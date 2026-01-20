"""Inference wrapper for camera-space POF.

Provides a simple interface for using the trained POF model:
- Load model from checkpoint
- Predict POF vectors from 2D keypoints
- Reconstruct 3D poses using MTC-style least-squares solver
"""

import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional, Tuple

from .model import CameraPOFModel, load_pof_model
from .bone_lengths import estimate_bone_lengths_array
from .reconstruction import (
    reconstruct_skeleton_from_pof,
    reconstruct_skeleton_batch,
    reconstruct_skeleton_least_squares,
    apply_pof_to_pose,
    apply_pof_to_pose_batch,
)
from .least_squares import (
    solve_depth_least_squares_pof,
    normalize_2d_for_pof,
)
from .dataset import normalize_pose_2d, compute_limb_features_2d
from .constants import NUM_JOINTS, NUM_LIMBS, LIMB_DEFINITIONS, HEIGHT_TO_TORSO_RATIO


class CameraPOFInference:
    """Camera-space POF inference wrapper.

    Provides convenient methods for:
    - Predicting POF vectors from 2D keypoints
    - Reconstructing 3D poses from POF using least-squares solver
    - Refining existing 3D poses using POF
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        use_least_squares: bool = True,
        verbose: bool = True,
    ):
        """Initialize inference wrapper.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            use_least_squares: If True (default), use MTC-style least-squares
                              solver. This is recommended as it ensures the
                              reconstructed skeleton projects back to observed
                              2D positions (reprojection error ≈ 0).
            verbose: Print loading information
        """
        self.model_path = Path(model_path)
        self.use_least_squares = use_least_squares

        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self.model = load_pof_model(
            str(self.model_path),
            device=self.device,
            verbose=verbose,
        )
        self.model.eval()

        if verbose:
            solver = "least-squares" if use_least_squares else "direct FK"
            print(f"[CameraPOFInference] Device: {self.device}, Solver: {solver}")

    @torch.no_grad()
    def predict_pof(
        self,
        keypoints_2d: np.ndarray,
        visibility: np.ndarray,
    ) -> np.ndarray:
        """Predict POF unit vectors from 2D keypoints.

        Input 2D keypoints are automatically normalized to pelvis-centered,
        unit-torso scale before being passed to the model.

        Args:
            keypoints_2d: (17, 2) or (N, 17, 2) raw 2D coordinates (e.g., [0,1] range)
            visibility: (17,) or (N, 17) confidence scores

        Returns:
            (14, 3) or (N, 14, 3) POF unit vectors
        """
        single_frame = keypoints_2d.ndim == 2
        if single_frame:
            keypoints_2d = keypoints_2d[np.newaxis, ...]
            visibility = visibility[np.newaxis, ...]

        batch_size = keypoints_2d.shape[0]

        # Normalize 2D and compute limb features for each frame
        pose_2d_norm = np.zeros((batch_size, NUM_JOINTS, 2), dtype=np.float32)
        limb_delta_2d = np.zeros((batch_size, NUM_LIMBS, 2), dtype=np.float32)
        limb_length_2d = np.zeros((batch_size, NUM_LIMBS), dtype=np.float32)

        for i in range(batch_size):
            pose_2d_norm[i], _, _ = normalize_pose_2d(keypoints_2d[i])
            limb_delta_2d[i], limb_length_2d[i] = compute_limb_features_2d(pose_2d_norm[i])

        # Convert to tensors
        pose_2d_t = torch.from_numpy(pose_2d_norm).to(self.device)
        vis_t = torch.from_numpy(visibility.astype(np.float32)).to(self.device)
        limb_delta_t = torch.from_numpy(limb_delta_2d).to(self.device)
        limb_length_t = torch.from_numpy(limb_length_2d).to(self.device)

        # Predict POF
        pof = self.model(pose_2d_t, vis_t, limb_delta_t, limb_length_t)
        pof = pof.cpu().numpy()

        if single_frame:
            pof = pof[0]

        return pof

    @torch.no_grad()
    def reconstruct_3d(
        self,
        keypoints_2d: np.ndarray,
        visibility: np.ndarray,
        height_m: float,
        pelvis_depth: float = 2.0,
        use_meter_coords: bool = True,
        fov_horizontal_deg: float = 60.0,
        aspect_ratio: float = 16/9,
    ) -> np.ndarray:
        """Reconstruct 3D pose from 2D keypoints.

        Uses MTC-style least-squares solver by default (self.use_least_squares=True).
        The LS solver keeps X,Y fixed from 2D observations and only solves for Z,
        ensuring the skeleton projects back to observed 2D positions.

        The output is in true metric scale when height_m is provided, using
        the relationship: torso_length = height / HEIGHT_TO_TORSO_RATIO (≈3.4).

        Args:
            keypoints_2d: (17, 2) or (N, 17, 2) normalized 2D coordinates
            visibility: (17,) or (N, 17) confidence scores
            height_m: Subject body height in meters (used for true metric output)
            pelvis_depth: Initial depth for pelvis (only used with direct FK)
            use_meter_coords: If True (default), convert normalized 2D to meter
                            coordinates using pinhole camera model. Set False
                            only if 2D coords are already in meters.
            fov_horizontal_deg: Horizontal FOV for meter conversion (default 60°)
            aspect_ratio: Image aspect ratio for meter conversion (default 16:9)

        Returns:
            (17, 3) or (N, 17, 3) reconstructed 3D poses in camera space (meters)
        """
        single_frame = keypoints_2d.ndim == 2
        if single_frame:
            keypoints_2d = keypoints_2d[np.newaxis, ...]
            visibility = visibility[np.newaxis, ...]

        # Predict POF vectors
        pof = self.predict_pof(keypoints_2d, visibility)
        if pof.ndim == 2:
            pof = pof[np.newaxis, ...]

        # Get bone lengths from height
        bone_lengths = estimate_bone_lengths_array(height_m)

        # Compute metric torso scale from known height
        # This gives true metric output instead of approximate 2D-derived scale
        metric_torso_scale = height_m / HEIGHT_TO_TORSO_RATIO

        # Reconstruct 3D poses
        if self.use_least_squares:
            # Use MTC-style least-squares solver (recommended)
            poses_3d = reconstruct_skeleton_least_squares(
                pof, keypoints_2d, bone_lengths,
                pelvis_depth=0.0,  # Normalized space
                denormalize=True,  # Scale to true meters
                output_depth=pelvis_depth,
                metric_torso_scale=metric_torso_scale,
            )
        else:
            # Use direct forward kinematics (original method)
            poses_3d = reconstruct_skeleton_batch(
                pof, bone_lengths, keypoints_2d, pelvis_depth,
                use_meter_coords, fov_horizontal_deg, aspect_ratio
            )

        if single_frame:
            poses_3d = poses_3d[0]

        return poses_3d

    @torch.no_grad()
    def reconstruct_3d_normalized(
        self,
        keypoints_2d: np.ndarray,
        visibility: np.ndarray,
        bone_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reconstruct 3D pose in normalized space (pelvis-centered, unit torso).

        Useful for training and evaluation where normalized poses are preferred.

        Args:
            keypoints_2d: (17, 2) or (N, 17, 2) normalized [0,1] 2D coordinates
            visibility: (17,) or (N, 17) confidence scores
            bone_lengths: Optional (14,) bone lengths for scale clamping

        Returns:
            (17, 3) or (N, 17, 3) reconstructed poses in normalized space
        """
        single_frame = keypoints_2d.ndim == 2
        if single_frame:
            keypoints_2d = keypoints_2d[np.newaxis, ...]
            visibility = visibility[np.newaxis, ...]

        # Predict POF vectors
        pof = self.predict_pof(keypoints_2d, visibility)
        if pof.ndim == 2:
            pof = pof[np.newaxis, ...]

        # Reconstruct using LS solver in normalized space
        poses_3d = reconstruct_skeleton_least_squares(
            pof, keypoints_2d, bone_lengths,
            pelvis_depth=0.0,
            denormalize=False,  # Stay in normalized space
        )

        if single_frame:
            poses_3d = poses_3d[0]

        return poses_3d

    @torch.no_grad()
    def refine_pose(
        self,
        pose_3d: np.ndarray,
        keypoints_2d: np.ndarray,
        visibility: np.ndarray,
        blend_weight: float = 0.5,
    ) -> np.ndarray:
        """Refine existing 3D pose using POF predictions.

        Blends POF-based directions with input pose, preserving
        input bone lengths while adjusting directions.

        Args:
            pose_3d: (17, 3) or (N, 17, 3) existing 3D pose
            keypoints_2d: (17, 2) or (N, 17, 2) 2D coordinates
            visibility: (17,) or (N, 17) confidence scores
            blend_weight: 0 = keep input, 1 = full POF direction

        Returns:
            (17, 3) or (N, 17, 3) refined pose
        """
        single_frame = pose_3d.ndim == 2
        if single_frame:
            pose_3d = pose_3d[np.newaxis, ...]
            keypoints_2d = keypoints_2d[np.newaxis, ...]
            visibility = visibility[np.newaxis, ...]

        # Predict POF vectors
        pof = self.predict_pof(keypoints_2d, visibility)
        if pof.ndim == 2:
            pof = pof[np.newaxis, ...]

        # Apply POF to refine poses
        refined = apply_pof_to_pose_batch(pose_3d, pof, blend_weight)

        if single_frame:
            refined = refined[0]

        return refined


def create_pof_inference(
    model_path: Union[str, Path] = "models/checkpoints/best_pof_model.pth",
    device: str = "auto",
) -> Optional[CameraPOFInference]:
    """Create POF inference wrapper if model exists.

    Convenience function that handles missing model gracefully.

    Args:
        model_path: Path to trained model
        device: Device to use

    Returns:
        CameraPOFInference instance or None if model not found
    """
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[CameraPOF] Model not found: {model_path}")
        return None

    try:
        return CameraPOFInference(model_path, device=device)
    except Exception as e:
        print(f"[CameraPOF] Failed to load model: {e}")
        return None
