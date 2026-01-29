"""Inference wrapper for camera-space POF.

Provides a simple interface for using the trained POF model:
- Load model from checkpoint
- Predict POF vectors from 2D keypoints
- Reconstruct 3D poses using MTC-style least-squares solver
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Union, Optional, Tuple, Dict

from .model import CameraPOFModel, load_pof_model
from .gnn_model import load_gnn_pof_model
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
    compute_limb_delta_2d,
)
from .bone_lengths import estimate_bone_lengths_array
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

        # Load model - auto-detect model type from checkpoint
        self.model = self._load_model_auto(verbose)
        self.model.eval()

    def _load_model_auto(self, verbose: bool):
        """Auto-detect model type from checkpoint and load appropriately."""
        # Peek at checkpoint to determine model type
        checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        model_type = config.get("model_type", "transformer")

        # GNN-based models (semgcn, semgcn-temporal, gcn)
        if model_type in ("semgcn", "semgcn-temporal", "gcn"):
            if verbose:
                print(f"[CameraPOFInference] Detected {model_type.upper()} model")
            return load_gnn_pof_model(str(self.model_path), device=self.device, verbose=verbose)
        else:
            # Default: Transformer model
            if verbose:
                print(f"[CameraPOFInference] Detected Transformer model")
            return load_pof_model(str(self.model_path), device=self.device, verbose=verbose)

        if verbose:
            solver = "least-squares" if use_least_squares else "direct FK"
            print(f"[CameraPOFInference] Device: {self.device}, Solver: {solver}")

    def _is_temporal_model(self) -> bool:
        """Check if model is a temporal model that needs sequential processing."""
        return hasattr(self.model, 'temporal_combine')

    def _has_zsign_head(self) -> bool:
        """Check if model has a Z-sign classification head."""
        return hasattr(self.model, 'z_sign_head')

    def _apply_z_sign_correction_hysteresis(
        self,
        pof: torch.Tensor,
        z_sign_logits: torch.Tensor,
        established_signs: Optional[torch.Tensor],
        establish_threshold: float = 0.7,
        flip_threshold: float = 0.85,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Correct POF Z direction with hysteresis to prevent flipping.

        Uses two thresholds:
        - establish_threshold: confidence needed to establish initial sign (0.7)
        - flip_threshold: confidence needed to flip an established sign (0.85)

        This prevents rapid flipping when z_sign is uncertain.

        Args:
            pof: (batch, 14, 3) POF unit vectors
            z_sign_logits: (batch, 14) logits for P(Z > 0)
            established_signs: (14,) tensor of established signs (+1/-1) or None
            establish_threshold: confidence to establish sign (default 0.7)
            flip_threshold: confidence to flip established sign (default 0.85)

        Returns:
            pof_corrected: (batch, 14, 3) POF with corrected Z signs
            new_established_signs: (14,) updated established signs
        """
        z_sign_prob = torch.sigmoid(z_sign_logits)  # (batch, 14)

        # Get current POF Z sign
        pof_z_sign = torch.sign(pof[:, :, 2])  # (batch, 14)
        pof_z_sign = torch.where(pof_z_sign == 0, torch.ones_like(pof_z_sign), pof_z_sign)

        if established_signs is None:
            # First frame: establish signs where confident, else use POF
            confident_positive = z_sign_prob > establish_threshold
            confident_negative = z_sign_prob < (1 - establish_threshold)

            # Determine target signs
            target_signs = torch.where(
                confident_positive,
                torch.ones_like(pof_z_sign),
                torch.where(
                    confident_negative,
                    -torch.ones_like(pof_z_sign),
                    pof_z_sign  # Use POF sign if uncertain
                )
            )
            established_signs = target_signs[0]  # (14,)
        else:
            # Subsequent frames: only flip if very confident
            established_signs = established_signs.to(pof.device)

            # Current established sign says Z should be positive/negative
            established_positive = established_signs > 0  # (14,)

            # To flip from positive to negative, need prob < (1 - flip_threshold)
            # To flip from negative to positive, need prob > flip_threshold
            should_flip_to_negative = established_positive & (z_sign_prob[0] < (1 - flip_threshold))
            should_flip_to_positive = (~established_positive) & (z_sign_prob[0] > flip_threshold)

            # Update established signs only where we flip
            new_signs = established_signs.clone()
            new_signs = torch.where(should_flip_to_negative, -torch.ones_like(new_signs), new_signs)
            new_signs = torch.where(should_flip_to_positive, torch.ones_like(new_signs), new_signs)
            established_signs = new_signs

            target_signs = established_signs.unsqueeze(0)  # (1, 14)

        # Apply correction: flip POF Z where it disagrees with target
        needs_flip = (pof_z_sign * target_signs) < 0  # Signs disagree

        pof_corrected = pof.clone()
        pof_corrected[:, :, 2] = torch.where(
            needs_flip,
            -pof[:, :, 2],
            pof[:, :, 2]
        )

        # Re-normalize
        pof_corrected = F.normalize(pof_corrected, dim=-1)

        return pof_corrected, established_signs

    def _apply_z_sign_correction(
        self,
        pof: torch.Tensor,
        z_sign_logits: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Correct POF Z direction using Z-sign classification.

        If z_sign predicts Z > 0 but POF has Z < 0 (or vice versa), flip the Z.
        This fixes the front-back ambiguity that causes L/R marker swapping.

        Args:
            pof: (batch, 14, 3) POF unit vectors
            z_sign_logits: (batch, 14) logits for P(Z > 0)
            threshold: classification threshold (default 0.5)

        Returns:
            (batch, 14, 3) POF with corrected Z signs
        """
        # Predict Z > 0 from logits
        z_sign_prob = torch.sigmoid(z_sign_logits)  # (batch, 14)
        z_should_be_positive = z_sign_prob > threshold  # (batch, 14)

        # Current Z sign from POF
        z_is_positive = pof[:, :, 2] > 0  # (batch, 14)

        # Flip Z where prediction disagrees with POF
        needs_flip = z_should_be_positive != z_is_positive  # (batch, 14)

        # Apply flip (multiply Z by -1 where needed)
        pof_corrected = pof.clone()
        pof_corrected[:, :, 2] = torch.where(
            needs_flip,
            -pof[:, :, 2],
            pof[:, :, 2]
        )

        # Re-normalize to ensure unit vector (Z flip shouldn't change norm, but be safe)
        pof_corrected = F.normalize(pof_corrected, dim=-1)

        return pof_corrected

    def _apply_z_sign_correction_with_info(
        self,
        pof: torch.Tensor,
        z_sign_logits: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Correct POF Z direction and return diagnostic info.

        Returns:
            pof_corrected: (batch, 14, 3) corrected POF
            z_sign_prob: (batch, 14) probabilities
            pof_before: (batch, 14, 3) POF before correction
            corrections: (batch, 14) bool mask of flipped limbs
        """
        pof_before = pof.clone()

        z_sign_prob = torch.sigmoid(z_sign_logits)
        z_should_be_positive = z_sign_prob > threshold
        z_is_positive = pof[:, :, 2] > 0
        needs_flip = z_should_be_positive != z_is_positive

        pof_corrected = pof.clone()
        pof_corrected[:, :, 2] = torch.where(
            needs_flip, -pof[:, :, 2], pof[:, :, 2]
        )
        pof_corrected = F.normalize(pof_corrected, dim=-1)

        return pof_corrected, z_sign_prob, pof_before, needs_flip

    @torch.no_grad()
    def predict_pof(
        self,
        keypoints_2d: np.ndarray,
        visibility: np.ndarray,
    ) -> np.ndarray:
        """Predict POF unit vectors from 2D keypoints.

        Input 2D keypoints are automatically normalized to pelvis-centered,
        unit-torso scale before being passed to the model.

        For temporal models (SemGCN-Temporal), frames are processed sequentially
        with proper temporal state tracking (previous frame's POF is passed to
        current frame).

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

        # For temporal models, process frame-by-frame with temporal state
        if self._is_temporal_model() and batch_size > 1:
            pof = self._predict_pof_temporal(
                pose_2d_norm, visibility, limb_delta_2d, limb_length_2d,
            )
        else:
            # Non-temporal models: process as batch
            pose_2d_t = torch.from_numpy(pose_2d_norm).to(self.device)
            vis_t = torch.from_numpy(visibility.astype(np.float32)).to(self.device)
            limb_delta_t = torch.from_numpy(limb_delta_2d).to(self.device)
            limb_length_t = torch.from_numpy(limb_length_2d).to(self.device)

            output = self.model(pose_2d_t, vis_t, limb_delta_t, limb_length_t)

            # Handle model output: tuple (pof, z_sign_logits) or just pof
            if isinstance(output, tuple):
                pof, z_sign_logits = output
                # Apply Z-sign correction for temporal models
                if self._has_zsign_head():
                    pof = self._apply_z_sign_correction(pof, z_sign_logits)
            else:
                pof = output
            pof = pof.cpu().numpy()

        if single_frame:
            pof = pof[0]

        return pof

    @torch.no_grad()
    def _predict_pof_temporal(
        self,
        pose_2d_norm: np.ndarray,
        visibility: np.ndarray,
        limb_delta_2d: np.ndarray,
        limb_length_2d: np.ndarray,
    ) -> np.ndarray:
        """Process frames sequentially for temporal models.

        Maintains prev_pof state across frames for proper temporal context.
        Uses hysteresis for Z-sign to prevent rapid flipping.
        """
        batch_size = pose_2d_norm.shape[0]
        pof_results = np.zeros((batch_size, NUM_LIMBS, 3), dtype=np.float32)
        has_zsign = self._has_zsign_head()

        # Initialize prev_pof as zeros for first frame
        prev_pof = torch.zeros(1, NUM_LIMBS, 3, device=self.device)

        # Track established Z signs for hysteresis (None = not established yet)
        # Once established, require higher confidence to flip
        established_z_signs = None  # (14,) tensor of +1/-1

        for i in range(batch_size):
            # Convert single frame to tensors
            pose_2d_t = torch.from_numpy(pose_2d_norm[i:i+1]).to(self.device)
            vis_t = torch.from_numpy(visibility[i:i+1].astype(np.float32)).to(self.device)
            limb_delta_t = torch.from_numpy(limb_delta_2d[i:i+1]).to(self.device)
            limb_length_t = torch.from_numpy(limb_length_2d[i:i+1]).to(self.device)

            # Call model with prev_pof
            output = self.model(pose_2d_t, vis_t, limb_delta_t, limb_length_t, prev_pof)

            # Handle model output: tuple (pof, z_sign_logits) or just pof
            if isinstance(output, tuple):
                pof, z_sign_logits = output
                # NOTE: Do NOT apply z-sign correction - it makes things worse.
                # The POF head already learns to predict the correct Z direction.
                # Z-sign head was an auxiliary training task, not for inference.
            else:
                pof = output

            # Store result and update prev_pof for next frame
            pof_results[i] = pof.cpu().numpy()[0]
            prev_pof = pof.detach()

        return pof_results

    @torch.no_grad()
    def _predict_pof_temporal_with_zsign(
        self,
        pose_2d_norm: np.ndarray,
        visibility: np.ndarray,
        limb_delta_2d: np.ndarray,
        limb_length_2d: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Process frames sequentially and return Z-sign diagnostic info.

        Returns POF after optional Z-sign correction along with diagnostic
        information about what corrections were applied.
        """
        batch_size = pose_2d_norm.shape[0]
        pof_results = np.zeros((batch_size, NUM_LIMBS, 3), dtype=np.float32)
        has_zsign = self._has_zsign_head()

        # Diagnostic arrays
        logits_all = np.zeros((batch_size, NUM_LIMBS), dtype=np.float32)
        probs_all = np.zeros((batch_size, NUM_LIMBS), dtype=np.float32)
        pof_before_all = np.zeros((batch_size, NUM_LIMBS, 3), dtype=np.float32)
        corrections_all = np.zeros((batch_size, NUM_LIMBS), dtype=bool)

        prev_pof = torch.zeros(1, NUM_LIMBS, 3, device=self.device)

        for i in range(batch_size):
            pose_2d_t = torch.from_numpy(pose_2d_norm[i:i+1]).to(self.device)
            vis_t = torch.from_numpy(visibility[i:i+1].astype(np.float32)).to(self.device)
            limb_delta_t = torch.from_numpy(limb_delta_2d[i:i+1]).to(self.device)
            limb_length_t = torch.from_numpy(limb_length_2d[i:i+1]).to(self.device)

            output = self.model(pose_2d_t, vis_t, limb_delta_t, limb_length_t, prev_pof)

            # Handle model output: tuple (pof, z_sign_logits) or just pof
            if isinstance(output, tuple):
                pof_raw, z_sign_logits = output
                pof_before_all[i] = pof_raw.cpu().numpy()[0]

                if has_zsign:
                    # Apply Z-sign correction and track corrections
                    pof, z_prob, _, corrections = \
                        self._apply_z_sign_correction_with_info(pof_raw, z_sign_logits)
                    logits_all[i] = z_sign_logits.cpu().numpy()[0]
                    probs_all[i] = z_prob.cpu().numpy()[0]
                    corrections_all[i] = corrections.cpu().numpy()[0]
                else:
                    pof = pof_raw
            else:
                pof = output
                pof_before_all[i] = pof.cpu().numpy()[0]

            pof_results[i] = pof.cpu().numpy()[0]
            prev_pof = pof.detach()

        zsign_info = {
            'logits': logits_all,
            'probs': probs_all,
            'pof_before': pof_before_all,
            'corrections': corrections_all,
        }
        return pof_results, zsign_info

    @torch.no_grad()
    def predict_pof_with_zsign(
        self,
        keypoints_2d: np.ndarray,
        visibility: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Predict POF and return Z-sign diagnostic information.

        For temporal models, processes frames sequentially with proper
        temporal state tracking.

        Args:
            keypoints_2d: (17, 2) or (N, 17, 2) raw 2D coordinates
            visibility: (17,) or (N, 17) confidence scores

        Returns:
            pof: (14, 3) or (N, 14, 3) POF unit vectors (after Z-sign correction)
            zsign_info: Dictionary containing:
                - 'logits': (N, 14) raw Z-sign logits
                - 'probs': (N, 14) sigmoid(logits), P(Z > 0)
                - 'pof_before': (N, 14, 3) POF before Z-sign correction
                - 'corrections': (N, 14) bool mask of flipped limbs
        """
        single_frame = keypoints_2d.ndim == 2
        if single_frame:
            keypoints_2d = keypoints_2d[np.newaxis, ...]
            visibility = visibility[np.newaxis, ...]

        batch_size = keypoints_2d.shape[0]

        # Normalize 2D and compute limb features
        pose_2d_norm = np.zeros((batch_size, NUM_JOINTS, 2), dtype=np.float32)
        limb_delta_2d = np.zeros((batch_size, NUM_LIMBS, 2), dtype=np.float32)
        limb_length_2d = np.zeros((batch_size, NUM_LIMBS), dtype=np.float32)

        for i in range(batch_size):
            pose_2d_norm[i], _, _ = normalize_pose_2d(keypoints_2d[i])
            limb_delta_2d[i], limb_length_2d[i] = compute_limb_features_2d(pose_2d_norm[i])

        # Use temporal processing for temporal models
        if self._is_temporal_model() and batch_size > 1:
            pof, zsign_info = self._predict_pof_temporal_with_zsign(
                pose_2d_norm, visibility, limb_delta_2d, limb_length_2d,
            )
        else:
            # Non-temporal: process as batch
            pose_2d_t = torch.from_numpy(pose_2d_norm).to(self.device)
            vis_t = torch.from_numpy(visibility.astype(np.float32)).to(self.device)
            limb_delta_t = torch.from_numpy(limb_delta_2d).to(self.device)
            limb_length_t = torch.from_numpy(limb_length_2d).to(self.device)

            output = self.model(pose_2d_t, vis_t, limb_delta_t, limb_length_t)

            # Handle model output: tuple (pof, z_sign_logits) or just pof
            if isinstance(output, tuple):
                pof_raw, z_sign_logits = output

                if self._has_zsign_head():
                    # Apply Z-sign correction and track corrections
                    pof, z_prob, pof_before, corrections = \
                        self._apply_z_sign_correction_with_info(pof_raw, z_sign_logits)

                    zsign_info = {
                        'logits': z_sign_logits.cpu().numpy(),
                        'probs': z_prob.cpu().numpy(),
                        'pof_before': pof_before.cpu().numpy(),
                        'corrections': corrections.cpu().numpy(),
                    }
                else:
                    pof = pof_raw
                    zsign_info = {
                        'logits': z_sign_logits.cpu().numpy(),
                        'probs': torch.sigmoid(z_sign_logits).cpu().numpy(),
                        'pof_before': pof_raw.cpu().numpy(),
                        'corrections': np.zeros((batch_size, NUM_LIMBS), dtype=bool),
                    }
            else:
                pof = output
                zsign_info = {
                    'logits': np.zeros((batch_size, NUM_LIMBS)),
                    'probs': np.full((batch_size, NUM_LIMBS), 0.5),
                    'pof_before': output.cpu().numpy(),
                    'corrections': np.zeros((batch_size, NUM_LIMBS), dtype=bool),
                }

            pof = pof.cpu().numpy()

        if single_frame:
            pof = pof[0]
            zsign_info = {k: v[0] for k, v in zsign_info.items()}

        return pof, zsign_info

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
                enforce_bones=True,  # Enforce bone lengths by adjusting Z
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
