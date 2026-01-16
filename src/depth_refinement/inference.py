"""
Inference module for depth refinement.

Usage:
    from src.depth_refinement.inference import DepthRefiner

    refiner = DepthRefiner('models/checkpoints/best_depth_model.pth')
    refined_pose = refiner.refine(pose_3d, visibility, pose_2d)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union

from .model import PoseAwareDepthRefiner
from .losses import COCO_BONES


def compute_torso_scale(pose: np.ndarray) -> float:
    """
    Compute torso scale for normalization.

    Uses hip-to-shoulder distance (average of left and right).
    Must match the training data normalization exactly.

    Args:
        pose: (17, 3) COCO keypoints (centered on pelvis)

    Returns:
        Torso scale, or 0 if invalid
    """
    # COCO indices: left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12
    left_torso = np.linalg.norm(pose[5] - pose[11])   # left shoulder to left hip
    right_torso = np.linalg.norm(pose[6] - pose[12])  # right shoulder to right hip

    # Average of both sides for robustness
    torso_scale = (left_torso + right_torso) / 2

    # Sanity check
    if torso_scale < 0.01 or torso_scale > 10.0:
        return 0.0

    return float(torso_scale)


class DepthRefiner:
    """Apply trained depth refinement to poses."""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
    ):
        """
        Initialize depth refiner.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Create model with saved config
        config = checkpoint.get('config', {})
        self.model = PoseAwareDepthRefiner(
            num_joints=config.get('num_joints', 17),
            d_model=config.get('d_model', 64),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 4),
            dim_feedforward=config.get('dim_feedforward', 256),
            dropout=config.get('dropout', 0.1),
            output_confidence=config.get('output_confidence', True),
            use_2d_pose=config.get('use_2d_pose', True),
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        print(f"[DepthRefiner] Loaded model from {model_path}")
        print(f"[DepthRefiner] Device: {self.device}")

    @torch.no_grad()
    def refine(
        self,
        pose_3d: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        pose_2d: Optional[np.ndarray] = None,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        return_confidence: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """
        Refine depth of a single pose or batch.

        IMPORTANT: Input pose_3d should already be:
        - Centered on pelvis
        - Transformed to training coordinate system (Y-up, Z-away)

        The model was trained on SCALE-NORMALIZED data. This method handles:
        1. Normalize input to unit torso scale
        2. Apply model corrections
        3. Denormalize back to original scale

        Args:
            pose_3d: (17, 3) or (N, 17, 3) COCO 3D pose(s) - pelvis-centered, Y-up, Z-away
            visibility: (17,) or (N, 17) visibility scores (default: all 1.0)
            pose_2d: (17, 2) or (N, 17, 2) normalized 2D coordinates (raw MediaPipe, NOT transformed)
            azimuth: Optional override for camera azimuth (0-360 degrees). If provided, skips prediction.
            elevation: Optional override for camera elevation (-90 to 90 degrees). If provided, skips prediction.
            return_confidence: Whether to return confidence scores

        Returns:
            Refined pose(s), optionally with confidence
        """
        # Handle batch dimension
        single_pose = pose_3d.ndim == 2
        if single_pose:
            pose_3d = pose_3d[np.newaxis, ...]
            if visibility is not None and visibility.ndim == 1:
                visibility = visibility[np.newaxis, ...]
            if pose_2d is not None and pose_2d.ndim == 2:
                pose_2d = pose_2d[np.newaxis, ...]

        batch_size = pose_3d.shape[0]

        # Default visibility
        if visibility is None:
            visibility = np.ones((batch_size, 17), dtype=np.float32)

        # === SCALE NORMALIZATION (to match training) ===
        # Compute torso scale for each pose and normalize to unit scale
        scales = np.zeros(batch_size, dtype=np.float32)
        pose_normalized = np.zeros_like(pose_3d)

        for i in range(batch_size):
            scale = compute_torso_scale(pose_3d[i])
            if scale > 0.01:
                scales[i] = scale
                pose_normalized[i] = pose_3d[i] / scale  # Normalize to unit scale
            else:
                # Fallback: use original pose if scale invalid
                scales[i] = 1.0
                pose_normalized[i] = pose_3d[i]

        # Convert to tensors
        pose_t = torch.from_numpy(pose_normalized.astype(np.float32)).to(self.device)
        vis_t = torch.from_numpy(visibility.astype(np.float32)).to(self.device)

        pose_2d_t = None
        if pose_2d is not None:
            pose_2d_t = torch.from_numpy(pose_2d.astype(np.float32)).to(self.device)

        # Prepare optional camera angle overrides
        az_t = None
        el_t = None
        if azimuth is not None and elevation is not None:
            az_t = torch.tensor([azimuth] * batch_size, dtype=torch.float32, device=self.device)
            el_t = torch.tensor([elevation] * batch_size, dtype=torch.float32, device=self.device)

        # Run model on normalized poses
        output = self.model(pose_t, vis_t, pose_2d=pose_2d_t, azimuth=az_t, elevation=el_t)

        # Get corrections (in normalized scale)
        delta_xyz = output['delta_xyz'].cpu().numpy()

        # Apply corrections and DENORMALIZE back to original scale
        refined_pose = pose_3d.copy()
        for i in range(batch_size):
            # Corrections are in normalized space, scale them back
            refined_pose[i] = pose_3d[i] + delta_xyz[i] * scales[i]

        # Return single or batch
        if single_pose:
            refined_pose = refined_pose[0]
            if return_confidence and 'confidence' in output:
                confidence = output['confidence'][0].cpu().numpy()
                return refined_pose, confidence
            return refined_pose

        if return_confidence and 'confidence' in output:
            confidence = output['confidence'].cpu().numpy()
            return refined_pose, confidence
        return refined_pose

    @torch.no_grad()
    def refine_sequence(
        self,
        poses_3d: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        poses_2d: Optional[np.ndarray] = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Refine depth of a sequence of poses.

        Args:
            poses_3d: (N, 17, 3) pose sequence
            visibility: (N, 17) visibility scores (default: all 1.0)
            poses_2d: (N, 17, 2) 2D pose sequence (optional but recommended)
            batch_size: Batch size for processing

        Returns:
            (N, 17, 3) refined poses
        """
        n_frames = poses_3d.shape[0]
        refined = np.zeros_like(poses_3d)

        if visibility is None:
            visibility = np.ones((n_frames, 17), dtype=np.float32)

        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            batch_poses = poses_3d[start:end]
            batch_vis = visibility[start:end]
            batch_2d = poses_2d[start:end] if poses_2d is not None else None
            refined[start:end] = self.refine(batch_poses, batch_vis, batch_2d)

        return refined

    @torch.no_grad()
    def refine_sequence_with_bone_locking(
        self,
        poses_3d: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        poses_2d: Optional[np.ndarray] = None,
        batch_size: int = 64,
        calibration_frames: int = 50,
    ) -> np.ndarray:
        """
        Refine depth with temporal bone length consistency.

        Same as refine_sequence, but applies bone locking:
        1. Refine all frames normally
        2. Compute median bone lengths from first N frames
        3. Project all frames to satisfy those bone lengths

        This ensures temporally consistent bone lengths (real bones don't change).

        Args:
            poses_3d: (N, 17, 3) pose sequence
            visibility: (N, 17) visibility scores (default: all 1.0)
            poses_2d: (N, 17, 2) 2D pose sequence (optional but recommended)
            batch_size: Batch size for processing
            calibration_frames: Number of frames to compute reference bone lengths from

        Returns:
            (N, 17, 3) refined poses with consistent bone lengths
        """
        # Step 1: Normal refinement
        refined = self.refine_sequence(poses_3d, visibility, poses_2d, batch_size)

        # Step 2: Compute reference bone lengths from calibration frames
        n_calib = min(calibration_frames, len(refined))
        reference_lengths = self._compute_median_bone_lengths(refined[:n_calib])

        # Step 3: Project all frames to bone length constraints
        locked = self._project_to_bone_lengths(refined, reference_lengths)

        return locked

    def _compute_median_bone_lengths(self, poses: np.ndarray) -> dict:
        """Compute median length for each bone across frames."""
        bone_lengths = {i: [] for i in range(len(COCO_BONES))}

        for pose in poses:
            for bone_idx, (i, j) in enumerate(COCO_BONES):
                length = np.linalg.norm(pose[i] - pose[j])
                bone_lengths[bone_idx].append(length)

        return {idx: np.median(lengths) for idx, lengths in bone_lengths.items()}

    def _project_to_bone_lengths(
        self,
        poses: np.ndarray,
        reference_lengths: dict,
        iterations: int = 3,
    ) -> np.ndarray:
        """Project poses to satisfy bone length constraints.

        Uses iterative projection: adjust child joint along bone direction
        to achieve target length, prioritizing depth (Z) corrections.
        """
        result = poses.copy()

        # Define parent-child relationships for each bone
        # Format: (bone_idx, child_joint_idx)
        BONE_HIERARCHY = [
            # Torso first (most stable)
            (0, 6),   # shoulder width: left shoulder (5) -> right (6)
            (1, 12),  # hip width: left hip (11) -> right (12)
            (2, 11),  # left torso: shoulder (5) -> hip (11)
            (3, 12),  # right torso: shoulder (6) -> hip (12)
            # Arms (shoulder → elbow → wrist)
            (4, 7),   # left upper arm: shoulder (5) -> elbow (7)
            (5, 9),   # left forearm: elbow (7) -> wrist (9)
            (6, 8),   # right upper arm: shoulder (6) -> elbow (8)
            (7, 10),  # right forearm: elbow (8) -> wrist (10)
            # Legs (hip → knee → ankle)
            (8, 13),  # left thigh: hip (11) -> knee (13)
            (9, 15),  # left shin: knee (13) -> ankle (15)
            (10, 14), # right thigh: hip (12) -> knee (14)
            (11, 16), # right shin: knee (14) -> ankle (16)
        ]

        for _ in range(iterations):
            for frame_idx in range(len(result)):
                pose = result[frame_idx]

                for bone_idx, child_idx in BONE_HIERARCHY:
                    bone = COCO_BONES[bone_idx]
                    parent_idx = bone[0] if bone[1] == child_idx else bone[1]
                    target_length = reference_lengths[bone_idx]

                    # Current bone vector
                    bone_vec = pose[child_idx] - pose[parent_idx]
                    current_length = np.linalg.norm(bone_vec)

                    if current_length < 1e-6:
                        continue

                    # Compute new child position along bone direction
                    bone_dir = bone_vec / current_length
                    new_child = pose[parent_idx] + bone_dir * target_length

                    # Blend: prioritize Z (depth) correction (80%), less XY change (20%)
                    pose[child_idx, :2] += 0.2 * (new_child[:2] - pose[child_idx, :2])
                    pose[child_idx, 2] += 0.8 * (new_child[2] - pose[child_idx, 2])

                result[frame_idx] = pose

        return result

    def get_prediction_info(
        self,
        pose_3d: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        pose_2d: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Get detailed prediction info including camera angle estimates.

        Args:
            pose_3d: (17, 3) or (N, 17, 3) COCO 3D pose(s) - pelvis-centered, Y-up, Z-away
            visibility: visibility scores
            pose_2d: 2D pose coordinates (raw MediaPipe, NOT transformed)

        Returns:
            dict with delta_xyz, confidence, pred_azimuth, pred_elevation, scale
        """
        single_pose = pose_3d.ndim == 2
        if single_pose:
            pose_3d = pose_3d[np.newaxis, ...]
            if visibility is not None and visibility.ndim == 1:
                visibility = visibility[np.newaxis, ...]
            if pose_2d is not None and pose_2d.ndim == 2:
                pose_2d = pose_2d[np.newaxis, ...]

        batch_size = pose_3d.shape[0]
        if visibility is None:
            visibility = np.ones((batch_size, 17), dtype=np.float32)

        # === SCALE NORMALIZATION (to match training) ===
        scales = np.zeros(batch_size, dtype=np.float32)
        pose_normalized = np.zeros_like(pose_3d)

        for i in range(batch_size):
            scale = compute_torso_scale(pose_3d[i])
            if scale > 0.01:
                scales[i] = scale
                pose_normalized[i] = pose_3d[i] / scale
            else:
                scales[i] = 1.0
                pose_normalized[i] = pose_3d[i]

        pose_t = torch.from_numpy(pose_normalized.astype(np.float32)).to(self.device)
        vis_t = torch.from_numpy(visibility.astype(np.float32)).to(self.device)
        pose_2d_t = None
        if pose_2d is not None:
            pose_2d_t = torch.from_numpy(pose_2d.astype(np.float32)).to(self.device)

        with torch.no_grad():
            output = self.model(pose_t, vis_t, pose_2d=pose_2d_t)

        # delta_xyz is in normalized space, scale back to original
        delta_xyz_normalized = output['delta_xyz'].cpu().numpy()
        delta_xyz = delta_xyz_normalized * scales[:, np.newaxis, np.newaxis]

        result = {
            'delta_xyz': delta_xyz,
            'delta_xyz_normalized': delta_xyz_normalized,  # For debugging
            'pred_azimuth': output['pred_azimuth'].cpu().numpy(),
            'pred_elevation': output['pred_elevation'].cpu().numpy(),
            'scale': scales,
        }
        if 'confidence' in output:
            result['confidence'] = output['confidence'].cpu().numpy()

        if single_pose:
            result = {k: v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v for k, v in result.items()}

        return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test depth refinement inference')
    parser.add_argument('--model', type=str, default='models/checkpoints/best_depth_model.pth')
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        print("Run training first: uv run --group neural python scripts/train_depth_model.py")
        exit(1)

    # Create refiner
    refiner = DepthRefiner(args.model)

    # Test with random data
    print("\nTesting with random pose...")
    pose_3d = np.random.randn(17, 3).astype(np.float32) * 0.3
    pose_3d[:, 1] += 0.5  # Move up
    visibility = np.random.rand(17).astype(np.float32)
    pose_2d = np.random.rand(17, 2).astype(np.float32)  # Simulated 2D

    refined, confidence = refiner.refine(pose_3d, visibility, pose_2d, return_confidence=True)

    print(f"Input pose range:")
    print(f"  X: {pose_3d[:, 0].min():.3f} to {pose_3d[:, 0].max():.3f}")
    print(f"  Y: {pose_3d[:, 1].min():.3f} to {pose_3d[:, 1].max():.3f}")
    print(f"  Z: {pose_3d[:, 2].min():.3f} to {pose_3d[:, 2].max():.3f}")
    print(f"Output pose range:")
    print(f"  X: {refined[:, 0].min():.3f} to {refined[:, 0].max():.3f}")
    print(f"  Y: {refined[:, 1].min():.3f} to {refined[:, 1].max():.3f}")
    print(f"  Z: {refined[:, 2].min():.3f} to {refined[:, 2].max():.3f}")
    print(f"3D corrections (mean ± std):")
    delta = refined - pose_3d
    print(f"  X: {delta[:, 0].mean():.4f} ± {delta[:, 0].std():.4f}")
    print(f"  Y: {delta[:, 1].mean():.4f} ± {delta[:, 1].std():.4f}")
    print(f"  Z: {delta[:, 2].mean():.4f} ± {delta[:, 2].std():.4f}")
    print(f"Confidence: {confidence.mean():.3f} ± {confidence.std():.3f}")

    # Test with prediction info
    print("\nGetting prediction details...")
    info = refiner.get_prediction_info(pose_3d, visibility, pose_2d)
    print(f"Predicted azimuth: {info['pred_azimuth']:.1f}°")
    print(f"Predicted elevation: {info['pred_elevation']:.1f}°")
    print(f"delta_xyz shape: {info['delta_xyz'].shape}")

    # Test batch
    print("\nTesting batch mode...")
    poses = np.random.randn(10, 17, 3).astype(np.float32) * 0.3
    refined_batch = refiner.refine(poses)
    print(f"Batch shape: {refined_batch.shape}")
