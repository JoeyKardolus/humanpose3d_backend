"""
Inference module for depth refinement.

Usage:
    from src.depth_refinement.inference import DepthRefiner

    refiner = DepthRefiner('models/checkpoints/best_model.pth')
    refined_pose = refiner.refine(pose, visibility, view_angle)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union

from .model import create_model


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

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model with saved config
        config = checkpoint.get('config', {})
        self.model = create_model(
            num_joints=config.get('num_joints', 17),
            d_model=config.get('d_model', 64),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 4),
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        print(f"[DepthRefiner] Loaded model from {model_path}")
        print(f"[DepthRefiner] Device: {self.device}")

    def compute_view_angle(self, pose: np.ndarray) -> float:
        """
        Compute view angle from torso plane.

        Args:
            pose: (17, 3) COCO pose

        Returns:
            View angle in degrees (0-90)
        """
        # COCO indices: left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12
        left_shoulder = pose[5]
        right_shoulder = pose[6]
        left_hip = pose[11]
        right_hip = pose[12]

        # Vectors defining torso plane
        shoulder_vec = right_shoulder - left_shoulder
        hip_vec = right_hip - left_hip

        # Normal to torso plane (points forward from body)
        torso_normal = np.cross(shoulder_vec, hip_vec)
        norm = np.linalg.norm(torso_normal)
        if norm < 1e-6:
            return 45.0  # Default if degenerate

        torso_normal = torso_normal / norm

        # Camera looks down +Z axis
        camera_dir = np.array([0, 0, 1])
        cos_angle = np.dot(torso_normal, camera_dir)

        # Return absolute angle (0° = frontal, 90° = profile)
        angle = np.degrees(np.arccos(np.clip(np.abs(cos_angle), -1, 1)))
        return angle

    @torch.no_grad()
    def refine(
        self,
        pose: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        view_angle: Optional[float] = None,
        return_confidence: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """
        Refine depth of a single pose.

        Args:
            pose: (17, 3) or (N, 17, 3) COCO pose(s)
            visibility: (17,) or (N, 17) visibility scores (default: all 1.0)
            view_angle: View angle in degrees, or None to auto-compute
            return_confidence: Whether to return confidence scores

        Returns:
            Refined pose(s), optionally with confidence
        """
        # Handle batch dimension
        single_pose = pose.ndim == 2
        if single_pose:
            pose = pose[np.newaxis, ...]

        batch_size = pose.shape[0]

        # Default visibility
        if visibility is None:
            visibility = np.ones((batch_size, 17))
        elif visibility.ndim == 1:
            visibility = visibility[np.newaxis, ...]

        # Compute or use provided view angle
        if view_angle is None:
            view_angles = np.array([
                self.compute_view_angle(pose[i]) for i in range(batch_size)
            ])
        else:
            view_angles = np.full(batch_size, view_angle)

        # Convert to tensors
        pose_t = torch.from_numpy(pose.astype(np.float32)).to(self.device)
        vis_t = torch.from_numpy(visibility.astype(np.float32)).to(self.device)
        angle_t = torch.from_numpy(view_angles.astype(np.float32)).to(self.device)

        # Run model
        output = self.model(pose_t, vis_t, angle_t)

        # Apply correction
        delta_z = output['delta_z'].cpu().numpy()
        refined_pose = pose.copy()
        refined_pose[:, :, 2] += delta_z

        # Return single or batch
        if single_pose:
            refined_pose = refined_pose[0]
            if return_confidence:
                confidence = output['confidence'][0].cpu().numpy()
                return refined_pose, confidence
            return refined_pose

        if return_confidence:
            confidence = output['confidence'].cpu().numpy()
            return refined_pose, confidence
        return refined_pose

    @torch.no_grad()
    def refine_sequence(
        self,
        poses: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Refine depth of a sequence of poses.

        Args:
            poses: (N, 17, 3) pose sequence
            visibility: (N, 17) visibility scores (default: all 1.0)
            batch_size: Batch size for processing

        Returns:
            (N, 17, 3) refined poses
        """
        n_frames = poses.shape[0]
        refined = np.zeros_like(poses)

        if visibility is None:
            visibility = np.ones((n_frames, 17))

        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            batch_poses = poses[start:end]
            batch_vis = visibility[start:end]
            refined[start:end] = self.refine(batch_poses, batch_vis)

        return refined


def apply_depth_refinement_to_trc(
    input_trc: Path,
    output_trc: Path,
    model_path: Path,
    coco_to_trc_mapping: dict = None,
) -> dict:
    """
    Apply depth refinement to a TRC file.

    This is a convenience function for applying the model to pose files.

    Args:
        input_trc: Input TRC file path
        output_trc: Output TRC file path
        model_path: Path to trained model
        coco_to_trc_mapping: Mapping from COCO joint indices to TRC marker names

    Returns:
        dict with statistics (before/after depth error, etc.)
    """
    # Default COCO to OpenCap marker mapping
    if coco_to_trc_mapping is None:
        coco_to_trc_mapping = {
            0: 'Nose',
            5: 'LShoulder',
            6: 'RShoulder',
            7: 'LElbow',
            8: 'RElbow',
            9: 'LWrist',
            10: 'RWrist',
            11: 'LHip',
            12: 'RHip',
            13: 'LKnee',
            14: 'RKnee',
            15: 'LAnkle',
            16: 'RAnkle',
        }

    # Load TRC
    from src.datastream.data_stream import read_trc
    trc_data = read_trc(input_trc)

    # Extract COCO joints from TRC
    # (Would need to implement TRC reading and COCO extraction)

    print(f"[apply_depth_refinement_to_trc] Not fully implemented yet")
    print(f"Input: {input_trc}")
    print(f"Output: {output_trc}")
    print(f"Model: {model_path}")

    return {'status': 'not_implemented'}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test depth refinement inference')
    parser.add_argument('--model', type=str, default='models/checkpoints/best_model.pth')
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        print("Run training first: uv run --group neural python scripts/train_depth_model.py")
        exit(1)

    # Create refiner
    refiner = DepthRefiner(args.model)

    # Test with random data
    print("\nTesting with random pose...")
    pose = np.random.randn(17, 3) * 0.3
    pose[:, 1] += 0.5  # Move up
    visibility = np.random.rand(17)

    refined, confidence = refiner.refine(pose, visibility, return_confidence=True)

    print(f"Input pose Z range: {pose[:, 2].min():.3f} to {pose[:, 2].max():.3f}")
    print(f"Output pose Z range: {refined[:, 2].min():.3f} to {refined[:, 2].max():.3f}")
    print(f"Depth corrections: {(refined[:, 2] - pose[:, 2]).mean():.4f} ± {(refined[:, 2] - pose[:, 2]).std():.4f}")
    print(f"Confidence: {confidence.mean():.3f} ± {confidence.std():.3f}")

    # Test batch
    print("\nTesting batch mode...")
    poses = np.random.randn(10, 17, 3) * 0.3
    refined_batch = refiner.refine(poses)
    print(f"Batch shape: {refined_batch.shape}")
