"""
Inference module for MainRefiner.

This provides a complete pipeline that:
1. Loads all three models (depth, joint, main refiner)
2. Runs them in sequence
3. Returns the fused refined pose

Usage:
    from src.main_refinement import MainRefinerPipeline

    pipeline = MainRefinerPipeline(
        depth_checkpoint='models/checkpoints/best_depth_model.pth',
        joint_checkpoint='models/checkpoints/best_joint_model.pth',
        main_checkpoint='models/checkpoints/best_main_refiner.pth',
    )
    refined_pose = pipeline.refine(pose_3d, visibility, pose_2d)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict

from .model import MainRefiner


def compute_torso_scale(pose: np.ndarray) -> float:
    """Compute torso scale for normalization.

    Args:
        pose: (17, 3) COCO keypoints

    Returns:
        Torso scale, or 0 if invalid
    """
    left_torso = np.linalg.norm(pose[5] - pose[11])
    right_torso = np.linalg.norm(pose[6] - pose[12])
    torso_scale = (left_torso + right_torso) / 2

    if torso_scale < 0.01 or torso_scale > 10.0:
        return 0.0

    return float(torso_scale)


class MainRefinerPipeline:
    """Complete refinement pipeline using all three models.

    This class handles:
    1. Loading depth, joint, and main refiner models
    2. Normalizing input poses
    3. Running all models in sequence
    4. Denormalizing outputs
    """

    def __init__(
        self,
        depth_checkpoint: Union[str, Path],
        joint_checkpoint: Union[str, Path],
        main_checkpoint: Union[str, Path],
        device: Optional[str] = None,
    ):
        """
        Initialize pipeline with all model checkpoints.

        Args:
            depth_checkpoint: Path to depth model checkpoint
            joint_checkpoint: Path to joint model checkpoint
            main_checkpoint: Path to main refiner checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load depth model
        from src.depth_refinement.model import PoseAwareDepthRefiner
        self.depth_model = self._load_depth_model(depth_checkpoint)

        # Load joint model
        from src.joint_refinement.model import JointConstraintRefiner
        self.joint_model = self._load_joint_model(joint_checkpoint)

        # Load main refiner
        self.main_model = self._load_main_model(main_checkpoint)

        print(f"MainRefinerPipeline loaded on {self.device}")
        print(f"  Depth model: {sum(p.numel() for p in self.depth_model.parameters()):,} params")
        print(f"  Joint model: {sum(p.numel() for p in self.joint_model.parameters()):,} params")
        print(f"  Main model: {sum(p.numel() for p in self.main_model.parameters()):,} params")

    def _load_depth_model(self, checkpoint_path: Union[str, Path]) -> torch.nn.Module:
        """Load depth refinement model."""
        from src.depth_refinement.model import PoseAwareDepthRefiner

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Depth model not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        model = PoseAwareDepthRefiner(
            num_joints=config.get('num_joints', 17),
            d_model=config.get('d_model', 64),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 4),
            dim_feedforward=config.get('dim_feedforward', 256),
            dropout=0.0,  # No dropout at inference
            output_confidence=config.get('output_confidence', True),
            use_2d_pose=config.get('use_2d_pose', True),
            use_elepose=config.get('use_elepose', False),
            use_limb_orientations=config.get('use_limb_orientations', False),
        )

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            raise KeyError(f"Checkpoint has no 'model_state_dict' or 'model' key")
        model.to(self.device)
        model.eval()

        return model

    def _load_joint_model(self, checkpoint_path: Union[str, Path]) -> torch.nn.Module:
        """Load joint refinement model."""
        from src.joint_refinement.model import JointConstraintRefiner

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Joint model not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        model = JointConstraintRefiner(
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 4),
            n_layers=config.get('n_layers', 4),
            dropout=0.0,  # No dropout at inference
        )

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            raise KeyError(f"Checkpoint has no 'model_state_dict' or 'model' key")
        model.to(self.device)
        model.eval()

        return model

    def _load_main_model(self, checkpoint_path: Union[str, Path]) -> MainRefiner:
        """Load main refiner model."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Main model not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        model = MainRefiner(
            d_model=config.get('d_model', 128),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 2),
            dropout=0.0,  # No dropout at inference
        )

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            raise KeyError(f"Checkpoint has no 'model_state_dict' or 'model' key")
        model.to(self.device)
        model.eval()

        return model

    @torch.no_grad()
    def refine(
        self,
        pose_3d: np.ndarray,
        visibility: np.ndarray,
        pose_2d: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Refine a 3D pose using the full pipeline.

        Args:
            pose_3d: (17, 3) or (N, 17, 3) 3D pose
            visibility: (17,) or (N, 17) per-joint visibility
            pose_2d: (17, 2) or (N, 17, 2) optional 2D pose (normalized [0,1])

        Returns:
            Dict with:
                'refined_pose': (17, 3) or (N, 17, 3) refined pose
                'depth_weights': (17,) or (N, 17) depth model weights
                'joint_weights': (17,) or (N, 17) joint model weights
                'confidence': (17,) or (N, 17) per-joint confidence
        """
        # Handle single pose input
        single_pose = pose_3d.ndim == 2
        if single_pose:
            pose_3d = pose_3d[np.newaxis, ...]
            visibility = visibility[np.newaxis, ...]
            if pose_2d is not None:
                pose_2d = pose_2d[np.newaxis, ...]

        batch_size = pose_3d.shape[0]
        results = []

        for i in range(batch_size):
            result = self._refine_single(
                pose_3d[i],
                visibility[i],
                pose_2d[i] if pose_2d is not None else None,
            )
            results.append(result)

        # Stack results
        output = {
            key: np.stack([r[key] for r in results])
            for key in results[0].keys()
        }

        # Remove batch dimension for single pose
        if single_pose:
            output = {key: value[0] for key, value in output.items()}

        return output

    def _refine_single(
        self,
        pose_3d: np.ndarray,
        visibility: np.ndarray,
        pose_2d: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Refine a single pose."""
        # Compute pelvis and normalize
        pelvis = (pose_3d[11] + pose_3d[12]) / 2
        pose_centered = pose_3d - pelvis

        torso_scale = compute_torso_scale(pose_centered)
        if torso_scale > 0:
            pose_normalized = pose_centered / torso_scale
        else:
            pose_normalized = pose_centered

        # Convert to tensors
        pose_t = torch.from_numpy(pose_normalized.astype(np.float32)).unsqueeze(0).to(self.device)
        vis_t = torch.from_numpy(visibility.astype(np.float32)).unsqueeze(0).to(self.device)

        if pose_2d is not None:
            pose_2d_t = torch.from_numpy(pose_2d.astype(np.float32)).unsqueeze(0).to(self.device)
        else:
            # Generate synthetic 2D from 3D
            pose_2d_t = pose_t[:, :, :2] / pose_t[:, :, 2:3].clamp(min=1e-4)
            pose_2d_t = (pose_2d_t + 1) / 2

        # 1. Run depth model
        depth_outputs = self.depth_model(pose_t, vis_t, pose_2d=pose_2d_t)

        # 2. Run joint model (compute angles first)
        joint_outputs = self._run_joint_model(pose_normalized, visibility)

        # 3. Run main refiner
        main_output = self.main_model(pose_t, vis_t, depth_outputs, joint_outputs)

        # Denormalize
        refined_pose = main_output['refined_pose'].squeeze(0).cpu().numpy()
        if torso_scale > 0:
            refined_pose = refined_pose * torso_scale
        refined_pose = refined_pose + pelvis

        return {
            'refined_pose': refined_pose,
            'depth_weights': main_output['depth_weights'].squeeze(0).cpu().numpy(),
            'joint_weights': main_output['joint_weights'].squeeze(0).cpu().numpy(),
            'confidence': main_output['confidence'].squeeze(0).cpu().numpy(),
        }

    def _run_joint_model(
        self,
        pose_normalized: np.ndarray,
        visibility: np.ndarray,
        precomputed_angles: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run joint model on pre-computed angles.

        Joint angles must be pre-computed from augmented TRC data using
        comprehensive_joint_angles.py. Cannot compute from COCO-17 directly.

        Args:
            pose_normalized: (17, 3) normalized pose (unused if precomputed_angles provided)
            visibility: (17,) visibility scores
            precomputed_angles: (12, 3) pre-computed joint angles from TRC

        Returns:
            Dict with 'refined_angles' and 'delta_angles' tensors
        """
        if precomputed_angles is None:
            # No pre-computed angles - return zeros
            # Joint angles require augmented TRC (64 markers) which isn't available here
            return {
                'refined_angles': torch.zeros(1, 12, 3, device=self.device),
                'delta_angles': torch.zeros(1, 12, 3, device=self.device),
            }

        angles = torch.from_numpy(precomputed_angles.astype(np.float32)).unsqueeze(0).to(self.device)
        vis_12 = torch.from_numpy(visibility[:12].astype(np.float32)).unsqueeze(0).to(self.device)

        refined_angles, delta_angles = self.joint_model(angles, vis_12)

        return {
            'refined_angles': refined_angles,
            'delta_angles': delta_angles,
        }


class DepthOnlyRefiner:
    """Simplified pipeline using only the depth model.

    Use this when only the depth model is available (no joint/main models).
    """

    def __init__(
        self,
        depth_checkpoint: Union[str, Path],
        device: Optional[str] = None,
    ):
        """
        Initialize with depth model only.

        Args:
            depth_checkpoint: Path to depth model checkpoint
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        from src.depth_refinement.model import PoseAwareDepthRefiner

        checkpoint_path = Path(depth_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Depth model not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        self.model = PoseAwareDepthRefiner(
            num_joints=config.get('num_joints', 17),
            d_model=config.get('d_model', 64),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 4),
            dim_feedforward=config.get('dim_feedforward', 256),
            dropout=0.0,
            output_confidence=config.get('output_confidence', True),
            use_2d_pose=config.get('use_2d_pose', True),
            use_elepose=config.get('use_elepose', False),
            use_limb_orientations=config.get('use_limb_orientations', False),
        )

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            raise KeyError(f"Checkpoint has no 'model_state_dict' or 'model' key")
        self.model.to(self.device)
        self.model.eval()

        print(f"DepthOnlyRefiner loaded on {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    @torch.no_grad()
    def refine(
        self,
        pose_3d: np.ndarray,
        visibility: np.ndarray,
        pose_2d: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Refine a 3D pose using depth model only.

        Args:
            pose_3d: (17, 3) 3D pose
            visibility: (17,) per-joint visibility
            pose_2d: (17, 2) optional 2D pose (normalized [0,1])

        Returns:
            (17, 3) refined pose
        """
        # Normalize
        pelvis = (pose_3d[11] + pose_3d[12]) / 2
        pose_centered = pose_3d - pelvis

        torso_scale = compute_torso_scale(pose_centered)
        if torso_scale > 0:
            pose_normalized = pose_centered / torso_scale
        else:
            pose_normalized = pose_centered

        # Convert to tensors
        pose_t = torch.from_numpy(pose_normalized.astype(np.float32)).unsqueeze(0).to(self.device)
        vis_t = torch.from_numpy(visibility.astype(np.float32)).unsqueeze(0).to(self.device)

        if pose_2d is not None:
            pose_2d_t = torch.from_numpy(pose_2d.astype(np.float32)).unsqueeze(0).to(self.device)
        else:
            pose_2d_t = pose_t[:, :, :2] / pose_t[:, :, 2:3].clamp(min=1e-4)
            pose_2d_t = (pose_2d_t + 1) / 2

        # Run model
        output = self.model(pose_t, vis_t, pose_2d=pose_2d_t)

        # Apply correction
        refined = pose_t + output['delta_xyz']
        refined = refined.squeeze(0).cpu().numpy()

        # Denormalize
        if torso_scale > 0:
            refined = refined * torso_scale
        refined = refined + pelvis

        return refined


if __name__ == '__main__':
    print("MainRefinerPipeline module")
    print("Usage: MainRefinerPipeline(depth_checkpoint, joint_checkpoint, main_checkpoint)")
