"""
Inference module for joint constraint refinement.

Supports both transformer-based and GNN-based models with auto-detection.

Usage:
    from src.joint_refinement.inference import JointRefiner

    # Auto-detect model type from checkpoint
    refiner = JointRefiner('models/checkpoints/best_joint_model.pth')
    refined_angles = refiner.refine(angles, visibility)

    # Explicit model type
    refiner = JointRefiner('models/checkpoints/best_joint_gnn_model.pth', model_type='semgcn-temporal')

The refiner loads the trained model and applies soft learned constraints
to refine joint angles computed by the validated ISB kinematics.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union

from .model import create_model, JointConstraintRefiner
from .gnn_model import create_gnn_joint_model, SemGCNTemporalJointRefiner


# Joint order (must match training)
JOINT_NAMES = [
    'pelvis', 'hip_R', 'hip_L', 'knee_R', 'knee_L',
    'ankle_R', 'ankle_L', 'trunk', 'shoulder_R', 'shoulder_L',
    'elbow_R', 'elbow_L',
]


class JointRefiner:
    """Joint constraint refinement inference wrapper.

    Supports both transformer-based and GNN-based models.
    Auto-detects model type from checkpoint if not specified.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'auto',
        model_type: str = 'auto',
    ):
        """
        Initialize the refiner.

        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda', 'cpu', or 'auto' (auto-detect)
            model_type: 'auto', 'transformer', 'gcn', 'semgcn', or 'semgcn-temporal'
        """
        self.model_path = Path(model_path)
        self.model_type_requested = model_type

        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model
        self.model, self.model_type = self._load_model()
        self.model.eval()

        # For temporal models, track previous angles
        self.prev_angles = None

        print(f"[JointRefiner] Loaded {self.model_type} model from {model_path}")
        print(f"[JointRefiner] Device: {self.device}")
        print(f"[JointRefiner] Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _load_model(self):
        """Load trained model from checkpoint with auto-detection."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Detect model type
        if self.model_type_requested == 'auto':
            # Check for config in checkpoint
            config = checkpoint.get('config', {})
            model_type = config.get('model_type', None)

            if model_type is None:
                # Fallback: check for d_model vs num_layers naming
                if 'd_model' in checkpoint and 'n_layers' in checkpoint:
                    model_type = 'transformer'
                elif 'config' in checkpoint:
                    model_type = checkpoint['config'].get('model_type', 'transformer')
                else:
                    model_type = 'transformer'
        else:
            model_type = self.model_type_requested

        # Create model based on type
        if model_type == 'transformer':
            d_model = checkpoint.get('d_model', 128)
            n_layers = checkpoint.get('n_layers', 4)
            model = create_model(d_model=d_model, n_layers=n_layers)
        else:
            # GNN models
            config = checkpoint.get('config', {})
            model = create_gnn_joint_model(
                model_type=model_type,
                d_model=config.get('d_model', 192),
                num_layers=config.get('num_layers', 4),
                use_gat=config.get('use_gat', False),
                verbose=False,
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        return model, model_type

    def refine(
        self,
        angles: np.ndarray,
        visibility: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Refine joint angles using learned constraints.

        For temporal models (semgcn-temporal), maintains internal state
        to track previous frame angles for context.

        Args:
            angles: (12, 3) or (N, 12, 3) joint angles in degrees
            visibility: (12,) or (N, 12) per-joint visibility (0-1, optional)

        Returns:
            Refined angles with same shape as input
        """
        # Handle single frame vs batch
        single_frame = angles.ndim == 2
        if single_frame:
            angles = angles[np.newaxis, ...]
            if visibility is not None:
                visibility = visibility[np.newaxis, ...]

        # Convert to tensors
        angles_t = torch.from_numpy(angles.astype(np.float32)).to(self.device)

        if visibility is not None:
            visibility_t = torch.from_numpy(visibility.astype(np.float32)).to(self.device)
        else:
            visibility_t = None

        # Run model
        with torch.no_grad():
            if self.model_type == 'semgcn-temporal':
                # Temporal model with previous frame context
                refined, delta, _ = self.model(angles_t, visibility_t, self.prev_angles)
                # Update state for next frame
                self.prev_angles = refined.detach()
            elif self.model_type in ('gcn', 'semgcn'):
                # GNN models without temporal
                refined, delta = self.model(angles_t, visibility_t)
            else:
                # Transformer model
                refined, delta = self.model(angles_t, visibility_t)

        # Convert back to numpy
        result = refined.cpu().numpy()

        if single_frame:
            result = result[0]

        return result

    def reset(self):
        """Reset temporal state. Call at start of new video sequence."""
        self.prev_angles = None

    def refine_batch(
        self,
        angles: np.ndarray,
        visibility: Optional[np.ndarray] = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Refine a large batch of angles efficiently.

        Args:
            angles: (N, 12, 3) joint angles in degrees
            visibility: (N, 12) per-joint visibility (optional)
            batch_size: Batch size for processing

        Returns:
            (N, 12, 3) refined angles
        """
        N = len(angles)
        results = []

        for i in range(0, N, batch_size):
            batch_angles = angles[i:i+batch_size]
            batch_vis = visibility[i:i+batch_size] if visibility is not None else None
            refined = self.refine(batch_angles, batch_vis)
            results.append(refined)

        return np.concatenate(results, axis=0)

    def get_delta(
        self,
        angles: np.ndarray,
        visibility: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get the predicted correction (delta) without applying it.

        Useful for analysis and visualization.

        Args:
            angles: (12, 3) or (N, 12, 3) joint angles in degrees
            visibility: (12,) or (N, 12) per-joint visibility (optional)

        Returns:
            Delta (correction) with same shape as input
        """
        # Handle single frame vs batch
        single_frame = angles.ndim == 2
        if single_frame:
            angles = angles[np.newaxis, ...]
            if visibility is not None:
                visibility = visibility[np.newaxis, ...]

        # Convert to tensors
        angles_t = torch.from_numpy(angles.astype(np.float32)).to(self.device)

        if visibility is not None:
            visibility_t = torch.from_numpy(visibility.astype(np.float32)).to(self.device)
        else:
            visibility_t = None

        # Run model
        with torch.no_grad():
            _, delta = self.model(angles_t, visibility_t)

        # Convert back to numpy
        result = delta.cpu().numpy()

        if single_frame:
            result = result[0]

        return result


def load_refiner(
    model_path: Union[str, Path] = 'models/checkpoints/best_joint_model.pth',
    device: str = 'auto',
    model_type: str = 'auto',
) -> JointRefiner:
    """
    Load a joint refiner from checkpoint.

    Args:
        model_path: Path to trained model
        device: Device to use
        model_type: 'auto', 'transformer', 'gcn', 'semgcn', or 'semgcn-temporal'

    Returns:
        JointRefiner instance
    """
    return JointRefiner(model_path, device, model_type)


if __name__ == '__main__':
    import sys

    # Test inference
    model_path = StoragePaths.load().checkpoints_root / "best_joint_model.pth"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: uv run python scripts/train_joint_model.py")
        sys.exit(1)

    # Load refiner
    refiner = JointRefiner(model_path)

    # Test with random angles
    test_angles = np.random.randn(12, 3) * 30  # Random angles
    test_vis = np.random.rand(12)  # Random visibility

    print(f"\nInput angles shape: {test_angles.shape}")
    print(f"Input visibility shape: {test_vis.shape}")

    # Refine
    refined = refiner.refine(test_angles, test_vis)
    delta = refiner.get_delta(test_angles, test_vis)

    print(f"\nRefined angles shape: {refined.shape}")
    print(f"Delta shape: {delta.shape}")

    print(f"\nDelta statistics:")
    print(f"  Mean: {np.abs(delta).mean():.2f}°")
    print(f"  Max: {np.abs(delta).max():.2f}°")

    # Test batch processing
    batch_angles = np.random.randn(100, 12, 3) * 30
    batch_vis = np.random.rand(100, 12)

    print(f"\nBatch processing test:")
    print(f"  Input: {batch_angles.shape}")
    refined_batch = refiner.refine_batch(batch_angles, batch_vis, batch_size=32)
    print(f"  Output: {refined_batch.shape}")
