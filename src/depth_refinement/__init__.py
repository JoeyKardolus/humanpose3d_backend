"""
Depth Refinement Module.

Trains and applies neural depth correction to MediaPipe poses using:
- REAL depth errors from AIST++ video/mocap pairs
- View angle conditioning (computed from torso orientation)
- Cross-joint attention for pose-aware inference

Usage:
    # Training
    uv run --group neural python scripts/train_depth_model.py

    # Inference
    from src.depth_refinement.inference import DepthRefiner
    refiner = DepthRefiner('models/checkpoints/best_model.pth')
    refined_pose = refiner.refine(pose, visibility)
"""

from .model import PoseAwareDepthRefiner, create_model
from .losses import DepthRefinementLoss
from .dataset import AISTPPDepthDataset, create_dataloaders
from .inference import DepthRefiner

__all__ = [
    'PoseAwareDepthRefiner',
    'create_model',
    'DepthRefinementLoss',
    'AISTPPDepthDataset',
    'create_dataloaders',
    'DepthRefiner',
]
