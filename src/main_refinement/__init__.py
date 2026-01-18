"""
Main Refinement Module - Learned Constraint Fusion.

This module provides the MainRefiner model that learns to optimally combine
outputs from the depth refinement and joint constraint models.

Architecture:
- Takes raw 3D pose + outputs from both constraint models
- Uses learned gating to decide when to trust each constraint source
- Cross-attention allows depth and joint features to inform each other
- Outputs final refined pose with per-joint confidence

Example usage:
    from src.main_refinement import MainRefiner, MainRefinerPipeline

    # Training: load individual models, create combined dataset
    pipeline = MainRefinerPipeline(
        depth_checkpoint='models/checkpoints/best_depth_model.pth',
        joint_checkpoint='models/checkpoints/best_joint_model.pth',
    )
    refined_pose = pipeline.refine(pose_3d, visibility, pose_2d)

    # Or just the fusion model (for training):
    from src.main_refinement.model import MainRefiner
    model = MainRefiner(d_model=128)
"""

from .model import (
    MainRefiner,
    DepthOutputEncoder,
    JointOutputEncoder,
    GatingNetwork,
    FusionHead,
    CrossModelAttention,
    LimbToJointMapper,
    Joint12To17Mapper,
)
from .losses import MainRefinerLoss
from .inference import MainRefinerPipeline

__all__ = [
    # Model components
    'MainRefiner',
    'DepthOutputEncoder',
    'JointOutputEncoder',
    'GatingNetwork',
    'FusionHead',
    'CrossModelAttention',
    'LimbToJointMapper',
    'Joint12To17Mapper',
    # Losses
    'MainRefinerLoss',
    # Inference
    'MainRefinerPipeline',
]
