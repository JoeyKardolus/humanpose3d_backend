"""Loss functions for joint constraint refinement.

This module re-exports from the losses/ subpackage for backward compatibility.
New code should import directly from src.joint_refinement.losses.
"""

# Re-export everything from the subpackage
from .losses import (
    # Basic losses
    angular_distance,
    kinematic_chain_loss,
    angle_sign_loss,
    temporal_smoothness_loss,
    sign_accuracy,
    # Constants
    SYMMETRIC_PAIRS,
    KINEMATIC_CHAINS,
    # Combined loss classes
    JointRefinementLoss,
    JointRefinementLossWithConstraints,
    GNNJointRefinementLoss,
)

__all__ = [
    # Basic losses
    "angular_distance",
    "kinematic_chain_loss",
    "angle_sign_loss",
    "temporal_smoothness_loss",
    "sign_accuracy",
    # Constants
    "SYMMETRIC_PAIRS",
    "KINEMATIC_CHAINS",
    # Combined loss classes
    "JointRefinementLoss",
    "JointRefinementLossWithConstraints",
    "GNNJointRefinementLoss",
]
