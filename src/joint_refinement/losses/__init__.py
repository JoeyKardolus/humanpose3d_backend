"""Loss functions for joint constraint refinement.

This package provides loss functions organized by purpose:
- angle_losses: Basic angle losses (angular_distance, sign, temporal)
- combined: Combined loss classes (JointRefinementLoss, etc.)
"""

from .angle_losses import (
    angular_distance,
    kinematic_chain_loss,
    angle_sign_loss,
    temporal_smoothness_loss,
    sign_accuracy,
    SYMMETRIC_PAIRS,
    KINEMATIC_CHAINS,
)

from .combined import (
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
