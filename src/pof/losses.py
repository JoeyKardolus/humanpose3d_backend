"""Loss functions for camera-space POF training.

This module re-exports from the losses/ subpackage for backward compatibility.
New code should import directly from src.pof.losses.
"""

# Re-export everything from the subpackage
from .losses import (
    # POF losses
    pof_cosine_loss,
    pof_angular_error,
    compute_limb_visibility,
    symmetry_loss,
    z_sign_loss,
    z_magnitude_loss,
    z_magnitude_l1_loss,
    z_sign_accuracy,
    smoothness_loss,
    # Solver losses
    projection_consistency_loss,
    scale_factor_regularization,
    solved_depth_loss,
    full_pose_loss,
    # Combined loss classes
    CameraPOFLoss,
    TemporalPOFLoss,
    LeastSquaresPOFLoss,
    ZSignOnlyLoss,
    CleanSeparationPOFLoss,
)

__all__ = [
    # POF losses
    "pof_cosine_loss",
    "pof_angular_error",
    "compute_limb_visibility",
    "symmetry_loss",
    "z_sign_loss",
    "z_magnitude_loss",
    "z_magnitude_l1_loss",
    "z_sign_accuracy",
    "smoothness_loss",
    # Solver losses
    "projection_consistency_loss",
    "scale_factor_regularization",
    "solved_depth_loss",
    "full_pose_loss",
    # Combined loss classes
    "CameraPOFLoss",
    "TemporalPOFLoss",
    "LeastSquaresPOFLoss",
    "ZSignOnlyLoss",
    "CleanSeparationPOFLoss",
]
