"""Loss functions for camera-space POF training.

This package provides loss functions organized by purpose:
- pof_losses: Core POF losses (cosine, angular error, symmetry, z-sign)
- solver_losses: Least-squares solver losses (depth, projection, scale)
- combined: Combined loss classes (CameraPOFLoss, TemporalPOFLoss, etc.)
"""

from .pof_losses import (
    pof_cosine_loss,
    pof_angular_error,
    compute_limb_visibility,
    symmetry_loss,
    z_sign_loss,
    z_magnitude_loss,
    z_magnitude_l1_loss,
    z_sign_accuracy,
    smoothness_loss,
)

from .solver_losses import (
    projection_consistency_loss,
    scale_factor_regularization,
    solved_depth_loss,
    full_pose_loss,
)

from .combined import (
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
