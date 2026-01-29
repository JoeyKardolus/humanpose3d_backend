"""Combined loss classes for POF training.

Contains loss classes that combine multiple loss functions:
- CameraPOFLoss: Basic POF training loss
- TemporalPOFLoss: POF with z-sign auxiliary task
- LeastSquaresPOFLoss: POF with depth solving
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .pof_losses import (
    pof_cosine_loss,
    pof_angular_error,
    symmetry_loss,
    z_sign_loss,
    z_sign_accuracy,
    smoothness_loss,
)
from .solver_losses import (
    projection_consistency_loss,
    scale_factor_regularization,
    solved_depth_loss,
)


class TemporalPOFLoss(nn.Module):
    """Combined loss for temporal POF model with Z-sign auxiliary task.

    Components:
    1. Cosine similarity loss (primary POF direction)
    2. Z-sign classification loss (depth direction)
    3. Symmetric limb consistency (soft constraint)
    4. Temporal smoothness (for video training)
    """

    def __init__(
        self,
        cosine_weight: float = 1.0,
        z_sign_weight: float = 0.2,
        symmetry_weight: float = 0.1,
        smoothness_weight: float = 0.0,
    ):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.z_sign_weight = z_sign_weight
        self.symmetry_weight = symmetry_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred_pof: torch.Tensor,
        z_sign_logits: torch.Tensor,
        gt_pof: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
        prev_pof: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        losses = {}
        device = pred_pof.device
        dtype = pred_pof.dtype

        losses["cosine"] = pof_cosine_loss(pred_pof, gt_pof, visibility)

        if self.z_sign_weight > 0:
            losses["z_sign"] = z_sign_loss(z_sign_logits, gt_pof)
            losses["z_sign_acc"] = z_sign_accuracy(z_sign_logits, gt_pof)
        else:
            losses["z_sign"] = torch.tensor(0.0, device=device, dtype=dtype)
            losses["z_sign_acc"] = torch.tensor(0.0, device=device)

        if self.symmetry_weight > 0:
            losses["symmetry"] = symmetry_loss(pred_pof)
        else:
            losses["symmetry"] = torch.tensor(0.0, device=device, dtype=dtype)

        if self.smoothness_weight > 0 and prev_pof is not None:
            losses["smoothness"] = smoothness_loss(pred_pof, prev_pof)
        else:
            losses["smoothness"] = torch.tensor(0.0, device=device, dtype=dtype)

        losses["total"] = (
            self.cosine_weight * losses["cosine"]
            + self.z_sign_weight * losses["z_sign"]
            + self.symmetry_weight * losses["symmetry"]
            + self.smoothness_weight * losses["smoothness"]
        )

        return losses


class ZSignOnlyLoss(nn.Module):
    """DEPRECATED: Loss for Z-sign only classification model."""

    def __init__(self):
        super().__init__()
        import warnings
        warnings.warn(
            "ZSignOnlyLoss is deprecated. Use TemporalPOFLoss instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def forward(
        self,
        z_sign_logits: torch.Tensor,
        gt_pof: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute Z-sign classification loss."""
        losses = {}
        losses["z_sign"] = z_sign_loss(z_sign_logits, gt_pof)
        losses["z_sign_acc"] = z_sign_accuracy(z_sign_logits, gt_pof)
        losses["total"] = losses["z_sign"]
        return losses


class CleanSeparationPOFLoss(nn.Module):
    """DEPRECATED: Use ZSignOnlyLoss instead."""

    def __init__(
        self,
        z_mag_weight: float = 1.0,
        z_sign_weight: float = 0.2,
        use_l1: bool = False,
    ):
        super().__init__()
        self.z_sign_weight = z_sign_weight
        import warnings
        warnings.warn(
            "CleanSeparationPOFLoss is deprecated. Use ZSignOnlyLoss instead.",
            DeprecationWarning
        )

    def forward(
        self,
        z_sign_logits: torch.Tensor,
        gt_pof: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute Z-sign loss only."""
        losses = {}
        losses["z_sign"] = z_sign_loss(z_sign_logits, gt_pof)
        losses["z_sign_acc"] = z_sign_accuracy(z_sign_logits, gt_pof)
        losses["total"] = self.z_sign_weight * losses["z_sign"]
        return losses


class CameraPOFLoss(nn.Module):
    """Combined loss for camera-space POF training.

    Components:
    1. Cosine similarity loss (primary)
    2. Symmetric limb consistency (soft constraint)
    3. Temporal smoothness (optional, for video training)
    """

    def __init__(
        self,
        cosine_weight: float = 1.0,
        symmetry_weight: float = 0.1,
        smoothness_weight: float = 0.0,
    ):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.symmetry_weight = symmetry_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred_pof: torch.Tensor,
        gt_pof: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
        prev_pof: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        losses = {}
        device = pred_pof.device
        dtype = pred_pof.dtype

        losses["cosine"] = pof_cosine_loss(pred_pof, gt_pof, visibility)

        if self.symmetry_weight > 0:
            losses["symmetry"] = symmetry_loss(pred_pof)
        else:
            losses["symmetry"] = torch.tensor(0.0, device=device, dtype=dtype)

        if self.smoothness_weight > 0 and prev_pof is not None:
            losses["smoothness"] = smoothness_loss(pred_pof, prev_pof)
        else:
            losses["smoothness"] = torch.tensor(0.0, device=device, dtype=dtype)

        losses["total"] = (
            self.cosine_weight * losses["cosine"]
            + self.symmetry_weight * losses["symmetry"]
            + self.smoothness_weight * losses["smoothness"]
        )

        return losses

    def get_metrics(
        self,
        pred_pof: torch.Tensor,
        gt_pof: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        with torch.no_grad():
            angular_err = pof_angular_error(pred_pof, gt_pof)
            return {
                "mean_angular_error_deg": float(angular_err.mean()),
                "max_angular_error_deg": float(angular_err.max()),
                "median_angular_error_deg": float(angular_err.median()),
            }


class LeastSquaresPOFLoss(nn.Module):
    """Combined loss for POF training with least-squares solver.

    Includes:
    1. POF cosine loss (direction accuracy)
    2. Depth loss (Z coordinate accuracy after LS solving)
    3. Scale factor regularization (penalize wrong directions)
    4. Projection consistency (sanity check, should be ~0)
    """

    def __init__(
        self,
        cosine_weight: float = 1.0,
        depth_weight: float = 1.0,
        scale_reg_weight: float = 0.1,
        projection_weight: float = 0.0,
        symmetry_weight: float = 0.1,
        depth_loss_type: str = "l1",
    ):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.depth_weight = depth_weight
        self.scale_reg_weight = scale_reg_weight
        self.projection_weight = projection_weight
        self.symmetry_weight = symmetry_weight
        self.depth_loss_type = depth_loss_type

    def forward(
        self,
        pred_pof: torch.Tensor,
        gt_pof: torch.Tensor,
        solved_pose: Optional[torch.Tensor] = None,
        gt_pose: Optional[torch.Tensor] = None,
        observed_2d: Optional[torch.Tensor] = None,
        scale_factors: Optional[torch.Tensor] = None,
        visibility: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        device = pred_pof.device
        dtype = pred_pof.dtype
        losses = {}

        losses["cosine"] = pof_cosine_loss(pred_pof, gt_pof, visibility)

        if self.symmetry_weight > 0:
            losses["symmetry"] = symmetry_loss(pred_pof)
        else:
            losses["symmetry"] = torch.tensor(0.0, device=device, dtype=dtype)

        if solved_pose is not None and gt_pose is not None and self.depth_weight > 0:
            losses["depth"] = solved_depth_loss(
                solved_pose, gt_pose, visibility, self.depth_loss_type
            )
        else:
            losses["depth"] = torch.tensor(0.0, device=device, dtype=dtype)

        if scale_factors is not None and self.scale_reg_weight > 0:
            losses["scale_reg"] = scale_factor_regularization(scale_factors)
        else:
            losses["scale_reg"] = torch.tensor(0.0, device=device, dtype=dtype)

        if solved_pose is not None and observed_2d is not None and self.projection_weight > 0:
            losses["projection"] = projection_consistency_loss(
                solved_pose, observed_2d, visibility
            )
        else:
            losses["projection"] = torch.tensor(0.0, device=device, dtype=dtype)

        losses["total"] = (
            self.cosine_weight * losses["cosine"]
            + self.symmetry_weight * losses["symmetry"]
            + self.depth_weight * losses["depth"]
            + self.scale_reg_weight * losses["scale_reg"]
            + self.projection_weight * losses["projection"]
        )

        return losses
