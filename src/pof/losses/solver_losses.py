"""Least-squares solver specific loss functions.

Contains losses for training POF models with depth solving:
- projection_consistency_loss: Verify solved X,Y match observed 2D
- scale_factor_regularization: Penalize negative scales
- solved_depth_loss: Primary training signal for depth accuracy
- full_pose_loss: Full 3D pose loss (X, Y, Z)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def projection_consistency_loss(
    solved_pose: torch.Tensor,
    observed_2d: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Verify solved X,Y match observed 2D positions.

    This loss should be approximately 0 if the LS solver is working correctly,
    since the solver keeps X,Y fixed from 2D observations by construction.

    Args:
        solved_pose: (batch, 17, 3) solved 3D pose (normalized)
        observed_2d: (batch, 17, 2) observed 2D positions (normalized)
        visibility: Optional (batch, 17) visibility mask
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Projection consistency loss (should be ~0)
    """
    projected_2d = solved_pose[:, :, :2]
    error = (projected_2d - observed_2d) ** 2
    error = error.sum(dim=-1)

    if visibility is not None:
        error = error * visibility

        if reduction == "mean":
            return error.sum() / visibility.sum().clamp(min=1e-6)
        elif reduction == "sum":
            return error.sum()
        return error

    if reduction == "mean":
        return error.mean()
    elif reduction == "sum":
        return error.sum()
    return error


def scale_factor_regularization(
    scale_factors: torch.Tensor,
    bone_lengths: Optional[torch.Tensor] = None,
    negative_penalty: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize negative scale factors (limb pointing wrong direction).

    When the LS solver produces a negative scale, it means the predicted
    limb orientation is reversed relative to the 2D observations.

    Args:
        scale_factors: (batch, 14) scale factors from LS solver
        bone_lengths: Optional (14,) or (batch, 14) for relative penalty
        negative_penalty: Weight for negative scales
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Regularization loss
    """
    neg_scales = F.relu(-scale_factors)

    if bone_lengths is not None:
        if bone_lengths.dim() == 1:
            bone_lengths = bone_lengths.unsqueeze(0)
        neg_scales = neg_scales * bone_lengths

    loss = negative_penalty * neg_scales

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def solved_depth_loss(
    solved_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    loss_type: str = "l1",
    reduction: str = "mean",
) -> torch.Tensor:
    """Primary training signal: solved Z should match ground truth.

    This is the main loss for training the depth solver to produce
    accurate depth values.

    Args:
        solved_pose: (batch, 17, 3) solved 3D pose
        gt_pose: (batch, 17, 3) ground truth 3D pose
        visibility: Optional (batch, 17) visibility mask
        loss_type: 'l1', 'l2', or 'smooth_l1'
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Depth loss
    """
    solved_z = solved_pose[:, :, 2]
    gt_z = gt_pose[:, :, 2]

    if loss_type == "l1":
        error = (solved_z - gt_z).abs()
    elif loss_type == "l2":
        error = (solved_z - gt_z) ** 2
    elif loss_type == "smooth_l1":
        error = F.smooth_l1_loss(solved_z, gt_z, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if visibility is not None:
        error = error * visibility

        if reduction == "mean":
            return error.sum() / visibility.sum().clamp(min=1e-6)
        elif reduction == "sum":
            return error.sum()
        return error

    if reduction == "mean":
        return error.mean()
    elif reduction == "sum":
        return error.sum()
    return error


def full_pose_loss(
    solved_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    loss_type: str = "l1",
    reduction: str = "mean",
) -> torch.Tensor:
    """Full 3D pose loss (X, Y, Z).

    Computes loss on all coordinates, not just depth.

    Args:
        solved_pose: (batch, 17, 3) solved 3D pose
        gt_pose: (batch, 17, 3) ground truth 3D pose
        visibility: Optional (batch, 17) visibility mask
        loss_type: 'l1', 'l2', or 'smooth_l1'
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Full pose loss
    """
    if loss_type == "l1":
        error = (solved_pose - gt_pose).abs().sum(dim=-1)
    elif loss_type == "l2":
        error = ((solved_pose - gt_pose) ** 2).sum(dim=-1)
    elif loss_type == "smooth_l1":
        error = F.smooth_l1_loss(
            solved_pose, gt_pose, reduction="none"
        ).sum(dim=-1)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if visibility is not None:
        error = error * visibility

        if reduction == "mean":
            return error.sum() / visibility.sum().clamp(min=1e-6)
        elif reduction == "sum":
            return error.sum()
        return error

    if reduction == "mean":
        return error.mean()
    elif reduction == "sum":
        return error.sum()
    return error
