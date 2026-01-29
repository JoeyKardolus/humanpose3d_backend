"""Core POF loss functions.

Contains losses for POF direction, symmetry, and z-sign classification.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from ..constants import LIMB_DEFINITIONS, NUM_LIMBS, LIMB_SWAP_PAIRS


def compute_limb_visibility(visibility: torch.Tensor) -> torch.Tensor:
    """Compute per-limb visibility from joint visibility.

    Limb visibility = min(parent_visibility, child_visibility)

    Args:
        visibility: (batch, 17) per-joint visibility scores

    Returns:
        (batch, 14) per-limb visibility scores
    """
    batch_size = visibility.size(0)
    limb_vis = torch.zeros(
        batch_size, NUM_LIMBS,
        device=visibility.device,
        dtype=visibility.dtype,
    )

    for limb_idx, (parent, child) in enumerate(LIMB_DEFINITIONS):
        limb_vis[:, limb_idx] = torch.min(
            visibility[:, parent],
            visibility[:, child],
        )

    return limb_vis


def pof_cosine_loss(
    pred_pof: torch.Tensor,
    gt_pof: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cosine similarity loss for POF unit vectors.

    Loss = 1 - cos(angle) where cos = dot(pred, gt)
    - Perfect alignment: loss = 0
    - Orthogonal: loss = 1
    - Opposite direction: loss = 2

    Args:
        pred_pof: (batch, 14, 3) predicted unit vectors
        gt_pof: (batch, 14, 3) ground truth unit vectors
        visibility: Optional (batch, 17) joint visibility for weighting
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Cosine loss value
    """
    pred_pof = F.normalize(pred_pof, dim=-1, eps=1e-6)
    cos_sim = (pred_pof * gt_pof).sum(dim=-1)
    loss = 1.0 - cos_sim

    if visibility is not None:
        limb_vis = compute_limb_visibility(visibility)
        loss = loss * limb_vis

        if reduction == "mean":
            return loss.sum() / limb_vis.sum().clamp(min=1e-6)
        elif reduction == "sum":
            return loss.sum()
        return loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def pof_angular_error(
    pred_pof: torch.Tensor,
    gt_pof: torch.Tensor,
) -> torch.Tensor:
    """Compute angular error in degrees between predicted and GT POF.

    Args:
        pred_pof: (batch, 14, 3) predicted unit vectors
        gt_pof: (batch, 14, 3) ground truth unit vectors

    Returns:
        (batch, 14) angular error in degrees
    """
    pred_pof = F.normalize(pred_pof, dim=-1, eps=1e-6)
    gt_pof = F.normalize(gt_pof, dim=-1, eps=1e-6)

    cos_theta = (pred_pof * gt_pof).sum(dim=-1).clamp(-1.0, 1.0)
    angle_rad = torch.acos(cos_theta)
    angle_deg = angle_rad * (180.0 / 3.14159265358979)

    return angle_deg


def symmetry_loss(pred_pof: torch.Tensor) -> torch.Tensor:
    """Encourage symmetric behavior for left/right limb pairs.

    For frontal poses, left and right limbs should have similar
    Z-component magnitudes.

    Args:
        pred_pof: (batch, 14, 3) predicted unit vectors

    Returns:
        Scalar symmetry loss
    """
    loss = torch.tensor(0.0, device=pred_pof.device, dtype=pred_pof.dtype)

    for left_idx, right_idx in LIMB_SWAP_PAIRS:
        left_z_mag = pred_pof[:, left_idx, 2].abs()
        right_z_mag = pred_pof[:, right_idx, 2].abs()
        loss = loss + (left_z_mag - right_z_mag).abs().mean()

    return loss / len(LIMB_SWAP_PAIRS)


def z_sign_loss(
    z_sign_logits: torch.Tensor,
    gt_pof: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary cross-entropy loss for Z-sign classification.

    Args:
        z_sign_logits: (batch, 14) raw logits for Z > 0 prediction
        gt_pof: (batch, 14, 3) ground truth POF vectors
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Binary cross-entropy loss
    """
    z_sign_gt = (gt_pof[:, :, 2] > 0).float()
    loss = F.binary_cross_entropy_with_logits(
        z_sign_logits, z_sign_gt, reduction="none"
    )

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def z_magnitude_loss(
    pred_z_mag: torch.Tensor,
    gt_pof: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE loss on |Z| magnitude (foreshortening).

    Args:
        pred_z_mag: (batch, 14) predicted |Z| magnitudes in [0, 1]
        gt_pof: (batch, 14, 3) ground truth POF unit vectors
        reduction: 'mean', 'sum', or 'none'

    Returns:
        MSE loss on |Z| magnitude
    """
    gt_z_mag = gt_pof[:, :, 2].abs()
    return F.mse_loss(pred_z_mag, gt_z_mag, reduction=reduction)


def z_magnitude_l1_loss(
    pred_z_mag: torch.Tensor,
    gt_pof: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """L1 loss on |Z| magnitude (foreshortening).

    Args:
        pred_z_mag: (batch, 14) predicted |Z| magnitudes in [0, 1]
        gt_pof: (batch, 14, 3) ground truth POF unit vectors
        reduction: 'mean', 'sum', or 'none'

    Returns:
        L1 loss on |Z| magnitude
    """
    gt_z_mag = gt_pof[:, :, 2].abs()
    return F.l1_loss(pred_z_mag, gt_z_mag, reduction=reduction)


def z_sign_accuracy(
    z_sign_logits: torch.Tensor,
    gt_pof: torch.Tensor,
) -> torch.Tensor:
    """Compute Z-sign classification accuracy.

    Args:
        z_sign_logits: (batch, 14) raw logits
        gt_pof: (batch, 14, 3) ground truth POF vectors

    Returns:
        Accuracy as a scalar tensor
    """
    with torch.no_grad():
        z_sign_gt = (gt_pof[:, :, 2] > 0).float()
        z_sign_pred = (torch.sigmoid(z_sign_logits) > 0.5).float()
        return (z_sign_pred == z_sign_gt).float().mean()


def smoothness_loss(
    pred_pof: torch.Tensor,
    prev_pof: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Temporal smoothness loss for video sequences.

    Args:
        pred_pof: (batch, 14, 3) current frame predictions
        prev_pof: (batch, 14, 3) previous frame predictions

    Returns:
        Smoothness loss (0 if prev_pof is None)
    """
    if prev_pof is None:
        return torch.tensor(0.0, device=pred_pof.device, dtype=pred_pof.dtype)

    diff = pred_pof - prev_pof
    return (diff ** 2).sum(dim=-1).mean()
