"""Loss functions for camera-space POF training.

Primary loss is cosine similarity between predicted and ground truth
POF unit vectors. Additional regularization losses encourage
symmetric limb behavior and consistent predictions.

Least-squares specific losses:
- projection_consistency_loss: Verify solved X,Y match observed 2D (should be ~0)
- scale_factor_regularization: Penalize negative scales (wrong direction)
- solved_depth_loss: Primary training signal for depth accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .constants import LIMB_DEFINITIONS, NUM_LIMBS, LIMB_SWAP_PAIRS


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
    # Ensure predictions are normalized
    pred_pof = F.normalize(pred_pof, dim=-1, eps=1e-6)

    # Cosine similarity: dot product of unit vectors
    cos_sim = (pred_pof * gt_pof).sum(dim=-1)  # (batch, 14)

    # Loss = 1 - cos(theta)
    loss = 1.0 - cos_sim  # (batch, 14)

    # Optional visibility weighting
    if visibility is not None:
        limb_vis = compute_limb_visibility(visibility)  # (batch, 14)
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
    # Ensure normalization
    pred_pof = F.normalize(pred_pof, dim=-1, eps=1e-6)
    gt_pof = F.normalize(gt_pof, dim=-1, eps=1e-6)

    # Cosine of angle
    cos_theta = (pred_pof * gt_pof).sum(dim=-1).clamp(-1.0, 1.0)

    # Convert to degrees
    angle_rad = torch.acos(cos_theta)
    angle_deg = angle_rad * (180.0 / 3.14159265358979)

    return angle_deg


def symmetry_loss(pred_pof: torch.Tensor) -> torch.Tensor:
    """Encourage symmetric behavior for left/right limb pairs.

    For frontal poses, left and right limbs should have similar
    Z-component magnitudes. This soft constraint helps during training.

    Args:
        pred_pof: (batch, 14, 3) predicted unit vectors

    Returns:
        Scalar symmetry loss
    """
    loss = torch.tensor(0.0, device=pred_pof.device, dtype=pred_pof.dtype)

    for left_idx, right_idx in LIMB_SWAP_PAIRS:
        # Compare Z-component magnitudes (depth extent)
        left_z_mag = pred_pof[:, left_idx, 2].abs()
        right_z_mag = pred_pof[:, right_idx, 2].abs()
        loss = loss + (left_z_mag - right_z_mag).abs().mean()

    return loss / len(LIMB_SWAP_PAIRS)


def smoothness_loss(
    pred_pof: torch.Tensor,
    prev_pof: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Temporal smoothness loss for video sequences.

    Penalizes large changes between consecutive frames.

    Args:
        pred_pof: (batch, 14, 3) current frame predictions
        prev_pof: (batch, 14, 3) previous frame predictions

    Returns:
        Smoothness loss (0 if prev_pof is None)
    """
    if prev_pof is None:
        return torch.tensor(0.0, device=pred_pof.device, dtype=pred_pof.dtype)

    # L2 distance between consecutive predictions
    diff = pred_pof - prev_pof
    return (diff ** 2).sum(dim=-1).mean()


class CameraPOFLoss(nn.Module):
    """Combined loss for camera-space POF training.

    Components:
    1. Cosine similarity loss (primary)
    2. Symmetric limb consistency (soft constraint)

    Optional:
    3. Temporal smoothness (for video training)
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
        """Compute all loss components.

        Args:
            pred_pof: (batch, 14, 3) predicted POF vectors
            gt_pof: (batch, 14, 3) ground truth POF vectors
            visibility: Optional (batch, 17) joint visibility
            prev_pof: Optional (batch, 14, 3) previous frame for smoothness

        Returns:
            Dictionary with individual loss components and total
        """
        losses = {}

        # Primary cosine loss
        losses["cosine"] = pof_cosine_loss(pred_pof, gt_pof, visibility)

        # Symmetry regularization
        if self.symmetry_weight > 0:
            losses["symmetry"] = symmetry_loss(pred_pof)
        else:
            losses["symmetry"] = torch.tensor(
                0.0, device=pred_pof.device, dtype=pred_pof.dtype
            )

        # Temporal smoothness (optional)
        if self.smoothness_weight > 0 and prev_pof is not None:
            losses["smoothness"] = smoothness_loss(pred_pof, prev_pof)
        else:
            losses["smoothness"] = torch.tensor(
                0.0, device=pred_pof.device, dtype=pred_pof.dtype
            )

        # Total weighted loss
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
        """Compute evaluation metrics (not for backprop).

        Args:
            pred_pof: (batch, 14, 3) predicted POF vectors
            gt_pof: (batch, 14, 3) ground truth POF vectors

        Returns:
            Dictionary with metric values
        """
        with torch.no_grad():
            angular_err = pof_angular_error(pred_pof, gt_pof)

            return {
                "mean_angular_error_deg": float(angular_err.mean()),
                "max_angular_error_deg": float(angular_err.max()),
                "median_angular_error_deg": float(angular_err.median()),
            }


# ============================================================================
# Least-Squares Solver Losses
# ============================================================================


def projection_consistency_loss(
    solved_pose: torch.Tensor,
    observed_2d: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Verify solved X,Y match observed 2D positions.

    This loss should be approximately 0 if the LS solver is working correctly,
    since the solver keeps X,Y fixed from 2D observations by construction.

    Useful for debugging and as a sanity check during training.

    Args:
        solved_pose: (batch, 17, 3) solved 3D pose (normalized)
        observed_2d: (batch, 17, 2) observed 2D positions (normalized)
        visibility: Optional (batch, 17) visibility mask
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Projection consistency loss (should be ~0)
    """
    # Extract X,Y from solved pose
    projected_2d = solved_pose[:, :, :2]  # (batch, 17, 2)

    # Compute squared error
    error = (projected_2d - observed_2d) ** 2  # (batch, 17, 2)
    error = error.sum(dim=-1)  # (batch, 17)

    # Apply visibility weighting if provided
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
    # Penalize negative scales using ReLU
    neg_scales = F.relu(-scale_factors)  # (batch, 14)

    # Optionally weight by bone length (bigger bones, bigger penalty)
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
    # Extract Z coordinates
    solved_z = solved_pose[:, :, 2]  # (batch, 17)
    gt_z = gt_pose[:, :, 2]          # (batch, 17)

    # Compute loss
    if loss_type == "l1":
        error = (solved_z - gt_z).abs()
    elif loss_type == "l2":
        error = (solved_z - gt_z) ** 2
    elif loss_type == "smooth_l1":
        error = F.smooth_l1_loss(solved_z, gt_z, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Apply visibility weighting if provided
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
    # Compute per-joint error
    if loss_type == "l1":
        error = (solved_pose - gt_pose).abs().sum(dim=-1)  # (batch, 17)
    elif loss_type == "l2":
        error = ((solved_pose - gt_pose) ** 2).sum(dim=-1)  # (batch, 17)
    elif loss_type == "smooth_l1":
        error = F.smooth_l1_loss(
            solved_pose, gt_pose, reduction="none"
        ).sum(dim=-1)  # (batch, 17)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Apply visibility weighting if provided
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
        projection_weight: float = 0.0,  # Usually 0, just for debugging
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
        """Compute all loss components.

        Args:
            pred_pof: (batch, 14, 3) predicted POF vectors
            gt_pof: (batch, 14, 3) ground truth POF vectors
            solved_pose: (batch, 17, 3) LS-solved pose (optional)
            gt_pose: (batch, 17, 3) ground truth pose (optional)
            observed_2d: (batch, 17, 2) observed 2D positions (optional)
            scale_factors: (batch, 14) scale factors from LS solver (optional)
            visibility: (batch, 17) joint visibility (optional)

        Returns:
            Dictionary with individual loss components and total
        """
        device = pred_pof.device
        dtype = pred_pof.dtype
        losses = {}

        # Primary POF cosine loss
        losses["cosine"] = pof_cosine_loss(pred_pof, gt_pof, visibility)

        # Symmetry regularization
        if self.symmetry_weight > 0:
            losses["symmetry"] = symmetry_loss(pred_pof)
        else:
            losses["symmetry"] = torch.tensor(0.0, device=device, dtype=dtype)

        # Depth loss (if solved pose and GT provided)
        if solved_pose is not None and gt_pose is not None and self.depth_weight > 0:
            losses["depth"] = solved_depth_loss(
                solved_pose, gt_pose, visibility, self.depth_loss_type
            )
        else:
            losses["depth"] = torch.tensor(0.0, device=device, dtype=dtype)

        # Scale factor regularization (if provided)
        if scale_factors is not None and self.scale_reg_weight > 0:
            losses["scale_reg"] = scale_factor_regularization(scale_factors)
        else:
            losses["scale_reg"] = torch.tensor(0.0, device=device, dtype=dtype)

        # Projection consistency (sanity check)
        if solved_pose is not None and observed_2d is not None and self.projection_weight > 0:
            losses["projection"] = projection_consistency_loss(
                solved_pose, observed_2d, visibility
            )
        else:
            losses["projection"] = torch.tensor(0.0, device=device, dtype=dtype)

        # Total weighted loss
        losses["total"] = (
            self.cosine_weight * losses["cosine"]
            + self.symmetry_weight * losses["symmetry"]
            + self.depth_weight * losses["depth"]
            + self.scale_reg_weight * losses["scale_reg"]
            + self.projection_weight * losses["projection"]
        )

        return losses
