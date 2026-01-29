"""Combined loss classes for joint refinement.

Contains loss classes that combine multiple loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .angle_losses import (
    angular_distance,
    kinematic_chain_loss,
    angle_sign_loss,
    temporal_smoothness_loss,
    SYMMETRIC_PAIRS,
)


class JointRefinementLoss(nn.Module):
    """Combined loss for joint constraint refinement.

    Components:
    1. Angle reconstruction (L1)
    2. Bilateral symmetry (L1 on symmetric pairs)
    3. Delta regularization (L2 on corrections)
    4. Kinematic chain consistency
    """

    def __init__(
        self,
        symmetry_weight: float = 0.1,
        delta_weight: float = 0.01,
        chain_weight: float = 0.1,
        use_visibility_weighting: bool = True,
        ignore_arm_abd_rot: bool = False,
        ignore_left_arm: bool = False,
        ignore_arms: bool = False,
    ):
        super().__init__()

        self.symmetry_weight = symmetry_weight
        self.delta_weight = delta_weight
        self.chain_weight = chain_weight
        self.use_visibility_weighting = use_visibility_weighting
        self.ignore_arm_abd_rot = ignore_arm_abd_rot
        self.ignore_left_arm = ignore_left_arm
        self.ignore_arms = ignore_arms

        self._dof_mask = None
        if ignore_arm_abd_rot or ignore_left_arm or ignore_arms:
            self.register_buffer('dof_mask', torch.ones(12, 3))
            if ignore_arms:
                for joint_idx in [8, 9, 10, 11]:
                    self.dof_mask[joint_idx, :] = 0.0
            else:
                if ignore_arm_abd_rot:
                    for joint_idx in [8, 9, 10, 11]:
                        self.dof_mask[joint_idx, 1] = 0.0
                        self.dof_mask[joint_idx, 2] = 0.0
                if ignore_left_arm:
                    for joint_idx in [9, 11]:
                        self.dof_mask[joint_idx, :] = 0.0
            self._dof_mask = True

    def forward(
        self,
        refined_angles: torch.Tensor,
        ground_truth_angles: torch.Tensor,
        delta: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses."""
        losses = {}

        if self.use_visibility_weighting and visibility is not None:
            vis_weights = visibility.unsqueeze(-1)
        else:
            vis_weights = torch.ones_like(refined_angles[..., :1])

        # 1. Angle reconstruction loss
        angle_error = angular_distance(refined_angles, ground_truth_angles)
        weighted_error = angle_error * vis_weights
        if self._dof_mask:
            weighted_error = weighted_error * self.dof_mask.unsqueeze(0)
            denom = (vis_weights * self.dof_mask.unsqueeze(0)).sum() + 1e-6
        else:
            denom = vis_weights.sum() * 3 + 1e-6
        reconstruction_loss = weighted_error.sum() / denom
        losses['reconstruction'] = reconstruction_loss

        # 2. Bilateral symmetry loss
        symmetry_loss = torch.tensor(0.0, device=refined_angles.device)

        for r_idx, l_idx in SYMMETRIC_PAIRS:
            r_angles = refined_angles[:, r_idx, :]
            l_angles = refined_angles[:, l_idx, :]

            flex_diff = angular_distance(r_angles[:, 0], l_angles[:, 0])
            abd_diff = angular_distance(r_angles[:, 1], -l_angles[:, 1])
            rot_diff = angular_distance(r_angles[:, 2], -l_angles[:, 2])

            if self.use_visibility_weighting and visibility is not None:
                pair_vis = visibility[:, r_idx] * visibility[:, l_idx]
            else:
                pair_vis = torch.ones(refined_angles.shape[0], device=refined_angles.device)

            pair_loss = (flex_diff + abd_diff + rot_diff) * pair_vis
            symmetry_loss = symmetry_loss + pair_loss.mean()

        symmetry_loss = symmetry_loss / len(SYMMETRIC_PAIRS)
        losses['symmetry'] = symmetry_loss

        # 3. Delta regularization
        delta_l2 = (delta ** 2).mean()
        losses['delta_reg'] = delta_l2

        # 4. Kinematic chain consistency
        chain_loss = kinematic_chain_loss(
            refined_angles, ground_truth_angles, visibility
        )
        losses['chain'] = chain_loss

        # Total loss
        total_loss = (
            reconstruction_loss +
            self.symmetry_weight * symmetry_loss +
            self.delta_weight * delta_l2 +
            self.chain_weight * chain_loss
        )
        losses['total'] = total_loss

        return losses


class JointRefinementLossWithConstraints(JointRefinementLoss):
    """Extended loss with soft learned constraints."""

    def __init__(
        self,
        symmetry_weight: float = 0.1,
        delta_weight: float = 0.01,
        chain_weight: float = 0.1,
        range_weight: float = 0.05,
        use_visibility_weighting: bool = True,
    ):
        super().__init__(symmetry_weight, delta_weight, chain_weight, use_visibility_weighting)
        self.range_weight = range_weight

        self.register_buffer('angle_mean', torch.zeros(12, 3))
        self.register_buffer('angle_std', torch.ones(12, 3) * 30.0)

    def update_statistics(self, ground_truth_batch: torch.Tensor):
        """Update running statistics from ground truth batch."""
        batch_mean = ground_truth_batch.mean(dim=0)
        batch_std = ground_truth_batch.std(dim=0)

        alpha = 0.01
        self.angle_mean = (1 - alpha) * self.angle_mean + alpha * batch_mean
        self.angle_std = (1 - alpha) * self.angle_std + alpha * batch_std

    def forward(
        self,
        refined_angles: torch.Tensor,
        ground_truth_angles: torch.Tensor,
        delta: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses with additional soft constraints."""
        losses = super().forward(
            refined_angles, ground_truth_angles, delta, visibility
        )

        if self.training:
            with torch.no_grad():
                self.update_statistics(ground_truth_angles)

        z_score = (refined_angles - self.angle_mean) / (self.angle_std + 1e-6)
        range_violation = F.relu(z_score.abs() - 2.0) ** 2
        range_loss = range_violation.mean()
        losses['range'] = range_loss

        losses['total'] = losses['total'] + self.range_weight * range_loss

        return losses


class GNNJointRefinementLoss(JointRefinementLoss):
    """Extended loss for GNN joint refinement with sign and temporal losses."""

    def __init__(
        self,
        symmetry_weight: float = 0.01,
        delta_weight: float = 0.001,
        chain_weight: float = 0.1,
        sign_weight: float = 0.1,
        temporal_weight: float = 0.05,
        use_visibility_weighting: bool = True,
    ):
        super().__init__(
            symmetry_weight=symmetry_weight,
            delta_weight=delta_weight,
            chain_weight=chain_weight,
            use_visibility_weighting=use_visibility_weighting,
        )
        self.sign_weight = sign_weight
        self.temporal_weight = temporal_weight

    def forward(
        self,
        refined_angles: torch.Tensor,
        ground_truth_angles: torch.Tensor,
        delta: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
        sign_logits: Optional[torch.Tensor] = None,
        prev_refined_angles: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses including sign and temporal components."""
        losses = super().forward(
            refined_angles, ground_truth_angles, delta, visibility
        )

        if sign_logits is not None:
            sign_loss = angle_sign_loss(sign_logits, ground_truth_angles, visibility)
            losses['sign'] = sign_loss
            losses['total'] = losses['total'] + self.sign_weight * sign_loss

        if prev_refined_angles is not None:
            smooth_loss = temporal_smoothness_loss(
                refined_angles, prev_refined_angles, visibility
            )
            losses['temporal'] = smooth_loss
            losses['total'] = losses['total'] + self.temporal_weight * smooth_loss

        return losses
