"""
Loss functions for joint constraint refinement.

Main losses:
1. Angle reconstruction: Angular distance (handles 360° periodicity)
2. Bilateral symmetry: Left/right consistency (learned from data)
3. Delta regularization: Penalize large corrections

The model learns soft constraints from data distribution,
not hard-coded joint limits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def angular_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute shortest angular distance accounting for 360° periodicity.

    Angles like 179° and -179° are only 2° apart, not 358°.
    This is critical for joint angles which can wrap around ±180°.

    Args:
        pred: Predicted angles in degrees (any shape)
        target: Target angles in degrees (same shape as pred)

    Returns:
        Absolute angular distance in degrees (same shape)
    """
    diff = pred - target
    # Wrap to [-180, 180] - the shortest path around the circle
    diff = ((diff + 180) % 360) - 180
    return diff.abs()


# Symmetric joint pairs (left/right)
SYMMETRIC_PAIRS = [
    (1, 2),   # hip_R, hip_L
    (3, 4),   # knee_R, knee_L
    (5, 6),   # ankle_R, ankle_L
    (8, 9),   # shoulder_R, shoulder_L
    (10, 11), # elbow_R, elbow_L
]

# Kinematic chains (parent_idx, child_idx)
# Connected joints should have consistent corrections
KINEMATIC_CHAINS = [
    # Lower body
    (0, 1),   # pelvis → hip_R
    (0, 2),   # pelvis → hip_L
    (1, 3),   # hip_R → knee_R
    (2, 4),   # hip_L → knee_L
    (3, 5),   # knee_R → ankle_R
    (4, 6),   # knee_L → ankle_L
    # Upper body
    (0, 7),   # pelvis → trunk
    (7, 8),   # trunk → shoulder_R
    (7, 9),   # trunk → shoulder_L
    (8, 10),  # shoulder_R → elbow_R
    (9, 11),  # shoulder_L → elbow_L
]


def kinematic_chain_loss(
    refined_angles: torch.Tensor,
    gt_angles: torch.Tensor,
    visibility: torch.Tensor = None,
) -> torch.Tensor:
    """Enforce consistency between connected joints in the kinematic chain.

    For each parent-child pair in the chain:
    - Compute the correction (error) for parent joint
    - Compute the correction (error) for child joint
    - Penalize if corrections diverge too much

    This ensures corrections propagate sensibly through the chain,
    preventing e.g. pelvis having huge correction while hip is unchanged.

    Args:
        refined_angles: (B, 12, 3) refined angles
        gt_angles: (B, 12, 3) ground truth angles
        visibility: (B, 12) per-joint visibility (optional)

    Returns:
        Scalar chain consistency loss
    """
    chain_loss = torch.tensor(0.0, device=refined_angles.device)

    for parent_idx, child_idx in KINEMATIC_CHAINS:
        # Error (correction) for parent and child
        parent_error = angular_distance(
            refined_angles[:, parent_idx],
            gt_angles[:, parent_idx]
        )  # (B, 3)
        child_error = angular_distance(
            refined_angles[:, child_idx],
            gt_angles[:, child_idx]
        )  # (B, 3)

        # Connected joints should have correlated error magnitudes
        # Large parent error with small child error indicates inconsistency
        error_diff = (parent_error - child_error).abs()  # (B, 3)

        # Weight by visibility if available
        if visibility is not None:
            weight = visibility[:, parent_idx] * visibility[:, child_idx]
            error_diff = error_diff * weight.unsqueeze(-1)

        chain_loss = chain_loss + error_diff.mean()

    return chain_loss / len(KINEMATIC_CHAINS)


class JointRefinementLoss(nn.Module):
    """Combined loss for joint constraint refinement.

    Components:
    1. Angle reconstruction (L1)
    2. Bilateral symmetry (L1 on symmetric pairs)
    3. Delta regularization (L2 on corrections)
    4. Kinematic chain consistency (connected joints have consistent corrections)

    All losses are visibility-weighted when visibility is provided.
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

        # Create mask for ignoring certain DOFs
        # Joint indices: 8=shoulder_R, 9=shoulder_L, 10=elbow_R, 11=elbow_L
        # DOF indices: 0=flex, 1=abd, 2=rot
        self._dof_mask = None
        if ignore_arm_abd_rot or ignore_left_arm or ignore_arms:
            # Mask: 1 = include in loss, 0 = ignore
            self.register_buffer('dof_mask', torch.ones(12, 3))
            if ignore_arms:
                # Ignore both arms entirely
                for joint_idx in [8, 9, 10, 11]:  # both shoulders and elbows
                    self.dof_mask[joint_idx, :] = 0.0
            else:
                if ignore_arm_abd_rot:
                    for joint_idx in [8, 9, 10, 11]:  # shoulders and elbows
                        self.dof_mask[joint_idx, 1] = 0.0  # abd
                        self.dof_mask[joint_idx, 2] = 0.0  # rot
                if ignore_left_arm:
                    for joint_idx in [9, 11]:  # shoulder_L, elbow_L
                        self.dof_mask[joint_idx, :] = 0.0  # all DOFs
            self._dof_mask = True

    def forward(
        self,
        refined_angles: torch.Tensor,
        ground_truth_angles: torch.Tensor,
        delta: torch.Tensor,
        visibility: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses.

        Args:
            refined_angles: (B, 12, 3) predicted refined angles
            ground_truth_angles: (B, 12, 3) ground truth angles
            delta: (B, 12, 3) predicted corrections
            visibility: (B, 12) per-joint visibility (optional)

        Returns:
            Dict with individual losses and total loss
        """
        losses = {}

        # Compute visibility weights
        if self.use_visibility_weighting and visibility is not None:
            # Expand visibility to (B, 12, 1) for broadcasting
            vis_weights = visibility.unsqueeze(-1)
        else:
            vis_weights = torch.ones_like(refined_angles[..., :1])

        # 1. Angle reconstruction loss (angular distance handles 360° periodicity)
        angle_error = angular_distance(refined_angles, ground_truth_angles)
        # Weight by visibility and take mean
        weighted_error = angle_error * vis_weights
        # Apply DOF mask if ignoring certain joints/DOFs
        if self._dof_mask:
            weighted_error = weighted_error * self.dof_mask.unsqueeze(0)
            denom = (vis_weights * self.dof_mask.unsqueeze(0)).sum() + 1e-6
        else:
            denom = vis_weights.sum() * 3 + 1e-6
        reconstruction_loss = weighted_error.sum() / denom
        losses['reconstruction'] = reconstruction_loss

        # 2. Bilateral symmetry loss
        # Left and right joints should have similar (but mirrored) constraints
        symmetry_loss = torch.tensor(0.0, device=refined_angles.device)

        for r_idx, l_idx in SYMMETRIC_PAIRS:
            r_angles = refined_angles[:, r_idx, :]  # (B, 3)
            l_angles = refined_angles[:, l_idx, :]  # (B, 3)

            # Flexion should be similar (use angular distance for periodicity)
            flex_diff = angular_distance(r_angles[:, 0], l_angles[:, 0])

            # Abd and rot should be opposite (mirrored) - use angular distance
            # For mirrored angles, r + l should equal 0, so compare r to -l
            abd_diff = angular_distance(r_angles[:, 1], -l_angles[:, 1])
            rot_diff = angular_distance(r_angles[:, 2], -l_angles[:, 2])

            # Get visibility weights for this pair
            if self.use_visibility_weighting and visibility is not None:
                pair_vis = visibility[:, r_idx] * visibility[:, l_idx]
            else:
                pair_vis = torch.ones(refined_angles.shape[0], device=refined_angles.device)

            pair_loss = (flex_diff + abd_diff + rot_diff) * pair_vis
            symmetry_loss = symmetry_loss + pair_loss.mean()

        symmetry_loss = symmetry_loss / len(SYMMETRIC_PAIRS)
        losses['symmetry'] = symmetry_loss

        # 3. Delta regularization (L2)
        # Penalize large corrections - model should make small adjustments
        delta_l2 = (delta ** 2).mean()
        losses['delta_reg'] = delta_l2

        # 4. Kinematic chain consistency
        # Connected joints should have consistent corrections
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
    """Extended loss with soft learned constraints.

    Adds:
    - Smoothness penalty (angle changes should be smooth)
    - Range penalty (learned from data distribution)
    """

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

        # Learned range statistics (updated during training)
        # These are soft constraints learned from data, not hard limits
        self.register_buffer('angle_mean', torch.zeros(12, 3))
        self.register_buffer('angle_std', torch.ones(12, 3) * 30.0)  # Default 30°

    def update_statistics(self, ground_truth_batch: torch.Tensor):
        """Update running statistics from ground truth batch."""
        # Online update of mean and std
        batch_mean = ground_truth_batch.mean(dim=0)
        batch_std = ground_truth_batch.std(dim=0)

        # Exponential moving average
        alpha = 0.01
        self.angle_mean = (1 - alpha) * self.angle_mean + alpha * batch_mean
        self.angle_std = (1 - alpha) * self.angle_std + alpha * batch_std

    def forward(
        self,
        refined_angles: torch.Tensor,
        ground_truth_angles: torch.Tensor,
        delta: torch.Tensor,
        visibility: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses with additional soft constraints."""

        # Get base losses
        losses = super().forward(
            refined_angles, ground_truth_angles, delta, visibility
        )

        # Update statistics (only during training)
        if self.training:
            with torch.no_grad():
                self.update_statistics(ground_truth_angles)

        # Soft range constraint
        # Penalize angles that deviate significantly from learned distribution
        z_score = (refined_angles - self.angle_mean) / (self.angle_std + 1e-6)
        # Use squared error for z-scores > 2 (outside 95% of distribution)
        range_violation = F.relu(z_score.abs() - 2.0) ** 2
        range_loss = range_violation.mean()
        losses['range'] = range_loss

        # Update total loss
        losses['total'] = losses['total'] + self.range_weight * range_loss

        return losses


if __name__ == '__main__':
    # Test losses
    B = 4

    refined = torch.randn(B, 12, 3) * 30
    gt = refined + torch.randn(B, 12, 3) * 5  # Small noise
    delta = torch.randn(B, 12, 3) * 2
    visibility = torch.rand(B, 12)

    loss_fn = JointRefinementLoss()
    losses = loss_fn(refined, gt, delta, visibility)

    print("Loss values:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")

    # Test extended loss
    print("\nWith soft constraints:")
    loss_fn_ext = JointRefinementLossWithConstraints()
    losses_ext = loss_fn_ext(refined, gt, delta, visibility)

    for name, value in losses_ext.items():
        print(f"  {name}: {value.item():.4f}")
