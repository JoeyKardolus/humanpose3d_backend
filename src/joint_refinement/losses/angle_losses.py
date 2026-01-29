"""Basic angle loss functions.

Contains individual loss functions for joint angle refinement.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def angular_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute shortest angular distance accounting for 360 periodicity.

    Angles like 179 and -179 are only 2 apart, not 358.

    Args:
        pred: Predicted angles in degrees (any shape)
        target: Target angles in degrees (same shape as pred)

    Returns:
        Absolute angular distance in degrees (same shape)
    """
    diff = pred - target
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
KINEMATIC_CHAINS = [
    # Lower body
    (0, 1),   # pelvis -> hip_R
    (0, 2),   # pelvis -> hip_L
    (1, 3),   # hip_R -> knee_R
    (2, 4),   # hip_L -> knee_L
    (3, 5),   # knee_R -> ankle_R
    (4, 6),   # knee_L -> ankle_L
    # Upper body
    (0, 7),   # pelvis -> trunk
    (7, 8),   # trunk -> shoulder_R
    (7, 9),   # trunk -> shoulder_L
    (8, 10),  # shoulder_R -> elbow_R
    (9, 11),  # shoulder_L -> elbow_L
]


def kinematic_chain_loss(
    refined_angles: torch.Tensor,
    gt_angles: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Enforce consistency between connected joints in the kinematic chain.

    Args:
        refined_angles: (B, 12, 3) refined angles
        gt_angles: (B, 12, 3) ground truth angles
        visibility: (B, 12) per-joint visibility (optional)

    Returns:
        Scalar chain consistency loss
    """
    chain_loss = torch.tensor(0.0, device=refined_angles.device)

    for parent_idx, child_idx in KINEMATIC_CHAINS:
        parent_error = angular_distance(
            refined_angles[:, parent_idx],
            gt_angles[:, parent_idx]
        )
        child_error = angular_distance(
            refined_angles[:, child_idx],
            gt_angles[:, child_idx]
        )

        error_diff = (parent_error - child_error).abs()

        if visibility is not None:
            weight = visibility[:, parent_idx] * visibility[:, child_idx]
            error_diff = error_diff * weight.unsqueeze(-1)

        chain_loss = chain_loss + error_diff.mean()

    return chain_loss / len(KINEMATIC_CHAINS)


def angle_sign_loss(
    sign_logits: torch.Tensor,
    gt_angles: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute BCE loss for abd/rot sign classification.

    Args:
        sign_logits: (B, 12, 2) logits for [P(abd > 0), P(rot > 0)]
        gt_angles: (B, 12, 3) ground truth angles in degrees
        visibility: (B, 12) optional per-joint visibility

    Returns:
        Scalar BCE loss
    """
    abd_sign_gt = (gt_angles[:, :, 1] > 0).float()
    rot_sign_gt = (gt_angles[:, :, 2] > 0).float()

    abd_loss = F.binary_cross_entropy_with_logits(
        sign_logits[:, :, 0],
        abd_sign_gt,
        reduction='none',
    )
    rot_loss = F.binary_cross_entropy_with_logits(
        sign_logits[:, :, 1],
        rot_sign_gt,
        reduction='none',
    )

    loss = abd_loss + rot_loss

    if visibility is not None:
        loss = loss * visibility
        return loss.sum() / (visibility.sum() + 1e-6)

    return loss.mean()


def temporal_smoothness_loss(
    refined_angles: torch.Tensor,
    prev_refined_angles: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize large frame-to-frame angle changes.

    Args:
        refined_angles: (B, 12, 3) refined angles for current frame
        prev_refined_angles: (B, 12, 3) refined angles for previous frame
        visibility: (B, 12) optional per-joint visibility

    Returns:
        Scalar smoothness loss
    """
    frame_diff = angular_distance(refined_angles, prev_refined_angles)

    if visibility is not None:
        vis_weight = visibility.unsqueeze(-1)
        frame_diff = frame_diff * vis_weight
        return frame_diff.sum() / (vis_weight.sum() * 3 + 1e-6)

    return frame_diff.mean()


def sign_accuracy(
    sign_logits: torch.Tensor,
    gt_angles: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute sign prediction accuracy metrics.

    Args:
        sign_logits: (B, 12, 2) logits for [P(abd > 0), P(rot > 0)]
        gt_angles: (B, 12, 3) ground truth angles in degrees
        visibility: (B, 12) optional per-joint visibility

    Returns:
        Dict with 'abd_acc', 'rot_acc', 'total_acc'
    """
    abd_pred = (torch.sigmoid(sign_logits[:, :, 0]) > 0.5).float()
    rot_pred = (torch.sigmoid(sign_logits[:, :, 1]) > 0.5).float()

    abd_gt = (gt_angles[:, :, 1] > 0).float()
    rot_gt = (gt_angles[:, :, 2] > 0).float()

    if visibility is not None:
        abd_correct = ((abd_pred == abd_gt).float() * visibility).sum()
        rot_correct = ((rot_pred == rot_gt).float() * visibility).sum()
        total = visibility.sum() + 1e-6
        abd_acc = (abd_correct / total).item()
        rot_acc = (rot_correct / total).item()
    else:
        abd_acc = (abd_pred == abd_gt).float().mean().item()
        rot_acc = (rot_pred == rot_gt).float().mean().item()

    return {
        'abd_acc': abd_acc,
        'rot_acc': rot_acc,
        'total_acc': (abd_acc + rot_acc) / 2,
    }
