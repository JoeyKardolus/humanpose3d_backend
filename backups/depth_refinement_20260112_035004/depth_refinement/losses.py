"""
Loss functions for depth refinement training.

Key losses:
1. Primary depth correction loss (L1)
2. Bone length consistency (corrected pose should have consistent bone lengths)
3. Symmetry (left/right limbs should have similar bone lengths)
4. Confidence calibration (high confidence should correlate with low error)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

# COCO 17 bone definitions (joint pairs)
COCO_BONES = [
    # Torso
    (5, 6),    # left_shoulder - right_shoulder
    (11, 12),  # left_hip - right_hip
    (5, 11),   # left_shoulder - left_hip
    (6, 12),   # right_shoulder - right_hip

    # Left arm
    (5, 7),    # left_shoulder - left_elbow
    (7, 9),    # left_elbow - left_wrist

    # Right arm
    (6, 8),    # right_shoulder - right_elbow
    (8, 10),   # right_elbow - right_wrist

    # Left leg
    (11, 13),  # left_hip - left_knee
    (13, 15),  # left_knee - left_ankle

    # Right leg
    (12, 14),  # right_hip - right_knee
    (14, 16),  # right_knee - right_ankle
]

# Symmetric bone pairs (left_bone_idx, right_bone_idx)
SYMMETRIC_PAIRS = [
    (4, 6),   # left upper arm vs right upper arm
    (5, 7),   # left forearm vs right forearm
    (8, 10),  # left thigh vs right thigh
    (9, 11),  # left shin vs right shin
]


def compute_bone_lengths(pose: torch.Tensor) -> torch.Tensor:
    """Compute bone lengths from pose.

    Args:
        pose: (batch, 17, 3) joint positions

    Returns:
        (batch, num_bones) bone lengths
    """
    lengths = []
    for i, j in COCO_BONES:
        bone_vec = pose[:, i] - pose[:, j]
        length = torch.norm(bone_vec, dim=-1)
        lengths.append(length)
    return torch.stack(lengths, dim=-1)  # (batch, num_bones)


def depth_correction_loss(
    pred_delta_z: torch.Tensor,
    corrupted_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    visibility: torch.Tensor = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Primary loss: Predicted depth correction should match needed correction.

    Args:
        pred_delta_z: (batch, 17) predicted depth corrections
        corrupted_pose: (batch, 17, 3) input pose with depth errors
        gt_pose: (batch, 17, 3) ground truth pose
        visibility: (batch, 17) optional visibility weights
        reduction: 'mean', 'sum', or 'none'

    Returns:
        L1 loss between predicted and actual depth corrections
    """
    # Actual depth correction needed
    gt_delta_z = gt_pose[:, :, 2] - corrupted_pose[:, :, 2]

    # L1 loss
    loss = F.l1_loss(pred_delta_z, gt_delta_z, reduction='none')  # (batch, 17)

    # Optional: weight by visibility (low visibility = uncertain GT too)
    if visibility is not None:
        # Higher visibility = more weight on getting it right
        weights = visibility.clamp(min=0.1)  # Don't zero out completely
        loss = loss * weights

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def bone_length_loss(
    pred_delta_z: torch.Tensor,
    corrupted_pose: torch.Tensor,
    gt_pose: torch.Tensor,
) -> torch.Tensor:
    """Bone length consistency loss.

    The corrected pose should have similar bone lengths to ground truth.

    Args:
        pred_delta_z: (batch, 17) predicted depth corrections
        corrupted_pose: (batch, 17, 3) input pose
        gt_pose: (batch, 17, 3) ground truth pose

    Returns:
        Mean absolute difference in bone lengths
    """
    # Apply correction
    corrected_pose = corrupted_pose.clone()
    corrected_pose[:, :, 2] = corrected_pose[:, :, 2] + pred_delta_z

    # Compute bone lengths
    corrected_lengths = compute_bone_lengths(corrected_pose)
    gt_lengths = compute_bone_lengths(gt_pose)

    # L1 loss on bone lengths
    return F.l1_loss(corrected_lengths, gt_lengths)


def symmetry_loss(
    pred_delta_z: torch.Tensor,
    corrupted_pose: torch.Tensor,
) -> torch.Tensor:
    """Symmetry loss: Left and right limbs should have similar bone lengths.

    This is a soft constraint - humans aren't perfectly symmetric, but
    large asymmetries usually indicate errors.

    Args:
        pred_delta_z: (batch, 17) predicted depth corrections
        corrupted_pose: (batch, 17, 3) input pose

    Returns:
        Mean asymmetry penalty
    """
    # Apply correction
    corrected_pose = corrupted_pose.clone()
    corrected_pose[:, :, 2] = corrected_pose[:, :, 2] + pred_delta_z

    # Compute bone lengths
    lengths = compute_bone_lengths(corrected_pose)

    # Compare symmetric pairs
    asymmetry = 0.0
    for left_idx, right_idx in SYMMETRIC_PAIRS:
        left_len = lengths[:, left_idx]
        right_len = lengths[:, right_idx]
        asymmetry = asymmetry + torch.abs(left_len - right_len).mean()

    return asymmetry / len(SYMMETRIC_PAIRS)


def confidence_calibration_loss(
    pred_delta_z: torch.Tensor,
    pred_confidence: torch.Tensor,
    corrupted_pose: torch.Tensor,
    gt_pose: torch.Tensor,
) -> torch.Tensor:
    """Confidence should correlate with actual accuracy.

    High confidence -> low error
    Low confidence -> high error

    This encourages the network to "know when it doesn't know."

    Args:
        pred_delta_z: (batch, 17) predicted corrections
        pred_confidence: (batch, 17) predicted confidence (0-1)
        corrupted_pose: (batch, 17, 3) input pose
        gt_pose: (batch, 17, 3) ground truth

    Returns:
        Negative correlation penalty
    """
    # Actual error per joint
    gt_delta_z = gt_pose[:, :, 2] - corrupted_pose[:, :, 2]
    error = torch.abs(pred_delta_z - gt_delta_z)

    # Normalize error to [0, 1] range for comparison
    error_norm = error / (error.max() + 1e-6)

    # We want: high confidence where error is low
    # Loss: confidence should be (1 - normalized_error)
    target_confidence = 1.0 - error_norm.detach()  # Detach to avoid gradient issues

    return F.mse_loss(pred_confidence, target_confidence)


class DepthRefinementLoss(nn.Module):
    """Combined loss for depth refinement training.

    Balances multiple objectives:
    1. Primary depth correction (most important)
    2. Bone length consistency (biomechanical validity)
    3. Symmetry (soft constraint)
    4. Confidence calibration (optional)
    """

    def __init__(
        self,
        depth_weight: float = 1.0,
        bone_weight: float = 0.1,
        symmetry_weight: float = 0.05,
        confidence_weight: float = 0.1,
    ):
        super().__init__()
        self.depth_weight = depth_weight
        self.bone_weight = bone_weight
        self.symmetry_weight = symmetry_weight
        self.confidence_weight = confidence_weight

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        corrupted_pose: torch.Tensor,
        gt_pose: torch.Tensor,
        visibility: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            model_output: dict with 'delta_z' and optionally 'confidence'
            corrupted_pose: (batch, 17, 3) input pose
            gt_pose: (batch, 17, 3) ground truth
            visibility: (batch, 17) visibility scores

        Returns:
            dict with 'total' and individual loss components
        """
        pred_delta_z = model_output['delta_z']

        losses = {}

        # 1. Primary depth loss
        losses['depth'] = depth_correction_loss(
            pred_delta_z, corrupted_pose, gt_pose, visibility
        )

        # 2. Bone length consistency
        losses['bone'] = bone_length_loss(pred_delta_z, corrupted_pose, gt_pose)

        # 3. Symmetry
        losses['symmetry'] = symmetry_loss(pred_delta_z, corrupted_pose)

        # 4. Confidence calibration (if available)
        if 'confidence' in model_output:
            losses['confidence'] = confidence_calibration_loss(
                pred_delta_z,
                model_output['confidence'],
                corrupted_pose,
                gt_pose,
            )
        else:
            losses['confidence'] = torch.tensor(0.0, device=pred_delta_z.device)

        # Combined loss
        losses['total'] = (
            self.depth_weight * losses['depth'] +
            self.bone_weight * losses['bone'] +
            self.symmetry_weight * losses['symmetry'] +
            self.confidence_weight * losses['confidence']
        )

        return losses


if __name__ == '__main__':
    # Quick test
    batch_size = 4
    corrupted = torch.randn(batch_size, 17, 3)
    gt = corrupted.clone()
    gt[:, :, 2] += torch.randn(batch_size, 17) * 0.1  # Add some depth difference

    visibility = torch.rand(batch_size, 17)

    pred_delta_z = torch.randn(batch_size, 17) * 0.05
    pred_conf = torch.rand(batch_size, 17)

    model_output = {'delta_z': pred_delta_z, 'confidence': pred_conf}

    loss_fn = DepthRefinementLoss()
    losses = loss_fn(model_output, corrupted, gt, visibility)

    print("Losses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
