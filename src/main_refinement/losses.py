"""
Loss functions for MainRefiner training.

Includes:
- Pose loss (L1 + depth-weighted)
- Gating supervision (teach optimal weighting from GT)
- Improvement loss (ensure fusion beats individual models)
- Bone length consistency
- Confidence calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# Bone definitions for COCO-17
COCO_BONES = [
    (5, 7),    # L shoulder -> elbow
    (7, 9),    # L elbow -> wrist
    (6, 8),    # R shoulder -> elbow
    (8, 10),   # R elbow -> wrist
    (11, 13),  # L hip -> knee
    (13, 15),  # L knee -> ankle
    (12, 14),  # R hip -> knee
    (14, 16),  # R knee -> ankle
    (5, 11),   # L shoulder -> hip (torso)
    (6, 12),   # R shoulder -> hip (torso)
    (5, 6),    # shoulder width
    (11, 12),  # hip width
]


def pose_loss(
    refined_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
    depth_weight: float = 2.0,
) -> torch.Tensor:
    """Compute pose reconstruction loss with depth emphasis.

    Args:
        refined_pose: (B, 17, 3) predicted pose
        gt_pose: (B, 17, 3) ground truth pose
        visibility: (B, 17) optional visibility weights
        depth_weight: Extra weight for depth (Z) errors

    Returns:
        Weighted L1 loss
    """
    # Per-joint L1 error
    error = (refined_pose - gt_pose).abs()  # (B, 17, 3)

    # Weight depth errors more (depth is hardest to predict)
    weights = torch.ones_like(error)
    weights[:, :, 2] = depth_weight  # Z dimension
    weighted_error = error * weights

    # Per-joint error magnitude
    joint_error = weighted_error.mean(dim=-1)  # (B, 17)

    if visibility is not None:
        # Weight by visibility - use clamp for BF16 stability
        weighted_joint_error = joint_error * visibility
        return weighted_joint_error.sum() / visibility.sum().clamp(min=1e-4)
    else:
        return joint_error.mean()


def bone_length_loss(
    refined_pose: torch.Tensor,
    gt_pose: torch.Tensor,
) -> torch.Tensor:
    """Encourage bone lengths to match ground truth.

    Args:
        refined_pose: (B, 17, 3) predicted pose
        gt_pose: (B, 17, 3) ground truth pose

    Returns:
        Mean bone length error
    """
    total_error = 0.0

    for parent, child in COCO_BONES:
        # Bone length in prediction
        pred_bone = refined_pose[:, child] - refined_pose[:, parent]
        pred_len = torch.norm(pred_bone, dim=-1)

        # Bone length in GT
        gt_bone = gt_pose[:, child] - gt_pose[:, parent]
        gt_len = torch.norm(gt_bone, dim=-1)

        # L1 error
        total_error = total_error + (pred_len - gt_len).abs().mean()

    return total_error / len(COCO_BONES)


def gate_supervision_loss(
    depth_weights: torch.Tensor,
    raw_pose: torch.Tensor,
    depth_delta: torch.Tensor,
    gt_pose: torch.Tensor,
) -> torch.Tensor:
    """Supervise gating to trust the better model.

    Computes which source (depth correction or raw) would give
    lower error, and encourages gating to weight that source higher.

    Args:
        depth_weights: (B, 17) predicted depth model weights
        raw_pose: (B, 17, 3) original input pose
        depth_delta: (B, 17, 3) depth model corrections
        gt_pose: (B, 17, 3) ground truth pose

    Returns:
        MSE loss for gating supervision
    """
    # Error if we use depth model
    depth_corrected = raw_pose + depth_delta
    depth_error = (depth_corrected - gt_pose).abs().mean(dim=-1)  # (B, 17)

    # Error if we use raw pose (no correction)
    raw_error = (raw_pose - gt_pose).abs().mean(dim=-1)  # (B, 17)

    # Target: use depth if its error is lower
    # Sigmoid of negative error difference -> higher weight for lower error
    error_diff = raw_error - depth_error  # Positive = depth is better
    target_depth_weight = torch.sigmoid(error_diff * 5.0)  # Scale for sharper sigmoid

    return F.mse_loss(depth_weights, target_depth_weight.detach())


def improvement_loss(
    refined_pose: torch.Tensor,
    raw_pose: torch.Tensor,
    depth_delta: torch.Tensor,
    gt_pose: torch.Tensor,
    margin: float = 0.9,
) -> torch.Tensor:
    """Encourage fusion to beat best individual model.

    Args:
        refined_pose: (B, 17, 3) fused output
        raw_pose: (B, 17, 3) original input
        depth_delta: (B, 17, 3) depth model corrections
        gt_pose: (B, 17, 3) ground truth
        margin: Must beat best single model by this factor (0.9 = 10% better)

    Returns:
        Penalty if fusion doesn't beat best individual
    """
    # Fusion error
    fusion_error = (refined_pose - gt_pose).abs().mean()

    # Depth-only error
    depth_corrected = raw_pose + depth_delta
    depth_error = (depth_corrected - gt_pose).abs().mean()

    # Raw error (no correction)
    raw_error = (raw_pose - gt_pose).abs().mean()

    # Best single model error
    best_single = torch.min(depth_error, raw_error)

    # Penalty: fusion should be at least margin * best_single
    # relu ensures no penalty if fusion is better
    return F.relu(fusion_error - best_single * margin)


def confidence_calibration_loss(
    confidence: torch.Tensor,
    refined_pose: torch.Tensor,
    gt_pose: torch.Tensor,
) -> torch.Tensor:
    """Calibrate confidence to reflect actual error.

    High confidence should mean low error.

    Args:
        confidence: (B, 17) predicted confidence
        refined_pose: (B, 17, 3) predicted pose
        gt_pose: (B, 17, 3) ground truth pose

    Returns:
        MSE loss for confidence calibration
    """
    # Actual per-joint error
    error = (refined_pose - gt_pose).norm(dim=-1)  # (B, 17)

    # Normalize to [0, 1] - use clamp for BF16 stability
    max_error = error.max().clamp(min=1e-4)
    error_norm = error / max_error

    # Target confidence = 1 - normalized_error
    target_confidence = (1.0 - error_norm).detach()

    return F.mse_loss(confidence, target_confidence)


class MainRefinerLoss(nn.Module):
    """Combined loss for MainRefiner training.

    Combines multiple loss components with configurable weights.
    """

    def __init__(
        self,
        pose_weight: float = 1.0,
        bone_weight: float = 0.5,
        gate_supervision_weight: float = 0.3,
        improvement_weight: float = 0.2,
        confidence_weight: float = 0.1,
        depth_emphasis: float = 2.0,
    ):
        """
        Args:
            pose_weight: Weight for pose reconstruction loss
            bone_weight: Weight for bone length consistency
            gate_supervision_weight: Weight for gating supervision
            improvement_weight: Weight for improvement loss
            confidence_weight: Weight for confidence calibration
            depth_emphasis: Extra weight for depth errors in pose loss
        """
        super().__init__()
        self.pose_weight = pose_weight
        self.bone_weight = bone_weight
        self.gate_supervision_weight = gate_supervision_weight
        self.improvement_weight = improvement_weight
        self.confidence_weight = confidence_weight
        self.depth_emphasis = depth_emphasis

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        raw_pose: torch.Tensor,
        gt_pose: torch.Tensor,
        depth_outputs: Dict[str, torch.Tensor],
        visibility: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            output: MainRefiner output dict
            raw_pose: (B, 17, 3) original input pose
            gt_pose: (B, 17, 3) ground truth pose
            depth_outputs: Depth model output dict (for delta_xyz)
            visibility: (B, 17) optional visibility weights

        Returns:
            Dict with individual losses and 'total'
        """
        losses = {}

        # 1. Primary pose loss
        losses['pose'] = pose_loss(
            output['refined_pose'],
            gt_pose,
            visibility,
            self.depth_emphasis,
        )

        # 2. Bone length consistency
        losses['bone'] = bone_length_loss(output['refined_pose'], gt_pose)

        # 3. Gating supervision
        losses['gate'] = gate_supervision_loss(
            output['depth_weights'],
            raw_pose,
            depth_outputs['delta_xyz'],
            gt_pose,
        )

        # 4. Improvement loss
        losses['improvement'] = improvement_loss(
            output['refined_pose'],
            raw_pose,
            depth_outputs['delta_xyz'],
            gt_pose,
        )

        # 5. Confidence calibration
        losses['confidence'] = confidence_calibration_loss(
            output['confidence'],
            output['refined_pose'],
            gt_pose,
        )

        # Total weighted loss
        losses['total'] = (
            self.pose_weight * losses['pose'] +
            self.bone_weight * losses['bone'] +
            self.gate_supervision_weight * losses['gate'] +
            self.improvement_weight * losses['improvement'] +
            self.confidence_weight * losses['confidence']
        )

        return losses


if __name__ == '__main__':
    # Test losses
    print("Testing MainRefinerLoss...")

    B = 4

    # Mock data
    raw_pose = torch.randn(B, 17, 3)
    gt_pose = raw_pose + torch.randn(B, 17, 3) * 0.1  # Small perturbation
    visibility = torch.rand(B, 17)

    # Mock model outputs
    output = {
        'refined_pose': raw_pose + torch.randn(B, 17, 3) * 0.05,
        'depth_weights': torch.rand(B, 17),
        'joint_weights': torch.rand(B, 17),
        'residual_weights': torch.rand(B, 17),
        'confidence': torch.rand(B, 17),
    }

    depth_outputs = {
        'delta_xyz': torch.randn(B, 17, 3) * 0.1,
    }

    # Test loss
    loss_fn = MainRefinerLoss()
    losses = loss_fn(output, raw_pose, gt_pose, depth_outputs, visibility)

    print("\nLoss values:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    print("\nLoss computation test passed!")
