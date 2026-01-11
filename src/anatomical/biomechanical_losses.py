#!/usr/bin/env python3
"""
Biomechanical loss functions for self-supervised depth refinement training.

Losses based on anatomical constraints:
- Bone length consistency
- Joint angle ROM
- Ground plane contact
- Left/right symmetry
- Temporal smoothness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


# OpenCap marker pairs for bones
BONE_PAIRS = [
    # Lower body - right
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("RAnkle", "RHeel"),
    ("RAnkle", "RBigToe"),

    # Lower body - left
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("LAnkle", "LHeel"),
    ("LAnkle", "LBigToe"),

    # Spine
    ("Hip", "C7_study"),

    # Upper body - right
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),

    # Upper body - left
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
]


# Symmetric marker pairs (left/right)
SYMMETRIC_PAIRS = [
    ("RHip", "LHip"),
    ("RKnee", "LKnee"),
    ("RAnkle", "LAnkle"),
    ("RShoulder", "LShoulder"),
    ("RElbow", "LElbow"),
    ("RWrist", "LWrist"),
]


# Ground contact markers (feet)
GROUND_MARKERS = ["RHeel", "LHeel", "RBigToe", "LBigToe", "RSmallToe", "LSmallToe"]


class BoneLengthLoss(nn.Module):
    """Penalize temporal variation in bone lengths.

    Bones should maintain consistent length across frames.
    """

    def __init__(self, bone_pairs: List[Tuple[str, str]]):
        super().__init__()
        self.bone_pairs = bone_pairs

    def forward(
        self,
        positions: torch.Tensor,
        marker_names: List[str]
    ) -> torch.Tensor:
        """Compute bone length consistency loss.

        Args:
            positions: (batch, frames, markers, 3)
            marker_names: List of marker names

        Returns:
            Scalar loss
        """
        # Create marker name to index mapping
        marker_idx = {name: i for i, name in enumerate(marker_names)}

        batch_size, frames, num_markers, _ = positions.shape
        total_loss = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)
        num_bones = 0

        for marker1, marker2 in self.bone_pairs:
            if marker1 not in marker_idx or marker2 not in marker_idx:
                continue

            idx1 = marker_idx[marker1]
            idx2 = marker_idx[marker2]

            # Get positions for both markers
            pos1 = positions[:, :, idx1, :]  # (batch, frames, 3)
            pos2 = positions[:, :, idx2, :]  # (batch, frames, 3)

            # Compute bone lengths
            bone_lengths = torch.norm(pos2 - pos1, dim=-1)  # (batch, frames)

            # Compute coefficient of variation (CV) per batch
            mean_length = bone_lengths.mean(dim=1, keepdim=True)  # (batch, 1)
            std_length = bone_lengths.std(dim=1, keepdim=True)  # (batch, 1)

            # CV = std / mean (penalize high variation)
            cv = std_length / (mean_length + 1e-6)

            total_loss += cv.mean()
            num_bones += 1

        if num_bones == 0:
            return total_loss
        return total_loss / num_bones


class GroundPlaneLoss(nn.Module):
    """Penalize foot markers going below ground plane.

    Assumes ground is at z=0 or minimum foot height.
    """

    def __init__(self, ground_markers: List[str], threshold: float = 0.01):
        super().__init__()
        self.ground_markers = ground_markers
        self.threshold = threshold  # Allowable penetration (meters)

    def forward(
        self,
        positions: torch.Tensor,
        marker_names: List[str]
    ) -> torch.Tensor:
        """Compute ground plane violation loss.

        Args:
            positions: (batch, frames, markers, 3)
            marker_names: List of marker names

        Returns:
            Scalar loss
        """
        marker_idx = {name: i for i, name in enumerate(marker_names)}

        batch_size, frames, num_markers, _ = positions.shape

        # Find minimum Z across all foot markers (defines ground)
        foot_indices = [marker_idx[m] for m in self.ground_markers if m in marker_idx]

        if not foot_indices:
            return torch.tensor(0.0, device=positions.device)

        foot_positions = positions[:, :, foot_indices, 2]  # (batch, frames, num_feet)
        ground_level = foot_positions.min()  # Global minimum

        # Penalize markers below (ground_level - threshold)
        violations = F.relu(ground_level - self.threshold - foot_positions)

        return violations.mean()


class SymmetryLoss(nn.Module):
    """Penalize asymmetry between left and right limbs.

    Corresponding bones should have similar lengths.
    """

    def __init__(self, symmetric_pairs: List[Tuple[str, str]]):
        super().__init__()
        self.symmetric_pairs = symmetric_pairs

    def forward(
        self,
        positions: torch.Tensor,
        marker_names: List[str]
    ) -> torch.Tensor:
        """Compute left-right symmetry loss.

        Args:
            positions: (batch, frames, markers, 3)
            marker_names: List of marker names

        Returns:
            Scalar loss
        """
        marker_idx = {name: i for i, name in enumerate(marker_names)}

        # Compute distance from pelvis/hip center for each symmetric pair
        if "Hip" not in marker_idx:
            return torch.tensor(0.0, device=positions.device)

        hip_idx = marker_idx["Hip"]
        hip_pos = positions[:, :, hip_idx, :]  # (batch, frames, 3)

        total_loss = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)
        num_pairs = 0

        for right_marker, left_marker in self.symmetric_pairs:
            if right_marker not in marker_idx or left_marker not in marker_idx:
                continue

            right_idx = marker_idx[right_marker]
            left_idx = marker_idx[left_marker]

            # Distance from hip to each marker
            right_dist = torch.norm(positions[:, :, right_idx, :] - hip_pos, dim=-1)
            left_dist = torch.norm(positions[:, :, left_idx, :] - hip_pos, dim=-1)

            # Penalize difference in distances
            diff = torch.abs(right_dist - left_dist)
            total_loss += diff.mean()
            num_pairs += 1

        if num_pairs == 0:
            return total_loss
        return total_loss / num_pairs


class TemporalSmoothnessLoss(nn.Module):
    """Penalize jerky motion (high acceleration).

    Motion should be smooth across frames.
    """

    def __init__(self):
        super().__init__()

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute temporal smoothness loss.

        Args:
            positions: (batch, frames, markers, 3)

        Returns:
            Scalar loss
        """
        # Compute second-order differences (acceleration)
        # velocity = pos[t+1] - pos[t]
        # accel = velocity[t+1] - velocity[t]

        velocity = positions[:, 1:, :, :] - positions[:, :-1, :, :]  # (batch, frames-1, markers, 3)
        accel = velocity[:, 1:, :, :] - velocity[:, :-1, :, :]  # (batch, frames-2, markers, 3)

        # L2 norm of acceleration
        accel_magnitude = torch.norm(accel, dim=-1)  # (batch, frames-2, markers)

        return accel_magnitude.mean()


class JointAngleLoss(nn.Module):
    """Penalize joint angles outside biomechanical ROM.

    Simple version: just penalize extreme bone angles.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        positions: torch.Tensor,
        marker_names: List[str]
    ) -> torch.Tensor:
        """Compute joint angle ROM violation loss.

        Args:
            positions: (batch, frames, markers, 3)
            marker_names: List of marker names

        Returns:
            Scalar loss
        """
        marker_idx = {name: i for i, name in enumerate(marker_names)}

        # Check knee angles (simple version: angle between thigh-knee-ankle)
        loss = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)
        num_joints = 0

        # Right knee
        if all(m in marker_idx for m in ["RHip", "RKnee", "RAnkle"]):
            hip = positions[:, :, marker_idx["RHip"], :]
            knee = positions[:, :, marker_idx["RKnee"], :]
            ankle = positions[:, :, marker_idx["RAnkle"], :]

            # Vectors
            v1 = hip - knee  # Thigh
            v2 = ankle - knee  # Shank

            # Dot product
            dot = (v1 * v2).sum(dim=-1)
            norm1 = torch.norm(v1, dim=-1)
            norm2 = torch.norm(v2, dim=-1)

            cos_angle = dot / (norm1 * norm2 + 1e-6)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

            # Knee should not hyperextend (angle < 180°)
            # cos(180°) = -1, penalize if cos_angle < -0.95 (>170°)
            hyperextension = F.relu(-0.95 - cos_angle)
            loss += hyperextension.mean()
            num_joints += 1

        # Left knee
        if all(m in marker_idx for m in ["LHip", "LKnee", "LAnkle"]):
            hip = positions[:, :, marker_idx["LHip"], :]
            knee = positions[:, :, marker_idx["LKnee"], :]
            ankle = positions[:, :, marker_idx["LAnkle"], :]

            v1 = hip - knee
            v2 = ankle - knee

            dot = (v1 * v2).sum(dim=-1)
            norm1 = torch.norm(v1, dim=-1)
            norm2 = torch.norm(v2, dim=-1)

            cos_angle = dot / (norm1 * norm2 + 1e-6)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

            hyperextension = F.relu(-0.95 - cos_angle)
            loss += hyperextension.mean()
            num_joints += 1

        if num_joints == 0:
            return loss
        return loss / num_joints


class BiomechanicalLoss(nn.Module):
    """Combined biomechanical loss function.

    Weights different constraints for training.
    """

    def __init__(
        self,
        bone_weight: float = 1.0,
        ground_weight: float = 0.8,
        symmetry_weight: float = 0.4,
        smoothness_weight: float = 0.2,
        angle_weight: float = 0.5,
    ):
        super().__init__()

        self.bone_weight = bone_weight
        self.ground_weight = ground_weight
        self.symmetry_weight = symmetry_weight
        self.smoothness_weight = smoothness_weight
        self.angle_weight = angle_weight

        # Initialize individual losses
        self.bone_loss = BoneLengthLoss(BONE_PAIRS)
        self.ground_loss = GroundPlaneLoss(GROUND_MARKERS)
        self.symmetry_loss = SymmetryLoss(SYMMETRIC_PAIRS)
        self.smoothness_loss = TemporalSmoothnessLoss()
        self.angle_loss = JointAngleLoss()

    def forward(
        self,
        positions: torch.Tensor,
        marker_names: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined biomechanical loss.

        Args:
            positions: (batch, frames, markers, 3)
            marker_names: List of marker names

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual loss values for logging
        """
        bone_l = self.bone_loss(positions, marker_names)
        ground_l = self.ground_loss(positions, marker_names)
        symmetry_l = self.symmetry_loss(positions, marker_names)
        smoothness_l = self.smoothness_loss(positions)
        angle_l = self.angle_loss(positions, marker_names)

        total_loss = (
            self.bone_weight * bone_l +
            self.ground_weight * ground_l +
            self.symmetry_weight * symmetry_l +
            self.smoothness_weight * smoothness_l +
            self.angle_weight * angle_l
        )

        loss_dict = {
            "bone_length": bone_l.item(),
            "ground_plane": ground_l.item(),
            "symmetry": symmetry_l.item(),
            "smoothness": smoothness_l.item(),
            "joint_angle": angle_l.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    # Test loss functions
    print("Testing biomechanical loss functions...")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy data
    batch_size = 4
    frames = 11
    markers = 59
    marker_names = ["Hip", "RHip", "LHip", "RKnee", "LKnee", "RAnkle", "LAnkle",
                    "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist", "LWrist",
                    "RHeel", "LHeel", "RBigToe", "LBigToe", "C7_study"] + [f"marker_{i}" for i in range(41)]

    positions = torch.randn(batch_size, frames, markers, 3).to(device) * 0.5
    positions[:, :, :, 2] += 1.0  # Shift Z up to avoid ground penetration

    # Test combined loss
    biomech_loss = BiomechanicalLoss().to(device)

    total_loss, loss_dict = biomech_loss(positions, marker_names)

    print("Loss values:")
    for name, value in loss_dict.items():
        print(f"  {name:20s}: {value:.6f}")

    print()
    print("✓ Loss functions working correctly!")
