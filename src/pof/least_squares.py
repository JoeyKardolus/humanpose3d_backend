"""Least-squares depth solver for POF-based 3D reconstruction.

Implements the MTC (MonocularTotalCapture) style least-squares solver that
keeps X,Y fixed from 2D observations and only solves for Z depths.

Key insight from MTC: Under orthographic projection, 3D X,Y ≈ 2D positions.
By normalizing both 2D input and POF predictions to the same scale (pelvis-
centered, unit torso), the least-squares solution becomes valid:

    scale = dot(delta_2d, orient_xy) / ||orient_xy||^2
    child_depth = parent_depth + scale * orient_z

This ensures the reconstructed 3D skeleton projects back to the observed 2D
positions (reprojection error ≈ 0 by construction).
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from .constants import (
    LIMB_DEFINITIONS,
    NUM_JOINTS,
    NUM_LIMBS,
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX,
    LEFT_SHOULDER_IDX,
    RIGHT_SHOULDER_IDX,
)


# Hierarchical solve order: hips -> torso -> extremities
# Matches the order in depth_refinement/model.py
SOLVE_ORDER = [
    # From hips: solve shoulders via torso
    10,  # L torso (5-11): L_hip → L_shoulder
    11,  # R torso (6-12): R_hip → R_shoulder
    # From shoulders: solve arms
    0,   # L upper arm (5-7): L_shoulder → L_elbow
    2,   # R upper arm (6-8): R_shoulder → R_elbow
    1,   # L forearm (7-9): L_elbow → L_wrist
    3,   # R forearm (8-10): R_elbow → R_wrist
    # From hips: solve legs
    4,   # L thigh (11-13): L_hip → L_knee
    6,   # R thigh (12-14): R_hip → R_knee
    5,   # L shin (13-15): L_knee → L_ankle
    7,   # R shin (14-16): R_knee → R_ankle
]


def normalize_2d_for_pof(
    keypoints_2d: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize 2D keypoints to POF training scale.

    Transforms [0,1] image coordinates to pelvis-centered, unit-torso scale.
    This matches the coordinate system used during POF training.

    Args:
        keypoints_2d: (batch, 17, 2) normalized [0,1] image coordinates
        eps: Small value to prevent division by zero

    Returns:
        normalized_2d: (batch, 17, 2) pelvis-centered, unit torso scale
        pelvis_2d: (batch, 2) pelvis position for denormalization
        torso_scale: (batch,) torso scale for denormalization
    """
    # Compute pelvis center (midpoint of hips)
    pelvis_2d = (
        keypoints_2d[:, LEFT_HIP_IDX] + keypoints_2d[:, RIGHT_HIP_IDX]
    ) / 2  # (batch, 2)

    # Center on pelvis
    centered = keypoints_2d - pelvis_2d.unsqueeze(1)  # (batch, 17, 2)

    # Compute torso scale (average of L/R shoulder-to-hip)
    l_torso = torch.norm(
        keypoints_2d[:, LEFT_SHOULDER_IDX] - keypoints_2d[:, LEFT_HIP_IDX],
        dim=-1
    )  # (batch,)
    r_torso = torch.norm(
        keypoints_2d[:, RIGHT_SHOULDER_IDX] - keypoints_2d[:, RIGHT_HIP_IDX],
        dim=-1
    )  # (batch,)
    torso_scale = (l_torso + r_torso) / 2  # (batch,)

    # Prevent division by zero
    safe_scale = torch.clamp(torso_scale, min=eps)

    # Scale to unit torso
    normalized_2d = centered / safe_scale.unsqueeze(-1).unsqueeze(-1)

    return normalized_2d, pelvis_2d, torso_scale


def solve_depth_least_squares_pof(
    pof: torch.Tensor,
    keypoints_2d: torch.Tensor,
    bone_lengths: Optional[torch.Tensor] = None,
    pelvis_depth: float = 0.0,
    normalize_input: bool = True,
    return_scale_factors: bool = False,
) -> torch.Tensor:
    """MTC-style depth solver using 2D keypoints and POF orientations.

    Pipeline:
    1. Normalize 2D: center on pelvis, scale to unit torso
    2. Copy normalized 2D X,Y as 3D X,Y (MTC insight)
    3. For each limb: solve scale = dot(delta_2d, orient_xy) / ||orient_xy||^2
    4. Compute depth: child_z = parent_z + scale * orient_z
    5. Return 3D pose in normalized camera space

    Same code used for training (differentiable) and inference.

    Args:
        pof: (batch, 14, 3) POF unit vectors from model
        keypoints_2d: (batch, 17, 2) 2D keypoints (normalized [0,1] if
                     normalize_input=True, or already normalized if False)
        bone_lengths: (14,) or (batch, 14) optional bone lengths for scale
                     clamping. If None, uses default clamp of [-1.5, 1.5]
        pelvis_depth: Initial Z for pelvis (default 0.0)
        normalize_input: If True, normalize 2D to pelvis-centered unit torso
        return_scale_factors: If True, also return scale factors for each limb

    Returns:
        pose_3d: (batch, 17, 3) reconstructed 3D pose in normalized space
                 (pelvis-centered, unit torso scale)
        scale_factors: (batch, 14) scale factors if return_scale_factors=True
    """
    batch_size = pof.size(0)
    device = pof.device
    dtype = pof.dtype

    # Step 1: Normalize 2D if needed
    if normalize_input:
        normalized_2d, _, _ = normalize_2d_for_pof(keypoints_2d)
    else:
        normalized_2d = keypoints_2d

    # Step 2: Initialize 3D pose with 2D X,Y (MTC insight)
    pose_3d = torch.zeros(batch_size, NUM_JOINTS, 3, device=device, dtype=dtype)
    pose_3d[:, :, :2] = normalized_2d  # Copy X,Y from 2D

    # Initialize pelvis (hips) depth
    pose_3d[:, LEFT_HIP_IDX, 2] = pelvis_depth
    pose_3d[:, RIGHT_HIP_IDX, 2] = pelvis_depth

    # Track scale factors for diagnostics/losses
    scale_factors = torch.zeros(batch_size, NUM_LIMBS, device=device, dtype=dtype)

    # Step 3-4: Solve depths hierarchically
    for limb_idx in SOLVE_ORDER:
        parent_idx, child_idx = LIMB_DEFINITIONS[limb_idx]
        orientation = pof[:, limb_idx]  # (batch, 3)

        # Get 2D positions (already normalized)
        parent_2d = normalized_2d[:, parent_idx]  # (batch, 2)
        child_2d = normalized_2d[:, child_idx]    # (batch, 2)
        parent_depth = pose_3d[:, parent_idx, 2]  # (batch,)

        # 2D displacement
        delta_2d = child_2d - parent_2d  # (batch, 2)
        delta_2d_len = torch.norm(delta_2d, dim=-1)  # (batch,)

        # Orientation components
        orient_xy = orientation[:, :2]  # (batch, 2)
        orient_z = orientation[:, 2]    # (batch,)

        # Least-squares solution for scale
        # scale = dot(delta_2d, orient_xy) / ||orient_xy||^2
        orient_xy_norm_sq = (orient_xy ** 2).sum(dim=-1)  # (batch,)

        # Edge case: when orient_xy is small (limb pointing at camera)
        valid_solve = orient_xy_norm_sq > 0.05

        # Safe division
        safe_norm_sq = torch.where(
            valid_solve,
            orient_xy_norm_sq,
            torch.ones_like(orient_xy_norm_sq)
        )
        scale_from_lstsq = (delta_2d * orient_xy).sum(dim=-1) / safe_norm_sq

        # Fallback: use 2D length with sign from Z component
        fallback_scale = delta_2d_len * torch.sign(orient_z + 1e-8)

        # Choose between lstsq and fallback
        scale = torch.where(valid_solve, scale_from_lstsq, fallback_scale)

        # Clamp scale to reasonable bounds
        if bone_lengths is not None:
            # Clamp to [-1.5, 1.5] * bone_length
            if bone_lengths.dim() == 1:
                max_scale = 1.5 * bone_lengths[limb_idx]
            else:
                max_scale = 1.5 * bone_lengths[:, limb_idx]
            scale = scale.clamp(-max_scale, max_scale)
        else:
            # Default clamp for unit torso scale
            scale = scale.clamp(-1.5, 1.5)

        # Store scale factor
        scale_factors[:, limb_idx] = scale

        # Compute child depth
        child_depth = parent_depth + scale * orient_z
        pose_3d[:, child_idx, 2] = child_depth

    # Handle head joints (not in kinematic chain)
    shoulder_center_depth = (
        pose_3d[:, LEFT_SHOULDER_IDX, 2] + pose_3d[:, RIGHT_SHOULDER_IDX, 2]
    ) / 2

    # Nose is typically slightly in front of shoulders
    pose_3d[:, 0, 2] = shoulder_center_depth + 0.05

    # Eyes and ears: same depth as nose
    for idx in [1, 2, 3, 4]:
        pose_3d[:, idx, 2] = pose_3d[:, 0, 2]

    # NaN safety
    nan_mask = torch.isnan(pose_3d).any(dim=-1, keepdim=True)
    if nan_mask.any():
        # Replace NaN with zeros
        pose_3d = torch.where(nan_mask, torch.zeros_like(pose_3d), pose_3d)

    if return_scale_factors:
        return pose_3d, scale_factors
    return pose_3d


def denormalize_pose_3d(
    pose_3d: torch.Tensor,
    pelvis_2d: torch.Tensor,
    torso_scale: torch.Tensor,
    output_depth: float = 2.0,
    metric_torso_scale: Optional[float] = None,
) -> torch.Tensor:
    """Denormalize pose from unit torso scale to meters.

    When metric_torso_scale is provided (computed from known subject height),
    the output is in true metric scale. Otherwise, uses the 2D-derived torso
    scale which gives approximate but not true metric output.

    Args:
        pose_3d: (batch, 17, 3) normalized pose (pelvis-centered, unit torso)
        pelvis_2d: (batch, 2) original pelvis position in image coords
        torso_scale: (batch,) original torso scale from 2D (used for X,Y only
                    when metric_torso_scale is provided)
        output_depth: Target depth for pelvis in output
        metric_torso_scale: If provided, use this for true metric scale output.
                           Computed as: subject_height / HEIGHT_TO_TORSO_RATIO

    Returns:
        (batch, 17, 3) denormalized pose in meter scale
    """
    batch_size = pose_3d.size(0)
    device = pose_3d.device
    dtype = pose_3d.dtype

    # Use metric scale if provided, otherwise fall back to 2D-derived scale
    if metric_torso_scale is not None:
        # True metric output using known subject height
        scale = torch.full((batch_size,), metric_torso_scale, device=device, dtype=dtype)
    else:
        # Approximate scale from 2D observations
        scale = torso_scale

    # Scale back
    output = pose_3d * scale.unsqueeze(-1).unsqueeze(-1)

    # Add back pelvis position (X,Y) scaled appropriately
    # When using metric scale, we still use 2D pelvis position but scale it
    # to maintain correct image-space relationship
    pelvis_3d = torch.zeros(batch_size, 1, 3, device=device, dtype=dtype)
    if metric_torso_scale is not None:
        # Scale pelvis X,Y by ratio of metric to 2D scale
        scale_ratio = scale / torch.clamp(torso_scale, min=1e-6)
        pelvis_3d[:, 0, :2] = pelvis_2d * scale_ratio.unsqueeze(-1)
    else:
        pelvis_3d[:, 0, :2] = pelvis_2d
    pelvis_3d[:, 0, 2] = output_depth
    output = output + pelvis_3d

    return output


def solve_with_denormalization(
    pof: torch.Tensor,
    keypoints_2d: torch.Tensor,
    bone_lengths: Optional[torch.Tensor] = None,
    output_depth: float = 2.0,
    metric_torso_scale: Optional[float] = None,
) -> torch.Tensor:
    """Solve depths and denormalize to meter scale.

    Convenience function that combines solving and denormalization.

    Args:
        pof: (batch, 14, 3) POF unit vectors
        keypoints_2d: (batch, 17, 2) normalized [0,1] image coordinates
        bone_lengths: Optional bone lengths for scale clamping
        output_depth: Target pelvis depth in output
        metric_torso_scale: If provided, use for true metric scale output.
                           Computed as: subject_height / HEIGHT_TO_TORSO_RATIO

    Returns:
        (batch, 17, 3) 3D pose in meter scale
    """
    # Normalize and remember original scale
    normalized_2d, pelvis_2d, torso_scale = normalize_2d_for_pof(keypoints_2d)

    # Solve depths
    pose_3d = solve_depth_least_squares_pof(
        pof, normalized_2d,
        bone_lengths=bone_lengths,
        pelvis_depth=0.0,
        normalize_input=False,
    )

    # Denormalize with metric scale if provided
    return denormalize_pose_3d(
        pose_3d, pelvis_2d, torso_scale, output_depth,
        metric_torso_scale=metric_torso_scale
    )
