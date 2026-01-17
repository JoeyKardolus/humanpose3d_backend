"""
Loss functions for 3D pose refinement training.

Key losses:
1. Primary 3D correction loss (L1 on all axes, with depth emphasized)
2. Bone length consistency (corrected pose should have consistent bone lengths)
3. Symmetry (left/right limbs should have similar bone lengths)
4. Confidence calibration (high confidence should correlate with low error)
5. Limb orientation loss (POF-inspired: predicted limb orientations should match GT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# Limb definitions for Part Orientation Fields (POF-inspired)
# Each limb is (parent_joint, child_joint) using COCO-17 indices
LIMBS = [
    (5, 7),    # 0: L shoulder → elbow
    (7, 9),    # 1: L elbow → wrist
    (6, 8),    # 2: R shoulder → elbow
    (8, 10),   # 3: R elbow → wrist
    (11, 13),  # 4: L hip → knee
    (13, 15),  # 5: L knee → ankle
    (12, 14),  # 6: R hip → knee
    (14, 16),  # 7: R knee → ankle
    (5, 6),    # 8: Shoulder width
    (11, 12),  # 9: Hip width
    (5, 11),   # 10: L torso
    (6, 12),   # 11: R torso
    # Cross-body diagonals (help with reaching poses and torso twist)
    (5, 12),   # 12: L shoulder → R hip
    (6, 11),   # 13: R shoulder → L hip
]


def compute_limb_orientations_from_pose(pose_3d: torch.Tensor) -> torch.Tensor:
    """Compute 3D unit vectors for each limb from a pose.

    Args:
        pose_3d: (batch, 17, 3) or (17, 3) joint positions

    Returns:
        (batch, 14, 3) or (14, 3) unit vectors for each limb
    """
    single_pose = pose_3d.dim() == 2
    if single_pose:
        pose_3d = pose_3d.unsqueeze(0)  # (1, 17, 3)

    orientations = []
    for parent, child in LIMBS:
        vec = pose_3d[:, child] - pose_3d[:, parent]  # (batch, 3)
        length = torch.norm(vec, dim=-1, keepdim=True)  # (batch, 1)
        unit_vec = vec / (length + 1e-8)  # (batch, 3)
        orientations.append(unit_vec)

    orientations = torch.stack(orientations, dim=1)  # (batch, 14, 3)

    if single_pose:
        orientations = orientations[0]  # (14, 3)

    return orientations


def limb_orientation_loss(
    pred_orientations: torch.Tensor,
    gt_orientations: torch.Tensor,
    visibility: torch.Tensor = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Cosine similarity loss for limb orientations (POF-inspired).

    Measures how well predicted limb orientations match ground truth.
    Uses 1 - cos(theta) so that perfect alignment = 0 loss.

    Args:
        pred_orientations: (batch, 14, 3) predicted unit vectors per limb
        gt_orientations: (batch, 14, 3) ground truth unit vectors per limb
        visibility: (batch, 17) optional per-joint visibility scores.
                   If provided, weights each limb by min(parent_vis, child_vis)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Cosine similarity loss (0 = perfect alignment, 2 = opposite directions)
    """
    # Cosine similarity: dot product of unit vectors
    # cos(theta) = pred · gt (both unit vectors)
    cos_sim = (pred_orientations * gt_orientations).sum(dim=-1)  # (batch, 14)

    # Loss = 1 - cos(theta)
    # cos(0°) = 1 → loss = 0 (perfect)
    # cos(90°) = 0 → loss = 1 (orthogonal)
    # cos(180°) = -1 → loss = 2 (opposite)
    loss = 1.0 - cos_sim  # (batch, 14)

    # Apply visibility weighting if provided
    if visibility is not None:
        # Compute per-limb visibility as min of parent and child joint visibility
        limb_vis = []
        for parent, child in LIMBS:
            vis = torch.min(visibility[:, parent], visibility[:, child])  # (batch,)
            limb_vis.append(vis)
        limb_vis = torch.stack(limb_vis, dim=1)  # (batch, 14)

        # Weight loss by visibility (low vis limbs contribute less)
        loss = loss * limb_vis

        if reduction == 'mean':
            # Weighted mean: sum(loss * weight) / sum(weight)
            return loss.sum() / (limb_vis.sum() + 1e-6)
        elif reduction == 'sum':
            return loss.sum()
        return loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def limb_orientation_angle_error(
    pred_orientations: torch.Tensor,
    gt_orientations: torch.Tensor,
) -> torch.Tensor:
    """Compute angular error between predicted and GT limb orientations in degrees.

    Args:
        pred_orientations: (batch, 14, 3) predicted unit vectors
        gt_orientations: (batch, 14, 3) ground truth unit vectors

    Returns:
        (batch, 14) angular error in degrees for each limb
    """
    # Cosine of angle between vectors
    cos_theta = (pred_orientations * gt_orientations).sum(dim=-1)  # (batch, 14)
    cos_theta = cos_theta.clamp(-1.0, 1.0)  # Numerical stability

    # Convert to degrees
    angle_rad = torch.acos(cos_theta)
    angle_deg = angle_rad * (180.0 / 3.14159265)

    return angle_deg


def projection_consistency_loss(
    corrected_3d: torch.Tensor,
    observed_2d: torch.Tensor,
    visibility: torch.Tensor = None,
    projection: str = 'ortho',
) -> torch.Tensor:
    """Loss to ensure 3D pose projects back to observed 2D positions.

    This is a key constraint from MonocularTotalCapture: the reconstructed 3D
    must be consistent with the 2D observations. If it doesn't project back
    correctly, the 3D is wrong.

    NOTE: For orthographic projection with least-squares solver, this loss is
    always 0 because the solver guarantees X,Y match. Use solved_depth_loss instead.

    Args:
        corrected_3d: (batch, 17, 3) corrected 3D pose
        observed_2d: (batch, 17, 2) observed 2D joint positions
        visibility: (batch, 17) optional visibility weights
        projection: 'ortho' (orthographic) or 'weak_persp' (weak perspective)

    Returns:
        Scalar loss measuring 2D reprojection error
    """
    # Handle any NaN/Inf in inputs
    corrected_3d = torch.nan_to_num(corrected_3d, nan=0.0, posinf=1.0, neginf=-1.0)

    if projection == 'ortho':
        # Orthographic: X, Y directly correspond to x, y
        projected_2d = corrected_3d[:, :, :2]  # (batch, 17, 2)
    else:
        # Weak perspective: x = X/Z, y = Y/Z
        z = corrected_3d[:, :, 2:3].clamp(min=0.1)  # Avoid division by zero
        projected_2d = corrected_3d[:, :, :2] / z  # (batch, 17, 2)

    # L2 error between projected and observed
    error = (projected_2d - observed_2d) ** 2  # (batch, 17, 2)
    error = error.sum(dim=-1)  # (batch, 17) - squared distance per joint

    # Clamp error to prevent extreme values
    error = error.clamp(max=10.0)

    if visibility is not None:
        # Weight by visibility
        weighted_error = error * visibility
        return weighted_error.sum() / (visibility.sum() + 1e-6)
    else:
        return error.mean()


def solved_depth_loss(
    solved_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    visibility: torch.Tensor = None,
) -> torch.Tensor:
    """Loss comparing depths from least-squares solver to ground truth.

    This is the KEY training signal for the MTC-style approach:
    "If we trust the predicted limb orientations and solve for depth,
    do we get the correct depths?"

    This trains the network to predict orientations that lead to correct depths.

    Args:
        solved_pose: (batch, 17, 3) pose from least-squares solver
        gt_pose: (batch, 17, 3) ground truth pose
        visibility: (batch, 17) optional visibility weights

    Returns:
        Scalar loss measuring depth error
    """
    # Handle any NaN/Inf
    solved_pose = torch.nan_to_num(solved_pose, nan=0.0, posinf=1.0, neginf=-1.0)

    # Compare depths (Z axis)
    depth_error = (solved_pose[:, :, 2] - gt_pose[:, :, 2]).abs()  # (batch, 17)

    # Clamp to prevent extreme values
    depth_error = depth_error.clamp(max=1.0)

    if visibility is not None:
        weighted_error = depth_error * visibility
        return weighted_error.sum() / (visibility.sum() + 1e-6)
    else:
        return depth_error.mean()


def scale_factor_regularization(
    scale_factors: torch.Tensor,
    target_scale: float = 0.3,
) -> torch.Tensor:
    """Regularize scale factors from least-squares solver.

    MTC insight: negative scale factors indicate wrong limb direction.
    This loss encourages positive scale factors close to expected bone lengths.

    Args:
        scale_factors: (batch, 14) scale factors from least-squares solver
        target_scale: expected average scale (related to bone length, ~0.3m)

    Returns:
        Scalar loss penalizing negative and extreme scale factors
    """
    # Handle any NaN/Inf in scale factors (shouldn't happen but be safe)
    scale_factors = torch.nan_to_num(scale_factors, nan=0.0, posinf=2.0, neginf=-2.0)

    # Penalize negative scale factors (wrong direction)
    negative_penalty = F.relu(-scale_factors).mean()

    # Penalize extreme scale factors (too large or too small)
    # Expected scale is roughly bone_length for unit orientation vectors
    deviation = (scale_factors.abs() - target_scale).abs()
    extreme_penalty = deviation.mean()

    return negative_penalty + 0.1 * extreme_penalty


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

# Symmetric bone pairs (left_bone_idx, right_bone_idx) - indices into COCO_BONES
SYMMETRIC_PAIRS = [
    (4, 6),   # left upper arm vs right upper arm
    (5, 7),   # left forearm vs right forearm
    (8, 10),  # left thigh vs right thigh
    (9, 11),  # left shin vs right shin
]

# Symmetric bone pairs as joint indices: ((left_joint1, left_joint2), (right_joint1, right_joint2))
# Used for visibility-weighted symmetric bone constraints
SYMMETRIC_BONE_PAIRS = [
    ((5, 7), (6, 8)),     # upper arms: L-shoulder→elbow vs R-shoulder→elbow
    ((7, 9), (8, 10)),    # forearms: L-elbow→wrist vs R-elbow→wrist
    ((11, 13), (12, 14)), # thighs: L-hip→knee vs R-hip→knee
    ((13, 15), (14, 16)), # shins: L-knee→ankle vs R-knee→ankle
]

# Core torso bones for 2D→3D width prediction
# Order matters: [shoulder_width, hip_width, left_torso, right_torso]
TORSO_BONE_INDICES = [
    (5, 6),    # shoulder width: left_shoulder - right_shoulder
    (11, 12),  # hip width: left_hip - right_hip
    (5, 11),   # left torso: left_shoulder - left_hip
    (6, 12),   # right torso: right_shoulder - right_hip
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


def pose_correction_loss(
    pred_delta_xyz: torch.Tensor,
    corrupted_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    visibility: torch.Tensor = None,
    reduction: str = 'mean',
    depth_weight: float = 2.0,
) -> torch.Tensor:
    """Primary loss: Predicted 3D correction should match needed correction.

    Args:
        pred_delta_xyz: (batch, 17, 3) predicted 3D corrections
        corrupted_pose: (batch, 17, 3) input pose with errors
        gt_pose: (batch, 17, 3) ground truth pose
        visibility: (batch, 17) optional visibility weights
        reduction: 'mean', 'sum', or 'none'
        depth_weight: Extra weight on Z (depth) axis since it has largest errors

    Returns:
        Weighted L1 loss between predicted and actual 3D corrections
    """
    # Actual 3D correction needed
    gt_delta_xyz = gt_pose - corrupted_pose  # (batch, 17, 3)

    # L1 loss per joint per axis
    loss = F.l1_loss(pred_delta_xyz, gt_delta_xyz, reduction='none')  # (batch, 17, 3)

    # Weight axes differently: depth (Z) typically has larger errors
    axis_weights = torch.tensor([1.0, 1.0, depth_weight], device=loss.device)
    loss = loss * axis_weights.view(1, 1, 3)  # (batch, 17, 3)

    # NOTE: Visibility weighting REMOVED - was causing train/eval mismatch
    # Training weighted low-visibility joints less, but eval measures all equally
    # Model can still use visibility as INPUT feature to know which joints are uncertain
    # If this doesn't help depth, try foreshortening-based uncertainty instead

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


# Backward compatibility alias
def depth_correction_loss(
    pred_delta_z: torch.Tensor,
    corrupted_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    visibility: torch.Tensor = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Legacy function - use pose_correction_loss for new code."""
    # Convert delta_z to delta_xyz (zeros for x, y)
    batch_size = pred_delta_z.shape[0]
    num_joints = pred_delta_z.shape[1]
    pred_delta_xyz = torch.zeros(batch_size, num_joints, 3, device=pred_delta_z.device)
    pred_delta_xyz[:, :, 2] = pred_delta_z
    return pose_correction_loss(pred_delta_xyz, corrupted_pose, gt_pose, visibility, reduction, depth_weight=1.0)


def bone_length_loss(
    pred_delta_xyz: torch.Tensor,
    corrupted_pose: torch.Tensor,
    gt_pose: torch.Tensor,
) -> torch.Tensor:
    """Bone length consistency loss.

    The corrected pose should have similar bone lengths to ground truth.

    Args:
        pred_delta_xyz: (batch, 17, 3) predicted 3D corrections
        corrupted_pose: (batch, 17, 3) input pose
        gt_pose: (batch, 17, 3) ground truth pose

    Returns:
        Mean absolute difference in bone lengths
    """
    # Apply 3D correction
    corrected_pose = corrupted_pose + pred_delta_xyz

    # Compute bone lengths
    corrected_lengths = compute_bone_lengths(corrected_pose)
    gt_lengths = compute_bone_lengths(gt_pose)

    # L1 loss on bone lengths
    return F.l1_loss(corrected_lengths, gt_lengths)


def bone_variance_loss(corrected_pose: torch.Tensor) -> torch.Tensor:
    """Penalize variance of bone lengths across batch.

    Encourages model to predict temporally consistent bone lengths.
    Real bones don't change length frame-to-frame.

    Args:
        corrected_pose: (batch, 17, 3) corrected poses

    Returns:
        Scalar loss = mean variance across all bones
    """
    bone_lengths = compute_bone_lengths(corrected_pose)  # (batch, num_bones)

    # Variance of each bone across batch
    variance_per_bone = bone_lengths.var(dim=0)  # (num_bones,)

    return variance_per_bone.mean()


def compute_median_bone_lengths(poses: torch.Tensor) -> torch.Tensor:
    """Compute median bone length for each bone across a temporal window.

    Args:
        poses: (window_size, 17, 3) poses from same person

    Returns:
        (num_bones,) median length for each bone
    """
    bone_lengths = compute_bone_lengths(poses)  # (window_size, num_bones)
    return bone_lengths.median(dim=0).values


def project_to_bone_lengths(
    poses: torch.Tensor,
    target_lengths: torch.Tensor,
    iterations: int = 3,
) -> torch.Tensor:
    """Project poses to satisfy bone length constraints.

    Uses iterative projection: adjust child joint along bone direction
    to achieve target length, prioritizing depth (Z) corrections.

    Args:
        poses: (N, 17, 3) poses to project
        target_lengths: (num_bones,) target length for each bone
        iterations: Number of projection iterations

    Returns:
        (N, 17, 3) projected poses
    """
    result = poses.clone()

    # Define parent-child relationships for each bone
    # Format: (bone_tuple, child_idx)
    # Parent is the more proximal/stable joint
    BONE_HIERARCHY = [
        # Torso first (most stable)
        (0, 6),   # shoulder width: left shoulder (5) is parent, right (6) is child
        (1, 12),  # hip width: left hip (11) is parent, right (12) is child
        (2, 11),  # left torso: left shoulder (5) is parent, left hip (11) is child
        (3, 12),  # right torso: right shoulder (6) is parent, right hip (12) is child
        # Arms (shoulder → elbow → wrist)
        (4, 7),   # left upper arm: shoulder (5) -> elbow (7)
        (5, 9),   # left forearm: elbow (7) -> wrist (9)
        (6, 8),   # right upper arm: shoulder (6) -> elbow (8)
        (7, 10),  # right forearm: elbow (8) -> wrist (10)
        # Legs (hip → knee → ankle)
        (8, 13),  # left thigh: hip (11) -> knee (13)
        (9, 15),  # left shin: knee (13) -> ankle (15)
        (10, 14), # right thigh: hip (12) -> knee (14)
        (11, 16), # right shin: knee (14) -> ankle (16)
    ]

    for _ in range(iterations):
        for bone_idx, child_idx in BONE_HIERARCHY:
            bone = COCO_BONES[bone_idx]
            parent_idx = bone[0] if bone[1] == child_idx else bone[1]
            target_len = target_lengths[bone_idx]

            # Current bone vector for all frames
            bone_vec = result[:, child_idx] - result[:, parent_idx]  # (N, 3)
            current_len = torch.norm(bone_vec, dim=-1, keepdim=True)  # (N, 1)

            # Skip if bone length is tiny (avoid division by zero)
            valid = current_len.squeeze(-1) > 1e-6
            if not valid.any():
                continue

            # Compute new child position along bone direction
            bone_dir = bone_vec / (current_len + 1e-8)  # (N, 3)
            new_child = result[:, parent_idx] + bone_dir * target_len

            # Blend: prioritize Z (depth) correction (80%), less XY change (20%)
            # This preserves the 2D appearance while fixing depth
            blend = torch.zeros_like(new_child)
            blend[:, :2] = 0.2 * (new_child[:, :2] - result[:, child_idx, :2])  # XY: 20%
            blend[:, 2] = 0.8 * (new_child[:, 2] - result[:, child_idx, 2])      # Z: 80%

            # Apply only to valid frames
            result[:, child_idx] = result[:, child_idx] + blend * valid.float().unsqueeze(-1)

    return result


def apply_bone_locking(
    corrected_poses: torch.Tensor,
    calibration_frames: int = 50,
) -> torch.Tensor:
    """Apply bone locking to a temporal window.

    1. Compute median bone lengths from first N frames
    2. Project all frames to satisfy those bone lengths

    Args:
        corrected_poses: (window_size, 17, 3) corrected poses from same person
        calibration_frames: Number of frames to compute median from

    Returns:
        (window_size, 17, 3) bone-locked poses
    """
    n_calib = min(calibration_frames, len(corrected_poses))

    # Compute reference bone lengths from calibration frames
    reference_lengths = compute_median_bone_lengths(corrected_poses[:n_calib])

    # Project all frames to those bone lengths
    locked = project_to_bone_lengths(corrected_poses, reference_lengths)

    return locked


def symmetry_loss(
    pred_delta_xyz: torch.Tensor,
    corrupted_pose: torch.Tensor,
) -> torch.Tensor:
    """Symmetry loss: Left and right limbs should have similar bone lengths.

    This is a soft constraint - humans aren't perfectly symmetric, but
    large asymmetries usually indicate errors.

    Args:
        pred_delta_xyz: (batch, 17, 3) predicted 3D corrections
        corrupted_pose: (batch, 17, 3) input pose

    Returns:
        Mean asymmetry penalty
    """
    # Apply 3D correction
    corrected_pose = corrupted_pose + pred_delta_xyz

    # Compute bone lengths
    lengths = compute_bone_lengths(corrected_pose)

    # Compare symmetric pairs
    asymmetry = 0.0
    for left_idx, right_idx in SYMMETRIC_PAIRS:
        left_len = lengths[:, left_idx]
        right_len = lengths[:, right_idx]
        asymmetry = asymmetry + torch.abs(left_len - right_len).mean()

    return asymmetry / len(SYMMETRIC_PAIRS)


def visibility_weighted_symmetric_bone_loss(
    pred_delta_xyz: torch.Tensor,
    corrupted_pose: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Visibility-weighted symmetric bone length constraint.

    Key insight: Trust high-visibility bones, constrain low-visibility bones to match.

    For each symmetric pair (e.g., left arm vs right arm):
    - Compute visibility-weighted reference bone length
    - Both bones should match this reference
    - Higher visibility side contributes more to reference

    This prevents shrinkage while allowing depth correction:
    - Oblique view: trust near side, correct far side
    - Frontal view: both sides contribute equally

    Args:
        pred_delta_xyz: (batch, 17, 3) predicted 3D corrections
        corrupted_pose: (batch, 17, 3) input pose
        visibility: (batch, 17) per-joint visibility scores

    Returns:
        Mean symmetric bone constraint loss
    """
    corrected_pose = corrupted_pose + pred_delta_xyz

    total_loss = 0.0
    for (li, lj), (ri, rj) in SYMMETRIC_BONE_PAIRS:
        # Compute bone lengths for corrected pose
        left_bone = torch.norm(corrected_pose[:, li] - corrected_pose[:, lj], dim=-1)
        right_bone = torch.norm(corrected_pose[:, ri] - corrected_pose[:, rj], dim=-1)

        # Visibility of each bone = min of both joints
        left_vis = torch.min(visibility[:, li], visibility[:, lj])   # (batch,)
        right_vis = torch.min(visibility[:, ri], visibility[:, rj])  # (batch,)

        # Visibility-weighted reference length
        total_vis = left_vis + right_vis + 1e-6
        ref_length = (left_bone * left_vis + right_bone * right_vis) / total_vis

        # Both bones should match reference, weighted by visibility
        # Higher visibility = more responsibility to be correct
        left_loss = left_vis * torch.abs(left_bone - ref_length)
        right_loss = right_vis * torch.abs(right_bone - ref_length)

        total_loss = total_loss + (left_loss + right_loss).mean()

    return total_loss / len(SYMMETRIC_BONE_PAIRS)


def scale_preservation_loss(
    pred_delta_xyz: torch.Tensor,
    corrupted_pose: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Prevent overall pose shrinkage, weighted by visibility.

    This is a backup constraint to prevent uniform scaling of the skeleton.

    Args:
        pred_delta_xyz: (batch, 17, 3) predicted 3D corrections
        corrupted_pose: (batch, 17, 3) input pose
        visibility: (batch, 17) per-joint visibility scores

    Returns:
        Mean scale preservation loss
    """
    corrected = corrupted_pose + pred_delta_xyz

    # Distance from origin (pelvis) for each joint
    input_distances = torch.norm(corrupted_pose, dim=-1)   # (batch, 17)
    output_distances = torch.norm(corrected, dim=-1)       # (batch, 17)

    # Visibility-weighted mean scale
    vis_sum = visibility.sum(dim=1, keepdim=True) + 1e-6
    input_scale = (input_distances * visibility).sum(dim=1) / vis_sum.squeeze()
    output_scale = (output_distances * visibility).sum(dim=1) / vis_sum.squeeze()

    return F.l1_loss(output_scale, input_scale)


def compute_torso_lengths(pose: torch.Tensor) -> torch.Tensor:
    """Compute torso bone lengths from pose.

    Args:
        pose: (batch, 17, 3) joint positions

    Returns:
        (batch, 4) bone lengths [shoulder_width, hip_width, left_torso, right_torso]
    """
    lengths = []
    for i, j in TORSO_BONE_INDICES:
        bone_vec = pose[:, i] - pose[:, j]
        length = torch.norm(bone_vec, dim=-1)
        lengths.append(length)
    return torch.stack(lengths, dim=-1)


def torso_width_prediction_loss(
    pred_delta_xyz: torch.Tensor,
    corrupted_pose: torch.Tensor,
    pred_torso_lengths: torch.Tensor,
    gt_torso_lengths: torch.Tensor,
    visibility: torch.Tensor,
) -> tuple:
    """Torso bone constraint using predicted reference lengths.

    Key insight: We can't use input bone lengths as reference (corrupted by depth errors).
    Instead, we predict expected 3D lengths from 2D foreshortening + view angle.

    Two loss components:
    1. width_loss: Corrected bone lengths should match PREDICTED reference
    2. predictor_loss: Train the predictor to predict GT lengths

    Args:
        pred_delta_xyz: (batch, 17, 3) predicted 3D corrections
        corrupted_pose: (batch, 17, 3) input pose
        pred_torso_lengths: (batch, 4) predicted reference [shoulder, hip, L-torso, R-torso]
        gt_torso_lengths: (batch, 4) GT bone lengths for training the predictor
        visibility: (batch, 17) per-joint visibility scores

    Returns:
        tuple: (width_loss, predictor_loss)
    """
    corrected_pose = corrupted_pose + pred_delta_xyz

    # Compute actual bone lengths after correction
    actual_lengths = compute_torso_lengths(corrected_pose)  # (batch, 4)

    # Visibility weights per bone (min of both joints)
    vis_weights = []
    for i, j in TORSO_BONE_INDICES:
        vis_weights.append(torch.min(visibility[:, i], visibility[:, j]))
    vis_weights = torch.stack(vis_weights, dim=-1)  # (batch, 4)

    # Loss 1: Corrected lengths match PREDICTED reference (not corrupted input!)
    width_loss = (vis_weights * (actual_lengths - pred_torso_lengths).abs()).mean()

    # Loss 2: Train predictor to predict GT lengths (supervised)
    predictor_loss = F.l1_loss(pred_torso_lengths, gt_torso_lengths)

    return width_loss, predictor_loss


def confidence_calibration_loss(
    pred_delta_xyz: torch.Tensor,
    pred_confidence: torch.Tensor,
    corrupted_pose: torch.Tensor,
    gt_pose: torch.Tensor,
) -> torch.Tensor:
    """Confidence should correlate with actual accuracy.

    High confidence -> low error
    Low confidence -> high error

    This encourages the network to "know when it doesn't know."

    Args:
        pred_delta_xyz: (batch, 17, 3) predicted 3D corrections
        pred_confidence: (batch, 17) predicted confidence (0-1)
        corrupted_pose: (batch, 17, 3) input pose
        gt_pose: (batch, 17, 3) ground truth

    Returns:
        Negative correlation penalty
    """
    # Actual 3D error per joint (euclidean distance)
    gt_delta_xyz = gt_pose - corrupted_pose
    error_per_joint = torch.norm(pred_delta_xyz - gt_delta_xyz, dim=-1)  # (batch, 17)

    # Normalize error to [0, 1] range for comparison
    error_norm = error_per_joint / (error_per_joint.max() + 1e-6)

    # We want: high confidence where error is low
    # Loss: confidence should be (1 - normalized_error)
    target_confidence = 1.0 - error_norm.detach()  # Detach to avoid gradient issues

    return F.mse_loss(pred_confidence, target_confidence)


class DepthRefinementLoss(nn.Module):
    """Combined loss for 3D pose refinement training.

    Balances multiple objectives:
    1. Primary 3D correction (most important, depth weighted higher)
    2. Bone length consistency (biomechanical validity)
    3. Symmetry (soft constraint)
    4. Confidence calibration (optional)
    5. Torso width prediction (prevents Z-axis flattening)
    """

    def __init__(
        self,
        pose_weight: float = 1.0,
        bone_weight: float = 0.1,
        symmetry_weight: float = 0.05,
        confidence_weight: float = 0.1,
        depth_axis_weight: float = 2.0,  # Extra weight on Z axis
        symmetric_bone_weight: float = 0.3,  # Visibility-weighted symmetric bone constraint
        scale_preservation_weight: float = 0.2,  # Prevent uniform shrinkage
        torso_width_weight: float = 0.5,  # Constraint on torso bone lengths
        torso_predictor_weight: float = 0.3,  # Train width predictor to match GT
    ):
        super().__init__()
        self.pose_weight = pose_weight
        self.bone_weight = bone_weight
        self.symmetry_weight = symmetry_weight
        self.confidence_weight = confidence_weight
        self.depth_axis_weight = depth_axis_weight
        self.symmetric_bone_weight = symmetric_bone_weight
        self.scale_preservation_weight = scale_preservation_weight
        self.torso_width_weight = torso_width_weight
        self.torso_predictor_weight = torso_predictor_weight

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        corrupted_pose: torch.Tensor,
        gt_pose: torch.Tensor,
        visibility: torch.Tensor,
        gt_torso_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            model_output: dict with 'delta_xyz', optionally 'confidence', 'pred_torso_lengths'
            corrupted_pose: (batch, 17, 3) input pose
            gt_pose: (batch, 17, 3) ground truth
            visibility: (batch, 17) visibility scores
            gt_torso_lengths: (batch, 4) GT torso bone lengths for training predictor

        Returns:
            dict with 'total' and individual loss components
        """
        pred_delta_xyz = model_output['delta_xyz']

        losses = {}

        # 1. Primary 3D correction loss (depth weighted higher)
        losses['pose'] = pose_correction_loss(
            pred_delta_xyz, corrupted_pose, gt_pose, visibility,
            depth_weight=self.depth_axis_weight
        )

        # 2. Bone length consistency
        losses['bone'] = bone_length_loss(pred_delta_xyz, corrupted_pose, gt_pose)

        # 3. Symmetry
        losses['symmetry'] = symmetry_loss(pred_delta_xyz, corrupted_pose)

        # 4. Confidence calibration (if available)
        if 'confidence' in model_output:
            losses['confidence'] = confidence_calibration_loss(
                pred_delta_xyz,
                model_output['confidence'],
                corrupted_pose,
                gt_pose,
            )
        else:
            losses['confidence'] = torch.tensor(0.0, device=pred_delta_xyz.device)

        # 5. Visibility-weighted symmetric bone constraint (prevents shrinkage)
        losses['symmetric_bone'] = visibility_weighted_symmetric_bone_loss(
            pred_delta_xyz, corrupted_pose, visibility
        )

        # 6. Scale preservation (backup to prevent uniform shrinkage)
        losses['scale'] = scale_preservation_loss(
            pred_delta_xyz, corrupted_pose, visibility
        )

        # 7. Torso width prediction loss (replaces old torso_preservation_loss)
        # Uses predicted reference lengths instead of corrupted input
        if 'pred_torso_lengths' in model_output and gt_torso_lengths is not None:
            torso_width, torso_predictor = torso_width_prediction_loss(
                pred_delta_xyz, corrupted_pose,
                model_output['pred_torso_lengths'], gt_torso_lengths,
                visibility
            )
            losses['torso_width'] = torso_width
            losses['torso_predictor'] = torso_predictor
        else:
            # Fallback: no torso constraint if predictor not available
            losses['torso_width'] = torch.tensor(0.0, device=pred_delta_xyz.device)
            losses['torso_predictor'] = torch.tensor(0.0, device=pred_delta_xyz.device)

        # Combined loss
        losses['total'] = (
            self.pose_weight * losses['pose'] +
            self.bone_weight * losses['bone'] +
            self.symmetry_weight * losses['symmetry'] +
            self.confidence_weight * losses['confidence'] +
            self.symmetric_bone_weight * losses['symmetric_bone'] +
            self.scale_preservation_weight * losses['scale'] +
            self.torso_width_weight * losses['torso_width'] +
            self.torso_predictor_weight * losses['torso_predictor']
        )

        return losses


if __name__ == '__main__':
    # Quick test
    batch_size = 4
    corrupted = torch.randn(batch_size, 17, 3)
    gt = corrupted.clone()
    gt[:, :, 2] += torch.randn(batch_size, 17) * 0.1  # Add some depth difference
    gt[:, :, :2] += torch.randn(batch_size, 17, 2) * 0.02  # Small XY difference

    visibility = torch.rand(batch_size, 17)

    # Now outputs (batch, 17, 3) for delta_xyz
    pred_delta_xyz = torch.randn(batch_size, 17, 3) * 0.05
    pred_conf = torch.rand(batch_size, 17)

    # Simulate torso predictor output
    pred_torso_lengths = torch.rand(batch_size, 4) * 0.5 + 0.2  # Random positive lengths
    gt_torso_lengths = compute_torso_lengths(gt)

    model_output = {
        'delta_xyz': pred_delta_xyz,
        'confidence': pred_conf,
        'pred_torso_lengths': pred_torso_lengths,
    }

    loss_fn = DepthRefinementLoss()
    losses = loss_fn(model_output, corrupted, gt, visibility, gt_torso_lengths)

    print("Losses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")

    # Test without torso predictor (backward compat)
    print("\nWithout torso predictor:")
    model_output_no_torso = {'delta_xyz': pred_delta_xyz, 'confidence': pred_conf}
    losses_no_torso = loss_fn(model_output_no_torso, corrupted, gt, visibility)
    for name, value in losses_no_torso.items():
        print(f"  {name}: {value.item():.4f}")
