"""3D skeleton reconstruction from POF vectors and bone lengths.

Provides two reconstruction methods:
1. Direct forward kinematics (original): child = parent + length * direction
2. Least-squares solver (MTC-style): keeps X,Y fixed, only solves for Z

The least-squares method is preferred as it guarantees the reconstructed
skeleton projects back to the observed 2D positions (reprojection error ≈ 0).
"""

import numpy as np
import torch
from typing import Optional, Dict, Union

from .constants import (
    RECONSTRUCTION_ORDER,
    KINEMATIC_CHAINS,
    JOINT_TO_LIMB,
    LIMB_DEFINITIONS,
    NUM_JOINTS,
    NUM_LIMBS,
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX,
    LEFT_SHOULDER_IDX,
    RIGHT_SHOULDER_IDX,
)
from .least_squares import (
    solve_depth_least_squares_pof,
    normalize_2d_for_pof,
    denormalize_pose_3d,
    enforce_bone_lengths,
)


def normalized_to_meters(
    keypoints_2d: np.ndarray,
    depth: float,
    fov_horizontal_deg: float = 60.0,
    aspect_ratio: float = 16/9,
) -> np.ndarray:
    """Convert normalized [0,1] 2D coordinates to approximate meter coordinates.

    Uses a pinhole camera model to estimate physical positions at a given depth.

    Args:
        keypoints_2d: (N, 2) or (17, 2) normalized [0,1] coordinates
        depth: Depth in meters from camera
        fov_horizontal_deg: Horizontal field of view in degrees
        aspect_ratio: Image width/height ratio

    Returns:
        (N, 2) or (17, 2) coordinates in meters
    """
    # Compute physical dimensions at this depth
    fov_rad = np.deg2rad(fov_horizontal_deg)
    width_m = 2 * depth * np.tan(fov_rad / 2)
    height_m = width_m / aspect_ratio

    # Transform: center at (0.5, 0.5), scale to meters
    # X: positive = right, Y: positive = down (camera convention)
    keypoints_m = keypoints_2d.copy()
    keypoints_m[..., 0] = (keypoints_2d[..., 0] - 0.5) * width_m
    keypoints_m[..., 1] = (keypoints_2d[..., 1] - 0.5) * height_m

    return keypoints_m


def reconstruct_skeleton_from_pof(
    pof: np.ndarray,
    bone_lengths: np.ndarray,
    keypoints_2d: Optional[np.ndarray] = None,
    pelvis_depth: float = 2.0,
    use_meter_coords: bool = False,
    fov_horizontal_deg: float = 60.0,
    aspect_ratio: float = 16/9,
) -> np.ndarray:
    """Reconstruct 3D skeleton from POF vectors and bone lengths.

    Algorithm:
    1. Initialize pelvis (midpoint of hips) at specified depth
    2. Place hips at pelvis with hip width offset
    3. For each joint in kinematic order:
       child_pos = parent_pos + bone_length * POF_unit_vector
    4. Optionally use 2D keypoints for X/Y reference

    Args:
        pof: (14, 3) unit vectors for each limb
        bone_lengths: (14,) lengths for each limb in meters
        keypoints_2d: Optional (17, 2) normalized 2D positions for X/Y reference
        pelvis_depth: Initial depth for pelvis (Z coordinate)
        use_meter_coords: If True, convert normalized 2D to meter coordinates
                         using pinhole camera model. Required for RTMPose.
        fov_horizontal_deg: Horizontal FOV for meter conversion (default 60°)
        aspect_ratio: Image aspect ratio for meter conversion (default 16:9)

    Returns:
        (17, 3) reconstructed 3D joint positions in camera space (meters)
    """
    pose_3d = np.zeros((NUM_JOINTS, 3), dtype=np.float32)

    # Initialize pelvis center
    if keypoints_2d is not None:
        # Convert to meters if requested (for 2D-only estimators like RTMPose)
        if use_meter_coords:
            keypoints_m = normalized_to_meters(
                keypoints_2d, pelvis_depth, fov_horizontal_deg, aspect_ratio
            )
            pelvis_xy = (keypoints_m[LEFT_HIP_IDX] + keypoints_m[RIGHT_HIP_IDX]) / 2
        else:
            # Use normalized coords directly (for MediaPipe with 3D)
            pelvis_xy = (keypoints_2d[LEFT_HIP_IDX] + keypoints_2d[RIGHT_HIP_IDX]) / 2
        pelvis_pos = np.array([pelvis_xy[0], pelvis_xy[1], pelvis_depth], dtype=np.float32)
    else:
        # Default: centered at origin (proper meter space)
        pelvis_pos = np.array([0.0, 0.0, pelvis_depth], dtype=np.float32)

    # Initialize hips from pelvis using hip width (limb 9)
    hip_width = bone_lengths[9]  # Hip width limb
    hip_pof = pof[9]  # Hip width POF (L hip -> R hip direction)

    # Left hip is half hip width from pelvis in opposite direction
    pose_3d[LEFT_HIP_IDX] = pelvis_pos - (hip_width / 2) * hip_pof
    # Right hip is half hip width from pelvis in POF direction
    pose_3d[RIGHT_HIP_IDX] = pelvis_pos + (hip_width / 2) * hip_pof

    # Reconstruct remaining joints following kinematic chains
    for joint_idx in RECONSTRUCTION_ORDER:
        if joint_idx in [LEFT_HIP_IDX, RIGHT_HIP_IDX]:
            # Hips already initialized
            continue

        parent_idx = KINEMATIC_CHAINS[joint_idx]
        limb_idx = JOINT_TO_LIMB[joint_idx]

        parent_pos = pose_3d[parent_idx]
        bone_length = bone_lengths[limb_idx]
        unit_vec = pof[limb_idx]

        # Check if POF direction is reversed for this limb
        # (i.e., POF parent != reconstruction parent)
        pof_parent, pof_child = LIMB_DEFINITIONS[limb_idx]
        if pof_parent != parent_idx:
            # POF points opposite direction, negate it
            unit_vec = -unit_vec

        # Child = Parent + length * direction
        pose_3d[joint_idx] = parent_pos + bone_length * unit_vec

    # Handle head joints (0-4) - not in kinematic chain
    # Place at shoulder center height with appropriate offsets
    shoulder_center = (pose_3d[LEFT_SHOULDER_IDX] + pose_3d[RIGHT_SHOULDER_IDX]) / 2

    for joint_idx in [0, 1, 2, 3, 4]:  # nose, eyes, ears
        if keypoints_2d is not None:
            # Use 2D X/Y from keypoints (converted to meters if needed)
            if use_meter_coords:
                # keypoints_m was computed earlier for pelvis
                head_kp = normalized_to_meters(
                    keypoints_2d[joint_idx:joint_idx+1],
                    shoulder_center[2],  # Use shoulder depth for head
                    fov_horizontal_deg, aspect_ratio
                )[0]
                pose_3d[joint_idx, 0] = head_kp[0]
                pose_3d[joint_idx, 1] = head_kp[1]
            else:
                pose_3d[joint_idx, 0] = keypoints_2d[joint_idx, 0]
                pose_3d[joint_idx, 1] = keypoints_2d[joint_idx, 1]
        else:
            # Approximate from shoulder center
            pose_3d[joint_idx, 0] = shoulder_center[0]
            pose_3d[joint_idx, 1] = shoulder_center[1] - 0.1  # Slightly above shoulders

        # Use shoulder center depth
        pose_3d[joint_idx, 2] = shoulder_center[2]

    return pose_3d


def reconstruct_skeleton_batch(
    pof: np.ndarray,
    bone_lengths: np.ndarray,
    keypoints_2d: Optional[np.ndarray] = None,
    pelvis_depth: float = 2.0,
    use_meter_coords: bool = False,
    fov_horizontal_deg: float = 60.0,
    aspect_ratio: float = 16/9,
) -> np.ndarray:
    """Batch version of reconstruct_skeleton_from_pof.

    Args:
        pof: (batch, 14, 3) unit vectors
        bone_lengths: (14,) or (batch, 14) bone lengths
        keypoints_2d: Optional (batch, 17, 2) 2D positions
        pelvis_depth: Initial depth for pelvis
        use_meter_coords: If True, convert normalized 2D to meter coordinates
        fov_horizontal_deg: Horizontal FOV for meter conversion
        aspect_ratio: Image aspect ratio for meter conversion

    Returns:
        (batch, 17, 3) reconstructed poses
    """
    batch_size = pof.shape[0]

    # Handle single bone_lengths array for whole batch
    if bone_lengths.ndim == 1:
        bone_lengths = np.tile(bone_lengths[np.newaxis, :], (batch_size, 1))

    poses_3d = np.zeros((batch_size, NUM_JOINTS, 3), dtype=np.float32)

    for i in range(batch_size):
        kp_2d = keypoints_2d[i] if keypoints_2d is not None else None
        poses_3d[i] = reconstruct_skeleton_from_pof(
            pof[i], bone_lengths[i], kp_2d, pelvis_depth,
            use_meter_coords, fov_horizontal_deg, aspect_ratio
        )

    return poses_3d


def apply_pof_to_pose(
    input_pose: np.ndarray,
    pof: np.ndarray,
    blend_weight: float = 0.5,
) -> np.ndarray:
    """Apply POF corrections to an existing 3D pose.

    Blends POF-based reconstruction with input pose for gradual
    refinement. Preserves input bone lengths while adjusting directions.

    Args:
        input_pose: (17, 3) existing 3D pose (e.g., from MediaPipe)
        pof: (14, 3) predicted POF unit vectors
        blend_weight: 0 = keep input, 1 = use full POF direction

    Returns:
        (17, 3) refined pose
    """
    result = input_pose.copy()

    # Apply POF directions while preserving input bone lengths
    for joint_idx in RECONSTRUCTION_ORDER:
        if joint_idx in [LEFT_HIP_IDX, RIGHT_HIP_IDX]:
            # Keep hips fixed as root
            continue

        parent_idx = KINEMATIC_CHAINS[joint_idx]
        limb_idx = JOINT_TO_LIMB[joint_idx]

        # Current bone vector and length from input
        current_vec = input_pose[joint_idx] - input_pose[parent_idx]
        current_length = np.linalg.norm(current_vec)

        if current_length < 1e-6:
            continue

        # POF-based position using input bone length
        pof_direction = pof[limb_idx]
        pof_child = result[parent_idx] + current_length * pof_direction

        # Blend between input and POF-based position
        result[joint_idx] = (
            (1 - blend_weight) * input_pose[joint_idx]
            + blend_weight * pof_child
        )

    return result


def apply_pof_to_pose_batch(
    input_poses: np.ndarray,
    pof: np.ndarray,
    blend_weight: float = 0.5,
) -> np.ndarray:
    """Batch version of apply_pof_to_pose.

    Args:
        input_poses: (batch, 17, 3) existing poses
        pof: (batch, 14, 3) predicted POF vectors
        blend_weight: Blend factor

    Returns:
        (batch, 17, 3) refined poses
    """
    batch_size = input_poses.shape[0]
    result = np.zeros_like(input_poses)

    for i in range(batch_size):
        result[i] = apply_pof_to_pose(input_poses[i], pof[i], blend_weight)

    return result


def compute_reconstruction_error(
    reconstructed: np.ndarray,
    ground_truth: np.ndarray,
) -> Dict[str, float]:
    """Compute reconstruction error metrics.

    Args:
        reconstructed: (17, 3) or (batch, 17, 3) reconstructed poses
        ground_truth: (17, 3) or (batch, 17, 3) ground truth poses

    Returns:
        Dictionary with error metrics
    """
    single_pose = reconstructed.ndim == 2
    if single_pose:
        reconstructed = reconstructed[np.newaxis, ...]
        ground_truth = ground_truth[np.newaxis, ...]

    # Per-joint error
    joint_errors = np.linalg.norm(reconstructed - ground_truth, axis=-1)  # (batch, 17)

    # Compute metrics
    mpjpe = float(joint_errors.mean())  # Mean Per Joint Position Error
    max_error = float(joint_errors.max())
    median_error = float(np.median(joint_errors))

    # Per-limb errors
    limb_errors = []
    for parent, child in LIMB_DEFINITIONS:
        parent_err = joint_errors[:, parent].mean()
        child_err = joint_errors[:, child].mean()
        limb_errors.append((parent_err + child_err) / 2)

    return {
        "mpjpe_m": mpjpe,
        "mpjpe_cm": mpjpe * 100,
        "max_error_cm": max_error * 100,
        "median_error_cm": median_error * 100,
        "mean_limb_error_cm": float(np.mean(limb_errors)) * 100,
    }


def normalize_pose_scale(
    pose_3d: np.ndarray,
    target_torso_length: float = 0.5,
) -> np.ndarray:
    """Normalize pose to consistent scale based on torso length.

    Args:
        pose_3d: (17, 3) or (batch, 17, 3) pose
        target_torso_length: Target length for shoulder-hip distance

    Returns:
        Scaled pose
    """
    single_pose = pose_3d.ndim == 2
    if single_pose:
        pose_3d = pose_3d[np.newaxis, ...]

    # Compute current torso length (average of L/R shoulder-to-hip)
    l_torso = np.linalg.norm(
        pose_3d[:, LEFT_SHOULDER_IDX] - pose_3d[:, LEFT_HIP_IDX], axis=-1
    )
    r_torso = np.linalg.norm(
        pose_3d[:, RIGHT_SHOULDER_IDX] - pose_3d[:, RIGHT_HIP_IDX], axis=-1
    )
    current_torso = (l_torso + r_torso) / 2  # (batch,)

    # Compute scale factor
    scale = target_torso_length / np.maximum(current_torso, 1e-6)
    scale = scale[:, np.newaxis, np.newaxis]  # (batch, 1, 1)

    # Center on pelvis, scale, then restore
    pelvis = (pose_3d[:, LEFT_HIP_IDX] + pose_3d[:, RIGHT_HIP_IDX]) / 2
    pelvis = pelvis[:, np.newaxis, :]  # (batch, 1, 3)

    pose_centered = pose_3d - pelvis
    pose_scaled = pose_centered * scale
    pose_3d = pose_scaled + pelvis

    if single_pose:
        pose_3d = pose_3d[0]

    return pose_3d


def reconstruct_skeleton_least_squares(
    pof: Union[np.ndarray, torch.Tensor],
    keypoints_2d: Union[np.ndarray, torch.Tensor],
    bone_lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pelvis_depth: float = 0.0,
    denormalize: bool = False,
    output_depth: float = 2.0,
    metric_torso_scale: Optional[float] = None,
    enforce_bones: bool = False,
    use_2d_xy: bool = True,
    enforce_width: bool = False,
) -> np.ndarray:
    """Reconstruct 3D skeleton using MTC-style least-squares solver.

    This is the preferred reconstruction method. It keeps X,Y coordinates
    fixed from 2D observations and only solves for Z depths, ensuring the
    skeleton projects back to the observed 2D positions.

    Args:
        pof: (14, 3) or (batch, 14, 3) POF unit vectors
        keypoints_2d: (17, 2) or (batch, 17, 2) normalized [0,1] 2D keypoints
        bone_lengths: Optional (14,) or (batch, 14) bone lengths for clamping
        pelvis_depth: Initial Z for pelvis in normalized space (default 0.0)
        denormalize: If True, denormalize output to meter scale
        output_depth: Target pelvis depth when denormalizing
        metric_torso_scale: If provided, use for true metric scale output.
                           Computed as: subject_height / HEIGHT_TO_TORSO_RATIO
                           When None, uses 2D-derived scale (approximate).
        enforce_bones: If True, enforce bone length constraints by adjusting Z.
                      Default False - rely on POF predictions for bone lengths.
        use_2d_xy: If True (default), derive orient_xy from 2D observations.
                  Only uses |Z| from POF. This fixes the torso collapse issue.
        enforce_width: If True, enforce hip and shoulder width bone lengths.
                      Fixes collapsed width in side views.

    Returns:
        (17, 3) or (batch, 17, 3) reconstructed 3D pose
    """
    # Convert to torch if needed
    if isinstance(pof, np.ndarray):
        pof_t = torch.from_numpy(pof.astype(np.float32))
        was_numpy = True
    else:
        pof_t = pof
        was_numpy = False

    if isinstance(keypoints_2d, np.ndarray):
        kp_t = torch.from_numpy(keypoints_2d.astype(np.float32))
    else:
        kp_t = keypoints_2d

    # Handle single frame
    single_frame = pof_t.ndim == 2
    if single_frame:
        pof_t = pof_t.unsqueeze(0)
        kp_t = kp_t.unsqueeze(0)

    # Convert bone lengths
    bl_t = None
    if bone_lengths is not None:
        if isinstance(bone_lengths, np.ndarray):
            bl_t = torch.from_numpy(bone_lengths.astype(np.float32))
        else:
            bl_t = bone_lengths

    # Solve using least-squares
    if denormalize:
        # Normalize, solve, then denormalize
        normalized_2d, pelvis_2d, torso_scale = normalize_2d_for_pof(kp_t)

        # Normalize bone_lengths to unit-torso scale (to match normalized 2D space)
        # In normalized space, torso = 1 unit, so divide by metric torso length
        # This fixes scale clamping: e.g., 0.5m / 0.52m ≈ 0.96 units instead of 0.5
        bl_normalized = None
        if bl_t is not None:
            if metric_torso_scale is not None:
                # Use known metric torso scale from subject height
                bl_normalized = bl_t / metric_torso_scale
            else:
                # Use average torso bone length as normalizer
                # Limbs 10, 11 are L/R torso (shoulder→hip)
                avg_torso = (bl_t[10] + bl_t[11]) / 2
                bl_normalized = bl_t / avg_torso.clamp(min=1e-6)

        # Solve in NORMALIZED space with NORMALIZED bone_lengths
        pose_3d = solve_depth_least_squares_pof(
            pof_t, normalized_2d,
            bone_lengths=bl_normalized,  # Now in unit-torso scale
            pelvis_depth=pelvis_depth,
            normalize_input=False,
            use_2d_xy=use_2d_xy,
            enforce_width=enforce_width,
        )
        pose_3d = denormalize_pose_3d(
            pose_3d, pelvis_2d, torso_scale, output_depth,
            metric_torso_scale=metric_torso_scale
        )
        # Optionally enforce bone lengths in metric space (bl_t is in meters)
        if enforce_bones and bl_t is not None:
            pose_3d = enforce_bone_lengths(pose_3d, bl_t, strength=1.0)
    else:
        # Stay in normalized space
        pose_3d = solve_depth_least_squares_pof(
            pof_t, kp_t,
            bone_lengths=bl_t,
            pelvis_depth=pelvis_depth,
            normalize_input=True,
            use_2d_xy=use_2d_xy,
            enforce_width=enforce_width,
        )

    # Handle single frame
    if single_frame:
        pose_3d = pose_3d.squeeze(0)

    # Convert back to numpy if input was numpy
    if was_numpy:
        pose_3d = pose_3d.numpy()

    return pose_3d


def reconstruct_skeleton_least_squares_batch(
    pof: Union[np.ndarray, torch.Tensor],
    keypoints_2d: Union[np.ndarray, torch.Tensor],
    bone_lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pelvis_depth: float = 0.0,
    denormalize: bool = False,
    output_depth: float = 2.0,
    metric_torso_scale: Optional[float] = None,
    enforce_bones: bool = False,
    use_2d_xy: bool = True,
) -> np.ndarray:
    """Batch version of reconstruct_skeleton_least_squares.

    Args:
        pof: (batch, 14, 3) POF unit vectors
        keypoints_2d: (batch, 17, 2) normalized [0,1] 2D keypoints
        bone_lengths: Optional (14,) or (batch, 14) bone lengths
        pelvis_depth: Initial Z for pelvis
        denormalize: If True, denormalize output
        output_depth: Target pelvis depth when denormalizing
        metric_torso_scale: If provided, use for true metric scale output
        enforce_bones: If True, enforce bone length constraints
        use_2d_xy: If True (default), derive orient_xy from 2D observations

    Returns:
        (batch, 17, 3) reconstructed 3D poses
    """
    return reconstruct_skeleton_least_squares(
        pof, keypoints_2d, bone_lengths,
        pelvis_depth, denormalize, output_depth,
        metric_torso_scale=metric_torso_scale,
        enforce_bones=enforce_bones,
        use_2d_xy=use_2d_xy,
    )
