"""
Shared utilities for depth refinement training data conversion.

Contains functions for body frame alignment between MediaPipe and ground truth poses.
"""

import numpy as np


def get_body_frame(pose: np.ndarray) -> np.ndarray:
    """
    Compute body-relative coordinate frame from a COCO-17 pose.

    The body frame is defined by:
    - Y axis: up (from pelvis to shoulder midpoint)
    - X axis: right (from left to right shoulder, orthogonalized)
    - Z axis: forward (cross product of X and Y)

    Args:
        pose: (17, 3) array of COCO-17 keypoints
              Indices: 5=L_shoulder, 6=R_shoulder, 11=L_hip, 12=R_hip

    Returns:
        (3, 3) rotation matrix where columns are [x_axis, y_axis, z_axis]
    """
    # Key points for body frame computation
    l_shoulder = pose[5]
    r_shoulder = pose[6]
    l_hip = pose[11]
    r_hip = pose[12]

    # Pelvis (hip midpoint) and shoulder midpoint
    pelvis = (l_hip + r_hip) / 2
    shoulder_mid = (l_shoulder + r_shoulder) / 2

    # Y axis: up (from pelvis toward shoulders)
    y_axis = shoulder_mid - pelvis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        return np.eye(3)
    y_axis = y_axis / y_norm

    # X axis: right (from left shoulder to right shoulder)
    x_axis = r_shoulder - l_shoulder

    # Orthogonalize X to Y (remove component parallel to Y)
    x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return np.eye(3)
    x_axis = x_axis / x_norm

    # Z axis: forward (cross product gives right-handed system)
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return np.eye(3)
    z_axis = z_axis / z_norm

    # Build rotation matrix with columns [x, y, z]
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def align_body_frames(source_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """
    Rotate source pose to align its body frame with target pose's body frame.

    This corrects rotation differences between MediaPipe and ground truth poses
    so the model learns actual depth errors, not rotation misalignment.

    Both poses should be centered on pelvis before calling this function.

    Args:
        source_pose: (17, 3) pose to be rotated (typically MediaPipe)
        target_pose: (17, 3) reference pose (typically ground truth)

    Returns:
        (17, 3) rotated source pose aligned with target's body frame

    Example:
        mp_centered = center_on_pelvis(mp_pose)
        gt_centered = center_on_pelvis(gt_pose)
        mp_aligned = align_body_frames(mp_centered, gt_centered)
    """
    # Get body frames for both poses
    source_frame = get_body_frame(source_pose)  # (3, 3)
    target_frame = get_body_frame(target_pose)  # (3, 3)

    # Compute rotation from source frame to target frame
    # R @ source_frame = target_frame
    # R = target_frame @ source_frame.T
    R = target_frame @ source_frame.T

    # Apply rotation to all joints (pose is centered, so just rotate)
    aligned_pose = (R @ source_pose.T).T

    return aligned_pose


def compute_body_frame_error(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """
    Compute rotation angle between two poses' body frames.

    Useful for verifying alignment worked correctly.

    Args:
        pose1: (17, 3) first pose
        pose2: (17, 3) second pose

    Returns:
        Rotation angle in degrees between the two body frames
    """
    frame1 = get_body_frame(pose1)
    frame2 = get_body_frame(pose2)

    # Relative rotation
    R_rel = frame2 @ frame1.T

    # Extract rotation angle from rotation matrix
    # angle = arccos((trace(R) - 1) / 2)
    trace = np.trace(R_rel)
    # Clamp to valid range for arccos
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    return np.degrees(angle_rad)


def compute_limb_orientation_errors(mp_pose: np.ndarray, gt_pose: np.ndarray) -> dict:
    """
    Compute limb orientation errors between MediaPipe and ground truth.

    Useful for diagnosing alignment issues - torso errors should be similar
    to arm/leg errors after proper alignment.

    Args:
        mp_pose: (17, 3) MediaPipe pose (normalized)
        gt_pose: (17, 3) ground truth pose (normalized)

    Returns:
        Dict with per-limb errors in degrees:
        - torso_errors: [L_torso, R_torso, L_cross, R_cross]
        - arm_errors: [L_upper, R_upper, L_fore, R_fore]
        - leg_errors: [L_thigh, R_thigh, L_shin, R_shin]
    """
    # Define limbs: (parent_idx, child_idx, name)
    torso_limbs = [
        (5, 11, 'L_torso'),    # L_shoulder -> L_hip
        (6, 12, 'R_torso'),    # R_shoulder -> R_hip
        (5, 12, 'L_cross'),    # L_shoulder -> R_hip
        (6, 11, 'R_cross'),    # R_shoulder -> L_hip
    ]

    arm_limbs = [
        (5, 7, 'L_upper_arm'),   # L_shoulder -> L_elbow
        (6, 8, 'R_upper_arm'),   # R_shoulder -> R_elbow
        (7, 9, 'L_forearm'),     # L_elbow -> L_wrist
        (8, 10, 'R_forearm'),    # R_elbow -> R_wrist
    ]

    leg_limbs = [
        (11, 13, 'L_thigh'),   # L_hip -> L_knee
        (12, 14, 'R_thigh'),   # R_hip -> R_knee
        (13, 15, 'L_shin'),    # L_knee -> L_ankle
        (14, 16, 'R_shin'),    # R_knee -> R_ankle
    ]

    def compute_limb_error(limbs):
        errors = []
        for p, c, name in limbs:
            mp_vec = mp_pose[c] - mp_pose[p]
            gt_vec = gt_pose[c] - gt_pose[p]

            mp_norm = np.linalg.norm(mp_vec)
            gt_norm = np.linalg.norm(gt_vec)

            if mp_norm < 1e-6 or gt_norm < 1e-6:
                errors.append(0.0)
                continue

            mp_unit = mp_vec / mp_norm
            gt_unit = gt_vec / gt_norm

            dot = np.clip(np.dot(mp_unit, gt_unit), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))
            errors.append(angle)
        return errors

    return {
        'torso_errors': compute_limb_error(torso_limbs),
        'arm_errors': compute_limb_error(arm_limbs),
        'leg_errors': compute_limb_error(leg_limbs),
    }
