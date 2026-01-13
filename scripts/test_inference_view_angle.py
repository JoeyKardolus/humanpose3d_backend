#!/usr/bin/env python3
"""
Test view angle computation at INFERENCE time (no camera calibration).

Key insight: View angle = subject's orientation relative to camera.
- At training: We know camera position, compute angle from that
- At inference: MediaPipe pose is in camera-relative space, so we can
  compute the same angle from the pose itself

MediaPipe coordinate system:
- X: right (subject's left)
- Y: down
- Z: forward (away from camera, into the scene)
- Camera is at origin looking down +Z

So if subject faces camera: their forward (from hips) points toward -Z
If subject is sideways: their forward points along X
"""

import numpy as np


def compute_view_angle_inference(pose_3d: np.ndarray) -> float:
    """
    Compute view angle from MediaPipe pose at inference time.

    Uses the same hip-based approach as training, but assumes
    camera is at origin looking down +Z (MediaPipe convention).

    Args:
        pose_3d: (17, 3) COCO keypoints from MediaPipe (in MediaPipe coords)

    Returns:
        View angle in degrees (0=frontal, 90=profile)
    """
    # COCO indices: left_hip=11, right_hip=12
    left_hip = pose_3d[11]
    right_hip = pose_3d[12]

    # Subject's lateral axis (hip to hip)
    # In MediaPipe: X is right, so right_hip - left_hip points in +X direction
    lateral = right_hip - left_hip
    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm < 1e-6:
        return 45.0
    lateral = lateral / lateral_norm

    # Up axis: In MediaPipe Y is down, so up is -Y
    # But for computing forward, we want world up
    # After Y-flip (which we do for training), Y is up
    # For now, assume we're working in Y-up space
    up = np.array([0, 1, 0])

    # Forward axis: perpendicular to lateral and up
    forward = np.cross(lateral, up)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return 45.0
    forward = forward / forward_norm

    # Camera direction: In MediaPipe (after Y-flip), camera looks down +Z
    # Actually, MediaPipe has camera at origin looking at subject
    # Subject is at positive Z, so camera looks toward +Z
    # For the view angle, we want angle between subject's forward and camera's view direction
    # Camera looks at subject, so view direction is roughly [0, 0, 1] (toward positive Z)
    # But we want to know: from camera's POV, what angle is the subject at?
    # If subject faces camera, their forward points toward camera = toward -Z
    camera_dir = np.array([0, 0, -1])  # Camera looks from origin toward subject

    # Project both onto horizontal plane
    forward_horiz = forward.copy()
    forward_horiz[1] = 0
    forward_norm = np.linalg.norm(forward_horiz)
    if forward_norm < 1e-6:
        return 45.0
    forward_horiz = forward_horiz / forward_norm

    # Angle between subject's forward and camera direction
    # 0° = subject faces camera (forward ≈ -Z)
    # 90° = subject is sideways (forward ≈ ±X)
    cos_angle = np.dot(forward_horiz, camera_dir)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    # We want 0-90 range (don't care if facing toward or away)
    if angle > 90:
        angle = 180 - angle

    return angle


def test_synthetic_poses():
    """Test with synthetic poses at known orientations."""

    print("=" * 60)
    print("INFERENCE VIEW ANGLE TEST")
    print("=" * 60)
    print()
    print("Testing with synthetic poses at known orientations...")
    print("(Simulating MediaPipe output in Y-up coordinates)")
    print()

    # Create a simple standing pose
    # COCO 17: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
    def make_pose(facing_angle_deg):
        """Create pose facing given angle from camera (0=frontal, 90=profile)."""
        # Base pose: standing, facing -Z (toward camera)
        pose = np.zeros((17, 3))

        # Vertical positions (Y-up)
        pose[0, 1] = 1.7   # nose
        pose[5, 1] = 1.4   # left shoulder
        pose[6, 1] = 1.4   # right shoulder
        pose[11, 1] = 1.0  # left hip
        pose[12, 1] = 1.0  # right hip
        pose[15, 1] = 0.0  # left ankle
        pose[16, 1] = 0.0  # right ankle

        # Horizontal spread (base: facing camera)
        shoulder_width = 0.4
        hip_width = 0.3

        # Rotate around Y axis by facing_angle
        angle_rad = np.radians(facing_angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Shoulders: spread along X when frontal, along Z when profile
        pose[5, 0] = -shoulder_width/2 * cos_a  # left shoulder X
        pose[5, 2] = -shoulder_width/2 * sin_a + 2.0  # left shoulder Z (2m from camera)
        pose[6, 0] = shoulder_width/2 * cos_a   # right shoulder X
        pose[6, 2] = shoulder_width/2 * sin_a + 2.0   # right shoulder Z

        # Hips: same rotation
        pose[11, 0] = -hip_width/2 * cos_a
        pose[11, 2] = -hip_width/2 * sin_a + 2.0
        pose[12, 0] = hip_width/2 * cos_a
        pose[12, 2] = hip_width/2 * sin_a + 2.0

        # Center on pelvis for consistency
        pelvis = (pose[11] + pose[12]) / 2
        pose = pose - pelvis

        return pose

    # Test at various angles
    test_angles = [0, 15, 30, 45, 60, 75, 90]

    print(f"{'True Angle':>12} | {'Computed':>10} | {'Error':>8}")
    print("-" * 40)

    max_error = 0
    for true_angle in test_angles:
        pose = make_pose(true_angle)
        computed = compute_view_angle_inference(pose)
        error = abs(computed - true_angle)
        max_error = max(max_error, error)

        status = "OK" if error < 5 else "WARN"
        print(f"{true_angle:>12}° | {computed:>9.1f}° | {error:>7.1f}° {status}")

    print("-" * 40)
    print(f"Max error: {max_error:.1f}°")
    print()

    if max_error < 5:
        print("SUCCESS: View angle computation works at inference time!")
    else:
        print("WARNING: Large errors - need to check coordinate conventions")

    print()
    print("=" * 60)


def test_with_bending():
    """Test that bending doesn't affect view angle (same as training)."""

    print()
    print("BENDING ROBUSTNESS TEST")
    print("=" * 60)
    print()

    def make_pose_with_bend(facing_angle_deg, bend_angle_deg):
        """Create pose with upper body bent forward."""
        from scipy.spatial.transform import Rotation

        pose = np.zeros((17, 3))

        # Base positions
        shoulder_width = 0.4
        hip_width = 0.3

        # Rotate around Y axis by facing_angle
        angle_rad = np.radians(facing_angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Hips (not affected by bend)
        pose[11] = np.array([-hip_width/2 * cos_a, 1.0, -hip_width/2 * sin_a + 2.0])
        pose[12] = np.array([hip_width/2 * cos_a, 1.0, hip_width/2 * sin_a + 2.0])

        # Shoulders (will be rotated by bend)
        pose[5] = np.array([-shoulder_width/2 * cos_a, 1.4, -shoulder_width/2 * sin_a + 2.0])
        pose[6] = np.array([shoulder_width/2 * cos_a, 1.4, shoulder_width/2 * sin_a + 2.0])
        pose[0] = np.array([0, 1.7, 2.0])  # nose

        # Apply bend (rotate upper body around hip lateral axis)
        if bend_angle_deg != 0:
            pelvis = (pose[11] + pose[12]) / 2
            hip_lateral = pose[12] - pose[11]
            hip_lateral = hip_lateral / np.linalg.norm(hip_lateral)

            R = Rotation.from_rotvec(hip_lateral * np.radians(bend_angle_deg)).as_matrix()

            for idx in [0, 5, 6]:  # nose, shoulders
                rel = pose[idx] - pelvis
                pose[idx] = pelvis + R @ rel

        # Center on pelvis
        pelvis = (pose[11] + pose[12]) / 2
        pose = pose - pelvis

        return pose

    # Test frontal view with different bends
    print("Frontal view (0°) with bending:")
    print(f"{'Bend':>8} | {'View Angle':>12} | {'Expected':>10}")
    print("-" * 40)

    for bend in [0, 30, 45, 60, 90]:
        pose = make_pose_with_bend(0, bend)
        computed = compute_view_angle_inference(pose)
        print(f"{bend:>7}° | {computed:>11.1f}° | {'~0°':>10}")

    print()
    print("45° side view with bending:")
    print(f"{'Bend':>8} | {'View Angle':>12} | {'Expected':>10}")
    print("-" * 40)

    for bend in [0, 30, 45, 60, 90]:
        pose = make_pose_with_bend(45, bend)
        computed = compute_view_angle_inference(pose)
        print(f"{bend:>7}° | {computed:>11.1f}° | {'~45°':>10}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    test_synthetic_poses()
    test_with_bending()
