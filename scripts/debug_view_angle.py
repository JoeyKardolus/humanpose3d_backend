#!/usr/bin/env python3
"""
Debug script to visualize view angle computation problem.

Shows:
1. Ground truth skeleton
2. Camera position (from AIST++ calibration)
3. Current (wrong) approach: torso normal as camera direction
4. Correct approach: actual camera-to-subject vector
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# COCO skeleton connections for visualization
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]


def load_camera_params(cameras_dir: Path, setting_name: str, camera_name: str = 'c01'):
    """Load camera extrinsics from AIST++ calibration."""
    setting_file = cameras_dir / f"{setting_name}.json"
    with open(setting_file, 'r') as f:
        cameras = json.load(f)

    for cam in cameras:
        if cam['name'] == camera_name:
            # Rodrigues to rotation matrix
            rvec = np.array(cam['rotation'])
            R = Rotation.from_rotvec(rvec).as_matrix()

            # Translation (in cm -> meters)
            t = np.array(cam['translation']) / 100.0

            # Camera position in world coords: C = -R^T @ t
            cam_pos = -R.T @ t

            return {
                'R': R,
                't': t,
                'position': cam_pos,
                'name': camera_name,
            }

    raise ValueError(f"Camera {camera_name} not found in {setting_file}")


def get_camera_setting(seq_name: str, mapping_file: Path) -> str:
    """Get camera setting name for sequence."""
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == seq_name:
                return parts[1]
    return 'setting1'  # Default


def compute_view_angle_OLD(pose_3d: np.ndarray) -> tuple:
    """OLD approach: torso normal as camera direction (WRONG)."""
    left_shoulder = pose_3d[5]
    right_shoulder = pose_3d[6]
    left_hip = pose_3d[11]
    right_hip = pose_3d[12]

    shoulder_vec = right_shoulder - left_shoulder
    hip_vec = right_hip - left_hip

    torso_normal = np.cross(shoulder_vec, hip_vec)
    norm = np.linalg.norm(torso_normal)
    if norm < 1e-6:
        return 45.0, np.array([0, 0, 1])
    torso_normal = torso_normal / norm

    camera_dir = np.array([0, 0, 1])
    cos_angle = np.dot(torso_normal, camera_dir)
    angle = np.degrees(np.arccos(np.clip(np.abs(cos_angle), -1, 1)))

    return angle, torso_normal


def compute_view_angle_NEW(pose_3d: np.ndarray, camera_pos: np.ndarray) -> tuple:
    """
    NEW approach: compute azimuth angle of camera relative to subject.

    Uses pelvis orientation to define subject's local frame:
    - X-axis: hip-to-hip (lateral)
    - Y-axis: up (vertical)
    - Z-axis: forward (cross product)

    Then computes camera azimuth in this frame.
    """
    # Subject center (pelvis)
    left_hip = pose_3d[11]
    right_hip = pose_3d[12]
    pelvis = (left_hip + right_hip) / 2

    # Vector from subject to camera
    cam_to_subj = camera_pos - pelvis

    # Subject's local coordinate frame from pelvis
    # Lateral axis: hip to hip
    lateral = right_hip - left_hip
    lateral = lateral / np.linalg.norm(lateral)

    # Up axis: world Y (we don't want this affected by bending)
    up = np.array([0, 1, 0])

    # Forward axis: perpendicular to lateral and up
    forward = np.cross(lateral, up)
    forward = forward / np.linalg.norm(forward)

    # Re-orthogonalize lateral
    lateral = np.cross(up, forward)
    lateral = lateral / np.linalg.norm(lateral)

    # Project camera direction onto horizontal plane (XZ in subject frame)
    cam_horiz = cam_to_subj.copy()
    cam_horiz[1] = 0  # Remove vertical component
    cam_horiz = cam_horiz / (np.linalg.norm(cam_horiz) + 1e-6)

    # Azimuth: angle in horizontal plane (0° = front, 90° = side)
    forward_component = np.dot(cam_horiz, forward)
    lateral_component = np.dot(cam_horiz, lateral)

    azimuth = np.degrees(np.arctan2(np.abs(lateral_component), forward_component))
    # Clamp to 0-90 range (we care about front vs side, not left vs right)
    azimuth = np.abs(azimuth)
    if azimuth > 90:
        azimuth = 180 - azimuth

    return azimuth, forward, cam_to_subj


def visualize_comparison(pose_3d: np.ndarray, camera_pos: np.ndarray,
                         old_angle: float, old_normal: np.ndarray,
                         new_angle: float, new_forward: np.ndarray,
                         cam_to_subj: np.ndarray,
                         title: str = "View Angle Comparison"):
    """Create visualization comparing old vs new approach."""

    fig = plt.figure(figsize=(16, 6))

    # Centering
    pelvis = (pose_3d[11] + pose_3d[12]) / 2
    pose_centered = pose_3d - pelvis
    cam_centered = camera_pos - pelvis

    def draw_skeleton(ax, pose, color='blue'):
        ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c=color, s=50)
        for i, j in COCO_CONNECTIONS:
            ax.plot([pose[i, 0], pose[j, 0]],
                   [pose[i, 2], pose[j, 2]],
                   [pose[i, 1], pose[j, 1]], c=color, linewidth=1.5)

    # Plot 1: Old approach (torso normal)
    ax1 = fig.add_subplot(131, projection='3d')
    draw_skeleton(ax1, pose_centered)

    # Draw torso normal (OLD - treating as camera direction)
    torso_center = (pose_centered[5] + pose_centered[6] +
                   pose_centered[11] + pose_centered[12]) / 4
    ax1.quiver(torso_center[0], torso_center[2], torso_center[1],
              old_normal[0], old_normal[2], old_normal[1],
              color='red', linewidth=3, arrow_length_ratio=0.2, length=0.5,
              label='Torso normal (WRONG)')

    # Draw assumed camera direction (Z-axis)
    ax1.quiver(0, 1, 0, 0, -1, 0, color='green', linewidth=2,
              arrow_length_ratio=0.1, label='Assumed camera (Z)')

    ax1.set_title(f"OLD: Torso Normal\nAngle = {old_angle:.1f}°")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_zlabel('Y')
    ax1.legend()

    # Plot 2: New approach (actual camera position)
    ax2 = fig.add_subplot(132, projection='3d')
    draw_skeleton(ax2, pose_centered)

    # Draw actual camera position
    ax2.scatter([cam_centered[0]], [cam_centered[2]], [cam_centered[1]],
               c='orange', s=200, marker='^', label='Camera')

    # Draw camera-to-subject line
    ax2.plot([cam_centered[0], 0], [cam_centered[2], 0], [cam_centered[1], 0],
            'g--', linewidth=2, label='Camera → Subject')

    # Draw subject's forward direction
    ax2.quiver(0, 0, 0, new_forward[0]*0.5, new_forward[2]*0.5, new_forward[1]*0.5,
              color='blue', linewidth=3, arrow_length_ratio=0.2,
              label='Subject forward')

    ax2.set_title(f"NEW: Camera Position\nAzimuth = {new_angle:.1f}°")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('Y')
    ax2.legend()

    # Plot 3: Top-down view (clearest for azimuth)
    ax3 = fig.add_subplot(133)

    # Skeleton from above
    ax3.scatter(pose_centered[:, 0], pose_centered[:, 2], c='blue', s=30)
    for i, j in COCO_CONNECTIONS:
        ax3.plot([pose_centered[i, 0], pose_centered[j, 0]],
                [pose_centered[i, 2], pose_centered[j, 2]], 'b-', linewidth=1)

    # Camera position
    ax3.scatter([cam_centered[0]], [cam_centered[2]], c='orange', s=200,
               marker='^', label='Camera', zorder=10)

    # Camera-to-subject line
    ax3.plot([cam_centered[0], 0], [cam_centered[2], 0], 'g--',
            linewidth=2, label='View direction')

    # Subject forward direction
    ax3.arrow(0, 0, new_forward[0]*0.3, new_forward[2]*0.3,
             head_width=0.05, head_length=0.03, fc='blue', ec='blue',
             label='Subject forward')

    # Old torso normal
    ax3.arrow(0, 0, old_normal[0]*0.3, old_normal[2]*0.3,
             head_width=0.05, head_length=0.03, fc='red', ec='red',
             label='Torso normal')

    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 5)
    ax3.set_aspect('equal')
    ax3.set_xlabel('X (lateral)')
    ax3.set_ylabel('Z (front-back)')
    ax3.set_title('Top-Down View\n(azimuth is angle from forward to camera)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def simulate_bent_forward(pose_3d: np.ndarray, bend_angle_deg: float = 45) -> np.ndarray:
    """Simulate subject bending forward by rotating upper body."""
    pose = pose_3d.copy()

    # Pelvis center as rotation origin
    pelvis = (pose[11] + pose[12]) / 2

    # Rotate upper body joints around pelvis X-axis (lateral)
    hip_vec = pose[12] - pose[11]
    hip_vec = hip_vec / np.linalg.norm(hip_vec)

    # Rotation matrix around hip axis
    angle_rad = np.radians(bend_angle_deg)
    R = Rotation.from_rotvec(hip_vec * angle_rad).as_matrix()

    # Upper body joints to rotate: head, nose, shoulders, elbows, wrists
    upper_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for idx in upper_joints:
        rel_pos = pose[idx] - pelvis
        pose[idx] = pelvis + R @ rel_pos

    return pose


def main():
    print("=" * 60)
    print("VIEW ANGLE COMPUTATION DEBUG")
    print("=" * 60)

    # Paths
    aistpp_dir = Path("data/AIST++")
    annotations_dir = aistpp_dir / "annotations"
    cameras_dir = annotations_dir / "cameras"
    keypoints_dir = annotations_dir / "keypoints3d"
    mapping_file = cameras_dir / "mapping.txt"

    # Find a sequence
    kp_files = sorted(keypoints_dir.glob("*.pkl"))[:5]

    for kp_file in kp_files:
        seq_name = kp_file.stem
        print(f"\nSequence: {seq_name}")

        # Load GT pose
        with open(kp_file, 'rb') as f:
            data = pickle.load(f)
        keypoints = data['keypoints3d_optim'] / 100.0  # cm -> m

        # Get camera setting
        setting = get_camera_setting(seq_name, mapping_file)
        print(f"  Camera setting: {setting}")

        # Load camera params
        try:
            cam_params = load_camera_params(cameras_dir, setting, 'c01')
        except Exception as e:
            print(f"  Failed to load camera: {e}")
            continue

        print(f"  Camera position: {cam_params['position']}")

        # Test on frame 0
        frame_idx = 0
        pose = keypoints[frame_idx]

        if np.isnan(pose).any():
            continue

        # Test 1: Normal pose
        print(f"\n  NORMAL POSE (frame {frame_idx}):")
        old_angle, old_normal = compute_view_angle_OLD(pose)
        new_angle, new_forward, cam_to_subj = compute_view_angle_NEW(
            pose, cam_params['position']
        )
        print(f"    OLD (torso normal): {old_angle:.1f}°")
        print(f"    NEW (camera pos):   {new_angle:.1f}°")

        fig = visualize_comparison(
            pose, cam_params['position'],
            old_angle, old_normal, new_angle, new_forward, cam_to_subj,
            title=f"{seq_name} - NORMAL POSE"
        )
        fig.savefig("view_angle_debug_normal.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: view_angle_debug_normal.png")

        # Test 2: Bent forward 45°
        print(f"\n  BENT FORWARD 45°:")
        pose_bent = simulate_bent_forward(pose, 45)
        old_angle_bent, old_normal_bent = compute_view_angle_OLD(pose_bent)
        new_angle_bent, new_forward_bent, cam_to_subj_bent = compute_view_angle_NEW(
            pose_bent, cam_params['position']
        )
        print(f"    OLD (torso normal): {old_angle_bent:.1f}°")
        print(f"    NEW (camera pos):   {new_angle_bent:.1f}°")
        print(f"    Change from normal:")
        print(f"      OLD: {abs(old_angle_bent - old_angle):.1f}°  <- UNSTABLE!")
        print(f"      NEW: {abs(new_angle_bent - new_angle):.1f}°  <- STABLE!")

        fig = visualize_comparison(
            pose_bent, cam_params['position'],
            old_angle_bent, old_normal_bent, new_angle_bent, new_forward_bent, cam_to_subj_bent,
            title=f"{seq_name} - BENT FORWARD 45°"
        )
        fig.savefig("view_angle_debug_bent.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: view_angle_debug_bent.png")

        # Test 3: Bent forward 90° (extreme)
        print(f"\n  BENT FORWARD 90° (extreme):")
        pose_extreme = simulate_bent_forward(pose, 90)
        old_angle_ext, old_normal_ext = compute_view_angle_OLD(pose_extreme)
        new_angle_ext, new_forward_ext, cam_to_subj_ext = compute_view_angle_NEW(
            pose_extreme, cam_params['position']
        )
        print(f"    OLD (torso normal): {old_angle_ext:.1f}°")
        print(f"    NEW (camera pos):   {new_angle_ext:.1f}°")
        print(f"    Change from normal:")
        print(f"      OLD: {abs(old_angle_ext - old_angle):.1f}°  <- COMPLETELY WRONG!")
        print(f"      NEW: {abs(new_angle_ext - new_angle):.1f}°  <- STILL CORRECT!")

        fig = visualize_comparison(
            pose_extreme, cam_params['position'],
            old_angle_ext, old_normal_ext, new_angle_ext, new_forward_ext, cam_to_subj_ext,
            title=f"{seq_name} - BENT FORWARD 90° (extreme)"
        )
        fig.savefig("view_angle_debug_extreme.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: view_angle_debug_extreme.png")

        break  # Just one sequence

    print("\n" + "=" * 60)
    print("Done! Check the generated PNG files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
