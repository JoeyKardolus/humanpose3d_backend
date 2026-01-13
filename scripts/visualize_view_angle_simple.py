#!/usr/bin/env python3
"""
Simple visualization of view angle: camera ray hitting torso plane.
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation


def load_camera_params(cameras_dir: Path, setting_name: str, camera_name: str = 'c01') -> dict:
    setting_file = cameras_dir / f"{setting_name}.json"
    with open(setting_file, 'r') as f:
        cameras = json.load(f)
    for cam in cameras:
        if cam['name'] == camera_name:
            rvec = np.array(cam['rotation'])
            R = Rotation.from_rotvec(rvec).as_matrix()
            t = np.array(cam['translation']) / 100.0
            cam_pos = -R.T @ t
            return {'position': cam_pos}
    raise ValueError(f"Camera {camera_name} not found")


def load_keypoints3d(pkl_path: Path) -> np.ndarray:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['keypoints3d_optim'] / 100.0


def compute_view_angle(pose_3d: np.ndarray, camera_pos: np.ndarray) -> dict:
    """
    Compute view angle in subject's local coordinate frame.

    Returns azimuth in full 0-360° range:
      0° = directly in front of subject
     90° = subject's right side (profile)
    180° = directly behind subject
    270° = subject's left side (profile)

    For depth distortion: profile views (90°, 270°) have most distortion
    on the far side of the body.
    """
    nose = pose_3d[0]
    ls, rs = pose_3d[5], pose_3d[6]
    lh, rh = pose_3d[11], pose_3d[12]

    torso_center = (ls + rs + lh + rh) / 4

    # === BUILD SUBJECT'S LOCAL COORDINATE FRAME ===
    # Right axis: left shoulder -> right shoulder
    right_axis = rs - ls
    right_axis = right_axis / np.linalg.norm(right_axis)

    # Up axis: world Y (stable, not affected by bending)
    up_axis = np.array([0.0, 1.0, 0.0])

    # Forward axis: perpendicular to right and up (where chest faces)
    # Using up × right to get forward (right-hand rule: chest direction)
    forward_axis = np.cross(up_axis, right_axis)
    forward_norm = np.linalg.norm(forward_axis)
    if forward_norm < 1e-6:
        forward_axis = np.array([0.0, 0.0, 1.0])
    else:
        forward_axis = forward_axis / forward_norm

    # Re-orthogonalize right axis
    right_axis = np.cross(forward_axis, up_axis)
    right_axis = right_axis / np.linalg.norm(right_axis)

    # === CAMERA DIRECTION IN SUBJECT'S FRAME ===
    # Vector from subject TO camera (so 0° = camera in front)
    subj_to_cam = camera_pos - torso_center
    cam_dist = np.linalg.norm(subj_to_cam)

    # Project onto subject's horizontal plane
    forward_component = np.dot(subj_to_cam, forward_axis)
    right_component = np.dot(subj_to_cam, right_axis)
    up_component = np.dot(subj_to_cam, up_axis)

    # === AZIMUTH: full 0-360° ===
    # atan2 gives -180 to +180, convert to 0-360
    azimuth = np.degrees(np.arctan2(right_component, forward_component))
    if azimuth < 0:
        azimuth += 360.0

    # === FRONT/BACK (for display purposes) ===
    # Front = 0-90° or 270-360°, Back = 90-180° or 180-270°
    viewing_front = (azimuth <= 90) or (azimuth >= 270)

    # === ELEVATION ===
    horiz_dist = np.sqrt(forward_component**2 + right_component**2)
    if horiz_dist < 1e-6:
        elevation = 90.0 if up_component > 0 else -90.0
    else:
        elevation = np.degrees(np.arctan2(up_component, horiz_dist))

    return {
        'azimuth': azimuth,
        'elevation': elevation,
        'viewing_front': viewing_front,
        'torso_center': torso_center,
        'forward_axis': forward_axis,
        'right_axis': right_axis,
        'camera_pos': camera_pos,
        'cam_dist': cam_dist,
        'corners': [ls, rs, rh, lh],
        'nose': nose,
    }


def visualize(pose_3d: np.ndarray, camera_pos: np.ndarray, title: str = ""):
    """Clean visualization of camera ray hitting torso plane."""
    va = compute_view_angle(pose_3d, camera_pos)

    fig = plt.figure(figsize=(16, 8))

    tc = va['torso_center']
    fwd = va['forward_axis']
    right = va['right_axis']
    cam = va['camera_pos']
    nose = va['nose']
    corners = va['corners']

    # === Panel 1: Top-down view (X-Z plane) ===
    ax1 = fig.add_subplot(2, 3, 1)

    # Torso plane
    corners_xz = [(c[0], c[2]) for c in corners + [corners[0]]]
    xs, zs = zip(*corners_xz)
    ax1.fill(xs[:-1], zs[:-1], alpha=0.3, color='cyan', label='Torso')
    ax1.plot(xs, zs, 'b-', linewidth=2)

    # Center and nose
    ax1.scatter([tc[0]], [tc[2]], c='green', s=100, marker='*', zorder=10)
    ax1.scatter([nose[0]], [nose[2]], c='orange', s=80, zorder=10, label='Nose')

    # Forward arrow (where subject faces)
    arrow_scale = 0.5
    ax1.annotate('', xy=(tc[0] + fwd[0]*arrow_scale, tc[2] + fwd[2]*arrow_scale),
                 xytext=(tc[0], tc[2]),
                 arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax1.text(tc[0] + fwd[0]*arrow_scale*1.2, tc[2] + fwd[2]*arrow_scale*1.2,
             'FWD', fontsize=8, color='green')

    # Right arrow
    ax1.annotate('', xy=(tc[0] + right[0]*arrow_scale*0.5, tc[2] + right[2]*arrow_scale*0.5),
                 xytext=(tc[0], tc[2]),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # Camera
    ax1.scatter([cam[0]], [cam[2]], c='red', s=150, marker='^', zorder=15, label='Camera')

    # Camera ray
    ax1.plot([cam[0], tc[0]], [cam[2], tc[2]], 'r-', linewidth=2)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('Top-Down View')

    # === Panel 2: Side view (Z-Y plane) for elevation ===
    ax2 = fig.add_subplot(2, 3, 2)

    # Torso as vertical line (shoulders to hips)
    torso_top_y = max(c[1] for c in corners)
    torso_bot_y = min(c[1] for c in corners)
    ax2.plot([tc[2], tc[2]], [torso_bot_y, torso_top_y], 'b-', linewidth=8,
             solid_capstyle='round', label='Torso')
    ax2.scatter([tc[2]], [tc[1]], c='green', s=100, marker='*', zorder=10)

    # Camera
    ax2.scatter([cam[2]], [cam[1]], c='red', s=150, marker='^', zorder=15, label='Camera')

    # Camera ray
    ax2.plot([cam[2], tc[2]], [cam[1], tc[1]], 'r-', linewidth=2)

    # Ground line
    ax2.axhline(y=0, color='brown', linestyle='--', alpha=0.5, label='Ground')

    ax2.set_xlabel('Z (depth)')
    ax2.set_ylabel('Y (height)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_title('Side View (Elevation)')

    # === Panel 3: Azimuth compass diagram (0-360°) ===
    ax3 = fig.add_subplot(2, 3, 4, projection='polar')

    # Draw subject as circle at center
    ax3.scatter([0], [0], c='blue', s=200, marker='o', zorder=10)

    # Mark cardinal directions
    ax3.annotate('0°\nFRONT', xy=(0, 1.15), ha='center', fontsize=8, color='green')
    ax3.annotate('90°\nRIGHT', xy=(np.pi/2, 1.15), ha='center', fontsize=8)
    ax3.annotate('180°\nBACK', xy=(np.pi, 1.15), ha='center', fontsize=8, color='red')
    ax3.annotate('270°\nLEFT', xy=(3*np.pi/2, 1.15), ha='center', fontsize=8)

    # Camera position on compass
    az_rad = np.radians(va['azimuth'])
    ax3.scatter([az_rad], [0.8], c='red', s=150, marker='^', zorder=15)
    ax3.plot([0, az_rad], [0, 0.8], 'r-', linewidth=2)

    # Angle arc
    arc_angles = np.linspace(0, az_rad, 50)
    ax3.plot(arc_angles, [0.4]*len(arc_angles), 'purple', linewidth=2)

    ax3.set_ylim(0, 1.3)
    ax3.set_yticks([])
    ax3.set_theta_zero_location('N')  # 0° at top
    ax3.set_theta_direction(-1)  # Clockwise
    ax3.set_title(f'Azimuth: {va["azimuth"]:.0f}°', pad=20)

    # === Panel 4: Elevation diagram ===
    ax4 = fig.add_subplot(2, 3, 5)

    # Ground line
    ax4.plot([-0.8, 0.8], [0, 0], 'brown', linewidth=3, linestyle='--')
    ax4.text(0, -0.15, 'GROUND', ha='center', fontsize=9, color='brown')

    # Subject as vertical line
    ax4.plot([0, 0], [0, 0.6], 'b-', linewidth=8, solid_capstyle='round')
    ax4.scatter([0], [0.3], c='green', s=80, marker='*', zorder=10)
    ax4.text(0.08, 0.3, 'Subject', fontsize=9, color='blue')

    # Camera ray at elevation angle
    elev_rad = np.radians(va['elevation'])
    ray_len = 0.7
    ray_x = -ray_len * np.cos(elev_rad)
    ray_y = 0.3 + ray_len * np.sin(elev_rad)

    ax4.annotate('', xy=(0, 0.3), xytext=(ray_x, ray_y),
                 arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax4.scatter([ray_x], [ray_y], c='red', s=100, marker='^', zorder=10)
    ax4.text(ray_x - 0.08, ray_y + 0.05, 'CAM', fontsize=9, color='red', ha='right')

    # Horizontal reference
    ax4.plot([-0.6, 0], [0.3, 0.3], 'gray', linewidth=1, linestyle=':')

    # Angle arc
    if abs(va['elevation']) > 1:
        arc_start = min(0, elev_rad)
        arc_end = max(0, elev_rad)
        arc_angles = np.linspace(arc_start, arc_end, 20)
        arc_r = 0.25
        ax4.plot(-arc_r * np.cos(arc_angles), 0.3 + arc_r * np.sin(arc_angles),
                 'orange', linewidth=2)

    ax4.text(-0.5, 0.5, f'{va["elevation"]:+.1f}°',
             fontsize=14, color='orange', fontweight='bold')

    ax4.set_xlim(-1, 0.6)
    ax4.set_ylim(-0.25, 1.0)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('Elevation (+ve=above, -ve=below)')

    # === Panel 5: Results ===
    ax5 = fig.add_subplot(2, 3, (3, 6))
    ax5.axis('off')

    front_back = "FRONT" if va['viewing_front'] else "BACK"
    az_color = 'green' if va['viewing_front'] else 'red'
    elev_color = 'orange'

    cam = va['camera_pos']
    tc = va['torso_center']
    result_text = f"""VIEW ANGLE RESULTS
══════════════════════

Azimuth:   {va['azimuth']:.1f}°
Elevation: {va['elevation']:+.1f}°
Side:      {front_back}

─────────────────────
AZIMUTH GUIDE
─────────────────────
  0° = front
 90° = right profile
180° = back
270° = left profile

─────────────────────
DEBUG INFO
─────────────────────
Camera pos:
  X={cam[0]:+.2f} Y={cam[1]:+.2f} Z={cam[2]:+.2f}

Torso center:
  X={tc[0]:+.2f} Y={tc[1]:+.2f} Z={tc[2]:+.2f}

Distance: {va['cam_dist']:.2f}m
"""
    ax5.text(0.05, 0.95, result_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Big angle displays
    ax5.text(0.3, 0.18, f'Az: {va["azimuth"]:+.0f}°', transform=ax5.transAxes,
             fontsize=32, ha='center', fontweight='bold', color=az_color)
    ax5.text(0.7, 0.18, f'El: {va["elevation"]:+.0f}°', transform=ax5.transAxes,
             fontsize=32, ha='center', fontweight='bold', color=elev_color)
    ax5.text(0.5, 0.02, front_back, transform=ax5.transAxes,
             fontsize=18, ha='center', fontweight='bold', color=az_color)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig, va


def main():
    print("=" * 50)
    print("VIEW ANGLE VISUALIZATION")
    print("=" * 50)

    aist_dir = Path("data/AIST++")
    keypoints_dir = aist_dir / "annotations" / "keypoints3d"
    cameras_dir = aist_dir / "annotations" / "cameras"
    videos_dir = aist_dir / "videos"

    video_files = sorted(videos_dir.glob("*.mp4"))[:3]

    for video_path in video_files:
        seq_name = video_path.stem
        parts = seq_name.split('_')

        cam_name = None
        for i, p in enumerate(parts):
            if p.startswith('c') and p[1:].isdigit():
                cam_name = p
                parts[i] = 'cAll'
                break
        if not cam_name:
            continue

        kp_path = keypoints_dir / f"{'_'.join(parts)}.pkl"
        if not kp_path.exists():
            continue

        try:
            cam = load_camera_params(cameras_dir, "setting1", cam_name)
            kp3d = load_keypoints3d(kp_path)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(f"\n{seq_name}")
        
        for fidx in [0, len(kp3d)//2]:
            fig, va = visualize(kp3d[fidx], cam['position'], 
                               f"{seq_name} - Frame {fidx}")
            
            out = f"view_angle_{seq_name}_f{fidx:04d}.png"
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            fb = "FRONT" if va['viewing_front'] else "BACK"
            print(f"  Frame {fidx}: Az={va['azimuth']:+.1f}° El={va['elevation']:+.1f}° ({fb}) -> {out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
