#!/usr/bin/env python3
"""Interactive visualization of POF reconstruction without TRC/augmentation pipeline.

Uses RTMPose for 2D detection (same as POF model training data).
"""

from pathlib import Path
import sys
import os
import argparse

# Set matplotlib backend before importing pyplot
import matplotlib
# Try TkAgg first, fall back to Agg for headless
if os.environ.get('MPLBACKEND') != 'Agg':
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mediastream.media_stream import read_video_rgb
from src.posedetector.rtmpose_detector import RTMPoseDetector
from src.pof.inference import CameraPOFInference
from src.pof.reconstruction import reconstruct_skeleton_least_squares

# COCO-17 skeleton
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

COCO_NAMES = [
    'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
    'L_shldr', 'R_shldr', 'L_elbow', 'R_elbow',
    'L_wrist', 'R_wrist', 'L_hip', 'R_hip',
    'L_knee', 'R_knee', 'L_ankle', 'R_ankle'
]


def get_pose_2d_and_visibility(frames, fps):
    """Extract 2D poses using RTMPose (same as POF training data)."""
    detector = RTMPoseDetector(model_size="m", device="cpu")
    result = detector.detect(frames, fps, visibility_min=0.1)
    return result.keypoints_2d, result.visibility


def visualize_pof_interactive(video_path: Path, height_m: float = 1.78, max_frames: int = 200):
    """Run POF reconstruction and visualize interactively."""
    print(f"Loading video: {video_path}")
    frames, fps = read_video_rgb(video_path)

    if len(frames) > max_frames:
        print(f"Using first {max_frames} frames")
        frames = frames[:max_frames]

    print(f"Extracting 2D poses from {len(frames)} frames...")
    pose_2d, visibility = get_pose_2d_and_visibility(frames, fps)

    print("Loading POF model...")
    pof = CameraPOFInference("models/checkpoints/best_pof_semgcn-temporal_model.pth")

    print("Running POF reconstruction...")
    # Use same code path as training data test (works correctly)
    pof_pred = pof.predict_pof(pose_2d, visibility)
    poses_3d = reconstruct_skeleton_least_squares(
        pof_pred, pose_2d, None,
        pelvis_depth=0.0, denormalize=False, enforce_width=True
    )

    print(f"Reconstructed {len(poses_3d)} frames")
    print(f"Coordinate ranges:")
    print(f"  X: {poses_3d[:, :, 0].min():.3f} to {poses_3d[:, :, 0].max():.3f}")
    print(f"  Y: {poses_3d[:, :, 1].min():.3f} to {poses_3d[:, :, 1].max():.3f}")
    print(f"  Z: {poses_3d[:, :, 2].min():.3f} to {poses_3d[:, :, 2].max():.3f}")

    # Interactive visualization
    show_interactive(poses_3d, visibility, video_path.stem)


def show_interactive(poses_3d, visibility, title="POF"):
    """Interactive 3D visualization with slider and play button."""
    n_frames = len(poses_3d)

    # Center on pelvis for each frame (skeleton-centric view)
    pelvis = (poses_3d[:, 11] + poses_3d[:, 12]) / 2  # L/R hip midpoint
    centered_poses = poses_3d - pelvis[:, np.newaxis, :]

    # Flip Y for display (camera Y-down -> display Y-up)
    display_poses = centered_poses.copy()
    display_poses[:, :, 1] = -centered_poses[:, :, 1]

    # Compute tight bounds around skeleton (not including translation)
    pad = 0.2
    x_range = display_poses[:, :, 0].max() - display_poses[:, :, 0].min()
    y_range = display_poses[:, :, 1].max() - display_poses[:, :, 1].min()
    z_range = display_poses[:, :, 2].max() - display_poses[:, :, 2].min()
    max_range = max(x_range, y_range, z_range) / 2 + pad

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2)

    # View from front-ish angle
    ax.view_init(elev=10, azim=-80)

    # Animation state
    state = {'playing': False, 'frame_idx': 0}

    def update_plot(frame_idx):
        ax.clear()
        pose = display_poses[frame_idx]
        vis = visibility[frame_idx]

        # Draw skeleton bones first (behind joints)
        for i, j in COCO_CONNECTIONS:
            if vis[i] > 0.3 and vis[j] > 0.3:
                ax.plot([pose[i, 0], pose[j, 0]],
                       [pose[i, 2], pose[j, 2]],
                       [pose[i, 1], pose[j, 1]],
                       'b-', linewidth=3, alpha=0.8)

        # Plot joints with visibility coloring
        for i in range(17):
            if vis[i] > 0.5:
                color = 'lime'
                size = 80
            elif vis[i] > 0.3:
                color = 'orange'
                size = 60
            else:
                color = 'red'
                size = 40
            ax.scatter(pose[i, 0], pose[i, 2], pose[i, 1],
                      c=color, s=size, edgecolor='black', linewidth=0.5)

        # Label all joints
        for i, name in enumerate(COCO_NAMES):
            if vis[i] > 0.2:
                ax.text(pose[i, 0] + 0.02, pose[i, 2], pose[i, 1] + 0.02,
                       name, fontsize=8, alpha=0.7)

        ax.set_xlabel('X (right)', fontsize=10)
        ax.set_ylabel('Z (depth)', fontsize=10)
        ax.set_zlabel('Y (up)', fontsize=10)
        ax.set_title(f'{title} - Frame {frame_idx + 1}/{n_frames}', fontsize=12)

        # Set equal aspect ratio, centered on skeleton
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        fig.canvas.draw_idle()

    # Frame slider
    ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)

    def on_slider(val):
        state['frame_idx'] = int(val)
        update_plot(state['frame_idx'])

    slider.on_changed(on_slider)

    # Play/Pause button
    ax_button = plt.axes([0.4, 0.02, 0.2, 0.04])
    button = Button(ax_button, 'Play')

    def on_button(event):
        state['playing'] = not state['playing']
        button.label.set_text('Pause' if state['playing'] else 'Play')

    button.on_clicked(on_button)

    # Animation timer
    def animate(event):
        if state['playing']:
            state['frame_idx'] = (state['frame_idx'] + 1) % n_frames
            slider.set_val(state['frame_idx'])

    timer = fig.canvas.new_timer(interval=40)  # ~25 fps
    timer.add_callback(animate, None)
    timer.start()

    # Initial plot
    update_plot(0)

    plt.show()


def save_frames(poses_3d, visibility, output_dir, title="POF"):
    """Save frames as images with multiple views."""
    os.makedirs(output_dir, exist_ok=True)
    n_frames = len(poses_3d)

    # Center on pelvis
    pelvis = (poses_3d[:, 11] + poses_3d[:, 12]) / 2
    centered = poses_3d - pelvis[:, np.newaxis, :]
    # Flip Y for display
    display_poses = centered.copy()
    display_poses[:, :, 1] = -centered[:, :, 1]

    # Compute bounds
    pad = 0.2
    max_range = max(
        display_poses[:, :, 0].max() - display_poses[:, :, 0].min(),
        display_poses[:, :, 1].max() - display_poses[:, :, 1].min(),
        display_poses[:, :, 2].max() - display_poses[:, :, 2].min()
    ) / 2 + pad

    # Joint names for labeling
    JOINT_NAMES = ['nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
                   'L_sh', 'R_sh', 'L_el', 'R_el', 'L_wr', 'R_wr',
                   'L_hip', 'R_hip', 'L_kn', 'R_kn', 'L_an', 'R_an']

    # Save every 5th frame with 3 views
    for frame_idx in range(0, n_frames, 5):
        fig = plt.figure(figsize=(18, 6))
        pose = display_poses[frame_idx]
        vis = visibility[frame_idx]

        views = [
            (1, 10, -80, 'Front-ish view'),
            (2, 90, -90, 'Top-down view (Y up)'),
            (3, 0, 0, 'Side view'),
        ]

        for subplot_idx, elev, azim, view_title in views:
            ax = fig.add_subplot(1, 3, subplot_idx, projection='3d')

            # Draw bones with L=blue, R=red
            for i, j in COCO_CONNECTIONS:
                if vis[i] > 0.3 and vis[j] > 0.3:
                    # Color by side: L=blue, R=red, center=green
                    if 'L_' in JOINT_NAMES[i] or 'L_' in JOINT_NAMES[j]:
                        color = 'blue'
                    elif 'R_' in JOINT_NAMES[i] or 'R_' in JOINT_NAMES[j]:
                        color = 'red'
                    else:
                        color = 'green'
                    ax.plot([pose[i, 0], pose[j, 0]],
                           [pose[i, 2], pose[j, 2]],
                           [pose[i, 1], pose[j, 1]],
                           color=color, linewidth=3, alpha=0.8)

            # Draw joints with L=blue, R=red
            for i in range(17):
                if vis[i] > 0.3:
                    if 'L_' in JOINT_NAMES[i]:
                        color = 'blue'
                    elif 'R_' in JOINT_NAMES[i]:
                        color = 'red'
                    else:
                        color = 'green'
                    ax.scatter(pose[i, 0], pose[i, 2], pose[i, 1], c=color, s=60, edgecolor='black')

            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
            ax.set_xlabel('X (right)')
            ax.set_ylabel('Z (depth)')
            ax.set_zlabel('Y (up)')
            ax.set_title(view_title)
            ax.view_init(elev=elev, azim=azim)

        fig.suptitle(f'{title} - Frame {frame_idx + 1}/{n_frames} (Blue=Left, Red=Right)', fontsize=14)
        plt.tight_layout()

        out_path = f'{output_dir}/{title}_frame{frame_idx:04d}.png'
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize POF reconstruction')
    parser.add_argument('video', nargs='?', default='data/input/PushUp.mp4', help='Input video path')
    parser.add_argument('height', nargs='?', type=float, default=1.78, help='Subject height in meters')
    parser.add_argument('max_frames', nargs='?', type=int, default=200, help='Max frames to process')
    parser.add_argument('--save', action='store_true', help='Save frames as images instead of interactive display')
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"Video not found: {video}")
        sys.exit(1)

    if args.save:
        # Save mode
        print(f"Loading video: {video}")
        from src.mediastream.media_stream import read_video_rgb
        from src.posedetector.rtmpose_detector import RTMPoseDetector

        frames, fps = read_video_rgb(video)
        if len(frames) > args.max_frames:
            print(f"Using first {args.max_frames} frames")
            frames = frames[:args.max_frames]

        print(f"Extracting 2D poses from {len(frames)} frames...")
        detector = RTMPoseDetector(model_size="m", device="cpu")
        result = detector.detect(frames, fps, visibility_min=0.1)
        pose_2d, visibility = result.keypoints_2d, result.visibility

        print("Loading POF model...")
        pof = CameraPOFInference("models/checkpoints/best_pof_semgcn-temporal_model.pth")

        print("Running POF reconstruction...")
        pof_pred = pof.predict_pof(pose_2d, visibility)
        poses_3d = reconstruct_skeleton_least_squares(
            pof_pred, pose_2d, None,
            pelvis_depth=0.0, denormalize=False, enforce_width=False
        )

        output_dir = f'data/output/pof_viz_{video.stem}'
        print(f"Saving frames to {output_dir}/...")
        save_frames(poses_3d, visibility, output_dir, video.stem)
        print("Done!")
    else:
        visualize_pof_interactive(video, args.height, args.max_frames)


if __name__ == "__main__":
    main()
