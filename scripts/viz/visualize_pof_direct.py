#!/usr/bin/env python3
"""Direct visualization of POF reconstruction without TRC/augmentation pipeline."""

import os
os.environ['MPLBACKEND'] = 'Agg'  # Use non-interactive backend

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mediastream.media_stream import read_video_rgb
from src.posedetector.pose_detector import extract_world_landmarks
from src.pof.inference import CameraPOFInference

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
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def get_pose_2d_and_visibility(frames, fps):
    """Extract 2D poses using MediaPipe."""
    from src.pipeline.refinement import _records_to_coco_arrays, OPENCAP_TO_COCO

    records, landmarks_2d = extract_world_landmarks(
        frames, fps,
        Path('models/pose_landmarker_heavy.task'),
        0.1,
        return_2d_landmarks=True,
    )

    timestamps, pose_3d, visibility, pose_2d, _ = _records_to_coco_arrays(records, landmarks_2d)
    return pose_2d, visibility


def visualize_pof(video_path: Path, height_m: float = 1.78, max_frames: int = 100):
    """Run POF reconstruction and visualize."""
    print(f"Loading video: {video_path}")
    frames, fps = read_video_rgb(video_path)

    if len(frames) > max_frames:
        print(f"Using first {max_frames} frames")
        frames = frames[:max_frames]

    print(f"Extracting 2D poses from {len(frames)} frames...")
    pose_2d, visibility = get_pose_2d_and_visibility(frames, fps)

    print("Loading POF model...")
    pof = CameraPOFInference("models/checkpoints/best_pof_model.pth")

    print("Running POF reconstruction...")
    # Use meter coords for camera-space output
    aspect_ratio = frames.shape[2] / frames.shape[1]
    poses_3d = pof.reconstruct_3d(
        pose_2d, visibility, height_m,
        use_meter_coords=True,
        aspect_ratio=aspect_ratio,
    )

    print(f"Reconstructed {len(poses_3d)} frames")
    print(f"Coordinate ranges:")
    print(f"  X: {poses_3d[:, :, 0].min():.3f} to {poses_3d[:, :, 0].max():.3f}")
    print(f"  Y: {poses_3d[:, :, 1].min():.3f} to {poses_3d[:, :, 1].max():.3f}")
    print(f"  Z: {poses_3d[:, :, 2].min():.3f} to {poses_3d[:, :, 2].max():.3f}")

    # Save visualization
    output_path = Path("data/output") / f"{video_path.stem}_pof_direct.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_3d(poses_3d, visibility, output_path)


def visualize_3d(poses_3d, visibility, output_path: Path):
    """Save multi-frame visualization to PNG."""
    n_frames = len(poses_3d)

    # Flip Y for display (camera Y-down -> display Y-up)
    display_poses = poses_3d.copy()
    display_poses[:, :, 1] = -poses_3d[:, :, 1]

    # Compute bounds
    x_min, x_max = display_poses[:, :, 0].min(), display_poses[:, :, 0].max()
    y_min, y_max = display_poses[:, :, 1].min(), display_poses[:, :, 1].max()
    z_min, z_max = display_poses[:, :, 2].min(), display_poses[:, :, 2].max()

    # Equal aspect ratio
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    # Create multi-panel figure showing several frames
    sample_frames = [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]
    sample_frames = [f for f in sample_frames if f < n_frames]

    fig = plt.figure(figsize=(20, 8))

    for panel_idx, frame_idx in enumerate(sample_frames):
        ax = fig.add_subplot(1, len(sample_frames), panel_idx + 1, projection='3d')
        ax.view_init(elev=15, azim=-70)

        pose = display_poses[frame_idx]
        vis = visibility[frame_idx]

        # Plot joints
        colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in vis]
        ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c=colors, s=50)

        # Label key joints
        for i, name in enumerate(COCO_NAMES):
            if vis[i] > 0.3 and i in [0, 5, 6, 9, 10, 11, 12, 15, 16]:  # Key joints only
                ax.text(pose[i, 0], pose[i, 2], pose[i, 1], name[:3], fontsize=7)

        # Draw skeleton
        for i, j in COCO_CONNECTIONS:
            if vis[i] > 0.3 and vis[j] > 0.3:
                ax.plot([pose[i, 0], pose[j, 0]],
                       [pose[i, 2], pose[j, 2]],
                       [pose[i, 1], pose[j, 1]],
                       'b-', linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f'Frame {frame_idx + 1}')

        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(z_mid - max_range, z_mid + max_range)
        ax.set_zlim(y_mid - max_range, y_mid + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visualization to: {output_path}")


def main():
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/input/PushUp.mp4')
    height = float(sys.argv[2]) if len(sys.argv) > 2 else 1.78
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    if not video.exists():
        print(f"Video not found: {video}")
        sys.exit(1)

    visualize_pof(video, height, max_frames)


if __name__ == "__main__":
    main()
