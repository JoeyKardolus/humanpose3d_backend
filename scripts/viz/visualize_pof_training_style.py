#!/usr/bin/env python3
"""Visualize POF reconstruction in training coordinate system (body-frame aligned).

This matches the coordinate system used during training - NOT meter-converted camera space.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mediastream.media_stream import read_video_rgb
from src.posedetector.pose_detector import extract_world_landmarks
from src.pof.inference import CameraPOFInference
from src.pof.reconstruction import reconstruct_skeleton_batch
from src.pof.bone_lengths import estimate_bone_lengths_array

COCO_CONNECTIONS = [
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

COCO_NAMES = [
    'nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
    'l_sho', 'r_sho', 'l_elb', 'r_elb',
    'l_wri', 'r_wri', 'l_hip', 'r_hip',
    'l_kne', 'r_kne', 'l_ank', 'r_ank'
]


def get_mediapipe_data(frames, fps):
    """Get MediaPipe 2D and visibility."""
    from src.pipeline.refinement import _records_to_coco_arrays

    records, landmarks_2d = extract_world_landmarks(
        frames, fps,
        Path('models/pose_landmarker_heavy.task'),
        0.1,
        return_2d_landmarks=True,
    )

    timestamps, pose_3d, visibility, pose_2d, _ = _records_to_coco_arrays(records, landmarks_2d)
    return pose_2d, visibility


def visualize_pof_training_style(video_path: Path, height_m: float = 1.78, max_frames: int = 50):
    """Visualize POF in training coordinate system."""
    print(f"Loading video: {video_path}")
    frames, fps = read_video_rgb(video_path)

    if len(frames) > max_frames:
        print(f"Using first {max_frames} frames")
        frames = frames[:max_frames]

    print(f"Extracting 2D poses...")
    pose_2d, visibility = get_mediapipe_data(frames, fps)

    print("Loading POF model...")
    pof_model = CameraPOFInference("models/checkpoints/best_pof_model.pth")

    print("Predicting POF vectors...")
    pof_vectors = pof_model.predict_pof(pose_2d, visibility)

    print("Reconstructing skeletons (training-style, NO meter conversion)...")
    bone_lengths = estimate_bone_lengths_array(height_m)

    # Reconstruct WITHOUT meter conversion - use normalized 2D directly
    # This matches training coordinate system
    poses_3d = reconstruct_skeleton_batch(
        pof_vectors,
        bone_lengths,
        pose_2d,  # normalized [0,1] 2D
        pelvis_depth=0.0,  # Center at origin for visualization
        use_meter_coords=False,  # NO meter conversion
    )

    # Center each frame on pelvis for cleaner visualization
    pelvis = (poses_3d[:, 11, :] + poses_3d[:, 12, :]) / 2
    poses_3d_centered = poses_3d - pelvis[:, np.newaxis, :]

    print(f"Reconstructed {len(poses_3d)} frames")
    print(f"Centered coordinate ranges:")
    print(f"  X: {poses_3d_centered[:, :, 0].min():.3f} to {poses_3d_centered[:, :, 0].max():.3f}")
    print(f"  Y: {poses_3d_centered[:, :, 1].min():.3f} to {poses_3d_centered[:, :, 1].max():.3f}")
    print(f"  Z: {poses_3d_centered[:, :, 2].min():.3f} to {poses_3d_centered[:, :, 2].max():.3f}")

    # Visualize
    output_path = Path("data/output") / f"{video_path.stem}_pof_training_style.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visualize_frames(poses_3d_centered, visibility, output_path)


def visualize_frames(poses_3d, visibility, output_path: Path):
    """Create multi-view visualization."""
    n_frames = len(poses_3d)

    # Sample frames
    sample_indices = [0, n_frames//4, n_frames//2, 3*n_frames//4, min(n_frames-1, n_frames)]
    sample_indices = sorted(set([i for i in sample_indices if i < n_frames]))[:5]

    # Compute bounds
    all_coords = poses_3d.reshape(-1, 3)
    valid = ~np.isnan(all_coords).any(axis=1)
    if valid.any():
        coords = all_coords[valid]
        max_range = np.abs(coords).max() * 1.2
    else:
        max_range = 1.0

    fig = plt.figure(figsize=(20, 8))

    for panel_idx, frame_idx in enumerate(sample_indices):
        ax = fig.add_subplot(1, len(sample_indices), panel_idx + 1, projection='3d')

        # View from front-ish angle
        ax.view_init(elev=10, azim=-80)

        pose = poses_3d[frame_idx]
        vis = visibility[frame_idx]

        # Swap Y and Z for visualization (Y-up display)
        # In training coords: X=right, Y=down in image, Z=depth
        # For display: X=right, Y=up, Z=depth
        display_pose = pose.copy()
        display_pose[:, 1] = -pose[:, 1]  # Flip Y

        # Plot joints
        colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in vis]
        ax.scatter(display_pose[:, 0], display_pose[:, 2], display_pose[:, 1],
                   c=colors, s=60, depthshade=True)

        # Draw skeleton
        for i, j in COCO_CONNECTIONS:
            if vis[i] > 0.3 and vis[j] > 0.3:
                ax.plot([display_pose[i, 0], display_pose[j, 0]],
                       [display_pose[i, 2], display_pose[j, 2]],
                       [display_pose[i, 1], display_pose[j, 1]],
                       'b-', linewidth=2.5)

        # Labels for key joints
        for i in [5, 6, 9, 10, 11, 12, 15, 16]:
            if vis[i] > 0.3:
                ax.text(display_pose[i, 0], display_pose[i, 2], display_pose[i, 1],
                       COCO_NAMES[i], fontsize=7)

        ax.set_xlabel('X')
        ax.set_ylabel('Z (depth)')
        ax.set_zlabel('Y (up)')
        ax.set_title(f'Frame {frame_idx + 1}')

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

    plt.suptitle('POF Reconstruction (Training Coordinate Style)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/input/PushUp.mp4')
    height = float(sys.argv[2]) if len(sys.argv) > 2 else 1.78
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    if not video.exists():
        print(f"Video not found: {video}")
        sys.exit(1)

    visualize_pof_training_style(video, height, max_frames)


if __name__ == "__main__":
    main()
