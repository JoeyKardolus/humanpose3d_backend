#!/usr/bin/env python3
"""Visualize AIST++ training pairs: video frame + MediaPipe + Ground Truth."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import cv2

# COCO skeleton connections for visualization
COCO_CONNECTIONS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes-ears
    # Upper body
    (5, 6),   # shoulders
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    # Torso
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),          # hips
    # Lower body
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]


def plot_skeleton(ax, joints, color='blue', label=None, alpha=1.0):
    """Plot 17-joint COCO skeleton on 3D axes."""
    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c=color, s=30, alpha=alpha)

    # Plot connections
    for i, j in COCO_CONNECTIONS:
        ax.plot(
            [joints[i, 0], joints[j, 0]],
            [joints[i, 2], joints[j, 2]],
            [joints[i, 1], joints[j, 1]],
            c=color, alpha=alpha, linewidth=2
        )

    if label:
        ax.scatter([], [], [], c=color, label=label)


def visualize_pair(npz_path: Path, video_dir: Path = None, save_path: Path = None):
    """Visualize a single training pair."""
    data = np.load(npz_path)
    corrupted = data['corrupted']
    ground_truth = data['ground_truth']
    frame_idx = int(data['frame_idx'])
    sequence = str(data['sequence'])

    fig = plt.figure(figsize=(15, 5))

    # Panel 1: Video frame (if available)
    ax1 = fig.add_subplot(131)
    if video_dir:
        video_path = video_dir / f"{sequence}.mp4"
        if not video_path.exists():
            video_path = video_dir / f"{sequence}.avi"
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Frame {frame_idx}')
    ax1.axis('off')

    # Panel 2: MediaPipe
    ax2 = fig.add_subplot(132, projection='3d')
    plot_skeleton(ax2, corrupted, color='blue', label='MediaPipe')
    ax2.set_title('MediaPipe (Input)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('Y')
    ax2.legend()

    max_range = max(np.abs(corrupted).max(), np.abs(ground_truth).max()) * 1.2
    ax2.set_xlim(-max_range, max_range)
    ax2.set_ylim(-max_range, max_range)
    ax2.set_zlim(-max_range, max_range)

    # Panel 3: Ground truth
    ax3 = fig.add_subplot(133, projection='3d')
    plot_skeleton(ax3, ground_truth, color='green', label='AIST++ GT')
    ax3.set_title('Ground Truth')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_zlabel('Y')
    ax3.legend()
    ax3.set_xlim(-max_range, max_range)
    ax3.set_ylim(-max_range, max_range)
    ax3.set_zlim(-max_range, max_range)

    plt.suptitle(f'{sequence}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    data_dir = Path("data/training/aistpp_converted")
    video_dir = Path("data/AIST++/videos")

    npz_files = sorted(data_dir.glob("*.npz"))[:10]

    if not npz_files:
        print("No training files found!")
        return

    print(f"Found {len(npz_files)} files, visualizing first few...")

    for npz_path in npz_files[:5]:
        save_path = data_dir / f"viz_{npz_path.stem}.png"
        visualize_pair(npz_path, video_dir, save_path)


if __name__ == "__main__":
    main()
