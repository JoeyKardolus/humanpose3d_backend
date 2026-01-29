#!/usr/bin/env python3
"""
Visualize training samples: 2D pose, MediaPipe 3D, and Ground Truth 3D side by side.

Usage:
    uv run python scripts/viz/visualize_training_samples.py [--num-samples 9] [--output samples.png]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# COCO-17 joint names
COCO_JOINTS = [
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
    'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist',
    'L_Hip', 'R_Hip',
    'L_Knee', 'R_Knee',
    'L_Ankle', 'R_Ankle',
]

# Skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Colors for left/right
LEFT_COLOR = '#3498db'   # Blue
RIGHT_COLOR = '#e74c3c'  # Red
CENTER_COLOR = '#2ecc71'  # Green


def get_limb_color(i, j):
    """Get color based on left/right side."""
    left_joints = {1, 3, 5, 7, 9, 11, 13, 15}
    right_joints = {2, 4, 6, 8, 10, 12, 14, 16}
    if i in left_joints or j in left_joints:
        return LEFT_COLOR
    elif i in right_joints or j in right_joints:
        return RIGHT_COLOR
    return CENTER_COLOR


def draw_pose_2d(ax, pose_2d, title='2D Pose'):
    """Draw 2D pose on image coordinates."""
    ax.set_aspect('equal')

    # Draw bones first
    for i, j in SKELETON:
        color = get_limb_color(i, j)
        ax.plot([pose_2d[i, 0], pose_2d[j, 0]],
                [pose_2d[i, 1], pose_2d[j, 1]],
                color=color, linewidth=2, alpha=0.8)

    # Draw joints
    for idx in range(17):
        if idx in {1, 3, 5, 7, 9, 11, 13, 15}:
            color = LEFT_COLOR
        elif idx in {2, 4, 6, 8, 10, 12, 14, 16}:
            color = RIGHT_COLOR
        else:
            color = CENTER_COLOR
        ax.scatter(pose_2d[idx, 0], pose_2d[idx, 1], c=color, s=40, zorder=5)

    ax.invert_yaxis()  # Image coordinates: Y increases downward
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(alpha=0.3)


def draw_pose_3d(ax, pose_3d, title='3D Pose', color_scheme='sided'):
    """Draw 3D pose with Y-up data shown upright (swap Y->Z for matplotlib)."""
    # Matplotlib 3D has Z as vertical, but our data has Y as vertical
    # So we plot: X -> X, Z -> Y, Y -> Z (vertical)

    # Draw bones
    for i, j in SKELETON:
        if color_scheme == 'sided':
            color = get_limb_color(i, j)
        else:
            color = color_scheme
        ax.plot([pose_3d[i, 0], pose_3d[j, 0]],
                [pose_3d[i, 2], pose_3d[j, 2]],  # Z -> Y axis
                [pose_3d[i, 1], pose_3d[j, 1]],  # Y -> Z axis (vertical)
                color=color, linewidth=2, alpha=0.8)

    # Draw joints
    for idx in range(17):
        if color_scheme == 'sided':
            if idx in {1, 3, 5, 7, 9, 11, 13, 15}:
                color = LEFT_COLOR
            elif idx in {2, 4, 6, 8, 10, 12, 14, 16}:
                color = RIGHT_COLOR
            else:
                color = CENTER_COLOR
        else:
            color = color_scheme
        ax.scatter(pose_3d[idx, 0], pose_3d[idx, 2], pose_3d[idx, 1],
                   c=color, s=40, zorder=5)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Z (depth)')
    ax.set_zlabel('Y (height)')

    # Equal aspect ratio
    max_range = np.abs(pose_3d).max() * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)


def load_samples(data_dir: Path, num_samples: int) -> list:
    """Load random training samples."""
    npz_files = list(data_dir.glob('**/*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    sample_files = random.sample(npz_files, min(num_samples, len(npz_files)))

    samples = []
    for f in sample_files:
        try:
            d = np.load(f)
            samples.append({
                'corrupted': d['corrupted'],
                'ground_truth': d['ground_truth'],
                'pose_2d': d['pose_2d'],
                'azimuth': float(d['azimuth']),
                'elevation': float(d['elevation']),
                'sequence': str(d['sequence']),
            })
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    return samples


def visualize_samples(samples: list, output_path: Path = None):
    """Create grid visualization of samples."""
    n = len(samples)

    fig = plt.figure(figsize=(15, 5 * n))

    for row, s in enumerate(samples):
        # Column 1: 2D pose
        ax1 = fig.add_subplot(n, 3, row * 3 + 1)
        draw_pose_2d(ax1, s['pose_2d'], f"2D Input (Az={s['azimuth']:.0f}°)")

        # Column 2: MediaPipe 3D (corrupted)
        ax2 = fig.add_subplot(n, 3, row * 3 + 2, projection='3d')
        draw_pose_3d(ax2, s['corrupted'], 'MediaPipe 3D', color_scheme='#e74c3c')
        ax2.view_init(elev=15, azim=45)

        # Column 3: Ground Truth 3D
        ax3 = fig.add_subplot(n, 3, row * 3 + 3, projection='3d')
        draw_pose_3d(ax3, s['ground_truth'], 'Ground Truth 3D', color_scheme='#3498db')
        ax3.view_init(elev=15, azim=45)

        # Compute error for this sample
        body_joints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        diff = s['corrupted'][body_joints] - s['ground_truth'][body_joints]
        err_3d = np.linalg.norm(diff, axis=1).mean() * 100
        err_z = np.abs(s['corrupted'][body_joints, 2] - s['ground_truth'][body_joints, 2]).mean() * 100

        # Add error text
        fig.text(0.5, 1 - (row + 0.95) / n,
                 f"3D Error: {err_3d:.1f}cm | Depth Error: {err_z:.1f}cm",
                 ha='center', fontsize=9, color='gray')

    plt.suptitle('Training Data: 2D Input → MediaPipe 3D (Red) vs Ground Truth (Blue)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize training samples')
    parser.add_argument('--data-dir', type=str,
                        default='data/training/aistpp_converted',
                        help='Training data directory')
    parser.add_argument('--num-samples', type=int, default=6,
                        help='Number of samples to show')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    print(f"Loading {args.num_samples} samples from {data_dir}...")
    samples = load_samples(data_dir, args.num_samples)
    print(f"Loaded {len(samples)} samples")

    output = Path(args.output) if args.output else None
    visualize_samples(samples, output)


if __name__ == '__main__':
    main()
