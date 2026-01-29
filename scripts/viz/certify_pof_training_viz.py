#!/usr/bin/env python3
"""Visualize POF alignment with 2D skeleton to verify training data correctness.

Creates a grid of samples showing POF arrows overlaid on 2D skeletons.
Arrows should point along limb directions if coordinate transformation is correct.

Usage:
    uv run python scripts/viz/certify_pof_training_viz.py
    uv run python scripts/viz/certify_pof_training_viz.py --samples 16 --output my_viz.png
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pof.dataset import world_to_camera_space, compute_gt_pof_from_3d
from src.pof.constants import LIMB_DEFINITIONS, LIMB_NAMES

# COCO skeleton for drawing
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


def compute_alignment_angles(pose_2d, pof):
    """Compute angle between POF XY and 2D segment for each limb."""
    angles = []
    for limb_idx, (parent, child) in enumerate(LIMB_DEFINITIONS):
        pof_xy = pof[limb_idx, :2]
        pof_xy_norm = pof_xy / (np.linalg.norm(pof_xy) + 1e-6)
        seg_2d = pose_2d[child] - pose_2d[parent]
        seg_2d_norm = seg_2d / (np.linalg.norm(seg_2d) + 1e-6)
        dot = np.clip(np.dot(pof_xy_norm, seg_2d_norm), -1, 1)
        angles.append(np.degrees(np.arccos(dot)))
    return angles


def visualize_pof_alignment(
    data_dir: Path,
    output_path: Path,
    num_samples: int = 9,
    seed: int = 123,
):
    """Create visualization grid showing POF alignment."""
    npz_files = list(data_dir.glob('*.npz'))
    if not npz_files:
        print(f"No NPZ files found in {data_dir}")
        return

    random.seed(seed)
    random.shuffle(npz_files)

    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size))
    if grid_size == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 14))

    sample_idx = 0
    plotted = 0

    for ax in axes:
        if plotted >= num_samples:
            ax.axis('off')
            continue

        # Find a valid sample with camera_R
        while sample_idx < len(npz_files):
            data = np.load(npz_files[sample_idx])
            sample_idx += 1
            if 'camera_R' in data:
                break
        else:
            ax.axis('off')
            continue

        pose_2d = data['pose_2d']
        gt = data['ground_truth']
        camera_R = data['camera_R']

        gt_camera = world_to_camera_space(gt, camera_R)
        pof = compute_gt_pof_from_3d(gt_camera)

        # Draw skeleton
        for i, j in SKELETON:
            ax.plot([pose_2d[i, 0], pose_2d[j, 0]],
                    [pose_2d[i, 1], pose_2d[j, 1]],
                    'b-', linewidth=2, alpha=0.5)

        # Draw joints
        ax.scatter(pose_2d[:, 0], pose_2d[:, 1], c='blue', s=30, zorder=5)

        # Draw POF arrows at midpoint of each limb
        for limb_idx, (parent, child) in enumerate(LIMB_DEFINITIONS):
            mid = (pose_2d[parent] + pose_2d[child]) / 2
            pof_xy = pof[limb_idx, :2]
            arrow_scale = 0.08
            ax.arrow(mid[0], mid[1],
                     pof_xy[0] * arrow_scale, pof_xy[1] * arrow_scale,
                     head_width=0.015, head_length=0.01,
                     fc=colors[limb_idx], ec=colors[limb_idx],
                     linewidth=2, zorder=10)

        # Compute mean alignment error for title
        angles = compute_alignment_angles(pose_2d, pof)
        ax.set_title(f'Mean error: {np.mean(angles):.1f}°', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Flip Y for image coords
        ax.set_aspect('equal')
        ax.axis('off')
        plotted += 1

    plt.suptitle('POF Vectors (arrows) vs 2D Skeleton (blue)\nArrows should point along limb direction', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize POF alignment with 2D skeleton')
    parser.add_argument('--data-dir', type=str, default='data/training/aistpp_converted',
                        help='Training data directory')
    parser.add_argument('--output', type=str, default='data/training/pof_alignment_verification.png',
                        help='Output image path')
    parser.add_argument('--samples', type=int, default=9,
                        help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    args = parser.parse_args()

    visualize_pof_alignment(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        num_samples=args.samples,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
