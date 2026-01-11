#!/usr/bin/env python3
"""
Visualize CMU training data to verify quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


def visualize_sample(sample_path: Path):
    """Visualize a single training sample."""
    data = np.load(sample_path)

    corrupted = data["corrupted"]  # (markers, 3)
    ground_truth = data["ground_truth"]  # (markers, 3)
    marker_names = data["marker_names"].tolist()
    camera_angle = data["camera_angle"]
    noise_std = data["noise_std"]

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 7))

    # Plot 1: Corrupted (with noise)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(corrupted[:, 0], corrupted[:, 1], corrupted[:, 2],
                c='red', marker='o', s=50, alpha=0.6, label='Corrupted')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Corrupted (noise={noise_std}m, camera={camera_angle}°)')
    ax1.legend()

    # Set equal aspect ratio
    max_range = np.array([
        corrupted[:, 0].max() - corrupted[:, 0].min(),
        corrupted[:, 1].max() - corrupted[:, 1].min(),
        corrupted[:, 2].max() - corrupted[:, 2].min()
    ]).max() / 2.0

    mid_x = (corrupted[:, 0].max() + corrupted[:, 0].min()) * 0.5
    mid_y = (corrupted[:, 1].max() + corrupted[:, 1].min()) * 0.5
    mid_z = (corrupted[:, 2].max() + corrupted[:, 2].min()) * 0.5

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # Plot 2: Ground truth
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                c='blue', marker='o', s=50, alpha=0.6, label='Ground Truth')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Ground Truth (CMU mocap)')
    ax2.legend()

    # Set equal aspect ratio
    max_range = np.array([
        ground_truth[:, 0].max() - ground_truth[:, 0].min(),
        ground_truth[:, 1].max() - ground_truth[:, 1].min(),
        ground_truth[:, 2].max() - ground_truth[:, 2].min()
    ]).max() / 2.0

    mid_x = (ground_truth[:, 0].max() + ground_truth[:, 0].min()) * 0.5
    mid_y = (ground_truth[:, 1].max() + ground_truth[:, 1].min()) * 0.5
    mid_z = (ground_truth[:, 2].max() + ground_truth[:, 2].min()) * 0.5

    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    # Print marker info
    print(f"\n{'='*80}")
    print(f"Sample: {sample_path.name}")
    print(f"Camera angle: {camera_angle}°")
    print(f"Noise std: {noise_std}m")
    print(f"Number of markers: {len(marker_names)}")
    print(f"\nMarkers:")
    for i, name in enumerate(marker_names):
        gt_pos = ground_truth[i]
        corr_pos = corrupted[i]
        diff = np.linalg.norm(gt_pos - corr_pos)
        print(f"  {i:2d}. {name:20s} | GT: [{gt_pos[0]:6.3f}, {gt_pos[1]:6.3f}, {gt_pos[2]:6.3f}] | Error: {diff:.4f}m")

    print(f"\n{'='*80}")

    return fig


def main():
    """Visualize multiple training samples."""
    data_dir = Path("data/training/cmu_converted")

    # Get a few random samples
    samples = sorted(data_dir.glob("*.npz"))

    if not samples:
        print(f"No training data found in {data_dir}")
        return

    print(f"Found {len(samples)} training samples")

    # Visualize first 3
    for i, sample_path in enumerate(samples[:3]):
        fig = visualize_sample(sample_path)
        output_path = f"training_sample_{i+1}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {output_path}")
        plt.close()

    print("\n✓ Training data visualization complete!")


if __name__ == "__main__":
    main()
