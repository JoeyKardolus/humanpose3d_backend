#!/usr/bin/env python3
"""
Visualize CMU training data with proper skeleton connections.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


# Skeleton connections (bones)
SKELETON_CONNECTIONS = [
    # Spine
    ("Hip", "C7_study"),
    ("C7_study", "Neck"),
    ("Neck", "Head"),

    # Right arm
    ("C7_study", "RShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),

    # Left arm
    ("C7_study", "LShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),

    # Right leg
    ("Hip", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("RAnkle", "RHeel"),
    ("RAnkle", "RBigToe"),

    # Left leg
    ("Hip", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("LAnkle", "LHeel"),
    ("LAnkle", "LBigToe"),
]


def plot_skeleton(ax, positions, marker_names, color, label, alpha=0.6):
    """Plot skeleton with connections."""
    marker_idx = {name: i for i, name in enumerate(marker_names)}

    # Plot markers
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=color, marker='o', s=30, alpha=alpha, label=label)

    # Plot connections
    for marker1, marker2 in SKELETON_CONNECTIONS:
        if marker1 in marker_idx and marker2 in marker_idx:
            idx1 = marker_idx[marker1]
            idx2 = marker_idx[marker2]

            pos1 = positions[idx1]
            pos2 = positions[idx2]

            # Draw line
            ax.plot([pos1[0], pos2[0]],
                   [pos1[1], pos2[1]],
                   [pos1[2], pos2[2]],
                   c=color, alpha=alpha*0.5, linewidth=2)


def visualize_sample(sample_path: Path):
    """Visualize a single training sample with skeleton."""
    data = np.load(sample_path)

    corrupted = data["corrupted"]  # (markers, 3)
    ground_truth = data["ground_truth"]  # (markers, 3)
    marker_names = data["marker_names"].tolist()
    camera_angle = data["camera_angle"]
    noise_std = data["noise_std"]

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(18, 8))

    # Plot 1: Corrupted (with noise)
    ax1 = fig.add_subplot(121, projection='3d')
    plot_skeleton(ax1, corrupted, marker_names, 'red', 'Corrupted (with noise)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
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
    plot_skeleton(ax2, ground_truth, marker_names, 'blue', 'Ground Truth (CMU)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
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

    # Print info
    print(f"\n{'='*80}")
    print(f"Sample: {sample_path.name}")
    print(f"Camera angle: {camera_angle}°")
    print(f"Noise std: {noise_std}m")
    print(f"Number of markers: {len(marker_names)}")

    # Compute error stats
    error = np.linalg.norm(corrupted - ground_truth, axis=1) * 1000  # mm
    print(f"\nCorruption error: mean={error.mean():.1f}mm, max={error.max():.1f}mm")

    # Check if skeleton looks reasonable
    marker_idx = {name: i for i, name in enumerate(marker_names)}
    if "Hip" in marker_idx and "Head" in marker_idx:
        hip_pos = ground_truth[marker_idx["Hip"]]
        head_pos = ground_truth[marker_idx["Head"]]
        height = head_pos[1] - hip_pos[1]
        print(f"Estimated torso height: {height:.2f}m")

    print(f"{'='*80}")

    return fig


def main():
    """Visualize multiple training samples with skeletons."""
    data_dir = Path("data/training/cmu_converted")

    # Get a few samples with different noise levels
    samples = []

    # Try to get one sample from each noise level
    for noise in ['n030', 'n050', 'n080']:
        matching = list(data_dir.glob(f"*{noise}.npz"))
        if matching:
            samples.append(matching[0])

    if not samples:
        samples = sorted(data_dir.glob("*.npz"))[:3]

    if not samples:
        print(f"No training data found in {data_dir}")
        return

    print(f"Found {len(samples)} samples to visualize")

    # Visualize each
    for i, sample_path in enumerate(samples):
        fig = visualize_sample(sample_path)
        output_path = f"training_skeleton_{i+1}_{sample_path.stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {output_path}")
        plt.close()

    print("\n✓ Training skeleton visualization complete!")


if __name__ == "__main__":
    main()
