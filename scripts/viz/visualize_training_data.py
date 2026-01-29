#!/usr/bin/env python3
"""
Visualization of training data for depth refinement model.

Shows:
- 3D skeleton comparison (MediaPipe vs Ground Truth)
- Per-joint error distribution
- Camera angle distribution
- Sample gallery from different viewpoints

Usage:
    uv run python scripts/viz/visualize_training_data.py [--num-samples 500] [--output training_viz.png]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import random
from collections import defaultdict

# COCO-17 joint names
COCO_JOINTS = [
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',  # 0-4
    'L_Shoulder', 'R_Shoulder',  # 5-6
    'L_Elbow', 'R_Elbow',  # 7-8
    'L_Wrist', 'R_Wrist',  # 9-10
    'L_Hip', 'R_Hip',  # 11-12
    'L_Knee', 'R_Knee',  # 13-14
    'L_Ankle', 'R_Ankle',  # 15-16
]

# Skeleton connections for COCO-17
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Limb joints for analysis (excluding face)
BODY_JOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def load_training_samples(data_dir: Path, num_samples: int = 500) -> list:
    """Load random training samples from npz files."""
    npz_files = list(data_dir.glob('**/*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    # Random sample
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
                'visibility': d['visibility'],
                'sequence': str(d['sequence']),
            })
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue

    return samples


def draw_skeleton_3d(ax, pose, color, alpha=0.8, label=None):
    """Draw 3D skeleton."""
    # Draw joints
    visible = np.ones(17, dtype=bool)  # All visible for normalized poses
    ax.scatter(pose[visible, 0], pose[visible, 1], pose[visible, 2],
               c=color, s=30, alpha=alpha, label=label)

    # Draw bones
    for i, j in SKELETON_CONNECTIONS:
        if visible[i] and visible[j]:
            ax.plot([pose[i, 0], pose[j, 0]],
                   [pose[i, 1], pose[j, 1]],
                   [pose[i, 2], pose[j, 2]],
                   color=color, linewidth=2, alpha=alpha)


def compute_joint_errors(samples: list) -> dict:
    """Compute per-joint position errors."""
    errors = defaultdict(list)

    for s in samples:
        diff = s['corrupted'] - s['ground_truth']
        dist = np.linalg.norm(diff, axis=1)  # (17,) per-joint errors
        for i, name in enumerate(COCO_JOINTS):
            errors[name].append(dist[i])

    return {k: np.array(v) for k, v in errors.items()}


def compute_depth_errors(samples: list) -> np.ndarray:
    """Compute depth (Z) errors only."""
    z_errors = []
    for s in samples:
        z_diff = np.abs(s['corrupted'][:, 2] - s['ground_truth'][:, 2])
        z_errors.append(z_diff.mean())
    return np.array(z_errors)


def create_visualization(samples: list, output_path: Path = None):
    """Create comprehensive training data visualization."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # --- Row 1: 3D Skeleton Examples ---
    # Pick 4 samples with different camera angles
    angle_samples = sorted(samples, key=lambda x: x['azimuth'])
    idx_step = len(angle_samples) // 4
    example_samples = [angle_samples[i * idx_step] for i in range(4)]

    for col, s in enumerate(example_samples):
        ax = fig.add_subplot(gs[0, col], projection='3d')

        # Draw both skeletons
        draw_skeleton_3d(ax, s['ground_truth'], 'blue', alpha=0.6,
                        label='Ground Truth' if col == 0 else None)
        draw_skeleton_3d(ax, s['corrupted'], 'red', alpha=0.6,
                        label='MediaPipe' if col == 0 else None)

        # Format
        ax.set_title(f"Az={s['azimuth']:.0f}° El={s['elevation']:.0f}°", fontsize=9)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Equal aspect
        max_range = 0.8
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        if col == 0:
            ax.legend(loc='upper left', fontsize=7)

    fig.text(0.5, 0.97, '3D Pose Comparison: Ground Truth (Blue) vs MediaPipe (Red)',
             ha='center', fontsize=12, fontweight='bold')

    # --- Row 2: Error Distributions ---

    # 2a. Per-joint error boxplot (body joints only)
    ax_joint = fig.add_subplot(gs[1, :2])
    joint_errors = compute_joint_errors(samples)

    # Filter to body joints
    body_names = [COCO_JOINTS[i] for i in BODY_JOINTS]
    body_errors = [joint_errors[name] * 100 for name in body_names]  # cm

    bp = ax_joint.boxplot(body_errors, labels=body_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax_joint.set_ylabel('Position Error (cm)')
    ax_joint.set_title('Per-Joint 3D Position Error Distribution', fontsize=10)
    ax_joint.tick_params(axis='x', rotation=45)
    ax_joint.grid(axis='y', alpha=0.3)

    # 2b. Depth error histogram
    ax_depth = fig.add_subplot(gs[1, 2])
    depth_errors = compute_depth_errors(samples) * 100  # cm
    ax_depth.hist(depth_errors, bins=40, color='coral', edgecolor='black', alpha=0.7)
    ax_depth.axvline(np.mean(depth_errors), color='red', linestyle='--',
                     label=f'Mean: {np.mean(depth_errors):.1f} cm')
    ax_depth.axvline(np.median(depth_errors), color='blue', linestyle='--',
                     label=f'Median: {np.median(depth_errors):.1f} cm')
    ax_depth.set_xlabel('Mean Depth Error (cm)')
    ax_depth.set_ylabel('Count')
    ax_depth.set_title('Depth (Z) Error Distribution', fontsize=10)
    ax_depth.legend(fontsize=8)
    ax_depth.grid(alpha=0.3)

    # 2c. Total 3D error histogram
    ax_total = fig.add_subplot(gs[1, 3])
    total_errors = []
    for s in samples:
        diff = s['corrupted'][BODY_JOINTS] - s['ground_truth'][BODY_JOINTS]
        total_errors.append(np.linalg.norm(diff, axis=1).mean() * 100)
    total_errors = np.array(total_errors)

    ax_total.hist(total_errors, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
    ax_total.axvline(np.mean(total_errors), color='red', linestyle='--',
                     label=f'Mean: {np.mean(total_errors):.1f} cm')
    ax_total.set_xlabel('Mean 3D Error (cm)')
    ax_total.set_ylabel('Count')
    ax_total.set_title('Total 3D Error Distribution', fontsize=10)
    ax_total.legend(fontsize=8)
    ax_total.grid(alpha=0.3)

    # --- Row 3: Camera Angle Analysis ---

    # 3a. Azimuth distribution
    ax_az = fig.add_subplot(gs[2, 0])
    azimuths = [s['azimuth'] for s in samples]
    ax_az.hist(azimuths, bins=36, color='green', edgecolor='black', alpha=0.7)
    ax_az.set_xlabel('Azimuth (degrees)')
    ax_az.set_ylabel('Count')
    ax_az.set_title('Camera Azimuth Distribution', fontsize=10)
    ax_az.set_xlim(0, 360)
    ax_az.grid(alpha=0.3)

    # 3b. Elevation distribution
    ax_el = fig.add_subplot(gs[2, 1])
    elevations = [s['elevation'] for s in samples]
    ax_el.hist(elevations, bins=20, color='purple', edgecolor='black', alpha=0.7)
    ax_el.set_xlabel('Elevation (degrees)')
    ax_el.set_ylabel('Count')
    ax_el.set_title('Camera Elevation Distribution', fontsize=10)
    ax_el.grid(alpha=0.3)

    # 3c. Azimuth vs Elevation scatter
    ax_scatter = fig.add_subplot(gs[2, 2])
    ax_scatter.scatter(azimuths, elevations, alpha=0.3, s=10, c='teal')
    ax_scatter.set_xlabel('Azimuth (degrees)')
    ax_scatter.set_ylabel('Elevation (degrees)')
    ax_scatter.set_title('Camera Angle Coverage', fontsize=10)
    ax_scatter.set_xlim(0, 360)
    ax_scatter.grid(alpha=0.3)

    # 3d. Error vs Azimuth
    ax_err_az = fig.add_subplot(gs[2, 3])
    # Bin errors by azimuth
    az_bins = np.linspace(0, 360, 13)  # 12 bins of 30 degrees
    az_centers = (az_bins[:-1] + az_bins[1:]) / 2
    az_errors = []
    for i in range(len(az_bins) - 1):
        bin_samples = [s for s in samples
                       if az_bins[i] <= s['azimuth'] < az_bins[i+1]]
        if bin_samples:
            errs = []
            for s in bin_samples:
                diff = s['corrupted'][BODY_JOINTS] - s['ground_truth'][BODY_JOINTS]
                errs.append(np.linalg.norm(diff, axis=1).mean() * 100)
            az_errors.append(np.mean(errs))
        else:
            az_errors.append(np.nan)

    ax_err_az.bar(az_centers, az_errors, width=25, color='orange',
                  edgecolor='black', alpha=0.7)
    ax_err_az.set_xlabel('Azimuth (degrees)')
    ax_err_az.set_ylabel('Mean 3D Error (cm)')
    ax_err_az.set_title('Error by Camera Angle', fontsize=10)
    ax_err_az.set_xlim(0, 360)
    ax_err_az.grid(alpha=0.3)

    # Overall title
    plt.suptitle(f'Training Data Visualization ({len(samples):,} samples)',
                 fontsize=14, fontweight='bold', y=1.0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize training data')
    parser.add_argument('--data-dir', type=str,
                        default='data/training/aistpp_converted',
                        help='Training data directory')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='Number of samples to visualize')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (displays if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    print(f"Loading {args.num_samples} samples from {data_dir}...")
    samples = load_training_samples(data_dir, args.num_samples)
    print(f"Loaded {len(samples)} samples")

    # Compute quick stats
    total_files = len(list(data_dir.glob('**/*.npz')))
    print(f"\nDataset statistics:")
    print(f"  Total samples: {total_files:,}")
    print(f"  Samples loaded: {len(samples):,}")

    # Error stats
    depth_errors = compute_depth_errors(samples) * 100
    print(f"\nDepth error (Z):")
    print(f"  Mean: {np.mean(depth_errors):.2f} cm")
    print(f"  Median: {np.median(depth_errors):.2f} cm")
    print(f"  Std: {np.std(depth_errors):.2f} cm")

    total_errors = []
    for s in samples:
        diff = s['corrupted'][BODY_JOINTS] - s['ground_truth'][BODY_JOINTS]
        total_errors.append(np.linalg.norm(diff, axis=1).mean() * 100)
    total_errors = np.array(total_errors)
    print(f"\nTotal 3D error (body joints):")
    print(f"  Mean: {np.mean(total_errors):.2f} cm")
    print(f"  Median: {np.median(total_errors):.2f} cm")
    print(f"  Std: {np.std(total_errors):.2f} cm")

    output = Path(args.output) if args.output else None
    create_visualization(samples, output)


if __name__ == '__main__':
    main()
