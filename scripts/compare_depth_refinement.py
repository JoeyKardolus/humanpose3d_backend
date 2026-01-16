#!/usr/bin/env python3
"""
Compare depth refinement model performance against baseline.

Metrics:
- Depth (Z) error: The main target - MediaPipe's weakest dimension
- Full 3D error (MPJPE): Mean Per-Joint Position Error
- Bone length consistency: Variation across frames/samples
- View angle prediction: Azimuth/Elevation accuracy

Usage:
    uv run python scripts/compare_depth_refinement.py
    uv run python scripts/compare_depth_refinement.py --samples 10000 --visualize
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

from src.depth_refinement.inference import DepthRefiner
from src.depth_refinement.dataset import AISTPPDepthDataset


# COCO joint pairs for bone length computation
BONE_PAIRS = [
    (5, 7),   # L shoulder - L elbow
    (7, 9),   # L elbow - L wrist
    (6, 8),   # R shoulder - R elbow
    (8, 10),  # R elbow - R wrist
    (11, 13), # L hip - L knee
    (13, 15), # L knee - L ankle
    (12, 14), # R hip - R knee
    (14, 16), # R knee - R ankle
    (5, 11),  # L shoulder - L hip
    (6, 12),  # R shoulder - R hip
    (5, 6),   # Shoulder width
    (11, 12), # Hip width
]

BONE_NAMES = [
    'L upper arm', 'L forearm', 'R upper arm', 'R forearm',
    'L thigh', 'L shin', 'R thigh', 'R shin',
    'L torso', 'R torso', 'Shoulder width', 'Hip width'
]


def compute_bone_lengths(poses: np.ndarray) -> np.ndarray:
    """Compute bone lengths for all samples.

    Args:
        poses: (N, 17, 3) pose array

    Returns:
        (N, 12) bone lengths
    """
    n_samples = poses.shape[0]
    bone_lengths = np.zeros((n_samples, len(BONE_PAIRS)))

    for i, (j1, j2) in enumerate(BONE_PAIRS):
        diff = poses[:, j1] - poses[:, j2]
        bone_lengths[:, i] = np.linalg.norm(diff, axis=1)

    return bone_lengths


def compute_metrics(
    corrupted: np.ndarray,
    refined: np.ndarray,
    ground_truth: np.ndarray,
    pred_az: Optional[np.ndarray] = None,
    pred_el: Optional[np.ndarray] = None,
    gt_az: Optional[np.ndarray] = None,
    gt_el: Optional[np.ndarray] = None,
) -> dict:
    """Compute comparison metrics.

    Args:
        corrupted: (N, 17, 3) MediaPipe poses (input)
        refined: (N, 17, 3) Refined poses (model output)
        ground_truth: (N, 17, 3) AIST++ ground truth
        pred_az/el: Predicted view angles
        gt_az/el: Ground truth view angles

    Returns:
        dict with metrics
    """
    n_samples = corrupted.shape[0]

    # Per-joint depth (Z) errors
    z_error_before = np.abs(corrupted[:, :, 2] - ground_truth[:, :, 2])
    z_error_after = np.abs(refined[:, :, 2] - ground_truth[:, :, 2])

    # Per-joint 3D errors (MPJPE)
    error_3d_before = np.linalg.norm(corrupted - ground_truth, axis=2)
    error_3d_after = np.linalg.norm(refined - ground_truth, axis=2)

    # Bone length consistency
    bones_corrupted = compute_bone_lengths(corrupted)
    bones_refined = compute_bone_lengths(refined)
    bones_gt = compute_bone_lengths(ground_truth)

    # Bone length error (relative to GT)
    bone_error_before = np.abs(bones_corrupted - bones_gt) / (bones_gt + 1e-6)
    bone_error_after = np.abs(bones_refined - bones_gt) / (bones_gt + 1e-6)

    # Bone length variation (CV = std/mean - lower is more consistent)
    bone_cv_before = bones_corrupted.std(axis=0) / (bones_corrupted.mean(axis=0) + 1e-6)
    bone_cv_after = bones_refined.std(axis=0) / (bones_refined.mean(axis=0) + 1e-6)
    bone_cv_gt = bones_gt.std(axis=0) / (bones_gt.mean(axis=0) + 1e-6)

    metrics = {
        'n_samples': n_samples,

        # Depth (Z) errors
        'z_error_before_cm': z_error_before.mean() * 100,
        'z_error_after_cm': z_error_after.mean() * 100,
        'z_error_improvement': (1 - z_error_after.mean() / z_error_before.mean()) * 100,
        'z_error_before_std_cm': z_error_before.std() * 100,
        'z_error_after_std_cm': z_error_after.std() * 100,

        # 3D errors (MPJPE)
        'mpjpe_before_cm': error_3d_before.mean() * 100,
        'mpjpe_after_cm': error_3d_after.mean() * 100,
        'mpjpe_improvement': (1 - error_3d_after.mean() / error_3d_before.mean()) * 100,

        # Bone length error vs GT
        'bone_error_before_pct': bone_error_before.mean() * 100,
        'bone_error_after_pct': bone_error_after.mean() * 100,
        'bone_error_improvement': (1 - bone_error_after.mean() / bone_error_before.mean()) * 100,

        # Bone length CV (consistency within batch)
        'bone_cv_before': bone_cv_before.mean(),
        'bone_cv_after': bone_cv_after.mean(),
        'bone_cv_gt': bone_cv_gt.mean(),

        # Per-bone CV
        'bone_cv_before_per_bone': bone_cv_before,
        'bone_cv_after_per_bone': bone_cv_after,
    }

    # View angle metrics (if available)
    if pred_az is not None and gt_az is not None:
        # Circular distance for azimuth
        az_diff = np.abs(pred_az - gt_az)
        az_diff = np.minimum(az_diff, 360 - az_diff)
        metrics['azimuth_error_deg'] = az_diff.mean()
        metrics['azimuth_error_std'] = az_diff.std()

    if pred_el is not None and gt_el is not None:
        el_diff = np.abs(pred_el - gt_el)
        metrics['elevation_error_deg'] = el_diff.mean()
        metrics['elevation_error_std'] = el_diff.std()

    return metrics


def print_comparison(metrics: dict, bone_names: list = BONE_NAMES):
    """Print formatted comparison table."""

    print("\n" + "=" * 70)
    print("DEPTH REFINEMENT MODEL COMPARISON")
    print("=" * 70)
    print(f"\nSamples evaluated: {metrics['n_samples']:,}")

    # Depth (Z) error
    print("\n" + "-" * 50)
    print("DEPTH (Z) ERROR - Main target")
    print("-" * 50)
    print(f"  Before (MediaPipe):  {metrics['z_error_before_cm']:6.2f} cm (± {metrics['z_error_before_std_cm']:.2f})")
    print(f"  After (Refined):     {metrics['z_error_after_cm']:6.2f} cm (± {metrics['z_error_after_std_cm']:.2f})")
    improvement = metrics['z_error_improvement']
    sign = '+' if improvement > 0 else ''
    color = '\033[92m' if improvement > 0 else '\033[91m'  # Green/Red
    print(f"  Improvement:         {color}{sign}{improvement:5.1f}%\033[0m")

    # 3D error (MPJPE)
    print("\n" + "-" * 50)
    print("FULL 3D ERROR (MPJPE)")
    print("-" * 50)
    print(f"  Before (MediaPipe):  {metrics['mpjpe_before_cm']:6.2f} cm")
    print(f"  After (Refined):     {metrics['mpjpe_after_cm']:6.2f} cm")
    improvement = metrics['mpjpe_improvement']
    sign = '+' if improvement > 0 else ''
    color = '\033[92m' if improvement > 0 else '\033[91m'
    print(f"  Improvement:         {color}{sign}{improvement:5.1f}%\033[0m")

    # Bone length error
    print("\n" + "-" * 50)
    print("BONE LENGTH ERROR (vs Ground Truth)")
    print("-" * 50)
    print(f"  Before (MediaPipe):  {metrics['bone_error_before_pct']:6.2f}%")
    print(f"  After (Refined):     {metrics['bone_error_after_pct']:6.2f}%")
    improvement = metrics['bone_error_improvement']
    sign = '+' if improvement > 0 else ''
    color = '\033[92m' if improvement > 0 else '\033[91m'
    print(f"  Improvement:         {color}{sign}{improvement:5.1f}%\033[0m")

    # Bone length consistency
    print("\n" + "-" * 50)
    print("BONE LENGTH CONSISTENCY (CV - lower is better)")
    print("-" * 50)
    print(f"  Ground Truth:        {metrics['bone_cv_gt']:.4f}")
    print(f"  Before (MediaPipe):  {metrics['bone_cv_before']:.4f}")
    print(f"  After (Refined):     {metrics['bone_cv_after']:.4f}")

    # Per-bone breakdown
    if 'bone_cv_before_per_bone' in metrics:
        print("\n  Per-bone CV:")
        print(f"  {'Bone':<16} {'GT':>8} {'Before':>8} {'After':>8} {'Change':>8}")
        print("  " + "-" * 48)
        for i, name in enumerate(bone_names):
            gt_cv = metrics.get('bone_cv_gt_per_bone', np.zeros(len(bone_names)))[i] if 'bone_cv_gt_per_bone' in metrics else 0
            before_cv = metrics['bone_cv_before_per_bone'][i]
            after_cv = metrics['bone_cv_after_per_bone'][i]
            change = after_cv - before_cv
            color = '\033[92m' if change < 0 else '\033[91m'
            print(f"  {name:<16} {gt_cv:>8.4f} {before_cv:>8.4f} {after_cv:>8.4f} {color}{change:>+8.4f}\033[0m")

    # View angle prediction
    if 'azimuth_error_deg' in metrics:
        print("\n" + "-" * 50)
        print("VIEW ANGLE PREDICTION")
        print("-" * 50)
        print(f"  Azimuth error:       {metrics['azimuth_error_deg']:6.1f}° (± {metrics['azimuth_error_std']:.1f})")
        print(f"  Elevation error:     {metrics['elevation_error_deg']:6.1f}° (± {metrics['elevation_error_std']:.1f})")

    print("\n" + "=" * 70)


def create_visualization(
    corrupted: np.ndarray,
    refined: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
    sample_idx: int = 0,
):
    """Create visualization of a sample comparison."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15, 5))

    titles = ['MediaPipe (Input)', 'Refined (Output)', 'Ground Truth']
    poses = [corrupted[sample_idx], refined[sample_idx], ground_truth[sample_idx]]

    for i, (title, pose) in enumerate(zip(titles, poses)):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        # Plot joints
        ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c='b', s=50)

        # Plot bones
        for j1, j2 in BONE_PAIRS:
            ax.plot(
                [pose[j1, 0], pose[j2, 0]],
                [pose[j1, 2], pose[j2, 2]],
                [pose[j1, 1], pose[j2, 1]],
                'b-', linewidth=2
            )

        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (depth, m)')
        ax.set_zlabel('Y (m)')

        # Set consistent view
        ax.view_init(elev=15, azim=45)

        # Set axis limits based on GT
        max_range = np.ptp(ground_truth[sample_idx]).max() * 0.6
        mid_x = ground_truth[sample_idx, :, 0].mean()
        mid_y = ground_truth[sample_idx, :, 1].mean()
        mid_z = ground_truth[sample_idx, :, 2].mean()
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(mid_y - max_range, mid_y + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visualization to: {output_path}")


def create_error_histogram(
    z_error_before: np.ndarray,
    z_error_after: np.ndarray,
    output_path: Path,
):
    """Create histogram comparing depth errors."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Flatten errors
    before = z_error_before.flatten() * 100  # cm
    after = z_error_after.flatten() * 100

    bins = np.linspace(0, 20, 50)

    ax.hist(before, bins=bins, alpha=0.5, label=f'MediaPipe (mean: {before.mean():.2f} cm)', color='red')
    ax.hist(after, bins=bins, alpha=0.5, label=f'Refined (mean: {after.mean():.2f} cm)', color='green')

    ax.set_xlabel('Depth Error (cm)')
    ax.set_ylabel('Count')
    ax.set_title('Depth Error Distribution: Before vs After Refinement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved error histogram to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare depth refinement model')
    parser.add_argument('--model', type=str, default='models/checkpoints/best_depth_model.pth',
                        help='Path to trained model')
    parser.add_argument('--data', type=str, default='data/training/aistpp_converted',
                        help='Path to validation data')
    parser.add_argument('--samples', type=int, default=50000,
                        help='Max samples to evaluate')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--output-dir', type=str, default='data/output/depth_comparison',
                        help='Output directory for visualizations')
    args = parser.parse_args()

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run training first: uv run python scripts/train_depth_model.py")
        return 1

    # Load model
    print(f"Loading model from: {model_path}")
    refiner = DepthRefiner(model_path)

    # Load validation data
    print(f"\nLoading validation data from: {args.data}")
    val_dataset = AISTPPDepthDataset(
        args.data,
        split='val',
        augment=False,
        max_samples=args.samples,
    )

    print(f"Evaluating on {len(val_dataset)} samples...")

    # Collect data
    all_corrupted = []
    all_refined = []
    all_gt = []
    all_pred_az = []
    all_pred_el = []
    all_gt_az = []
    all_gt_el = []

    # Process in batches
    batch_size = args.batch_size
    for start_idx in tqdm(range(0, len(val_dataset), batch_size), desc='Evaluating'):
        end_idx = min(start_idx + batch_size, len(val_dataset))

        # Collect batch
        batch_corrupted = []
        batch_gt = []
        batch_vis = []
        batch_2d = []
        batch_az = []
        batch_el = []

        for i in range(start_idx, end_idx):
            sample = val_dataset[i]
            batch_corrupted.append(sample['corrupted'].numpy())
            batch_gt.append(sample['ground_truth'].numpy())
            batch_vis.append(sample['visibility'].numpy())
            batch_2d.append(sample['pose_2d'].numpy())
            batch_az.append(sample['azimuth'].item())
            batch_el.append(sample['elevation'].item())

        batch_corrupted = np.stack(batch_corrupted)
        batch_gt = np.stack(batch_gt)
        batch_vis = np.stack(batch_vis)
        batch_2d = np.stack(batch_2d)

        # Refine
        batch_refined = refiner.refine(batch_corrupted, batch_vis, batch_2d)

        # Get view angle predictions
        info = refiner.get_prediction_info(batch_corrupted, batch_vis, batch_2d)

        all_corrupted.append(batch_corrupted)
        all_refined.append(batch_refined)
        all_gt.append(batch_gt)
        all_pred_az.append(info['pred_azimuth'])
        all_pred_el.append(info['pred_elevation'])
        all_gt_az.extend(batch_az)
        all_gt_el.extend(batch_el)

    # Concatenate
    all_corrupted = np.concatenate(all_corrupted)
    all_refined = np.concatenate(all_refined)
    all_gt = np.concatenate(all_gt)
    all_pred_az = np.concatenate(all_pred_az)
    all_pred_el = np.concatenate(all_pred_el)
    all_gt_az = np.array(all_gt_az)
    all_gt_el = np.array(all_gt_el)

    # Compute metrics
    metrics = compute_metrics(
        all_corrupted, all_refined, all_gt,
        all_pred_az, all_pred_el,
        all_gt_az, all_gt_el,
    )

    # Print results
    print_comparison(metrics)

    # Generate visualizations
    if args.visualize:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sample comparison
        create_visualization(
            all_corrupted, all_refined, all_gt,
            output_dir / 'pose_comparison.png',
            sample_idx=0,
        )

        # Error histogram
        z_error_before = np.abs(all_corrupted[:, :, 2] - all_gt[:, :, 2])
        z_error_after = np.abs(all_refined[:, :, 2] - all_gt[:, :, 2])
        create_error_histogram(
            z_error_before, z_error_after,
            output_dir / 'error_histogram.png',
        )

        # Multiple sample comparisons
        for i in range(min(5, len(all_corrupted))):
            create_visualization(
                all_corrupted, all_refined, all_gt,
                output_dir / f'pose_comparison_{i}.png',
                sample_idx=i,
            )

    return 0


if __name__ == '__main__':
    sys.exit(main())
