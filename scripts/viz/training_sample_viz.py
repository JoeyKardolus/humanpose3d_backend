#!/usr/bin/env python3
"""
Visualize training samples with alignment verification.

Shows:
- 2D pose overlay
- 3D corrupted (MediaPipe) vs ground truth
- Depth errors per joint
- Limb orientation errors (arms/legs/torso breakdown)
- Body frame axes for alignment verification

Usage:
    uv run python scripts/viz/training_sample_viz.py
    uv run python scripts/viz/training_sample_viz.py --dataset mtc
    uv run python scripts/viz/training_sample_viz.py --sample path/to/sample.npz
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.depth_refinement.data_utils import (
    get_body_frame,
    compute_body_frame_error,
    compute_limb_orientation_errors,
)

# COCO-17 skeleton connections
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes-ears
    (5, 6),                          # shoulders
    (5, 7), (7, 9),                  # left arm
    (6, 8), (8, 10),                 # right arm
    (5, 11), (6, 12),                # shoulders to hips
    (11, 12),                        # hips
    (11, 13), (13, 15),              # left leg
    (12, 14), (14, 16),              # right leg
]

COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

LIMB_NAMES = {
    'torso': ['L_torso', 'R_torso', 'L_cross', 'R_cross'],
    'arms': ['L_upper', 'R_upper', 'L_fore', 'R_fore'],
    'legs': ['L_thigh', 'R_thigh', 'L_shin', 'R_shin'],
}


def plot_skeleton_3d(ax, joints, color='blue', label=None, alpha=1.0, linestyle='-'):
    """Plot 17-joint COCO skeleton on 3D axes."""
    ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1],
               c=color, s=30, alpha=alpha)

    for i, j in COCO_CONNECTIONS:
        ax.plot(
            [joints[i, 0], joints[j, 0]],
            [joints[i, 2], joints[j, 2]],
            [joints[i, 1], joints[j, 1]],
            c=color, alpha=alpha, linewidth=2, linestyle=linestyle
        )

    if label:
        ax.scatter([], [], [], c=color, label=label)


def plot_body_frame(ax, pose, scale=0.3, alpha=0.8):
    """Plot body frame axes on 3D plot."""
    frame = get_body_frame(pose)
    pelvis = (pose[11] + pose[12]) / 2

    # X axis (right) - red
    ax.quiver(pelvis[0], pelvis[2], pelvis[1],
              frame[0, 0] * scale, frame[2, 0] * scale, frame[1, 0] * scale,
              color='red', alpha=alpha, arrow_length_ratio=0.1, linewidth=2)

    # Y axis (up) - green
    ax.quiver(pelvis[0], pelvis[2], pelvis[1],
              frame[0, 1] * scale, frame[2, 1] * scale, frame[1, 1] * scale,
              color='green', alpha=alpha, arrow_length_ratio=0.1, linewidth=2)

    # Z axis (forward) - blue
    ax.quiver(pelvis[0], pelvis[2], pelvis[1],
              frame[0, 2] * scale, frame[2, 2] * scale, frame[1, 2] * scale,
              color='blue', alpha=alpha, arrow_length_ratio=0.1, linewidth=2)


def plot_pose_2d(ax, pose_2d, title='2D Pose', color='blue'):
    """Plot 2D pose on image axes."""
    ax.scatter(pose_2d[:, 0], pose_2d[:, 1], c=color, s=30)

    for i, j in COCO_CONNECTIONS:
        ax.plot(
            [pose_2d[i, 0], pose_2d[j, 0]],
            [pose_2d[i, 1], pose_2d[j, 1]],
            c=color, linewidth=2, alpha=0.7
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Flip Y for image coords
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def compute_depth_errors(mp_pose, gt_pose):
    """Compute per-joint depth (Z) errors."""
    return np.abs(mp_pose[:, 2] - gt_pose[:, 2])


def visualize_sample(npz_path: Path, save_path: Path = None, show_frames: bool = True):
    """
    Visualize a single training sample with comprehensive diagnostics.

    Layout (2 rows, 5 cols):
    - Row 1: 2D MediaPipe, 2D GT Projected, 3D overlay, depth errors bar, limb errors bar
    - Row 2: 3D MP only, 3D GT only, body frame alignment, error stats text, quality flags
    """
    data = np.load(npz_path)
    corrupted = data['corrupted']  # MediaPipe (normalized)
    ground_truth = data['ground_truth']  # GT (normalized)

    # Optional fields
    pose_2d = data.get('pose_2d', None)
    projected_2d = data.get('projected_2d', None)
    visibility = data.get('visibility', None)
    azimuth = data.get('azimuth', 0.0)
    elevation = data.get('elevation', 0.0)
    mp_scale = data.get('mp_scale', 0.0)
    sequence = str(data.get('sequence', npz_path.stem))
    frame_idx = int(data.get('frame_idx', 0))

    fig = plt.figure(figsize=(25, 10))

    # ===== Row 1 =====

    # 1. 2D Pose (MediaPipe - from video)
    ax1 = fig.add_subplot(2, 5, 1)
    if pose_2d is not None:
        plot_pose_2d(ax1, pose_2d, '2D MediaPipe\n(from video)', color='red')
    else:
        ax1.text(0.5, 0.5, 'No MediaPipe 2D', ha='center', va='center')
        ax1.set_title('2D MediaPipe')

    # 2. 2D Pose (GT projected - from 3D + camera)
    ax2 = fig.add_subplot(2, 5, 2)
    if projected_2d is not None:
        plot_pose_2d(ax2, projected_2d, '2D GT Projected\n(from 3D + camera)', color='green')
    else:
        ax2.text(0.5, 0.5, 'No projected 2D', ha='center', va='center')
        ax2.set_title('2D GT Projected')

    # 3. 3D Overlay
    ax3 = fig.add_subplot(2, 5, 3, projection='3d')
    plot_skeleton_3d(ax3, corrupted, color='red', label='MediaPipe', alpha=0.7, linestyle='--')
    plot_skeleton_3d(ax3, ground_truth, color='green', label='Ground Truth', alpha=0.9)
    ax3.set_title('3D Overlay')
    ax3.legend()
    set_3d_axes_equal(ax3, corrupted, ground_truth)

    # 4. Depth errors per joint
    ax4 = fig.add_subplot(2, 5, 4)
    depth_errors = compute_depth_errors(corrupted, ground_truth)
    colors = ['red' if e > 0.1 else 'orange' if e > 0.05 else 'green' for e in depth_errors]
    ax4.bar(range(17), depth_errors, color=colors)
    ax4.set_xticks(range(17))
    ax4.set_xticklabels([k[:3] for k in COCO_KEYPOINTS], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Depth Error (normalized)')
    ax4.set_title(f'Per-Joint Depth Errors (mean: {depth_errors.mean():.3f})')
    ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='High')
    ax4.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Medium')

    # 5. Limb orientation errors
    ax5 = fig.add_subplot(2, 5, 5)
    limb_errors = compute_limb_orientation_errors(corrupted, ground_truth)

    # Prepare data for grouped bar chart
    x = np.arange(4)
    width = 0.25

    torso_vals = limb_errors['torso_errors']
    arm_vals = limb_errors['arm_errors']
    leg_vals = limb_errors['leg_errors']

    ax5.bar(x - width, torso_vals, width, label=f'Torso (mean: {np.mean(torso_vals):.1f}°)', color='purple')
    ax5.bar(x, arm_vals, width, label=f'Arms (mean: {np.mean(arm_vals):.1f}°)', color='blue')
    ax5.bar(x + width, leg_vals, width, label=f'Legs (mean: {np.mean(leg_vals):.1f}°)', color='orange')

    ax5.set_xticks(x)
    ax5.set_xticklabels(['Left 1', 'Right 1', 'Left 2', 'Right 2'])
    ax5.set_ylabel('Orientation Error (degrees)')
    ax5.set_title('Limb Orientation Errors')
    ax5.legend(fontsize=8)
    ax5.axhline(y=30, color='red', linestyle='--', alpha=0.5)

    # ===== Row 2 =====

    # 6. 3D MediaPipe only with frame
    ax6 = fig.add_subplot(2, 5, 6, projection='3d')
    plot_skeleton_3d(ax6, corrupted, color='red', label='MediaPipe')
    if show_frames:
        plot_body_frame(ax6, corrupted, scale=0.3)
    ax6.set_title('MediaPipe + Body Frame')
    set_3d_axes_equal(ax6, corrupted)

    # 7. 3D GT only with frame
    ax7 = fig.add_subplot(2, 5, 7, projection='3d')
    plot_skeleton_3d(ax7, ground_truth, color='green', label='Ground Truth')
    if show_frames:
        plot_body_frame(ax7, ground_truth, scale=0.3)
    ax7.set_title('Ground Truth + Body Frame')
    set_3d_axes_equal(ax7, ground_truth)

    # 8. Body frame alignment comparison (both frames overlaid)
    ax8 = fig.add_subplot(2, 5, 8, projection='3d')
    plot_body_frame(ax8, corrupted, scale=0.4, alpha=0.6)
    plot_body_frame(ax8, ground_truth, scale=0.4, alpha=1.0)

    frame_error = compute_body_frame_error(corrupted, ground_truth)
    ax8.set_title(f'Frame Alignment\n(Error: {frame_error:.1f}°)')
    ax8.set_xlim(-0.5, 0.5)
    ax8.set_ylim(-0.5, 0.5)
    ax8.set_zlim(-0.5, 0.5)

    # 9. Statistics text panel
    ax9 = fig.add_subplot(2, 5, 9)
    ax9.axis('off')

    # Compute torso visibility if available
    torso_vis_str = "N/A"
    if visibility is not None:
        torso_vis = visibility[[5, 6, 11, 12]].min()
        torso_vis_str = f"{torso_vis:.2f}"

    stats_text = f"""
Sample: {sequence}
Frame: {frame_idx}
View: Az={float(azimuth):.1f}°, El={float(elevation):.1f}°
MP Scale: {float(mp_scale):.3f}
Torso Visibility: {torso_vis_str}

--- Error Summary ---
Body Frame Error: {frame_error:.1f}°
Mean Depth Error: {depth_errors.mean():.4f}
Max Depth Error: {depth_errors.max():.4f}

--- Limb Orientation ---
Torso Mean: {np.mean(torso_vals):.1f}°
Arms Mean: {np.mean(arm_vals):.1f}°
Legs Mean: {np.mean(leg_vals):.1f}°
"""
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 10. Quality flags panel
    ax10 = fig.add_subplot(2, 5, 10)
    ax10.axis('off')

    # Quality checks
    quality_checks = []
    quality_checks.append(('Frame Align', frame_error < 15, f'{frame_error:.1f}° < 15°'))
    quality_checks.append(('Torso Error', np.mean(torso_vals) < 30, f'{np.mean(torso_vals):.1f}° < 30°'))

    if visibility is not None:
        torso_vis = visibility[[5, 6, 11, 12]].min()
        quality_checks.append(('Torso Vis', torso_vis >= 0.3, f'{torso_vis:.2f} ≥ 0.3'))

    if mp_scale > 0:
        quality_checks.append(('MP Scale', 0.05 <= mp_scale <= 5.0, f'{float(mp_scale):.2f} in [0.05, 5.0]'))

    # Check if both 2D views are similar (camera perspective match)
    if pose_2d is not None and projected_2d is not None:
        # Compare pelvis position (should be close if camera is correct)
        mp_pelvis_2d = (pose_2d[11] + pose_2d[12]) / 2
        gt_pelvis_2d = (projected_2d[11] + projected_2d[12]) / 2
        pelvis_2d_dist = np.linalg.norm(mp_pelvis_2d - gt_pelvis_2d)
        quality_checks.append(('2D Alignment', pelvis_2d_dist < 0.2, f'{pelvis_2d_dist:.3f} < 0.2'))

    quality_text = "=== Quality Checks ===\n\n"
    for name, passed, detail in quality_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        color_code = "" if passed else "(!)"
        quality_text += f"{status} {name}: {detail} {color_code}\n"

    overall_pass = all(p for _, p, _ in quality_checks)
    quality_text += f"\n{'='*25}\n"
    quality_text += f"Overall: {'GOOD SAMPLE' if overall_pass else 'CHECK SAMPLE'}"

    ax10.text(0.05, 0.95, quality_text, transform=ax10.transAxes,
              fontsize=9, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round',
                       facecolor='lightgreen' if overall_pass else 'lightyellow',
                       alpha=0.5))

    plt.suptitle(f'{npz_path.name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.savefig(npz_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {npz_path.with_suffix('.png')}")

    plt.close()

    return {
        'frame_error': frame_error,
        'depth_mean': depth_errors.mean(),
        'torso_mean': np.mean(torso_vals),
        'arms_mean': np.mean(arm_vals),
        'legs_mean': np.mean(leg_vals),
    }


def set_3d_axes_equal(ax, *poses):
    """Set equal aspect ratio for 3D plot."""
    all_points = np.vstack(poses)
    max_range = np.abs(all_points).max() * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')


def aggregate_stats(stats_list):
    """Compute aggregate statistics from multiple samples."""
    if not stats_list:
        return {}

    return {
        'frame_error': {
            'mean': np.mean([s['frame_error'] for s in stats_list]),
            'std': np.std([s['frame_error'] for s in stats_list]),
            'max': np.max([s['frame_error'] for s in stats_list]),
        },
        'depth_mean': {
            'mean': np.mean([s['depth_mean'] for s in stats_list]),
            'std': np.std([s['depth_mean'] for s in stats_list]),
        },
        'torso_mean': {
            'mean': np.mean([s['torso_mean'] for s in stats_list]),
            'std': np.std([s['torso_mean'] for s in stats_list]),
        },
        'arms_mean': {
            'mean': np.mean([s['arms_mean'] for s in stats_list]),
            'std': np.std([s['arms_mean'] for s in stats_list]),
        },
        'legs_mean': {
            'mean': np.mean([s['legs_mean'] for s in stats_list]),
            'std': np.std([s['legs_mean'] for s in stats_list]),
        },
    }


def visualize_motion_sync(sequence_samples: list, output_path: Path = None):
    """
    Visualize motion sync between MediaPipe and GT over consecutive frames.

    Plots pelvis Y position (vertical motion) for both MP and GT to detect
    timing/sync issues between video and motion capture.

    Args:
        sequence_samples: List of paths to consecutive samples from same sequence
        output_path: Where to save the plot
    """
    if len(sequence_samples) < 3:
        print("Need at least 3 consecutive samples for motion sync visualization")
        return

    # Extract motion data
    frames = []
    mp_pelvis_y = []
    gt_pelvis_y = []
    mp_pelvis_x = []
    gt_pelvis_x = []

    for sample_path in sorted(sequence_samples):
        data = np.load(sample_path)
        frame_idx = int(data.get('frame_idx', 0))
        corrupted = data['corrupted']
        ground_truth = data['ground_truth']

        # Pelvis is midpoint of hips (indices 11, 12)
        mp_pelvis = (corrupted[11] + corrupted[12]) / 2
        gt_pelvis = (ground_truth[11] + ground_truth[12]) / 2

        frames.append(frame_idx)
        mp_pelvis_y.append(mp_pelvis[1])  # Y = vertical
        gt_pelvis_y.append(gt_pelvis[1])
        mp_pelvis_x.append(mp_pelvis[0])  # X = horizontal
        gt_pelvis_x.append(gt_pelvis[0])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Pelvis Y (vertical motion)
    ax1 = axes[0, 0]
    ax1.plot(frames, mp_pelvis_y, 'r-o', label='MediaPipe', markersize=4)
    ax1.plot(frames, gt_pelvis_y, 'g-s', label='Ground Truth', markersize=4)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Pelvis Y (normalized)')
    ax1.set_title('Vertical Motion (Y) - Should Move In Sync')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Pelvis X (horizontal motion)
    ax2 = axes[0, 1]
    ax2.plot(frames, mp_pelvis_x, 'r-o', label='MediaPipe', markersize=4)
    ax2.plot(frames, gt_pelvis_x, 'g-s', label='Ground Truth', markersize=4)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Pelvis X (normalized)')
    ax2.set_title('Horizontal Motion (X) - Should Move In Sync')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Motion correlation (scatter)
    ax3 = axes[1, 0]
    ax3.scatter(gt_pelvis_y, mp_pelvis_y, c=frames, cmap='viridis', s=30)
    ax3.set_xlabel('GT Pelvis Y')
    ax3.set_ylabel('MP Pelvis Y')
    ax3.set_title('Motion Correlation (color = frame)')

    # Add diagonal line for perfect correlation
    min_val = min(min(gt_pelvis_y), min(mp_pelvis_y))
    max_val = max(max(gt_pelvis_y), max(mp_pelvis_y))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect sync')
    ax3.legend()

    # Compute correlation coefficient
    corr = np.corrcoef(gt_pelvis_y, mp_pelvis_y)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Motion velocity comparison
    ax4 = axes[1, 1]
    if len(frames) > 1:
        mp_velocity = np.diff(mp_pelvis_y)
        gt_velocity = np.diff(gt_pelvis_y)
        frame_mids = [(frames[i] + frames[i+1])/2 for i in range(len(frames)-1)]

        ax4.plot(frame_mids, mp_velocity, 'r-o', label='MP Velocity', markersize=4)
        ax4.plot(frame_mids, gt_velocity, 'g-s', label='GT Velocity', markersize=4)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Frame Index')
        ax4.set_ylabel('Δ Pelvis Y')
        ax4.set_title('Motion Velocity - Sign Should Match')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Check for sign mismatches (motion in opposite directions)
        sign_mismatches = np.sum(np.sign(mp_velocity) != np.sign(gt_velocity))
        ax4.text(0.05, 0.95, f'Sign mismatches: {sign_mismatches}/{len(mp_velocity)}',
                 transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Extract sequence name from first sample
    seq_name = str(np.load(sequence_samples[0]).get('sequence', 'unknown'))

    plt.suptitle(f'Motion Sync Analysis: {seq_name}\n({len(frames)} frames)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved motion sync plot: {output_path}")
    else:
        plt.savefig(f'motion_sync_{seq_name}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: motion_sync_{seq_name}.png")

    plt.close()

    return {
        'correlation': corr,
        'frames': len(frames),
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize training samples')
    parser.add_argument('--dataset', choices=['aistpp', 'mtc', 'both'], default='aistpp',
                        help='Dataset to visualize')
    parser.add_argument('--sample', type=str, default=None,
                        help='Path to specific sample .npz file')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualizations')
    parser.add_argument('--no-frames', action='store_true',
                        help='Hide body frame axes')
    parser.add_argument('--motion-sync', type=str, default=None,
                        help='Sequence prefix for motion sync analysis (e.g., "gBR_sBM_cAll_d04")')
    args = parser.parse_args()

    # Single sample mode
    if args.sample:
        sample_path = Path(args.sample)
        if not sample_path.exists():
            print(f"Sample not found: {sample_path}")
            return

        # Default to data/training/viz/ directory to avoid deletion with training data
        if args.output_dir:
            viz_dir = Path(args.output_dir)
        else:
            viz_dir = Path('data/training/viz')
        viz_dir.mkdir(parents=True, exist_ok=True)
        save_path = viz_dir / f"viz_{sample_path.stem}.png"
        stats = visualize_sample(sample_path, save_path, show_frames=not args.no_frames)
        print(f"\nStats: {stats}")
        return

    # Motion sync mode - analyze consecutive frames from a sequence
    if args.motion_sync:
        datasets = {
            'aistpp': Path('data/training/aistpp_converted'),
            'mtc': Path('data/training/mtc_converted'),
        }
        ds_path = datasets[args.dataset]

        if not ds_path.exists():
            print(f"Dataset directory not found: {ds_path}")
            return

        # Find all samples matching the sequence prefix
        pattern = f"{args.motion_sync}*.npz"
        matching_samples = sorted(ds_path.glob(pattern))

        if len(matching_samples) < 3:
            print(f"Found only {len(matching_samples)} samples matching '{args.motion_sync}*'")
            print("Need at least 3 consecutive samples for motion sync analysis")
            return

        print(f"Found {len(matching_samples)} samples for sequence: {args.motion_sync}")

        output_dir = Path(args.output_dir) if args.output_dir else ds_path / 'viz'
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"motion_sync_{args.motion_sync}.png"
        result = visualize_motion_sync(matching_samples, output_path)

        if result:
            print(f"\nMotion Sync Results:")
            print(f"  Frames analyzed: {result['frames']}")
            print(f"  Correlation: {result['correlation']:.3f}")
            if result['correlation'] > 0.7:
                print("  Status: GOOD - Motion is well synchronized")
            elif result['correlation'] > 0.4:
                print("  Status: MODERATE - Some sync issues may exist")
            else:
                print("  Status: WARNING - Poor sync, check video/mocap alignment")
        return

    # Dataset mode
    datasets = {
        'aistpp': Path('data/training/aistpp_converted'),
        'mtc': Path('data/training/mtc_converted'),
    }

    to_process = ['aistpp', 'mtc'] if args.dataset == 'both' else [args.dataset]

    for ds_name in to_process:
        ds_path = datasets[ds_name]

        if not ds_path.exists():
            print(f"Dataset directory not found: {ds_path}")
            continue

        npz_files = sorted(ds_path.glob('*.npz'))
        if not npz_files:
            print(f"No .npz files found in {ds_path}")
            continue

        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name.upper()} ({len(npz_files)} samples)")
        print(f"{'='*60}")

        # Sample evenly across dataset
        if len(npz_files) > args.num_samples:
            indices = np.linspace(0, len(npz_files) - 1, args.num_samples, dtype=int)
            selected = [npz_files[i] for i in indices]
        else:
            selected = npz_files[:args.num_samples]

        output_dir = Path(args.output_dir) if args.output_dir else ds_path / 'viz'
        output_dir.mkdir(parents=True, exist_ok=True)

        stats_list = []
        for npz_path in selected:
            print(f"\nProcessing: {npz_path.name}")
            save_path = output_dir / f"viz_{npz_path.stem}.png"
            stats = visualize_sample(npz_path, save_path, show_frames=not args.no_frames)
            stats_list.append(stats)

        # Print aggregate stats
        agg = aggregate_stats(stats_list)
        print(f"\n{'='*60}")
        print(f"AGGREGATE STATS ({ds_name.upper()})")
        print(f"{'='*60}")
        print(f"Body Frame Error: {agg['frame_error']['mean']:.1f}° ± {agg['frame_error']['std']:.1f}°")
        print(f"Mean Depth Error: {agg['depth_mean']['mean']:.4f} ± {agg['depth_mean']['std']:.4f}")
        print(f"Torso Orientation: {agg['torso_mean']['mean']:.1f}° ± {agg['torso_mean']['std']:.1f}°")
        print(f"Arms Orientation:  {agg['arms_mean']['mean']:.1f}° ± {agg['arms_mean']['std']:.1f}°")
        print(f"Legs Orientation:  {agg['legs_mean']['mean']:.1f}° ± {agg['legs_mean']['std']:.1f}°")

        if agg['torso_mean']['mean'] > 30:
            print(f"\nWARNING: High torso error ({agg['torso_mean']['mean']:.1f}°) suggests rotation misalignment!")
            print("Check that align_body_frames() is being called in the conversion script.")


if __name__ == '__main__':
    main()
