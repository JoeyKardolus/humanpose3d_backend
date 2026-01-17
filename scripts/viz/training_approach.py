#!/usr/bin/env python3
"""
Visualize the training approach for depth refinement.

Shows:
1. MediaPipe (corrupted) vs Ground Truth skeleton comparison
2. Depth errors (Z difference) per joint
3. View angle distribution across dataset
4. What the model needs to learn: depth correction conditioned on view angle
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# COCO 17 keypoint names
COCO_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# COCO skeleton connections
COCO_CONNECTIONS = [
    (5, 6),   # Shoulders
    (5, 7), (7, 9),   # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12), # Torso
    (11, 12),         # Hips
    (11, 13), (13, 15), # Left leg
    (12, 14), (14, 16), # Right leg
]


def load_training_samples(data_dir: Path, n_samples: int = 100):
    """Load training samples from NPZ files."""
    files = sorted(data_dir.glob("*.npz"))[:n_samples]

    samples = []
    for f in files:
        data = np.load(f)
        samples.append({
            'corrupted': data['corrupted'],
            'ground_truth': data['ground_truth'],
            'visibility': data['visibility'],
            'view_angle': float(data['view_angle']) if 'view_angle' in data else
                         float(data.get('azimuth', 45)),
            'sequence': str(data['sequence']),
            'frame_idx': int(data['frame_idx']),
        })

    return samples


def draw_skeleton(ax, pose, color='blue', alpha=1.0, label=None):
    """Draw 3D skeleton."""
    ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c=color, s=30, alpha=alpha)
    for i, (j1, j2) in enumerate(COCO_CONNECTIONS):
        ax.plot([pose[j1, 0], pose[j2, 0]],
               [pose[j1, 2], pose[j2, 2]],
               [pose[j1, 1], pose[j2, 1]],
               c=color, linewidth=1.5, alpha=alpha,
               label=label if i == 0 else None)


def visualize_single_sample(sample, idx=0):
    """Visualize a single training sample."""
    corrupted = sample['corrupted']
    gt = sample['ground_truth']
    visibility = sample['visibility']
    view_angle = sample['view_angle']

    # Compute depth errors
    depth_errors = corrupted[:, 2] - gt[:, 2]

    fig = plt.figure(figsize=(16, 10))

    # === Panel 1: 3D Skeleton Comparison ===
    ax1 = fig.add_subplot(231, projection='3d')
    draw_skeleton(ax1, gt, color='green', alpha=0.8, label='Ground Truth')
    draw_skeleton(ax1, corrupted, color='red', alpha=0.8, label='MediaPipe (noisy)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z (depth)')
    ax1.set_zlabel('Y')
    ax1.set_title(f'Sample {idx}: 3D Skeleton Comparison\nView Angle: {view_angle:.1f}°')
    ax1.legend()

    # === Panel 2: Top-down view (X-Z plane) - shows depth errors ===
    ax2 = fig.add_subplot(232)
    ax2.scatter(gt[:, 0], gt[:, 2], c='green', s=50, label='Ground Truth', zorder=5)
    ax2.scatter(corrupted[:, 0], corrupted[:, 2], c='red', s=50, marker='x', label='MediaPipe', zorder=5)

    # Draw arrows showing depth error
    for i in range(17):
        if abs(depth_errors[i]) > 0.01:  # Only show significant errors
            ax2.annotate('', xy=(corrupted[i, 0], corrupted[i, 2]),
                        xytext=(gt[i, 0], gt[i, 2]),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))

    ax2.set_xlabel('X (lateral)')
    ax2.set_ylabel('Z (depth)')
    ax2.set_title('Top-Down View\n(arrows show depth errors)')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # === Panel 3: Depth error per joint ===
    ax3 = fig.add_subplot(233)
    colors = ['red' if e > 0 else 'blue' for e in depth_errors]
    bars = ax3.barh(range(17), depth_errors * 100, color=colors, alpha=0.7)
    ax3.set_yticks(range(17))
    ax3.set_yticklabels([f'{i}:{COCO_NAMES[i][:6]}' for i in range(17)], fontsize=8)
    ax3.set_xlabel('Depth Error (cm)')
    ax3.set_title('Depth Error per Joint\n(red=too far, blue=too close)')
    ax3.axvline(x=0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='x')

    # === Panel 4: Visibility scores ===
    ax4 = fig.add_subplot(234)
    colors = plt.cm.RdYlGn(visibility)
    ax4.barh(range(17), visibility, color=colors)
    ax4.set_yticks(range(17))
    ax4.set_yticklabels([f'{i}:{COCO_NAMES[i][:6]}' for i in range(17)], fontsize=8)
    ax4.set_xlabel('Visibility Score')
    ax4.set_title('MediaPipe Visibility\n(green=high, red=low)')
    ax4.set_xlim(0, 1)
    ax4.grid(True, alpha=0.3, axis='x')

    # === Panel 5: What model learns ===
    ax5 = fig.add_subplot(235)
    ax5.text(0.5, 0.9, 'MODEL INPUT:', ha='center', fontsize=12, fontweight='bold',
            transform=ax5.transAxes)
    ax5.text(0.5, 0.75, f'• Noisy pose (17, 3)', ha='center', fontsize=10,
            transform=ax5.transAxes)
    ax5.text(0.5, 0.65, f'• Visibility (17,)', ha='center', fontsize=10,
            transform=ax5.transAxes)

    ax5.text(0.5, 0.5, 'MODEL PREDICTS:', ha='center', fontsize=12, fontweight='bold',
            transform=ax5.transAxes)
    ax5.text(0.5, 0.35, f'• View angle: {view_angle:.1f}° (internal)', ha='center', fontsize=10,
            transform=ax5.transAxes, color='purple')
    ax5.text(0.5, 0.25, f'• Depth correction Δz (17,)', ha='center', fontsize=10,
            transform=ax5.transAxes, color='blue')

    ax5.text(0.5, 0.1, f'Target Δz range: [{depth_errors.min()*100:.1f}, {depth_errors.max()*100:.1f}] cm',
            ha='center', fontsize=9, transform=ax5.transAxes, style='italic')

    ax5.axis('off')
    ax5.set_title('Training Objective')

    # === Panel 6: Error statistics ===
    ax6 = fig.add_subplot(236)
    ax6.text(0.5, 0.85, 'DEPTH ERROR STATS:', ha='center', fontsize=12, fontweight='bold',
            transform=ax6.transAxes)
    ax6.text(0.5, 0.70, f'Mean: {np.mean(depth_errors)*100:.2f} cm', ha='center', fontsize=10,
            transform=ax6.transAxes)
    ax6.text(0.5, 0.58, f'Std: {np.std(depth_errors)*100:.2f} cm', ha='center', fontsize=10,
            transform=ax6.transAxes)
    ax6.text(0.5, 0.46, f'Max: {np.max(np.abs(depth_errors))*100:.2f} cm', ha='center', fontsize=10,
            transform=ax6.transAxes)

    # Worst joints
    worst_idx = np.argsort(np.abs(depth_errors))[-3:][::-1]
    ax6.text(0.5, 0.30, 'Worst joints:', ha='center', fontsize=10, fontweight='bold',
            transform=ax6.transAxes)
    for i, idx in enumerate(worst_idx):
        ax6.text(0.5, 0.20 - i*0.08, f'{COCO_NAMES[idx]}: {depth_errors[idx]*100:.1f} cm',
                ha='center', fontsize=9, transform=ax6.transAxes)

    ax6.axis('off')
    ax6.set_title('Error Analysis')

    plt.tight_layout()
    return fig


def visualize_dataset_overview(samples):
    """Visualize overall dataset statistics."""

    view_angles = [s['view_angle'] for s in samples]

    # Collect all depth errors
    all_depth_errors = []
    per_joint_errors = {i: [] for i in range(17)}

    for s in samples:
        errors = s['corrupted'][:, 2] - s['ground_truth'][:, 2]
        all_depth_errors.extend(errors)
        for i in range(17):
            per_joint_errors[i].append(errors[i])

    fig = plt.figure(figsize=(16, 10))

    # === Panel 1: View angle distribution ===
    ax1 = fig.add_subplot(231)
    ax1.hist(view_angles, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('View Angle (degrees)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'View Angle Distribution\n(n={len(samples)} samples)')
    ax1.axvline(np.mean(view_angles), color='red', linestyle='--', label=f'Mean: {np.mean(view_angles):.1f}°')
    ax1.legend()

    # === Panel 2: Overall depth error distribution ===
    ax2 = fig.add_subplot(232)
    ax2.hist(np.array(all_depth_errors) * 100, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Depth Error (cm)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Depth Error Distribution\n(all joints, all samples)')
    ax2.axvline(0, color='black', linestyle='-', linewidth=2)

    # === Panel 3: Depth error per joint (box plot) ===
    ax3 = fig.add_subplot(233)
    box_data = [np.array(per_joint_errors[i]) * 100 for i in range(17)]
    bp = ax3.boxplot(box_data, vert=False, patch_artist=True)
    ax3.set_yticklabels([COCO_NAMES[i][:8] for i in range(17)], fontsize=8)
    ax3.set_xlabel('Depth Error (cm)')
    ax3.set_title('Depth Error by Joint')
    ax3.axvline(0, color='red', linestyle='--', alpha=0.5)

    # === Panel 4: Depth error vs view angle ===
    ax4 = fig.add_subplot(234)
    for s in samples[:50]:  # Plot subset
        errors = s['corrupted'][:, 2] - s['ground_truth'][:, 2]
        ax4.scatter([s['view_angle']] * 17, errors * 100, alpha=0.3, s=10)
    ax4.set_xlabel('View Angle (degrees)')
    ax4.set_ylabel('Depth Error (cm)')
    ax4.set_title('Depth Error vs View Angle\n(each dot = one joint)')
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # === Panel 5: Mean absolute error per joint ===
    ax5 = fig.add_subplot(235)
    mae_per_joint = [np.mean(np.abs(per_joint_errors[i])) * 100 for i in range(17)]
    colors = plt.cm.Reds(np.array(mae_per_joint) / max(mae_per_joint))
    ax5.barh(range(17), mae_per_joint, color=colors)
    ax5.set_yticks(range(17))
    ax5.set_yticklabels([COCO_NAMES[i][:8] for i in range(17)], fontsize=8)
    ax5.set_xlabel('Mean Absolute Error (cm)')
    ax5.set_title('MAE by Joint\n(darker = worse)')

    # === Panel 6: Training approach summary ===
    ax6 = fig.add_subplot(236)
    ax6.text(0.5, 0.95, 'TRAINING APPROACH', ha='center', fontsize=14, fontweight='bold',
            transform=ax6.transAxes)

    ax6.text(0.5, 0.80, '1. View Angle Head predicts:', ha='center', fontsize=10,
            transform=ax6.transAxes)
    ax6.text(0.5, 0.72, '   (azimuth, elevation) from pose', ha='center', fontsize=9,
            transform=ax6.transAxes, color='purple')

    ax6.text(0.5, 0.58, '2. Depth Head uses view angles to predict:', ha='center', fontsize=10,
            transform=ax6.transAxes)
    ax6.text(0.5, 0.50, '   Δz correction per joint', ha='center', fontsize=9,
            transform=ax6.transAxes, color='blue')

    ax6.text(0.5, 0.36, '3. Joint loss:', ha='center', fontsize=10,
            transform=ax6.transAxes)
    ax6.text(0.5, 0.28, '   L = L_depth + λ·L_viewangle', ha='center', fontsize=9,
            transform=ax6.transAxes, family='monospace')

    ax6.text(0.5, 0.14, '4. At inference: single forward pass', ha='center', fontsize=10,
            transform=ax6.transAxes)
    ax6.text(0.5, 0.06, '   (no camera calibration needed)', ha='center', fontsize=9,
            transform=ax6.transAxes, style='italic')

    ax6.axis('off')

    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("TRAINING APPROACH VISUALIZATION")
    print("=" * 70)

    data_dir = Path("data/training/aistpp_converted")

    if not data_dir.exists():
        print(f"ERROR: Training data not found at {data_dir}")
        return

    # Count files
    n_files = len(list(data_dir.glob("*.npz")))
    print(f"Found {n_files} training samples")

    # Load samples
    print("Loading samples...")
    samples = load_training_samples(data_dir, n_samples=min(500, n_files))
    print(f"Loaded {len(samples)} samples")

    # Visualize single samples
    print("\nGenerating single sample visualizations...")
    for i in [0, len(samples)//4, len(samples)//2]:
        fig = visualize_single_sample(samples[i], idx=i)
        filename = f"training_sample_{i}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {filename}")

    # Visualize dataset overview
    print("\nGenerating dataset overview...")
    fig = visualize_dataset_overview(samples)
    fig.savefig("training_overview.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: training_overview.png")

    print("\n" + "=" * 70)
    print("Done! Check the generated PNG files.")
    print("=" * 70)


if __name__ == "__main__":
    main()
