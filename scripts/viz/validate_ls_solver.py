#!/usr/bin/env python
"""Validate the least-squares solver for POF reconstruction.

Tests:
1. Reprojection error: LS-solved X,Y should match input 2D (~0 by construction)
2. Compare direct FK vs LS reconstruction
3. Evaluate depth accuracy on training samples (if available)

Usage:
    uv run python scripts/viz/validate_ls_solver.py
    uv run python scripts/viz/validate_ls_solver.py --data data/training/aistpp_converted
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pof.least_squares import (
    solve_depth_least_squares_pof,
    normalize_2d_for_pof,
)
from src.pof.reconstruction import (
    reconstruct_skeleton_from_pof,
    reconstruct_skeleton_least_squares,
)
from src.pof.dataset import compute_gt_pof_from_3d_torch
from src.pof.bone_lengths import estimate_bone_lengths_array
from src.pof.constants import (
    LIMB_DEFINITIONS,
    COCO_JOINT_NAMES,
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX,
    LEFT_SHOULDER_IDX,
    RIGHT_SHOULDER_IDX,
)


def create_synthetic_pose():
    """Create a synthetic 3D pose for testing."""
    # Simple standing pose
    pose_3d = np.array([
        [0.0, -0.4, 0.0],   # 0: nose
        [-0.03, -0.45, 0.0], # 1: left_eye
        [0.03, -0.45, 0.0],  # 2: right_eye
        [-0.08, -0.4, 0.0],  # 3: left_ear
        [0.08, -0.4, 0.0],   # 4: right_ear
        [-0.15, -0.2, 0.0],  # 5: left_shoulder
        [0.15, -0.2, 0.0],   # 6: right_shoulder
        [-0.25, 0.0, 0.05],  # 7: left_elbow
        [0.25, 0.0, -0.05],  # 8: right_elbow
        [-0.25, 0.2, 0.1],   # 9: left_wrist
        [0.25, 0.2, -0.1],   # 10: right_wrist
        [-0.1, 0.3, 0.0],    # 11: left_hip
        [0.1, 0.3, 0.0],     # 12: right_hip
        [-0.1, 0.6, 0.02],   # 13: left_knee
        [0.1, 0.6, -0.02],   # 14: right_knee
        [-0.1, 0.9, 0.0],    # 15: left_ankle
        [0.1, 0.9, 0.0],     # 16: right_ankle
    ], dtype=np.float32)
    return pose_3d


def normalize_pose(pose_3d):
    """Normalize pose to pelvis-centered, unit torso scale."""
    pelvis = (pose_3d[LEFT_HIP_IDX] + pose_3d[RIGHT_HIP_IDX]) / 2
    centered = pose_3d - pelvis

    l_torso = np.linalg.norm(pose_3d[LEFT_SHOULDER_IDX] - pose_3d[LEFT_HIP_IDX])
    r_torso = np.linalg.norm(pose_3d[RIGHT_SHOULDER_IDX] - pose_3d[RIGHT_HIP_IDX])
    torso_scale = (l_torso + r_torso) / 2

    return centered / torso_scale


def test_reprojection_error():
    """Test 1: LS-solved X,Y should match input 2D."""
    print("\n=== Test 1: Reprojection Error ===")

    # Create synthetic pose
    pose_3d = create_synthetic_pose()
    pose_3d_norm = normalize_pose(pose_3d)

    # Use X,Y as "2D" (MTC insight)
    pose_2d = pose_3d_norm[:, :2]

    # Compute ground truth POF from 3D
    pose_3d_t = torch.from_numpy(pose_3d_norm).unsqueeze(0)
    gt_pof = compute_gt_pof_from_3d_torch(pose_3d_t)

    # Solve using LS
    pose_2d_t = torch.from_numpy(pose_2d).unsqueeze(0)
    solved_t = solve_depth_least_squares_pof(
        gt_pof, pose_2d_t,
        normalize_input=False,  # Already normalized
    )
    solved = solved_t.squeeze(0).numpy()

    # Check X,Y match
    xy_error = np.linalg.norm(solved[:, :2] - pose_2d, axis=-1)
    mean_xy_error = xy_error.mean()
    max_xy_error = xy_error.max()

    print(f"  Mean X,Y error: {mean_xy_error:.6f} (should be ~0)")
    print(f"  Max X,Y error:  {max_xy_error:.6f} (should be ~0)")

    # Check depth accuracy
    z_error = np.abs(solved[:, 2] - pose_3d_norm[:, 2])
    mean_z_error = z_error.mean()
    max_z_error = z_error.max()

    print(f"  Mean Z error:   {mean_z_error:.4f}")
    print(f"  Max Z error:    {max_z_error:.4f}")

    return mean_xy_error < 1e-5, mean_z_error


def test_direct_vs_ls():
    """Test 2: Compare direct FK vs LS reconstruction."""
    print("\n=== Test 2: Direct FK vs LS Comparison ===")

    # Create synthetic pose and get bone lengths
    pose_3d = create_synthetic_pose()
    pose_3d_norm = normalize_pose(pose_3d)
    pose_2d = pose_3d_norm[:, :2]

    # Compute GT POF
    pose_3d_t = torch.from_numpy(pose_3d_norm).unsqueeze(0)
    gt_pof = compute_gt_pof_from_3d_torch(pose_3d_t).squeeze(0).numpy()

    # Compute bone lengths from pose
    bone_lengths = np.zeros(14, dtype=np.float32)
    for i, (parent, child) in enumerate(LIMB_DEFINITIONS):
        bone_lengths[i] = np.linalg.norm(pose_3d_norm[child] - pose_3d_norm[parent])

    # Direct FK reconstruction
    direct_recon = reconstruct_skeleton_from_pof(
        gt_pof, bone_lengths,
        keypoints_2d=pose_2d,
        pelvis_depth=0.0,
        use_meter_coords=False,
    )

    # LS reconstruction
    ls_recon = reconstruct_skeleton_least_squares(
        gt_pof, pose_2d, bone_lengths,
        pelvis_depth=0.0,
        denormalize=False,
    )

    # Compare to GT
    direct_error = np.linalg.norm(direct_recon - pose_3d_norm, axis=-1)
    ls_error = np.linalg.norm(ls_recon - pose_3d_norm, axis=-1)

    print(f"  Direct FK MPJPE: {direct_error.mean():.4f}")
    print(f"  LS solver MPJPE: {ls_error.mean():.4f}")

    # Reprojection error comparison
    direct_reproj = np.linalg.norm(direct_recon[:, :2] - pose_2d, axis=-1)
    ls_reproj = np.linalg.norm(ls_recon[:, :2] - pose_2d, axis=-1)

    print(f"  Direct FK reproj error: {direct_reproj.mean():.6f}")
    print(f"  LS solver reproj error: {ls_reproj.mean():.6f}")

    return ls_reproj.mean() < direct_reproj.mean() + 1e-5


def test_on_training_data(data_dir: Path, num_samples: int = 100):
    """Test 3: Evaluate on training samples."""
    print(f"\n=== Test 3: Evaluation on Training Data ===")

    if not data_dir.exists():
        print(f"  Data directory not found: {data_dir}")
        print("  Skipping training data test.")
        return None

    # Find .npz files
    npz_files = list(data_dir.glob("**/*.npz"))
    if not npz_files:
        print(f"  No .npz files found in {data_dir}")
        return None

    print(f"  Found {len(npz_files)} training files")

    # Sample random files
    rng = np.random.default_rng(42)
    sample_files = rng.choice(npz_files, size=min(num_samples, len(npz_files)), replace=False)

    direct_errors = []
    ls_errors = []
    direct_reproj_errors = []
    ls_reproj_errors = []

    for npz_path in sample_files:
        try:
            data = np.load(npz_path)

            if 'ground_truth' not in data or 'corrupted' not in data:
                continue

            gt_pose = data['ground_truth']  # (17, 3)
            corrupted = data['corrupted']   # (17, 3) - MediaPipe with errors

            # Use corrupted X,Y as 2D (simulating real scenario)
            pose_2d = corrupted[:, :2]

            # Compute POF from corrupted pose (simulating model prediction)
            corrupted_t = torch.from_numpy(corrupted).unsqueeze(0)
            pred_pof = compute_gt_pof_from_3d_torch(corrupted_t).squeeze(0).numpy()

            # Compute bone lengths from GT
            bone_lengths = np.zeros(14, dtype=np.float32)
            for i, (parent, child) in enumerate(LIMB_DEFINITIONS):
                bone_lengths[i] = np.linalg.norm(gt_pose[child] - gt_pose[parent])

            # Direct FK reconstruction
            direct_recon = reconstruct_skeleton_from_pof(
                pred_pof, bone_lengths,
                keypoints_2d=pose_2d,
                pelvis_depth=0.0,
                use_meter_coords=False,
            )

            # LS reconstruction
            ls_recon = reconstruct_skeleton_least_squares(
                pred_pof, pose_2d, bone_lengths,
                pelvis_depth=0.0,
                denormalize=False,
            )

            # Compute errors vs GT
            direct_errors.append(np.linalg.norm(direct_recon - gt_pose, axis=-1).mean())
            ls_errors.append(np.linalg.norm(ls_recon - gt_pose, axis=-1).mean())

            # Reprojection errors
            direct_reproj_errors.append(
                np.linalg.norm(direct_recon[:, :2] - pose_2d, axis=-1).mean()
            )
            ls_reproj_errors.append(
                np.linalg.norm(ls_recon[:, :2] - pose_2d, axis=-1).mean()
            )

        except Exception as e:
            continue

    if not direct_errors:
        print("  Failed to process any samples")
        return None

    print(f"  Processed {len(direct_errors)} samples")
    print(f"\n  MPJPE (vs GT):")
    print(f"    Direct FK: {np.mean(direct_errors):.4f} +/- {np.std(direct_errors):.4f}")
    print(f"    LS solver: {np.mean(ls_errors):.4f} +/- {np.std(ls_errors):.4f}")

    print(f"\n  Reprojection Error (2D consistency):")
    print(f"    Direct FK: {np.mean(direct_reproj_errors):.6f} +/- {np.std(direct_reproj_errors):.6f}")
    print(f"    LS solver: {np.mean(ls_reproj_errors):.6f} +/- {np.std(ls_reproj_errors):.6f}")

    return {
        'direct_mpjpe': np.mean(direct_errors),
        'ls_mpjpe': np.mean(ls_errors),
        'direct_reproj': np.mean(direct_reproj_errors),
        'ls_reproj': np.mean(ls_reproj_errors),
    }


def visualize_comparison(output_path: Path):
    """Create visualization comparing direct FK vs LS reconstruction."""
    print(f"\n=== Creating Visualization ===")

    pose_3d = create_synthetic_pose()
    pose_3d_norm = normalize_pose(pose_3d)
    pose_2d = pose_3d_norm[:, :2]

    # Compute GT POF
    pose_3d_t = torch.from_numpy(pose_3d_norm).unsqueeze(0)
    gt_pof = compute_gt_pof_from_3d_torch(pose_3d_t).squeeze(0).numpy()

    # Compute bone lengths
    bone_lengths = np.zeros(14, dtype=np.float32)
    for i, (parent, child) in enumerate(LIMB_DEFINITIONS):
        bone_lengths[i] = np.linalg.norm(pose_3d_norm[child] - pose_3d_norm[parent])

    # Reconstructions
    direct_recon = reconstruct_skeleton_from_pof(
        gt_pof, bone_lengths,
        keypoints_2d=pose_2d,
        pelvis_depth=0.0,
        use_meter_coords=False,
    )

    ls_recon = reconstruct_skeleton_least_squares(
        gt_pof, pose_2d, bone_lengths,
        pelvis_depth=0.0,
        denormalize=False,
    )

    # Plot - FRONT VIEW (X vs Y)
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))

    # Skeleton connections
    connections = [
        (0, 5), (0, 6),      # Nose to shoulders (neck proxy)
        (5, 7), (7, 9),      # Left arm
        (6, 8), (8, 10),     # Right arm
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
        (5, 6),              # Shoulders
        (11, 12),            # Hips
        (5, 11), (6, 12),    # Torso sides
    ]

    def plot_skeleton(ax, pose, title, color='b'):
        # Front view: X horizontal, Y vertical (inverted so head is up)
        ax.scatter(pose[:, 0], -pose[:, 1], c=color, s=50, zorder=5)
        for i, j in connections:
            ax.plot([pose[i, 0], pose[j, 0]], [-pose[i, 1], -pose[j, 1]],
                   color=color, linewidth=2)
        ax.set_xlabel('X (left/right)')
        ax.set_ylabel('Y (up/down)')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

    # GT
    plot_skeleton(axes[0], pose_3d_norm, 'Ground Truth', 'g')

    # Direct FK
    direct_mpjpe = np.linalg.norm(direct_recon - pose_3d_norm, axis=-1).mean()
    plot_skeleton(axes[1], direct_recon, f'Direct FK\nMPJPE: {direct_mpjpe:.4f}', 'r')

    # LS
    ls_mpjpe = np.linalg.norm(ls_recon - pose_3d_norm, axis=-1).mean()
    plot_skeleton(axes[2], ls_recon, f'LS Solver\nMPJPE: {ls_mpjpe:.4f}', 'b')

    plt.suptitle('Front View (X vs Y) - Synthetic Pose with Perfect POF', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate LS solver for POF reconstruction")
    parser.add_argument("--data", type=str, default="data/training/aistpp_converted",
                        help="Path to training data directory")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to test")
    parser.add_argument("--output", type=str, default="data/output/ls_solver_validation.png",
                        help="Output path for visualization")
    args = parser.parse_args()

    print("=" * 60)
    print("Least-Squares Solver Validation")
    print("=" * 60)

    # Run tests
    test1_passed, test1_z_error = test_reprojection_error()
    test2_passed = test_direct_vs_ls()
    test3_results = test_on_training_data(Path(args.data), args.num_samples)

    # Create visualization
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_comparison(output_path)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Test 1 (Reprojection): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"  Test 2 (Direct vs LS): {'PASSED' if test2_passed else 'FAILED'}")
    if test3_results:
        print(f"  Test 3 (Training data):")
        print(f"    LS solver reduces reproj error by: {test3_results['direct_reproj'] - test3_results['ls_reproj']:.6f}")


if __name__ == "__main__":
    main()
