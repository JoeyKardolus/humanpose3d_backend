#!/usr/bin/env python3
"""
Convert Human3.6M dataset to training pairs for depth refinement.

Human3.6M has:
- 32 joints (3D ground truth from mocap)
- 4 camera views per action
- Real video footage

This script:
1. Loads Human3.6M 3D poses (ground truth)
2. Projects to 2D using camera parameters
3. Optionally runs MediaPipe on videos to get realistic corrupted poses
4. Creates training pairs: (corrupted, ground_truth)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import h5py
from tqdm import tqdm


# Human3.6M joint names (32 joints)
H36M_JOINTS = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck/Nose', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]


def load_h36m_poses(h5_file: Path) -> np.ndarray:
    """Load 3D poses from Human3.6M H5 file.

    Args:
        h5_file: Path to .h5 file with 3D poses

    Returns:
        poses: (num_frames, 32, 3) array of 3D joint positions
    """
    with h5py.File(h5_file, 'r') as f:
        # Human3.6M format: (num_frames, 32*3)
        poses_flat = f['3D_positions'][:]

    # Reshape to (num_frames, 32, 3)
    num_frames = poses_flat.shape[0]
    poses = poses_flat.reshape(num_frames, 32, 3)

    return poses


def project_to_2d(poses_3d: np.ndarray, camera_params: Dict) -> np.ndarray:
    """Project 3D poses to 2D using camera parameters.

    Args:
        poses_3d: (num_frames, num_joints, 3)
        camera_params: Camera intrinsics and extrinsics

    Returns:
        poses_2d: (num_frames, num_joints, 2)
    """
    # TODO: Implement camera projection
    # For now, just return dummy 2D projection
    num_frames, num_joints, _ = poses_3d.shape
    poses_2d = poses_3d[:, :, :2]  # Simple orthographic projection

    return poses_2d


def simulate_mediapipe_noise(poses_3d: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
    """Add MediaPipe-style depth noise to 3D poses.

    Args:
        poses_3d: (num_frames, num_joints, 3) ground truth poses
        noise_std: Standard deviation of depth noise (meters)

    Returns:
        corrupted: (num_frames, num_joints, 3) noisy poses
    """
    corrupted = poses_3d.copy()

    # Add more noise to depth (Z-axis)
    depth_noise = np.random.randn(*corrupted[:, :, 2].shape) * noise_std
    corrupted[:, :, 2] += depth_noise

    # Add less noise to X, Y
    xy_noise = np.random.randn(*corrupted[:, :, :2].shape) * (noise_std * 0.1)
    corrupted[:, :, :2] += xy_noise

    return corrupted


def convert_h36m_to_training_data(
    h5_file: Path,
    output_dir: Path,
    noise_levels: list = [30.0, 50.0, 80.0],  # mm
    downsample: int = 4  # Use every Nth frame
):
    """Convert Human3.6M sequence to training pairs.

    Args:
        h5_file: Path to Human3.6M .h5 file
        output_dir: Output directory for training .npz files
        noise_levels: Depth noise levels to simulate (mm)
        downsample: Use every Nth frame (Human3.6M is 50 FPS, high redundancy)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load 3D poses
    print(f"Loading {h5_file.name}...")
    poses_3d = load_h36m_poses(h5_file)
    num_frames = poses_3d.shape[0]

    print(f"  Frames: {num_frames}")
    print(f"  Joints: {poses_3d.shape[1]}")

    # Downsample frames
    poses_3d = poses_3d[::downsample]
    num_frames = poses_3d.shape[0]
    print(f"  After downsampling (every {downsample}): {num_frames} frames")

    # Generate training pairs
    examples_generated = 0
    action_name = h5_file.stem

    for frame_idx in tqdm(range(num_frames), desc="Converting"):
        ground_truth = poses_3d[frame_idx]  # (32, 3)

        # Center on pelvis (Hip is joint 0)
        pelvis = ground_truth[0]
        ground_truth_centered = ground_truth - pelvis

        for noise_std_mm in noise_levels:
            noise_std = noise_std_mm / 1000.0  # Convert mm to meters

            # Simulate MediaPipe corruption
            corrupted = simulate_mediapipe_noise(
                ground_truth_centered[np.newaxis],  # Add batch dim
                noise_std=noise_std
            )[0]  # Remove batch dim

            # Save training pair
            example_name = f"{action_name}_f{frame_idx:04d}_n{int(noise_std_mm):03d}"
            output_path = output_dir / f"{example_name}.npz"

            np.savez_compressed(
                output_path,
                corrupted=corrupted,
                ground_truth=ground_truth_centered,
                joint_names=H36M_JOINTS,
                noise_std=noise_std,
            )

            examples_generated += 1

    return examples_generated


def main():
    """Convert all Human3.6M data to training format."""

    # Paths
    h36m_dir = Path("data/human36m")
    output_dir = Path("data/training/human36m_converted")

    if not h36m_dir.exists():
        print(f"ERROR: Human3.6M data not found at {h36m_dir}")
        print("Run: bash scripts/setup_human36m.sh")
        return

    print("="*80)
    print("HUMAN3.6M → TRAINING DATA CONVERSION")
    print("="*80)
    print()

    # Find all H5 files
    h5_files = sorted(h36m_dir.glob("**/*.h5"))

    if not h5_files:
        print("No .h5 files found. Trying .txt format...")
        # Try alternative format
        h5_files = sorted(h36m_dir.glob("**/*positions*.txt"))

    if not h5_files:
        print("ERROR: No pose files found")
        print("Expected: .h5 or .txt files with 3D positions")
        return

    print(f"Found {len(h5_files)} pose files")
    print()

    # Convert first few for testing
    total_examples = 0
    for h5_file in h5_files[:3]:  # Start with 3 files
        num_examples = convert_h36m_to_training_data(h5_file, output_dir)
        total_examples += num_examples
        print(f"  Generated {num_examples} training examples")
        print()

    print("="*80)
    print(f"✓ Generated {total_examples} training examples")
    print(f"✓ Saved to: {output_dir}")
    print()
    print("Next steps:")
    print("1. Validate training data quality")
    print("2. Start training depth refinement model")
    print("="*80)


if __name__ == "__main__":
    main()
