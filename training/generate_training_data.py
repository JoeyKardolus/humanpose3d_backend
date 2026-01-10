#!/usr/bin/env python3
"""
Generate training data from CMU Motion Capture.

Converts BVH files → OpenCap markers → Training pairs (corrupted + ground truth)
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

from bvh_to_positions import extract_all_positions, BVHJoint
from cmu_to_opencap_mapping import DIRECT_MAPPING, ESTIMATED_MARKERS


def apply_cmu_mapping(
    cmu_positions: Dict[str, np.ndarray],
    subject_height: float = 1.75
) -> Dict[str, np.ndarray]:
    """Apply CMU → OpenCap mapping to convert joint positions to markers.

    Args:
        cmu_positions: Dict of CMU joint name → 3D position
        subject_height: Subject height in meters (for scaling)

    Returns:
        Dict of OpenCap marker name → 3D position
    """
    opencap_markers = {}

    # Apply direct mappings
    for cmu_joint, opencap_marker in DIRECT_MAPPING.items():
        if cmu_joint in cmu_positions:
            opencap_markers[opencap_marker] = cmu_positions[cmu_joint]

    # Apply estimated markers
    for marker_name, info in ESTIMATED_MARKERS.items():
        method = info["method"]

        if method == "offset_from_joint":
            base_joint = info["base_joint"]
            offset = np.array(info["offset"])
            if base_joint in cmu_positions:
                opencap_markers[marker_name] = cmu_positions[base_joint] + offset

        elif method == "direct_copy":
            base_joint = info["base_joint"]
            if base_joint in cmu_positions:
                opencap_markers[marker_name] = cmu_positions[base_joint].copy()

        elif method == "midpoint":
            joints = info["joints"]
            positions = [cmu_positions[j] for j in joints if j in cmu_positions]
            if len(positions) == len(joints):
                opencap_markers[marker_name] = np.mean(positions, axis=0)

        elif method == "medial_offset":
            base_joint = info["base_joint"]
            offset_distance = info["offset_distance"]
            if base_joint in cmu_positions:
                # For now, just offset along X-axis (lateral)
                # TODO: Proper medial direction based on limb orientation
                offset = np.array([offset_distance, 0, 0])
                opencap_markers[marker_name] = cmu_positions[base_joint] + offset

        elif method == "thigh_cluster":
            base_joint = info["base_joint"]
            target_joint = info["target_joint"]
            position = info["position"]
            if base_joint in cmu_positions and target_joint in cmu_positions:
                # Interpolate along femur
                base_pos = cmu_positions[base_joint]
                target_pos = cmu_positions[target_joint]
                opencap_markers[marker_name] = base_pos + position * (target_pos - base_pos)

        elif method == "shoulder_cluster":
            base_joint = info["base_joint"]
            index = info["index"]
            if base_joint in cmu_positions:
                # Place cluster markers around shoulder
                # TODO: Better placement based on shoulder-elbow vector
                offsets = [
                    np.array([0.03, 0.02, 0]),
                    np.array([0.03, -0.02, 0]),
                    np.array([0.03, 0, 0.02]),
                ]
                opencap_markers[marker_name] = cmu_positions[base_joint] + offsets[index]

        elif method == "bell_regression":
            # Hip joint centers - use simple geometric estimate
            if "Hips" in cmu_positions:
                pelvis = cmu_positions["Hips"]
                # Bell et al. 1990: HJC ~0.36*height lateral, ~0.19*height inferior from pelvis
                if "RHJC" in marker_name:
                    opencap_markers[marker_name] = pelvis + np.array([0.36*subject_height*0.2, -0.19*subject_height*0.2, 0])
                else:  # LHJC
                    opencap_markers[marker_name] = pelvis + np.array([-0.36*subject_height*0.2, -0.19*subject_height*0.2, 0])

    return opencap_markers


def simulate_mediapipe_depth_error(
    ground_truth: np.ndarray,
    camera_angle_deg: float = 45.0,
    noise_std_mm: float = 50.0,
) -> np.ndarray:
    """Simulate MediaPipe depth estimation errors.

    Args:
        ground_truth: (num_frames, num_markers, 3) array
        camera_angle_deg: Camera viewing angle from frontal (degrees)
        noise_std_mm: Standard deviation of depth noise in millimeters

    Returns:
        Corrupted positions with realistic depth errors
    """
    corrupted = ground_truth.copy()

    # Convert to mm for easier manipulation
    noise_std = noise_std_mm / 1000.0  # Convert to meters

    # Simulate viewpoint-dependent depth error
    # When camera is at angle, depth estimation becomes less reliable
    angle_rad = np.deg2rad(camera_angle_deg)

    # Depth noise increases with distance from camera center
    # and with steeper viewing angle
    depth_noise_scale = np.sin(angle_rad) * noise_std

    # Add Gaussian noise to Z-coordinate (depth)
    num_frames, num_markers, _ = corrupted.shape
    depth_noise = np.random.randn(num_frames, num_markers) * depth_noise_scale
    corrupted[:, :, 2] += depth_noise

    # Add small noise to X,Y as well (smaller magnitude)
    xy_noise_scale = noise_std * 0.2
    xy_noise = np.random.randn(num_frames, num_markers, 2) * xy_noise_scale
    corrupted[:, :, :2] += xy_noise

    return corrupted


def convert_bvh_to_training_data(
    bvh_path: Path,
    output_dir: Path,
    num_camera_angles: int = 6,
    noise_levels: list = [30.0, 50.0, 80.0],  # mm
) -> int:
    """Convert single BVH file to multiple training examples.

    Args:
        bvh_path: Path to BVH file
        output_dir: Directory to save training pairs
        num_camera_angles: Number of camera viewpoints to simulate
        noise_levels: Different noise levels to try (mm)

    Returns:
        Number of training examples generated
    """
    # Extract 3D positions from BVH
    joints, all_positions, frame_time = extract_all_positions(bvh_path)
    num_frames = all_positions.shape[0]
    joint_names = sorted(joints.keys())

    # Convert to dict format for easier mapping
    examples_generated = 0

    # Process each frame
    for frame_idx in range(num_frames):
        # Get CMU joint positions for this frame
        cmu_positions = {}
        for joint_idx, joint_name in enumerate(joint_names):
            cmu_positions[joint_name] = all_positions[frame_idx, joint_idx]

        # Apply CMU → OpenCap mapping
        opencap_markers = apply_cmu_mapping(cmu_positions)

        # Convert to array format
        marker_names = sorted(opencap_markers.keys())
        num_markers = len(marker_names)
        ground_truth_frame = np.zeros((num_markers, 3))
        for marker_idx, marker_name in enumerate(marker_names):
            ground_truth_frame[marker_idx] = opencap_markers[marker_name]

        # Simulate multiple camera angles and noise levels
        for angle_idx in range(num_camera_angles):
            camera_angle = angle_idx * 15.0  # 0°, 15°, 30°, 45°, 60°, 75°

            for noise_std in noise_levels:
                # Add depth corruption
                corrupted_frame = simulate_mediapipe_depth_error(
                    ground_truth_frame[np.newaxis, :, :],  # Add batch dim
                    camera_angle_deg=camera_angle,
                    noise_std_mm=noise_std
                )[0]  # Remove batch dim

                # Save training pair
                # Format: {bvh_name}_frame{idx}_angle{angle}_noise{noise}.npz
                example_name = f"{bvh_path.stem}_f{frame_idx:04d}_a{int(camera_angle):02d}_n{int(noise_std):03d}"
                output_path = output_dir / f"{example_name}.npz"

                np.savez_compressed(
                    output_path,
                    corrupted=corrupted_frame,
                    ground_truth=ground_truth_frame,
                    marker_names=marker_names,
                    camera_angle=camera_angle,
                    noise_std=noise_std,
                )

                examples_generated += 1

    return examples_generated


def main():
    """Generate training data from CMU mocap dataset."""

    # Paths
    cmu_dir = Path("data/training/cmu_mocap/cmu-mocap/data")
    output_dir = Path("data/training/cmu_converted")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CMU MOCAP → TRAINING DATA CONVERSION")
    print("=" * 80)
    print()

    # Start with just Subject 001 for testing (14 sequences)
    subject_dir = cmu_dir / "001"

    if not subject_dir.exists():
        print(f"ERROR: CMU mocap not found at {subject_dir}")
        print("Run: cd data/training/cmu_mocap && git clone https://github.com/una-dinosauria/cmu-mocap.git")
        return

    # Find all BVH files in subject 001
    bvh_files = sorted(subject_dir.glob("*.bvh"))
    print(f"Found {len(bvh_files)} BVH files in Subject 001")
    print()

    # Process first 3 files for quick test
    total_examples = 0
    for bvh_path in bvh_files[:3]:
        print(f"Processing: {bvh_path.name}")
        num_examples = convert_bvh_to_training_data(
            bvh_path,
            output_dir,
            num_camera_angles=6,
            noise_levels=[30.0, 50.0, 80.0],
        )
        total_examples += num_examples
        print(f"  Generated {num_examples} training examples")
        print()

    print("=" * 80)
    print(f"✓ Generated {total_examples} training examples")
    print(f"✓ Saved to: {output_dir}")
    print()
    print("Next steps:")
    print("1. Verify training data quality")
    print("2. Process all subjects (001-143) for full dataset")
    print("3. Implement PoseFormer architecture")
    print("4. Start training on RTX 5080!")
    print("=" * 80)


if __name__ == "__main__":
    main()
