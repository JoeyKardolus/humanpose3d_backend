#!/usr/bin/env python3
"""Generate joint angle training data from AIST++ NPZ files.

This script:
1. Groups NPZ files by sequence (same video/camera)
2. Loads all frames for a sequence
3. Estimates missing markers (feet from ankles, neck from shoulders)
4. Runs Pose2Sim augmentation on the full sequence
5. Computes joint angles using validated ISB kinematics
6. Saves extended NPZ files to aistpp_joint_angles/

Usage:
    uv run python scripts/generate_joint_angle_training.py [--max-sequences N] [--workers N]
"""

import argparse
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.datastream.data_stream import LandmarkRecord, ORDER_22, write_landmark_csv
from src.markeraugmentation.gpu_config import patch_pose2sim_gpu


# COCO 17 joint names (matching AIST++ format)
COCO_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# COCO index to ORDER_22 marker name mapping
COCO_TO_MARKER = {
    0: 'Nose',
    5: 'LShoulder',
    6: 'RShoulder',
    7: 'LElbow',
    8: 'RElbow',
    9: 'LWrist',
    10: 'RWrist',
    11: 'LHip',
    12: 'RHip',
    13: 'LKnee',
    14: 'RKnee',
    15: 'LAnkle',
    16: 'RAnkle',
}


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm


def compute_pose_scale(pose: np.ndarray) -> float:
    """Compute torso scale (hip-to-shoulder distance) for scale-aware calculations."""
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12

    left_torso = np.linalg.norm(pose[L_SHOULDER] - pose[L_HIP])
    right_torso = np.linalg.norm(pose[R_SHOULDER] - pose[R_HIP])
    torso_scale = (left_torso + right_torso) / 2

    if torso_scale < 0.01:
        return 1.0  # Fallback
    return torso_scale


def estimate_feet_from_pose(pose: np.ndarray) -> dict:
    """Estimate heel and toe positions from COCO 17 pose.

    Args:
        pose: (17, 3) COCO keypoints (may be scale-normalized)

    Returns:
        Dict with estimated marker positions

    Note: Foot dimensions are scaled relative to the pose's torso size,
    so this works correctly with both real-scale and normalized poses.
    """
    L_HIP, R_HIP = 11, 12
    L_ANKLE, R_ANKLE = 15, 16
    L_SHOULDER, R_SHOULDER = 5, 6

    result = {}

    # Right direction (left to right hip)
    right_vec = normalize(pose[R_HIP] - pose[L_HIP])

    # Up direction FROM BODY (hip center to shoulder center)
    # This handles poses with variable orientation after body frame alignment
    hip_center = (pose[L_HIP] + pose[R_HIP]) / 2
    shoulder_center = (pose[L_SHOULDER] + pose[R_SHOULDER]) / 2
    up_vec = normalize(shoulder_center - hip_center)

    # Forward direction (perpendicular to right and up)
    forward_vec = normalize(np.cross(up_vec, right_vec))
    # Re-orthogonalize right to ensure orthonormal frame
    right_vec = normalize(np.cross(forward_vec, up_vec))

    # === SCALE-AWARE FOOT GEOMETRY ===
    # Base values are for typical human scale (torso ~0.5m)
    # Scale them relative to actual pose torso size
    pose_scale = compute_pose_scale(pose)
    REFERENCE_TORSO = 0.5  # Typical human torso in meters

    scale_factor = pose_scale / REFERENCE_TORSO

    # Foot geometry (scaled to match pose)
    HEEL_BACK = 0.05 * scale_factor
    HEEL_DOWN = 0.03 * scale_factor
    FOOT_LENGTH = 0.20 * scale_factor
    TOE_LATERAL = 0.03 * scale_factor

    for ankle_idx, prefix, lateral_sign in [
        (R_ANKLE, 'R', 1.0),
        (L_ANKLE, 'L', -1.0),
    ]:
        ankle = pose[ankle_idx]

        # Heel: behind and below ankle
        heel = ankle - forward_vec * HEEL_BACK - up_vec * HEEL_DOWN
        result[f'{prefix}Heel'] = heel

        # Big toe: in front of heel
        big_toe = heel + forward_vec * FOOT_LENGTH
        result[f'{prefix}BigToe'] = big_toe

        # Small toe: lateral to big toe
        lateral_vec = right_vec * lateral_sign
        small_toe = big_toe + lateral_vec * TOE_LATERAL
        result[f'{prefix}SmallToe'] = small_toe

    return result


def estimate_all_markers(pose: np.ndarray, visibility: np.ndarray) -> dict:
    """Estimate all ORDER_22 markers from COCO 17 pose.

    Note: Works with both real-scale and normalized poses by scaling
    derived marker offsets relative to the pose's torso size.
    """
    markers = {}

    # Direct mappings from COCO
    for coco_idx, marker_name in COCO_TO_MARKER.items():
        markers[marker_name] = pose[coco_idx]

    # Derived markers
    markers['Neck'] = (pose[5] + pose[6]) / 2
    markers['Hip'] = (pose[11] + pose[12]) / 2

    # Scale-aware head offset
    pose_scale = compute_pose_scale(pose)
    REFERENCE_TORSO = 0.5  # Typical human torso in meters
    scale_factor = pose_scale / REFERENCE_TORSO

    # Head: extrapolate from nose and neck (scaled offset)
    nose = pose[0]
    neck = markers['Neck']
    nose_to_head = normalize(nose - neck)
    HEAD_OFFSET = 0.15 * scale_factor  # ~15cm for typical human
    markers['Head'] = nose + nose_to_head * HEAD_OFFSET

    # Feet (already scale-aware)
    feet = estimate_feet_from_pose(pose)
    markers.update(feet)

    return markers


def parse_sequence_name(filename: str) -> Tuple[str, int]:
    """Extract sequence name and frame index from NPZ filename.

    Example: gBR_sBM_cAll_d04_mBR0_ch01_c01_f000000.npz
    Returns: ('gBR_sBM_cAll_d04_mBR0_ch01_c01', 0)
    """
    match = re.match(r'(.+)_f(\d+)\.npz$', filename)
    if match:
        return match.group(1), int(match.group(2))
    return filename.replace('.npz', ''), 0


def group_files_by_sequence(npz_files: List[Path]) -> Dict[str, List[Tuple[Path, int]]]:
    """Group NPZ files by sequence name."""
    groups = defaultdict(list)
    for path in npz_files:
        seq_name, frame_idx = parse_sequence_name(path.name)
        groups[seq_name].append((path, frame_idx))

    # Sort each group by frame index
    for seq_name in groups:
        groups[seq_name].sort(key=lambda x: x[1])

    return dict(groups)


def sequence_to_trc(
    frames: List[Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    fps: float = 30.0,
) -> Path:
    """Convert sequence of COCO 17 poses to TRC file.

    Args:
        frames: List of (pose, visibility) tuples, each pose is (17, 3)
        output_path: Path for output TRC file
        fps: Frame rate

    Returns:
        Path to created TRC file
    """
    from src.datastream.data_stream import csv_to_trc_strict

    records = []

    for frame_idx, (pose, visibility) in enumerate(frames):
        timestamp = frame_idx / fps
        markers = estimate_all_markers(pose, visibility)

        for marker_name in ORDER_22:
            if marker_name in markers:
                pos = markers[marker_name]
                # Default visibility for estimated markers
                vis = 0.5
                # Use actual visibility for direct COCO mappings
                for coco_idx, name in COCO_TO_MARKER.items():
                    if name == marker_name:
                        vis = visibility[coco_idx]
                        break

                records.append(LandmarkRecord(
                    timestamp_s=timestamp,
                    landmark=marker_name,
                    x_m=float(pos[0]),
                    y_m=float(pos[1]),
                    z_m=float(pos[2]),
                    visibility=float(vis),
                ))

    # Write CSV
    csv_path = output_path.with_suffix('.csv')
    write_landmark_csv(csv_path, records)

    # Convert to TRC
    trc_path = output_path.with_suffix('.trc')
    csv_to_trc_strict(csv_path, trc_path, ORDER_22)

    # Clean up CSV
    csv_path.unlink()

    return trc_path


def estimate_height_from_poses(frames: list) -> float:
    """Estimate subject height from pose sequence for Pose2Sim config.

    For scale-normalized poses (torso=1.0), this returns ~3.4 (scaled height).
    For real-scale poses (torso~0.5m), this returns ~1.7m.
    """
    # Typical human ratio: height / torso_length ≈ 3.4
    TYPICAL_HEIGHT_TO_TORSO_RATIO = 3.4

    torso_scales = []
    for pose, _ in frames:
        scale = compute_pose_scale(pose)
        if scale > 0.01:
            torso_scales.append(scale)

    if not torso_scales:
        return 1.7  # Default

    avg_torso = np.mean(torso_scales)
    return float(avg_torso * TYPICAL_HEIGHT_TO_TORSO_RATIO)


def run_augmentation(trc_path: Path, output_dir: Path, estimated_height: float = 1.7) -> Path:
    """Run Pose2Sim augmentation on TRC file.

    Note: Height is used for Pose2Sim config but doesn't affect
    marker augmentation LSTM output significantly. We estimate it
    from pose scale for consistency.
    """
    from src.markeraugmentation.markeraugmentation import run_pose2sim_augment

    # Mass estimation from height (BMI ~22 assumption)
    estimated_mass = 22.0 * (estimated_height ** 2)

    augmented_path = run_pose2sim_augment(
        trc_path=trc_path,
        out_dir=output_dir,
        height=estimated_height,
        mass=estimated_mass,
        age=25,
        sex='male',
        augmentation_cycles=5,
    )

    return augmented_path


def compute_angles_from_trc(trc_path: Path) -> Dict[str, np.ndarray]:
    """Compute joint angles from augmented TRC file.

    Returns:
        Dict mapping joint name to (n_frames, 3) angle array
    """
    from src.kinematics.comprehensive_joint_angles import compute_all_joint_angles

    angles = compute_all_joint_angles(
        trc_path,
        smooth_window=5,        # Light smoothing to reduce gimbal lock noise
        unwrap=True,            # Remove 360° discontinuities (CRITICAL for training!)
        zero_mode='first_frame',
        verbose=False,
    )

    # Convert DataFrames to numpy arrays
    result = {}
    for joint_name, df in angles.items():
        result[joint_name] = df.values.astype(np.float32)

    return result


def angles_to_array(angles: Dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
    """Extract single frame angles as (12, 3) array.

    Note: DataFrame has columns [time_s, flex, abd, rot], so we skip column 0.
    Elbow only has 1 DOF (flex), so we pad with zeros for abd/rot.
    Pelvis angles are wrapped to ±180° to prevent unbounded accumulation.
    """
    joint_order = [
        'pelvis', 'hip_R', 'hip_L', 'knee_R', 'knee_L',
        'ankle_R', 'ankle_L', 'trunk', 'shoulder_R', 'shoulder_L',
        'elbow_R', 'elbow_L',
    ]

    result = np.zeros((12, 3), dtype=np.float32)

    for i, joint in enumerate(joint_order):
        if joint in angles:
            joint_angles = angles[joint]
            if frame_idx < len(joint_angles):
                vals = joint_angles[frame_idx]
                # Skip column 0 (time_s), take angle columns
                angle_vals = vals[1:]

                # Handle joints with fewer than 3 DOFs (e.g., elbow has only flex)
                n_dofs = min(len(angle_vals), 3)
                result[i, :n_dofs] = angle_vals[:n_dofs]
                # Remaining DOFs stay as zeros (already initialized)

    # Wrap ALL joint angles to ±180° to prevent unbounded accumulation from unwrap
    result = ((result + 180) % 360) - 180

    return result


def process_sequence(args: Tuple[str, List[Tuple[Path, int]], Path]) -> Tuple[int, int]:
    """Process a single sequence.

    Args:
        args: (sequence_name, list of (npz_path, frame_idx), output_dir)

    Returns:
        (success_count, fail_count)
    """
    seq_name, file_list, output_dir = args

    # Check how many outputs already exist
    existing_count = sum(
        1 for path, _ in file_list
        if (output_dir / path.name).exists()
    )

    # Skip if we already have a reasonable fraction of outputs
    # Quality filters drop 10-30% of frames, so 40% threshold catches truly incomplete
    completion_ratio = existing_count / len(file_list) if file_list else 0
    if completion_ratio >= 0.40:
        return existing_count, len(file_list) - existing_count

    try:
        # Load all frames
        frames_corrupted = []
        frames_gt = []
        original_data = []

        for npz_path, frame_idx in file_list:
            data = np.load(npz_path)
            frames_corrupted.append((data['corrupted'], data['visibility']))
            frames_gt.append((data['ground_truth'], data['visibility']))
            original_data.append(dict(data))

        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Estimate heights from pose scales (works with normalized poses)
            corrupted_height = estimate_height_from_poses(frames_corrupted)
            gt_height = estimate_height_from_poses(frames_gt)

            # Process corrupted sequence
            corrupted_trc = sequence_to_trc(
                frames_corrupted,
                temp_path / 'corrupted',
            )
            corrupted_aug = run_augmentation(corrupted_trc, temp_path / 'corrupted_aug', corrupted_height)
            corrupted_angles = compute_angles_from_trc(corrupted_aug)

            # Process ground truth sequence
            gt_trc = sequence_to_trc(
                frames_gt,
                temp_path / 'ground_truth',
            )
            gt_aug = run_augmentation(gt_trc, temp_path / 'gt_aug', gt_height)
            gt_angles = compute_angles_from_trc(gt_aug)

            # Save individual frame NPZ files
            success = 0
            for i, (npz_path, frame_idx) in enumerate(file_list):
                output_path = output_dir / npz_path.name

                if output_path.exists():
                    success += 1
                    continue

                try:
                    corrupted_angle_array = angles_to_array(corrupted_angles, i)
                    gt_angle_array = angles_to_array(gt_angles, i)

                    # Skip frames where GT angle computation failed (bad ground truth)
                    if np.allclose(gt_angle_array, 0):
                        continue  # GT failed - unusable

                    # Keep corrupted data even if very wrong - that's what we're training to fix!
                    # Only skip if corrupted is all zeros (computation completely failed)
                    if np.allclose(corrupted_angle_array, 0):
                        continue  # Corrupted computation failed completely

                    # Save with original data plus angles
                    # Preserve ALL fields from input NPZ (including scale fields)
                    orig = original_data[i]

                    # Build output dict with all original fields
                    output_data = {k: v for k, v in orig.items()}

                    # Add computed joint angles
                    output_data['corrupted_angles'] = corrupted_angle_array
                    output_data['ground_truth_angles'] = gt_angle_array

                    np.savez_compressed(output_path, **output_data)
                    success += 1
                except Exception as e:
                    print(f"    Error saving {npz_path.name}: {e}")

            return success, len(file_list) - success

    except Exception as e:
        print(f"  Error processing sequence {seq_name}: {e}")
        return 0, len(file_list)


def main():
    parser = argparse.ArgumentParser(description='Generate joint angle training data')
    parser.add_argument('--input-dir', type=Path,
                        default=Path('data/training/aistpp_converted'),
                        help='Input directory with NPZ files')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/training/aistpp_joint_angles'),
                        help='Output directory for extended NPZ files')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Maximum number of sequences to process')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    args = parser.parse_args()

    print("=" * 60)
    print("AIST++ Joint Angle Training Data Generator")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print()

    # Enable GPU acceleration for Pose2Sim LSTM
    patch_pose2sim_gpu()

    # Find and group input files
    input_files = sorted(args.input_dir.glob('*.npz'))
    print(f"Found {len(input_files)} input files")

    sequences = group_files_by_sequence(input_files)
    print(f"Grouped into {len(sequences)} sequences")

    if args.max_sequences:
        seq_names = list(sequences.keys())[:args.max_sequences]
        sequences = {k: sequences[k] for k in seq_names}
        print(f"Processing {len(sequences)} sequences (--max-sequences)")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare tasks
    tasks = [
        (seq_name, file_list, args.output_dir)
        for seq_name, file_list in sequences.items()
    ]

    # Process
    total_success = 0
    total_failed = 0

    if args.workers == 1:
        for i, task in enumerate(tasks):
            seq_name = task[0]
            num_frames = len(task[1])
            print(f"[{i+1}/{len(tasks)}] {seq_name} ({num_frames} frames)")

            success, failed = process_sequence(task)
            total_success += success
            total_failed += failed
    else:
        with Pool(args.workers) as pool:
            for i, (success, failed) in enumerate(pool.imap_unordered(process_sequence, tasks)):
                total_success += success
                total_failed += failed
                print(f"  [{i+1}/{len(tasks)}] +{success} samples (total: {total_success})", flush=True)

    print()
    print(f"Done! Success: {total_success}, Failed: {total_failed}")


if __name__ == '__main__':
    main()
