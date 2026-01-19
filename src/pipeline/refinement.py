"""Neural refinement functions for the pose estimation pipeline.

This module contains functions for applying neural depth and joint refinement
to landmark records and joint angles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from collections import defaultdict

from src.datastream.data_stream import LandmarkRecord
from src.depth_refinement.inference import DepthRefiner
from src.joint_refinement.inference import JointRefiner

# Mapping from OpenCap/MediaPipe marker names to COCO 17 joint indices
# COCO 17: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
#          left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
#          left_knee, right_knee, left_ankle, right_ankle
OPENCAP_TO_COCO = {
    "Nose": 0,
    "LShoulder": 5,
    "RShoulder": 6,
    "LElbow": 7,
    "RElbow": 8,
    "LWrist": 9,
    "RWrist": 10,
    "LHip": 11,
    "RHip": 12,
    "LKnee": 13,
    "RKnee": 14,
    "LAnkle": 15,
    "RAnkle": 16,
}

# Foot markers that follow ankle corrections
LEFT_FOOT_MARKERS = {'LHeel', 'LBigToe', 'LSmallToe'}
RIGHT_FOOT_MARKERS = {'RHeel', 'RBigToe', 'RSmallToe'}


def _records_to_coco_arrays(records: list) -> tuple:
    """Convert landmark records to COCO format arrays.

    Returns:
        timestamps: Sorted list of timestamps
        pose_3d: (n_frames, 17, 3) array
        visibility: (n_frames, 17) array
        pose_2d: (n_frames, 17, 2) array
        frames_data: Dict mapping timestamp -> landmark -> (x, y, z, vis)
    """
    frames_data = defaultdict(dict)
    for rec in records:
        frames_data[rec.timestamp_s][rec.landmark] = (rec.x_m, rec.y_m, rec.z_m, rec.visibility)

    timestamps = sorted(frames_data.keys())
    if not timestamps:
        return [], np.array([]), np.array([]), np.array([]), {}

    n_frames = len(timestamps)
    pose_3d = np.zeros((n_frames, 17, 3), dtype=np.float32)
    visibility = np.zeros((n_frames, 17), dtype=np.float32)
    pose_2d = np.zeros((n_frames, 17, 2), dtype=np.float32)

    for fi, ts in enumerate(timestamps):
        frame_landmarks = frames_data[ts]
        for name, coco_idx in OPENCAP_TO_COCO.items():
            if name in frame_landmarks:
                x, y, z, vis = frame_landmarks[name]
                pose_3d[fi, coco_idx] = [x, y, z]
                visibility[fi, coco_idx] = vis
                pose_2d[fi, coco_idx] = [x, y]

    return timestamps, pose_3d, visibility, pose_2d, frames_data


def _transform_to_training_coords(pose_3d: np.ndarray) -> tuple:
    """Transform poses to training convention (pelvis-centered, Y-up, Z-away).

    Returns:
        pose_centered: Transformed poses
        pelvis: Original pelvis positions for un-centering
    """
    # Compute pelvis (midpoint of hips: COCO left_hip=11, right_hip=12)
    pelvis = (pose_3d[:, 11, :] + pose_3d[:, 12, :]) / 2

    # Center on pelvis
    pose_centered = pose_3d - pelvis[:, np.newaxis, :]

    # Flip Y (down->up) and Z (toward->away)
    pose_centered[:, :, 1] = -pose_centered[:, :, 1]
    pose_centered[:, :, 2] = -pose_centered[:, :, 2]

    return pose_centered, pelvis


def _transform_from_training_coords(pose_centered: np.ndarray, pelvis: np.ndarray) -> np.ndarray:
    """Transform poses back from training convention to original coordinates."""
    # Flip Y and Z back
    pose_centered[:, :, 1] = -pose_centered[:, :, 1]
    pose_centered[:, :, 2] = -pose_centered[:, :, 2]

    # Un-center (add pelvis back)
    return pose_centered + pelvis[:, np.newaxis, :]


def _update_records_with_refined_poses(
    records: list,
    timestamps: list,
    refined_poses: np.ndarray,
    corrections: np.ndarray,
) -> list:
    """Update landmark records with refined pose values."""
    timestamp_to_fi = {ts: fi for fi, ts in enumerate(timestamps)}

    new_records = []
    for rec in records:
        if rec.landmark in OPENCAP_TO_COCO:
            fi = timestamp_to_fi[rec.timestamp_s]
            coco_idx = OPENCAP_TO_COCO[rec.landmark]
            new_records.append(LandmarkRecord(
                timestamp_s=rec.timestamp_s,
                landmark=rec.landmark,
                x_m=float(refined_poses[fi, coco_idx, 0]),
                y_m=float(refined_poses[fi, coco_idx, 1]),
                z_m=float(refined_poses[fi, coco_idx, 2]),
                visibility=rec.visibility,
            ))
        elif rec.landmark in LEFT_FOOT_MARKERS:
            # Propagate LAnkle correction to left foot markers
            fi = timestamp_to_fi[rec.timestamp_s]
            ankle_correction = corrections[fi, 15]  # LAnkle = COCO index 15
            new_records.append(LandmarkRecord(
                timestamp_s=rec.timestamp_s,
                landmark=rec.landmark,
                x_m=rec.x_m + float(ankle_correction[0]),
                y_m=rec.y_m + float(ankle_correction[1]),
                z_m=rec.z_m + float(ankle_correction[2]),
                visibility=rec.visibility,
            ))
        elif rec.landmark in RIGHT_FOOT_MARKERS:
            # Propagate RAnkle correction to right foot markers
            fi = timestamp_to_fi[rec.timestamp_s]
            ankle_correction = corrections[fi, 16]  # RAnkle = COCO index 16
            new_records.append(LandmarkRecord(
                timestamp_s=rec.timestamp_s,
                landmark=rec.landmark,
                x_m=rec.x_m + float(ankle_correction[0]),
                y_m=rec.y_m + float(ankle_correction[1]),
                z_m=rec.z_m + float(ankle_correction[2]),
                visibility=rec.visibility,
            ))
        else:
            new_records.append(rec)

    return new_records


def _report_corrections(corrections: np.ndarray, visibility: np.ndarray, label: str) -> None:
    """Print correction statistics."""
    mask = visibility > 0.3
    mean_correction_xyz = np.abs(corrections[mask]).mean(axis=0)
    total_correction = np.linalg.norm(corrections[mask], axis=-1).mean()
    print(f"[{label}] Applied neural 3D refinement:")
    print(f"        Mean |correction|: X={mean_correction_xyz[0]*100:.2f}cm, "
          f"Y={mean_correction_xyz[1]*100:.2f}cm, Z={mean_correction_xyz[2]*100:.2f}cm")
    print(f"        Total 3D: {total_correction*100:.2f} cm")


def apply_neural_depth_refinement(
    records: list,
    model_path: str | Path,
) -> list:
    """Apply neural 3D pose refinement to landmark records.

    Uses a trained transformer model to correct MediaPipe pose errors
    on all three axes (X, Y, Z), with emphasis on depth (Z) corrections.

    Args:
        records: List of LandmarkRecord named tuples
        model_path: Path to trained pose refinement model

    Returns:
        Updated records with refined x, y, z coordinates
    """
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[depth] WARNING: Model not found at {model_path}, skipping refinement")
        return records

    try:
        refiner = DepthRefiner(model_path)
    except Exception as e:
        print(f"[depth] WARNING: Failed to load model: {e}")
        return records

    timestamps, pose_3d, visibility, pose_2d, _ = _records_to_coco_arrays(records)
    if not timestamps:
        return records

    # Transform to training coordinates
    pose_centered, pelvis = _transform_to_training_coords(pose_3d)

    # Apply refinement with bone locking for temporal consistency
    try:
        refined_centered = refiner.refine_sequence_with_bone_locking(
            pose_centered, visibility, pose_2d, calibration_frames=50
        )
    except Exception as e:
        print(f"[depth] WARNING: Refinement failed: {e}")
        return records

    # Transform back
    refined_poses = _transform_from_training_coords(refined_centered, pelvis)
    corrections = refined_poses - pose_3d

    _report_corrections(corrections, visibility, "depth")

    return _update_records_with_refined_poses(records, timestamps, refined_poses, corrections)


def apply_main_refiner(
    records: list,
    depth_model_path: str | Path,
    joint_model_path: str | Path,
    main_model_path: str | Path,
) -> list:
    """Apply MainRefiner (fusion of depth + joint models) for best pose refinement.

    The MainRefiner combines outputs from depth and joint constraint models
    using learned gating to produce optimal refined poses.

    Args:
        records: List of LandmarkRecord named tuples
        depth_model_path: Path to depth model checkpoint
        joint_model_path: Path to joint model checkpoint
        main_model_path: Path to main refiner checkpoint

    Returns:
        Updated records with refined coordinates
    """
    depth_model_path = Path(depth_model_path)
    joint_model_path = Path(joint_model_path)
    main_model_path = Path(main_model_path)

    # Check all models exist
    for path, name in [(depth_model_path, "depth"), (joint_model_path, "joint"), (main_model_path, "main")]:
        if not path.exists():
            print(f"[main_refiner] WARNING: {name} model not found at {path}, skipping")
            return records

    try:
        from src.main_refinement.inference import MainRefinerPipeline
        pipeline = MainRefinerPipeline(
            depth_checkpoint=depth_model_path,
            joint_checkpoint=joint_model_path,
            main_checkpoint=main_model_path,
        )
    except Exception as e:
        print(f"[main_refiner] WARNING: Failed to load pipeline: {e}")
        return records

    timestamps, pose_3d, visibility, pose_2d, _ = _records_to_coco_arrays(records)
    if not timestamps:
        return records

    n_frames = len(timestamps)
    pose_centered, pelvis = _transform_to_training_coords(pose_3d)

    # Apply refinement frame by frame
    refined_centered = np.zeros_like(pose_centered)
    print(f"[main_refiner] Refining {n_frames} frames...")

    try:
        for fi in range(n_frames):
            result = pipeline.refine(
                pose_centered[fi],
                visibility[fi],
                pose_2d[fi],
            )
            refined_centered[fi] = result['refined_pose']

            if (fi + 1) % 100 == 0:
                print(f"[main_refiner] Processed {fi + 1}/{n_frames} frames")
    except Exception as e:
        print(f"[main_refiner] WARNING: Refinement failed at frame {fi}: {e}")
        return records

    # Transform back
    refined_poses = _transform_from_training_coords(refined_centered, pelvis)
    corrections = refined_poses - pose_3d

    _report_corrections(corrections, visibility, "main_refiner")

    return _update_records_with_refined_poses(records, timestamps, refined_poses, corrections)


def apply_neural_joint_refinement(
    angle_results: dict,
    model_path: str | Path,
) -> dict:
    """Apply neural joint constraint refinement to computed angles.

    Uses a trained transformer model to refine joint angles with learned
    soft constraints from AIST++ data.

    Args:
        angle_results: Dict mapping joint names to DataFrames with angle columns
        model_path: Path to trained joint refinement model

    Returns:
        Dict with refined angle DataFrames
    """
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[joint] WARNING: Model not found at {model_path}, skipping refinement")
        return angle_results

    JOINT_ORDER = [
        'pelvis', 'hip_R', 'hip_L', 'knee_R', 'knee_L',
        'ankle_R', 'ankle_L', 'trunk', 'shoulder_R', 'shoulder_L',
        'elbow_R', 'elbow_L',
    ]

    refiner = JointRefiner(model_path)

    # Get number of frames from first available joint
    n_frames = None
    for joint in JOINT_ORDER:
        if joint in angle_results:
            n_frames = len(angle_results[joint])
            break

    if n_frames is None:
        print("[joint] WARNING: No joint angle data found")
        return angle_results

    # Extract angles into (n_frames, 12, 3) array
    angles = np.zeros((n_frames, 12, 3), dtype=np.float32)

    for i, joint in enumerate(JOINT_ORDER):
        if joint in angle_results:
            df = angle_results[joint]
            flex_col = f"{joint}_flex_deg"
            abd_col = f"{joint}_abd_deg"
            rot_col = f"{joint}_rot_deg"

            if flex_col in df.columns:
                angles[:, i, 0] = df[flex_col].values
            if abd_col in df.columns:
                angles[:, i, 1] = df[abd_col].values
            if rot_col in df.columns:
                angles[:, i, 2] = df[rot_col].values

    # Apply refinement
    refined_angles = refiner.refine_batch(angles, batch_size=64)

    # Report statistics
    delta = refined_angles - angles
    mean_delta = np.abs(delta).mean()
    max_delta = np.abs(delta).max()
    print(f"[joint] Applied neural joint constraint refinement:")
    print(f"        Mean |correction|: {mean_delta:.2f}°")
    print(f"        Max |correction|: {max_delta:.2f}°")

    # Update angle_results with refined values
    refined_results = {}
    for joint_name, df in angle_results.items():
        if joint_name in JOINT_ORDER:
            i = JOINT_ORDER.index(joint_name)
            refined_df = df.copy()

            flex_col = f"{joint_name}_flex_deg"
            abd_col = f"{joint_name}_abd_deg"
            rot_col = f"{joint_name}_rot_deg"

            if flex_col in refined_df.columns:
                refined_df[flex_col] = refined_angles[:, i, 0]
            if abd_col in refined_df.columns:
                refined_df[abd_col] = refined_angles[:, i, 1]
            if rot_col in refined_df.columns:
                refined_df[rot_col] = refined_angles[:, i, 2]

            refined_results[joint_name] = refined_df
        else:
            refined_results[joint_name] = df

    return refined_results
