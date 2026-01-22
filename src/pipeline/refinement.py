"""Neural refinement functions for the pose estimation pipeline.

This module contains functions for applying:
- Camera-space POF reconstruction (primary 3D source)
- Neural joint constraint refinement (post-augmentation)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from collections import defaultdict

from src.datastream.data_stream import LandmarkRecord
from src.joint_refinement.inference import JointRefiner
from src.pof.inference import CameraPOFInference

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


def _records_to_coco_arrays(records: list, landmarks_2d: dict = None) -> tuple:
    """Convert landmark records to COCO format arrays.

    Args:
        records: List of LandmarkRecord
        landmarks_2d: Optional dict of 2D normalized image coords {(timestamp, name): (x, y)}
                     If provided, used for pose_2d instead of world coords

    Returns:
        timestamps: Sorted list of timestamps
        pose_3d: (n_frames, 17, 3) array
        visibility: (n_frames, 17) array
        pose_2d: (n_frames, 17, 2) array - normalized image coords if landmarks_2d provided
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
                # Use actual 2D image coords if available, otherwise fall back to world X,Y
                if landmarks_2d and (ts, name) in landmarks_2d:
                    pose_2d[fi, coco_idx] = landmarks_2d[(ts, name)]
                else:
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


def apply_camera_pof_reconstruction(
    records: list,
    model_path: str | Path,
    landmarks_2d: dict,
    height_m: float,
    image_size: tuple[int, int] = None,
    is_primary_3d: bool = False,
) -> list:
    """Apply camera-space POF reconstruction to landmark records.

    Uses a trained POF model to predict limb orientations from 2D keypoints
    and reconstructs 3D poses in camera space. This is an alternative to
    the world-space depth refinement approach.

    Args:
        records: List of LandmarkRecord named tuples
        model_path: Path to trained POF model
        landmarks_2d: Dict of 2D normalized image coords from pose estimator
                     {(timestamp, landmark_name): (x, y)} where x,y are 0-1 normalized
        height_m: Subject body height in meters (for bone length estimation)
        image_size: Optional (height, width) for aspect ratio calculation
        is_primary_3d: If True, POF is the primary 3D source (e.g., RTMPose).
                      Skips correction stats and uses meter coord conversion.

    Returns:
        Updated records with reconstructed 3D coordinates
    """
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[pof] WARNING: Model not found at {model_path}, skipping POF reconstruction")
        return records

    if not landmarks_2d:
        print("[pof] WARNING: 2D landmarks required for POF reconstruction")
        return records

    try:
        pof_inference = CameraPOFInference(model_path)
    except Exception as e:
        print(f"[pof] WARNING: Failed to load POF model: {e}")
        return records

    timestamps, pose_3d, visibility, pose_2d, frames_data = _records_to_coco_arrays(records, landmarks_2d)
    if not timestamps:
        return records

    n_frames = len(timestamps)
    print(f"[pof] Reconstructing {n_frames} frames with camera-space POF...")

    # Calculate aspect ratio from image size if provided
    aspect_ratio = 16/9  # default
    if image_size is not None:
        h, w = image_size
        aspect_ratio = w / h

    # Reconstruct 3D poses from 2D keypoints using POF
    try:
        # For 2D-only estimators (RTMPose), use_meter_coords=True converts
        # normalized [0,1] 2D coords to approximate meter coordinates
        reconstructed = pof_inference.reconstruct_3d(
            pose_2d, visibility, height_m,
            use_meter_coords=is_primary_3d,
            aspect_ratio=aspect_ratio,
        )
    except Exception as e:
        print(f"[pof] WARNING: POF reconstruction failed: {e}")
        return records

    # Report statistics
    mask = visibility > 0.3
    if mask.any():
        if is_primary_3d:
            # For primary 3D (RTMPose), report reconstruction stats instead of corrections
            # Compute bone lengths to verify reconstruction quality
            from src.pof.bone_lengths import estimate_bone_lengths_array
            expected_bones = estimate_bone_lengths_array(height_m)

            # Sample: compute shoulder-hip distance (torso)
            l_torso = np.linalg.norm(
                reconstructed[:, 5] - reconstructed[:, 11], axis=-1
            ).mean()
            r_torso = np.linalg.norm(
                reconstructed[:, 6] - reconstructed[:, 12], axis=-1
            ).mean()

            pelvis_z = (reconstructed[:, 11, 2] + reconstructed[:, 12, 2]).mean() / 2
            print(f"[pof] POF 3D reconstruction (primary):")
            print(f"      Pelvis depth: {pelvis_z:.2f}m")
            print(f"      Torso length: L={l_torso*100:.1f}cm, R={r_torso*100:.1f}cm (expected ~{expected_bones[10]*100:.1f}cm)")
        else:
            # For refinement mode, report corrections from original 3D
            corrections = reconstructed - pose_3d
            mean_correction_xyz = np.abs(corrections[mask]).mean(axis=0)
            total_correction = np.linalg.norm(corrections[mask], axis=-1).mean()
            print(f"[pof] Applied camera-space POF refinement:")
            print(f"      Mean |correction|: X={mean_correction_xyz[0]*100:.2f}cm, "
                  f"Y={mean_correction_xyz[1]*100:.2f}cm, Z={mean_correction_xyz[2]*100:.2f}cm")
            print(f"      Total 3D: {total_correction*100:.2f} cm")

    # For primary 3D mode, corrections are not meaningful (comparing against placeholder zeros)
    # Create a dummy corrections array
    corrections = reconstructed - pose_3d

    return _update_records_with_refined_poses(records, timestamps, reconstructed, corrections)
