"""Anatomical constraints and biomechanical post-processing.

This module applies anatomical constraints to pose landmarks:
1. Constant bone lengths (arms and legs) across frames
2. Pelvis Z-depth smoothing for stability
3. Ground plane estimation and foot contact enforcement
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.datastream.data_stream import LandmarkRecord


# Bone pairs for limb length constraints (parent, child)
BONE_PAIRS = [
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
]

# Foot landmarks for ground plane estimation
FOOT_LANDMARKS = [
    "LAnkle",
    "RAnkle",
    "LHeel",
    "RHeel",
    "LBigToe",
    "RBigToe",
]


def apply_anatomical_constraints(
    records: List[LandmarkRecord],
    smooth_window: int = 21,
    ground_percentile: float = 5.0,
    foot_visibility_threshold: float = 0.5,
    ground_margin: float = 0.02,
) -> List[LandmarkRecord]:
    """Apply anatomical constraints to landmark records.

    Args:
        records: List of LandmarkRecord objects (long format)
        smooth_window: Smoothing window size for pelvis Z (frames, must be odd)
        ground_percentile: Percentile for ground plane estimation (0-100)
        foot_visibility_threshold: Minimum visibility for foot landmarks
        ground_margin: Tolerance margin for foot contact (meters)

    Returns:
        List of modified LandmarkRecord objects with constraints applied

    Raises:
        ValueError: If records are empty or missing required landmarks
    """
    if not records:
        raise ValueError("No landmark records provided")

    # Ensure smooth_window is odd and >= 1
    if smooth_window < 1:
        smooth_window = 1
    if smooth_window % 2 == 0:
        smooth_window += 1

    # Build frame structure
    frames_dict: Dict[float, Dict[str, Tuple[float, float, float, float]]] = {}
    for rec in records:
        if rec.timestamp_s not in frames_dict:
            frames_dict[rec.timestamp_s] = {}
        frames_dict[rec.timestamp_s][rec.landmark] = (
            rec.x_m,
            rec.y_m,
            rec.z_m,
            rec.visibility,
        )

    # Sort timestamps
    timestamps = sorted(frames_dict.keys())
    n_frames = len(timestamps)

    # Get all unique landmarks
    all_landmarks = set()
    for frame_data in frames_dict.values():
        all_landmarks.update(frame_data.keys())
    landmarks = sorted(all_landmarks)
    n_landmarks = len(landmarks)

    # Build index mappings
    lm_to_idx = {name: i for i, name in enumerate(landmarks)}
    ts_to_frame = {ts: i for i, ts in enumerate(timestamps)}

    # Build coordinate arrays: (frames, landmarks, 3=xyz)
    coords = np.zeros((n_frames, n_landmarks, 3), dtype=float)
    visibility = np.zeros((n_frames, n_landmarks), dtype=float)

    for ts, frame_data in frames_dict.items():
        f = ts_to_frame[ts]
        for lm_name, (x, y, z, vis) in frame_data.items():
            j = lm_to_idx[lm_name]
            coords[f, j] = [x, y, z]
            visibility[f, j] = vis

    # Validate bone pairs exist
    valid_bones = []
    for parent_name, child_name in BONE_PAIRS:
        if parent_name in lm_to_idx and child_name in lm_to_idx:
            valid_bones.append((parent_name, child_name))

    if not valid_bones:
        print(
            "[anatomical_constraints] WARNING: No valid bone pairs found, "
            "skipping bone length constraints"
        )
    else:
        # Calculate average bone lengths
        rest_lengths: Dict[Tuple[str, str], float] = {}
        for parent_name, child_name in valid_bones:
            p_idx = lm_to_idx[parent_name]
            c_idx = lm_to_idx[child_name]
            dists = []
            for f in range(n_frames):
                p = coords[f, p_idx]
                c = coords[f, c_idx]
                v = c - p
                d = np.linalg.norm(v)
                if d > 1e-6:
                    dists.append(d)
            if dists:
                rest_lengths[(parent_name, child_name)] = float(np.mean(dists))
            else:
                rest_lengths[(parent_name, child_name)] = 0.0

        # Apply bone length constraints per frame
        for f in range(n_frames):
            for parent_name, child_name in valid_bones:
                p_idx = lm_to_idx[parent_name]
                c_idx = lm_to_idx[child_name]
                p = coords[f, p_idx]
                c = coords[f, c_idx]
                v = c - p
                d = np.linalg.norm(v)
                target_len = rest_lengths.get((parent_name, child_name), 0.0)
                if d > 1e-6 and target_len > 0:
                    coords[f, c_idx] = p + v * (target_len / d)

        print(
            f"[anatomical_constraints] Applied bone length constraints to {len(valid_bones)} bone pairs"
        )

    # Pelvis Z-smoothing
    if "LHip" in lm_to_idx and "RHip" in lm_to_idx:
        left_hip_idx = lm_to_idx["LHip"]
        right_hip_idx = lm_to_idx["RHip"]

        root_z = (coords[:, left_hip_idx, 2] + coords[:, right_hip_idx, 2]) / 2.0

        pad = smooth_window // 2
        root_z_padded = np.pad(root_z, (pad, pad), mode="edge")
        kernel = np.ones(smooth_window) / smooth_window
        root_z_smooth = np.convolve(root_z_padded, kernel, mode="valid")

        delta_z = root_z_smooth - root_z
        coords[:, :, 2] += delta_z[:, None]

        print(
            f"[anatomical_constraints] Applied pelvis Z-smoothing (window={smooth_window})"
        )
    else:
        print(
            "[anatomical_constraints] WARNING: LHip/RHip not found, "
            "skipping pelvis Z-smoothing"
        )

    # Ground plane estimation and foot contact enforcement
    valid_foot_lms = [lm for lm in FOOT_LANDMARKS if lm in lm_to_idx]
    if valid_foot_lms:
        foot_indices = [lm_to_idx[name] for name in valid_foot_lms]

        all_feet_y = coords[:, foot_indices, 1].reshape(-1)
        all_feet_vis = visibility[:, foot_indices].reshape(-1)

        # Filter on visibility
        mask = all_feet_vis > foot_visibility_threshold
        if np.any(mask):
            ground_y = float(np.percentile(all_feet_y[mask], ground_percentile))
        else:
            ground_y = float(np.percentile(all_feet_y, ground_percentile))

        # Enforce foot contact within margin
        for f in range(n_frames):
            for idx in foot_indices:
                y = coords[f, idx, 1]
                if y <= ground_y + ground_margin:
                    coords[f, idx, 1] = ground_y

        print(
            f"[anatomical_constraints] Applied ground plane (y={ground_y:.6f}m) "
            f"to {len(valid_foot_lms)} foot landmarks"
        )
    else:
        print(
            "[anatomical_constraints] WARNING: No foot landmarks found, "
            "skipping ground plane estimation"
        )

    # Convert back to LandmarkRecord list
    output_records = []
    for ts, frame_data in frames_dict.items():
        f = ts_to_frame[ts]
        for lm_name in frame_data.keys():
            j = lm_to_idx[lm_name]
            x, y, z = coords[f, j]
            vis = visibility[f, j]
            output_records.append(
                LandmarkRecord(
                    timestamp_s=ts,
                    landmark=lm_name,
                    x_m=x,
                    y_m=y,
                    z_m=z,
                    visibility=vis,
                )
            )

    return output_records
