"""
Enhanced ground plane refinement with stance detection and depth propagation.

Improves depth (z-axis) accuracy by:
1. Detecting stance phases (foot contact with ground)
2. Using foot contacts as reliable depth anchors
3. Propagating depth corrections up kinematic chains (foot → knee → hip → spine)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from src.datastream.data_stream import LandmarkRecord


# Kinematic chains for depth propagation (from distal to proximal)
KINEMATIC_CHAINS = {
    "right_leg": ["RAnkle", "RKnee", "RHip"],
    "left_leg": ["LAnkle", "LKnee", "LHip"],
}

# Foot landmarks for ground contact detection
FOOT_CONTACT_MARKERS = {
    "right": ["RAnkle", "RHeel", "RBigToe"],
    "left": ["LAnkle", "LHeel", "LBigToe"],
}


def detect_stance_phases(
    coords: np.ndarray,
    lm_to_idx: Dict[str, int],
    ground_y: float,
    contact_threshold: float = 0.03,
    min_contact_frames: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Detect stance phases for left and right feet.

    Args:
        coords: Array of shape (n_frames, n_landmarks, 3)
        lm_to_idx: Mapping of landmark names to indices
        ground_y: Estimated ground plane y-coordinate
        contact_threshold: Max distance from ground to consider contact (meters)
        min_contact_frames: Minimum consecutive frames for valid stance

    Returns:
        Dict with 'left' and 'right' boolean arrays indicating stance frames
    """
    n_frames = coords.shape[0]
    stance_phases = {}

    for side, markers in FOOT_CONTACT_MARKERS.items():
        # Get indices of available markers
        available_markers = [m for m in markers if m in lm_to_idx]
        if not available_markers:
            stance_phases[side] = np.zeros(n_frames, dtype=bool)
            continue

        marker_indices = [lm_to_idx[m] for m in available_markers]

        # Check if any foot marker is near ground
        contact_frames = np.zeros(n_frames, dtype=bool)
        for frame_idx in range(n_frames):
            for marker_idx in marker_indices:
                y = coords[frame_idx, marker_idx, 1]
                if abs(y - ground_y) < contact_threshold:
                    contact_frames[frame_idx] = True
                    break

        # Apply minimum contact duration filter
        # Convert to runs of contact
        stance = np.zeros(n_frames, dtype=bool)
        in_contact = False
        contact_start = 0

        for i in range(n_frames):
            if contact_frames[i] and not in_contact:
                # Start of contact
                in_contact = True
                contact_start = i
            elif not contact_frames[i] and in_contact:
                # End of contact
                contact_duration = i - contact_start
                if contact_duration >= min_contact_frames:
                    stance[contact_start:i] = True
                in_contact = False

        # Handle contact that extends to end
        if in_contact:
            contact_duration = n_frames - contact_start
            if contact_duration >= min_contact_frames:
                stance[contact_start:] = True

        stance_phases[side] = stance

    return stance_phases


def calculate_depth_offset_from_ground(
    coords: np.ndarray,
    lm_to_idx: Dict[str, int],
    ground_y: float,
    stance_phases: Dict[str, np.ndarray],
    subject_height: float = 1.78,
) -> np.ndarray:
    """
    Calculate depth offset corrections based on foot contacts during stance.

    Uses anthropometric ratios to estimate depth from known ground contacts.

    Args:
        coords: Array of shape (n_frames, n_landmarks, 3)
        lm_to_idx: Mapping of landmark names to indices
        ground_y: Estimated ground plane y-coordinate
        stance_phases: Dict with 'left'/'right' boolean stance arrays
        subject_height: Subject height in meters (for anthropometric scaling)

    Returns:
        Array of shape (n_frames,) with depth offset corrections
    """
    n_frames = coords.shape[0]
    depth_offsets = np.zeros(n_frames)

    # Anthropometric ratio: ankle height / body height ≈ 0.04-0.05
    ankle_height_ratio = 0.045
    expected_ankle_height = subject_height * ankle_height_ratio

    for side in ["left", "right"]:
        ankle_marker = "LAnkle" if side == "left" else "RAnkle"
        if ankle_marker not in lm_to_idx:
            continue

        ankle_idx = lm_to_idx[ankle_marker]
        stance = stance_phases[side]

        for frame_idx in range(n_frames):
            if not stance[frame_idx]:
                continue

            # During stance, ankle should be at expected height above ground
            ankle_y = coords[frame_idx, ankle_idx, 1]
            ankle_z = coords[frame_idx, ankle_idx, 2]

            # Calculate expected z-depth from y-position
            # If ankle is on ground (y ≈ ground_y), use that as reference
            y_above_ground = ankle_y - ground_y

            # Depth error: difference between expected and actual height
            height_error = y_above_ground - expected_ankle_height

            # Convert height error to depth correction
            # Simple heuristic: depth errors cause apparent height changes
            # Correct depth proportionally to height error
            if abs(height_error) > 0.01:  # 1cm threshold
                depth_offsets[frame_idx] = height_error * 0.5  # Conservative correction

    # Smooth depth offsets temporally
    if np.any(depth_offsets != 0):
        # Use median filter to remove outliers
        window_size = 5
        depth_offsets = uniform_filter1d(depth_offsets, size=window_size, mode="nearest")

    return depth_offsets


def propagate_depth_corrections(
    coords: np.ndarray,
    lm_to_idx: Dict[str, int],
    stance_phases: Dict[str, np.ndarray],
    depth_offsets: np.ndarray,
    propagation_weight: float = 0.7,
) -> np.ndarray:
    """
    Propagate depth corrections from feet up kinematic chains.

    Args:
        coords: Array of shape (n_frames, n_landmarks, 3)
        lm_to_idx: Mapping of landmark names to indices
        stance_phases: Dict with 'left'/'right' boolean stance arrays
        depth_offsets: Array of shape (n_frames,) with base depth corrections
        propagation_weight: Weight decay for propagation up chain (0-1)

    Returns:
        Modified coords array with depth corrections applied
    """
    n_frames, n_landmarks, _ = coords.shape
    corrected_coords = coords.copy()

    # Apply depth offsets during stance, propagating up kinematic chains
    for side, chain in KINEMATIC_CHAINS.items():
        stance_key = side.split("_")[0]  # "right_leg" → "right"
        if stance_key not in stance_phases:
            continue

        stance = stance_phases[stance_key]

        # Get chain indices
        chain_indices = [lm_to_idx[marker] for marker in chain if marker in lm_to_idx]
        if len(chain_indices) < 2:
            continue

        # Propagate depth corrections up the chain
        for frame_idx in range(n_frames):
            if not stance[frame_idx]:
                continue

            base_correction = depth_offsets[frame_idx]
            if abs(base_correction) < 0.001:  # Skip negligible corrections
                continue

            # Apply with decreasing weight up the chain
            for i, marker_idx in enumerate(chain_indices):
                weight = propagation_weight ** i  # Decay: 1.0 → 0.7 → 0.49 → ...
                corrected_coords[frame_idx, marker_idx, 2] += base_correction * weight

    return corrected_coords


def apply_enhanced_ground_plane_refinement(
    records: List[LandmarkRecord],
    ground_y: float | None = None,
    subject_height: float = 1.78,
    contact_threshold: float = 0.03,
    min_contact_frames: int = 3,
    propagation_weight: float = 0.7,
) -> List[LandmarkRecord]:
    """
    Apply enhanced ground plane refinement with stance detection and depth propagation.

    Args:
        records: List of landmark records
        ground_y: Pre-computed ground plane (if None, will be estimated)
        subject_height: Subject height for anthropometric scaling
        contact_threshold: Max distance from ground for contact detection
        min_contact_frames: Minimum stance duration
        propagation_weight: Weight decay for depth propagation up kinematic chain

    Returns:
        List of corrected landmark records
    """
    if not records:
        return records

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

    # Build coordinate arrays
    coords = np.zeros((n_frames, n_landmarks, 3), dtype=float)
    visibility = np.zeros((n_frames, n_landmarks), dtype=float)

    for ts, frame_data in frames_dict.items():
        f = ts_to_frame[ts]
        for lm_name, (x, y, z, vis) in frame_data.items():
            j = lm_to_idx[lm_name]
            coords[f, j] = [x, y, z]
            visibility[f, j] = vis

    # Estimate ground plane if not provided
    if ground_y is None:
        foot_markers = ["LAnkle", "RAnkle", "LHeel", "RHeel", "LBigToe", "RBigToe"]
        available_foot_markers = [m for m in foot_markers if m in lm_to_idx]

        if available_foot_markers:
            foot_indices = [lm_to_idx[m] for m in available_foot_markers]
            all_feet_y = coords[:, foot_indices, 1].reshape(-1)
            all_feet_y = all_feet_y[~np.isnan(all_feet_y)]  # Filter NaNs
            if len(all_feet_y) > 0:
                ground_y = float(np.percentile(all_feet_y, 5.0))
            else:
                ground_y = 0.0
        else:
            ground_y = 0.0

    print(f"[ground_plane_refinement] Ground plane estimated at y={ground_y:.4f}m")

    # Detect stance phases
    stance_phases = detect_stance_phases(
        coords, lm_to_idx, ground_y, contact_threshold, min_contact_frames
    )

    left_stance_count = np.sum(stance_phases.get("left", np.array([])))
    right_stance_count = np.sum(stance_phases.get("right", np.array([])))
    print(
        f"[ground_plane_refinement] Detected stance: "
        f"left={left_stance_count}/{n_frames} frames, "
        f"right={right_stance_count}/{n_frames} frames"
    )

    # Calculate depth offsets from ground contacts
    depth_offsets = calculate_depth_offset_from_ground(
        coords, lm_to_idx, ground_y, stance_phases, subject_height
    )

    corrections_count = np.sum(np.abs(depth_offsets) > 0.001)
    if corrections_count > 0:
        print(
            f"[ground_plane_refinement] Applying depth corrections to "
            f"{corrections_count}/{n_frames} frames"
        )

        # Propagate depth corrections up kinematic chains
        coords = propagate_depth_corrections(
            coords, lm_to_idx, stance_phases, depth_offsets, propagation_weight
        )
    else:
        print("[ground_plane_refinement] No significant depth corrections needed")

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
