"""Compute lower limb joint angles using Euler decompositions.

Computes hip, knee, and ankle joint angles from augmented TRC files using
anatomically correct coordinate systems and Euler angle decompositions.

Output angles follow biomechanical conventions:
- Flexion/Extension: Sagittal plane rotation
- Abduction/Adduction: Frontal plane rotation
- Internal/External Rotation: Transverse plane rotation

Based on the approach from toevoegen/compute_lower_limb_kinematics_euler_remap.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .angle_processing import (
    clamp_biomechanical_angles,
    euler_xyz,
    median_filter_angles,
    smooth_moving_average,
    unwrap_angles_deg,
    zero_angles,
)
from .segment_coordinate_systems import femur_axes, foot_axes, pelvis_axes, tibia_axes


def read_trc(trc_path: Path) -> Tuple[Dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    """Read TRC file and extract marker coordinates.

    IMPORTANT: For augmented TRC files (Pose2Sim output), the header may list
    only 22 markers but the data contains 65 markers. This function reads ALL
    marker data columns, not just those listed in the header.

    Args:
        trc_path: Path to TRC file

    Returns:
        Tuple of:
        - marker_index: Dict mapping marker names to column indices
        - frames: Frame numbers (N,)
        - times: Time stamps in seconds (N,)
        - coords: Marker coordinates (N, num_markers, 3)
    """
    lines = trc_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Parse data to determine ACTUAL number of markers
    # (header may be incomplete for augmented files)
    data_lines = [line for line in lines[6:] if line.strip()]
    if not data_lines:
        raise ValueError(f"No data rows found in {trc_path}")

    first_data_row = data_lines[0].split("\t")
    num_data_cols = len(first_data_row)
    # Cols: Frame# + Time + (marker_x, marker_y, marker_z) * N
    # So: num_data_cols = 2 + 3 * num_markers
    num_markers_in_data = (num_data_cols - 2) // 3

    # Parse header marker names (may be incomplete!)
    header_line = lines[3].split("\t")
    marker_names = []
    k = 2  # Skip Frame# and Time columns

    while k < len(header_line):
        name = header_line[k].strip()
        if name:
            marker_names.append(name)
        k += 3

    # If data has more markers than header, use standard OpenCap augmented marker names
    if num_markers_in_data > len(marker_names):
        # These are augmented markers added by Pose2Sim (from VisualizeData.load_trc_frames)
        augmented_names = [
            "C7_study", "r_shoulder_study", "L_shoulder_study",
            "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study",
            "r_knee_study", "L_knee_study", "r_mknee_study", "L_mknee_study",
            "r_ankle_study", "L_ankle_study", "r_mankle_study", "L_mankle_study",
            "r_calc_study", "L_calc_study", "r_toe_study", "L_toe_study",
            "r_5meta_study", "L_5meta_study",
            "r_lelbow_study", "L_lelbow_study", "r_melbow_study", "L_melbow_study",
            "r_lwrist_study", "L_lwrist_study", "r_mwrist_study", "L_mwrist_study",
            "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
            "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
            "r_sh1_study", "r_sh2_study", "r_sh3_study",
            "L_sh1_study", "L_sh2_study", "L_sh3_study",
            "RHJC_study", "LHJC_study",
        ]
        num_augmented_needed = num_markers_in_data - len(marker_names)
        marker_names.extend(augmented_names[:num_augmented_needed])

    marker_index = {name: i for i, name in enumerate(marker_names)}

    num_frames = len(data_lines)

    frames = np.zeros(num_frames, dtype=int)
    times = np.zeros(num_frames, dtype=float)
    coords = np.full((num_frames, num_markers_in_data, 3), np.nan, dtype=float)

    for fi, line in enumerate(data_lines):
        cols = line.split("\t")

        try:
            frames[fi] = int(float(cols[0]))
            times[fi] = float(cols[1])
        except (ValueError, IndexError):
            continue

        for mi in range(num_markers_in_data):
            cx, cy, cz = 2 + 3 * mi, 3 + 3 * mi, 4 + 3 * mi

            if cx < len(cols) and cy < len(cols) and cz < len(cols):
                try:
                    coords[fi, mi, 0] = float(cols[cx])
                    coords[fi, mi, 1] = float(cols[cy])
                    coords[fi, mi, 2] = float(cols[cz])
                except ValueError:
                    pass  # Leave as NaN

    return marker_index, frames, times, coords


def get_marker(
    coords: np.ndarray,
    marker_index: Dict[str, int],
    frame: int,
    name: str,
) -> Optional[np.ndarray]:
    """Extract marker position for a specific frame.

    Args:
        coords: Marker coordinates array (N, M, 3)
        marker_index: Marker name to index mapping
        frame: Frame index
        name: Marker name

    Returns:
        Marker position (3,) or None if not available/invalid
    """
    if name not in marker_index:
        return None

    marker_idx = marker_index[name]
    pos = coords[frame, marker_idx]

    if not np.isfinite(pos).all():
        return None

    return pos


def compute_lower_limb_angles(
    trc_path: Path,
    side: Literal["R", "L"] = "R",
    smooth_window: int = 21,
    unwrap: bool = True,
    zero_mode: Literal["first_frame", "first_n_seconds", "global_mean"] = "first_n_seconds",
    zero_window_s: float = 0.5,
) -> pd.DataFrame:
    """Compute hip, knee, and ankle joint angles from augmented TRC.

    Args:
        trc_path: Path to augmented TRC file (_LSTM.trc or _LSTM_fixed.trc)
        side: "R" or "L" for right/left leg
        smooth_window: Moving average window size (0 or 1 = no smoothing, odd number recommended)
        unwrap: Remove 360° discontinuities
        zero_mode: Reference configuration for zeroing angles
        zero_window_s: Time window for "first_n_seconds" mode

    Returns:
        DataFrame with columns:
        - time_s: Time stamps
        - hip_flex_deg, hip_abd_deg, hip_rot_deg
        - knee_flex_deg, knee_abd_deg, knee_rot_deg
        - ankle_flex_deg, ankle_abd_deg, ankle_rot_deg

    Raises:
        FileNotFoundError: If TRC file doesn't exist
        KeyError: If required markers are missing
    """
    if not trc_path.exists():
        raise FileNotFoundError(f"TRC file not found: {trc_path}")

    # Read TRC
    marker_idx, frames, times, coords = read_trc(trc_path)

    # Apply smoothing to all marker coordinates
    if smooth_window and smooth_window > 1:
        num_frames, num_markers, _ = coords.shape
        smoothed = np.empty_like(coords)

        for mi in range(num_markers):
            for axis in range(3):
                smoothed[:, mi, axis] = smooth_moving_average(
                    coords[:, mi, axis],
                    smooth_window
                )

        coords = smoothed

    # Define required markers for this side with fallback names
    # Pose2Sim may use different naming conventions (_study suffix or not)
    if side == "R":
        marker_candidates = {
            "asis_r": ["r.ASIS_study", "r.ASIS", "RASIS", "R_ASIS"],
            "asis_l": ["L.ASIS_study", "L.ASIS", "LASIS", "L_ASIS"],
            "psis_r": ["r.PSIS_study", "r.PSIS", "RPSIS", "R_PSIS"],
            "psis_l": ["L.PSIS_study", "L.PSIS", "LPSIS", "L_PSIS"],
            "hjc": ["RHJC_study", "RHJC", "R_HJC"],
            "knee_lat": ["r_knee_study", "r_knee", "RKnee_lat", "R_knee"],
            "knee_med": ["r_mknee_study", "r_mknee", "RKnee_med", "R_mknee"],
            "ankle_lat": ["r_ankle_study", "r_ankle", "RAnkle_lat", "R_ankle"],
            "ankle_med": ["r_mankle_study", "r_mankle", "RAnkle_med", "R_mankle"],
            "heel": ["r_calc_study", "r_calc", "RHeel", "R_calc"],
            "toe": ["r_toe_study", "r_toe", "RBigToe", "R_toe"],
            "meta5": ["r_5meta_study", "r_5meta", "R_5meta"],
        }
    else:  # Left side
        marker_candidates = {
            "asis_r": ["r.ASIS_study", "r.ASIS", "RASIS", "R_ASIS"],
            "asis_l": ["L.ASIS_study", "L.ASIS", "LASIS", "L_ASIS"],
            "psis_r": ["r.PSIS_study", "r.PSIS", "RPSIS", "R_PSIS"],
            "psis_l": ["L.PSIS_study", "L.PSIS", "LPSIS", "L_PSIS"],
            "hjc": ["LHJC_study", "LHJC", "L_HJC"],
            "knee_lat": ["L_knee_study", "L_knee", "LKnee_lat", "L_knee"],
            "knee_med": ["L_mknee_study", "L_mknee", "LKnee_med", "L_mknee"],
            "ankle_lat": ["L_ankle_study", "L_ankle", "LAnkle_lat", "L_ankle"],
            "ankle_med": ["L_mankle_study", "L_mankle", "LAnkle_med", "L_mankle"],
            "heel": ["L_calc_study", "L_calc", "LHeel", "L_calc"],
            "toe": ["L_toe_study", "L_toe", "LBigToe", "L_toe"],
            "meta5": ["L_5meta_study", "L_5meta", "L_5meta"],
        }

    # Resolve actual marker names (pick first match from candidates)
    markers = {}
    for key, candidates in marker_candidates.items():
        for candidate in candidates:
            if candidate in marker_idx:
                markers[key] = candidate
                break
        else:
            # No match found for this marker
            markers[key] = None

    # Check required markers exist
    missing = [key for key, name in markers.items() if name is None]
    if missing:
        available = sorted(marker_idx.keys())
        raise KeyError(
            f"Missing required markers for {side} leg: {missing}\n"
            f"Available markers: {available[:20]}... ({len(available)} total)"
        )

    num_frames = len(times)

    # Initialize angle arrays
    hip_angles = np.full((num_frames, 3), np.nan, dtype=float)
    knee_angles = np.full((num_frames, 3), np.nan, dtype=float)
    ankle_angles = np.full((num_frames, 3), np.nan, dtype=float)

    # Track previous axes for continuity
    prev_pelvis = None
    prev_femur = None
    prev_tibia = None
    prev_foot = None

    # Process each frame
    for fi in range(num_frames):
        # Extract pelvis markers
        rasis = get_marker(coords, marker_idx, fi, markers["asis_r"])
        lasis = get_marker(coords, marker_idx, fi, markers["asis_l"])
        rpsis = get_marker(coords, marker_idx, fi, markers["psis_r"])
        lpsis = get_marker(coords, marker_idx, fi, markers["psis_l"])

        # Build pelvis coordinate system
        pelvis = pelvis_axes(rasis, lasis, rpsis, lpsis, prev_pelvis)
        if pelvis is None:
            continue
        prev_pelvis = pelvis

        # Extract femur markers
        hjc = get_marker(coords, marker_idx, fi, markers["hjc"])
        knee_lat = get_marker(coords, marker_idx, fi, markers["knee_lat"])
        knee_med = get_marker(coords, marker_idx, fi, markers["knee_med"])

        # Build femur coordinate system
        femur = femur_axes(hjc, knee_lat, knee_med, pelvis[:, 2], prev_femur)
        if femur is None:
            continue
        prev_femur = femur

        # Extract tibia markers
        ankle_lat = get_marker(coords, marker_idx, fi, markers["ankle_lat"])
        ankle_med = get_marker(coords, marker_idx, fi, markers["ankle_med"])

        # Build tibia coordinate system
        tibia = tibia_axes(
            knee_lat, knee_med, ankle_lat, ankle_med,
            pelvis[:, 2], prev_tibia
        )
        if tibia is None:
            continue
        prev_tibia = tibia

        # Extract foot markers
        heel = get_marker(coords, marker_idx, fi, markers["heel"])
        toe = get_marker(coords, marker_idx, fi, markers["toe"])
        meta5 = get_marker(coords, marker_idx, fi, markers["meta5"])

        # Build foot coordinate system
        foot = foot_axes(heel, toe, meta5, pelvis[:, 2], prev_foot)
        if foot is None:
            continue
        prev_foot = foot

        # Compute relative rotation matrices
        # Hip: pelvis -> femur
        R_hip = pelvis.T @ femur

        # Knee: femur -> tibia
        R_knee = femur.T @ tibia

        # Ankle: tibia -> foot
        R_ankle = tibia.T @ foot

        # Extract Euler angles (XYZ sequence)
        hip_angles[fi] = euler_xyz(R_hip)
        knee_angles[fi] = euler_xyz(R_knee)
        ankle_angles[fi] = euler_xyz(R_ankle)

    # Remap Euler components to anatomical terms
    # For XYZ sequence: X=flex/ext, Y=abd/add, Z=rotation
    # (This matches the toevoegen script's JOINT_REMAP with flex=Z, abd=X, rot=Y for XYZ input)
    # We use the default XYZ convention where:
    # - Index 0 (X) = Flexion/Extension
    # - Index 1 (Y) = Abduction/Adduction
    # - Index 2 (Z) = Rotation

    # For left side, negate abduction to maintain R+ convention
    if side == "L":
        hip_angles[:, 1] = -hip_angles[:, 1]
        knee_angles[:, 1] = -knee_angles[:, 1]
        ankle_angles[:, 1] = -ankle_angles[:, 1]

    # Apply median filter to remove outliers from gimbal lock/bad data
    # This prevents unwrapping from amplifying isolated spikes
    hip_angles = median_filter_angles(hip_angles, window_size=5)
    knee_angles = median_filter_angles(knee_angles, window_size=5)
    ankle_angles = median_filter_angles(ankle_angles, window_size=5)

    # Unwrap angles to remove discontinuities
    if unwrap:
        hip_angles = unwrap_angles_deg(hip_angles)
        knee_angles = unwrap_angles_deg(knee_angles)
        ankle_angles = unwrap_angles_deg(ankle_angles)

    # Zero angles to reference configuration FIRST
    # This prevents clamping from being undone by large zero offsets
    hip_angles = zero_angles(hip_angles, times, zero_mode, zero_window_s)
    knee_angles = zero_angles(knee_angles, times, zero_mode, zero_window_s)
    ankle_angles = zero_angles(ankle_angles, times, zero_mode, zero_window_s)

    # THEN clamp to biomechanically reasonable ranges
    # This ensures final output respects joint limits
    hip_angles[:, 0] = clamp_biomechanical_angles(hip_angles[:, 0], "hip", "flex")
    hip_angles[:, 1] = clamp_biomechanical_angles(hip_angles[:, 1], "hip", "abd")
    hip_angles[:, 2] = clamp_biomechanical_angles(hip_angles[:, 2], "hip", "rot")

    knee_angles[:, 0] = clamp_biomechanical_angles(knee_angles[:, 0], "knee", "flex")
    knee_angles[:, 1] = clamp_biomechanical_angles(knee_angles[:, 1], "knee", "abd")
    knee_angles[:, 2] = clamp_biomechanical_angles(knee_angles[:, 2], "knee", "rot")

    ankle_angles[:, 0] = clamp_biomechanical_angles(ankle_angles[:, 0], "ankle", "flex")
    ankle_angles[:, 1] = clamp_biomechanical_angles(ankle_angles[:, 1], "ankle", "abd")
    ankle_angles[:, 2] = clamp_biomechanical_angles(ankle_angles[:, 2], "ankle", "rot")

    # Build output DataFrame
    return pd.DataFrame({
        "time_s": times,
        "hip_flex_deg": hip_angles[:, 0],
        "hip_abd_deg": hip_angles[:, 1],
        "hip_rot_deg": hip_angles[:, 2],
        "knee_flex_deg": knee_angles[:, 0],
        "knee_abd_deg": knee_angles[:, 1],
        "knee_rot_deg": knee_angles[:, 2],
        "ankle_flex_deg": ankle_angles[:, 0],
        "ankle_abd_deg": ankle_angles[:, 1],
        "ankle_rot_deg": ankle_angles[:, 2],
    })
