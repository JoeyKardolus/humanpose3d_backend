"""Compute upper body joint angles using anatomical coordinate systems.

Computes trunk, shoulder, and elbow joint angles from augmented TRC files using
biomechanically accurate coordinate systems and Euler/geometric decompositions.

Output angles follow biomechanical conventions:
- Trunk: Flexion/Extension, Lateral Flexion, Axial Rotation (XYZ sequence)
- Shoulder: Exo/Endorotation, Flexion/Extension, Abduction/Adduction (ZXY sequence)
- Elbow: Flexion/Extension (geometric, 0° = extended, 180° = flexed)

Based on the throwing kinematics approach from toevoegen/throwing_kinematics_definitive_trc_only2.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .angle_processing import (
    clamp_biomechanical_angles,
    euler_xyz,
    euler_zxy,
    geometric_elbow_flexion,
    median_filter_angles,
    smooth_moving_average,
    unwrap_angles_deg,
    wrap_to_180,
    zero_angles,
)
from .joint_angles_euler import get_marker, read_trc
from .segment_coordinate_systems import (
    forearm_axes,
    humerus_axes,
    normalize,
    pelvis_axes,
    trunk_axes,
)


def compute_upper_body_angles(
    trc_path: Path,
    side: Literal["R", "L"] = "R",
    smooth_window: int = 9,
    unwrap: bool = True,
    zero_mode: Literal["first_frame", "first_n_seconds", "global_mean"] = "first_n_seconds",
    zero_window_s: float = 0.5,
) -> pd.DataFrame:
    """Compute trunk, shoulder, and elbow joint angles from augmented TRC.

    Args:
        trc_path: Path to augmented TRC file (_LSTM.trc or _LSTM_complete.trc)
        side: "R" or "L" for right/left arm
        smooth_window: Moving average window size (0 or 1 = no smoothing)
        unwrap: Remove 360° discontinuities
        zero_mode: Reference configuration for zeroing angles
        zero_window_s: Time window for "first_n_seconds" mode

    Returns:
        DataFrame with columns:
        - time_s: Time stamps
        - trunk_flex_deg, trunk_lateral_deg, trunk_rot_deg
        - shoulder_exo_deg, shoulder_flex_deg, shoulder_abd_deg
        - elbow_flex_deg

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

    # Define required markers with fallback names
    # Both sides need pelvis/hip markers for trunk reference
    marker_candidates = {
        # Pelvis markers (optional - fallback to basic hip markers)
        "asis_r": ["r.ASIS_study", "r.ASIS", "RASIS", "R_ASIS", "RHip"],
        "asis_l": ["L.ASIS_study", "L.ASIS", "LASIS", "L_ASIS", "LHip"],
        "psis_r": ["r.PSIS_study", "r.PSIS", "RPSIS", "R_PSIS"],
        "psis_l": ["L.PSIS_study", "L.PSIS", "LPSIS", "L_PSIS"],
        "c7": ["C7_study", "C7", "Neck"],
    }

    # Add side-specific arm markers
    # Prioritize augmented markers (_study suffix) as they have better data
    if side == "R":
        marker_candidates.update({
            "shoulder_r": ["r_shoulder_study", "RShoulder", "r_shoulder", "R_shoulder"],
            "shoulder_l": ["L_shoulder_study", "LShoulder", "L_shoulder"],
            "elbow": ["r_lelbow_study", "RElbow", "r_elbow", "R_elbow"],
            "wrist": ["r_lwrist_study", "RWrist", "r_wrist", "R_wrist"],
        })
    else:  # Left side
        marker_candidates.update({
            "shoulder_r": ["r_shoulder_study", "RShoulder", "r_shoulder", "R_shoulder"],
            "shoulder_l": ["L_shoulder_study", "LShoulder", "L_shoulder"],
            "elbow": ["L_lelbow_study", "LElbow", "L_elbow"],
            "wrist": ["L_lwrist_study", "LWrist", "L_wrist"],
        })

    # Resolve actual marker names (prioritize markers with valid data)
    markers = {}
    for key, candidates in marker_candidates.items():
        for candidate in candidates:
            if candidate in marker_idx:
                # Check if this marker has any valid data
                idx = marker_idx[candidate]
                # Check first 10 frames for valid data
                has_data = False
                for fi in range(min(10, coords.shape[0])):
                    if np.isfinite(coords[fi, idx]).all():
                        has_data = True
                        break

                if has_data:
                    markers[key] = candidate
                    break
        else:
            # No match found with valid data
            markers[key] = None

    # Check required markers exist (PSIS and wrist are optional)
    required_markers = ["asis_r", "asis_l", "c7", "shoulder_r", "shoulder_l", "elbow"]
    missing = [key for key in required_markers if markers.get(key) is None]
    if missing:
        available = sorted(marker_idx.keys())
        raise KeyError(
            f"Missing required markers for {side} arm: {missing}\n"
            f"Available markers: {available[:20]}... ({len(available)} total)"
        )

    # Warn if wrist is missing (elbow angles will be NaN)
    if markers.get("wrist") is None:
        print(f"[WARNING] Wrist marker not found for {side} arm - elbow angles will be unavailable")

    num_frames = len(times)

    # Initialize angle arrays
    trunk_angles = np.full((num_frames, 3), np.nan, dtype=float)
    shoulder_angles = np.full((num_frames, 3), np.nan, dtype=float)
    elbow_flexion = np.full(num_frames, np.nan, dtype=float)

    # Track previous axes for continuity
    prev_pelvis = None
    prev_trunk = None
    prev_humerus = None

    # Process each frame
    for fi in range(num_frames):
        # Extract pelvis/hip markers (needed for trunk reference)
        rasis = get_marker(coords, marker_idx, fi, markers["asis_r"])
        lasis = get_marker(coords, marker_idx, fi, markers["asis_l"])
        rpsis = get_marker(coords, marker_idx, fi, markers.get("psis_r"))
        lpsis = get_marker(coords, marker_idx, fi, markers.get("psis_l"))

        # Build pelvis coordinate system if PSIS available, otherwise create simple frame
        if rpsis is not None and lpsis is not None:
            pelvis = pelvis_axes(rasis, lasis, rpsis, lpsis, prev_pelvis)
            if pelvis is None:
                continue
            prev_pelvis = pelvis
        else:
            # Fallback: use simple frame from hip markers
            if rasis is None or lasis is None:
                continue
            # Simple pelvis Z-axis from right to left hip
            pelvis_z = normalize(rasis - lasis)
            if np.isnan(pelvis_z).any():
                continue
            # Use placeholder for full pelvis (we only need Z-axis for trunk)
            pelvis = np.eye(3)
            pelvis[:, 2] = pelvis_z

        # Pelvis origin for trunk (hip midpoint)
        pelvis_origin = 0.5 * (rasis + lasis) if rasis is not None and lasis is not None else None
        if pelvis_origin is None:
            continue

        # Extract trunk markers
        c7 = get_marker(coords, marker_idx, fi, markers["c7"])
        r_shoulder = get_marker(coords, marker_idx, fi, markers["shoulder_r"])
        l_shoulder = get_marker(coords, marker_idx, fi, markers["shoulder_l"])

        # Build trunk coordinate system
        trunk = trunk_axes(c7, r_shoulder, l_shoulder, pelvis_origin, pelvis[:, 2], prev_trunk)
        if trunk is None:
            continue
        prev_trunk = trunk

        # Extract arm markers (side-specific)
        shoulder = r_shoulder if side == "R" else l_shoulder
        elbow = get_marker(coords, marker_idx, fi, markers["elbow"])
        wrist = get_marker(coords, marker_idx, fi, markers.get("wrist")) if markers.get("wrist") else None

        # Build humerus coordinate system (requires wrist for proper orientation)
        if wrist is not None:
            humerus = humerus_axes(shoulder, elbow, wrist, trunk[:, 2], prev_humerus)
            if humerus is None:
                # Skip shoulder angles if can't build humerus
                pass
            else:
                prev_humerus = humerus

                # Shoulder: trunk -> humerus
                R_shoulder = trunk.T @ humerus

                # Shoulder: ZXY (exo/endo, flex/ext, abd/add)
                shoulder_angles[fi] = euler_zxy(R_shoulder)

                # Elbow: Geometric flexion
                if shoulder is not None and elbow is not None:
                    elbow_flexion[fi] = geometric_elbow_flexion(shoulder, elbow, wrist)

        # Compute trunk angles (always possible if we got here)
        # Trunk: pelvis -> trunk
        R_trunk = pelvis.T @ trunk

        # Trunk: XYZ (flexion/extension, lateral flexion, rotation)
        trunk_angles[fi] = euler_xyz(R_trunk)

    # For left side, negate abduction/lateral angles to maintain R+ convention
    if side == "L":
        trunk_angles[:, 1] = -trunk_angles[:, 1]  # Lateral flexion
        shoulder_angles[:, 2] = -shoulder_angles[:, 2]  # Abduction

    # Apply median filter to remove outliers from gimbal lock/bad data
    trunk_angles = median_filter_angles(trunk_angles, window_size=5)
    shoulder_angles = median_filter_angles(shoulder_angles, window_size=5)
    elbow_flexion = median_filter_angles(elbow_flexion, window_size=5)

    # Unwrap angles to remove discontinuities
    if unwrap:
        trunk_angles = unwrap_angles_deg(trunk_angles)
        shoulder_angles = unwrap_angles_deg(shoulder_angles)
        # Elbow flexion is bounded [0, 180] so unwrapping not needed

    # Zero angles to reference configuration FIRST
    # This prevents clamping from being undone by large zero offsets
    trunk_angles = zero_angles(trunk_angles, times, zero_mode, zero_window_s)
    shoulder_angles = zero_angles(shoulder_angles, times, zero_mode, zero_window_s)

    # THEN clamp to biomechanically reasonable ranges
    # This ensures final output respects joint limits
    trunk_angles[:, 0] = clamp_biomechanical_angles(trunk_angles[:, 0], "trunk", "flex")
    trunk_angles[:, 1] = clamp_biomechanical_angles(trunk_angles[:, 1], "trunk", "abd")
    trunk_angles[:, 2] = clamp_biomechanical_angles(trunk_angles[:, 2], "trunk", "rot")

    shoulder_angles[:, 0] = clamp_biomechanical_angles(shoulder_angles[:, 0], "shoulder", "exo")
    shoulder_angles[:, 1] = clamp_biomechanical_angles(shoulder_angles[:, 1], "shoulder", "flex")
    shoulder_angles[:, 2] = clamp_biomechanical_angles(shoulder_angles[:, 2], "shoulder", "abd")

    elbow_flexion = clamp_biomechanical_angles(elbow_flexion, "elbow", "flex")

    # Zero elbow flexion
    elbow_zeroed = elbow_flexion - np.nanmean(
        elbow_flexion[times <= (times[0] + zero_window_s)]
    ) if zero_mode == "first_n_seconds" else elbow_flexion - elbow_flexion[0]

    # Build output DataFrame
    return pd.DataFrame({
        "time_s": times,
        "trunk_flex_deg": trunk_angles[:, 0],
        "trunk_lateral_deg": trunk_angles[:, 1],
        "trunk_rot_deg": trunk_angles[:, 2],
        "shoulder_exo_deg": shoulder_angles[:, 0],
        "shoulder_flex_deg": shoulder_angles[:, 1],
        "shoulder_abd_deg": shoulder_angles[:, 2],
        "elbow_flex_deg": elbow_zeroed,
    })
