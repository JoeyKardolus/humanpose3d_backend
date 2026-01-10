"""Comprehensive joint angle computation for full body using ISB coordinate systems.

Computes all joint angles (lower + upper body) using ISB-compliant anatomical
coordinate systems and Euler decompositions.

References:
- ISB Standards (Wu et al. 2002, 2005): Joint coordinate system definitions
- Grood & Suntay (1983): Joint coordinate system for knee
- MANIKIN (ECCV 2024): Biomechanically Accurate Neural IK

Output angles follow ISB biomechanical conventions:
- Flexion/Extension (+/- around X axis)
- Abduction/Adduction (+/- around Y axis)
- Internal/External Rotation (+/- around Z axis)
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
    zero_angles,
)
from .joint_angles_euler import get_marker, read_trc
from .segment_coordinate_systems import (
    femur_axes,
    foot_axes,
    forearm_axes,
    humerus_axes,
    normalize,
    pelvis_axes,
    tibia_axes,
    trunk_axes,
)


def compute_all_joint_angles(
    trc_path: Path,
    smooth_window: int = 21,
    unwrap: bool = True,
    zero_mode: Literal["first_frame", "first_n_seconds", "global_mean"] = "first_n_seconds",
    zero_window_s: float = 0.5,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Compute all joint angles (lower + upper body) from augmented TRC.

    Uses ISB-compliant segment coordinate systems and Euler decompositions
    to compute biomechanically accurate 3-DOF angles for all major joints.

    Args:
        trc_path: Path to augmented TRC file (_LSTM_complete.trc recommended)
        smooth_window: Moving average window size (0 or 1 = no smoothing)
        unwrap: Remove 360° discontinuities
        zero_mode: Reference configuration for zeroing angles
        zero_window_s: Time window for "first_n_seconds" mode
        verbose: Print progress messages

    Returns:
        Dict with DataFrames for each joint group:
        - "pelvis": pelvis_tilt, pelvis_obliquity, pelvis_rotation
        - "hip_R": hip_flex, hip_abd, hip_rot (right)
        - "hip_L": hip_flex, hip_abd, hip_rot (left)
        - "knee_R": knee_flex, knee_abd, knee_rot (right)
        - "knee_L": knee_flex, knee_abd, knee_rot (left)
        - "ankle_R": ankle_flex, ankle_abd, ankle_rot (right)
        - "ankle_L": ankle_flex, ankle_abd, ankle_rot (left)
        - "trunk": trunk_flex, trunk_lateral, trunk_rot
        - "shoulder_R": shoulder_exo, shoulder_flex, shoulder_abd (right)
        - "shoulder_L": shoulder_exo, shoulder_flex, shoulder_abd (left)
        - "elbow_R": elbow_flex (right)
        - "elbow_L": elbow_flex (left)

    Raises:
        FileNotFoundError: If TRC file doesn't exist
        KeyError: If required markers are missing
    """
    if not trc_path.exists():
        raise FileNotFoundError(f"TRC file not found: {trc_path}")

    if verbose:
        print(f"[comprehensive_angles] Reading TRC: {trc_path.name}")

    # Read TRC
    marker_idx, frames, times, coords = read_trc(trc_path)
    num_frames = len(times)

    # Apply smoothing to all marker coordinates (optional)
    if smooth_window and smooth_window > 1:
        if verbose:
            print(f"[comprehensive_angles] Applying smoothing (window={smooth_window})")

        num_markers = coords.shape[1]
        smoothed = np.empty_like(coords)

        for mi in range(num_markers):
            for axis in range(3):
                smoothed[:, mi, axis] = smooth_moving_average(
                    coords[:, mi, axis],
                    smooth_window
                )

        coords = smoothed

    # Define marker candidates with priority order
    marker_candidates = {
        # Pelvis/hip markers
        "asis_r": ["r.ASIS_study", "RASIS", "RHip"],
        "asis_l": ["L.ASIS_study", "LASIS", "LHip"],
        "psis_r": ["r.PSIS_study", "RPSIS"],
        "psis_l": ["L.PSIS_study", "LPSIS"],

        # Lower body - Right
        "hjc_r": ["RHJC_study", "RHJC"],
        "knee_lat_r": ["r_knee_study", "RKnee"],
        "knee_med_r": ["r_mknee_study"],
        "ankle_lat_r": ["r_ankle_study", "RAnkle"],
        "ankle_med_r": ["r_mankle_study"],
        "heel_r": ["r_calc_study", "RHeel"],
        "toe_r": ["r_toe_study", "RBigToe"],
        "meta5_r": ["r_5meta_study"],

        # Lower body - Left
        "hjc_l": ["LHJC_study", "LHJC"],
        "knee_lat_l": ["L_knee_study", "LKnee"],
        "knee_med_l": ["L_mknee_study"],
        "ankle_lat_l": ["L_ankle_study", "LAnkle"],
        "ankle_med_l": ["L_mankle_study"],
        "heel_l": ["L_calc_study", "LHeel"],
        "toe_l": ["L_toe_study", "LBigToe"],
        "meta5_l": ["L_5meta_study"],

        # Upper body
        "c7": ["C7_study", "Neck"],
        "shoulder_r": ["r_shoulder_study", "RShoulder"],
        "shoulder_l": ["L_shoulder_study", "LShoulder"],
        "elbow_r": ["r_lelbow_study", "RElbow"],
        "elbow_l": ["L_lelbow_study", "LElbow"],
        "wrist_r": ["r_lwrist_study", "RWrist"],
        "wrist_l": ["L_lwrist_study", "LWrist"],
    }

    # Resolve markers (find first available with valid data)
    markers = {}
    for key, candidates in marker_candidates.items():
        markers[key] = None
        for candidate in candidates:
            if candidate in marker_idx:
                idx = marker_idx[candidate]
                # Check if has valid data in first 10 frames
                for fi in range(min(10, num_frames)):
                    if np.isfinite(coords[fi, idx]).all():
                        markers[key] = candidate
                        break
                if markers[key]:
                    break

    if verbose:
        available_markers = [k for k, v in markers.items() if v is not None]
        missing_markers = [k for k, v in markers.items() if v is None]
        print(f"[comprehensive_angles] Found {len(available_markers)}/{ len(markers)} required markers")
        if missing_markers:
            print(f"[comprehensive_angles] Missing: {', '.join(missing_markers[:10])}")

    # Initialize angle arrays for all joints
    results = {}

    # Pelvis angles (global orientation)
    pelvis_angles = np.full((num_frames, 3), np.nan)

    # Lower body angles (both sides)
    hip_r_angles = np.full((num_frames, 3), np.nan)
    hip_l_angles = np.full((num_frames, 3), np.nan)
    knee_r_angles = np.full((num_frames, 3), np.nan)
    knee_l_angles = np.full((num_frames, 3), np.nan)
    ankle_r_angles = np.full((num_frames, 3), np.nan)
    ankle_l_angles = np.full((num_frames, 3), np.nan)

    # Upper body angles (both sides)
    trunk_angles = np.full((num_frames, 3), np.nan)
    shoulder_r_angles = np.full((num_frames, 3), np.nan)
    shoulder_l_angles = np.full((num_frames, 3), np.nan)
    elbow_r_flex = np.full(num_frames, np.nan)
    elbow_l_flex = np.full(num_frames, np.nan)

    # Track previous coordinate systems for continuity
    prev_pelvis = None
    prev_trunk = None
    prev_femur_r = None
    prev_femur_l = None
    prev_tibia_r = None
    prev_tibia_l = None
    prev_humerus_r = None
    prev_humerus_l = None

    if verbose:
        print(f"[comprehensive_angles] Computing angles for {num_frames} frames...")

    # Process each frame
    for fi in range(num_frames):
        # =====================================================================
        # PELVIS AND HIP JOINT CENTERS
        # =====================================================================
        rasis = get_marker(coords, marker_idx, fi, markers.get("asis_r"))
        lasis = get_marker(coords, marker_idx, fi, markers.get("asis_l"))
        rpsis = get_marker(coords, marker_idx, fi, markers.get("psis_r"))
        lpsis = get_marker(coords, marker_idx, fi, markers.get("psis_l"))

        # Build pelvis coordinate system
        if all(m is not None for m in [rasis, lasis, rpsis, lpsis]):
            pelvis = pelvis_axes(rasis, lasis, rpsis, lpsis, prev_pelvis)
            if pelvis is not None:
                prev_pelvis = pelvis

                # Pelvis angles: GLOBAL orientation using ZXY Euler sequence
                # Following reference implementation compute_pelvis_global_angles.py
                # pelvis matrix = rotation from pelvis to world (columns = pelvis axes in world)
                # ZXY Euler of pelvis gives orientation of pelvis relative to world
                # ZXY sequence returns [flex_Z, abd_X, rot_Y]:
                #   - Flex/Ext: Rotation around Z (right) axis - sagittal plane tilt
                #   - Abd/Add: Rotation around X (anterior) axis - frontal plane tilt
                #   - Rotation: Rotation around Y (superior) axis - axial rotation
                # Note: These are GLOBAL angles (pelvis orientation in world frame)
                # The zero_angles() step later will subtract mean/first frame to get
                # deviations from neutral posture
                pelvis_angles[fi] = euler_zxy(pelvis)  # Pelvis orientation in world

        # Pelvis origin for trunk
        pelvis_origin = 0.5 * (rasis + lasis) if rasis is not None and lasis is not None else None

        # =====================================================================
        # LOWER BODY - RIGHT SIDE (Hip, Knee, Ankle)
        # =====================================================================
        if pelvis is not None:
            # Right hip
            hjc_r = get_marker(coords, marker_idx, fi, markers.get("hjc_r"))
            knee_lat_r = get_marker(coords, marker_idx, fi, markers.get("knee_lat_r"))
            knee_med_r = get_marker(coords, marker_idx, fi, markers.get("knee_med_r"))
            ankle_lat_r = get_marker(coords, marker_idx, fi, markers.get("ankle_lat_r"))
            ankle_med_r = get_marker(coords, marker_idx, fi, markers.get("ankle_med_r"))
            heel_r = get_marker(coords, marker_idx, fi, markers.get("heel_r"))
            toe_r = get_marker(coords, marker_idx, fi, markers.get("toe_r"))

            # Right femur (hip -> knee)
            if all(m is not None for m in [hjc_r, knee_lat_r, knee_med_r]):
                femur_r = femur_axes(hjc_r, knee_lat_r, knee_med_r, pelvis[:, 2], prev_femur_r)
                if femur_r is not None:
                    prev_femur_r = femur_r
                    # Hip: pelvis -> femur
                    R_hip_r = pelvis.T @ femur_r
                    hip_r_angles[fi] = euler_xyz(R_hip_r)

                    # Right tibia (knee -> ankle)
                    if all(m is not None for m in [ankle_lat_r, ankle_med_r]):
                        tibia_r = tibia_axes(knee_lat_r, knee_med_r, ankle_lat_r, ankle_med_r, pelvis[:, 2], prev_tibia_r)
                        if tibia_r is not None:
                            prev_tibia_r = tibia_r
                            # Knee: femur -> tibia
                            R_knee_r = femur_r.T @ tibia_r
                            knee_r_angles[fi] = euler_xyz(R_knee_r)

                            # Right foot (ankle -> foot)
                            if all(m is not None for m in [heel_r, toe_r]):
                                meta5_r = get_marker(coords, marker_idx, fi, markers.get("meta5_r"))
                                foot_r = foot_axes(heel_r, toe_r, meta5_r, pelvis[:, 2])
                                if foot_r is not None:
                                    # Ankle: tibia -> foot
                                    R_ankle_r = tibia_r.T @ foot_r
                                    ankle_r_angles[fi] = euler_xyz(R_ankle_r)

        # =====================================================================
        # LOWER BODY - LEFT SIDE (Hip, Knee, Ankle)
        # =====================================================================
        if pelvis is not None:
            # Left hip
            hjc_l = get_marker(coords, marker_idx, fi, markers.get("hjc_l"))
            knee_lat_l = get_marker(coords, marker_idx, fi, markers.get("knee_lat_l"))
            knee_med_l = get_marker(coords, marker_idx, fi, markers.get("knee_med_l"))
            ankle_lat_l = get_marker(coords, marker_idx, fi, markers.get("ankle_lat_l"))
            ankle_med_l = get_marker(coords, marker_idx, fi, markers.get("ankle_med_l"))
            heel_l = get_marker(coords, marker_idx, fi, markers.get("heel_l"))
            toe_l = get_marker(coords, marker_idx, fi, markers.get("toe_l"))

            # Left femur (hip -> knee)
            if all(m is not None for m in [hjc_l, knee_lat_l, knee_med_l]):
                femur_l = femur_axes(hjc_l, knee_lat_l, knee_med_l, pelvis[:, 2], prev_femur_l)
                if femur_l is not None:
                    prev_femur_l = femur_l
                    # Hip: pelvis -> femur
                    R_hip_l = pelvis.T @ femur_l
                    hip_l_angles[fi] = euler_xyz(R_hip_l)

                    # Left tibia (knee -> ankle)
                    if all(m is not None for m in [ankle_lat_l, ankle_med_l]):
                        tibia_l = tibia_axes(knee_lat_l, knee_med_l, ankle_lat_l, ankle_med_l, pelvis[:, 2], prev_tibia_l)
                        if tibia_l is not None:
                            prev_tibia_l = tibia_l
                            # Knee: femur -> tibia
                            R_knee_l = femur_l.T @ tibia_l
                            knee_l_angles[fi] = euler_xyz(R_knee_l)

                            # Left foot (ankle -> foot)
                            if all(m is not None for m in [heel_l, toe_l]):
                                meta5_l = get_marker(coords, marker_idx, fi, markers.get("meta5_l"))
                                foot_l = foot_axes(heel_l, toe_l, meta5_l, pelvis[:, 2])
                                if foot_l is not None:
                                    # Ankle: tibia -> foot
                                    R_ankle_l = tibia_l.T @ foot_l
                                    ankle_l_angles[fi] = euler_xyz(R_ankle_l)

        # =====================================================================
        # UPPER BODY - TRUNK
        # =====================================================================
        if pelvis is not None and pelvis_origin is not None:
            c7 = get_marker(coords, marker_idx, fi, markers.get("c7"))
            r_shoulder = get_marker(coords, marker_idx, fi, markers.get("shoulder_r"))
            l_shoulder = get_marker(coords, marker_idx, fi, markers.get("shoulder_l"))

            if all(m is not None for m in [c7, r_shoulder, l_shoulder]):
                trunk = trunk_axes(c7, r_shoulder, l_shoulder, pelvis_origin, pelvis[:, 2], prev_trunk)
                if trunk is not None:
                    prev_trunk = trunk
                    # Trunk: pelvis -> trunk
                    R_trunk = pelvis.T @ trunk
                    trunk_angles[fi] = euler_xyz(R_trunk)

                    # =====================================================================
                    # UPPER BODY - RIGHT ARM (Shoulder, Elbow)
                    # =====================================================================
                    elbow_r = get_marker(coords, marker_idx, fi, markers.get("elbow_r"))
                    wrist_r = get_marker(coords, marker_idx, fi, markers.get("wrist_r"))

                    if all(m is not None for m in [r_shoulder, elbow_r, wrist_r]):
                        humerus_r = humerus_axes(r_shoulder, elbow_r, wrist_r, trunk[:, 2], prev_humerus_r)
                        if humerus_r is not None:
                            prev_humerus_r = humerus_r
                            # Shoulder: trunk -> humerus (ZXY sequence)
                            R_shoulder_r = trunk.T @ humerus_r
                            shoulder_r_angles[fi] = euler_zxy(R_shoulder_r)

                            # Elbow flexion (geometric)
                            elbow_r_flex[fi] = geometric_elbow_flexion(r_shoulder, elbow_r, wrist_r)

                    # =====================================================================
                    # UPPER BODY - LEFT ARM (Shoulder, Elbow)
                    # =====================================================================
                    elbow_l = get_marker(coords, marker_idx, fi, markers.get("elbow_l"))
                    wrist_l = get_marker(coords, marker_idx, fi, markers.get("wrist_l"))

                    if all(m is not None for m in [l_shoulder, elbow_l, wrist_l]):
                        humerus_l = humerus_axes(l_shoulder, elbow_l, wrist_l, trunk[:, 2], prev_humerus_l)
                        if humerus_l is not None:
                            prev_humerus_l = humerus_l
                            # Shoulder: trunk -> humerus (ZXY sequence)
                            R_shoulder_l = trunk.T @ humerus_l
                            shoulder_l_angles[fi] = euler_zxy(R_shoulder_l)

                            # Elbow flexion (geometric)
                            elbow_l_flex[fi] = geometric_elbow_flexion(l_shoulder, elbow_l, wrist_l)

    if verbose:
        print("[comprehensive_angles] Post-processing angles (filter, unwrap, zero, clamp)...")

    # Post-process all angles: median filter -> unwrap -> zero -> clamp
    def process_angle_array(angles, joint_type, dof_names, skip_clamp=False):
        """Apply full processing pipeline to angle array."""
        # Median filter to remove outliers
        filtered = median_filter_angles(angles, window_size=5)

        # Unwrap discontinuities
        if unwrap:
            filtered = unwrap_angles_deg(filtered)

        # Zero to reference configuration
        filtered = zero_angles(filtered, times, zero_mode, zero_window_s)

        # Clamp to biomechanical limits (skip for pelvis - uses global angles)
        if not skip_clamp:
            if filtered.ndim == 2:  # Multi-DOF
                for i, dof in enumerate(dof_names):
                    filtered[:, i] = clamp_biomechanical_angles(filtered[:, i], joint_type, dof)
            else:  # Single DOF
                filtered = clamp_biomechanical_angles(filtered, joint_type, dof_names[0])

        return filtered

    # Process all angle arrays
    # Pelvis: skip clamping (global angles, already centered by zeroing)
    pelvis_angles = process_angle_array(pelvis_angles, "pelvis", ["flex", "abd", "rot"], skip_clamp=True)
    hip_r_angles = process_angle_array(hip_r_angles, "hip", ["flex", "abd", "rot"])
    hip_l_angles = process_angle_array(hip_l_angles, "hip", ["flex", "abd", "rot"])
    knee_r_angles = process_angle_array(knee_r_angles, "knee", ["flex", "abd", "rot"])
    knee_l_angles = process_angle_array(knee_l_angles, "knee", ["flex", "abd", "rot"])
    ankle_r_angles = process_angle_array(ankle_r_angles, "ankle", ["flex", "abd", "rot"])
    ankle_l_angles = process_angle_array(ankle_l_angles, "ankle", ["flex", "abd", "rot"])
    trunk_angles = process_angle_array(trunk_angles, "trunk", ["flex", "abd", "rot"])
    shoulder_r_angles = process_angle_array(shoulder_r_angles, "shoulder", ["exo", "flex", "abd"])
    shoulder_l_angles = process_angle_array(shoulder_l_angles, "shoulder", ["exo", "flex", "abd"])
    elbow_r_flex = process_angle_array(elbow_r_flex, "elbow", ["flex"])
    elbow_l_flex = process_angle_array(elbow_l_flex, "elbow", ["flex"])

    # Build output DataFrames
    results = {
        "pelvis": pd.DataFrame({
            "time_s": times,
            "pelvis_tilt_deg": pelvis_angles[:, 0],
            "pelvis_obliquity_deg": pelvis_angles[:, 1],
            "pelvis_rotation_deg": pelvis_angles[:, 2],
        }),
        "hip_R": pd.DataFrame({
            "time_s": times,
            "hip_flex_deg": hip_r_angles[:, 0],
            "hip_abd_deg": hip_r_angles[:, 1],
            "hip_rot_deg": hip_r_angles[:, 2],
        }),
        "hip_L": pd.DataFrame({
            "time_s": times,
            "hip_flex_deg": hip_l_angles[:, 0],
            "hip_abd_deg": hip_l_angles[:, 1],
            "hip_rot_deg": hip_l_angles[:, 2],
        }),
        "knee_R": pd.DataFrame({
            "time_s": times,
            "knee_flex_deg": knee_r_angles[:, 0],
            "knee_abd_deg": knee_r_angles[:, 1],
            "knee_rot_deg": knee_r_angles[:, 2],
        }),
        "knee_L": pd.DataFrame({
            "time_s": times,
            "knee_flex_deg": knee_l_angles[:, 0],
            "knee_abd_deg": knee_l_angles[:, 1],
            "knee_rot_deg": knee_l_angles[:, 2],
        }),
        "ankle_R": pd.DataFrame({
            "time_s": times,
            "ankle_flex_deg": ankle_r_angles[:, 0],
            "ankle_abd_deg": ankle_r_angles[:, 1],
            "ankle_rot_deg": ankle_r_angles[:, 2],
        }),
        "ankle_L": pd.DataFrame({
            "time_s": times,
            "ankle_flex_deg": ankle_l_angles[:, 0],
            "ankle_abd_deg": ankle_l_angles[:, 1],
            "ankle_rot_deg": ankle_l_angles[:, 2],
        }),
        "trunk": pd.DataFrame({
            "time_s": times,
            "trunk_flex_deg": trunk_angles[:, 0],
            "trunk_lateral_deg": trunk_angles[:, 1],
            "trunk_rot_deg": trunk_angles[:, 2],
        }),
        "shoulder_R": pd.DataFrame({
            "time_s": times,
            "shoulder_exo_deg": shoulder_r_angles[:, 0],
            "shoulder_flex_deg": shoulder_r_angles[:, 1],
            "shoulder_abd_deg": shoulder_r_angles[:, 2],
        }),
        "shoulder_L": pd.DataFrame({
            "time_s": times,
            "shoulder_exo_deg": shoulder_l_angles[:, 0],
            "shoulder_flex_deg": shoulder_l_angles[:, 1],
            "shoulder_abd_deg": shoulder_l_angles[:, 2],
        }),
        "elbow_R": pd.DataFrame({
            "time_s": times,
            "elbow_flex_deg": elbow_r_flex,
        }),
        "elbow_L": pd.DataFrame({
            "time_s": times,
            "elbow_flex_deg": elbow_l_flex,
        }),
    }

    if verbose:
        # Report data availability
        for joint_name, df in results.items():
            valid_rows = df.notna().all(axis=1).sum()
            print(f"[comprehensive_angles] {joint_name:12s}: {valid_rows}/{num_frames} frames with data")

    return results
