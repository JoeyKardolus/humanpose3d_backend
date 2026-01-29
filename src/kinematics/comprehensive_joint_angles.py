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
from typing import Callable, Dict, Literal

import numpy as np
import pandas as pd

from .angle_processing import (
    euler_xyz,
    euler_zxy,
    geometric_elbow_flexion,
    median_filter_angles,
    smooth_moving_average,
    unwrap_angles_deg,
    zero_angles,
)
from .trc_utils import get_marker, read_trc
from .segment_coordinate_systems import (
    femur_axes,
    foot_axes,
    humerus_axes,
    pelvis_axes,
    tibia_axes,
    trunk_axes,
)

# Fixed marker names from Pose2Sim augmentation
MARKER_NAMES = {
    # Pelvis
    "asis_r": "r.ASIS_study",
    "asis_l": "L.ASIS_study",
    "psis_r": "r.PSIS_study",
    "psis_l": "L.PSIS_study",
    # Hip joint centers
    "hjc_r": "RHJC_study",
    "hjc_l": "LHJC_study",
    # Knee (lateral/medial)
    "knee_lat_r": "r_knee_study",
    "knee_med_r": "r_mknee_study",
    "knee_lat_l": "L_knee_study",
    "knee_med_l": "L_mknee_study",
    # Ankle (lateral/medial)
    "ankle_lat_r": "r_ankle_study",
    "ankle_med_r": "r_mankle_study",
    "ankle_lat_l": "L_ankle_study",
    "ankle_med_l": "L_mankle_study",
    # Foot
    "heel_r": "r_calc_study",
    "heel_l": "L_calc_study",
    "toe_r": "r_toe_study",
    "toe_l": "L_toe_study",
    "meta5_r": "r_5meta_study",
    "meta5_l": "L_5meta_study",
    # Upper body
    "c7": "C7_study",
    "shoulder_r": "r_shoulder_study",
    "shoulder_l": "L_shoulder_study",
    "elbow_r": "r_lelbow_study",
    "elbow_l": "L_lelbow_study",
    "wrist_r": "r_lwrist_study",
    "wrist_l": "L_lwrist_study",
}


def _compute_joint(
    parent_frame: np.ndarray | None,
    child_func: Callable,
    markers: list,
    prev_frames: dict,
    segment_name: str,
    euler_func: Callable = euler_xyz,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Compute joint angles from parent->child rotation.

    Args:
        parent_frame: Parent segment coordinate system (3x3 rotation matrix)
        child_func: Function to build child coordinate system
        markers: List of markers required by child_func (last is prev_frame)
        prev_frames: Dict tracking previous frames for continuity
        segment_name: Key for prev_frames dict
        euler_func: Euler decomposition function (euler_xyz or euler_zxy)

    Returns:
        (child_frame, angles) where angles = [raw Euler angles] or [nan, nan, nan]
    """
    nan_angles = np.array([np.nan, np.nan, np.nan])

    if parent_frame is None or any(m is None for m in markers[:-1]):
        return None, nan_angles

    # Build child coordinate system (last marker slot is prev_frame)
    prev = prev_frames.get(segment_name)
    child_frame = child_func(*markers[:-1], prev)

    if child_frame is None:
        return None, nan_angles

    prev_frames[segment_name] = child_frame
    R_joint = parent_frame.T @ child_frame
    return child_frame, euler_func(R_joint)


def compute_all_joint_angles(
    trc_path: Path,
    smooth_window: int = 9,
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
        unwrap: Remove 360 degree discontinuities
        zero_mode: Reference configuration for zeroing angles
        zero_window_s: Time window for "first_n_seconds" mode
        verbose: Print progress messages

    Returns:
        Dict with DataFrames for each joint group (pelvis, hip_R, hip_L, etc.)
    """
    if not trc_path.exists():
        raise FileNotFoundError(f"TRC file not found: {trc_path}")

    if verbose:
        print(f"[comprehensive_angles] Reading TRC: {trc_path.name}")

    marker_idx, frames, times, coords = read_trc(trc_path)
    num_frames = len(times)

    # Apply smoothing to marker coordinates
    if smooth_window and smooth_window > 1:
        if verbose:
            print(f"[comprehensive_angles] Applying smoothing (window={smooth_window})")
        smoothed = np.empty_like(coords)
        for mi in range(coords.shape[1]):
            for axis in range(3):
                smoothed[:, mi, axis] = smooth_moving_average(coords[:, mi, axis], smooth_window)
        coords = smoothed

    # Resolve markers - check which fixed names are present
    markers = {}
    for key, trc_name in MARKER_NAMES.items():
        if trc_name in marker_idx:
            idx = marker_idx[trc_name]
            for fi in range(min(10, num_frames)):
                if np.isfinite(coords[fi, idx]).all():
                    markers[key] = trc_name
                    break
        if key not in markers:
            markers[key] = None

    if verbose:
        available = [k for k, v in markers.items() if v is not None]
        missing = [k for k, v in markers.items() if v is None]
        print(f"[comprehensive_angles] Found {len(available)}/{len(markers)} required markers")
        if missing:
            print(f"[comprehensive_angles] Missing: {', '.join(missing[:10])}")

    # Initialize angle storage with dict-based structure for bilateral joints
    pelvis_angles = np.full((num_frames, 3), np.nan)
    trunk_angles = np.full((num_frames, 3), np.nan)
    hip_angles = {"r": np.full((num_frames, 3), np.nan), "l": np.full((num_frames, 3), np.nan)}
    knee_angles = {"r": np.full((num_frames, 3), np.nan), "l": np.full((num_frames, 3), np.nan)}
    ankle_angles = {"r": np.full((num_frames, 3), np.nan), "l": np.full((num_frames, 3), np.nan)}
    shoulder_angles = {"r": np.full((num_frames, 3), np.nan), "l": np.full((num_frames, 3), np.nan)}
    elbow_flex = {"r": np.full(num_frames, np.nan), "l": np.full(num_frames, np.nan)}

    # Track previous coordinate systems for continuity
    prev_frames = {}

    if verbose:
        print(f"[comprehensive_angles] Computing angles for {num_frames} frames...")

    # Helper to get marker for current frame
    def gm(name):
        return get_marker(coords, marker_idx, fi, markers.get(name))

    for fi in range(num_frames):
        pelvis = None

        # Pelvis coordinate system
        rasis, lasis = gm("asis_r"), gm("asis_l")
        rpsis, lpsis = gm("psis_r"), gm("psis_l")

        if all(m is not None for m in [rasis, lasis, rpsis, lpsis]):
            pelvis = pelvis_axes(rasis, lasis, rpsis, lpsis, prev_frames.get("pelvis"))
            if pelvis is not None:
                prev_frames["pelvis"] = pelvis
                pelvis_angles[fi] = euler_zxy(pelvis)

        pelvis_origin = 0.5 * (rasis + lasis) if rasis is not None and lasis is not None else None

        # Lower body - loop over sides
        if pelvis is not None:
            for side in ["r", "l"]:
                hjc = gm(f"hjc_{side}")
                knee_lat, knee_med = gm(f"knee_lat_{side}"), gm(f"knee_med_{side}")
                ankle_lat, ankle_med = gm(f"ankle_lat_{side}"), gm(f"ankle_med_{side}")
                heel, toe, meta5 = gm(f"heel_{side}"), gm(f"toe_{side}"), gm(f"meta5_{side}")

                # Hip (femur)
                femur, hip_angles[side][fi] = _compute_joint(
                    pelvis, femur_axes,
                    [hjc, knee_lat, knee_med, pelvis[:, 2], None],
                    prev_frames, f"femur_{side}"
                )

                # Knee (tibia)
                tibia, knee_angles[side][fi] = _compute_joint(
                    femur, tibia_axes,
                    [knee_lat, knee_med, ankle_lat, ankle_med, pelvis[:, 2], None],
                    prev_frames, f"tibia_{side}"
                )

                # Ankle (foot)
                _, ankle_angles[side][fi] = _compute_joint(
                    tibia, foot_axes,
                    [heel, toe, meta5, pelvis[:, 2], None],
                    prev_frames, f"foot_{side}"
                )

        # Upper body - trunk
        if pelvis is not None and pelvis_origin is not None:
            c7 = gm("c7")
            r_shoulder, l_shoulder = gm("shoulder_r"), gm("shoulder_l")

            if all(m is not None for m in [c7, r_shoulder, l_shoulder]):
                trunk = trunk_axes(c7, r_shoulder, l_shoulder, pelvis_origin, pelvis[:, 2],
                                   prev_frames.get("trunk"))
                if trunk is not None:
                    prev_frames["trunk"] = trunk
                    trunk_angles[fi] = euler_xyz(pelvis.T @ trunk)

                    # Arms - loop over sides
                    for side in ["r", "l"]:
                        shoulder = gm(f"shoulder_{side}")
                        elbow = gm(f"elbow_{side}")
                        wrist = gm(f"wrist_{side}")

                        if all(m is not None for m in [shoulder, elbow, wrist]):
                            humerus, shoulder_angles[side][fi] = _compute_joint(
                                trunk, humerus_axes,
                                [shoulder, elbow, wrist, trunk[:, 2], None],
                                prev_frames, f"humerus_{side}",
                                euler_func=euler_zxy
                            )
                            if humerus is not None:
                                elbow_flex[side][fi] = geometric_elbow_flexion(shoulder, elbow, wrist)

    if verbose:
        print("[comprehensive_angles] Post-processing angles (filter, unwrap, zero)...")

    def process_angle_array(angles, use_global_mean=False, skip_median_filter=False):
        """Apply full processing pipeline to angle array."""
        filtered = angles.copy() if skip_median_filter else median_filter_angles(angles, window_size=5)
        if unwrap:
            filtered = unwrap_angles_deg(filtered)
        actual_zero_mode = "global_mean" if use_global_mean else zero_mode
        return zero_angles(filtered, times, actual_zero_mode, zero_window_s)

    # Process all angle arrays
    pelvis_angles = process_angle_array(pelvis_angles, use_global_mean=True, skip_median_filter=True)
    trunk_angles = process_angle_array(trunk_angles)
    for side in ["r", "l"]:
        hip_angles[side] = process_angle_array(hip_angles[side])
        knee_angles[side] = process_angle_array(knee_angles[side])
        ankle_angles[side] = process_angle_array(ankle_angles[side])
        shoulder_angles[side] = process_angle_array(shoulder_angles[side])
        elbow_flex[side] = process_angle_array(elbow_flex[side])

    # Build output DataFrames
    # XYZ Euler: Index 0=ABD, Index 1=ROT, Index 2=FLEX
    # ZXY Euler: Index 0=ROT, Index 1=FLEX, Index 2=ABD (pelvis, shoulder)
    results = {
        "pelvis": pd.DataFrame({
            "time_s": times,
            "pelvis_flex_deg": pelvis_angles[:, 0],
            "pelvis_abd_deg": pelvis_angles[:, 1],
            "pelvis_rot_deg": pelvis_angles[:, 2],
        }),
        "trunk": pd.DataFrame({
            "time_s": times,
            "trunk_flex_deg": trunk_angles[:, 2],
            "trunk_abd_deg": trunk_angles[:, 0],
            "trunk_rot_deg": trunk_angles[:, 1],
        }),
    }

    # Add bilateral joints
    for side in ["r", "l"]:
        SIDE = side.upper()
        results[f"hip_{SIDE}"] = pd.DataFrame({
            "time_s": times,
            "hip_flex_deg": hip_angles[side][:, 2],
            "hip_abd_deg": hip_angles[side][:, 0],
            "hip_rot_deg": hip_angles[side][:, 1],
        })
        results[f"knee_{SIDE}"] = pd.DataFrame({
            "time_s": times,
            "knee_flex_deg": knee_angles[side][:, 2],
            "knee_abd_deg": knee_angles[side][:, 0],
            "knee_rot_deg": knee_angles[side][:, 1],
        })
        results[f"ankle_{SIDE}"] = pd.DataFrame({
            "time_s": times,
            "ankle_flex_deg": ankle_angles[side][:, 2],
            "ankle_abd_deg": ankle_angles[side][:, 0],
            "ankle_rot_deg": ankle_angles[side][:, 1],
        })
        results[f"shoulder_{SIDE}"] = pd.DataFrame({
            "time_s": times,
            "shoulder_flex_deg": shoulder_angles[side][:, 1],
            "shoulder_abd_deg": shoulder_angles[side][:, 2],
            "shoulder_rot_deg": shoulder_angles[side][:, 0],
        })
        results[f"elbow_{SIDE}"] = pd.DataFrame({
            "time_s": times,
            "elbow_flex_deg": elbow_flex[side],
        })

    if verbose:
        for joint_name, df in results.items():
            valid_rows = df.notna().all(axis=1).sum()
            print(f"[comprehensive_angles] {joint_name:12s}: {valid_rows}/{num_frames} frames with data")

    return results
