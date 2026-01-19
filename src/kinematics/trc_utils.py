"""TRC file reading utilities.

Provides functions to read TRC (Track Row Column) files and extract marker data
for joint angle computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Import for official Pose2Sim marker names
try:
    from ..datastream.trc_header_fix import get_pose2sim_augmented_markers
except ImportError:
    get_pose2sim_augmented_markers = None


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

    # If data has more markers than header, use official Pose2Sim augmented marker names
    if num_markers_in_data > len(marker_names):
        # Get official marker names from Pose2Sim (or fallback to hardcoded list)
        if get_pose2sim_augmented_markers is not None:
            augmented_names = get_pose2sim_augmented_markers()
        else:
            # Fallback: hardcoded list matching Pose2Sim 0.9.x output order
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
