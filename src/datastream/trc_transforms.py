"""Coordinate transforms between camera-space and Pose2Sim/kinematics conventions.

Two-stage transform approach:
1. camera_to_pose2sim: Center on pelvis only (keeps camera Y-down convention)
   - Pose2Sim LSTM was trained on height-normalized, pelvis-centered data
   - The LSTM learns relative marker positions, not absolute coordinates

2. pose2sim_to_kinematics: Invert Y for ISB-compliant kinematics
   - ISB standards use Y-up convention
   - Apply after augmentation, before joint angle computation
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np


# Marker names in order matching TRC file format (21 original markers)
MARKER_NAMES = [
    "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
    "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
    "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist",
    "Hip", "Nose"
]

# Indices for hip markers (0-indexed in the marker list)
RHIP_IDX = 3   # RHip
LHIP_IDX = 4   # LHip
HIP_IDX = 19   # Hip (midpoint)


def camera_to_pose2sim(
    marker_data: np.ndarray,
    marker_names: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Transform camera-space coordinates to Pose2Sim convention.

    Pose2Sim LSTM expects:
    - Pelvis-centered (all markers relative to hip midpoint)
    - Height-normalized (implicitly handled by Pose2Sim)
    - Same axis orientation as input (NO Y-inversion here)

    Args:
        marker_data: (n_frames, n_markers, 3) marker positions in camera-space
        marker_names: List of marker names in order

    Returns:
        transformed: (n_frames, n_markers, 3) pelvis-centered coordinates
        pelvis_offset: (n_frames, 3) pelvis position for inverse transform
        scale: Scale factor (currently 1.0, reserved for future use)
    """
    if marker_names is None:
        marker_names = MARKER_NAMES

    # Find hip marker indices
    try:
        rhip_idx = marker_names.index("RHip")
        lhip_idx = marker_names.index("LHip")
    except ValueError:
        rhip_idx = RHIP_IDX
        lhip_idx = LHIP_IDX

    # Compute pelvis center (midpoint of hips)
    pelvis = (marker_data[:, rhip_idx] + marker_data[:, lhip_idx]) / 2

    # Center on pelvis only - NO Y-inversion
    transformed = marker_data - pelvis[:, np.newaxis, :]

    return transformed, pelvis, 1.0


def pose2sim_to_kinematics(marker_data: np.ndarray) -> np.ndarray:
    """Transform Pose2Sim output to kinematics convention (Y-up).

    Kinematics (ISB standard) expects:
    - Y-up convention (vertical axis points up)
    - Pelvis-centered (already satisfied from Pose2Sim)

    This inverts Y to convert from camera convention (Y-down) to
    biomechanics convention (Y-up).

    Args:
        marker_data: (n_frames, n_markers, 3) from Pose2Sim augmentation

    Returns:
        (n_frames, n_markers, 3) with Y-up for kinematics
    """
    transformed = marker_data.copy()
    transformed[:, :, 1] = -transformed[:, :, 1]
    return transformed


def kinematics_to_pose2sim(marker_data: np.ndarray) -> np.ndarray:
    """Transform kinematics coordinates back to Pose2Sim convention.

    Inverse of pose2sim_to_kinematics (inverts Y again).

    Args:
        marker_data: (n_frames, n_markers, 3) in kinematics convention

    Returns:
        (n_frames, n_markers, 3) in Pose2Sim convention
    """
    transformed = marker_data.copy()
    transformed[:, :, 1] = -transformed[:, :, 1]
    return transformed


def pose2sim_to_camera(
    marker_data: np.ndarray,
    pelvis_offset: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Transform Pose2Sim coordinates back to camera-space.

    Inverse of camera_to_pose2sim (adds back pelvis offset).

    Args:
        marker_data: (n_frames, n_markers, 3) pelvis-centered
        pelvis_offset: (n_frames, 3) pelvis position from original camera-space
        scale: Scale factor (must match value from camera_to_pose2sim)

    Returns:
        (n_frames, n_markers, 3) in camera-space
    """
    return marker_data + pelvis_offset[:, np.newaxis, :]


def parse_trc_to_array(trc_path: Path) -> Tuple[np.ndarray, List[str], float]:
    """Parse TRC file to numpy array.

    Handles Pose2Sim's LSTM output which has mismatched header/data marker counts.
    Uses data column count as the authoritative marker count.

    Args:
        trc_path: Path to TRC file

    Returns:
        data: (n_frames, n_markers, 3) marker positions
        marker_names: List of marker names
        frame_rate: Data rate from header
    """
    from src.markeraugmentation.markeraugmentation import ALL_MARKERS_64

    with open(trc_path, 'r') as f:
        lines = f.readlines()

    # Parse header (Line 1: DataRate info)
    header_parts = lines[1].strip().split('\t')
    frame_rate = float(header_parts[1])

    # Line 3: Marker names (each name repeated 3x for X, Y, Z)
    name_line = lines[3].strip().split('\t')
    header_marker_names = []
    for i in range(2, len(name_line), 3):
        if name_line[i] and name_line[i] not in header_marker_names:
            header_marker_names.append(name_line[i])

    # Find data start (first line with numeric frame number)
    data_start = 0
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        if parts and parts[0].replace('.', '', 1).replace('-', '', 1).isdigit():
            data_start = i
            break

    # Parse data rows
    data_rows = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if not parts[0].replace('.', '', 1).replace('-', '', 1).isdigit():
            continue

        coords = []
        for val in parts[2:]:
            try:
                coords.append(float(val) if val.strip() else np.nan)
            except ValueError:
                coords.append(np.nan)
        data_rows.append(coords)

    # Convert to array
    data = np.array(data_rows, dtype=np.float32)
    n_frames = data.shape[0]
    n_data_cols = data.shape[1]
    n_markers = n_data_cols // 3

    # Build marker names list
    if n_markers == 64 and len(header_marker_names) < 64:
        marker_names = ALL_MARKERS_64
    elif n_markers <= len(header_marker_names):
        marker_names = header_marker_names[:n_markers]
    else:
        marker_names = header_marker_names.copy()
        for i in range(len(header_marker_names), n_markers):
            marker_names.append(f"Marker{i+1}")

    # Pad data if needed
    expected_cols = n_markers * 3
    if n_data_cols < expected_cols:
        padding = np.full((n_frames, expected_cols - n_data_cols), np.nan)
        data = np.hstack([data, padding])
    elif n_data_cols > expected_cols:
        data = data[:, :expected_cols]

    data = data.reshape(n_frames, n_markers, 3)

    return data, marker_names, frame_rate


def array_to_trc(
    data: np.ndarray,
    marker_names: List[str],
    output_path: Path,
    frame_rate: float = 30.0,
) -> None:
    """Write marker data to TRC file.

    Args:
        data: (n_frames, n_markers, 3) marker positions
        marker_names: List of marker names
        output_path: Output TRC file path
        frame_rate: Data frame rate
    """
    n_frames, n_markers, _ = data.shape
    filename = Path(output_path).name
    lines = []

    # Line 0
    lines.append(f"PathFileType\t4\t(X/Y/Z)\t{filename}\n")

    # Line 1
    lines.append(
        f"DataRate\t{frame_rate:.2f}\tCameraRate\t{frame_rate:.2f}\t"
        f"NumFrames\t{n_frames}\tNumMarkers\t{n_markers}\tUnits\tm\n"
    )

    # Line 2 (empty)
    lines.append("\n")

    # Line 3: marker names (each 3x for X, Y, Z)
    name_parts = ["Frame#", "Time"]
    for name in marker_names:
        name_parts.extend([name, name, name])
    lines.append("\t".join(name_parts) + "\n")

    # Line 4: X Y Z labels
    xyz_parts = ["", ""]
    for i in range(n_markers):
        xyz_parts.extend([f"X{i+1}", f"Y{i+1}", f"Z{i+1}"])
    lines.append("\t".join(xyz_parts) + "\n")

    # Data rows
    for frame_idx in range(n_frames):
        time = frame_idx / frame_rate
        row_parts = [str(frame_idx + 1), f"{time:.6f}"]
        for marker_idx in range(n_markers):
            for coord_idx in range(3):
                val = data[frame_idx, marker_idx, coord_idx]
                if np.isnan(val):
                    row_parts.append("")
                else:
                    row_parts.append(f"{val:.6f}")
        lines.append("\t".join(row_parts) + "\n")

    with open(output_path, 'w') as f:
        f.writelines(lines)
