"""Temporal smoothing for TRC marker data using moving average."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
from scipy.ndimage import uniform_filter1d


def _read_trc_direct(trc_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Read TRC file directly, preserving all data lines.

    Args:
        trc_path: Path to TRC file

    Returns:
        Tuple of (header_lines, frames, times, coords)
    """
    lines = trc_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # TRC header: lines 0-4, data starts at line 5
    header_lines = lines[:5]
    data_lines = [line for line in lines[5:] if line.strip()]

    if not data_lines:
        raise ValueError(f"No data rows found in {trc_path}")

    # Determine marker count from first data line
    first_row = data_lines[0].split("\t")
    num_data_cols = len(first_row) - 2  # Subtract Frame# and Time
    num_markers = num_data_cols // 3

    num_frames = len(data_lines)
    frames = np.zeros(num_frames, dtype=int)
    times = np.zeros(num_frames, dtype=float)
    coords = np.full((num_frames, num_markers, 3), np.nan, dtype=float)

    for fi, line in enumerate(data_lines):
        cols = line.split("\t")
        try:
            frames[fi] = int(float(cols[0]))
            times[fi] = float(cols[1])
        except (ValueError, IndexError):
            continue

        for mi in range(num_markers):
            cx, cy, cz = 2 + 3 * mi, 3 + 3 * mi, 4 + 3 * mi
            if cz < len(cols):
                try:
                    coords[fi, mi, 0] = float(cols[cx]) if cols[cx] else np.nan
                    coords[fi, mi, 1] = float(cols[cy]) if cols[cy] else np.nan
                    coords[fi, mi, 2] = float(cols[cz]) if cols[cz] else np.nan
                except ValueError:
                    pass

    return header_lines, frames, times, coords


def smooth_trc(
    input_path: Path,
    output_path: Path | None = None,
    window: int = 3,
) -> Path:
    """Smooth marker positions using simple moving average.

    Each marker is smoothed independently using uniform_filter1d with 'reflect'
    mode to avoid edge artifacts. Markers with NaN values are left unchanged.

    Args:
        input_path: Input TRC file path
        output_path: Output path (default: input with _smooth suffix)
        window: Smoothing window size (odd number, default 3)

    Returns:
        Path to smoothed TRC file
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_smooth.trc"
    output_path = Path(output_path)

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    # Read TRC data directly (preserves all frames)
    header_lines, frames, times, coords = _read_trc_direct(input_path)
    num_frames, num_markers, _ = coords.shape

    print(f"[smooth] Simple temporal smoothing (window={window}) on {num_frames} frames, {num_markers} markers")

    # Create smoothed coordinates array
    smoothed = np.copy(coords)
    smoothed_count = 0
    skipped_count = 0

    # Apply smoothing to each marker independently
    for mi in range(num_markers):
        marker_data = coords[:, mi, :]
        has_nan = np.isnan(marker_data).any()

        if has_nan:
            skipped_count += 1
            continue

        smoothed_count += 1
        for axis in range(3):
            smoothed[:, mi, axis] = uniform_filter1d(
                coords[:, mi, axis],
                size=window,
                mode='reflect'
            )

    print(f"[smooth] Smoothed {smoothed_count} markers, skipped {skipped_count} with NaN")

    # Build new data lines
    data_lines = []
    for fi in range(num_frames):
        row = [str(frames[fi]), f"{times[fi]:.6f}"]
        for mi in range(num_markers):
            x, y, z = smoothed[fi, mi]
            if np.isnan(x):
                row.extend(["", "", ""])
            else:
                row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        data_lines.append("\t".join(row))

    # Write output
    output_content = "\n".join(header_lines + data_lines) + "\n"
    output_path.write_text(output_content, encoding="utf-8")

    print(f"[smooth] Wrote smoothed TRC to {output_path}")

    return output_path


def hide_markers_in_trc(
    input_path: Path,
    markers_to_hide: list[str],
    output_path: Path | None = None,
) -> Path:
    """Hide specified markers by setting their coordinates to NaN.

    Also hides related augmented markers (e.g., hiding RElbow also hides r_lelbow_study).

    Args:
        input_path: Input TRC file path
        markers_to_hide: List of marker names to hide (e.g., ['RElbow', 'RWrist'])
        output_path: Output path (default: overwrite input)

    Returns:
        Path to modified TRC file
    """
    # Mapping from base markers to their augmented variants
    AUGMENTED_MARKERS = {
        'RElbow': ['r_lelbow_study', 'r_melbow_study'],
        'LElbow': ['L_lelbow_study', 'L_melbow_study'],
        'RWrist': ['r_lwrist_study', 'r_mwrist_study'],
        'LWrist': ['L_lwrist_study', 'L_mwrist_study'],
    }

    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path
    output_path = Path(output_path)

    if not markers_to_hide:
        return output_path

    # Expand markers_to_hide to include augmented variants
    expanded_markers = set(markers_to_hide)
    for marker in markers_to_hide:
        if marker in AUGMENTED_MARKERS:
            expanded_markers.update(AUGMENTED_MARKERS[marker])
    markers_to_hide = list(expanded_markers)

    # Read file and parse header to find marker positions
    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    if len(lines) < 5:
        print(f"[hide] Warning: TRC file too short, skipping")
        return output_path

    # Line 3 (index 2) contains marker names
    # Format: Frame#\tTime\tMarker1\t\t\tMarker2\t\t\t...
    marker_line = lines[3]
    marker_cols = marker_line.split("\t")

    # Find column indices for markers to hide
    # Marker names appear 3 times each (X, Y, Z) - we want the FIRST occurrence
    marker_indices = {}  # marker_name -> column index (0-based in data)
    for i, col in enumerate(marker_cols):
        col = col.strip()
        if col and col not in ("Frame#", "Time", "") and col not in marker_indices:
            # Only store first occurrence of each marker name
            marker_indices[col] = i

    # Find which markers to hide
    cols_to_hide = set()
    hidden_markers = []
    for marker in markers_to_hide:
        if marker in marker_indices:
            base_col = marker_indices[marker]
            # Each marker has 3 columns: X, Y, Z
            cols_to_hide.add(base_col)
            cols_to_hide.add(base_col + 1)
            cols_to_hide.add(base_col + 2)
            hidden_markers.append(marker)

    if not hidden_markers:
        print(f"[hide] No matching markers found to hide")
        return output_path

    # Process data lines (starting at line 5, index 5)
    new_lines = lines[:5]  # Keep header
    for line in lines[5:]:
        if not line.strip():
            new_lines.append(line)
            continue

        cols = line.split("\t")
        for col_idx in cols_to_hide:
            if col_idx < len(cols):
                cols[col_idx] = ""
        new_lines.append("\t".join(cols))

    # Write output
    output_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    print(f"[hide] Hidden {len(hidden_markers)} markers: {', '.join(hidden_markers)}")

    return output_path
