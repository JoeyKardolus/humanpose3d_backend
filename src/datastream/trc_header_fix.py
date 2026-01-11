"""Fix TRC header for Pose2Sim augmented files.

Pose2Sim's marker augmentation outputs TRC files with a header mismatch:
- Header declares 22 markers (original input)
- Data contains 65 markers (22 original + 43 augmented)

This module provides utilities to:
1. Detect header/data mismatch
2. Get official augmented marker names from Pose2Sim
3. Reconstruct the header with correct marker names
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple


def get_pose2sim_augmented_markers() -> List[str]:
    """Get official augmented marker names from Pose2Sim.

    Returns the marker names in the exact order that Pose2Sim outputs them.
    These come from getOpenPoseMarkers_lowerExtremity2() and
    getMarkers_upperExtremity_noPelvis2() in Pose2Sim.markerAugmentation.

    Returns:
        List of 43 augmented marker names in Pose2Sim output order
    """
    try:
        from Pose2Sim.markerAugmentation import (
            getOpenPoseMarkers_lowerExtremity2,
            getMarkers_upperExtremity_noPelvis2,
        )
        _, resp_lower = getOpenPoseMarkers_lowerExtremity2()
        _, resp_upper = getMarkers_upperExtremity_noPelvis2()
        return resp_lower + resp_upper
    except ImportError:
        # Fallback to hardcoded list if Pose2Sim not available
        # This list matches Pose2Sim 0.9.x output order
        return [
            "C7_study",
            "r_shoulder_study", "L_shoulder_study",
            "r.ASIS_study", "L.ASIS_study",
            "r.PSIS_study", "L.PSIS_study",
            "r_knee_study", "L_knee_study",
            "r_mknee_study", "L_mknee_study",
            "r_ankle_study", "L_ankle_study",
            "r_mankle_study", "L_mankle_study",
            "r_calc_study", "L_calc_study",
            "r_toe_study", "L_toe_study",
            "r_5meta_study", "L_5meta_study",
            "r_lelbow_study", "L_lelbow_study",
            "r_melbow_study", "L_melbow_study",
            "r_lwrist_study", "L_lwrist_study",
            "r_mwrist_study", "L_mwrist_study",
            "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
            "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
            "r_sh1_study", "r_sh2_study", "r_sh3_study",
            "L_sh1_study", "L_sh2_study", "L_sh3_study",
            "RHJC_study", "LHJC_study",
        ]


def count_markers_in_trc(trc_path: Path) -> Tuple[int, int]:
    """Count markers declared in header vs present in data.

    Args:
        trc_path: Path to TRC file

    Returns:
        Tuple of (header_count, data_count)
    """
    lines = trc_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    if len(lines) < 6:
        raise ValueError(f"TRC file too short: {trc_path}")

    # Count markers in header (line 3)
    header_line = lines[3].split("\t")
    header_count = 0
    k = 2
    while k < len(header_line):
        if header_line[k].strip():
            header_count += 1
        k += 3

    # Count markers in data (first data line after header)
    data_lines = [ln for ln in lines[6:] if ln.strip()]
    if not data_lines:
        raise ValueError(f"No data rows in TRC: {trc_path}")

    data_cols = len(data_lines[0].split("\t"))
    # Cols: Frame# + Time + (x, y, z) * num_markers
    data_count = (data_cols - 2) // 3

    return header_count, data_count


def fix_trc_header(
    input_path: Path,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> Path:
    """Fix TRC header to match actual marker count in data.

    Args:
        input_path: Path to TRC file with mismatched header
        output_path: Output path (default: input_FIXED.trc)
        verbose: Print progress messages

    Returns:
        Path to fixed TRC file
    """
    if output_path is None:
        output_path = input_path.with_name(
            input_path.stem.replace("_LSTM", "_LSTM_fixed") + ".trc"
        )
        if output_path == input_path:
            output_path = input_path.with_name(input_path.stem + "_fixed.trc")

    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    if len(lines) < 6:
        raise ValueError(f"TRC file too short: {input_path}")

    h0, h1, h2, h3, h4, h5 = lines[:6]
    data = [ln for ln in lines[6:] if ln.strip()]

    if not data:
        raise ValueError(f"No data rows in TRC: {input_path}")

    # Parse existing header marker names
    h3_parts = h3.split("\t")
    existing_names = []
    k = 2
    while k < len(h3_parts):
        name = h3_parts[k].strip()
        if name:
            existing_names.append(name)
        k += 3

    # Count actual markers in data
    data_cols = len(data[0].split("\t"))
    num_markers_in_data = (data_cols - 2) // 3

    if verbose:
        print(f"[trc_header_fix] Header declares {len(existing_names)} markers")
        print(f"[trc_header_fix] Data contains {num_markers_in_data} markers")

    if num_markers_in_data == len(existing_names):
        if verbose:
            print("[trc_header_fix] Header already matches data - copying file")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    if num_markers_in_data < len(existing_names):
        raise ValueError(
            f"Data has fewer markers ({num_markers_in_data}) than header ({len(existing_names)})"
        )

    # Get augmented marker names from Pose2Sim
    augmented_names = get_pose2sim_augmented_markers()
    num_needed = num_markers_in_data - len(existing_names)

    if num_needed > len(augmented_names):
        if verbose:
            print(
                f"[trc_header_fix] Warning: need {num_needed} names but only have "
                f"{len(augmented_names)} augmented marker names"
            )

    names_to_add = augmented_names[:num_needed]

    if verbose:
        print(f"[trc_header_fix] Adding {len(names_to_add)} augmented marker names")

    # Build new marker name list
    all_names = existing_names + names_to_add

    # Update header line 2 (NumMarkers field)
    h2_parts = h2.split("\t")
    h2_parts[3] = str(num_markers_in_data)
    new_h2 = "\t".join(h2_parts)

    # Build new header line 3 (marker names)
    new_h3_parts = ["Frame#", "Time"]
    for name in all_names:
        new_h3_parts.extend([name, "", ""])
    new_h3 = "\t".join(new_h3_parts)

    # Build new header line 4 (X/Y/Z subheaders)
    new_h4_parts = ["", ""]
    for i in range(1, num_markers_in_data + 1):
        new_h4_parts.extend([f"X{i}", f"Y{i}", f"Z{i}"])
    new_h4 = "\t".join(new_h4_parts)

    # Write fixed file
    output_lines = [h0, h1, new_h2, new_h3, new_h4, h5] + data
    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    if verbose:
        print(f"[trc_header_fix] Fixed TRC written to: {output_path}")

    return output_path


def ensure_trc_header_fixed(trc_path: Path, verbose: bool = True) -> Path:
    """Ensure TRC file has correct header, fixing if needed.

    If the header already matches the data, returns the original path.
    If there's a mismatch, creates a fixed version and returns that path.

    Args:
        trc_path: Path to TRC file
        verbose: Print progress messages

    Returns:
        Path to TRC file with correct header (may be same as input)
    """
    header_count, data_count = count_markers_in_trc(trc_path)

    if header_count == data_count:
        if verbose:
            print(f"[trc_header_fix] Header OK ({header_count} markers)")
        return trc_path

    if verbose:
        print(
            f"[trc_header_fix] Header mismatch: {header_count} in header, "
            f"{data_count} in data - fixing..."
        )

    return fix_trc_header(trc_path, verbose=verbose)
