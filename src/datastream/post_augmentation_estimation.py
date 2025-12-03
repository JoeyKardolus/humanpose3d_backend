"""Estimate remaining markers after Pose2Sim augmentation to reach 100%."""
from __future__ import annotations

import numpy as np
from pathlib import Path


def estimate_shoulder_clusters_and_hjc(trc_path: Path) -> Path:
    """Fill in shoulder clusters and hip joint centers that Pose2Sim's LSTM missed.

    Uses the VisualizeData loader which properly reads all 65 markers from data.
    """
    from ..visualizedata.visualize_data import VisualizeData

    # Load TRC with proper 65-marker support
    viz = VisualizeData()
    marker_names, frames = viz.load_trc_frames(trc_path)

    # Create marker index map
    marker_idx = {name: i for i, name in enumerate(marker_names)}

    def get_marker(frame_data, name: str) -> np.ndarray:
        """Get marker coords as numpy array."""
        if name not in marker_idx:
            return np.array([np.nan, np.nan, np.nan])
        idx = marker_idx[name]
        if idx >= len(frame_data):
            return np.array([np.nan, np.nan, np.nan])
        return frame_data[idx].copy()

    def set_marker(frame_data, name: str, coords: np.ndarray):
        """Set marker coords."""
        if name not in marker_idx:
            return
        idx = marker_idx[name]
        if idx < len(frame_data):
            frame_data[idx] = coords

    # Process each frame
    for frame in frames:
        # Estimate right shoulder cluster from RShoulder and RElbow
        r_shoulder = get_marker(frame, 'RShoulder')
        r_elbow = get_marker(frame, 'RElbow')

        if not np.isnan(r_shoulder).any() and not np.isnan(r_elbow).any():
            # Create 3 points around the shoulder
            vec = r_elbow - r_shoulder
            vec_len = np.linalg.norm(vec)
            if vec_len > 0.01:  # Valid vector
                vec_norm = vec / vec_len

                # Perpendicular vectors for cluster
                perp1 = np.array([-vec_norm[2], 0, vec_norm[0]])  # Perpendicular in XZ plane
                perp1_len = np.linalg.norm(perp1)
                if perp1_len > 0.001:
                    perp1 = perp1 / perp1_len
                    perp2 = np.cross(vec_norm, perp1)

                    offset = 0.05  # 5cm offset
                    set_marker(frame, 'r_sh1_study', r_shoulder + perp1 * offset)
                    set_marker(frame, 'r_sh2_study', r_shoulder + perp2 * offset)
                    set_marker(frame, 'r_sh3_study', r_shoulder - perp1 * offset)

        # Estimate left shoulder cluster
        l_shoulder = get_marker(frame, 'LShoulder')
        l_elbow = get_marker(frame, 'LElbow')

        if not np.isnan(l_shoulder).any() and not np.isnan(l_elbow).any():
            vec = l_elbow - l_shoulder
            vec_len = np.linalg.norm(vec)
            if vec_len > 0.01:
                vec_norm = vec / vec_len

                perp1 = np.array([-vec_norm[2], 0, vec_norm[0]])
                perp1_len = np.linalg.norm(perp1)
                if perp1_len > 0.001:
                    perp1 = perp1 / perp1_len
                    perp2 = np.cross(vec_norm, perp1)

                    offset = 0.05
                    set_marker(frame, 'L_sh1_study', l_shoulder + perp1 * offset)
                    set_marker(frame, 'L_sh2_study', l_shoulder + perp2 * offset)
                    set_marker(frame, 'L_sh3_study', l_shoulder - perp1 * offset)

        # Estimate hip joint centers from ASIS and PSIS markers
        r_asis = get_marker(frame, 'r.ASIS_study')
        l_asis = get_marker(frame, 'L.ASIS_study')
        r_psis = get_marker(frame, 'r.PSIS_study')
        l_psis = get_marker(frame, 'L.PSIS_study')

        if not any(np.isnan(m).any() for m in [r_asis, l_asis, r_psis, l_psis]):
            # Hip joint center regression from Bell et al. 1990
            pelvis_width = np.linalg.norm(r_asis - l_asis)
            pelvis_depth = np.linalg.norm(r_asis - r_psis)

            if pelvis_width > 0.01 and pelvis_depth > 0.01:
                # Right HJC
                r_mid_pelvis = (r_asis + r_psis) / 2
                lateral_offset = -0.24 * pelvis_width  # Lateral
                post_offset = -0.16 * pelvis_depth      # Posterior
                inf_offset = -0.28 * pelvis_width       # Inferior

                # Create pelvis coordinate system
                pelvic_x = (r_asis - l_asis) / pelvis_width  # Lateral axis
                forward = (r_asis + l_asis) / 2 - (r_psis + l_psis) / 2
                forward_len = np.linalg.norm(forward)
                if forward_len > 0.01:
                    pelvic_z = forward / forward_len  # Forward axis
                    pelvic_y = np.cross(pelvic_z, pelvic_x)  # Up axis

                    r_hjc = r_mid_pelvis + lateral_offset * pelvic_x + inf_offset * pelvic_y + post_offset * pelvic_z
                    set_marker(frame, 'RHJC_study', r_hjc)

                    # Left HJC (mirror)
                    l_mid_pelvis = (l_asis + l_psis) / 2
                    l_hjc = l_mid_pelvis - lateral_offset * pelvic_x + inf_offset * pelvic_y + post_offset * pelvic_z
                    set_marker(frame, 'LHJC_study', l_hjc)

    # Write updated TRC
    output_path = trc_path.parent / f"{trc_path.stem}_complete.trc"
    _write_trc(output_path, marker_names, frames, trc_path)

    return output_path


def _write_trc(output_path: Path, marker_names: list, frames: list, original_trc: Path):
    """Write TRC file with all markers."""
    # Read original header
    original_lines = original_trc.read_text(encoding='utf-8').splitlines()

    # Find data start
    data_start = next(i for i, line in enumerate(original_lines) if line.startswith('Frame#')) + 2
    while data_start < len(original_lines) and not original_lines[data_start].strip():
        data_start += 1

    # Write new file with updated data
    with output_path.open('w', encoding='utf-8') as f:
        # Copy header lines
        for line in original_lines[:data_start]:
            f.write(line + '\n')

        # Write data with all markers
        for frame_idx, frame in enumerate(frames):
            row = [str(frame_idx + 1), f'{frame_idx / 30.0:.6f}']  # Frame# and Time

            # Add all marker coords
            for coords in frame:
                if np.isnan(coords).any():
                    row.extend(['', '', ''])
                else:
                    row.extend([f'{coords[0]:.6f}', f'{coords[1]:.6f}', f'{coords[2]:.6f}'])

            f.write('\t'.join(row) + '\n')
