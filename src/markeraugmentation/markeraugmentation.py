from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np


# Complete marker set after Pose2Sim LSTM augmentation (64 markers total)
# Order: Original 21 + Lower body 35 + Upper body 8
ORIGINAL_MARKERS = [
    "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
    "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
    "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist",
    "Hip", "Nose"
]  # 21 markers (Head removed - not used by pipeline)

LOWER_BODY_AUGMENTED = [
    "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study",
    "r_knee_study", "r_mknee_study", "r_ankle_study", "r_mankle_study",
    "r_toe_study", "r_5meta_study", "r_calc_study",
    "L_knee_study", "L_mknee_study", "L_ankle_study", "L_mankle_study",
    "L_toe_study", "L_calc_study", "L_5meta_study",
    "r_shoulder_study", "L_shoulder_study", "C7_study",
    "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
    "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
    "r_sh1_study", "r_sh2_study", "r_sh3_study",
    "L_sh1_study", "L_sh2_study", "L_sh3_study",
    "RHJC_study", "LHJC_study"
]

UPPER_BODY_AUGMENTED = [
    "r_lelbow_study", "r_melbow_study",  # lateral/medial right elbow
    "r_lwrist_study", "r_mwrist_study",  # lateral/medial right wrist
    "L_lelbow_study", "L_melbow_study",  # lateral/medial left elbow
    "L_lwrist_study", "L_mwrist_study"   # lateral/medial left wrist
]

ALL_MARKERS_64 = ORIGINAL_MARKERS + LOWER_BODY_AUGMENTED + UPPER_BODY_AUGMENTED


def _resolve_pose2sim_command(config_path: Path) -> list[str]:
    """Derive the Pose2Sim invocation command for augment_markers_all."""
    cmd_env = os.environ.get("POSE2SIM_CMD")
    if cmd_env:
        return shlex.split(cmd_env) + [str(config_path)]

    repo_root = Path(__file__).resolve().parents[2]
    local_cli = repo_root / ".venv" / "bin" / "pose2sim"
    if local_cli.is_file():
        return [str(local_cli), "augment_markers_all", str(config_path)]

    local_py = repo_root / ".venv" / "bin" / "python"
    shim = (
        "import sys,pathlib,tomllib;"
        "from Pose2Sim import markerAugmentation;"
        "cfg=pathlib.Path(sys.argv[1]);"
        "config=tomllib.load(cfg.open('rb'));"
        "markerAugmentation.augment_markers_all(config)"
    )
    if local_py.exists():
        return [str(local_py), "-c", shim, str(config_path)]

    return [sys.executable, "-c", shim, str(config_path)]


def run_pose2sim_augment(
    trc_path: Path,
    out_dir: Path,
    height: float,
    mass: float,
    augmentation_cycles: int = 1,
    camera_space: bool = False,
) -> Path:
    """Invoke Pose2Sim's augment_markers_all with multi-cycle averaging for better results.

    Args:
        trc_path: Path to input TRC file
        out_dir: Output directory
        height: Subject height in meters
        mass: Subject mass in kg
        augmentation_cycles: Number of cycles for averaging (default 1)
        camera_space: If True, transform camera-space input to Pose2Sim convention
                     before augmentation, and transform output back to camera-space.
                     Required when using POF 3D reconstruction.

    Returns:
        Path to augmented TRC file
    """
    if augmentation_cycles < 1:
        raise ValueError(f"augmentation_cycles must be >= 1, got {augmentation_cycles}")

    # Handle camera-space input by transforming coordinates
    if camera_space:
        from src.datastream.trc_transforms import (
            parse_trc_to_array,
            array_to_trc,
            camera_to_pose2sim,
            pose2sim_to_camera,
        )

        print("[augment] transforming camera-space input to Pose2Sim convention")
        marker_data, marker_names, frame_rate = parse_trc_to_array(trc_path)
        transformed_data, pelvis_offset, scale = camera_to_pose2sim(marker_data, marker_names)

        # Write transformed TRC for Pose2Sim
        transformed_trc = out_dir / f"{trc_path.stem}_transformed.trc"
        array_to_trc(transformed_data, marker_names, transformed_trc, frame_rate)
        working_trc = transformed_trc
    else:
        working_trc = trc_path

    if augmentation_cycles == 1:
        # Single cycle - use original fast path
        augmented_output = _run_single_augmentation_cycle(
            working_trc, out_dir, height, mass, cycle_num=0
        )
    else:
        # Multi-cycle averaging approach
        print(f"[augment] running {augmentation_cycles} augmentation cycles with averaging")
        cycle_results = []

        for cycle_num in range(augmentation_cycles):
            print(f"[augment] cycle {cycle_num + 1}/{augmentation_cycles}")
            try:
                cycle_output = _run_single_augmentation_cycle(
                    working_trc, out_dir, height, mass, cycle_num
                )
                cycle_results.append(cycle_output)
            except Exception as exc:
                print(f"[augment] warning: cycle {cycle_num + 1} failed: {exc}")
                continue

        if not cycle_results:
            raise RuntimeError("All augmentation cycles failed")

        # Average the results
        augmented_output = _average_trc_files(cycle_results, out_dir, working_trc.stem)
        print(
            f"[augment] averaged {len(cycle_results)}/{augmentation_cycles} successful cycles"
        )

    # Clean up intermediate cycle files and project directories
    if augmentation_cycles > 1:
        for cycle_num in range(augmentation_cycles):
            # Remove cycle-specific TRC files
            cycle_trc = out_dir / f"{working_trc.stem}_LSTM_cycle{cycle_num}.trc"
            if cycle_trc.exists():
                cycle_trc.unlink()

            # Remove cycle-specific config files
            cycle_config = out_dir / f"Config_cycle{cycle_num}.toml"
            if cycle_config.exists():
                cycle_config.unlink()

            # Remove cycle-specific project directories
            cycle_project = out_dir / f"pose2sim_project_cycle{cycle_num}"
            if cycle_project.exists():
                shutil.rmtree(cycle_project)

        print(f"[augment] cleaned up {augmentation_cycles} intermediate cycle files")

    # Handle camera-space output
    # Transform output to kinematics convention (Y-up) for ISB-compliant joint angles
    if camera_space:
        from src.datastream.trc_transforms import parse_trc_to_array, array_to_trc, pose2sim_to_kinematics

        print("[augment] transforming output to kinematics convention (Y-up)")

        # Read augmented markers (still in camera Y-down convention)
        aug_data, aug_marker_names, aug_frame_rate = parse_trc_to_array(augmented_output)

        # Apply Y-inversion for kinematics (camera Y-down → kinematics Y-up)
        kinematics_data = pose2sim_to_kinematics(aug_data)

        # Write final output with original filename convention
        final_output = out_dir / f"{trc_path.stem}_LSTM.trc"
        array_to_trc(kinematics_data, aug_marker_names, final_output, aug_frame_rate)

        # Clean up intermediate files
        if transformed_trc.exists():
            transformed_trc.unlink()
        if augmented_output != final_output and augmented_output.exists():
            augmented_output.unlink()

        return final_output

    return augmented_output


def _run_single_augmentation_cycle(
    trc_path: Path,
    out_dir: Path,
    height: float,
    mass: float,
    cycle_num: int,
) -> Path:
    """Run a single Pose2Sim augmentation cycle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    project_dir = out_dir / f"pose2sim_project_cycle{cycle_num}"
    pose3d_dir = project_dir / "pose-3d"
    pose3d_dir.mkdir(parents=True, exist_ok=True)

    trc_copy = pose3d_dir / trc_path.name

    # For multi-cycle runs (cycle_num > 0), add small perturbations to create variation
    # This allows averaging to smooth noise and potentially fill gaps
    if cycle_num > 0:
        _copy_trc_with_perturbation(trc_path, trc_copy, cycle_num)
    else:
        shutil.copy2(trc_path, trc_copy)

    config_path = out_dir / f"Config_cycle{cycle_num}.toml"
    config_path.write_text(
        dedent(
            f"""
            [project]
            project_dir = "{project_dir}"
            participant_height = {height}
            participant_mass = {mass}
            frame_range = "all"

            [markerAugmentation]
            feet_on_floor = false
            make_c3d = false

            [kinematics]
            default_height = {height}
            fastest_frames_to_remove_percent = 0.1
            close_to_zero_speed_m = 0.2
            large_hip_knee_angles = 45
            trimmed_extrema_percent = 0.5
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    command = _resolve_pose2sim_command(config_path)

    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Pose2Sim CLI not found. Install pose2sim, set POSE2SIM_CMD, or provide "
            "a local `.venv/bin/pose2sim` executable."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Pose2Sim invocation failed (cycle {cycle_num}). "
            "Ensure pose2sim is installed in `.venv`, declared in pyproject, or override POSE2SIM_CMD."
        ) from exc

    lstm_source = pose3d_dir / f"{trc_path.stem}_LSTM.trc"
    if not lstm_source.exists():
        raise RuntimeError(f"Pose2Sim run did not produce {lstm_source}")

    cycle_output = out_dir / f"{trc_path.stem}_LSTM_cycle{cycle_num}.trc"
    shutil.copy2(lstm_source, cycle_output)
    return cycle_output


def _average_trc_files(trc_files: list[Path], out_dir: Path, base_name: str) -> Path:
    """Average multiple TRC files and write the result."""
    if not trc_files:
        raise ValueError("No TRC files to average")

    # Read header from first file to get expected structure
    with open(trc_files[0], "r") as f:
        all_lines = f.readlines()

    # Find where data starts
    data_start_idx = 0
    for i, line in enumerate(all_lines):
        parts = line.strip().split("\t")
        if parts and parts[0].replace(".", "", 1).replace("-", "", 1).isdigit():
            data_start_idx = i
            break

    # Determine expected number of columns from first data line
    first_data_parts = all_lines[data_start_idx].strip().split("\t")
    expected_cols = len(first_data_parts)
    n_markers = (expected_cols - 2) // 3  # Subtract Frame#, Time; divide by 3 coords

    print(f"[augment] expected columns per row: {expected_cols}")

    # Read all TRC files
    all_data = []
    for trc_file in trc_files:
        with open(trc_file, "r") as f:
            lines = f.readlines()

        # Parse data rows
        data_rows = []
        in_data = False
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split("\t")

            # Check if this is a data row
            if not in_data:
                if parts and parts[0].replace(".", "", 1).replace("-", "", 1).isdigit():
                    in_data = True
                else:
                    continue

            if in_data:
                # Parse all columns
                row = []
                for i, val in enumerate(parts):
                    if val and val.strip():
                        try:
                            row.append(float(val))
                        except ValueError:
                            row.append(np.nan)
                    else:
                        row.append(np.nan)

                # Ensure row has exactly expected_cols columns
                while len(row) < expected_cols:
                    row.append(np.nan)
                if len(row) > expected_cols:
                    row = row[:expected_cols]

                data_rows.append(row)

        if data_rows:
            all_data.append(np.array(data_rows, dtype=float))

    if not all_data:
        raise RuntimeError("No valid data found in TRC files")

    # Ensure all have same number of frames (use minimum)
    min_frames = min(arr.shape[0] for arr in all_data)
    all_data = [arr[:min_frames, :] for arr in all_data]

    # Verify shapes match
    shapes = [arr.shape for arr in all_data]
    print(f"[augment] data shapes: {shapes[0]} ({len(all_data)} files)")

    # Stack and average
    stacked = np.stack(all_data, axis=0)  # shape: (n_files, n_frames, n_cols)
    averaged_data = np.nanmean(stacked, axis=0)  # shape: (n_frames, n_cols)

    # Build proper header with all marker names
    # Pose2Sim bug: header doesn't include augmented marker names, but data does
    output_filename = f"{base_name}_LSTM.trc"
    header_lines = _build_trc_header(
        filename=output_filename,
        n_frames=len(averaged_data),
        n_markers=n_markers,
        data_rate=30.0,
    )

    # Write output
    output_path = out_dir / output_filename
    with open(output_path, "w") as f:
        # Write header
        for line in header_lines:
            f.write(line)

        # Write averaged data
        for row in averaged_data:
            formatted_row = "\t".join(
                "" if np.isnan(val) else f"{val:.6f}" for val in row
            )
            f.write(formatted_row + "\n")

    print(f"[augment] wrote {len(averaged_data)} frames with {expected_cols} columns")

    return output_path


def _build_trc_header(
    filename: str,
    n_frames: int,
    n_markers: int,
    data_rate: float = 30.0,
) -> list[str]:
    """Build a proper TRC header with all marker names.

    Fixes Pose2Sim bug where augmented marker names aren't written to header.
    """
    # Use ALL_MARKERS_64 if we have 64 markers, otherwise use what we have
    if n_markers == 64:
        markers = ALL_MARKERS_64
    elif n_markers == 21:
        markers = ORIGINAL_MARKERS
    else:
        # Fallback: generate generic marker names
        markers = [f"Marker{i+1}" for i in range(n_markers)]

    # Line 0: PathFileType
    line0 = f"PathFileType\t4\t(X/Y/Z)\t{filename}\n"

    # Line 1: DataRate, CameraRate, NumFrames, NumMarkers, Units
    line1 = f"DataRate\t{data_rate:.2f}\tCameraRate\t{data_rate:.2f}\tNumFrames\t{n_frames}\tNumMarkers\t{n_markers}\tUnits\tm\n"

    # Line 2: Empty
    line2 = "\n"

    # Line 3: Frame#, Time, then marker names (each repeated 3x for X, Y, Z)
    marker_names = "\t".join(f"{m}\t{m}\t{m}" for m in markers)
    line3 = f"Frame#\tTime\t{marker_names}\n"

    # Line 4: Empty, Empty, then X1 Y1 Z1 X2 Y2 Z2 ... for each marker
    xyz_labels = "\t".join(f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(n_markers))
    line4 = f"\t\t{xyz_labels}\n"

    return [line0, line1, line2, line3, line4]


def _copy_trc_with_perturbation(
    input_trc: Path, output_trc: Path, seed: int, noise_scale: float = 0.0005
) -> None:
    """Copy TRC file with small random perturbations to create variation for averaging.

    Args:
        input_trc: Source TRC file
        output_trc: Destination TRC file
        seed: Random seed for reproducibility
        noise_scale: Scale of Gaussian noise to add (in meters, default 0.5mm)
    """
    np.random.seed(seed)

    with open(input_trc, "r") as f:
        lines = f.readlines()

    # Find where data starts
    data_start_idx = 0
    for i, line in enumerate(lines):
        parts = line.strip().split("\t")
        if parts and parts[0].replace(".", "", 1).isdigit():
            data_start_idx = i
            break

    # Write header unchanged
    output_lines = lines[:data_start_idx]

    # Add noise to data rows
    for line in lines[data_start_idx:]:
        if not line.strip():
            continue

        parts = line.strip().split("\t")
        perturbed_parts = []

        for i, val in enumerate(parts):
            if i < 2:
                # Keep Frame# and Time unchanged
                perturbed_parts.append(val)
            elif val and val.strip():
                # Add small Gaussian noise to coordinate data
                try:
                    original = float(val)
                    noise = np.random.normal(0, noise_scale)
                    perturbed = original + noise
                    perturbed_parts.append(f"{perturbed:.6f}")
                except ValueError:
                    perturbed_parts.append(val)
            else:
                # Keep empty values empty
                perturbed_parts.append(val)

        output_lines.append("\t".join(perturbed_parts) + "\n")

    with open(output_trc, "w") as f:
        f.writelines(output_lines)
