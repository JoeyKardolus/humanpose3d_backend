from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np


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
    age: int,
    sex: str,
    augmentation_cycles: int = 1,
) -> Path:
    """Invoke Pose2Sim's augment_markers_all with multi-cycle averaging for better results."""
    if augmentation_cycles < 1:
        raise ValueError(f"augmentation_cycles must be >= 1, got {augmentation_cycles}")

    if augmentation_cycles == 1:
        # Single cycle - use original fast path
        return _run_single_augmentation_cycle(
            trc_path, out_dir, height, mass, age, sex, cycle_num=0
        )

    # Multi-cycle averaging approach
    print(f"[augment] running {augmentation_cycles} augmentation cycles with averaging")
    cycle_results = []

    for cycle_num in range(augmentation_cycles):
        print(f"[augment] cycle {cycle_num + 1}/{augmentation_cycles}")
        try:
            cycle_output = _run_single_augmentation_cycle(
                trc_path, out_dir, height, mass, age, sex, cycle_num
            )
            cycle_results.append(cycle_output)
        except Exception as exc:
            print(f"[augment] warning: cycle {cycle_num + 1} failed: {exc}")
            continue

    if not cycle_results:
        raise RuntimeError("All augmentation cycles failed")

    # Average the results
    averaged_output = _average_trc_files(cycle_results, out_dir, trc_path.stem)
    print(
        f"[augment] averaged {len(cycle_results)}/{augmentation_cycles} successful cycles"
    )

    # Clean up intermediate cycle files and project directories
    for cycle_num in range(augmentation_cycles):
        # Remove cycle-specific TRC files
        cycle_trc = out_dir / f"{trc_path.stem}_LSTM_cycle{cycle_num}.trc"
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

    return averaged_output


def _run_single_augmentation_cycle(
    trc_path: Path,
    out_dir: Path,
    height: float,
    mass: float,
    age: int,
    sex: str,
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

    header_lines = all_lines[:data_start_idx]

    # Determine expected number of columns from first data line
    first_data_parts = all_lines[data_start_idx].strip().split("\t")
    expected_cols = len(first_data_parts)

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

    # Write output
    output_path = out_dir / f"{base_name}_LSTM.trc"
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
