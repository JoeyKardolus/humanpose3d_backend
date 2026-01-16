"""
Automated feature testing script for HumanPose3D pipeline.

Systematically tests each feature and combination to measure impact on:
- Bone length consistency (depth accuracy)
- Augmentation success rate
- Processing time
- Visual quality
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Test configurations
TEST_CONFIGS = {
    "config_0_baseline": {
        "name": "Baseline (No Optional Features)",
        "flags": [],
        "description": "Minimal pipeline with 20 augmentation cycles",
    },
    "config_1_gaussian": {
        "name": "Baseline + Gaussian Smoothing",
        "flags": ["--gaussian-smooth", "2.5"],
        "description": "Temporal Gaussian smoothing",
    },
    "config_2_flk": {
        "name": "Baseline + FLK Filtering",
        "flags": ["--flk-filter", "--flk-passes", "1"],
        "description": "Spatio-temporal filtering with biomechanical constraints",
    },
    "config_3_anatomical": {
        "name": "Baseline + Anatomical Constraints",
        "flags": ["--anatomical-constraints"],
        "description": "Bone length smoothing, ground plane, pelvis filtering",
    },
    "config_4_bone_length": {
        "name": "Baseline + Bone Length Constraints (NEW)",
        "flags": ["--bone-length-constraints", "--bone-length-report"],
        "description": "Enforces consistent bone lengths (depth correction)",
    },
    "config_5_estimation": {
        "name": "Baseline + Marker Estimation",
        "flags": ["--estimate-missing"],
        "description": "Fills missing markers using anatomical symmetry",
    },
    "config_6_multicycle": {
        "name": "Baseline + Multi-cycle Augmentation (30)",
        "flags": ["--augmentation-cycles", "30"],
        "description": "30 LSTM cycles vs default 20",
    },
    "config_7_combo_bone_est": {
        "name": "Bone Length + Estimation",
        "flags": ["--bone-length-constraints", "--estimate-missing"],
        "description": "Combination of depth correction and marker filling",
    },
    "config_8_combo_gaussian_bone_est": {
        "name": "Gaussian + Bone Length + Estimation",
        "flags": ["--gaussian-smooth", "2.5", "--bone-length-constraints", "--estimate-missing"],
        "description": "Temporal smoothing + depth correction + marker filling",
    },
    "config_9_combo_anatomical_bone_est": {
        "name": "Anatomical + Bone Length + Estimation (RECOMMENDED)",
        "flags": ["--anatomical-constraints", "--bone-length-constraints", "--estimate-missing"],
        "description": "Full anatomical pipeline without heavy filtering",
    },
    "config_10_maximum": {
        "name": "All Features (Maximum Quality)",
        "flags": [
            "--gaussian-smooth", "2.5",
            "--flk-filter", "--flk-passes", "2",
            "--anatomical-constraints",
            "--bone-length-constraints",
            "--estimate-missing",
            "--force-complete",
            "--augmentation-cycles", "30",
        ],
        "description": "Every feature enabled for maximum quality",
    },
    "config_11_ground_plane": {
        "name": "Baseline + Ground Plane Refinement (NEW)",
        "flags": ["--ground-plane-refinement"],
        "description": "Stance detection and depth propagation from foot contacts",
    },
    "config_12_ground_plane_bone": {
        "name": "Ground Plane + Bone Length",
        "flags": ["--ground-plane-refinement", "--bone-length-constraints"],
        "description": "Combined ground plane and bone length depth correction",
    },
    "config_13_force_complete": {
        "name": "Anatomical + Bone Length + Estimation + Force Complete",
        "flags": ["--anatomical-constraints", "--bone-length-constraints", "--estimate-missing", "--force-complete"],
        "description": "Full pipeline with post-augmentation completion for 100% markers",
    },
    "config_14_recommended_plus_ground": {
        "name": "RECOMMENDED + Ground Plane",
        "flags": ["--anatomical-constraints", "--bone-length-constraints", "--estimate-missing", "--ground-plane-refinement"],
        "description": "Best configuration + ground plane refinement",
    },
}

# Standard test video and subject parameters
TEST_VIDEO = "data/input/joey.mp4"
SUBJECT_PARAMS = ["--height", "1.78", "--mass", "75.0", "--age", "30", "--sex", "male"]

# Default augmentation cycles (overridden by config)
DEFAULT_CYCLES = ["--augmentation-cycles", "20"]

# Output directory
OUTPUT_ROOT = Path("data/output/feature-tests")


def load_trc_data(trc_path: Path) -> Tuple[List[str], np.ndarray]:
    """Load TRC file and return marker names and frames data."""
    with open(trc_path) as f:
        lines = f.readlines()

    # Parse header to get marker names
    marker_line = lines[3].strip().split("\t")
    marker_names = []
    for i in range(2, len(marker_line), 3):  # Skip Frame#, Time
        if marker_line[i]:
            marker_names.append(marker_line[i])

    # Parse data - need to handle variable length rows
    data_lines = lines[6:]  # Skip header rows
    frames = []
    max_cols = 0

    for line in data_lines:
        if line.strip():
            parts = line.strip().split("\t")
            values = []
            for v in parts:
                try:
                    values.append(float(v) if v else np.nan)
                except ValueError:
                    values.append(np.nan)
            frames.append(values)
            max_cols = max(max_cols, len(values))

    # Pad all rows to same length
    padded_frames = []
    for frame in frames:
        if len(frame) < max_cols:
            frame.extend([np.nan] * (max_cols - len(frame)))
        padded_frames.append(frame)

    return marker_names, np.array(padded_frames)


def calculate_bone_length_statistics(trc_path: Path) -> Dict[str, float]:
    """
    Calculate bone length consistency statistics from TRC file.

    Returns dict with:
    - bone_pair → std_dev for each tracked bone
    - average_std → overall consistency metric
    """
    marker_names, frames = load_trc_data(trc_path)

    # Map marker names to column indices
    marker_indices = {}
    for i, name in enumerate(marker_names):
        # Each marker has 3 columns (x, y, z)
        col_start = 2 + i * 3  # Skip Frame# and Time columns
        marker_indices[name] = col_start

    # Define bone pairs to track
    bone_pairs = [
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
        ("RHip", "RKnee"),
        ("RKnee", "RAnkle"),
        ("LHip", "LKnee"),
        ("LKnee", "LAnkle"),
    ]

    bone_stats = {}
    valid_stds = []

    for parent_name, child_name in bone_pairs:
        if parent_name not in marker_indices or child_name not in marker_indices:
            continue

        parent_idx = marker_indices[parent_name]
        child_idx = marker_indices[child_name]

        # Calculate bone length for each frame
        lengths = []
        for frame in frames:
            parent_x, parent_y, parent_z = frame[parent_idx:parent_idx + 3]
            child_x, child_y, child_z = frame[child_idx:child_idx + 3]

            # Skip frames with missing data
            if any(np.isnan([parent_x, parent_y, parent_z, child_x, child_y, child_z])):
                continue

            dx = child_x - parent_x
            dy = child_y - parent_y
            dz = child_z - parent_z
            length = np.sqrt(dx**2 + dy**2 + dz**2)
            lengths.append(length)

        if lengths:
            std_dev = float(np.std(lengths))
            bone_stats[f"{parent_name}-{child_name}"] = std_dev
            valid_stds.append(std_dev)

    # Calculate average std across all bones
    if valid_stds:
        bone_stats["average_std"] = float(np.mean(valid_stds))
    else:
        bone_stats["average_std"] = 0.0

    return bone_stats


def count_augmented_markers(trc_path: Path) -> Tuple[int, int]:
    """
    Count how many of the 43 augmented markers have valid data.

    Returns (successful_count, total_augmented_markers=43)
    """
    marker_names, frames = load_trc_data(trc_path)

    # Augmented TRC files have 22 markers in header but 65 in data
    # Need to count based on actual data columns, not header
    if len(frames) == 0:
        return 0, 43

    n_cols = frames.shape[1]
    # Calculate actual number of markers from data columns
    # Format: Frame# (1 col) + Time (1 col) + markers * 3 coords
    n_markers_actual = (n_cols - 2) // 3

    if n_markers_actual <= 22:
        # No augmented markers
        return 0, 43

    # Count markers that have at least some valid data across frames
    augmented_count = 0
    total_augmented = 43

    # Augmented markers are indices 22-64 (43 markers after the first 22)
    for marker_idx in range(22, min(n_markers_actual, 65)):
        col_start = 2 + marker_idx * 3  # Skip Frame# and Time columns

        if col_start + 3 > n_cols:
            break

        # Check if any frame has valid (non-zero, non-NaN) data for this marker
        marker_data = frames[:, col_start:col_start + 3]

        # Check for valid data: not all zeros and not all NaNs
        has_non_zero = np.any(np.abs(marker_data) > 0.001)  # Threshold for "real" data
        has_non_nan = np.any(~np.isnan(marker_data))

        if has_non_zero and has_non_nan:
            augmented_count += 1

    return augmented_count, total_augmented


def run_pipeline_test(
    config_id: str,
    config: Dict,
    output_dir: Path,
) -> Dict:
    """
    Run a single pipeline configuration and collect metrics.

    Returns dict with:
    - config_id, name, description
    - processing_time
    - bone_length_stats
    - augmentation_success (count, percentage)
    - output_file
    """
    print(f"\n{'='*80}")
    print(f"Running: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}\n")

    # Build command
    cmd = ["uv", "run", "python", "main.py", "--video", TEST_VIDEO]
    cmd.extend(SUBJECT_PARAMS)

    # Add config-specific flags
    cmd.extend(config["flags"])

    # Add default augmentation cycles if not specified
    if "--augmentation-cycles" not in config["flags"]:
        cmd.extend(DEFAULT_CYCLES)

    # Run pipeline and measure time
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        processing_time = time.time() - start_time
        print(result.stdout)

        if result.stderr:
            print("STDERR:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Pipeline failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {
            "config_id": config_id,
            "name": config["name"],
            "description": config["description"],
            "error": str(e),
            "processing_time": time.time() - start_time,
        }

    # Find output TRC file
    video_stem = Path(TEST_VIDEO).stem
    output_path = Path("data/output/pose-3d") / video_stem

    # Try to find LSTM or complete TRC
    trc_candidates = [
        output_path / f"{video_stem}_complete.trc",
        output_path / f"{video_stem}_LSTM.trc",
        output_path / f"{video_stem}.trc",
    ]

    trc_file = None
    for candidate in trc_candidates:
        if candidate.exists():
            trc_file = candidate
            break

    if not trc_file:
        print(f"ERROR: Could not find output TRC file in {output_path}")
        return {
            "config_id": config_id,
            "name": config["name"],
            "description": config["description"],
            "error": "TRC file not found",
            "processing_time": processing_time,
        }

    print(f"\nAnalyzing output: {trc_file}")

    # Calculate metrics
    bone_stats = calculate_bone_length_statistics(trc_file)
    aug_count, aug_total = count_augmented_markers(trc_file)
    aug_percentage = (aug_count / aug_total * 100) if aug_total > 0 else 0.0

    results = {
        "config_id": config_id,
        "name": config["name"],
        "description": config["description"],
        "processing_time": processing_time,
        "bone_length_stats": bone_stats,
        "augmentation_success": {
            "count": aug_count,
            "total": aug_total,
            "percentage": aug_percentage,
        },
        "output_file": str(trc_file),
    }

    print(f"\n📊 Results:")
    print(f"   Processing Time: {processing_time:.1f} seconds")
    print(f"   Bone Length Std (avg): {bone_stats.get('average_std', 0):.4f} m")
    print(f"   Augmentation Success: {aug_count}/{aug_total} ({aug_percentage:.1f}%)")
    print(f"   Output: {trc_file}")

    return results


def main():
    """Run all test configurations and save results."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results_file = OUTPUT_ROOT / "test_results.json"
    all_results = []

    print("HumanPose3D Feature Testing")
    print("="*80)
    print(f"Test Video: {TEST_VIDEO}")
    print(f"Output Directory: {OUTPUT_ROOT}")
    print(f"Configurations to Test: {len(TEST_CONFIGS)}")
    print("="*80)

    # Run each configuration
    for config_id, config in TEST_CONFIGS.items():
        result = run_pipeline_test(config_id, config, OUTPUT_ROOT)
        all_results.append(result)

        # Save results incrementally
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("TESTING COMPLETE - SUMMARY")
    print("="*80)

    baseline = None
    for result in all_results:
        if "error" in result:
            print(f"\n❌ {result['name']}: FAILED")
            print(f"   Error: {result['error']}")
            continue

        is_baseline = "baseline" in result["config_id"].lower()
        if is_baseline:
            baseline = result

        bone_std = result["bone_length_stats"].get("average_std", 0)
        aug_pct = result["augmentation_success"]["percentage"]
        time_s = result["processing_time"]

        improvement = ""
        if baseline and not is_baseline:
            baseline_std = baseline["bone_length_stats"].get("average_std", 0)
            if baseline_std > 0:
                improvement_pct = (baseline_std - bone_std) / baseline_std * 100
                improvement = f" ({improvement_pct:+.1f}% vs baseline)"

        print(f"\n✅ {result['name']}")
        print(f"   Time: {time_s:.1f}s | Bone Std: {bone_std:.4f}m{improvement} | Aug: {aug_pct:.1f}%")

    print(f"\n📁 Detailed results saved to: {results_file}")
    print("\n💡 Next: Review results and update docs/TESTING_REPORT.md")


if __name__ == "__main__":
    main()
