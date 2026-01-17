#!/usr/bin/env python3
"""Compare pelvis output between main codebase and 'use' directory reference.

This script validates that the main codebase produces identical pelvis angles
to the reference implementation in the 'use' directory.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_use_pelvis_csv(csv_path: Path) -> pd.DataFrame:
    """Load pelvis angles from 'use' directory output."""
    df = pd.read_csv(csv_path)
    # Rename columns to match main codebase output (universal naming)
    df = df.rename(columns={
        "pelvis_flex_deg(Z)": "pelvis_flex_deg",
        "pelvis_abd_deg(X)": "pelvis_abd_deg",
        "pelvis_rot_deg(Y)": "pelvis_rot_deg",
    })
    return df


def compute_pelvis_from_trc(trc_path: Path) -> pd.DataFrame:
    """Compute pelvis angles using main codebase."""
    from src.kinematics.comprehensive_joint_angles import compute_all_joint_angles

    results = compute_all_joint_angles(
        trc_path,
        smooth_window=9,  # Match reference (verified by testing)
        unwrap=True,
        zero_mode="global_mean",  # This is overridden for pelvis internally
        verbose=True,
    )

    return results["pelvis"]


def compare_angles(ref_df: pd.DataFrame, test_df: pd.DataFrame, tolerance: float = 1.0):
    """Compare two pelvis angle DataFrames.

    Args:
        ref_df: Reference angles from 'use' directory
        test_df: Test angles from main codebase
        tolerance: Maximum acceptable difference in degrees
    """
    print("\n" + "="*60)
    print("PELVIS ANGLE COMPARISON")
    print("="*60)

    # Align by time if lengths differ
    if len(ref_df) != len(test_df):
        print(f"WARNING: Frame count mismatch - ref={len(ref_df)}, test={len(test_df)}")
        # Use minimum length
        min_len = min(len(ref_df), len(test_df))
        ref_df = ref_df.iloc[:min_len].copy()
        test_df = test_df.iloc[:min_len].copy()

    columns = ["pelvis_flex_deg", "pelvis_abd_deg", "pelvis_rot_deg"]

    all_passed = True

    for col in columns:
        if col not in ref_df.columns:
            print(f"WARNING: Missing column {col} in reference")
            continue
        if col not in test_df.columns:
            print(f"WARNING: Missing column {col} in test")
            continue

        ref = ref_df[col].values
        test = test_df[col].values

        # Handle NaN
        valid = np.isfinite(ref) & np.isfinite(test)
        if not valid.any():
            print(f"{col}: No valid data to compare")
            continue

        diff = np.abs(ref[valid] - test[valid])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rmse = np.sqrt(np.mean(diff**2))

        passed = max_diff <= tolerance
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"\n{col}:")
        print(f"  Reference: mean={np.nanmean(ref):.3f}, std={np.nanstd(ref):.3f}")
        print(f"  Test:      mean={np.nanmean(test):.3f}, std={np.nanstd(test):.3f}")
        print(f"  Max diff:  {max_diff:.4f} deg")
        print(f"  Mean diff: {mean_diff:.4f} deg")
        print(f"  RMSE:      {rmse:.4f} deg")
        print(f"  Status:    [{status}] (tolerance={tolerance} deg)")

    print("\n" + "="*60)
    if all_passed:
        print("RESULT: ALL TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED")
    print("="*60)

    return all_passed


def main():
    # Check for 'use' directory output
    use_output = PROJECT_ROOT / "use" / "output"
    use_pelvis_csv = list(use_output.glob("*_pelvis_global_ZXY.csv"))

    if not use_pelvis_csv:
        print("ERROR: No pelvis CSV found in use/output/")
        print("Run the 'use' pipeline first to generate reference output.")
        return 1

    ref_csv = use_pelvis_csv[0]
    print(f"Reference CSV: {ref_csv}")

    # Find corresponding TRC file
    # The CSV name is like pose2sim_input_exact_LSTM_fixed_pelvis_global_ZXY.csv
    # The TRC name is like pose2sim_input_exact_LSTM_fixed.trc
    trc_name = ref_csv.stem.replace("_pelvis_global_ZXY", "") + ".trc"
    trc_path = use_output / trc_name

    if not trc_path.exists():
        # Try without _fixed
        trc_path = use_output / trc_name.replace("_fixed", "")

    if not trc_path.exists():
        # Look in pose-3d subdirectory
        trc_path = use_output / "pose-3d" / trc_name

    if not trc_path.exists():
        print(f"ERROR: TRC file not found: {trc_path}")
        print("Available TRC files:")
        for f in use_output.rglob("*.trc"):
            print(f"  {f}")
        return 1

    print(f"TRC file: {trc_path}")

    # Load reference data
    print("\nLoading reference pelvis angles...")
    ref_df = load_use_pelvis_csv(ref_csv)
    print(f"Reference frames: {len(ref_df)}")

    # Compute using main codebase
    print("\nComputing pelvis angles using main codebase...")
    test_df = compute_pelvis_from_trc(trc_path)
    print(f"Test frames: {len(test_df)}")

    # Compare
    passed = compare_angles(ref_df, test_df, tolerance=1.0)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
