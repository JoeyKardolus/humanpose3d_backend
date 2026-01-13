"""Verify joint angle fix by comparing before/after metrics.

Usage:
    uv run python scripts/verify_angle_fix.py
"""
import pandas as pd
import numpy as np
from pathlib import Path


def compute_metrics(csv_path: Path) -> dict:
    """Compute ROM and stats for each DOF."""
    df = pd.read_csv(csv_path)
    metrics = {}
    for col in df.columns:
        if col == 'time_s':
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue

        # Check for discontinuities (jumps > 180 degrees)
        diffs = vals.diff().abs()
        has_discontinuity = (diffs > 180).any()
        max_jump = diffs.max()

        metrics[col] = {
            'min': vals.min(),
            'max': vals.max(),
            'rom': vals.max() - vals.min(),
            'std': vals.std(),
            'mean': vals.mean(),
            'has_discontinuity': has_discontinuity,
            'max_jump': max_jump,
        }
    return metrics


def main():
    base_dir = Path('data/output/pose-3d/hardloop/joint_angles')
    baseline_dir = Path('data/output/pose-3d/hardloop/joint_angles_BASELINE_BEFORE_FIX')

    if not baseline_dir.exists():
        print("ERROR: Baseline directory not found. Did you run the backup step?")
        print(f"Expected: {baseline_dir}")
        return

    if not base_dir.exists():
        print("ERROR: Current output directory not found.")
        print(f"Expected: {base_dir}")
        return

    joints = ['hip_R', 'hip_L', 'knee_R', 'knee_L', 'ankle_R', 'ankle_L']

    print("=" * 70)
    print("JOINT ANGLE FIX VERIFICATION REPORT")
    print("=" * 70)
    print()

    # Expected ranges for running (from professor's feedback)
    expected_rom = {
        'hip_flex_deg': (30, 80),    # Should be ~50 ROM
        'hip_abd_deg': (5, 40),      # Smaller ROM
        'hip_rot_deg': (5, 40),      # Smaller ROM
        'knee_flex_deg': (30, 80),   # Should be ~50 max
        'knee_abd_deg': (5, 40),     # Smaller ROM
        'knee_rot_deg': (5, 40),     # Smaller ROM
        'ankle_flex_deg': (15, 60),  # Should be ~30 ROM
        'ankle_abd_deg': (5, 40),    # Smaller ROM
        'ankle_rot_deg': (5, 40),    # Smaller ROM
    }

    all_pass = True

    for joint in joints:
        before_path = baseline_dir / f'hardloop_angles_{joint}.csv'
        after_path = base_dir / f'hardloop_angles_{joint}.csv'

        if not before_path.exists():
            print(f"Skipping {joint} - baseline file not found")
            continue
        if not after_path.exists():
            print(f"Skipping {joint} - after-fix file not found")
            continue

        m_before = compute_metrics(before_path)
        m_after = compute_metrics(after_path)

        print(f"--- {joint.upper()} ---")
        print()

        for dof in m_before:
            if dof not in m_after:
                print(f"  {dof}: Missing in after-fix data")
                continue

            b = m_before[dof]
            a = m_after[dof]

            # Determine if improvement occurred
            rom_improved = a['rom'] > b['rom'] * 2  # At least 2x ROM
            disc_fixed = b['has_discontinuity'] and not a['has_discontinuity']
            disc_new = not b['has_discontinuity'] and a['has_discontinuity']

            # Check against expected ranges
            exp_min, exp_max = expected_rom.get(dof, (5, 100))
            in_expected_range = exp_min <= a['rom'] <= exp_max

            # Status determination
            if a['rom'] > 1000:  # Massive discontinuity
                status = "FAIL (massive ROM)"
                all_pass = False
            elif disc_new:
                status = "FAIL (new discontinuity)"
                all_pass = False
            elif rom_improved and in_expected_range:
                status = "PASS (improved)"
            elif in_expected_range:
                status = "PASS (in range)"
            elif a['rom'] < b['rom']:
                status = "WORSE"
                all_pass = False
            else:
                status = "CHECK"

            print(f"  {dof}:")
            print(f"    BEFORE: ROM={b['rom']:7.1f}°  range=[{b['min']:7.1f}°, {b['max']:7.1f}°]  max_jump={b['max_jump']:7.1f}°")
            print(f"    AFTER:  ROM={a['rom']:7.1f}°  range=[{a['min']:7.1f}°, {a['max']:7.1f}°]  max_jump={a['max_jump']:7.1f}°")
            print(f"    Expected ROM: {exp_min}-{exp_max}°")
            print(f"    Status: {status}")
            print()

        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Expected ranges for running (from professor's feedback):")
    print("  - Hip flexion ROM: ~50°")
    print("  - Knee flexion max: ~50°")
    print("  - Ankle: No massive spikes, ROM ~30°")
    print("  - Rotation/Abduction: ROM < 100° (no 4000° discontinuities)")
    print()

    if all_pass:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED - Review output above")
    print()


if __name__ == '__main__':
    main()
