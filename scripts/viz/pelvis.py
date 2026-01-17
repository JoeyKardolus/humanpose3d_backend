#!/usr/bin/env python3
"""Create visualization comparing pelvis angles: reference vs main codebase."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    # Paths
    ref_csv = PROJECT_ROOT / "use" / "output" / "pose2sim_input_exact_LSTM_fixed_pelvis_global_ZXY.csv"
    trc_path = PROJECT_ROOT / "use" / "output" / "pose2sim_input_exact_LSTM_fixed.trc"
    output_path = PROJECT_ROOT / "pelvis_validation_comparison.png"

    print(f"Reference CSV: {ref_csv}")
    print(f"TRC file: {trc_path}")

    # Load reference data
    ref_df = pd.read_csv(ref_csv)
    ref_df = ref_df.rename(columns={
        "pelvis_flex_deg(Z)": "pelvis_tilt_deg",
        "pelvis_abd_deg(X)": "pelvis_obliquity_deg",
        "pelvis_rot_deg(Y)": "pelvis_rotation_deg",
    })

    # Compute using main codebase
    from src.kinematics.comprehensive_joint_angles import compute_all_joint_angles

    print("\nComputing pelvis angles using main codebase...")
    results = compute_all_joint_angles(
        trc_path,
        smooth_window=9,
        unwrap=True,
        zero_mode="global_mean",
        verbose=False,
    )
    test_df = results["pelvis"]

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    time = ref_df["time_s"].values
    columns = [
        ("pelvis_tilt_deg", "Pelvis Tilt (Flex/Ext)", "Z-axis"),
        ("pelvis_obliquity_deg", "Pelvis Obliquity (Abd/Add)", "X-axis"),
        ("pelvis_rotation_deg", "Pelvis Rotation", "Y-axis"),
    ]

    for i, (col, title, axis_label) in enumerate(columns):
        ref_vals = ref_df[col].values
        test_vals = test_df[col].values

        # Left: Overlay comparison
        ax_left = axes[i, 0]
        ax_left.plot(time, ref_vals, 'b-', linewidth=1.5, label='Reference (use/)', alpha=0.8)
        ax_left.plot(time, test_vals, 'r--', linewidth=1.5, label='Main codebase', alpha=0.8)
        ax_left.set_ylabel("Angle (deg)")
        ax_left.set_title(f"{title} - Comparison")
        ax_left.legend(loc='upper right')
        ax_left.grid(True, alpha=0.3)
        ax_left.axhline(0, color='k', linewidth=0.5)

        # Right: Difference
        ax_right = axes[i, 1]
        valid = np.isfinite(ref_vals) & np.isfinite(test_vals)
        diff = np.full_like(ref_vals, np.nan)
        diff[valid] = test_vals[valid] - ref_vals[valid]

        ax_right.plot(time, diff * 1000, 'g-', linewidth=1)  # Convert to millidegrees
        ax_right.set_ylabel("Difference (mdeg)")
        ax_right.set_title(f"{title} - Difference (Test - Ref)")
        ax_right.grid(True, alpha=0.3)
        ax_right.axhline(0, color='k', linewidth=0.5)

        # Add stats
        if valid.any():
            max_diff = np.max(np.abs(diff[valid])) * 1000
            mean_diff = np.mean(np.abs(diff[valid])) * 1000
            ax_right.text(0.02, 0.98, f"Max: {max_diff:.2f} mdeg\nMean: {mean_diff:.2f} mdeg",
                         transform=ax_right.transAxes, verticalalignment='top',
                         fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    fig.suptitle("Pelvis Angle Validation: Main Codebase vs Reference Implementation\n"
                 f"TRC: {trc_path.name} | Frames: {len(time)} | Smoothing: window=9",
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Also show stats
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for col, title, _ in columns:
        ref_vals = ref_df[col].values
        test_vals = test_df[col].values
        valid = np.isfinite(ref_vals) & np.isfinite(test_vals)
        diff = np.abs(test_vals[valid] - ref_vals[valid])
        print(f"{title:30s}: max={diff.max()*1000:.2f} mdeg, mean={diff.mean()*1000:.2f} mdeg")

    print("="*60)
    print(f"Output: {output_path}")

    return output_path


if __name__ == "__main__":
    main()
