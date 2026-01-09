"""Comprehensive joint angle visualization for full body.

Creates multi-panel plots showing all joint angles (lower + upper body) over time,
following biomechanical conventions and ISB standards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_comprehensive_joint_angles(
    angle_results: Dict[str, pd.DataFrame],
    output_path: Optional[Path] = None,
    title_prefix: str = "",
    dpi: int = 150,
) -> None:
    """Create comprehensive multi-panel visualization of all joint angles.

    Displays angles for all joints (pelvis, lower body, trunk, upper body) in
    a grid layout with separate panels for each joint.

    Args:
        angle_results: Dict of DataFrames from compute_all_joint_angles()
        output_path: If provided, save figure to this path
        title_prefix: Prefix for figure title (e.g., video name)
        dpi: Figure resolution for saved image

    Layout:
        - Column 1: Right side (Hip_R, Knee_R, Ankle_R, Shoulder_R, Elbow_R)
        - Column 2: Left side (Hip_L, Knee_L, Ankle_L, Shoulder_L, Elbow_L)
        - Top rows: Pelvis and Trunk (span both columns)
    """
    # Define plot layout: (rows, cols) for each joint
    # We'll use a 6x2 grid
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(7, 2, hspace=0.4, wspace=0.3)

    # Color scheme for DOFs
    colors = {
        "flex": "#1f77b4",  # Blue
        "abd": "#ff7f0e",   # Orange
        "rot": "#2ca02c",   # Green
        "exo": "#d62728",   # Red
        "lateral": "#9467bd",  # Purple
    }

    # Joint configurations: (result_key, panel_title, DOF_labels)
    joint_configs = [
        # Row 0: Pelvis (span 2 columns)
        ("pelvis", "Pelvis", [("pelvis_tilt_deg", "Tilt", "flex"),
                               ("pelvis_obliquity_deg", "Obliquity", "abd"),
                               ("pelvis_rotation_deg", "Rotation", "rot")]),

        # Row 1: Trunk (span 2 columns)
        ("trunk", "Trunk", [("trunk_flex_deg", "Flex/Ext", "flex"),
                            ("trunk_lateral_deg", "Lateral", "lateral"),
                            ("trunk_rot_deg", "Rotation", "rot")]),

        # Row 2: Hips (R | L)
        ("hip_R", "Hip (Right)", [("hip_flex_deg", "Flex/Ext", "flex"),
                                   ("hip_abd_deg", "Abd/Add", "abd"),
                                   ("hip_rot_deg", "Int/Ext Rot", "rot")]),
        ("hip_L", "Hip (Left)", [("hip_flex_deg", "Flex/Ext", "flex"),
                                  ("hip_abd_deg", "Abd/Add", "abd"),
                                  ("hip_rot_deg", "Int/Ext Rot", "rot")]),

        # Row 3: Knees (R | L)
        ("knee_R", "Knee (Right)", [("knee_flex_deg", "Flex/Ext", "flex"),
                                     ("knee_abd_deg", "Varus/Valgus", "abd"),
                                     ("knee_rot_deg", "Tibial Rot", "rot")]),
        ("knee_L", "Knee (Left)", [("knee_flex_deg", "Flex/Ext", "flex"),
                                    ("knee_abd_deg", "Varus/Valgus", "abd"),
                                    ("knee_rot_deg", "Tibial Rot", "rot")]),

        # Row 4: Ankles (R | L)
        ("ankle_R", "Ankle (Right)", [("ankle_flex_deg", "Dorsi/Plantar", "flex"),
                                       ("ankle_abd_deg", "Inv/Eversion", "abd"),
                                       ("ankle_rot_deg", "Rotation", "rot")]),
        ("ankle_L", "Ankle (Left)", [("ankle_flex_deg", "Dorsi/Plantar", "flex"),
                                      ("ankle_abd_deg", "Inv/Eversion", "abd"),
                                      ("ankle_rot_deg", "Rotation", "rot")]),

        # Row 5: Shoulders (R | L)
        ("shoulder_R", "Shoulder (Right)", [("shoulder_exo_deg", "Exo/Endo", "exo"),
                                             ("shoulder_flex_deg", "Flex/Ext", "flex"),
                                             ("shoulder_abd_deg", "Abd/Add", "abd")]),
        ("shoulder_L", "Shoulder (Left)", [("shoulder_exo_deg", "Exo/Endo", "exo"),
                                            ("shoulder_flex_deg", "Flex/Ext", "flex"),
                                            ("shoulder_abd_deg", "Abd/Add", "abd")]),

        # Row 6: Elbows (R | L)
        ("elbow_R", "Elbow (Right)", [("elbow_flex_deg", "Flexion", "flex")]),
        ("elbow_L", "Elbow (Left)", [("elbow_flex_deg", "Flexion", "flex")]),
    ]

    # Map joints to subplot positions
    subplot_positions = {
        "pelvis": (0, slice(None)),  # Row 0, span columns
        "trunk": (1, slice(None)),   # Row 1, span columns
        "hip_R": (2, 0),
        "hip_L": (2, 1),
        "knee_R": (3, 0),
        "knee_L": (3, 1),
        "ankle_R": (4, 0),
        "ankle_L": (4, 1),
        "shoulder_R": (5, 0),
        "shoulder_L": (5, 1),
        "elbow_R": (6, 0),
        "elbow_L": (6, 1),
    }

    # Create subplots and plot data
    for joint_key, panel_title, dof_specs in joint_configs:
        if joint_key not in angle_results:
            continue

        df = angle_results[joint_key]
        times = df["time_s"].values

        # Get subplot position
        if joint_key not in subplot_positions:
            continue

        row, col = subplot_positions[joint_key]

        if isinstance(col, slice):  # Spanning subplot
            ax = fig.add_subplot(gs[row, col])
        else:
            ax = fig.add_subplot(gs[row, col])

        # Check if data exists
        has_data = False

        # Plot each DOF
        for col_name, label, color_key in dof_specs:
            if col_name in df.columns:
                angles = df[col_name].values

                # Check if has valid data
                if not np.all(np.isnan(angles)):
                    has_data = True
                    ax.plot(times, angles, label=label, color=colors.get(color_key, "#333333"),
                           linewidth=1.5, alpha=0.8)

        if has_data:
            ax.set_title(panel_title, fontsize=11, fontweight="bold", loc="left")
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("Angle (deg)", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            ax.tick_params(labelsize=8)

            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        else:
            # No data - add message
            ax.text(0.5, 0.5, f"{panel_title}\n(No data available)",
                   ha='center', va='center', fontsize=10, color='gray',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    # Add overall title
    if title_prefix:
        fig.suptitle(f"{title_prefix} — Comprehensive Joint Angles (ISB)",
                    fontsize=14, fontweight="bold", y=0.995)
    else:
        fig.suptitle("Comprehensive Joint Angles (ISB)",
                    fontsize=14, fontweight="bold", y=0.995)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[visualize_comprehensive] Saved figure: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_side_by_side_comparison(
    angle_results: Dict[str, pd.DataFrame],
    output_path: Optional[Path] = None,
    title_prefix: str = "",
    dpi: int = 150,
) -> None:
    """Create side-by-side comparison of right vs left joint angles.

    Args:
        angle_results: Dict of DataFrames from compute_all_joint_angles()
        output_path: If provided, save figure to this path
        title_prefix: Prefix for figure title
        dpi: Figure resolution
    """
    # Joint pairs to compare
    joint_pairs = [
        ("hip_R", "hip_L", "Hip"),
        ("knee_R", "knee_L", "Knee"),
        ("ankle_R", "ankle_L", "Ankle"),
        ("shoulder_R", "shoulder_L", "Shoulder"),
        ("elbow_R", "elbow_L", "Elbow"),
    ]

    fig, axes = plt.subplots(len(joint_pairs), 1, figsize=(12, 10), sharex=True)
    if len(joint_pairs) == 1:
        axes = [axes]

    colors_right = {"flex": "#1f77b4", "abd": "#ff7f0e", "rot": "#2ca02c", "exo": "#d62728"}
    colors_left = {"flex": "#aec7e8", "abd": "#ffbb78", "rot": "#98df8a", "exo": "#ff9896"}

    for idx, (right_key, left_key, joint_name) in enumerate(joint_pairs):
        ax = axes[idx]

        # Plot right side (solid lines)
        if right_key in angle_results:
            df_r = angle_results[right_key]
            times = df_r["time_s"].values

            for col in df_r.columns:
                if col == "time_s":
                    continue
                angles = df_r[col].values
                if not np.all(np.isnan(angles)):
                    # Extract DOF type
                    dof_type = "flex" if "flex" in col else ("abd" if "abd" in col else "rot" if "rot" in col else "exo")
                    label = col.replace("_deg", "").replace("_", " ").title() + " (R)"
                    ax.plot(times, angles, label=label, color=colors_right.get(dof_type, "#1f77b4"),
                           linestyle="-", linewidth=1.5, alpha=0.9)

        # Plot left side (dashed lines)
        if left_key in angle_results:
            df_l = angle_results[left_key]
            times = df_l["time_s"].values

            for col in df_l.columns:
                if col == "time_s":
                    continue
                angles = df_l[col].values
                if not np.all(np.isnan(angles)):
                    dof_type = "flex" if "flex" in col else ("abd" if "abd" in col else "rot" if "rot" in col else "exo")
                    label = col.replace("_deg", "").replace("_", " ").title() + " (L)"
                    ax.plot(times, angles, label=label, color=colors_left.get(dof_type, "#aec7e8"),
                           linestyle="--", linewidth=1.5, alpha=0.9)

        ax.set_title(joint_name, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel("Angle (deg)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=10)

    if title_prefix:
        fig.suptitle(f"{title_prefix} — Right vs Left Joint Angles",
                    fontsize=13, fontweight="bold")
    else:
        fig.suptitle("Right vs Left Joint Angles Comparison",
                    fontsize=13, fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[visualize_comprehensive] Saved comparison: {output_path}")
        plt.close()
    else:
        plt.show()


def save_comprehensive_angles_csv(
    angle_results: Dict[str, pd.DataFrame],
    output_dir: Path,
    basename: str,
) -> None:
    """Save all joint angle data to separate CSV files.

    Args:
        angle_results: Dict of DataFrames from compute_all_joint_angles()
        output_dir: Directory to save CSV files
        basename: Base name for files (e.g., "joey")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for joint_name, df in angle_results.items():
        csv_path = output_dir / f"{basename}_angles_{joint_name}.csv"
        df.to_csv(csv_path, index=False, float_format="%.3f")
        print(f"[save_angles] {csv_path.name}")

    print(f"[save_angles] Saved {len(angle_results)} CSV files to {output_dir}")
