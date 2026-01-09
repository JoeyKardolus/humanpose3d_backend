"""Visualization of joint angle time series.

Provides publication-quality plots of joint angles matching biomechanics
standards. Generates 3-panel plots (hip, knee, ankle) with all 3 DOF per joint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_joint_angles_time_series(
    angles_df: pd.DataFrame,
    side: Literal["R", "L"],
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 150,
    show: bool = False,
) -> None:
    """Plot joint angles as 3-panel time series (hip, knee, ankle).

    Creates a figure with 3 vertically stacked subplots matching the style
    from the toevoegen visualization example.

    Args:
        angles_df: DataFrame with columns from compute_lower_limb_angles()
        side: "R" or "L" for plot labels
        output_path: Where to save figure (PNG). If None, only displays.
        title: Optional main title. Defaults to "Joint Angles - {side} Leg"
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure
        show: Whether to call plt.show() (for interactive use)
    """
    if title is None:
        title = f"Joint Angles — {side} Leg"

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    time = angles_df["time_s"].to_numpy()

    # Colors matching the reference visualization
    color_flex = "#1f77b4"  # Blue
    color_abd = "#ff7f0e"   # Orange
    color_rot = "#2ca02c"   # Green

    # --- Hip panel (top) ---
    ax = axes[0]
    ax.plot(time, angles_df["hip_flex_deg"], label="Hip Flex/Ext", color=color_flex, linewidth=1.2)
    ax.plot(time, angles_df["hip_abd_deg"], label="Hip Abd/Add", color=color_abd, linewidth=1.2)
    ax.plot(time, angles_df["hip_rot_deg"], label="Hip Rot", color=color_rot, linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_ylabel("Angle (deg)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Hip", fontsize=11, fontweight="bold", loc="left")

    # --- Knee panel (middle) ---
    ax = axes[1]
    ax.plot(time, angles_df["knee_flex_deg"], label="Knee Flex/Ext", color=color_flex, linewidth=1.2)
    ax.plot(time, angles_df["knee_abd_deg"], label="Knee Abd/Add", color=color_abd, linewidth=1.2)
    ax.plot(time, angles_df["knee_rot_deg"], label="Knee Rot", color=color_rot, linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_ylabel("Angle (deg)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Knee", fontsize=11, fontweight="bold", loc="left")

    # --- Ankle panel (bottom) ---
    ax = axes[2]
    ax.plot(time, angles_df["ankle_flex_deg"], label="Ankle Flex/Ext", color=color_flex, linewidth=1.2)
    ax.plot(time, angles_df["ankle_abd_deg"], label="Ankle Abd/Add", color=color_abd, linewidth=1.2)
    ax.plot(time, angles_df["ankle_rot_deg"], label="Ankle Rot", color=color_rot, linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_ylabel("Angle (deg)", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Ankle", fontsize=11, fontweight="bold", loc="left")

    # Overall title
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved joint angle plot: {output_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cycle_normalized(
    angles_df: pd.DataFrame,
    cycle_starts: np.ndarray,
    cycle_ends: np.ndarray,
    side: Literal["R", "L"],
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    n_points: int = 101,
    show: bool = False,
) -> None:
    """Plot cycle-normalized joint angles (0-100% of gait cycle).

    Overlays all detected cycles and shows mean ± SD envelope.

    Args:
        angles_df: DataFrame with joint angles
        cycle_starts: Frame indices for cycle starts (e.g., heel strikes)
        cycle_ends: Frame indices for cycle ends
        side: "R" or "L"
        output_path: Where to save figure
        title: Optional title
        n_points: Number of points in normalized cycle (default 101 for 0-100%)
        show: Whether to display interactively
    """
    if title is None:
        title = f"Gait Cycles — {side} Leg (0-100% normalized)"

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

    angle_cols = [
        ("hip_flex_deg", "hip_abd_deg", "hip_rot_deg"),
        ("knee_flex_deg", "knee_abd_deg", "knee_rot_deg"),
        ("ankle_flex_deg", "ankle_abd_deg", "ankle_rot_deg"),
    ]

    joint_names = ["Hip", "Knee", "Ankle"]
    dof_names = ["Flex/Ext", "Abd/Add", "Rotation"]

    for row, (joint_cols, joint_name) in enumerate(zip(angle_cols, joint_names)):
        for col, (angle_col, dof_name) in enumerate(zip(joint_cols, dof_names)):
            ax = axes[row, col]

            # Normalize all cycles
            normalized_cycles = []
            for start, end in zip(cycle_starts, cycle_ends):
                if end <= start or start < 0 or end >= len(angles_df):
                    continue

                segment = angles_df[angle_col].iloc[start:end+1].to_numpy()

                if len(segment) < 5:  # Too short
                    continue

                # Resample to 0-100%
                x_old = np.linspace(0, 100, len(segment))
                x_new = np.linspace(0, 100, n_points)
                resampled = np.interp(x_new, x_old, segment)

                normalized_cycles.append(resampled)

            if not normalized_cycles:
                ax.text(0.5, 0.5, "No valid cycles", ha="center", va="center", transform=ax.transAxes)
                continue

            # Convert to array and compute statistics
            cycles_array = np.array(normalized_cycles)  # (n_cycles, n_points)
            mean_curve = np.nanmean(cycles_array, axis=0)
            std_curve = np.nanstd(cycles_array, axis=0)

            x_pct = np.linspace(0, 100, n_points)

            # Plot individual cycles (transparent)
            for cycle in normalized_cycles:
                ax.plot(x_pct, cycle, color="gray", alpha=0.2, linewidth=0.8)

            # Plot mean ± SD
            ax.plot(x_pct, mean_curve, color="black", linewidth=2, label="Mean")
            ax.fill_between(
                x_pct,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color="blue",
                alpha=0.2,
                label="±1 SD"
            )

            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)
            ax.set_ylabel("Angle (deg)", fontsize=9)
            ax.set_title(f"{joint_name} — {dof_name}", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)

            if row == 0 and col == 2:  # Top-right panel
                ax.legend(fontsize=8)

            if row == 2:  # Bottom row
                ax.set_xlabel("Cycle %", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved cycle-normalized plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_upper_body_angles(
    angles_df: pd.DataFrame,
    side: Literal["R", "L"],
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 150,
    show: bool = False,
) -> None:
    """Plot upper body joint angles in 3-panel format.

    Creates a 3-panel time-series plot:
    - Top: Trunk angles (flexion/extension, lateral flexion, rotation)
    - Middle: Shoulder angles (exo/endorotation, flexion/extension, abduction/adduction)
    - Bottom: Elbow flexion

    Args:
        angles_df: DataFrame with upper body angle columns
        side: "R" or "L" for right/left side
        output_path: Optional path to save PNG
        title: Optional plot title
        figsize: Figure size (width, height)
        dpi: Output resolution
        show: Show interactive plot window

    Expected DataFrame columns:
        - time_s
        - trunk_flex_deg, trunk_lateral_deg, trunk_rot_deg
        - shoulder_exo_deg, shoulder_flex_deg, shoulder_abd_deg
        - elbow_flex_deg
    """
    time = angles_df["time_s"].values

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Panel 1: Trunk
    axes[0].plot(time, angles_df["trunk_flex_deg"], color="#1f77b4", linewidth=1.5, label="Flexion/Extension")
    axes[0].plot(time, angles_df["trunk_lateral_deg"], color="#ff7f0e", linewidth=1.5, label="Lateral Flexion")
    axes[0].plot(time, angles_df["trunk_rot_deg"], color="#2ca02c", linewidth=1.5, label="Rotation")
    axes[0].axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Angle (deg)", fontsize=10)
    axes[0].set_title("Trunk", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Shoulder
    axes[1].plot(time, angles_df["shoulder_exo_deg"], color="#1f77b4", linewidth=1.5, label="Exo/Endo Rotation")
    axes[1].plot(time, angles_df["shoulder_flex_deg"], color="#ff7f0e", linewidth=1.5, label="Flexion/Extension")
    axes[1].plot(time, angles_df["shoulder_abd_deg"], color="#2ca02c", linewidth=1.5, label="Abduction/Adduction")
    axes[1].axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Angle (deg)", fontsize=10)
    axes[1].set_title("Shoulder", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Elbow
    axes[2].plot(time, angles_df["elbow_flex_deg"], color="#d62728", linewidth=1.5, label="Flexion")
    axes[2].axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)
    axes[2].set_ylabel("Angle (deg)", fontsize=10)
    axes[2].set_xlabel("Time (s)", fontsize=10)
    axes[2].set_title("Elbow", fontsize=11, fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # Overall title
    if title is None:
        title = f"Upper Body Joint Angles ({side} Arm)"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved upper body angle plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

