"""Visualization utilities for camera-space POF.

Provides functions for:
- Plotting POF vectors in 3D
- Visualizing reconstructed skeletons
- Comparing ground truth vs predicted poses
- Debug visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple, Dict
from pathlib import Path

from .constants import (
    LIMB_DEFINITIONS,
    LIMB_NAMES,
    COCO_JOINT_NAMES,
    NUM_LIMBS,
    NUM_JOINTS,
)


# Bone connections for skeleton visualization (using COCO-17 indices)
SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    # Torso
    (5, 6),    # Shoulders
    (11, 12),  # Hips
    (5, 11),   # L side
    (6, 12),   # R side
    # Left arm
    (5, 7),    # L shoulder -> elbow
    (7, 9),    # L elbow -> wrist
    # Right arm
    (6, 8),    # R shoulder -> elbow
    (8, 10),   # R elbow -> wrist
    # Left leg
    (11, 13),  # L hip -> knee
    (13, 15),  # L knee -> ankle
    # Right leg
    (12, 14),  # R hip -> knee
    (14, 16),  # R knee -> ankle
    # Head
    (0, 5),    # Nose to L shoulder
    (0, 6),    # Nose to R shoulder
]


def plot_pof_vectors(
    pof: np.ndarray,
    pose_3d: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "POF Vectors",
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Visualize POF unit vectors in 3D.

    Shows each limb's orientation as an arrow from origin.
    Optionally shows skeleton context.

    Args:
        pof: (14, 3) unit vectors
        pose_3d: Optional (17, 3) 3D pose for context
        output_path: Save figure to path (show if None)
        title: Figure title
        figsize: Figure size in inches
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Color map for limbs
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_LIMBS))

    # Plot POF vectors from origin
    for i, (parent, child) in enumerate(LIMB_DEFINITIONS):
        vec = pof[i]
        ax.quiver(
            0, 0, 0,
            vec[0], vec[1], vec[2],
            color=colors[i],
            label=LIMB_NAMES[i],
            arrow_length_ratio=0.1,
            linewidth=2,
        )

    # If pose provided, show skeleton with offset
    if pose_3d is not None:
        offset = np.array([2.0, 0, 0])  # Offset skeleton to the right
        pose_offset = pose_3d + offset

        # Plot joints
        ax.scatter(
            pose_offset[:, 0],
            pose_offset[:, 1],
            pose_offset[:, 2],
            c="blue",
            s=50,
            alpha=0.7,
        )

        # Plot bones
        for parent, child in SKELETON_CONNECTIONS:
            ax.plot(
                [pose_offset[parent, 0], pose_offset[child, 0]],
                [pose_offset[parent, 1], pose_offset[child, 1]],
                [pose_offset[parent, 2], pose_offset[child, 2]],
                "b-",
                alpha=0.5,
                linewidth=2,
            )

    # Axis labels
    ax.set_xlabel("X (left/right)")
    ax.set_ylabel("Y (up/down)")
    ax.set_zlabel("Z (depth)")
    ax.set_title(title)

    # Legend (outside plot for readability)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.05, 1))

    # Equal aspect ratio
    max_range = 1.5
    ax.set_xlim(-max_range, max_range + 2)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved POF visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_skeleton_3d(
    pose_3d: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "3D Skeleton",
    color: str = "blue",
    figsize: Tuple[int, int] = (10, 10),
) -> None:
    """Visualize 3D skeleton.

    Args:
        pose_3d: (17, 3) joint positions
        output_path: Save figure to path (show if None)
        title: Figure title
        color: Skeleton color
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot joints
    ax.scatter(
        pose_3d[:, 0],
        pose_3d[:, 1],
        pose_3d[:, 2],
        c=color,
        s=50,
        zorder=5,
    )

    # Plot bones
    for parent, child in SKELETON_CONNECTIONS:
        ax.plot(
            [pose_3d[parent, 0], pose_3d[child, 0]],
            [pose_3d[parent, 1], pose_3d[child, 1]],
            [pose_3d[parent, 2], pose_3d[child, 2]],
            f"{color[0]}-",
            linewidth=2,
        )

    # Joint labels
    for i, name in enumerate(COCO_JOINT_NAMES):
        ax.text(
            pose_3d[i, 0],
            pose_3d[i, 1],
            pose_3d[i, 2],
            f"  {i}",
            fontsize=8,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Set equal aspect ratio
    _set_axes_equal(ax)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved skeleton visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_reconstruction_comparison(
    gt_pose: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "POF Reconstruction vs Ground Truth",
    figsize: Tuple[int, int] = (16, 6),
) -> None:
    """Compare ground truth and reconstructed poses side by side.

    Args:
        gt_pose: (17, 3) ground truth pose
        reconstructed: (17, 3) reconstructed pose
        output_path: Save figure to path (show if None)
        title: Figure title
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)

    # Ground truth
    ax1 = fig.add_subplot(131, projection="3d")
    _plot_skeleton_on_axis(ax1, gt_pose, "blue", "Ground Truth")

    # Reconstructed
    ax2 = fig.add_subplot(132, projection="3d")
    _plot_skeleton_on_axis(ax2, reconstructed, "red", "Reconstructed")

    # Overlay comparison
    ax3 = fig.add_subplot(133, projection="3d")
    _plot_skeleton_on_axis(ax3, gt_pose, "blue", "Overlay", alpha=0.6)
    _plot_skeleton_on_axis(ax3, reconstructed, "red", None, alpha=0.6)
    ax3.set_title("Overlay (Blue=GT, Red=Pred)")

    # Compute error
    error = np.linalg.norm(gt_pose - reconstructed, axis=-1).mean()
    fig.suptitle(f"{title}\nMean Error: {error * 100:.2f} cm")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_pof_error_distribution(
    pred_pof: np.ndarray,
    gt_pof: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "POF Angular Error Distribution",
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """Plot angular error distribution per limb.

    Args:
        pred_pof: (N, 14, 3) or (14, 3) predicted POF
        gt_pof: (N, 14, 3) or (14, 3) ground truth POF
        output_path: Save figure to path
        title: Figure title
        figsize: Figure size
    """
    if pred_pof.ndim == 2:
        pred_pof = pred_pof[np.newaxis, ...]
        gt_pof = gt_pof[np.newaxis, ...]

    # Compute angular errors
    cos_sim = (pred_pof * gt_pof).sum(axis=-1).clip(-1, 1)
    angles_deg = np.arccos(cos_sim) * (180 / np.pi)  # (N, 14)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Box plot per limb
    ax1 = axes[0]
    ax1.boxplot(angles_deg, labels=[LIMB_NAMES[i][:8] for i in range(NUM_LIMBS)])
    ax1.set_ylabel("Angular Error (degrees)")
    ax1.set_xlabel("Limb")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_title("Per-Limb Error")

    # Histogram of all errors
    ax2 = axes[1]
    ax2.hist(angles_deg.flatten(), bins=50, edgecolor="black")
    ax2.set_xlabel("Angular Error (degrees)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Overall Distribution\nMean: {angles_deg.mean():.2f}°")
    ax2.axvline(angles_deg.mean(), color="red", linestyle="--", label="Mean")
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved error distribution to {output_path}")
    else:
        plt.show()

    plt.close()


def _plot_skeleton_on_axis(
    ax: Axes3D,
    pose: np.ndarray,
    color: str,
    title: Optional[str],
    alpha: float = 1.0,
) -> None:
    """Helper to plot skeleton on given axis."""
    ax.scatter(
        pose[:, 0],
        pose[:, 1],
        pose[:, 2],
        c=color,
        s=30,
        alpha=alpha,
    )

    for parent, child in SKELETON_CONNECTIONS:
        ax.plot(
            [pose[parent, 0], pose[child, 0]],
            [pose[parent, 1], pose[child, 1]],
            [pose[parent, 2], pose[child, 2]],
            f"{color[0]}-",
            linewidth=2,
            alpha=alpha,
        )

    if title:
        ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    _set_axes_equal(ax)


def _set_axes_equal(ax: Axes3D) -> None:
    """Set equal aspect ratio for 3D axis."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    center = limits.mean(axis=1)
    radius = 0.5 * (limits[:, 1] - limits[:, 0]).max()

    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])


def create_debug_visualization(
    keypoints_2d: np.ndarray,
    visibility: np.ndarray,
    pred_pof: np.ndarray,
    gt_pof: Optional[np.ndarray] = None,
    gt_pose: Optional[np.ndarray] = None,
    reconstructed: Optional[np.ndarray] = None,
    output_dir: str = "debug_output",
    prefix: str = "debug",
) -> None:
    """Create comprehensive debug visualizations.

    Args:
        keypoints_2d: (17, 2) 2D keypoints
        visibility: (17,) visibility scores
        pred_pof: (14, 3) predicted POF
        gt_pof: Optional (14, 3) ground truth POF
        gt_pose: Optional (17, 3) ground truth 3D pose
        reconstructed: Optional (17, 3) reconstructed pose
        output_dir: Directory for output files
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot predicted POF vectors
    plot_pof_vectors(
        pred_pof,
        pose_3d=reconstructed,
        output_path=str(output_dir / f"{prefix}_pof_vectors.png"),
        title="Predicted POF Vectors",
    )

    # 2. If reconstructed, plot skeleton
    if reconstructed is not None:
        plot_skeleton_3d(
            reconstructed,
            output_path=str(output_dir / f"{prefix}_reconstructed.png"),
            title="Reconstructed 3D Skeleton",
            color="red",
        )

    # 3. If GT available, plot comparison
    if gt_pose is not None and reconstructed is not None:
        plot_reconstruction_comparison(
            gt_pose,
            reconstructed,
            output_path=str(output_dir / f"{prefix}_comparison.png"),
        )

    # 4. If GT POF available, plot error distribution
    if gt_pof is not None:
        plot_pof_error_distribution(
            pred_pof[np.newaxis, ...],
            gt_pof[np.newaxis, ...],
            output_path=str(output_dir / f"{prefix}_pof_errors.png"),
        )

    print(f"Debug visualizations saved to {output_dir}/")
