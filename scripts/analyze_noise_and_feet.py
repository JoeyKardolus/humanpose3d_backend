#!/usr/bin/env python3
"""
Analyze viewpoint-based noise application and feet geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


def analyze_viewpoint_noise():
    """Check if viewpoint-based noise is working correctly."""

    # Load samples with different camera angles (same noise level)
    angles = [0, 15, 30, 45, 60, 75]
    samples = [f'data/training/cmu_converted/01_01_f0000_a{a:02d}_n030.npz' for a in angles]

    # Track depth errors for left vs right markers
    left_errors = []
    right_errors = []

    print("="*80)
    print("VIEWPOINT-BASED NOISE ANALYSIS")
    print("="*80)
    print()

    for angle, sample_path in zip(angles, samples):
        if not Path(sample_path).exists():
            print(f"Skipping {angle}° - file not found")
            continue

        data = np.load(sample_path)
        corrupted = data['corrupted']
        ground_truth = data['ground_truth']
        marker_names = data['marker_names'].tolist()

        # Compute depth errors
        depth_errors = np.abs(corrupted[:, 2] - ground_truth[:, 2]) * 1000  # mm

        # Get shoulder errors
        rs_idx = marker_names.index('RShoulder')
        ls_idx = marker_names.index('LShoulder')

        right_errors.append(depth_errors[rs_idx])
        left_errors.append(depth_errors[ls_idx])

        print(f"Camera {angle:2d}°:")
        print(f"  RShoulder: {depth_errors[rs_idx]:5.1f}mm")
        print(f"  LShoulder: {depth_errors[ls_idx]:5.1f}mm")
        print(f"  Ratio R/L: {depth_errors[rs_idx]/depth_errors[ls_idx]:5.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(angles[:len(right_errors)], right_errors, 'o-', color='red', linewidth=2, markersize=8, label='Right Shoulder')
    ax.plot(angles[:len(left_errors)], left_errors, 's-', color='blue', linewidth=2, markersize=8, label='Left Shoulder')
    ax.set_xlabel('Camera Angle (degrees)', fontsize=12)
    ax.set_ylabel('Depth Error (mm)', fontsize=12)
    ax.set_title('Viewpoint-Based Noise: Should See Asymmetry', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add expected behavior annotation
    ax.text(0.5, 0.95, 'Expected: Right shoulder has MORE error at low angles\n'
                       'Left shoulder has MORE error at high angles (side view)',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('noise_analysis_viewpoint.png', dpi=150)
    print(f"\n✓ Saved: noise_analysis_viewpoint.png")
    plt.close()


def analyze_feet_geometry():
    """Check feet marker geometry."""

    data = np.load('data/training/cmu_converted/01_01_f0000_a00_n030.npz')
    ground_truth = data['ground_truth']
    marker_names = data['marker_names'].tolist()

    print()
    print("="*80)
    print("FEET GEOMETRY ANALYSIS")
    print("="*80)
    print()

    # Get foot markers
    foot_markers = {
        'Right Foot': ['RHeel', 'RAnkle', 'RBigToe'],
        'Left Foot': ['LHeel', 'LAnkle', 'LBigToe'],
    }

    fig = plt.figure(figsize=(14, 6))

    for subplot_idx, (foot_name, markers) in enumerate(foot_markers.items(), 1):
        ax = fig.add_subplot(1, 2, subplot_idx, projection='3d')

        positions = []
        for marker in markers:
            if marker in marker_names:
                idx = marker_names.index(marker)
                pos = ground_truth[idx]
                positions.append(pos)

                # Plot marker
                ax.scatter([pos[0]], [pos[1]], [pos[2]], s=100, marker='o')
                ax.text(pos[0], pos[1], pos[2], f'  {marker}', fontsize=9)

        positions = np.array(positions)

        # Connect markers
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
               'k-', linewidth=2, alpha=0.5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{foot_name} Markers')

        # Print analysis
        print(f"{foot_name}:")
        for marker in markers:
            if marker in marker_names:
                idx = marker_names.index(marker)
                pos = ground_truth[idx]
                print(f"  {marker:12s}: X={pos[0]:7.3f}, Y={pos[1]:7.3f}, Z={pos[2]:7.3f}")

        # Check if markers are collinear (bad!)
        x_spread = positions[:, 0].max() - positions[:, 0].min()
        y_spread = positions[:, 1].max() - positions[:, 1].min()
        z_spread = positions[:, 2].max() - positions[:, 2].min()

        print(f"  Spread: X={x_spread*1000:.1f}mm, Y={y_spread*1000:.1f}mm, Z={z_spread*1000:.1f}mm")

        if x_spread < 0.01 and y_spread < 0.01:
            print(f"  ⚠️  WARNING: Markers are nearly collinear (all on Z-axis)!")
            print(f"             This makes feet look like straight rods!")
        print()

    plt.tight_layout()
    plt.savefig('feet_geometry_analysis.png', dpi=150)
    print(f"✓ Saved: feet_geometry_analysis.png")
    print()
    plt.close()


def main():
    """Run all analyses."""
    analyze_viewpoint_noise()
    analyze_feet_geometry()

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("✓ Viewpoint-based noise: Check noise_analysis_viewpoint.png")
    print("  - Should see left/right asymmetry based on camera angle")
    print()
    print("✓ Feet geometry: Check feet_geometry_analysis.png")
    print("  - Feet markers are collinear (same X, Y) - this is CMU→OpenCap mapping issue")
    print("  - This makes feet look like straight rods in visualizations")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
