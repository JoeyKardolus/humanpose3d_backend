"""Joint kinematics computation and visualization.

This module provides tools for computing 3D joint angles from augmented TRC files
using anatomically correct Euler decompositions. It supports:
- Lower limb: hip, knee, ankle (3-DOF each, XYZ Euler)
- Upper body: trunk, shoulder (3-DOF, XYZ and ZXY Euler), elbow (geometric flexion)
- Comprehensive: all joints (pelvis, lower body, trunk, upper body) in one call

Key modules:
- segment_coordinate_systems: Build ISB-compliant anatomical reference frames
- joint_angles_euler: Compute lower limb Euler angles with unwrapping and filtering
- joint_angles_upper_body: Compute upper body angles (trunk, shoulder, elbow)
- comprehensive_joint_angles: Compute ALL joint angles (lower + upper body)
- visualize_angles: Generate time-series plots matching biomechanics standards
- visualize_comprehensive_angles: Multi-panel plots for all joints
- angle_processing: Utilities for smoothing, unwrapping, and zeroing
"""

from .comprehensive_joint_angles import compute_all_joint_angles
from .joint_angles_euler import compute_lower_limb_angles
from .joint_angles_upper_body import compute_upper_body_angles
from .visualize_angles import plot_joint_angles_time_series, plot_upper_body_angles
from .visualize_comprehensive_angles import (
    plot_comprehensive_joint_angles,
    plot_side_by_side_comparison,
    save_comprehensive_angles_csv,
)

__all__ = [
    "compute_lower_limb_angles",
    "compute_upper_body_angles",
    "compute_all_joint_angles",
    "plot_joint_angles_time_series",
    "plot_upper_body_angles",
    "plot_comprehensive_joint_angles",
    "plot_side_by_side_comparison",
    "save_comprehensive_angles_csv",
]
