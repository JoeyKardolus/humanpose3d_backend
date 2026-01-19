"""Joint kinematics computation and visualization.

This module provides tools for computing 3D joint angles from augmented TRC files
using anatomically correct Euler decompositions. It supports:
- Full body: pelvis, hip, knee, ankle, trunk, shoulder, elbow (both sides)
- ISB-compliant coordinate systems and Euler decompositions

Key modules:
- segment_coordinate_systems: Build ISB-compliant anatomical reference frames
- comprehensive_joint_angles: Compute ALL joint angles (lower + upper body)
- visualize_comprehensive_angles: Multi-panel plots for all joints
- angle_processing: Utilities for smoothing, unwrapping, and zeroing
- trc_utils: TRC file reading utilities
"""

from .comprehensive_joint_angles import compute_all_joint_angles
from .trc_utils import get_marker, read_trc
from .visualize_comprehensive_angles import (
    plot_comprehensive_joint_angles,
    plot_side_by_side_comparison,
    save_comprehensive_angles_csv,
)

__all__ = [
    "compute_all_joint_angles",
    "read_trc",
    "get_marker",
    "plot_comprehensive_joint_angles",
    "plot_side_by_side_comparison",
    "save_comprehensive_angles_csv",
]
