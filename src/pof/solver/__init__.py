"""Least-squares depth solving for POF reconstruction.

This subpackage provides depth solving functionality:
- solve_depth_least_squares_pof: Core depth solver
- enforce_bone_lengths: Bone length enforcement
- normalize_2d_for_pof, denormalize_pose_3d: Coordinate normalization
"""

# Import from parent least_squares for backward compatibility
from ..least_squares import (
    solve_depth_least_squares_pof,
    enforce_bone_lengths,
    normalize_2d_for_pof,
    denormalize_pose_3d,
)

__all__ = [
    "solve_depth_least_squares_pof",
    "enforce_bone_lengths",
    "normalize_2d_for_pof",
    "denormalize_pose_3d",
]
