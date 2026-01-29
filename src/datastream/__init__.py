"""Data stream module for TRC file processing and coordinate transforms.

Provides utilities for:
- TRC file I/O and manipulation
- Coordinate transforms between camera/Pose2Sim/kinematics conventions
- Marker data smoothing and processing
"""

from .trc_transforms import (
    camera_to_pose2sim,
    pose2sim_to_kinematics,
    kinematics_to_pose2sim,
    pose2sim_to_camera,
    parse_trc_to_array,
    array_to_trc,
    MARKER_NAMES,
)

from .trc_processing import (
    smooth_trc,
    hide_markers_in_trc,
)

__all__ = [
    # Transforms
    "camera_to_pose2sim",
    "pose2sim_to_kinematics",
    "kinematics_to_pose2sim",
    "pose2sim_to_camera",
    "parse_trc_to_array",
    "array_to_trc",
    "MARKER_NAMES",
    # Processing
    "smooth_trc",
    "hide_markers_in_trc",
]
