"""Marker augmentation module using Pose2Sim LSTM.

Provides marker augmentation from 21/22 original markers to 64 markers.
"""

from .markeraugmentation import (
    run_pose2sim_augment,
    ALL_MARKERS_64,
    ORIGINAL_MARKERS,
    LOWER_BODY_AUGMENTED,
    UPPER_BODY_AUGMENTED,
)

from .gpu_config import patch_pose2sim_gpu

__all__ = [
    "run_pose2sim_augment",
    "patch_pose2sim_gpu",
    "ALL_MARKERS_64",
    "ORIGINAL_MARKERS",
    "LOWER_BODY_AUGMENTED",
    "UPPER_BODY_AUGMENTED",
]
