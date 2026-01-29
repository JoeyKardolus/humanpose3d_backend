"""Shared utilities for HumanPose3D pipeline.

This module contains reusable components extracted from various modules:
- nn_layers: Neural network building blocks (GCN, GAT)
- training: Training utilities (device setup, checkpointing)
- constants: COCO skeleton definitions and visualization colors
- suppress_warnings: Common warning suppression for TensorFlow/ONNX
"""

from .nn_layers import GCNLayer, GATLayer
from .training import setup_device, setup_amp, save_checkpoint, load_checkpoint
from .constants import (
    COCO_SKELETON_CONNECTIONS,
    COCO_JOINT_NAMES,
    JOINT_COLORS,
    LIMB_COLORS,
)
from .suppress_warnings import suppress_common_warnings

__all__ = [
    # NN layers
    "GCNLayer",
    "GATLayer",
    # Training
    "setup_device",
    "setup_amp",
    "save_checkpoint",
    "load_checkpoint",
    # Constants
    "COCO_SKELETON_CONNECTIONS",
    "COCO_JOINT_NAMES",
    "JOINT_COLORS",
    "LIMB_COLORS",
    # Utilities
    "suppress_common_warnings",
]
