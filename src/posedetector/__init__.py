"""Pose detection module with pluggable backends.

Provides a unified interface for 2D/3D pose estimation from video frames.
Supports MediaPipe (default) and RTMPose backends.

Example:
    >>> from src.posedetector import create_pose_estimator
    >>> estimator = create_pose_estimator("mediapipe")
    >>> result = estimator.detect(frames, fps=30.0)
    >>> print(result.keypoints_2d.shape)  # (N, 17, 2)

Note: Detector imports are lazy to avoid loading TensorFlow/MediaPipe on startup.
      Use create_pose_estimator() or import the detector class directly when needed.
"""

from .base import (
    PoseEstimator,
    PoseDetectionResult,
    COCO_KEYPOINT_NAMES,
    COCO_TO_MARKER_NAME,
)
from .factory import create_pose_estimator, list_available_estimators

# Lazy imports for detectors - these load heavy dependencies (TensorFlow, ONNX)
# Import them directly when needed: from src.posedetector.mediapipe_detector import MediaPipeDetector


def __getattr__(name):
    """Lazy import for detector classes and their constants."""
    if name in ("MediaPipeDetector", "MEDIAPIPE_TO_COCO", "COCO_TO_MEDIAPIPE", "MEDIAPIPE_TO_MARKER_NAME"):
        from . import mediapipe_detector
        return getattr(mediapipe_detector, name)
    elif name in ("RTMPoseDetector", "RTMPOSE_MODELS"):
        from . import rtmpose_detector
        return getattr(rtmpose_detector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "PoseEstimator",
    "PoseDetectionResult",
    # Factory
    "create_pose_estimator",
    "list_available_estimators",
    # MediaPipe (lazy)
    "MediaPipeDetector",
    "MEDIAPIPE_TO_COCO",
    "COCO_TO_MEDIAPIPE",
    "MEDIAPIPE_TO_MARKER_NAME",
    # RTMPose (lazy)
    "RTMPoseDetector",
    "RTMPOSE_MODELS",
    # Constants
    "COCO_KEYPOINT_NAMES",
    "COCO_TO_MARKER_NAME",
]
