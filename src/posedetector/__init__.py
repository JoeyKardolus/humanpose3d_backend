"""Pose detection module with pluggable backends.

Provides a unified interface for 2D/3D pose estimation from video frames.
Supports MediaPipe (default) and RTMPose backends.

Example:
    >>> from src.posedetector import create_pose_estimator
    >>> estimator = create_pose_estimator("mediapipe")
    >>> result = estimator.detect(frames, fps=30.0)
    >>> print(result.keypoints_2d.shape)  # (N, 17, 2)
"""

from .base import (
    PoseEstimator,
    PoseDetectionResult,
    COCO_KEYPOINT_NAMES,
    COCO_TO_MARKER_NAME,
)
from .factory import create_pose_estimator, list_available_estimators
from .mediapipe_detector import (
    MediaPipeDetector,
    MEDIAPIPE_TO_COCO,
    COCO_TO_MEDIAPIPE,
    MEDIAPIPE_TO_MARKER_NAME,
)
from .rtmpose_detector import RTMPoseDetector, RTMPOSE_MODELS

__all__ = [
    # Base classes
    "PoseEstimator",
    "PoseDetectionResult",
    # Factory
    "create_pose_estimator",
    "list_available_estimators",
    # MediaPipe
    "MediaPipeDetector",
    "MEDIAPIPE_TO_COCO",
    "COCO_TO_MEDIAPIPE",
    "MEDIAPIPE_TO_MARKER_NAME",
    # RTMPose
    "RTMPoseDetector",
    "RTMPOSE_MODELS",
    # Constants
    "COCO_KEYPOINT_NAMES",
    "COCO_TO_MARKER_NAME",
]
