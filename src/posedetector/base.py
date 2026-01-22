"""Base classes for pose estimation.

Defines the PoseEstimator protocol that all pose detectors must implement,
providing a unified interface for MediaPipe, RTMPose, and future estimators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class PoseDetectionResult:
    """Unified output format for all pose estimators.

    All estimators must convert their native output to this format,
    ensuring consistent downstream processing regardless of the
    detector used.

    Attributes:
        keypoints_2d: (N, 17, 2) normalized [0,1] image coordinates in COCO-17 format.
                      X increases right, Y increases down.
        keypoints_3d: (N, 17, 3) world coordinates in meters. Only provided by
                      estimators that support 3D (e.g., MediaPipe). None otherwise.
        visibility: (N, 17) per-joint confidence scores in [0, 1] range.
        timestamps: (N,) frame timestamps in seconds.
        image_size: (height, width) of the input images for denormalization.
        metadata: Estimator-specific additional data (e.g., segmentation masks).
    """

    keypoints_2d: np.ndarray  # (N, 17, 2) normalized [0,1]
    visibility: np.ndarray  # (N, 17) confidence scores
    timestamps: np.ndarray  # (N,) seconds
    image_size: tuple[int, int]  # (height, width)
    keypoints_3d: Optional[np.ndarray] = None  # (N, 17, 3) meters
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate shapes and types."""
        n_frames = len(self.timestamps)

        if self.keypoints_2d.shape != (n_frames, 17, 2):
            raise ValueError(
                f"keypoints_2d shape mismatch: expected ({n_frames}, 17, 2), "
                f"got {self.keypoints_2d.shape}"
            )

        if self.visibility.shape != (n_frames, 17):
            raise ValueError(
                f"visibility shape mismatch: expected ({n_frames}, 17), "
                f"got {self.visibility.shape}"
            )

        if self.keypoints_3d is not None:
            if self.keypoints_3d.shape != (n_frames, 17, 3):
                raise ValueError(
                    f"keypoints_3d shape mismatch: expected ({n_frames}, 17, 3), "
                    f"got {self.keypoints_3d.shape}"
                )

    @property
    def num_frames(self) -> int:
        """Number of frames in the result."""
        return len(self.timestamps)

    @property
    def has_3d(self) -> bool:
        """Whether this result includes 3D world coordinates."""
        return self.keypoints_3d is not None

    def get_frame(self, idx: int) -> Dict[str, np.ndarray]:
        """Get keypoints for a single frame.

        Args:
            idx: Frame index.

        Returns:
            Dict with 'keypoints_2d', 'visibility', 'timestamp', and
            optionally 'keypoints_3d'.
        """
        result = {
            "keypoints_2d": self.keypoints_2d[idx],
            "visibility": self.visibility[idx],
            "timestamp": self.timestamps[idx],
        }
        if self.keypoints_3d is not None:
            result["keypoints_3d"] = self.keypoints_3d[idx]
        return result

    def filter_by_visibility(
        self, min_visibility: float = 0.3
    ) -> "PoseDetectionResult":
        """Create a copy with low-visibility keypoints masked.

        Args:
            min_visibility: Minimum confidence threshold.

        Returns:
            New PoseDetectionResult with low-visibility joints set to NaN.
        """
        mask = self.visibility < min_visibility

        keypoints_2d = self.keypoints_2d.copy()
        keypoints_2d[mask] = np.nan

        keypoints_3d = None
        if self.keypoints_3d is not None:
            keypoints_3d = self.keypoints_3d.copy()
            keypoints_3d[mask] = np.nan

        return PoseDetectionResult(
            keypoints_2d=keypoints_2d,
            visibility=self.visibility.copy(),
            timestamps=self.timestamps.copy(),
            image_size=self.image_size,
            keypoints_3d=keypoints_3d,
            metadata=self.metadata.copy(),
        )


class PoseEstimator(ABC):
    """Abstract base class for 2D/3D pose estimators.

    All pose estimators must implement this interface to ensure
    consistent integration with the pipeline. The interface is
    designed to be minimal while supporting both 2D-only (RTMPose)
    and 2D+3D (MediaPipe) estimators.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the estimator (e.g., 'mediapipe', 'rtmpose')."""
        pass

    @property
    @abstractmethod
    def provides_3d(self) -> bool:
        """Whether this estimator outputs 3D world coordinates.

        If True, the detect() method will populate keypoints_3d.
        If False, 3D must be computed downstream (e.g., via POF model).
        """
        pass

    @abstractmethod
    def detect(
        self,
        frames: np.ndarray,
        fps: float,
        visibility_min: float = 0.3,
    ) -> PoseDetectionResult:
        """Run pose detection on video frames.

        Args:
            frames: (N, H, W, 3) RGB video frames as uint8 or float32.
            fps: Frame rate for computing timestamps.
            visibility_min: Minimum confidence threshold. Joints below this
                           threshold may be excluded or marked.

        Returns:
            PoseDetectionResult with COCO-17 format keypoints.
        """
        pass

    def detect_single(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        visibility_min: float = 0.3,
    ) -> PoseDetectionResult:
        """Convenience method for single-frame detection.

        Args:
            frame: (H, W, 3) single RGB frame.
            timestamp: Timestamp for this frame.
            visibility_min: Minimum confidence threshold.

        Returns:
            PoseDetectionResult with single frame.
        """
        frames = frame[np.newaxis, ...]  # (1, H, W, 3)
        result = self.detect(frames, fps=1.0, visibility_min=visibility_min)
        # Override timestamp
        result.timestamps[0] = timestamp
        return result


# COCO-17 joint names in standard order
COCO_KEYPOINT_NAMES: List[str] = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Mapping from COCO-17 indices to marker names used in the pipeline
COCO_TO_MARKER_NAME: Dict[int, str] = {
    0: "Nose",
    5: "LShoulder",
    6: "RShoulder",
    7: "LElbow",
    8: "RElbow",
    9: "LWrist",
    10: "RWrist",
    11: "LHip",
    12: "RHip",
    13: "LKnee",
    14: "RKnee",
    15: "LAnkle",
    16: "RAnkle",
}
