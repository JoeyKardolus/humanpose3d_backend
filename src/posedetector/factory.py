"""Factory for creating pose estimators.

Provides a unified interface for instantiating different pose detection
backends (MediaPipe, RTMPose) with appropriate model configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

from .base import PoseEstimator


EstimatorType = Literal["mediapipe", "rtmpose"]


def create_pose_estimator(
    estimator_type: EstimatorType = "mediapipe",
    model_path: Optional[Path | str] = None,
    rtmpose_model_size: str = "m",
) -> PoseEstimator:
    """Create a pose estimator of the specified type.

    Args:
        estimator_type: Type of estimator ('mediapipe' or 'rtmpose').
        model_path: Path to model file. For MediaPipe, this is the .task file.
                   For RTMPose, if None, auto-downloads the model.
        rtmpose_model_size: RTMPose model size ('s', 'm', 'l'). Default 'm'.
                           Only used when estimator_type='rtmpose'.

    Returns:
        Configured PoseEstimator instance.

    Raises:
        ValueError: If estimator_type is not supported.
        FileNotFoundError: If model_path is specified but doesn't exist.
    """
    if estimator_type == "mediapipe":
        from .mediapipe_detector import MediaPipeDetector

        if model_path is None:
            # Default MediaPipe model path
            default_path = Path("models/pose_landmarker_heavy.task")
            if not default_path.exists():
                raise FileNotFoundError(
                    f"MediaPipe model not found at {default_path}. "
                    "Download from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker"
                )
            model_path = default_path

        return MediaPipeDetector(model_path)

    elif estimator_type == "rtmpose":
        from .rtmpose_detector import RTMPoseDetector

        return RTMPoseDetector(model_size=rtmpose_model_size)

    else:
        raise ValueError(
            f"Unknown estimator type: {estimator_type}. "
            f"Supported types: mediapipe, rtmpose"
        )


def list_available_estimators() -> list[str]:
    """List available pose estimator types.

    Returns:
        List of estimator type names.
    """
    return ["mediapipe", "rtmpose"]
