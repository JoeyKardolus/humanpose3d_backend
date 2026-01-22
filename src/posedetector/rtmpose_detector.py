"""RTMPose pose estimator implementation.

Uses rtmlib for efficient ONNX Runtime-based inference.
Provides 2D keypoints only (no 3D) in COCO-17 format natively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import PoseEstimator, PoseDetectionResult


# RTMPose mode configurations
# rtmlib uses "mode" parameter to select model complexity
RTMPOSE_MODELS = {
    "s": {
        "mode": "lightweight",
        "desc": "Small - fastest, lower accuracy (uses lightweight mode)",
    },
    "m": {
        "mode": "balanced",
        "desc": "Medium - balanced speed/accuracy (recommended)",
    },
    "l": {
        "mode": "performance",
        "desc": "Large - highest accuracy, slower",
    },
}


class RTMPoseDetector(PoseEstimator):
    """RTMPose-based pose estimator using rtmlib.

    RTMPose provides higher 2D accuracy than MediaPipe (~3-4px vs 6-8px)
    but outputs 2D only. For 3D reconstruction, use with the POF model.

    Attributes:
        model_size: Size of RTMPose model ('s', 'm', 'l').
    """

    def __init__(
        self,
        model_size: str = "m",
        device: str = "cuda",
        backend: str = "onnxruntime",
    ):
        """Initialize RTMPose detector.

        Args:
            model_size: Model size - 's' (small), 'm' (medium), 'l' (large).
            device: Device for inference ('cuda' or 'cpu').
            backend: Inference backend ('onnxruntime' recommended).
        """
        if model_size not in RTMPOSE_MODELS:
            raise ValueError(
                f"Invalid model_size '{model_size}'. "
                f"Choose from: {list(RTMPOSE_MODELS.keys())}"
            )

        self.model_size = model_size
        self.device = device
        self.backend = backend
        self._model = None

    def _ensure_model_loaded(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        try:
            from rtmlib import Wholebody
        except ImportError:
            raise ImportError(
                "rtmlib not installed. Install with: uv pip install rtmlib"
            )

        config = RTMPOSE_MODELS[self.model_size]

        # rtmlib auto-downloads models from OpenMMLab hub
        # mode: 'lightweight', 'balanced', or 'performance'
        self._model = Wholebody(
            mode=config["mode"],
            backend=self.backend,
            device=self.device,
        )

    @property
    def name(self) -> str:
        return f"rtmpose-{self.model_size}"

    @property
    def provides_3d(self) -> bool:
        return False  # RTMPose is 2D only

    def detect(
        self,
        frames: np.ndarray,
        fps: float,
        visibility_min: float = 0.3,
    ) -> PoseDetectionResult:
        """Run RTMPose detection on video frames.

        Args:
            frames: (N, H, W, 3) RGB frames.
            fps: Frame rate for timestamps.
            visibility_min: Minimum confidence (used in post-processing).

        Returns:
            PoseDetectionResult with COCO-17 keypoints (2D only).
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected (N, H, W, 3) frames, got {frames.shape}")

        self._ensure_model_loaded()

        n_frames, height, width, _ = frames.shape

        # Output arrays
        keypoints_2d = np.zeros((n_frames, 17, 2), dtype=np.float32)
        visibility = np.zeros((n_frames, 17), dtype=np.float32)
        timestamps = np.zeros(n_frames, dtype=np.float32)

        for idx, frame in enumerate(frames):
            timestamps[idx] = idx / fps

            # rtmlib expects BGR, convert from RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Run detection - returns (keypoints, scores)
            # keypoints shape: (N_persons, N_keypoints, 2)
            # scores shape: (N_persons, N_keypoints)
            keypoints, scores = self._model(frame_bgr)

            if len(keypoints) == 0:
                # No detection - leave as zeros
                continue

            # Take first detected person
            # RTMPose Wholebody outputs 133 keypoints, first 17 are COCO body
            kp = keypoints[0][:17]  # (17, 2) pixel coordinates
            sc = scores[0][:17]  # (17,) confidence scores

            # Normalize to [0, 1] range
            keypoints_2d[idx, :, 0] = kp[:, 0] / width
            keypoints_2d[idx, :, 1] = kp[:, 1] / height
            visibility[idx] = sc

        return PoseDetectionResult(
            keypoints_2d=keypoints_2d,
            keypoints_3d=None,  # RTMPose doesn't provide 3D
            visibility=visibility,
            timestamps=timestamps,
            image_size=(height, width),
            metadata={
                "estimator": "rtmpose",
                "model_size": self.model_size,
                "backend": self.backend,
            },
        )

    def detect_with_preview(
        self,
        frames: np.ndarray,
        fps: float,
        visibility_min: float = 0.3,
        preview_output: Optional[Path] = None,
        display: bool = False,
    ) -> PoseDetectionResult:
        """Run detection with optional visualization.

        Args:
            frames: (N, H, W, 3) RGB frames.
            fps: Frame rate.
            visibility_min: Minimum confidence threshold.
            preview_output: Path to write preview video.
            display: Whether to show live preview window.

        Returns:
            PoseDetectionResult with COCO-17 keypoints.
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected (N, H, W, 3) frames, got {frames.shape}")

        self._ensure_model_loaded()

        n_frames, height, width, _ = frames.shape

        # Output arrays
        keypoints_2d = np.zeros((n_frames, 17, 2), dtype=np.float32)
        visibility = np.zeros((n_frames, 17), dtype=np.float32)
        timestamps = np.zeros(n_frames, dtype=np.float32)

        # COCO-17 skeleton connections for visualization
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
        ]

        # Preview writer
        preview_writer = None
        if preview_output:
            preview_output = Path(preview_output)
            preview_output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            preview_writer = cv2.VideoWriter(
                str(preview_output), fourcc, fps, (width, height)
            )

        try:
            for idx, frame in enumerate(frames):
                timestamps[idx] = idx / fps

                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                keypoints, scores = self._model(frame_bgr)

                if len(keypoints) > 0:
                    kp = keypoints[0][:17]
                    sc = scores[0][:17]

                    keypoints_2d[idx, :, 0] = kp[:, 0] / width
                    keypoints_2d[idx, :, 1] = kp[:, 1] / height
                    visibility[idx] = sc

                # Visualization
                if display or preview_writer:
                    annotated = frame_bgr.copy()

                    if len(keypoints) > 0:
                        kp = keypoints[0][:17].astype(int)
                        sc = scores[0][:17]

                        # Draw skeleton
                        for i, j in skeleton:
                            if sc[i] > visibility_min and sc[j] > visibility_min:
                                cv2.line(
                                    annotated,
                                    tuple(kp[i]),
                                    tuple(kp[j]),
                                    (0, 255, 0),
                                    2,
                                )

                        # Draw keypoints
                        for i, (x, y) in enumerate(kp):
                            if sc[i] > visibility_min:
                                cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)

                    if display:
                        try:
                            cv2.imshow("RTMPose", annotated)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                        except cv2.error:
                            display = False

                    if preview_writer:
                        preview_writer.write(annotated)

        finally:
            if display:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            if preview_writer:
                preview_writer.release()

        return PoseDetectionResult(
            keypoints_2d=keypoints_2d,
            keypoints_3d=None,
            visibility=visibility,
            timestamps=timestamps,
            image_size=(height, width),
            metadata={
                "estimator": "rtmpose",
                "model_size": self.model_size,
                "backend": self.backend,
            },
        )
