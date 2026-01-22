"""MediaPipe pose estimator implementation.

Wraps MediaPipe Pose Landmarker to implement the PoseEstimator interface.
Provides both 2D and 3D world coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision

from .base import PoseEstimator, PoseDetectionResult, COCO_KEYPOINT_NAMES


# MediaPipe BlazePose (33 landmarks) to COCO-17 mapping
# MediaPipe indices: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
MEDIAPIPE_TO_COCO: Dict[int, int] = {
    0: 0,    # nose
    2: 1,    # left_eye (MediaPipe: left_eye_inner, closest match)
    5: 2,    # right_eye (MediaPipe: right_eye_inner, closest match)
    7: 3,    # left_ear
    8: 4,    # right_ear
    11: 5,   # left_shoulder
    12: 6,   # right_shoulder
    13: 7,   # left_elbow
    14: 8,   # right_elbow
    15: 9,   # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16,  # right_ankle
}

# Reverse mapping: COCO index -> MediaPipe index
COCO_TO_MEDIAPIPE: Dict[int, int] = {v: k for k, v in MEDIAPIPE_TO_COCO.items()}

# MediaPipe index to marker name (for LandmarkRecord compatibility)
MEDIAPIPE_TO_MARKER_NAME: Dict[int, str] = {
    mp.solutions.pose.PoseLandmark.NOSE.value: "Nose",
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value: "RShoulder",
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value: "LShoulder",
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value: "RElbow",
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value: "LElbow",
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value: "RWrist",
    mp.solutions.pose.PoseLandmark.LEFT_WRIST.value: "LWrist",
    mp.solutions.pose.PoseLandmark.RIGHT_HIP.value: "RHip",
    mp.solutions.pose.PoseLandmark.LEFT_HIP.value: "LHip",
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value: "RKnee",
    mp.solutions.pose.PoseLandmark.LEFT_KNEE.value: "LKnee",
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value: "RAnkle",
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value: "LAnkle",
    mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value: "RHeel",
    mp.solutions.pose.PoseLandmark.LEFT_HEEL.value: "LHeel",
    mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value: "RBigToe",
    mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value: "LBigToe",
}


class MediaPipeDetector(PoseEstimator):
    """MediaPipe Pose Landmarker implementation.

    Uses MediaPipe's BlazePose model to detect 33 body landmarks,
    then maps to COCO-17 format. Provides both 2D normalized image
    coordinates and 3D world coordinates.

    Attributes:
        model_path: Path to the MediaPipe .task model file.
    """

    def __init__(self, model_path: Path | str):
        """Initialize MediaPipe detector.

        Args:
            model_path: Path to pose_landmarker_*.task model file.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {self.model_path}")

    @property
    def name(self) -> str:
        return "mediapipe"

    @property
    def provides_3d(self) -> bool:
        return True  # MediaPipe outputs 3D world coordinates

    def detect(
        self,
        frames: np.ndarray,
        fps: float,
        visibility_min: float = 0.3,
    ) -> PoseDetectionResult:
        """Run MediaPipe pose detection.

        Args:
            frames: (N, H, W, 3) RGB frames.
            fps: Frame rate for timestamps.
            visibility_min: Minimum confidence (used in post-processing).

        Returns:
            PoseDetectionResult with COCO-17 keypoints.
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected (N, H, W, 3) frames, got {frames.shape}")

        n_frames, height, width, _ = frames.shape

        # Initialize detector
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        # Output arrays
        keypoints_2d = np.zeros((n_frames, 17, 2), dtype=np.float32)
        keypoints_3d = np.zeros((n_frames, 17, 3), dtype=np.float32)
        visibility = np.zeros((n_frames, 17), dtype=np.float32)
        timestamps = np.zeros(n_frames, dtype=np.float32)

        try:
            for idx, frame in enumerate(frames):
                timestamp = idx / fps
                timestamps[idx] = timestamp

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = detector.detect_for_video(mp_image, int(timestamp * 1000))

                world_landmarks = result.pose_world_landmarks or []
                image_landmarks = result.pose_landmarks or []

                if not world_landmarks or not image_landmarks:
                    # No detection - leave as zeros (will have zero visibility)
                    continue

                # Extract first person
                world = world_landmarks[0]
                image = image_landmarks[0]

                # Map MediaPipe -> COCO-17
                for mp_idx, coco_idx in MEDIAPIPE_TO_COCO.items():
                    if mp_idx < len(world) and mp_idx < len(image):
                        # 3D world coordinates (in meters)
                        keypoints_3d[idx, coco_idx] = [
                            world[mp_idx].x,
                            world[mp_idx].y,
                            world[mp_idx].z,
                        ]
                        # 2D normalized image coordinates
                        keypoints_2d[idx, coco_idx] = [
                            image[mp_idx].x,
                            image[mp_idx].y,
                        ]
                        # Visibility/confidence
                        visibility[idx, coco_idx] = image[mp_idx].visibility

        finally:
            detector.close()

        return PoseDetectionResult(
            keypoints_2d=keypoints_2d,
            keypoints_3d=keypoints_3d,
            visibility=visibility,
            timestamps=timestamps,
            image_size=(height, width),
            metadata={"estimator": "mediapipe", "model": str(self.model_path)},
        )

    def detect_with_preview(
        self,
        frames: np.ndarray,
        fps: float,
        visibility_min: float = 0.3,
        preview_output: Optional[Path] = None,
        display: bool = False,
    ) -> PoseDetectionResult:
        """Run detection with optional visualization/preview output.

        This is an extended version of detect() that supports the
        visualization features from the original extract_world_landmarks().

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

        n_frames, height, width, _ = frames.shape

        # Initialize detector
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        # Drawing setup
        draw_connections = None
        landmark_style = None
        if display or preview_output:
            draw_connections = mp.solutions.pose.POSE_CONNECTIONS
            landmark_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

        # Output arrays
        keypoints_2d = np.zeros((n_frames, 17, 2), dtype=np.float32)
        keypoints_3d = np.zeros((n_frames, 17, 3), dtype=np.float32)
        visibility = np.zeros((n_frames, 17), dtype=np.float32)
        timestamps = np.zeros(n_frames, dtype=np.float32)

        # Preview writer
        preview_writer = None
        if preview_output:
            preview_output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            preview_writer = cv2.VideoWriter(
                str(preview_output), fourcc, fps, (width, height)
            )

        try:
            for idx, frame in enumerate(frames):
                timestamp = idx / fps
                timestamps[idx] = timestamp

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = detector.detect_for_video(mp_image, int(timestamp * 1000))

                world_landmarks = result.pose_world_landmarks or []
                image_landmarks = result.pose_landmarks or []

                if world_landmarks and image_landmarks:
                    world = world_landmarks[0]
                    image = image_landmarks[0]

                    for mp_idx, coco_idx in MEDIAPIPE_TO_COCO.items():
                        if mp_idx < len(world) and mp_idx < len(image):
                            keypoints_3d[idx, coco_idx] = [
                                world[mp_idx].x,
                                world[mp_idx].y,
                                world[mp_idx].z,
                            ]
                            keypoints_2d[idx, coco_idx] = [
                                image[mp_idx].x,
                                image[mp_idx].y,
                            ]
                            visibility[idx, coco_idx] = image[mp_idx].visibility

                # Visualization
                if display or preview_writer:
                    annotated = frame.copy()
                    for pose_landmarks in result.pose_landmarks or []:
                        pose_proto = landmark_pb2.NormalizedLandmarkList()
                        pose_proto.landmark.extend(
                            landmark_pb2.NormalizedLandmark(
                                x=lm.x, y=lm.y, z=lm.z
                            )
                            for lm in pose_landmarks
                        )
                        mp.solutions.drawing_utils.draw_landmarks(
                            annotated,
                            pose_proto,
                            draw_connections,
                            landmark_style,
                        )

                    if display:
                        try:
                            cv2.imshow("MediaPipe Pose", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                        except cv2.error:
                            display = False

                    if preview_writer:
                        preview_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        finally:
            detector.close()
            if display:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            if preview_writer:
                preview_writer.release()

        return PoseDetectionResult(
            keypoints_2d=keypoints_2d,
            keypoints_3d=keypoints_3d,
            visibility=visibility,
            timestamps=timestamps,
            image_size=(height, width),
            metadata={"estimator": "mediapipe", "model": str(self.model_path)},
        )
