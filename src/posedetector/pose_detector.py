from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


class PoseDetector:
    def __init__(self, model_path: Path) -> None:
        self.model_path: Path = model_path
        self.landmarks: list = []

    def detect_pose_landmarks(self, video: np.ndarray, show_video: bool) -> list:
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        fps = int(1000 / 30)  # Default 30 FPS, pass actual FPS as parameter if needed

        self.landmarks = []

        if show_video:
            pose_connections = mp.solutions.pose.POSE_CONNECTIONS
            landmark_style = (
                mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

        for idx, frame in enumerate(video):
            timestamp_ms = idx * fps
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            self.landmarks.append(detection_result.pose_landmarks)

            if show_video and detection_result.pose_landmarks:
                annotated_frame = frame.copy()

                for pose_landmarks in detection_result.pose_landmarks:
                    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    pose_landmarks_proto.landmark.extend(
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in pose_landmarks
                    )
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated_frame,
                        pose_landmarks_proto,
                        pose_connections,
                        landmark_style,
                    )

                cv2.imshow(
                    "Pose Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if show_video:
            cv2.destroyAllWindows()

        detector.close()
        return self.landmarks
