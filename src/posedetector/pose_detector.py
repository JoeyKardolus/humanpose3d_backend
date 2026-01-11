from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision

from src.datastream.data_stream import LandmarkRecord

POSE_NAME_MAP: Dict[int, str] = {
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


def extract_world_landmarks(
    video_rgb: np.ndarray,
    fps: float,
    model_path: Path,
    visibility_min: float,
    display: bool = False,
    return_raw_landmarks: bool = False,
    preview_output: Path | None = None,
) -> List[LandmarkRecord] | tuple[List[LandmarkRecord], List[List[landmark_pb2.NormalizedLandmark]]]:
    """Run MediaPipe Pose world landmarks and optionally preview detections."""
    if video_rgb.size == 0:
        raise ValueError("Video contains no frames")

    fps = fps or 30.0
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(model_path),
        delegate=mp.tasks.BaseOptions.Delegate.GPU,
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    draw_connections = None
    landmark_style = None
    if display:
        draw_connections = mp.solutions.pose.POSE_CONNECTIONS
        landmark_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

    raw_frames: List[List[landmark_pb2.NormalizedLandmark]] = []
    records: List[LandmarkRecord] = []
    preview_writer = None
    preview_failed = False

    def write_preview(frame_rgb: np.ndarray) -> None:
        nonlocal preview_writer
        if preview_output is None:
            return
        if preview_writer is None:
            height, width = frame_rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            preview_output.parent.mkdir(parents=True, exist_ok=True)
            preview_writer = cv2.VideoWriter(
                str(preview_output), fourcc, fps or 30.0, (width, height)
            )
        preview_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    try:
        for idx, frame in enumerate(video_rgb):
            timestamp = idx / fps
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = detector.detect_for_video(mp_image, int(timestamp * 1000))
            world_landmarks = result.pose_world_landmarks or []
            vis_landmarks = result.pose_landmarks or []

            if not world_landmarks:
                if display:
                    try:
                        cv2.imshow("Pose Detection", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        display = False
                        if preview_output is not None and not preview_failed:
                            print(
                                f"[pose] GUI unavailable; writing preview video to {preview_output}"
                            )
                        preview_failed = True
                if preview_output is not None:
                    write_preview(frame)
                if return_raw_landmarks:
                    raw_frames.append([])
                continue

            world = world_landmarks[0]
            vis = vis_landmarks[0] if vis_landmarks else None
            if return_raw_landmarks:
                raw_frames.append(list(world))

            for lm_idx, landmark in enumerate(world):
                mapped_name = POSE_NAME_MAP.get(lm_idx)
                if mapped_name is None:
                    continue
                visibility = vis[lm_idx].visibility if vis else 1.0
                if visibility < visibility_min:
                    continue
                records.append(
                    LandmarkRecord(
                        timestamp_s=timestamp,
                        landmark=mapped_name,
                        x_m=float(landmark.x),
                        y_m=float(landmark.y),
                        z_m=float(landmark.z),
                        visibility=float(visibility),
                    )
                )

            need_annotation = display or preview_output is not None
            annotated = frame.copy() if need_annotation else None
            if need_annotation:
                for pose_landmarks in result.pose_landmarks or []:
                    pose_proto = landmark_pb2.NormalizedLandmarkList()
                    pose_proto.landmark.extend(
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in pose_landmarks
                    )
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated,
                        pose_proto,
                        draw_connections,
                        landmark_style,
                    )

            if display and annotated is not None:
                try:
                    cv2.imshow("Pose Detection", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    display = False
                    if preview_output is not None and not preview_failed:
                        print(
                            f"[pose] GUI unavailable; writing preview video to {preview_output}"
                        )
                    preview_failed = True

            if preview_output is not None and annotated is not None:
                write_preview(annotated)
    finally:
        if display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        if preview_writer is not None:
            preview_writer.release()
        detector.close()

    if return_raw_landmarks:
        return records, raw_frames
    return records
