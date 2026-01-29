from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Dict, List, TYPE_CHECKING

import cv2
import numpy as np

from src.datastream.data_stream import LandmarkRecord

# Lazy import for mediapipe - only loaded when extract_world_landmarks is called
# This avoids loading TensorFlow (~3s) on module import
if TYPE_CHECKING:
    import mediapipe as mp

# MediaPipe pose landmark indices (stable, from PoseLandmark enum)
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_NAME_MAP: Dict[int, str] = {
    0: "Nose",           # NOSE
    12: "RShoulder",     # RIGHT_SHOULDER
    11: "LShoulder",     # LEFT_SHOULDER
    14: "RElbow",        # RIGHT_ELBOW
    13: "LElbow",        # LEFT_ELBOW
    16: "RWrist",        # RIGHT_WRIST
    15: "LWrist",        # LEFT_WRIST
    24: "RHip",          # RIGHT_HIP
    23: "LHip",          # LEFT_HIP
    26: "RKnee",         # RIGHT_KNEE
    25: "LKnee",         # LEFT_KNEE
    28: "RAnkle",        # RIGHT_ANKLE
    27: "LAnkle",        # LEFT_ANKLE
    30: "RHeel",         # RIGHT_HEEL
    29: "LHeel",         # LEFT_HEEL
    32: "RBigToe",       # RIGHT_FOOT_INDEX
    31: "LBigToe",       # LEFT_FOOT_INDEX
}


def extract_world_landmarks(
    video_rgb: np.ndarray,
    fps: float,
    model_path: Path,
    visibility_min: float,
    display: bool = False,
    return_raw_landmarks: bool = False,
    return_2d_landmarks: bool = False,
    preview_output: Path | None = None,
    preview_rotation_degrees: int = 0,
) -> List[LandmarkRecord] | tuple:
    """Run MediaPipe Pose world landmarks and optionally preview detections.

    Args:
        return_2d_landmarks: If True, also returns dict of 2D normalized image coordinates
                            for depth refinement. Format: {(timestamp, landmark): (x, y)}
    """
    # Lazy import mediapipe to avoid loading TensorFlow (~3s) unless actually needed
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe.tasks.python import vision

    if video_rgb.size == 0:
        raise ValueError("Video contains no frames")

    fps = fps or 30.0
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    draw_connections = None
    landmark_style = None
    if display or preview_output is not None:
        draw_connections = mp.solutions.pose.POSE_CONNECTIONS
        landmark_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

    raw_frames: List[List[landmark_pb2.NormalizedLandmark]] = []
    records: List[LandmarkRecord] = []
    landmarks_2d: Dict[tuple, tuple] = {}  # {(timestamp, landmark_name): (x, y)}
    preview_writer = None
    preview_failed = False
    wrote_preview = False

    def write_preview(frame_rgb: np.ndarray) -> None:
        nonlocal preview_writer, preview_failed, wrote_preview
        if preview_output is None:
            return
        if preview_writer is None:
            height, width = frame_rgb.shape[:2]
            preview_output.parent.mkdir(parents=True, exist_ok=True)
            for codec in ("avc1", "mp4v"):
                fourcc = cv2.VideoWriter_fourcc(*codec)
                preview_writer = cv2.VideoWriter(
                    str(preview_output), fourcc, fps or 30.0, (width, height)
                )
                if preview_writer.isOpened():
                    break
                preview_writer.release()
                preview_writer = None
            if preview_writer is None:
                preview_failed = True
                return
        preview_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        wrote_preview = True
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
                # Store 2D normalized image coordinates for depth refinement
                if return_2d_landmarks and vis:
                    lm_2d = vis[lm_idx]
                    landmarks_2d[(timestamp, mapped_name)] = (float(lm_2d.x), float(lm_2d.y))

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
        if preview_output is not None:
            if wrote_preview:
                _transcode_preview(preview_output, preview_rotation_degrees)
            else:
                try:
                    preview_output.unlink(missing_ok=True)
                except OSError:
                    pass
        detector.close()

    # Return based on flags
    if return_raw_landmarks and return_2d_landmarks:
        return records, raw_frames, landmarks_2d
    elif return_raw_landmarks:
        return records, raw_frames
    elif return_2d_landmarks:
        return records, landmarks_2d
    return records


def _transcode_preview(preview_output: Path, rotation_degrees: int) -> None:
    """Transcode preview video to a browser-friendly H.264 MP4 when possible."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not preview_output.exists():
        return
    temp_path = preview_output.with_suffix(".h264.mp4")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(preview_output),
    ]
    filter_chain = _rotation_filter(rotation_degrees)
    if filter_chain:
        command += ["-vf", filter_chain]
    command += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    try:
        temp_path.replace(preview_output)
    except OSError:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _rotation_filter(rotation_degrees: int) -> str | None:
    if rotation_degrees == 90:
        return "transpose=1"
    if rotation_degrees == 180:
        return "transpose=1,transpose=1"
    if rotation_degrees == 270:
        return "transpose=2"
    return None
