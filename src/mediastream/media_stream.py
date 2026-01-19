from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Tuple

import cv2 as cv
import numpy as np


def read_video_rgb(video_path: Path) -> Tuple[np.ndarray, float]:
    """Load RGB frames deterministically so strict downstream steps retain true timing."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS) or 0.0
    frames = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    stack = np.stack(frames)
    return stack, float(fps or 0.0)


def probe_video_rotation(video_path: Path) -> int:
    """Return rotation in degrees (0/90/180/270) if metadata is available."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream_tags=rotate",
        "-of",
        "default=nk=1:nw=1",
        str(video_path),
    ]
    try:
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return 0
    try:
        rotation = int(output.decode("utf-8").strip())
    except ValueError:
        return 0
    rotation = rotation % 360
    if rotation in {0, 90, 180, 270}:
        return rotation
    return 0




class MediaStream:
    """Keeps backward compatibility with earlier imperative usage."""

    def __init__(self):
        self.video_rgb: np.ndarray | None = None
        self.fps: float = 0.0

    def read_video(self, video_path: Path) -> np.ndarray:
        """Loads video and returns it in RGB format."""
        self.video_rgb, self.fps = read_video_rgb(video_path)
        return self.video_rgb
