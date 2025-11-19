from pathlib import Path
import numpy as np
import cv2 as cv


class MediaStream:
    """Handles media streaming tasks"""

    def __init__(self):
        self.video_rgb = None
        self.fps: int = 0

    def read_video(self, video_path: Path) -> np.ndarray:
        """Loads video and returns it in RGB format."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv.VideoCapture(str(video_path))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))

        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

        video_frames_rgb = np.empty((frame_count, height, width, 3), dtype=np.uint8)

        idx = 0
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            video_frames_rgb[idx] = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            idx += 1

        cap.release()

        self.video_rgb = video_frames_rgb[:idx]
        return self.video_rgb
