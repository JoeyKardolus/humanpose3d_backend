"""Service for persisting uploaded videos."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from django.core.files.uploadedfile import UploadedFile

# Maximum video duration in seconds (1 minute)
MAX_VIDEO_DURATION_SECONDS = 60


class UploadService:
    """Persist incoming uploads to the configured directory."""

    def __init__(self, upload_root: Path) -> None:
        self._upload_root = upload_root

    def get_video_duration(self, video_path: Path) -> float | None:
        """Get video duration in seconds using ffprobe.

        Returns None if duration cannot be determined.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        return None

    def check_video_duration(self, video_path: Path) -> str | None:
        """Check if video duration is within limits.

        Returns error message if video is too long, None if OK.
        """
        duration = self.get_video_duration(video_path)
        if duration is None:
            # Can't determine duration - allow it but warn
            return None
        if duration > MAX_VIDEO_DURATION_SECONDS:
            return (
                f"Video is too long ({duration:.1f}s). "
                f"Maximum allowed duration is {MAX_VIDEO_DURATION_SECONDS} seconds (1 minute)."
            )
        return None

    def save_upload(self, uploaded: UploadedFile, safe_run_id: str) -> Path:
        """Write the uploaded file to disk and return its path."""
        upload_dir = self._upload_root / safe_run_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / f"{safe_run_id}{Path(uploaded.name).suffix}"
        with upload_path.open("wb") as handle:
            for chunk in uploaded.chunks():
                handle.write(chunk)
        return upload_path

    def remove_upload(self, safe_run_id: str) -> None:
        """Remove persisted uploads for a run if they exist."""
        upload_dir = self._upload_root / safe_run_id
        if not upload_dir.exists():
            return
        try:
            shutil.rmtree(upload_dir)
        except OSError:
            return
