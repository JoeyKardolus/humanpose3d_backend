"""Service for post-processing pipeline output."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class PipelineResultService:
    """Handle output movement and optional data fixes."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root

    def move_output(self, pipeline_run_dir: Path, output_dir: Path) -> None:
        """Move pipeline output into the requested directory."""
        if output_dir != pipeline_run_dir:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pipeline_run_dir), str(output_dir))


    def persist_input_video(
        self, upload_path: Path, output_dir: Path, safe_run_id: str
    ) -> None:
        """Copy the source video into the run output directory."""
        if not upload_path.exists():
            return
        target_dir = output_dir / "source"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{safe_run_id}{upload_path.suffix}"
        try:
            shutil.copy2(upload_path, target_path)
        except OSError:
            return
        rotation = self._probe_video_rotation(upload_path)
        metadata = {
            "rotation_degrees": rotation,
            "rotation_applied": False,
        }
        metadata_path = target_dir / "video_metadata.json"
        try:
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except OSError:
            return

    @staticmethod
    def _probe_video_rotation(video_path: Path) -> int:
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
