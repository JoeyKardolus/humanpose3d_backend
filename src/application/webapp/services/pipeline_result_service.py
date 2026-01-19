from __future__ import annotations

"""Service for post-processing pipeline output."""

import shutil
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

    def apply_header_fix(self, output_dir: Path, safe_run_id: str) -> None:
        """Apply strict TRC header fixes when requested."""
        from src.datastream.data_stream import header_fix_strict

        final_trc = output_dir / f"{safe_run_id}_final.trc"
        if final_trc.exists():
            header_fix_strict(final_trc)

    def persist_input_video(self, upload_path: Path, output_dir: Path, safe_run_id: str) -> None:
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
