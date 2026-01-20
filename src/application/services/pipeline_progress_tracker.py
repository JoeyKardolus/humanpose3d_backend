"""Service that derives progress updates from pipeline output."""

from __future__ import annotations

import re
import time

from src.application.repositories.run_status_repository import (
    RunStatusRepository,
)


class PipelineProgressTracker:
    """Parse pipeline output lines to update progress."""

    def __init__(self, status_repo: RunStatusRepository) -> None:
        self._status_repo = status_repo
        self._progress_augment_start = 35.0
        self._progress_augment_end = 70.0

    def update_from_line(self, run_key: str, line: str) -> None:
        """Update progress status based on a single output line."""
        progress = None
        stage = None
        # Map known log markers to progress and stage labels.
        if "Created TensorFlow Lite delegate for GPU" in line or "[GPU]" in line:
            progress = 3.0
            stage = "Initializing ML backends"
        elif "gl_context" in line or "cuda" in line or "TensorFlow" in line:
            progress = 4.0
            stage = "Loading models"
        elif "[main] estimated" in line:
            progress = 10.0
            stage = "Estimating missing markers"
        elif "[anatomical_constraints]" in line:
            progress = 16.0
            stage = "Applying anatomical constraints"
        elif "[bone_length]" in line:
            progress = 20.0
            stage = "Applying bone length constraints"
        elif "[main] step1 CSV" in line:
            progress = 15.0
            stage = "Landmarks extracted"
        elif "[main] step2 TRC" in line:
            progress = 30.0
            stage = "TRC generated"
        elif "[pose] preview rotation detected" in line:
            progress = 28.0
            stage = "Correcting preview rotation"
        elif "[media] tesseract not found" in line:
            progress = 28.0
            stage = "Skipping OCR rotation (tesseract missing)"
        elif "[main] step3 augment" in line:
            progress = self._progress_augment_start
            stage = "Augmenting markers"
        elif "[augment] cycle" in line:
            match = re.search(r"cycle (\d+)/(\d+)", line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                if total > 0:
                    span = self._progress_augment_end - self._progress_augment_start
                    progress = self._progress_augment_start + span * (current / total)
                    stage = f"Augmenting markers ({current}/{total})"
        elif "[main] step3.5 force-complete" in line:
            progress = 75.0
            stage = "Completing marker set"
        elif "[main] step3.6 multi-constraint-optimization" in line:
            progress = 85.0
            stage = "Optimizing constraints"
        elif "Multi-Constraint Optimization" in line:
            progress = 82.0
            stage = "Optimizing constraints"
        elif "Computing Comprehensive Joint Angles" in line:
            progress = 90.0
            stage = "Computing joint angles"
        elif "[main] step5 joint angles" in line:
            progress = 92.0
            stage = "Computing joint angles"
        elif "[main] step6 upper body angles" in line:
            progress = 94.0
            stage = "Computing upper body angles"
        elif "[save_angles]" in line:
            progress = 96.0
            stage = "Saving joint angle outputs"
        elif "[main] finished pipeline" in line:
            progress = 100.0
            stage = "Finalizing output"

        if progress is None and stage is None:
            return

        status = self._status_repo.get_status(run_key)
        if status is None:
            return
        current_progress = float(status.get("progress", 0.0))
        if progress is not None and progress <= current_progress:
            progress = None

        updates: dict[str, object] = {}
        if progress is not None:
            updates["progress"] = min(progress, 100.0)
        if stage is not None:
            updates["stage"] = stage
        if updates:
            updates["last_update"] = time.monotonic()
            self._status_repo.set_status(run_key, updates)
