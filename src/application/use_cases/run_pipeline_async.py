"""Use case for asynchronous pipeline execution."""

from __future__ import annotations

import logging
import threading
import time
from typing import Mapping

from src.application.dto.pipeline_run_spec import PipelineRunSpec
from src.application.repositories.run_status_repository import (
    RunStatusRepository,
)
from src.application.services.analytics_service import AnalyticsService
from src.application.services.kinematics_quality_service import KinematicsQualityService
from src.application.services.pipeline_command_builder import (
    PipelineCommandBuilder,
)
from src.application.services.pipeline_progress_tracker import (
    PipelineProgressTracker,
)
from src.application.services.pipeline_result_service import (
    PipelineResultService,
)
from src.application.services.pipeline_runner import PipelineRunner
from src.application.services.upload_service import UploadService

logger = logging.getLogger(__name__)


class RunPipelineAsyncUseCase:
    """Queue a pipeline run and report progress via shared status."""

    def __init__(
        self,
        command_builder: PipelineCommandBuilder,
        pipeline_runner: PipelineRunner,
        result_service: PipelineResultService,
        upload_service: UploadService,
        status_repo: RunStatusRepository,
        progress_tracker: PipelineProgressTracker,
        analytics_service: AnalyticsService | None = None,
        quality_service: KinematicsQualityService | None = None,
    ) -> None:
        self._command_builder = command_builder
        self._pipeline_runner = pipeline_runner
        self._result_service = result_service
        self._upload_service = upload_service
        self._status_repo = status_repo
        self._progress_tracker = progress_tracker
        self._analytics_service = analytics_service
        self._quality_service = quality_service

    def enqueue(
        self,
        spec: PipelineRunSpec,
        form_data: Mapping[str, str],
        fix_header: bool,
        results_url: str,
    ) -> None:
        """Spawn a background worker for the pipeline run."""
        self._status_repo.set_status(
            spec.run_key,
            {
                "progress": 1.0,
                "stage": "Queued",
                "done": False,
                "error": None,
                "results_url": results_url,
            },
        )
        # Track run started for analytics
        if self._analytics_service:
            try:
                self._analytics_service.track_run_started(spec.run_key, form_data)
            except Exception as e:
                logger.warning(f"Failed to track run started: {e}")

        worker = threading.Thread(
            target=self._run_worker,
            args=(spec, form_data, fix_header),
            daemon=True,
        )
        worker.start()

    def _run_worker(
        self,
        spec: PipelineRunSpec,
        form_data: Mapping[str, str],
        fix_header: bool,
    ) -> None:
        """Execute the pipeline and update status for progress polling."""
        self._status_repo.set_status(
            spec.run_key,
            {
                "progress": 2.0,
                "stage": "Starting pipeline",
                "done": False,
                "error": None,
                "started_at": time.monotonic(),
                "last_update": time.monotonic(),
            },
        )

        command = self._command_builder.build(spec.upload_path, form_data)

        try:
            execution = self._pipeline_runner.run(
                command,
                on_line=lambda line: self._progress_tracker.update_from_line(
                    spec.run_key, line
                ),
            )
        except (OSError, RuntimeError) as exc:
            self._upload_service.remove_upload(spec.safe_run_id)
            self._status_repo.set_status(
                spec.run_key,
                {
                    "done": True,
                    "error": f"Failed to start pipeline: {exc}",
                    "stage": "Failed",
                },
            )
            # Track error for analytics
            if self._analytics_service:
                try:
                    self._analytics_service.track_run_error(
                        spec.run_key, "startup_error", str(exc)
                    )
                except Exception as e:
                    logger.warning(f"Failed to track run error: {e}")
            return

        if execution.return_code != 0:
            log_path = spec.pipeline_run_dir / "pipeline_error.log"
            log_path.write_text(
                "\n".join([
                    "[stdout]",
                    execution.stdout_text.strip(),
                    "",
                    "[stderr]",
                    execution.stderr_text.strip(),
                ]) + "\n",
                encoding="utf-8",
            )
            self._upload_service.remove_upload(spec.safe_run_id)
            self._status_repo.set_status(
                spec.run_key,
                {
                    "done": True,
                    "error": "Pipeline failed. Check logs for details.",
                    "stage": "Failed",
                    "last_update": time.monotonic(),
                },
            )
            # Track error for analytics
            if self._analytics_service:
                try:
                    # Extract error type from stderr
                    stderr_lower = execution.stderr_text.lower()
                    if "no landmarks" in stderr_lower or "no pose" in stderr_lower:
                        error_type = "no_landmarks_detected"
                    elif "out of memory" in stderr_lower:
                        error_type = "out_of_memory"
                    elif "cuda" in stderr_lower and "error" in stderr_lower:
                        error_type = "cuda_error"
                    elif "video duration" in stderr_lower:
                        error_type = "video_too_long"
                    else:
                        error_type = "pipeline_error"
                    self._analytics_service.track_run_error(
                        spec.run_key, error_type, execution.stderr_text[:500]
                    )
                except Exception as e:
                    logger.warning(f"Failed to track run error: {e}")
            return

        log_path = spec.pipeline_run_dir / "pipeline.log"
        log_path.write_text(
            "\n".join([
                "[stdout]",
                execution.stdout_text.strip(),
                "",
                "[stderr]",
                execution.stderr_text.strip(),
            ]) + "\n",
            encoding="utf-8",
        )
        self._result_service.move_output(spec.pipeline_run_dir, spec.output_dir)
        # Skip saving source video - only keep marker data
        # self._result_service.persist_input_video(
        #     spec.upload_path,
        #     spec.output_dir,
        #     spec.safe_run_id,
        # )
        self._upload_service.remove_upload(spec.safe_run_id)

        # Analyze kinematics quality and track completion
        quality_metrics = None
        if self._quality_service:
            try:
                joint_angles_dir = spec.output_dir / "joint_angles"
                quality_metrics = self._quality_service.analyze(joint_angles_dir)
                if quality_metrics:
                    self._quality_service.save_metrics(quality_metrics, spec.output_dir)
            except Exception as e:
                logger.warning(f"Failed to analyze kinematics quality: {e}")

        # Track successful completion for analytics
        if self._analytics_service:
            try:
                self._analytics_service.track_run_completed(
                    spec.run_key, spec.output_dir, quality_metrics
                )
            except Exception as e:
                logger.warning(f"Failed to track run completed: {e}")

        self._status_repo.set_status(
            spec.run_key,
            {
                "progress": 100.0,
                "stage": "Complete",
                "done": True,
                "last_update": time.monotonic(),
            },
        )
