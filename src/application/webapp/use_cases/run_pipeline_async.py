"""Use case for asynchronous pipeline execution."""

from __future__ import annotations

import threading
import time
from typing import Mapping

from src.application.webapp.dto.pipeline_run_spec import PipelineRunSpec
from src.application.webapp.repositories.run_status_repository import (
    RunStatusRepository,
)
from src.application.webapp.services.pipeline_command_builder import (
    PipelineCommandBuilder,
)
from src.application.webapp.services.pipeline_log_service import PipelineLogService
from src.application.webapp.services.pipeline_progress_tracker import (
    PipelineProgressTracker,
)
from src.application.webapp.services.pipeline_result_service import (
    PipelineResultService,
)
from src.application.webapp.services.pipeline_runner import PipelineRunner
from src.application.webapp.services.upload_service import UploadService


class RunPipelineAsyncUseCase:
    """Queue a pipeline run and report progress via shared status."""

    def __init__(
        self,
        command_builder: PipelineCommandBuilder,
        pipeline_runner: PipelineRunner,
        log_service: PipelineLogService,
        result_service: PipelineResultService,
        upload_service: UploadService,
        status_repo: RunStatusRepository,
        progress_tracker: PipelineProgressTracker,
    ) -> None:
        self._command_builder = command_builder
        self._pipeline_runner = pipeline_runner
        self._log_service = log_service
        self._result_service = result_service
        self._upload_service = upload_service
        self._status_repo = status_repo
        self._progress_tracker = progress_tracker

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
                "stage": "Initializing models",
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
            return

        if execution.return_code != 0:
            log_path = spec.pipeline_run_dir / "pipeline_error.log"
            self._log_service.write_log(
                log_path, execution.stdout_text, execution.stderr_text
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
            return

        log_path = spec.pipeline_run_dir / "pipeline.log"
        self._log_service.write_log(
            log_path, execution.stdout_text, execution.stderr_text
        )
        self._result_service.move_output(spec.pipeline_run_dir, spec.output_dir)
        if fix_header:
            self._result_service.apply_header_fix(spec.output_dir, spec.safe_run_id)
        self._result_service.persist_input_video(
            spec.upload_path,
            spec.output_dir,
            spec.safe_run_id,
        )
        self._upload_service.remove_upload(spec.safe_run_id)

        self._status_repo.set_status(
            spec.run_key,
            {
                "progress": 100.0,
                "stage": "Complete",
                "done": True,
                "last_update": time.monotonic(),
            },
        )
