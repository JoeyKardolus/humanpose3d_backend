"""Use case for synchronous pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.application.dto.pipeline_run_spec import PipelineRunSpec
from src.application.services.pipeline_command_builder import (
    PipelineCommandBuilder,
)
from src.application.services.pipeline_result_service import (
    PipelineResultService,
)
from src.application.services.pipeline_runner import PipelineRunner
from src.application.services.upload_service import UploadService


@dataclass(frozen=True)
class PipelineSyncResult:
    """Captured output for synchronous pipeline runs."""

    return_code: int
    stdout_text: str
    stderr_text: str


class RunPipelineSyncUseCase:
    """Runs the pipeline in-process and handles output persistence."""

    def __init__(
        self,
        command_builder: PipelineCommandBuilder,
        pipeline_runner: PipelineRunner,
        result_service: PipelineResultService,
        upload_service: UploadService,
    ) -> None:
        self._command_builder = command_builder
        self._pipeline_runner = pipeline_runner
        self._result_service = result_service
        self._upload_service = upload_service

    def execute(
        self, spec: PipelineRunSpec, form_data: Mapping[str, str]
    ) -> PipelineSyncResult:
        """Execute the pipeline and record logs/results."""
        command = self._command_builder.build(spec.upload_path, form_data)
        try:
            execution = self._pipeline_runner.run(command)
        except (OSError, RuntimeError):
            self._upload_service.remove_upload(spec.safe_run_id)
            raise
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
            return PipelineSyncResult(
                return_code=execution.return_code,
                stdout_text=execution.stdout_text,
                stderr_text=execution.stderr_text,
            )

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
        return PipelineSyncResult(
            return_code=execution.return_code,
            stdout_text=execution.stdout_text,
            stderr_text=execution.stderr_text,
        )

