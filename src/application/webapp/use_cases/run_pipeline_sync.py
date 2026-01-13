from __future__ import annotations

"""Use case for synchronous pipeline execution."""

from dataclasses import dataclass
from typing import Mapping

from src.application.webapp.dto.pipeline_run_spec import PipelineRunSpec
from src.application.webapp.services.pipeline_command_builder import PipelineCommandBuilder
from src.application.webapp.services.pipeline_log_service import PipelineLogService
from src.application.webapp.services.pipeline_result_service import PipelineResultService
from src.application.webapp.services.pipeline_runner import PipelineRunner


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
        log_service: PipelineLogService,
        result_service: PipelineResultService,
    ) -> None:
        self._command_builder = command_builder
        self._pipeline_runner = pipeline_runner
        self._log_service = log_service
        self._result_service = result_service

    def execute(self, spec: PipelineRunSpec, form_data: Mapping[str, str]) -> PipelineSyncResult:
        """Execute the pipeline and record logs/results."""
        command = self._command_builder.build(spec.upload_path, form_data, spec.sex_raw)
        execution = self._pipeline_runner.run(command)
        if execution.return_code != 0:
            log_path = spec.pipeline_run_dir / "pipeline_error.log"
            self._log_service.write_log(log_path, execution.stdout_text, execution.stderr_text)
            return PipelineSyncResult(
                return_code=execution.return_code,
                stdout_text=execution.stdout_text,
                stderr_text=execution.stderr_text,
            )

        log_path = spec.pipeline_run_dir / "pipeline.log"
        self._log_service.write_log(log_path, execution.stdout_text, execution.stderr_text)
        self._result_service.move_output(spec.pipeline_run_dir, spec.output_dir)
        return PipelineSyncResult(
            return_code=execution.return_code,
            stdout_text=execution.stdout_text,
            stderr_text=execution.stderr_text,
        )

    def apply_header_fix(self, spec: PipelineRunSpec) -> None:
        """Apply the TRC header fix if requested."""
        self._result_service.apply_header_fix(spec.output_dir, spec.safe_run_id)
