"""Use case for preparing a pipeline run from request input."""

from __future__ import annotations

from src.application.webapp.dto.pipeline_preparation_result import (
    PipelinePreparationResult,
)
from src.application.webapp.dto.pipeline_request import PipelineRequestData
from src.application.webapp.dto.pipeline_run_spec import PipelineRunSpec
from src.application.webapp.services.output_directory_service import (
    OutputDirectoryService,
)
from src.application.webapp.services.run_id_factory import RunIdFactory
from src.application.webapp.services.run_key_service import RunKeyService
from src.application.webapp.services.upload_service import UploadService
from src.application.webapp.validators.run_request_validator import RunRequestValidator


class PreparePipelineRunUseCase:
    """Validate inputs, persist uploads, and prepare output directories."""

    def __init__(
        self,
        validator: RunRequestValidator,
        run_id_factory: RunIdFactory,
        run_key_service: RunKeyService,
        upload_service: UploadService,
        output_directory_service: OutputDirectoryService,
    ) -> None:
        self._validator = validator
        self._run_id_factory = run_id_factory
        self._run_key_service = run_key_service
        self._upload_service = upload_service
        self._output_directory_service = output_directory_service

    def execute(self, request_data: PipelineRequestData) -> PipelinePreparationResult:
        """Prepare a pipeline run or return validation errors."""
        uploaded = request_data.files.get("video")
        errors, sex_raw, output_location = self._validator.validate(
            request_data.form_data,
            has_upload=uploaded is not None,
        )
        if errors:
            return PipelinePreparationResult(errors=errors, prepared=None)
        if uploaded is None:
            return PipelinePreparationResult(
                errors=["Unable to prepare pipeline run."],
                prepared=None,
            )

        safe_run_id = self._run_id_factory.create(uploaded.name)
        run_key = self._run_key_service.build_run_key(output_location, safe_run_id)
        upload_path = self._upload_service.save_upload(uploaded, safe_run_id)

        output_dirs, output_errors = self._output_directory_service.prepare_directories(
            safe_run_id,
            run_key,
        )
        if output_errors:
            return PipelinePreparationResult(errors=output_errors, prepared=None)

        spec = PipelineRunSpec(
            sex_raw=sex_raw,
            run_key=run_key,
            safe_run_id=safe_run_id,
            upload_path=upload_path,
            pipeline_run_dir=output_dirs.pipeline_run_dir,
            output_dir=output_dirs.output_dir,
        )
        return PipelinePreparationResult(errors=[], prepared=spec)
