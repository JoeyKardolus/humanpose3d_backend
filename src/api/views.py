"""HTTP API endpoints for the HumanPose3D pipeline."""

from __future__ import annotations

from pathlib import Path

from django.http import (
    FileResponse,
    Http404,
    HttpRequest,
    HttpResponse,
    JsonResponse,
)
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from src.application.webapp.config.paths import AppPaths
from src.application.webapp.dto.pipeline_request import PipelineRequestData
from src.application.webapp.repositories.run_status_repository import (
    RunStatusRepository,
)
from src.application.webapp.services.media_service import MediaService
from src.application.webapp.services.output_directory_service import (
    OutputDirectoryService,
)
from src.application.webapp.services.output_history_service import OutputHistoryService
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
from src.application.webapp.services.progress_service import ProgressService
from src.application.webapp.services.results_archive_service import (
    ResultsArchiveService,
)
from src.application.webapp.services.results_service import ResultsService
from src.application.webapp.services.run_cleanup_service import RunCleanupService
from src.application.webapp.services.run_id_factory import RunIdFactory
from src.application.webapp.services.run_key_service import RunKeyService
from src.application.webapp.services.statistics_service import StatisticsService
from src.application.webapp.services.upload_service import UploadService
from src.application.webapp.use_cases.prepare_pipeline_run import (
    PreparePipelineRunUseCase,
)
from src.application.webapp.use_cases.run_pipeline_async import RunPipelineAsyncUseCase
from src.application.webapp.use_cases.run_pipeline_sync import RunPipelineSyncUseCase
from src.application.webapp.validators.path_validator import PathValidator
from src.application.webapp.validators.run_request_validator import RunRequestValidator


_APP_PATHS = AppPaths.from_anchor(Path(__file__))
_PATH_VALIDATOR = PathValidator()
_STATUS_REPO = RunStatusRepository()

_PIPELINE_RUNNER = PipelineRunner(_APP_PATHS.repo_root)
_PIPELINE_LOGGER = PipelineLogService()
_PIPELINE_RESULTS = PipelineResultService(_APP_PATHS.repo_root)
_PIPELINE_COMMANDS = PipelineCommandBuilder(_APP_PATHS.repo_root)
_PROGRESS_TRACKER = PipelineProgressTracker(_STATUS_REPO)
_PROGRESS_SERVICE = ProgressService(_STATUS_REPO)

_UPLOAD_SERVICE = UploadService(_APP_PATHS.upload_root)
_OUTPUT_DIRS = OutputDirectoryService(_APP_PATHS.output_root, _APP_PATHS.repo_root)
_RUN_ID_FACTORY = RunIdFactory()
_RUN_KEY_SERVICE = RunKeyService(_PATH_VALIDATOR)
_RUN_VALIDATOR = RunRequestValidator(_PATH_VALIDATOR)
_OUTPUT_HISTORY = OutputHistoryService(_APP_PATHS.output_root)
_RUN_CLEANUP = RunCleanupService(
    _APP_PATHS.output_root, _APP_PATHS.upload_root, _PATH_VALIDATOR
)

_PREPARE_PIPELINE = PreparePipelineRunUseCase(
    validator=_RUN_VALIDATOR,
    run_id_factory=_RUN_ID_FACTORY,
    run_key_service=_RUN_KEY_SERVICE,
    upload_service=_UPLOAD_SERVICE,
    output_directory_service=_OUTPUT_DIRS,
)
_RUN_PIPELINE_SYNC = RunPipelineSyncUseCase(
    command_builder=_PIPELINE_COMMANDS,
    pipeline_runner=_PIPELINE_RUNNER,
    log_service=_PIPELINE_LOGGER,
    result_service=_PIPELINE_RESULTS,
    upload_service=_UPLOAD_SERVICE,
)
_RUN_PIPELINE_ASYNC = RunPipelineAsyncUseCase(
    command_builder=_PIPELINE_COMMANDS,
    pipeline_runner=_PIPELINE_RUNNER,
    log_service=_PIPELINE_LOGGER,
    result_service=_PIPELINE_RESULTS,
    upload_service=_UPLOAD_SERVICE,
    status_repo=_STATUS_REPO,
    progress_tracker=_PROGRESS_TRACKER,
)

_RESULTS_SERVICE = ResultsService()
_RESULTS_ARCHIVE = ResultsArchiveService()
_STATISTICS_SERVICE = StatisticsService()
_MEDIA_SERVICE = MediaService(
    _APP_PATHS.output_root,
    _APP_PATHS.upload_root,
    _PATH_VALIDATOR,
)


def _json_error(message: str | list[str], status: int = 400) -> JsonResponse:
    payload = {"errors": message if isinstance(message, list) else [message]}
    return JsonResponse(payload, status=status)


@method_decorator(csrf_exempt, name="dispatch")
class RunListView(View):
    """List runs or trigger new asynchronous runs."""

    def get(self, request: HttpRequest) -> HttpResponse:
        runs = [
            {
                "run_key": entry.run_key,
                "display_name": entry.display_name,
                "modified_time": entry.modified_time,
            }
            for entry in _OUTPUT_HISTORY.list_runs()
        ]
        return JsonResponse({"runs": runs})

    def post(self, request: HttpRequest) -> HttpResponse:
        preparation = _PREPARE_PIPELINE.execute(
            PipelineRequestData.from_django_request(request)
        )
        if preparation.errors:
            return _json_error(preparation.errors)
        if preparation.prepared is None:
            return _json_error("Unable to prepare pipeline run.")

        spec = preparation.prepared
        results_url = reverse("api_run_detail", kwargs={"run_key": spec.run_key})
        progress_url = reverse("api_progress", kwargs={"run_key": spec.run_key})

        _RUN_PIPELINE_ASYNC.enqueue(
            spec,
            request.POST,
            request.POST.get("fix_header") is not None,
            results_url,
        )

        return JsonResponse(
            {
                "run_key": spec.run_key,
                "progress_url": progress_url,
                "results_url": results_url,
            },
            status=202,
        )


@method_decorator(csrf_exempt, name="dispatch")
class RunSyncView(View):
    """Run the pipeline synchronously and return results."""

    def post(self, request: HttpRequest) -> HttpResponse:
        preparation = _PREPARE_PIPELINE.execute(
            PipelineRequestData.from_django_request(request)
        )
        if preparation.errors:
            return _json_error(preparation.errors)
        if preparation.prepared is None:
            return _json_error("Unable to prepare pipeline run.")

        spec = preparation.prepared
        try:
            result = _RUN_PIPELINE_SYNC.execute(spec, request.POST)
        except (OSError, RuntimeError) as exc:
            return _json_error(f"Failed to start pipeline: {exc}")

        if result.return_code != 0:
            tail_lines = (result.stderr_text or result.stdout_text).splitlines()[-12:]
            return JsonResponse(
                {
                    "run_key": spec.run_key,
                    "status": "failed",
                    "return_code": result.return_code,
                    "log_tail": "\n".join(tail_lines),
                },
                status=500,
            )

        if request.POST.get("fix_header") is not None:
            _RUN_PIPELINE_SYNC.apply_header_fix(spec)

        return JsonResponse(
            {
                "run_key": spec.run_key,
                "status": "completed",
                "results_url": reverse(
                    "api_run_detail", kwargs={"run_key": spec.run_key}
                ),
            }
        )


class RunDetailView(View):
    """Return file listings and metadata for a run."""

    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        run_dir = _PATH_VALIDATOR.resolve_output_dir(_APP_PATHS.output_root, run_key)
        if not run_dir.exists():
            raise Http404("Run not found.")

        files = _RESULTS_SERVICE.list_files(run_dir)
        return JsonResponse(
            {
                "run_key": run_key,
                "files": files,
            }
        )


class PipelineProgressView(View):
    """Progress polling endpoint for running pipelines."""

    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        payload = _PROGRESS_SERVICE.build_payload(run_key)
        if payload is None:
            raise Http404("Run not found.")
        return JsonResponse(
            {
                "run_key": payload.run_key,
                "progress": payload.progress,
                "stage": payload.stage,
                "done": payload.done,
                "error": payload.error,
                "results_url": payload.results_url,
            }
        )


class StatisticsView(View):
    """Return joint angle and landmark statistics as JSON."""

    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        run_dir = _PATH_VALIDATOR.resolve_output_dir(_APP_PATHS.output_root, run_key)
        if not run_dir.exists():
            raise Http404("Run not found.")

        context = _STATISTICS_SERVICE.build_context(run_dir, run_key)
        return JsonResponse(context)


class DownloadAllView(View):
    """Archive download endpoint for run outputs."""

    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        run_dir = _PATH_VALIDATOR.resolve_output_dir(_APP_PATHS.output_root, run_key)
        if not run_dir.exists():
            raise Http404("Run not found.")

        archive = _RESULTS_ARCHIVE.build_archive(run_dir, run_key)
        return FileResponse(
            archive,
            as_attachment=True,
            filename=f"{run_key}.zip",
        )


class DownloadView(View):
    """File download endpoint for run outputs."""

    def get(self, request: HttpRequest, run_key: str, file_path: str) -> HttpResponse:
        target = _MEDIA_SERVICE.resolve_output_file(run_key, file_path)
        return FileResponse(target.open("rb"), as_attachment=True, filename=target.name)


class MediaView(View):
    """Media streaming endpoint for output files."""

    def get(self, request: HttpRequest, run_key: str, file_path: str) -> HttpResponse:
        target = _MEDIA_SERVICE.resolve_output_file(run_key, file_path)
        return FileResponse(target.open("rb"), as_attachment=False)


class UploadMediaView(View):
    """Media streaming endpoint for uploaded source files."""

    def get(self, request: HttpRequest, run_key: str, file_path: str) -> HttpResponse:
        target = _MEDIA_SERVICE.resolve_upload_file(run_key, file_path)
        return FileResponse(target.open("rb"), as_attachment=False)


@method_decorator(csrf_exempt, name="dispatch")
class DeleteRunView(View):
    """Delete an output run and uploaded input data."""

    def post(self, request: HttpRequest, run_key: str) -> HttpResponse:
        deleted = _RUN_CLEANUP.delete_run(run_key)
        if not deleted:
            return _json_error("Run not found.", status=404)
        return JsonResponse({"deleted": True})
