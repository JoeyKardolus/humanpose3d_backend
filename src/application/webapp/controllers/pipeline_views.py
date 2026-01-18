from __future__ import annotations

"""Django views for the webapp pipeline experience."""

import mimetypes
from pathlib import Path

from django.http import (
    FileResponse,
    Http404,
    HttpRequest,
    HttpResponse,
    JsonResponse,
)
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views import View

from src.application.webapp.config.paths import AppPaths
from src.application.webapp.dto.pipeline_request import PipelineRequestData
from src.application.webapp.repositories.run_status_repository import RunStatusRepository
from src.application.webapp.services.media_service import MediaService
from src.application.webapp.services.output_directory_service import OutputDirectoryService
from src.application.webapp.services.output_history_service import OutputHistoryService
from src.application.webapp.services.pipeline_command_builder import PipelineCommandBuilder
from src.application.webapp.services.pipeline_log_service import PipelineLogService
from src.application.webapp.services.pipeline_progress_tracker import PipelineProgressTracker
from src.application.webapp.services.pipeline_result_service import PipelineResultService
from src.application.webapp.services.pipeline_runner import PipelineRunner
from src.application.webapp.services.progress_service import ProgressService
from src.application.webapp.services.results_service import ResultsService
from src.application.webapp.services.results_archive_service import ResultsArchiveService
from src.application.webapp.services.run_id_factory import RunIdFactory
from src.application.webapp.services.run_key_service import RunKeyService
from src.application.webapp.services.statistics_service import StatisticsService
from src.application.webapp.services.upload_service import UploadService
from src.application.webapp.use_cases.prepare_pipeline_run import PreparePipelineRunUseCase
from src.application.webapp.use_cases.run_pipeline_async import RunPipelineAsyncUseCase
from src.application.webapp.use_cases.run_pipeline_sync import RunPipelineSyncUseCase
from src.application.webapp.validators.path_validator import PathValidator
from src.application.webapp.validators.run_request_validator import RunRequestValidator


# Module-level wiring keeps view classes thin and testable.
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
)
_RUN_PIPELINE_ASYNC = RunPipelineAsyncUseCase(
    command_builder=_PIPELINE_COMMANDS,
    pipeline_runner=_PIPELINE_RUNNER,
    log_service=_PIPELINE_LOGGER,
    result_service=_PIPELINE_RESULTS,
    status_repo=_STATUS_REPO,
    progress_tracker=_PROGRESS_TRACKER,
)

_RESULTS_SERVICE = ResultsService()
_RESULTS_ARCHIVE = ResultsArchiveService()
_STATISTICS_SERVICE = StatisticsService(_APP_PATHS.upload_root)
_MEDIA_SERVICE = MediaService(
    _APP_PATHS.output_root,
    _APP_PATHS.upload_root,
    _PATH_VALIDATOR,
)


class HomeView(View):
    """Home page view for uploading and running the pipeline."""

    def get(self, request: HttpRequest) -> HttpResponse:
        """Render the home page."""
        return render(request, "home.html", self._build_context())

    def post(self, request: HttpRequest) -> HttpResponse:
        """Run the pipeline synchronously and redirect to results."""
        preparation = _PREPARE_PIPELINE.execute(
            PipelineRequestData.from_django_request(request)
        )
        if preparation.errors:
            return render(
                request,
                "home.html",
                self._build_context(errors=preparation.errors),
            )
        if preparation.prepared is None:
            return render(
                request,
                "home.html",
                self._build_context(errors=["Unable to prepare pipeline run."]),
            )

        spec = preparation.prepared
        try:
            result = _RUN_PIPELINE_SYNC.execute(spec, request.POST)
        except (OSError, RuntimeError) as exc:
            return render(
                request,
                "home.html",
                self._build_context(errors=[f"Failed to start pipeline: {exc}"]),
            )

        if result.return_code != 0:
            log_path = spec.pipeline_run_dir / "pipeline_error.log"
            tail_lines = (result.stderr_text or result.stdout_text).splitlines()[-12:]
            tail_text = "\n".join(tail_lines)
            return render(
                request,
                "home.html",
                self._build_context(
                    errors=[
                        "Pipeline failed to run. Check the server logs for details.",
                        f"Exit code: {result.return_code}",
                        f"Log file: {log_path.relative_to(_APP_PATHS.repo_root)}",
                        f"Last output:\n{tail_text}"
                        if tail_text
                        else "No output captured.",
                    ]
                ),
            )

        if request.POST.get("fix_header") is not None:
            _RUN_PIPELINE_SYNC.apply_header_fix(spec)

        return redirect("results", run_key=spec.run_key)

    def _build_context(self, **kwargs: object) -> dict[str, object]:
        """Build template context shared by home page renders."""
        context: dict[str, object] = {"previous_runs": _OUTPUT_HISTORY.list_runs()}
        context.update(kwargs)
        return context


class RunPipelineView(View):
    """Async pipeline trigger view for AJAX submissions."""

    def post(self, request: HttpRequest) -> HttpResponse:
        """Start a pipeline run and return progress endpoints."""
        preparation = _PREPARE_PIPELINE.execute(
            PipelineRequestData.from_django_request(request)
        )
        if preparation.errors:
            return JsonResponse({"errors": preparation.errors}, status=400)
        if preparation.prepared is None:
            return JsonResponse(
                {"errors": ["Unable to prepare pipeline run."]},
                status=400,
            )

        spec = preparation.prepared
        results_url = reverse("results", kwargs={"run_key": spec.run_key})
        progress_url = reverse("progress", kwargs={"run_key": spec.run_key})

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
            }
        )


class PipelineProgressView(View):
    """Progress polling endpoint for running pipelines."""

    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        """Return JSON progress for a running pipeline."""
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


class ResultsView(View):
    """Results landing page for finished runs."""

    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        """Render the results list for the run output."""
        run_dir = _PATH_VALIDATOR.resolve_output_dir(_APP_PATHS.output_root, run_key)
        if not run_dir.exists():
            raise Http404("Run not found.")

        files = _RESULTS_SERVICE.list_files(run_dir)
        return render(
            request,
            "results.html",
            {
                "run_key": run_key,
                "files": files,
            },
        )


class DownloadAllView(View):
    """Archive download endpoint for run outputs."""

    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        """Download all output files as a zip archive."""
        run_dir = _PATH_VALIDATOR.resolve_output_dir(_APP_PATHS.output_root, run_key)
        if not run_dir.exists():
            raise Http404("Run not found.")

        archive = _RESULTS_ARCHIVE.build_archive(run_dir, run_key)
        return FileResponse(
            archive,
            as_attachment=True,
            filename=f"{run_key}.zip",
        )


class StatisticsView(View):
    """Statistics view for joint angle and landmark plots."""

    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        """Render statistics charts and video previews."""
        run_dir = _PATH_VALIDATOR.resolve_output_dir(_APP_PATHS.output_root, run_key)
        if not run_dir.exists():
            raise Http404("Run not found.")

        context = _STATISTICS_SERVICE.build_context(run_dir, run_key)
        return render(request, "statistics.html", context)


class DownloadView(View):
    """File download endpoint for run outputs."""

    def get(self, request: HttpRequest, run_key: str, file_path: str) -> HttpResponse:
        """Download an output file."""
        target = _MEDIA_SERVICE.resolve_output_file(run_key, file_path)
        return FileResponse(target.open("rb"), as_attachment=True, filename=target.name)


class MediaView(View):
    """Media streaming endpoint for output files."""

    def get(self, request: HttpRequest, run_key: str, file_path: str) -> HttpResponse:
        """Stream an output file without attachment headers."""
        target = _MEDIA_SERVICE.resolve_output_file(run_key, file_path)
        content_type, _ = mimetypes.guess_type(target.name)
        return FileResponse(
            target.open("rb"),
            as_attachment=False,
            content_type=content_type or "application/octet-stream",
        )


class UploadMediaView(View):
    """Media streaming endpoint for uploaded source files."""
    def get(self, request: HttpRequest, run_key: str, file_path: str) -> HttpResponse:
        """Stream an uploaded source file."""
        target = _MEDIA_SERVICE.resolve_upload_file(run_key, file_path)
        content_type, _ = mimetypes.guess_type(target.name)
        return FileResponse(
            target.open("rb"),
            as_attachment=False,
            content_type=content_type or "application/octet-stream",
        )
