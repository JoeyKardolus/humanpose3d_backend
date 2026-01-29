"""Django views for the webapp pipeline experience."""

from __future__ import annotations

import json
import mimetypes
import tempfile
import zipfile
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

from src.application.config.paths import AppPaths, StoragePaths
from src.application.dto.dof_config import DofConfig
from src.application.dto.pipeline_request import PipelineRequestData
from src.application.repositories.analytics_repository import AnalyticsRepository
from src.application.repositories.run_status_repository import (
    RunStatusRepository,
)
from src.application.services.analytics_service import AnalyticsService
from src.application.services.kinematics_quality_service import KinematicsQualityService
from src.application.services.media_service import MediaService
from src.application.services.output_directory_service import (
    OutputDirectoryService,
)
from src.application.services.output_history_service import OutputHistoryService
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
from src.application.services.progress_service import ProgressService
from src.application.services.joint_angle_recompute_service import (
    JointAngleRecomputeService,
)
from src.application.services.run_cleanup_service import RunCleanupService
from src.application.services.run_id_factory import RunIdFactory
from src.application.services.run_key_service import RunKeyService
from src.application.services.statistics_service import StatisticsService
from src.application.services.upload_service import UploadService
from src.application.use_cases.prepare_pipeline_run import (
    PreparePipelineRunUseCase,
)
from src.application.use_cases.run_pipeline_async import RunPipelineAsyncUseCase
from src.application.use_cases.run_pipeline_sync import RunPipelineSyncUseCase
from src.application.validators.path_validator import PathValidator
from src.application.validators.run_request_validator import RunRequestValidator


def _parse_pipeline_error(
    return_code: int, stderr_text: str, stdout_text: str
) -> tuple[list[str], str]:
    """Parse pipeline error output into user-friendly messages and technical details.

    Returns a tuple of (user_messages, technical_details).
    """
    output_text = stderr_text or stdout_text or ""
    output_lower = output_text.lower()

    # Common error patterns and their user-friendly messages
    user_messages: list[str] = []

    if "video duration exceeds" in output_lower or "video is too long" in output_lower:
        user_messages.append(
            "Video is too long. Please upload a video under 1 minute."
        )
    elif "ffprobe" in output_lower and ("not found" in output_lower or "no such file" in output_lower):
        user_messages.append(
            "Video processing tools are not installed on the server. "
            "Please contact the administrator."
        )
    elif "out of memory" in output_lower or "memory" in output_lower and "error" in output_lower:
        user_messages.append(
            "Server ran out of memory processing this video. "
            "Try a shorter or lower-resolution video."
        )
    elif "no landmarks detected" in output_lower or "no pose detected" in output_lower:
        user_messages.append(
            "Could not detect a person in the video. "
            "Ensure the subject is fully visible and well-lit."
        )
    elif "cuda" in output_lower and "error" in output_lower:
        user_messages.append(
            "GPU processing failed. The server will retry with CPU."
        )
    elif "permission denied" in output_lower:
        user_messages.append(
            "Server file permission error. Please contact the administrator."
        )
    elif "invalid video" in output_lower or "corrupt" in output_lower:
        user_messages.append(
            "The video file appears to be invalid or corrupted. "
            "Please try a different video."
        )
    else:
        user_messages.append(
            "Pipeline failed to complete. This may be due to video quality issues."
        )

    # Add exit code hint only if non-standard
    if return_code not in (0, 1):
        user_messages.append(f"Process exited with code {return_code}.")

    # Build technical details for collapsible section
    tail_lines = output_text.splitlines()[-20:]
    technical_details = "\n".join(tail_lines) if tail_lines else "No output captured."

    return user_messages, technical_details


# Module-level wiring keeps view classes thin and testable.
_APP_PATHS = AppPaths.from_anchor(Path(__file__))
_STORAGE_PATHS = StoragePaths.load()
_PATH_VALIDATOR = PathValidator()
_STATUS_REPO = RunStatusRepository()

_PIPELINE_RUNNER = PipelineRunner(_APP_PATHS.repo_root)
_PIPELINE_RESULTS = PipelineResultService(_APP_PATHS.repo_root)
_PIPELINE_COMMANDS = PipelineCommandBuilder(_APP_PATHS.repo_root)
_PROGRESS_TRACKER = PipelineProgressTracker(_STATUS_REPO)
_PROGRESS_SERVICE = ProgressService(_STATUS_REPO)

# Analytics and quality services
_ANALYTICS_REPO = AnalyticsRepository(_STORAGE_PATHS.logs_root)
_ANALYTICS_SERVICE = AnalyticsService(_ANALYTICS_REPO)
_QUALITY_SERVICE = KinematicsQualityService()

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
    result_service=_PIPELINE_RESULTS,
    upload_service=_UPLOAD_SERVICE,
)
_RUN_PIPELINE_ASYNC = RunPipelineAsyncUseCase(
    command_builder=_PIPELINE_COMMANDS,
    pipeline_runner=_PIPELINE_RUNNER,
    result_service=_PIPELINE_RESULTS,
    upload_service=_UPLOAD_SERVICE,
    status_repo=_STATUS_REPO,
    progress_tracker=_PROGRESS_TRACKER,
    analytics_service=_ANALYTICS_SERVICE,
    quality_service=_QUALITY_SERVICE,
)

_STATISTICS_SERVICE = StatisticsService()
_JOINT_ANGLE_RECOMPUTE_SERVICE = JointAngleRecomputeService()
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
            user_messages, technical_details = _parse_pipeline_error(
                result.return_code,
                result.stderr_text or "",
                result.stdout_text or "",
            )
            return render(
                request,
                "home.html",
                self._build_context(
                    errors=user_messages,
                    error_details=technical_details,
                ),
            )

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

        # List all files in run directory
        files = [
            {
                "name": path.name,
                "relative_path": path.relative_to(run_dir).as_posix(),
                "size_kb": max(path.stat().st_size // 1024, 1),
            }
            for path in sorted(run_dir.rglob("*"))
            if path.is_file()
        ]
        preview_video = None
        preview_video_type = None
        preview_candidates = sorted(run_dir.rglob("*_preview.*"))
        for candidate in preview_candidates:
            if (
                candidate.suffix.lower() in {".mp4", ".webm"}
                and candidate.stat().st_size > 1024
            ):
                preview_video = candidate
                preview_video_type = mimetypes.guess_type(candidate.name)[0]
                break

        source_video = None
        source_video_type = None
        source_dir = run_dir / "source"
        if source_dir.exists():
            for candidate in sorted(source_dir.iterdir()):
                if candidate.is_file() and candidate.suffix.lower() in {
                    ".mp4",
                    ".mov",
                    ".webm",
                    ".m4v",
                }:
                    source_video = candidate
                    source_video_type = mimetypes.guess_type(candidate.name)[0]
                    break
        return render(
            request,
            "results.html",
            {
                "run_key": run_key,
                "files": files,
                "source_video_path": source_video.relative_to(run_dir).as_posix()
                if source_video
                else None,
                "source_video_type": source_video_type,
                "preview_video_path": preview_video.relative_to(run_dir).as_posix()
                if preview_video
                else None,
                "preview_video_type": preview_video_type,
            },
        )


class DownloadAllView(View):
    """Archive download endpoint for run outputs."""

    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        """Download all output files as a zip archive."""
        run_dir = _PATH_VALIDATOR.resolve_output_dir(_APP_PATHS.output_root, run_key)
        if not run_dir.exists():
            raise Http404("Run not found.")

        # Create zip archive
        archive = tempfile.NamedTemporaryFile(suffix=f"_{run_key}.zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for path in sorted(run_dir.rglob("*")):
                if path.is_file():
                    zip_file.write(path, arcname=path.relative_to(run_dir).as_posix())
        archive.seek(0)

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


class DeleteRunView(View):
    """Delete an output run and uploaded input data."""

    def post(self, request: HttpRequest, run_key: str) -> HttpResponse:
        """Delete stored results for a run."""
        deleted = _RUN_CLEANUP.delete_run(run_key)
        if not deleted:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return JsonResponse({"error": "Run not found."}, status=404)
            raise Http404("Run not found.")
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"deleted": True})
        return redirect("home")


class JointAngleConfigView(View):
    """API endpoint for recomputing joint angles with DOF configuration."""

    def post(self, request: HttpRequest, run_key: str) -> HttpResponse:
        """Recompute joint angles with specified DOF configuration.

        Request body:
        {
            "dof_config": {
                "pelvis": ["flex", "abd", "rot"],
                "hip_R": ["flex"],
                ...
            }
        }

        Response:
        {
            "series": { ... },
            "success": true
        }
        """
        run_dir = _PATH_VALIDATOR.resolve_output_dir(_APP_PATHS.output_root, run_key)
        if not run_dir.exists():
            return JsonResponse({"error": "Run not found."}, status=404)

        try:
            body = json.loads(request.body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JsonResponse({"error": "Invalid JSON body."}, status=400)

        dof_config_data = body.get("dof_config", {})
        try:
            dof_config = DofConfig.from_json(dof_config_data)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)

        try:
            series = _JOINT_ANGLE_RECOMPUTE_SERVICE.recompute(run_dir, dof_config)
        except FileNotFoundError as exc:
            return JsonResponse({"error": str(exc)}, status=404)
        except Exception as exc:
            return JsonResponse(
                {"error": f"Failed to recompute joint angles: {exc}"},
                status=500,
            )

        return JsonResponse({"series": series, "success": True})
