"""HTTP API endpoints for the HumanPose3D pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from time import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

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

from src.application.config.paths import AppPaths
from src.application.config.paths import StoragePaths
from src.application.dto.pipeline_request import PipelineRequestData
from src.application.repositories.run_status_repository import (
    RunStatusRepository,
)
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
from src.application.services.results_archive_service import (
    ResultsArchiveService,
)
from src.application.services.results_service import ResultsService
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


_APP_PATHS = AppPaths.from_anchor(Path(__file__))
_PATH_VALIDATOR = PathValidator()
_STATUS_REPO = RunStatusRepository()

_PIPELINE_RUNNER = PipelineRunner(_APP_PATHS.repo_root)
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
)

_RESULTS_SERVICE = ResultsService()
_RESULTS_ARCHIVE = ResultsArchiveService()
_STATISTICS_SERVICE = StatisticsService()
_MEDIA_SERVICE = MediaService(
    _APP_PATHS.output_root,
    _APP_PATHS.upload_root,
    _PATH_VALIDATOR,
)

_MODELS_BASE_URL = "https://raw.githubusercontent.com/JoeyKardolus/humanpose3d_backend/models"
_MODEL_DOWNLOAD_LOCK = Lock()
_MODEL_DOWNLOAD_JOBS: dict[str, "ModelDownloadJob"] = {}


@dataclass(frozen=True)
class ModelAsset:
    name: str
    source_path: str
    target_path: Path


@dataclass
class ModelDownloadJob:
    total_bytes: int
    downloaded_bytes: int = 0
    downloaded_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    status: str = "running"
    started_at: float = field(default_factory=time)


def _get_repo_root() -> Path:
    """Resolve the repository root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "models").exists() and (parent / "src").exists():
            return parent
    return current.parents[3]


def _resolve_model_path(
    name: str, source_path: str, storage_path: Path, repo_models: Path
) -> ModelAsset:
    """Resolve model path, preferring repo location if exists."""
    repo_path = repo_models / Path(source_path).relative_to("models")
    target = repo_path if repo_path.exists() else storage_path
    return ModelAsset(name=name, source_path=source_path, target_path=target)


def _get_model_assets(storage_paths: StoragePaths) -> list[ModelAsset]:
    """Get required model assets - only core models needed for basic pipeline.

    Note: GRU.h5 is bundled with Pose2Sim (FLK package) and doesn't need separate download.
    Only pose_landmarker_heavy.task needs to be downloaded from Google's model garden.
    """
    repo_models = _get_repo_root() / "models"

    # Core models required for basic pipeline
    # GRU.h5 is installed with Pose2Sim via the FLK package, no need to distribute separately
    return [
        _resolve_model_path(
            "pose_landmarker_heavy.task",
            "models/pose_landmarker_heavy.task",
            storage_paths.models_root / "pose_landmarker_heavy.task",
            repo_models,
        ),
    ]


def _get_optional_model_assets(storage_paths: StoragePaths) -> list[ModelAsset]:
    """Get optional model assets for experimental features (POF, joint refinement)."""
    repo_models = _get_repo_root() / "models"

    return [
        # POF model - only needed when --camera-pof is enabled
        _resolve_model_path(
            "best_pof_semgcn-temporal_model.pth",
            "models/checkpoints/best_pof_semgcn-temporal_model.pth",
            storage_paths.checkpoints_root / "best_pof_semgcn-temporal_model.pth",
            repo_models,
        ),
        # Joint refinement model - only needed when --joint-refinement is enabled
        _resolve_model_path(
            "best_joint_model.pth",
            "models/checkpoints/best_joint_model.pth",
            storage_paths.checkpoints_root / "best_joint_model.pth",
            repo_models,
        ),
    ]


def _head_content_length(url: str) -> int | None:
    request = Request(url, method="HEAD")
    try:
        with urlopen(request, timeout=10) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (OSError, URLError, ValueError):
        return None


def _download_to_path(url: str, target_path: Path, job_id: str, total_bytes: int) -> None:
    with urlopen(url, timeout=30) as response:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                with _MODEL_DOWNLOAD_LOCK:
                    job = _MODEL_DOWNLOAD_JOBS.get(job_id)
                    if job is None:
                        return
                    job.downloaded_bytes += len(chunk)


def _start_model_download(job_id: str, assets: list[ModelAsset], total_bytes: int) -> None:
    for asset in assets:
        url = f"{_MODELS_BASE_URL}/{asset.source_path}"
        try:
            _download_to_path(url, asset.target_path, job_id, total_bytes)
            with _MODEL_DOWNLOAD_LOCK:
                job = _MODEL_DOWNLOAD_JOBS.get(job_id)
                if job is None:
                    return
                job.downloaded_files.append(asset.name)
        except (OSError, URLError) as exc:
            with _MODEL_DOWNLOAD_LOCK:
                job = _MODEL_DOWNLOAD_JOBS.get(job_id)
                if job is None:
                    return
                job.errors.append(f"{asset.name}: {exc}")
                job.status = "failed"
            return

    with _MODEL_DOWNLOAD_LOCK:
        job = _MODEL_DOWNLOAD_JOBS.get(job_id)
        if job is None:
            return
        job.status = "completed"


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


class ModelsStatusView(View):
    """Expose model asset availability for the web UI."""

    def get(self, request: HttpRequest) -> JsonResponse:
        storage_paths = StoragePaths.load()
        assets = _get_model_assets(storage_paths)
        missing = [asset.name for asset in assets if not asset.target_path.exists()]
        return JsonResponse(
            {
                "missing": missing,
                "expected": [asset.name for asset in assets],
                "storage_root": str(storage_paths.root),
            }
        )


@method_decorator(csrf_exempt, name="dispatch")
class ModelsDownloadView(View):
    """Download missing model assets to the shared storage directory."""

    def post(self, request: HttpRequest) -> JsonResponse:
        storage_paths = StoragePaths.load()
        assets = _get_model_assets(storage_paths)
        missing_assets = [asset for asset in assets if not asset.target_path.exists()]
        if not missing_assets:
            return JsonResponse({"downloaded": [], "missing": []})

        total_bytes = 0
        for asset in missing_assets:
            url = f"{_MODELS_BASE_URL}/{asset.source_path}"
            length = _head_content_length(url)
            if length is None:
                total_bytes = 0
                break
            total_bytes += length

        job_id = uuid4().hex
        job = ModelDownloadJob(total_bytes=total_bytes)
        with _MODEL_DOWNLOAD_LOCK:
            _MODEL_DOWNLOAD_JOBS[job_id] = job

        worker = Thread(
            target=_start_model_download,
            args=(job_id, missing_assets, total_bytes),
            daemon=True,
        )
        worker.start()

        return JsonResponse(
            {
                "job_id": job_id,
                "progress_url": reverse(
                    "api_models_download_progress", kwargs={"job_id": job_id}
                ),
                "missing": [asset.name for asset in missing_assets],
            },
            status=202,
        )


class ModelsDownloadProgressView(View):
    """Return download progress for model assets."""

    def get(self, request: HttpRequest, job_id: str) -> JsonResponse:
        with _MODEL_DOWNLOAD_LOCK:
            job = _MODEL_DOWNLOAD_JOBS.get(job_id)
            if job is None:
                return JsonResponse({"error": "Download job not found."}, status=404)
            total = job.total_bytes
            downloaded = job.downloaded_bytes
            status = job.status
            errors = list(job.errors)
            files = list(job.downloaded_files)

        progress = 0
        if total > 0:
            progress = min(100, int((downloaded / total) * 100))
        elif status == "completed":
            progress = 100

        payload = {
            "status": status,
            "progress": progress,
            "downloaded_files": files,
            "errors": errors,
        }
        return JsonResponse(payload)


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
