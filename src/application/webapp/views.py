import csv
import mimetypes
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Mapping, TextIO

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


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = (REPO_ROOT / "data" / "output" / "pose-3d").resolve()
UPLOAD_ROOT = (REPO_ROOT / "data" / "input" / "webapp").resolve()

RUN_STATUS: dict[str, dict[str, object]] = {}
RUN_STATUS_LOCK = threading.Lock()

PROGRESS_AUGMENT_START = 35.0
PROGRESS_AUGMENT_END = 70.0
PROGRESS_INIT_CAP = 12.0


def _safe_relative_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return None
    if ".." in candidate.parts:
        return None
    return candidate


def _coerce_bool(value: str | None) -> bool:
    return value is not None


def _resolve_output_dir(run_key: str) -> Path:
    run_dir = (OUTPUT_ROOT / run_key).resolve()
    if OUTPUT_ROOT not in run_dir.parents and run_dir != OUTPUT_ROOT:
        raise Http404("Invalid output path.")
    return run_dir


def _set_run_status(run_key: str, **updates: object) -> None:
    with RUN_STATUS_LOCK:
        status = RUN_STATUS.setdefault(run_key, {})
        status.update(updates)


def _get_run_status(run_key: str) -> dict[str, object] | None:
    with RUN_STATUS_LOCK:
        status = RUN_STATUS.get(run_key)
        return dict(status) if status is not None else None


def _update_progress_from_line(run_key: str, line: str) -> None:
    progress = None
    stage = None
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
    elif "[main] step3 augment" in line:
        progress = PROGRESS_AUGMENT_START
        stage = "Augmenting markers"
    elif "[augment] cycle" in line:
        match = re.search(r"cycle (\d+)/(\d+)", line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                span = PROGRESS_AUGMENT_END - PROGRESS_AUGMENT_START
                progress = PROGRESS_AUGMENT_START + span * (current / total)
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

    status = _get_run_status(run_key)
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
        _set_run_status(run_key, **updates)
        _set_run_status(run_key, last_update=time.monotonic())


def _build_pipeline_command(
    upload_path: Path, form_data: Mapping[str, str], sex_raw: str
) -> list[str]:
    main_path = REPO_ROOT / "main.py"
    command = [
        sys.executable,
        "-u",
        str(main_path),
        "--video",
        str(upload_path),
    ]

    def _add_flag(name: str, value: str | None, default: str | None = None) -> None:
        if value is None or value == "":
            if default is not None:
                command.extend([name, default])
            return
        command.extend([name, value])

    _add_flag("--height", form_data.get("height"), default="1.78")
    _add_flag("--mass", form_data.get("weight"), default="75.0")
    _add_flag("--age", form_data.get("age"), default="30")
    if sex_raw in {"male", "female"}:
        _add_flag("--sex", sex_raw)
    _add_flag("--visibility-min", form_data.get("visibility_min"), default="0.3")
    _add_flag(
        "--augmentation-cycles", form_data.get("augmentation_cycles"), default="20"
    )
    _add_flag(
        "--joint-angle-smooth-window",
        form_data.get("joint_angle_smooth_window"),
        default="9",
    )
    _add_flag("--bone-smooth-window", form_data.get("bone_smooth_window"), default="21")
    _add_flag("--ground-percentile", form_data.get("ground_percentile"), default="5.0")
    _add_flag("--ground-margin", form_data.get("ground_margin"), default="0.02")
    _add_flag(
        "--bone-length-tolerance",
        form_data.get("bone_length_tolerance"),
        default="0.15",
    )
    _add_flag("--bone-depth-weight", form_data.get("bone_depth_weight"), default="0.8")
    _add_flag(
        "--bone-length-iterations", form_data.get("bone_length_iterations"), default="3"
    )
    _add_flag(
        "--multi-constraint-iterations",
        form_data.get("multi_constraint_iterations"),
        default="10",
    )

    if _coerce_bool(form_data.get("estimate_missing")):
        command.append("--estimate-missing")
    if _coerce_bool(form_data.get("force_complete")):
        command.append("--force-complete")
    if _coerce_bool(form_data.get("anatomical_constraints")):
        command.append("--anatomical-constraints")
    if _coerce_bool(form_data.get("bone_length_constraints")):
        command.append("--bone-length-constraints")
    if _coerce_bool(form_data.get("multi_constraint_optimization")):
        command.append("--multi-constraint-optimization")
    if _coerce_bool(form_data.get("compute_all_joint_angles")):
        command.append("--compute-all-joint-angles")
    return command


def _write_pipeline_log(log_path: Path, stdout: str, stderr: str) -> None:
    log_path.write_text(
        "\n".join(
            [
                "[stdout]",
                stdout.strip(),
                "",
                "[stderr]",
                stderr.strip(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_pipeline_command(
    command: list[str], on_line: Callable[[str], None] | None = None
) -> tuple[int, str, str]:
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _stream_reader(stream: TextIO, sink: TextIO, collector: list[str]) -> None:
        for line in iter(stream.readline, ""):
            collector.append(line)
            if on_line:
                on_line(line)
            sink.write(line)
            sink.flush()

    with subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as process:
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Failed to attach to pipeline output streams.")

        stdout_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stdout, sys.stdout, stdout_lines),
        )
        stderr_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stderr, sys.stderr, stderr_lines),
        )
        stdout_thread.start()
        stderr_thread.start()
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()

    return return_code, "".join(stdout_lines), "".join(stderr_lines)


def _prepare_pipeline_run(
    request: HttpRequest,
) -> tuple[list[str], dict[str, object] | None]:
    errors: list[str] = []
    uploaded = request.FILES.get("video")
    if not uploaded:
        errors.append("Please upload a video file before submitting.")

    if request.POST.get("consent") != "accepted":
        errors.append("You must confirm participant consent before running an analysis.")

    output_location = _safe_relative_path(request.POST.get("output_location", "").strip())
    if request.POST.get("output_location") and not output_location:
        errors.append("Output location must be a relative path (no .. segments).")

    sex_raw = request.POST.get("sex", "").strip().lower()
    if sex_raw and sex_raw not in {"male", "female"}:
        errors.append("Sex must be Male or Female for the current pipeline.")

    if errors:
        return errors, None

    run_id = f"{Path(uploaded.name).stem}-{uuid.uuid4().hex[:8]}"
    safe_run_id = "".join(
        ch for ch in run_id if ch.isalnum() or ch in {"-", "_"}
    ).strip("-_")
    if not safe_run_id:
        safe_run_id = uuid.uuid4().hex[:12]
    run_key = safe_run_id
    if output_location:
        run_key = (output_location / safe_run_id).as_posix()

    upload_dir = UPLOAD_ROOT / safe_run_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"{safe_run_id}{Path(uploaded.name).suffix}"
    with upload_path.open("wb") as handle:
        for chunk in uploaded.chunks():
            handle.write(chunk)

    pipeline_run_dir = OUTPUT_ROOT / safe_run_id
    output_dir = OUTPUT_ROOT / run_key
    pipeline_run_dir.mkdir(parents=True, exist_ok=True)
    if output_dir != pipeline_run_dir and output_dir.exists():
        errors.append("Output location already exists. Choose a different output folder.")
        errors.append(f"Output path: {output_dir.relative_to(REPO_ROOT)}")
        return errors, None

    return (
        [],
        {
            "sex_raw": sex_raw,
            "run_key": run_key,
            "safe_run_id": safe_run_id,
            "upload_path": upload_path,
            "pipeline_run_dir": pipeline_run_dir,
            "output_dir": output_dir,
        },
    )


def _pipeline_worker(
    run_key: str,
    safe_run_id: str,
    output_dir: Path,
    pipeline_run_dir: Path,
    command: list[str],
    fix_header: bool,
) -> None:
    _set_run_status(
        run_key,
        progress=2.0,
        stage="Initializing models",
        done=False,
        error=None,
        started_at=time.monotonic(),
        last_update=time.monotonic(),
    )

    def _handle_line(line: str) -> None:
        _update_progress_from_line(run_key, line)

    try:
        return_code, stdout_text, stderr_text = _run_pipeline_command(
            command,
            on_line=_handle_line,
        )
    except (OSError, RuntimeError) as exc:
        _set_run_status(
            run_key,
            done=True,
            error=f"Failed to start pipeline: {exc}",
            stage="Failed",
        )
        return

    if return_code != 0:
        log_path = pipeline_run_dir / "pipeline_error.log"
        _write_pipeline_log(log_path, stdout_text, stderr_text)
        _set_run_status(
            run_key,
            done=True,
            error="Pipeline failed. Check logs for details.",
            stage="Failed",
            last_update=time.monotonic(),
        )
        return

    log_path = pipeline_run_dir / "pipeline.log"
    _write_pipeline_log(
        log_path,
        stdout_text,
        stderr_text,
    )

    if output_dir != pipeline_run_dir:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pipeline_run_dir), str(output_dir))

    if fix_header:
        from src.datastream.data_stream import header_fix_strict

        final_trc = output_dir / f"{safe_run_id}_final.trc"
        if final_trc.exists():
            header_fix_strict(final_trc)

    _set_run_status(
        run_key,
        progress=100.0,
        stage="Complete",
        done=True,
        last_update=time.monotonic(),
    )


class HomeView(View):
    def get(self, request: HttpRequest) -> HttpResponse:
        return render(request, "home.html")

    def post(self, request: HttpRequest) -> HttpResponse:
        errors, prepared = _prepare_pipeline_run(request)
        if errors:
            return render(
                request,
                "home.html",
                {"errors": errors},
            )
        if prepared is None:
            return render(
                request,
                "home.html",
                {"errors": ["Unable to prepare pipeline run."]},
            )

        sex_raw = prepared["sex_raw"]
        run_key = prepared["run_key"]
        safe_run_id = prepared["safe_run_id"]
        upload_path = prepared["upload_path"]
        pipeline_run_dir = prepared["pipeline_run_dir"]
        output_dir = prepared["output_dir"]

        command = _build_pipeline_command(
            upload_path, request.POST, str(sex_raw)
        )

        try:
            return_code, stdout_text, stderr_text = _run_pipeline_command(command)
        except (OSError, RuntimeError) as exc:
            return render(
                request,
                "home.html",
                {"errors": [f"Failed to start pipeline: {exc}"]},
            )
        if return_code != 0:
            log_path = pipeline_run_dir / "pipeline_error.log"
            _write_pipeline_log(log_path, stdout_text, stderr_text)
            tail_lines = (stderr_text or stdout_text).splitlines()[-12:]
            tail_text = "\n".join(tail_lines)
            return render(
                request,
                "home.html",
                {
                    "errors": [
                        "Pipeline failed to run. Check the server logs for details.",
                        f"Exit code: {return_code}",
                        f"Log file: {log_path.relative_to(REPO_ROOT)}",
                        f"Last output:\n{tail_text}"
                        if tail_text
                        else "No output captured.",
                    ]
                },
            )
        log_path = pipeline_run_dir / "pipeline.log"
        _write_pipeline_log(
            log_path,
            stdout_text,
            stderr_text,
        )

        if output_dir != pipeline_run_dir:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pipeline_run_dir), str(output_dir))

        if _coerce_bool(request.POST.get("fix_header")):
            from src.datastream.data_stream import header_fix_strict

            final_trc = output_dir / f"{safe_run_id}_final.trc"
            if final_trc.exists():
                header_fix_strict(final_trc)

        return redirect("results", run_key=run_key)


class RunPipelineView(View):
    def post(self, request: HttpRequest) -> HttpResponse:
        errors, prepared = _prepare_pipeline_run(request)
        if errors:
            return JsonResponse({"errors": errors}, status=400)
        if prepared is None:
            return JsonResponse(
                {"errors": ["Unable to prepare pipeline run."]},
                status=400,
            )

        sex_raw = prepared["sex_raw"]
        run_key = prepared["run_key"]
        safe_run_id = prepared["safe_run_id"]
        upload_path = prepared["upload_path"]
        pipeline_run_dir = prepared["pipeline_run_dir"]
        output_dir = prepared["output_dir"]

        command = _build_pipeline_command(upload_path, request.POST, str(sex_raw))
        results_url = reverse("results", kwargs={"run_key": run_key})
        progress_url = reverse("progress", kwargs={"run_key": run_key})

        _set_run_status(
            run_key,
            progress=1.0,
            stage="Queued",
            done=False,
            error=None,
            results_url=results_url,
        )

        worker = threading.Thread(
            target=_pipeline_worker,
            args=(
                run_key,
                str(safe_run_id),
                output_dir,
                pipeline_run_dir,
                command,
                _coerce_bool(request.POST.get("fix_header")),
            ),
            daemon=True,
        )
        worker.start()

        return JsonResponse(
            {
                "run_key": run_key,
                "progress_url": progress_url,
                "results_url": results_url,
            }
        )


class PipelineProgressView(View):
    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        status = _get_run_status(run_key)
        if status is None:
            raise Http404("Run not found.")
        payload = {
            "run_key": run_key,
            "progress": float(status.get("progress", 0.0)),
            "stage": status.get("stage", "Running"),
            "done": bool(status.get("done", False)),
            "error": status.get("error"),
            "results_url": status.get("results_url"),
        }
        if not payload["done"]:
            started_at = status.get("started_at")
            if isinstance(started_at, (int, float)) and payload["progress"] < PROGRESS_INIT_CAP:
                elapsed = max(time.monotonic() - started_at, 0.0)
                warm_progress = min(PROGRESS_INIT_CAP, elapsed * 0.4)
                if warm_progress > payload["progress"]:
                    payload["progress"] = warm_progress
                    payload["stage"] = payload["stage"] or "Extracting landmarks"
                    _set_run_status(
                        run_key,
                        progress=payload["progress"],
                        stage=payload["stage"],
                        last_update=time.monotonic(),
                    )
        return JsonResponse(payload)


class ResultsView(View):
    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        run_dir = _resolve_output_dir(run_key)
        if not run_dir.exists():
            raise Http404("Run not found.")

        files = []
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "name": path.name,
                        "relative_path": path.relative_to(run_dir).as_posix(),
                        "size_kb": max(path.stat().st_size // 1024, 1),
                    }
                )

        return render(
            request,
            "results.html",
            {
                "run_key": run_key,
                "files": files,
            },
        )


class StatisticsView(View):
    def get(self, request: HttpRequest, run_key: str) -> HttpResponse:
        run_dir = _resolve_output_dir(run_key)
        if not run_dir.exists():
            raise Http404("Run not found.")

        series: dict[str, dict[str, list[float] | list[str]]] = {}
        joint_options: list[dict[str, str]] = []

        angle_files = sorted(run_dir.rglob("*_angles_*.csv"))
        joint_files: dict[str, Path] = {}
        for csv_path in angle_files:
            stem = csv_path.stem
            if "_angles_" not in stem:
                continue
            joint_name = stem.split("_angles_", 1)[1]
            if joint_name in {"R", "L"}:
                continue
            joint_files[joint_name] = csv_path

        for joint_name, csv_path in joint_files.items():
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                fieldnames = list(reader.fieldnames)
                time_key = (
                    "time_s"
                    if "time_s" in fieldnames
                    else "time"
                    if "time" in fieldnames
                    else None
                )
                if time_key is None:
                    continue

                angle_columns = [name for name in fieldnames if name != time_key]
                if not angle_columns:
                    continue
                angle_columns = angle_columns[:3]

                entry = {
                    "t": [],
                    "x": [],
                    "y": [],
                    "z": [],
                    "labels": [
                        name.replace("_deg", "").replace("_", " ").title()
                        for name in angle_columns
                    ],
                }
                for row in reader:
                    try:
                        t_value = float(row[time_key])
                    except (TypeError, ValueError):
                        continue
                    entry["t"].append(t_value)
                    values: list[float | None] = []
                    for col in angle_columns:
                        try:
                            value = float(row[col])
                            values.append(value)
                        except (TypeError, ValueError):
                            values.append(None)
                    while len(values) < 3:
                        values.append(None)
                    entry["x"].append(values[0])
                    entry["y"].append(values[1])
                    entry["z"].append(values[2])

                key = f"joint:{joint_name}"
                series[key] = entry
                joint_options.append(
                    {
                        "value": key,
                        "label": f"Joint — {joint_name.replace('_', ' ').title()}",
                    }
                )

        landmark_csv = next(run_dir.rglob("*_raw_landmarks.csv"), None)
        if landmark_csv is None:
            landmark_csv = next(run_dir.rglob("*.csv"), None)
        if landmark_csv and landmark_csv.exists():
            landmark_series: dict[str, dict[str, list[float]]] = {}
            with landmark_csv.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames and {
                    "timestamp_s",
                    "landmark",
                    "x_m",
                    "y_m",
                    "z_m",
                }.issubset(reader.fieldnames):
                    for row in reader:
                        marker = row.get("landmark")
                        if not marker:
                            continue
                        entry = landmark_series.setdefault(
                            marker,
                            {"t": [], "x": [], "y": [], "z": []},
                        )
                        try:
                            entry["t"].append(float(row["timestamp_s"]))
                            entry["x"].append(float(row["x_m"]))
                            entry["y"].append(float(row["y_m"]))
                            entry["z"].append(float(row["z_m"]))
                        except (TypeError, ValueError):
                            continue

            for marker, entry in landmark_series.items():
                if not entry["t"]:
                    continue
                key = f"marker:{marker}"
                entry["labels"] = ["X", "Y", "Z"]
                series[key] = entry

        if not series:
            return render(
                request,
                "statistics.html",
                {
                    "run_key": run_key,
                    "markers": [],
                    "series": {},
                    "error": "No joint angles or landmark CSVs found for this run.",
                },
            )

        markers = sorted(joint_options, key=lambda item: item["label"])
        preview_video = None
        preview_video_type = None
        preview_candidates = sorted(run_dir.rglob("*_preview.*"))
        for candidate in preview_candidates:
            if candidate.suffix.lower() in {".mp4", ".webm"}:
                preview_video = candidate
                preview_video_type = mimetypes.guess_type(candidate.name)[0]
                break

        upload_video = None
        upload_video_type = None
        safe_run_id = Path(run_key).name
        upload_dir = UPLOAD_ROOT / safe_run_id
        if upload_dir.exists():
            for candidate in sorted(upload_dir.iterdir()):
                if candidate.is_file() and candidate.suffix.lower() in {
                    ".mp4",
                    ".mov",
                    ".webm",
                    ".m4v",
                }:
                    upload_video = candidate
                    upload_video_type = mimetypes.guess_type(candidate.name)[0]
                    break

        video_path = None
        video_type = None
        video_route = None
        video_label = None
        if preview_video:
            video_path = preview_video.relative_to(run_dir).as_posix()
            video_type = preview_video_type
            video_route = "media"
            video_label = "Overlay preview synced to the time slider."
        elif upload_video:
            video_path = upload_video.relative_to(upload_dir).as_posix()
            video_type = upload_video_type
            video_route = "upload_media"
            video_label = "Uploaded video synced to the time slider."
        return render(
            request,
            "statistics.html",
            {
                "run_key": run_key,
                "markers": markers,
                "series": series,
                "source_csv": "joint angle CSVs + raw landmarks",
                "video_path": video_path,
                "video_type": video_type,
                "video_route": video_route,
                "video_label": video_label,
            },
        )


class DownloadView(View):
    def get(self, request: HttpRequest, run_key: str, file_path: str) -> HttpResponse:
        run_dir = _resolve_output_dir(run_key)
        target = (run_dir / file_path).resolve()
        if run_dir not in target.parents and target != run_dir:
            raise Http404("Invalid file path.")
        if not target.exists() or not target.is_file():
            raise Http404("File not found.")

        response = FileResponse(
            target.open("rb"), as_attachment=True, filename=target.name
        )
        return response


class MediaView(View):
    def get(self, request: HttpRequest, run_key: str, file_path: str) -> HttpResponse:
        run_dir = _resolve_output_dir(run_key)
        target = (run_dir / file_path).resolve()
        if run_dir not in target.parents and target != run_dir:
            raise Http404("Invalid file path.")
        if not target.exists() or not target.is_file():
            raise Http404("File not found.")

        content_type, _ = mimetypes.guess_type(target.name)
        response = FileResponse(
            target.open("rb"),
            as_attachment=False,
            content_type=content_type or "application/octet-stream",
        )
        return response


class UploadMediaView(View):
    def get(self, request: HttpRequest, run_key: str, file_path: str) -> HttpResponse:
        safe_run_id = Path(run_key).name
        upload_dir = (UPLOAD_ROOT / safe_run_id).resolve()
        target = (upload_dir / file_path).resolve()
        if upload_dir not in target.parents and target != upload_dir:
            raise Http404("Invalid file path.")
        if not target.exists() or not target.is_file():
            raise Http404("File not found.")

        content_type, _ = mimetypes.guess_type(target.name)
        response = FileResponse(
            target.open("rb"),
            as_attachment=False,
            content_type=content_type or "application/octet-stream",
        )
        return response
