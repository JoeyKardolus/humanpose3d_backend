import csv
import mimetypes
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Mapping

from django.http import FileResponse, Http404, HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views import View


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = (REPO_ROOT / "data" / "output" / "pose-3d").resolve()
UPLOAD_ROOT = (REPO_ROOT / "data" / "input" / "webapp").resolve()


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


def _build_pipeline_command(
    upload_path: Path, form_data: Mapping[str, str], sex_raw: str
) -> list[str]:
    main_path = REPO_ROOT / "main.py"
    command = [
        sys.executable,
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


class HomeView(View):
    def get(self, request: HttpRequest) -> HttpResponse:
        return render(request, "home.html")

    def post(self, request: HttpRequest) -> HttpResponse:
        errors: list[str] = []
        uploaded = request.FILES.get("video")
        if not uploaded:
            errors.append("Please upload a video file before submitting.")

        if request.POST.get("consent") != "accepted":
            errors.append(
                "You must confirm participant consent before running an analysis."
            )

        output_location = _safe_relative_path(
            request.POST.get("output_location", "").strip()
        )
        if request.POST.get("output_location") and not output_location:
            errors.append("Output location must be a relative path (no .. segments).")

        sex_raw = request.POST.get("sex", "").strip().lower()
        if sex_raw and sex_raw not in {"male", "female"}:
            errors.append("Sex must be Male or Female for the current pipeline.")

        if errors:
            return render(
                request,
                "home.html",
                {"errors": errors},
            )

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
            return render(
                request,
                "home.html",
                {
                    "errors": [
                        "Output location already exists. Choose a different output folder.",
                        f"Output path: {output_dir.relative_to(REPO_ROOT)}",
                    ]
                },
            )
        command = _build_pipeline_command(upload_path, request.POST, sex_raw)

        try:
            completed = subprocess.run(
                command,
                check=True,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            log_path = pipeline_run_dir / "pipeline_error.log"
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            _write_pipeline_log(log_path, stdout, stderr)
            tail_lines = (stderr or stdout).splitlines()[-12:]
            tail_text = "\n".join(tail_lines)
            return render(
                request,
                "home.html",
                {
                    "errors": [
                        "Pipeline failed to run. Check the server logs for details.",
                        f"Exit code: {exc.returncode}",
                        f"Log file: {log_path.relative_to(REPO_ROOT)}",
                        f"Last output:\n{tail_text}"
                        if tail_text
                        else "No output captured.",
                    ]
                },
            )
        else:
            log_path = pipeline_run_dir / "pipeline.log"
            _write_pipeline_log(
                log_path,
                completed.stdout or "",
                completed.stderr or "",
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
