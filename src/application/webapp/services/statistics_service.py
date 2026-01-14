from __future__ import annotations

"""Service that assembles statistics context from run outputs."""

import csv
import mimetypes
from pathlib import Path

from src.application.webapp.services.trc_plot_service import TrcPlotService


class StatisticsService:
    """Prepare joint angle and landmark series for the statistics view."""

    def __init__(self, upload_root: Path) -> None:
        self._upload_root = upload_root
        self._trc_plot_service = TrcPlotService()

    def build_context(self, run_dir: Path, run_key: str) -> dict[str, object]:
        """Build the template context for statistics rendering."""
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

                entry: dict[str, list[float] | list[str]] = {
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
            return {
                "run_key": run_key,
                "markers": [],
                "series": {},
                "error": "No joint angles or landmark CSVs found for this run.",
            }

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
        upload_dir = self._upload_root / safe_run_id
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
        if preview_video:
            video_path = preview_video.relative_to(run_dir).as_posix()
            video_type = preview_video_type
            video_route = "media"
        elif upload_video:
            video_path = upload_video.relative_to(upload_dir).as_posix()
            video_type = upload_video_type
            video_route = "upload_media"

        plot_payload = self._trc_plot_service.build_plot_payload(run_dir)
        plot_data = plot_payload.__dict__ if plot_payload else None

        return {
            "run_key": run_key,
            "markers": markers,
            "series": series,
            "source_csv": "joint angle CSVs + raw landmarks",
            "video_path": video_path,
            "video_type": video_type,
            "video_route": video_route,
            "plot_data": plot_data,
        }
