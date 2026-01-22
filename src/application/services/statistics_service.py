"""Service that assembles statistics context from run outputs."""

from __future__ import annotations

import csv
import json
import math
import mimetypes
import shutil
import subprocess
from pathlib import Path

from src.application.services.landmark_plot_service import LandmarkPlotService
from src.application.services.trc_plot_service import TrcPlotService

ZERO_MODE = "first_n_seconds"
ZERO_WINDOW_S = 0.01


class StatisticsService:
    """Prepare joint angle and landmark series for the statistics view."""

    def __init__(self) -> None:
        self._trc_plot_service = TrcPlotService()
        self._landmark_plot_service = LandmarkPlotService()

    def build_context(self, run_dir: Path, run_key: str) -> dict[str, object]:
        """Build the template context for statistics rendering."""
        series: dict[str, dict[str, list[float] | list[str]]] = {}
        joint_options: list[dict[str, str]] = []

        angle_files = sorted(run_dir.rglob("*_angles_*.csv"))
        source_csv: str | None = None
        joint_files: dict[str, Path] = {}
        for csv_path in angle_files:
            stem = csv_path.stem
            if "_angles_" not in stem:
                continue
            joint_name = stem.split("_angles_", 1)[1]
            if joint_name in {"R", "L"}:
                continue
            joint_files[joint_name] = csv_path
            if source_csv is None:
                source_csv = csv_path.name

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

                entry = self._zero_series(entry)
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
            if source_csv is None:
                source_csv = landmark_csv.name
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
                "source_csv": source_csv or "",
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

        video_path = None
        video_type = None
        video_route = None
        if preview_video:
            video_path = preview_video.relative_to(run_dir).as_posix()
            video_type = preview_video_type
            video_route = "media"
        elif source_video:
            video_path = source_video.relative_to(run_dir).as_posix()
            video_type = source_video_type
            video_route = "media"

        rotation_degrees = self._resolve_preview_rotation(preview_video)
        mirror_depth = rotation_degrees is not None and rotation_degrees != 0
        mirror_y = rotation_degrees is not None and rotation_degrees != 0
        if rotation_degrees is None:
            rotation_degrees, _ = self._resolve_rotation(source_video, run_dir)
            mirror_depth = False
            mirror_y = False
        skeleton_payload = self._landmark_plot_service.build_plot_payload(run_dir)
        augmented_payload = self._trc_plot_service.build_plot_payload(run_dir)
        if rotation_degrees:
            if skeleton_payload is not None:
                skeleton_payload = self._rotate_plot_payload(
                    skeleton_payload, rotation_degrees, mirror_depth, mirror_y
                )
            if augmented_payload is not None:
                augmented_payload = self._rotate_plot_payload(
                    augmented_payload, rotation_degrees, mirror_depth, mirror_y
                )
        plot_skeleton_data = skeleton_payload.__dict__ if skeleton_payload else None
        plot_augmented_data = augmented_payload.__dict__ if augmented_payload else None

        return {
            "run_key": run_key,
            "markers": markers,
            "series": series,
            "source_csv": source_csv or "",
            "video_path": video_path,
            "video_type": video_type,
            "video_route": video_route,
            "plot_skeleton_data": plot_skeleton_data,
            "plot_augmented_data": plot_augmented_data,
        }

    def _zero_series(
        self, entry: dict[str, list[float] | list[str]]
    ) -> dict[str, list[float] | list[str]]:
        times = entry.get("t")
        if not isinstance(times, list) or not times:
            return entry
        for axis in ("x", "y", "z"):
            values = entry.get(axis)
            if not isinstance(values, list) or not values:
                continue
            offset = self._resolve_zero_offset(times, values)
            if offset is None:
                continue
            entry[axis] = [
                value - offset if self._is_finite_number(value) else value
                for value in values
            ]
        return entry

    def _resolve_zero_offset(
        self, times: list[float], values: list[float | None]
    ) -> float | None:
        if ZERO_MODE == "global_mean":
            sample = [value for value in values if self._is_finite_number(value)]
        elif ZERO_MODE == "first_n_seconds":
            tmax = times[0] + ZERO_WINDOW_S
            sample = [
                value
                for time_value, value in zip(times, values)
                if time_value <= tmax and self._is_finite_number(value)
            ]
        elif ZERO_MODE == "first_frame":
            first_value = values[0] if values else None
            sample = [first_value] if self._is_finite_number(first_value) else []
        else:
            sample = []
        if not sample:
            return None
        return sum(sample) / len(sample)

    @staticmethod
    def _is_finite_number(value: float | None) -> bool:
        return value is not None and math.isfinite(value)

    def _resolve_rotation(
        self, source_video: Path | None, run_dir: Path
    ) -> tuple[int, bool]:
        metadata_path = run_dir / "source" / "video_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return 0, True
            rotation = metadata.get("rotation_degrees")
            rotation_applied = metadata.get("rotation_applied")
            if isinstance(rotation, int) and rotation in {0, 90, 180, 270}:
                return rotation, bool(rotation_applied)
            return 0, True

        if source_video is None:
            return 0, True
        rotation = self._probe_video_rotation(source_video)
        if rotation == 0:
            return 0, True
        return rotation, False

    @staticmethod
    def _probe_video_rotation(video_path: Path) -> int:
        """Return rotation in degrees (0/90/180/270) if metadata is available."""
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return 0
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream_tags=rotate",
            "-of",
            "default=nk=1:nw=1",
            str(video_path),
        ]
        try:
            output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return 0
        try:
            rotation = int(output.decode("utf-8").strip())
        except ValueError:
            return 0
        rotation = rotation % 360
        if rotation in {0, 90, 180, 270}:
            return rotation
        return 0

    @staticmethod
    def _resolve_preview_rotation(preview_video: Path | None) -> int | None:
        if preview_video is None:
            return None
        metadata_path = preview_video.with_suffix(".json")
        if not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        rotation = metadata.get("rotation_degrees")
        if isinstance(rotation, int) and rotation in {0, 90, 180, 270}:
            return rotation
        return None

    def _rotate_plot_payload(
        self, payload, rotation: int, mirror_depth: bool, mirror_y: bool
    ):
        rotated_frames = [
            [
                self._rotate_point(point, rotation, mirror_depth, mirror_y)
                for point in frame
            ]
            for frame in payload.frames
        ]
        bounds = self._compute_bounds(rotated_frames)
        return payload.__class__(
            markers=payload.markers,
            frames=rotated_frames,
            times=payload.times,
            connections=payload.connections,
            bounds=bounds,
        )

    @staticmethod
    def _rotate_point(
        point: list[float | None],
        rotation: int,
        mirror_depth: bool,
        mirror_y: bool,
    ) -> list[float | None]:
        if len(point) != 3:
            return point
        x, y, z = point
        if x is None or y is None:
            return [x, y, z]
        if mirror_y:
            y = -y
        if rotation == 90:
            x, y = -y, x
        if rotation == 180:
            x, y = -x, -y
        if rotation == 270:
            x, y = y, -x
        if mirror_depth and z is not None:
            z = -z
        return [x, y, z]

    @staticmethod
    def _compute_bounds(
        frames: list[list[list[float | None]]],
    ) -> dict[str, list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for frame in frames:
            for point in frame:
                if len(point) != 3:
                    continue
                x, y, z = point
                if StatisticsService._is_valid_number(x):
                    xs.append(float(x))
                if StatisticsService._is_valid_number(y):
                    ys.append(float(y))
                if StatisticsService._is_valid_number(z):
                    zs.append(float(z))
        if not xs or not ys or not zs:
            return {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [-1.0, 1.0]}
        padding = 0.05
        return {
            "x": [min(xs) - padding, max(xs) + padding],
            "y": [min(ys) - padding, max(ys) + padding],
            "z": [min(zs) - padding, max(zs) + padding],
        }

    @staticmethod
    def _is_valid_number(value: float | None) -> bool:
        if value is None:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
