"""Service for preparing landmark-based skeleton plot data."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class LandmarkPlotPayload:
    """Container for landmark-derived 3D plot data."""

    markers: list[str]
    frames: list[list[list[float | None]]]
    times: list[float | None]
    connections: list[list[int]]
    bounds: dict[str, list[float]]


class LandmarkPlotService:
    """Parse raw landmark CSVs into a 3D plot payload."""

    _MAX_FRAMES = 300
    _MARKER_ORDER = [
        "Nose",
        "LEyeInner",
        "LEye",
        "LEyeOuter",
        "REyeInner",
        "REye",
        "REyeOuter",
        "LEar",
        "REar",
        "MouthLeft",
        "MouthRight",
        "LShoulder",
        "RShoulder",
        "LElbow",
        "RElbow",
        "LWrist",
        "RWrist",
        "LPinky",
        "RPinky",
        "LIndex",
        "RIndex",
        "LThumb",
        "RThumb",
        "LHip",
        "RHip",
        "LKnee",
        "RKnee",
        "LAnkle",
        "RAnkle",
        "LHeel",
        "RHeel",
        "LBigToe",
        "RBigToe",
    ]
    _CONNECTIONS = [
        (8, 6),
        (6, 5),
        (5, 4),
        (4, 0),
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (12, 11),
        (11, 23),
        (23, 24),
        (24, 12),
        (12, 14),
        (14, 16),
        (16, 18),
        (18, 20),
        (20, 16),
        (16, 22),
        (11, 13),
        (13, 15),
        (15, 17),
        (17, 19),
        (19, 15),
        (15, 21),
        (24, 26),
        (26, 28),
        (28, 30),
        (30, 32),
        (32, 28),
        (23, 25),
        (25, 27),
        (27, 29),
        (29, 31),
        (31, 27),
    ]

    def build_plot_payload(self, run_dir: Path) -> LandmarkPlotPayload | None:
        """Locate a raw landmark CSV and assemble the plot payload."""
        csv_path = self._find_landmark_csv(run_dir)
        if csv_path is None:
            return None

        frames, times, markers = self._load_landmarks(csv_path)
        if not frames:
            return None

        frames, times = self._downsample_frames(frames, times)
        connections = self._resolve_connections(len(markers))
        bounds = self._compute_bounds(frames)
        return LandmarkPlotPayload(
            markers=markers,
            frames=frames,
            times=times,
            connections=connections,
            bounds=bounds,
        )

    def _find_landmark_csv(self, run_dir: Path) -> Path | None:
        """Find a raw landmark CSV output for the run."""
        candidates = sorted(run_dir.rglob("*_raw_landmarks.csv"))
        if candidates:
            return candidates[0]
        return None

    def _load_landmarks(
        self, csv_path: Path
    ) -> tuple[list[list[list[float | None]]], list[float | None], list[str]]:
        """Load landmarks into per-frame coordinate arrays."""
        rows = csv_path.read_text(encoding="utf-8").splitlines()
        if not rows:
            return [], [], []

        header = rows[0].split(",")
        try:
            timestamp_idx = header.index("timestamp_s")
            landmark_idx = header.index("landmark")
            x_idx = header.index("x_m")
            y_idx = header.index("y_m")
            z_idx = header.index("z_m")
        except ValueError:
            return [], [], []

        frames: dict[float, dict[str, list[float | None]]] = {}
        for line in rows[1:]:
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) <= max(timestamp_idx, landmark_idx, x_idx, y_idx, z_idx):
                continue
            timestamp = self._parse_float(parts[timestamp_idx])
            if timestamp is None:
                continue
            landmark = parts[landmark_idx].strip()
            if not landmark:
                continue
            coords = [
                self._parse_float(parts[x_idx]),
                self._parse_float(parts[y_idx]),
                self._parse_float(parts[z_idx]),
            ]
            frame = frames.setdefault(timestamp, {})
            frame[landmark] = coords

        if not frames:
            return [], [], []

        available_markers = {name for frame in frames.values() for name in frame.keys()}
        if not any(name in available_markers for name in self._MARKER_ORDER):
            return [], [], []
        markers = list(self._MARKER_ORDER)

        times = sorted(frames.keys())
        frame_list: list[list[list[float | None]]] = []
        time_list: list[float | None] = []
        for timestamp in times:
            row = frames[timestamp]
            frame_list.append(
                [row.get(marker, [None, None, None]) for marker in markers]
            )
            time_list.append(timestamp)

        filled_frames = self._fill_missing_markers(frame_list)
        return filled_frames, time_list, markers

    def _downsample_frames(
        self,
        frames: list[list[list[float | None]]],
        times: list[float | None],
    ) -> tuple[list[list[list[float | None]]], list[float | None]]:
        """Reduce frame count to a manageable size for the UI."""
        if len(frames) <= self._MAX_FRAMES:
            return frames, times
        stride = max(math.ceil(len(frames) / self._MAX_FRAMES), 1)
        return frames[::stride], times[::stride]

    def _resolve_connections(self, marker_count: int) -> list[list[int]]:
        """Resolve skeleton connections based on MediaPipe indices."""
        resolved: list[list[int]] = []
        for start_idx, end_idx in self._CONNECTIONS:
            if start_idx < marker_count and end_idx < marker_count:
                resolved.append([start_idx, end_idx])
        return resolved

    def _compute_bounds(
        self, frames: list[list[list[float | None]]]
    ) -> dict[str, list[float]]:
        """Compute axis bounds across frames for stable plotting."""
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for frame in frames:
            for point in frame:
                if len(point) != 3:
                    continue
                x, y, z = point
                if self._is_valid_number(x):
                    xs.append(float(x))
                if self._is_valid_number(y):
                    ys.append(float(y))
                if self._is_valid_number(z):
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
    def _fill_missing_markers(
        frames: list[list[list[float | None]]],
    ) -> list[list[list[float | None]]]:
        """Forward/backward fill missing marker positions."""
        if not frames:
            return frames
        marker_count = len(frames[0])
        filled = [list(frame) for frame in frames]

        last_seen: list[list[float | None] | None] = [None] * marker_count
        for frame in filled:
            for idx, point in enumerate(frame):
                if point and all(value is not None for value in point):
                    last_seen[idx] = point
                elif last_seen[idx] is not None:
                    frame[idx] = list(last_seen[idx])  # copy to avoid aliasing

        next_seen: list[list[float | None] | None] = [None] * marker_count
        for frame in reversed(filled):
            for idx, point in enumerate(frame):
                if point and all(value is not None for value in point):
                    next_seen[idx] = point
                elif next_seen[idx] is not None:
                    frame[idx] = list(next_seen[idx])
        return filled

    @staticmethod
    def _parse_float(value: str | None) -> float | None:
        """Parse a float value or return None when invalid."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _is_valid_number(value: float | None) -> bool:
        """Check if a value is a finite float."""
        if value is None:
            return False
        return math.isfinite(value)
