from __future__ import annotations

"""Service for preparing TRC-based 3D plot data for the statistics view."""

from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class TrcPlotPayload:
    """Container for TRC-derived 3D plot data."""

    markers: list[str]
    frames: list[list[list[float | None]]]
    times: list[float | None]
    connections: list[list[int]]
    bounds: dict[str, list[float]]


class TrcPlotService:
    """Parse TRC files and build an embeddable 3D plot payload."""

    _MAX_FRAMES = 300

    _MARKER_ALIASES = {
        "rbigtoe": ["rbigtoe", "rtoestudy", "r_toe_study", "rtoe", "r_bigtoe"],
        "lbigtoe": ["lbigtoe", "ltoestudy", "l_toe_study", "ltoe", "l_bigtoe"],
        "rankle": ["rankle", "r_ankle_study", "ranklestudy", "r_mankle_study"],
        "lankle": ["lankle", "l_ankle_study", "lanklestudy", "l_mankle_study"],
        "rknee": ["rknee", "r_knee_study", "rkneestudy", "r_mknee_study"],
        "lknee": ["lknee", "l_knee_study", "lkneestudy", "l_mknee_study"],
        "rhip": ["rhip", "r_asis_study", "r.asis_study", "rhjc_study"],
        "lhip": ["lhip", "l_asis_study", "l.asis_study", "lhjc_study"],
        "hipcenter": ["hipcenter", "pelvis", "pelvis_study", "sacrum"],
        "neck": ["neck", "c7", "c7_study"],
        "rshoulder": [
            "rshoulder",
            "r_shoulder_study",
            "r_sh1_study",
            "r_sh2_study",
            "r_sh3_study",
        ],
        "lshoulder": [
            "lshoulder",
            "l_shoulder_study",
            "l_sh1_study",
            "l_sh2_study",
            "l_sh3_study",
        ],
        "relbow": ["relbow", "r_lelbow_study", "r_melbow_study"],
        "lelbow": ["lelbow", "l_lelbow_study", "l_melbow_study"],
        "rwrist": ["rwrist", "r_lwrist_study", "r_mwrist_study"],
        "lwrist": ["lwrist", "l_lwrist_study", "l_mwrist_study"],
    }

    _CONNECTIONS = [
        ("rbigtoe", "rankle"),
        ("rankle", "rknee"),
        ("rknee", "rhip"),
        ("lbigtoe", "lankle"),
        ("lankle", "lknee"),
        ("lknee", "lhip"),
        ("rhip", "lhip"),
        ("hipcenter", "neck"),
        ("rshoulder", "lshoulder"),
        ("rshoulder", "relbow"),
        ("relbow", "rwrist"),
        ("lshoulder", "lelbow"),
        ("lelbow", "lwrist"),
    ]
    _FALLBACK_CONNECTION_INDICES = [
        (13, 7),
        (7, 5),
        (5, 3),
        (14, 8),
        (8, 6),
        (6, 4),
        (3, 4),
        (19, 0),
        (1, 2),
        (1, 15),
        (15, 17),
        (2, 16),
        (16, 18),
    ]

    def build_plot_payload(self, run_dir: Path) -> TrcPlotPayload | None:
        """Locate a TRC file and assemble the plot payload."""
        trc_path = self._find_trc_path(run_dir)
        if trc_path is None:
            return None

        markers, times, frames = self._load_trc_frames(trc_path)
        if not frames:
            return None

        frames, times = self._downsample_frames(frames, times)
        connections = self._resolve_connections(markers)
        bounds = self._compute_bounds(frames)
        return TrcPlotPayload(
            markers=markers,
            frames=frames,
            times=times,
            connections=connections,
            bounds=bounds,
        )

    def _find_trc_path(self, run_dir: Path) -> Path | None:
        """Find the most relevant TRC file within the run directory."""
        candidates = sorted(run_dir.rglob("*.trc"))
        if not candidates:
            return None

        def rank(candidate: Path) -> int:
            name = candidate.name.lower()
            if name.endswith("_final.trc"):
                return 0
            if "_lstm" in name:
                return 1
            return 2

        return sorted(candidates, key=rank)[0]

    def _load_trc_frames(
        self, trc_path: Path
    ) -> tuple[list[str], list[float | None], list[list[list[float | None]]]]:
        """Parse marker names, timestamps, and coordinates from a TRC file."""
        lines = trc_path.read_text(encoding="utf-8").splitlines()
        name_line_idx = next(
            (idx for idx, line in enumerate(lines) if line.startswith("Frame#")), None
        )
        if name_line_idx is None:
            return [], [], []

        axis_line_idx = name_line_idx + 1
        data_start_idx = axis_line_idx + 1
        while data_start_idx < len(lines) and not lines[data_start_idx].strip():
            data_start_idx += 1

        name_tokens = lines[name_line_idx].split("\t")
        header_markers = []
        for idx in range(2, len(name_tokens), 3):
            label = name_tokens[idx].strip()
            if not label:
                label = f"Marker{len(header_markers) + 1}"
            header_markers.append(label)

        first_data_line = None
        for line in lines[data_start_idx:]:
            if line.strip():
                first_data_line = line.rstrip("\n").split("\t")
                break
        if first_data_line is None:
            return [], [], []

        data_columns = max(len(first_data_line) - 2, 0)
        actual_marker_count = data_columns // 3
        marker_names = header_markers[:]
        while len(marker_names) < actual_marker_count:
            marker_names.append(f"AugMarker{len(marker_names) + 1}")

        frames: list[list[list[float | None]]] = []
        times: list[float | None] = []
        for line in lines[data_start_idx:]:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue

            times.append(self._parse_float(parts[1]))
            data = parts[2:]
            frame: list[list[float | None]] = []
            for marker_idx in range(actual_marker_count):
                base = marker_idx * 3
                triple = data[base : base + 3] if base + 2 < len(data) else []
                frame.append(
                    [
                        self._parse_float(triple[0]) if len(triple) > 0 else None,
                        self._parse_float(triple[1]) if len(triple) > 1 else None,
                        self._parse_float(triple[2]) if len(triple) > 2 else None,
                    ]
                )
            frames.append(frame)

        return marker_names, times, frames

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

    def _resolve_connections(self, markers: list[str]) -> list[list[int]]:
        """Match known skeleton connections against available marker names."""
        normalized_map = {
            self._normalize_marker_name(name): idx for idx, name in enumerate(markers)
        }
        index_map: dict[str, int] = {}
        for canonical, aliases in self._MARKER_ALIASES.items():
            for alias in aliases:
                alias_key = self._normalize_marker_name(alias)
                if alias_key in normalized_map:
                    index_map[canonical] = normalized_map[alias_key]
                    break

        resolved: list[list[int]] = []
        for start, end in self._CONNECTIONS:
            start_idx = index_map.get(start)
            end_idx = index_map.get(end)
            if start_idx is None or end_idx is None:
                continue
            resolved.append([start_idx, end_idx])
        if resolved:
            return resolved
        return self._fallback_connections(len(markers))

    def _fallback_connections(self, marker_count: int) -> list[list[int]]:
        """Provide index-based connections when marker names are unavailable."""
        if marker_count < 20:
            return []
        resolved: list[list[int]] = []
        for start, end in self._FALLBACK_CONNECTION_INDICES:
            if start < marker_count and end < marker_count:
                resolved.append([start, end])
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
    def _normalize_marker_name(name: str) -> str:
        """Normalize marker names for fuzzy matching."""
        return "".join(char for char in name.lower() if char.isalnum())

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
