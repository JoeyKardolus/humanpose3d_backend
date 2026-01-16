from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ORDER_22: Tuple[str, ...] = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RHip",
    "LHip",
    "RKnee",
    "LKnee",
    "RAnkle",
    "LAnkle",
    "RHeel",
    "LHeel",
    "RSmallToe",
    "LSmallToe",
    "RBigToe",
    "LBigToe",
    "RElbow",
    "LElbow",
    "RWrist",
    "LWrist",
    "Hip",
    "Nose",
)  # 21 markers (Head removed - not used by pipeline)

DERIVED_PARENTS = {
    "Hip": ("LHip", "RHip"),
    "Neck": ("LShoulder", "RShoulder"),
}

CSV_HEADERS = ["timestamp_s", "landmark", "x_m", "y_m", "z_m", "visibility"]


@dataclass(frozen=True)
class LandmarkRecord:
    """Strict-mode snapshot of a single landmark measurement."""

    timestamp_s: float
    landmark: str
    x_m: float
    y_m: float
    z_m: float
    visibility: float

    def as_csv_row(self) -> List[str]:
        """Return the record as the canonical CSV row for deterministic exports."""
        return [
            f"{self.timestamp_s:.6f}",
            self.landmark,
            f"{self.x_m:.6f}",
            f"{self.y_m:.6f}",
            f"{self.z_m:.6f}",
            f"{self.visibility:.3f}",
        ]


def write_landmark_csv(output_path: Path, records: Sequence[LandmarkRecord]) -> int:
    """Write strict-mode CSV rows sorted by timestamp and marker name."""
    if not records:
        raise ValueError("No landmark records to write")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda r: (r.timestamp_s, r.landmark))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADERS)
        for record in sorted_records:
            writer.writerow(record.as_csv_row())
    return len(sorted_records)


def _load_frames(csv_path: Path) -> Dict[float, Dict[str, Tuple[float, float, float]]]:
    """Load CSV rows and group them per timestamp."""
    frames: Dict[float, Dict[str, Tuple[float, float, float]]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = float(row["timestamp_s"])
            coords = (
                float(row["x_m"]),
                float(row["y_m"]),
                float(row["z_m"]),
            )
            frames.setdefault(timestamp, {})[row["landmark"]] = coords
    return dict(sorted(frames.items()))


def _apply_derivatives(markers: Dict[str, Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
    """Insert derived markers using strict averaging rules."""
    derived = {}
    for name, (left, right) in DERIVED_PARENTS.items():
        if left in markers and right in markers:
            lx, ly, lz = markers[left]
            rx, ry, rz = markers[right]
            derived[name] = ((lx + rx) / 2, (ly + ry) / 2, (lz + rz) / 2)
    enriched = markers.copy()
    enriched.update(derived)
    return enriched


def _estimate_rate(timestamps: Sequence[float]) -> float:
    """Estimate frame rate from timestamps."""
    if len(timestamps) < 2:
        return 0.0
    deltas = [
        max(timestamps[idx + 1] - timestamps[idx], 1e-9)
        for idx in range(len(timestamps) - 1)
    ]
    median_delta = sorted(deltas)[len(deltas) // 2]
    return round(1.0 / median_delta, 2)


def csv_to_trc_strict(csv_path: Path, trc_path: Path, order: Sequence[str] = ORDER_22) -> Tuple[int, int]:
    """Convert strict CSV rows into an OpenSim-compatible TRC file."""
    frames = _load_frames(csv_path)
    timestamps = list(frames.keys())
    num_frames = len(timestamps)
    if num_frames == 0:
        raise ValueError(f"No frames parsed from {csv_path}")

    data_rate = _estimate_rate(timestamps)
    trc_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"PathFileType\t4\t(X/Y/Z)\t{trc_path.name}")
    lines.append(
        f"DataRate\t{data_rate:.2f}\tCameraRate\t{data_rate:.2f}\t"
        f"NumFrames\t{num_frames}\tNumMarkers\t{len(order)}\tUnits\tm"
    )
    lines.append("")  # compatibility spacer so Pose2Sim reads headers correctly
    name_tokens = ["Frame#", "Time"]
    for marker in order:
        name_tokens.extend([marker, marker, marker])
    lines.append("\t".join(name_tokens))

    axis_tokens = ["", ""]
    for _ in order:
        axis_tokens.extend(["X", "Y", "Z"])
    lines.append("\t".join(axis_tokens))

    for frame_idx, timestamp in enumerate(timestamps, start=1):
        markers = _apply_derivatives(frames[timestamp])
        row = [str(frame_idx), f"{timestamp:.6f}"]
        for marker in order:
            coords = markers.get(marker)
            if coords:
                row.extend([f"{coords[0]:.6f}", f"{coords[1]:.6f}", f"{coords[2]:.6f}"])
            else:
                row.extend(["", "", ""])
        lines.append("\t".join(row))

    trc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return num_frames, len(order)


def header_fix_strict(trc_path: Path) -> Path:
    """Rebuild header metadata so column counts reflect the strict-mode TRC payload."""
    lines = trc_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise ValueError(f"{trc_path} is too short to be a TRC file")

    data_lines = lines[5:]
    if not data_lines:
        raise ValueError("TRC file contains no data rows to inspect")

    first_row = data_lines[0].split()
    if len(first_row) < 2:
        raise ValueError("TRC data row missing frame/time columns")

    num_markers = max((len(first_row) - 2) // 3, 0)

    meta_parts = lines[1].split()
    for idx, token in enumerate(meta_parts):
        if token == "NumMarkers" and idx + 1 < len(meta_parts):
            meta_parts[idx + 1] = str(num_markers)
            break
    lines[1] = "\t".join(meta_parts)

    name_tokens = lines[3].split()
    base = name_tokens[:2]
    marker_labels: List[str] = []
    remainder = name_tokens[2:]
    for idx in range(0, len(remainder), 3):
        marker_labels.append(remainder[idx])

    if len(marker_labels) < num_markers:
        marker_labels.extend([""] * (num_markers - len(marker_labels)))
    else:
        marker_labels = marker_labels[:num_markers]

    rebuilt_names = base.copy()
    for label in marker_labels:
        rebuilt_names.extend([label, label, label])
    lines[3] = "\t".join(rebuilt_names)

    axis_tokens = ["", ""]
    for _ in range(num_markers):
        axis_tokens.extend(["X", "Y", "Z"])
    lines[4] = "\t".join(axis_tokens)

    fixed_path = trc_path.with_name(f"{trc_path.stem}_fixed.trc")
    fixed_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fixed_path
