"""Service for recomputing joint angles with DOF filtering."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from src.application.dto.dof_config import DofConfig
from src.kinematics.comprehensive_joint_angles import compute_all_joint_angles

ZERO_MODE = "first_n_seconds"
ZERO_WINDOW_S = 0.01


class JointAngleRecomputeService:
    """Recompute joint angles with configurable DOF filtering."""

    def recompute(
        self,
        run_dir: Path,
        dof_config: DofConfig,
    ) -> dict[str, dict[str, list[float] | list[str]]]:
        """Recompute joint angles and filter to specified DOF.

        Args:
            run_dir: Path to run output directory
            dof_config: Configuration specifying which DOF to include

        Returns:
            Series dict matching StatisticsService format for chart rendering

        Raises:
            FileNotFoundError: If no augmented TRC file found
        """
        trc_path = self._find_augmented_trc(run_dir)
        if trc_path is None:
            raise FileNotFoundError(
                f"No augmented TRC file found in {run_dir}"
            )

        joint_angles = compute_all_joint_angles(
            trc_path,
            smooth_window=9,
            unwrap=True,
            zero_mode="first_n_seconds",
            zero_window_s=0.5,
            verbose=False,
        )

        joint_angles_dir = self._ensure_joint_angles_dir(run_dir, trc_path)
        series: dict[str, dict[str, list[float] | list[str]]] = {}

        for joint_name, df in joint_angles.items():
            enabled_suffixes = dof_config.get_column_filter(joint_name)
            if not enabled_suffixes:
                continue

            time_col = "time_s"
            base_name = joint_name.lower().replace("_", "")
            if joint_name.endswith("_R") or joint_name.endswith("_L"):
                base_name = joint_name[:-2].lower()

            columns_to_keep = [time_col]
            for suffix in enabled_suffixes:
                for col in df.columns:
                    if col.endswith(suffix) and col != time_col:
                        columns_to_keep.append(col)
                        break

            filtered_df = df[columns_to_keep].copy()
            csv_path = joint_angles_dir / f"{trc_path.stem}_angles_{joint_name}.csv"
            filtered_df.to_csv(csv_path, index=False)

            series_key = f"joint:{joint_name}"
            entry = self._build_series_entry(filtered_df, time_col)
            if entry:
                series[series_key] = entry

        return series

    def _find_augmented_trc(self, run_dir: Path) -> Path | None:
        """Find the augmented TRC file in priority order."""
        final_files = list(run_dir.rglob("*_final.trc"))
        if final_files:
            return max(final_files, key=lambda p: p.stat().st_mtime)

        lstm_files = list(run_dir.rglob("*_LSTM*.trc"))
        if lstm_files:
            return max(lstm_files, key=lambda p: p.stat().st_mtime)

        all_trc = list(run_dir.rglob("*.trc"))
        if all_trc:
            return max(all_trc, key=lambda p: p.stat().st_mtime)

        return None

    def _ensure_joint_angles_dir(self, run_dir: Path, trc_path: Path) -> Path:
        """Create joint_angles directory if needed."""
        joint_angles_dir = trc_path.parent / "joint_angles"
        joint_angles_dir.mkdir(parents=True, exist_ok=True)
        return joint_angles_dir

    def _build_series_entry(
        self,
        df,
        time_col: str,
    ) -> dict[str, list[float] | list[str]] | None:
        """Build series entry in StatisticsService format."""
        if df.empty:
            return None

        angle_columns = [c for c in df.columns if c != time_col]
        if not angle_columns:
            return None

        angle_columns = angle_columns[:3]

        entry: dict[str, list[float] | list[str]] = {
            "t": df[time_col].tolist(),
            "x": [],
            "y": [],
            "z": [],
            "labels": [
                name.replace("_deg", "").replace("_", " ").title()
                for name in angle_columns
            ],
        }

        for i, col in enumerate(angle_columns):
            axis = ["x", "y", "z"][i]
            entry[axis] = df[col].tolist()

        while len(entry["labels"]) < 3:
            entry["labels"].append("")
        for axis in ["x", "y", "z"]:
            if not entry[axis]:
                entry[axis] = [None] * len(entry["t"])

        entry = self._zero_series(entry)
        return entry

    def _zero_series(
        self, entry: dict[str, list[float] | list[str]]
    ) -> dict[str, list[float] | list[str]]:
        """Apply zero-point baseline correction."""
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
        """Calculate zero offset based on first N seconds."""
        if not times or not values:
            return None
        tmax = times[0] + ZERO_WINDOW_S
        sample = [
            value
            for time_value, value in zip(times, values)
            if time_value <= tmax and self._is_finite_number(value)
        ]
        if not sample:
            return None
        return sum(sample) / len(sample)

    @staticmethod
    def _is_finite_number(value: float | None) -> bool:
        """Check if value is a finite number."""
        return value is not None and math.isfinite(value)
