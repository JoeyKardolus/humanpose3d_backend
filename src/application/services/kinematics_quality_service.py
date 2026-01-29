"""Service for analyzing kinematics quality from joint angle outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.application.dto.quality_metrics import JointRangeMetrics, QualityMetrics


class KinematicsQualityService:
    """Analyze joint angles for anatomical plausibility."""

    # Physiological joint angle limits in degrees
    # Based on ISB recommendations and clinical reference ranges
    PHYSIOLOGICAL_LIMITS: dict[str, tuple[float, float]] = {
        # Lower body
        "pelvis_flex": (-30, 45),
        "pelvis_abd": (-30, 30),
        "pelvis_rot": (-45, 45),
        "r_hip_flex": (-30, 125),
        "l_hip_flex": (-30, 125),
        "r_hip_abd": (-45, 50),
        "l_hip_abd": (-45, 50),
        "r_hip_rot": (-45, 45),
        "l_hip_rot": (-45, 45),
        "r_knee_flex": (-10, 160),
        "l_knee_flex": (-10, 160),
        "r_knee_abd": (-15, 15),
        "l_knee_abd": (-15, 15),
        "r_knee_rot": (-30, 30),
        "l_knee_rot": (-30, 30),
        "r_ankle_flex": (-50, 30),
        "l_ankle_flex": (-50, 30),
        "r_ankle_abd": (-25, 25),
        "l_ankle_abd": (-25, 25),
        "r_ankle_rot": (-20, 20),
        "l_ankle_rot": (-20, 20),
        # Upper body
        "trunk_flex": (-30, 90),
        "trunk_abd": (-45, 45),
        "trunk_rot": (-60, 60),
        "r_shoulder_flex": (-60, 180),
        "l_shoulder_flex": (-60, 180),
        "r_shoulder_abd": (-50, 180),
        "l_shoulder_abd": (-50, 180),
        "r_shoulder_rot": (-90, 90),
        "l_shoulder_rot": (-90, 90),
        "r_elbow_flex": (-10, 150),
        "l_elbow_flex": (-10, 150),
    }

    def analyze(self, joint_angles_dir: Path) -> QualityMetrics | None:
        """Analyze joint angles from a run's output directory.

        Args:
            joint_angles_dir: Path to the joint_angles/ subdirectory

        Returns:
            QualityMetrics if analysis successful, None otherwise
        """
        if not joint_angles_dir.exists():
            return None

        csv_files = list(joint_angles_dir.glob("*_angles_*.csv"))
        if not csv_files:
            return None

        joint_metrics: list[JointRangeMetrics] = []
        out_of_range_summary: dict[str, int] = {}
        total_frames = 0
        smoothness_values: list[float] = []

        for csv_file in csv_files:
            try:
                metrics = self._analyze_joint_csv(csv_file)
                if metrics:
                    joint_metrics.extend(metrics)
                    for m in metrics:
                        if m.out_of_range_frames > 0:
                            out_of_range_summary[m.joint_name] = m.out_of_range_frames
                        total_frames = max(total_frames, m.total_frames)
                        smoothness_values.append(m.within_limits_percent)
            except (OSError, ValueError):
                continue

        if not joint_metrics:
            return None

        # Calculate overall plausibility score
        if joint_metrics:
            total_within = sum(m.within_limits_percent for m in joint_metrics)
            plausibility_score = total_within / len(joint_metrics) / 100.0
        else:
            plausibility_score = 0.0

        # Calculate motion smoothness (average of per-joint smoothness)
        motion_smoothness = (
            sum(smoothness_values) / len(smoothness_values) / 100.0
            if smoothness_values
            else 0.0
        )

        return QualityMetrics(
            plausibility_score=round(plausibility_score, 3),
            total_frames=total_frames,
            analyzed_joints=len(joint_metrics),
            out_of_range_summary=out_of_range_summary,
            motion_smoothness=round(motion_smoothness, 3),
            bone_length_variance=0.0,  # TODO: Calculate from TRC if needed
            marker_dropout_rate=0.0,  # TODO: Calculate from raw landmarks
            joint_metrics=joint_metrics,
        )

    def _analyze_joint_csv(self, csv_path: Path) -> list[JointRangeMetrics]:
        """Analyze a single joint angle CSV file."""
        metrics: list[JointRangeMetrics] = []

        # Parse joint name from filename (e.g., joey_angles_r_knee.csv)
        stem = csv_path.stem
        parts = stem.split("_angles_")
        if len(parts) < 2:
            return metrics
        joint_base = parts[1]  # e.g., "r_knee" or "pelvis"

        # Read CSV data
        lines = csv_path.read_text().strip().split("\n")
        if len(lines) < 2:
            return metrics

        header = lines[0].split(",")
        data_rows = [line.split(",") for line in lines[1:] if line.strip()]
        if not data_rows:
            return metrics

        # Find angle columns (flex, abd, rot)
        angle_cols = {}
        for i, col in enumerate(header):
            col_lower = col.lower().strip()
            if "flex" in col_lower or "extension" in col_lower:
                angle_cols["flex"] = i
            elif "abd" in col_lower or "adduction" in col_lower:
                angle_cols["abd"] = i
            elif "rot" in col_lower or "rotation" in col_lower:
                angle_cols["rot"] = i

        if not angle_cols:
            return metrics

        total_frames = len(data_rows)

        for angle_type, col_idx in angle_cols.items():
            joint_name = f"{joint_base}_{angle_type}"
            limits = self.PHYSIOLOGICAL_LIMITS.get(joint_name)

            values = []
            for row in data_rows:
                if col_idx < len(row):
                    try:
                        values.append(float(row[col_idx]))
                    except ValueError:
                        continue

            if not values:
                continue

            arr = np.array(values)
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))

            # Count out-of-range frames
            out_of_range = 0
            if limits:
                out_of_range = int(np.sum((arr < limits[0]) | (arr > limits[1])))

            within_percent = (
                (total_frames - out_of_range) / total_frames * 100.0
                if total_frames > 0
                else 0.0
            )

            metrics.append(
                JointRangeMetrics(
                    joint_name=joint_name,
                    min_value=round(min_val, 2),
                    max_value=round(max_val, 2),
                    mean_value=round(mean_val, 2),
                    std_value=round(std_val, 2),
                    out_of_range_frames=out_of_range,
                    total_frames=total_frames,
                    within_limits_percent=round(within_percent, 2),
                )
            )

        return metrics

    def save_metrics(self, metrics: QualityMetrics, output_dir: Path) -> Path:
        """Save quality metrics to a JSON file in the output directory."""
        output_file = output_dir / "quality_metrics.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        return output_file

    def load_metrics(self, output_dir: Path) -> QualityMetrics | None:
        """Load quality metrics from a run's output directory."""
        metrics_file = output_dir / "quality_metrics.json"
        if not metrics_file.exists():
            return None
        try:
            with metrics_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return QualityMetrics.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
