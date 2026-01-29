"""DTO for kinematics quality metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class JointRangeMetrics:
    """Metrics for a single joint's range of motion."""

    joint_name: str
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    out_of_range_frames: int
    total_frames: int
    within_limits_percent: float


@dataclass(frozen=True)
class QualityMetrics:
    """Comprehensive kinematics quality analysis results."""

    plausibility_score: float
    total_frames: int
    analyzed_joints: int
    out_of_range_summary: dict[str, int] = field(default_factory=dict)
    motion_smoothness: float = 0.0
    bone_length_variance: float = 0.0
    marker_dropout_rate: float = 0.0
    joint_metrics: list[JointRangeMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "plausibility_score": self.plausibility_score,
            "total_frames": self.total_frames,
            "analyzed_joints": self.analyzed_joints,
            "out_of_range_summary": self.out_of_range_summary,
            "motion_smoothness": self.motion_smoothness,
            "bone_length_variance": self.bone_length_variance,
            "marker_dropout_rate": self.marker_dropout_rate,
            "joint_metrics": [
                {
                    "joint_name": jm.joint_name,
                    "min_value": jm.min_value,
                    "max_value": jm.max_value,
                    "mean_value": jm.mean_value,
                    "std_value": jm.std_value,
                    "out_of_range_frames": jm.out_of_range_frames,
                    "total_frames": jm.total_frames,
                    "within_limits_percent": jm.within_limits_percent,
                }
                for jm in self.joint_metrics
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualityMetrics":
        """Create QualityMetrics from a dictionary."""
        joint_metrics = [
            JointRangeMetrics(
                joint_name=jm["joint_name"],
                min_value=jm["min_value"],
                max_value=jm["max_value"],
                mean_value=jm["mean_value"],
                std_value=jm["std_value"],
                out_of_range_frames=jm["out_of_range_frames"],
                total_frames=jm["total_frames"],
                within_limits_percent=jm["within_limits_percent"],
            )
            for jm in data.get("joint_metrics", [])
        ]
        return cls(
            plausibility_score=data["plausibility_score"],
            total_frames=data["total_frames"],
            analyzed_joints=data["analyzed_joints"],
            out_of_range_summary=data.get("out_of_range_summary", {}),
            motion_smoothness=data.get("motion_smoothness", 0.0),
            bone_length_variance=data.get("bone_length_variance", 0.0),
            marker_dropout_rate=data.get("marker_dropout_rate", 0.0),
            joint_metrics=joint_metrics,
        )
