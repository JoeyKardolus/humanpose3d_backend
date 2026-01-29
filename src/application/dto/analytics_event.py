"""DTO for analytics event data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AnalyticsEvent:
    """Represents a single analytics event for pipeline runs."""

    timestamp: datetime
    event_type: str
    run_key: str
    settings: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float | None = None
    quality_metrics: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "run_key": self.run_key,
            "settings": self.settings,
            "duration_seconds": self.duration_seconds,
            "quality_metrics": self.quality_metrics,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyticsEvent":
        """Create an AnalyticsEvent from a dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=data["event_type"],
            run_key=data["run_key"],
            settings=data.get("settings", {}),
            duration_seconds=data.get("duration_seconds"),
            quality_metrics=data.get("quality_metrics"),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
        )


@dataclass(frozen=True)
class PipelineSettings:
    """Captured pipeline settings for analytics correlation."""

    height: float | None = None
    mass: float | None = None
    visibility_min: float | None = None
    augmentation_cycles: int | None = None
    estimate_missing: bool = False
    force_complete: bool = False
    camera_pof: bool = False
    joint_refinement: bool = False
    compute_all_joint_angles: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "height": self.height,
            "mass": self.mass,
            "visibility_min": self.visibility_min,
            "augmentation_cycles": self.augmentation_cycles,
            "estimate_missing": self.estimate_missing,
            "force_complete": self.force_complete,
            "camera_pof": self.camera_pof,
            "joint_refinement": self.joint_refinement,
            "compute_all_joint_angles": self.compute_all_joint_angles,
        }

    @classmethod
    def from_form_data(cls, form_data: dict[str, Any]) -> "PipelineSettings":
        """Create settings from form data."""

        def parse_float(value: Any) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        def parse_int(value: Any) -> int | None:
            if value is None or value == "":
                return None
            try:
                return int(value)
            except (ValueError, TypeError):
                return None

        def is_checked(value: Any) -> bool:
            return value is not None and value != ""

        return cls(
            height=parse_float(form_data.get("height")),
            mass=parse_float(form_data.get("weight")),
            visibility_min=parse_float(form_data.get("visibility_min")),
            augmentation_cycles=parse_int(form_data.get("augmentation_cycles")),
            estimate_missing=is_checked(form_data.get("estimate_missing")),
            force_complete=is_checked(form_data.get("force_complete")),
            camera_pof=is_checked(form_data.get("camera_pof")),
            joint_refinement=is_checked(form_data.get("joint_refinement")),
            compute_all_joint_angles=is_checked(form_data.get("compute_all_joint_angles")),
        )
