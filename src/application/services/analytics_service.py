"""Service for tracking pipeline analytics events."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from src.application.dto.analytics_event import AnalyticsEvent, PipelineSettings
from src.application.dto.quality_metrics import QualityMetrics
from src.application.repositories.analytics_repository import AnalyticsRepository


class AnalyticsService:
    """Track pipeline usage and quality metrics."""

    def __init__(self, repository: AnalyticsRepository) -> None:
        self._repository = repository
        self._run_start_times: dict[str, float] = {}

    def track_run_started(
        self,
        run_key: str,
        form_data: Mapping[str, Any],
    ) -> None:
        """Track when a pipeline run starts."""
        self._run_start_times[run_key] = time.monotonic()
        settings = PipelineSettings.from_form_data(dict(form_data))
        event = AnalyticsEvent(
            timestamp=datetime.utcnow(),
            event_type="run_started",
            run_key=run_key,
            settings=settings.to_dict(),
        )
        self._repository.append_event(event)

    def track_run_completed(
        self,
        run_key: str,
        output_dir: Path | None = None,
        quality_metrics: QualityMetrics | None = None,
    ) -> None:
        """Track when a pipeline run completes successfully."""
        duration = None
        if run_key in self._run_start_times:
            duration = time.monotonic() - self._run_start_times.pop(run_key)

        # Get settings from the started event if available
        events = self._repository.list_events(event_type="run_started", limit=100)
        settings = {}
        for event in reversed(events):
            if event.run_key == run_key:
                settings = event.settings
                break

        quality_dict = quality_metrics.to_dict() if quality_metrics else None

        event = AnalyticsEvent(
            timestamp=datetime.utcnow(),
            event_type="run_completed",
            run_key=run_key,
            settings=settings,
            duration_seconds=duration,
            quality_metrics=quality_dict,
        )
        self._repository.append_event(event)

    def track_run_error(
        self,
        run_key: str,
        error_type: str,
        error_message: str,
    ) -> None:
        """Track when a pipeline run fails."""
        duration = None
        if run_key in self._run_start_times:
            duration = time.monotonic() - self._run_start_times.pop(run_key)

        # Get settings from the started event if available
        events = self._repository.list_events(event_type="run_started", limit=100)
        settings = {}
        for event in reversed(events):
            if event.run_key == run_key:
                settings = event.settings
                break

        event = AnalyticsEvent(
            timestamp=datetime.utcnow(),
            event_type="run_error",
            run_key=run_key,
            settings=settings,
            duration_seconds=duration,
            error_type=error_type,
            error_message=error_message,
        )
        self._repository.append_event(event)

    def get_dashboard_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics for the analytics dashboard."""
        stats = self._repository.get_stats(days=30)
        today = self._repository.get_today_stats()
        week = self._repository.get_week_stats()

        return {
            **stats,
            **today,
            **week,
        }

    def get_quality_trend(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent quality scores with their settings for trend analysis."""
        events = self._repository.list_events(event_type="run_completed", limit=limit)
        trend = []
        for event in reversed(events):
            if event.quality_metrics:
                trend.append({
                    "timestamp": event.timestamp.isoformat(),
                    "run_key": event.run_key,
                    "plausibility_score": event.quality_metrics.get("plausibility_score"),
                    "settings": event.settings,
                })
        return trend

    def get_settings_correlation(self) -> dict[str, dict[str, float]]:
        """Analyze which settings correlate with better quality scores."""
        stats = self._repository.get_stats(days=30)
        settings_quality = stats.get("settings_quality", [])

        if not settings_quality:
            return {}

        correlations: dict[str, dict[str, list[float]]] = {
            "camera_pof": {"enabled": [], "disabled": []},
            "joint_refinement": {"enabled": [], "disabled": []},
            "estimate_missing": {"enabled": [], "disabled": []},
        }

        for settings, quality in settings_quality:
            for flag in correlations:
                key = "enabled" if settings.get(flag) else "disabled"
                correlations[flag][key].append(quality)

        result = {}
        for flag, values in correlations.items():
            enabled_avg = sum(values["enabled"]) / len(values["enabled"]) if values["enabled"] else 0
            disabled_avg = sum(values["disabled"]) / len(values["disabled"]) if values["disabled"] else 0
            result[flag] = {
                "enabled_avg": enabled_avg,
                "disabled_avg": disabled_avg,
                "enabled_count": len(values["enabled"]),
                "disabled_count": len(values["disabled"]),
            }

        return result
