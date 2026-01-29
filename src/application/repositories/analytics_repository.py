"""Repository for storing analytics events in JSONL format."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from src.application.dto.analytics_event import AnalyticsEvent


class AnalyticsRepository:
    """Thread-safe JSONL file storage for analytics events."""

    def __init__(self, logs_root: Path) -> None:
        self._logs_root = logs_root
        self._lock = threading.Lock()
        self._events_file = logs_root / "analytics.jsonl"

    def _ensure_dir(self) -> None:
        """Create logs directory if it doesn't exist."""
        self._logs_root.mkdir(parents=True, exist_ok=True)

    def append_event(self, event: AnalyticsEvent) -> None:
        """Append an event to the JSONL log file."""
        with self._lock:
            self._ensure_dir()
            with self._events_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

    def list_events(
        self,
        since: datetime | None = None,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[AnalyticsEvent]:
        """List events with optional filtering."""
        events: list[AnalyticsEvent] = []
        with self._lock:
            if not self._events_file.exists():
                return events
            for event in self._iter_events_unlocked():
                if since and event.timestamp < since:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                events.append(event)
                if limit and len(events) >= limit:
                    break
        return events

    def _iter_events_unlocked(self) -> Iterator[AnalyticsEvent]:
        """Iterate over events without locking (caller must hold lock)."""
        if not self._events_file.exists():
            return
        with self._events_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield AnalyticsEvent.from_dict(data)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    def get_stats(self, days: int = 30) -> dict[str, object]:
        """Calculate aggregate statistics for the analytics dashboard."""
        since = datetime.utcnow() - timedelta(days=days)
        events = self.list_events(since=since)

        total_runs = 0
        completed_runs = 0
        error_runs = 0
        total_duration = 0.0
        error_types: dict[str, int] = {}
        settings_quality: list[tuple[dict, float]] = []

        for event in events:
            if event.event_type == "run_started":
                total_runs += 1
            elif event.event_type == "run_completed":
                completed_runs += 1
                if event.duration_seconds:
                    total_duration += event.duration_seconds
                if event.quality_metrics:
                    plausibility = event.quality_metrics.get("plausibility_score")
                    if plausibility is not None:
                        settings_quality.append((event.settings, plausibility))
            elif event.event_type == "run_error":
                error_runs += 1
                if event.error_type:
                    error_types[event.error_type] = error_types.get(event.error_type, 0) + 1

        success_rate = completed_runs / total_runs if total_runs > 0 else 0.0
        avg_duration = total_duration / completed_runs if completed_runs > 0 else 0.0

        return {
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "error_runs": error_runs,
            "success_rate": success_rate,
            "avg_duration_seconds": avg_duration,
            "error_types": error_types,
            "settings_quality": settings_quality,
        }

    def get_today_stats(self) -> dict[str, int]:
        """Get run counts for today."""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        events = self.list_events(since=today)

        started = sum(1 for e in events if e.event_type == "run_started")
        completed = sum(1 for e in events if e.event_type == "run_completed")
        errors = sum(1 for e in events if e.event_type == "run_error")

        return {
            "today_started": started,
            "today_completed": completed,
            "today_errors": errors,
        }

    def get_week_stats(self) -> dict[str, int]:
        """Get run counts for the past week."""
        week_ago = datetime.utcnow() - timedelta(days=7)
        events = self.list_events(since=week_ago)

        started = sum(1 for e in events if e.event_type == "run_started")
        completed = sum(1 for e in events if e.event_type == "run_completed")
        errors = sum(1 for e in events if e.event_type == "run_error")

        return {
            "week_started": started,
            "week_completed": completed,
            "week_errors": errors,
        }
