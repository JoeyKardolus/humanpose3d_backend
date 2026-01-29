"""Repository for storing bug reports as JSON files."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from src.application.dto.bug_report import BugReport


class BugReportRepository:
    """Thread-safe JSON file storage for bug reports."""

    def __init__(self, logs_root: Path) -> None:
        self._logs_root = logs_root
        self._reports_dir = logs_root / "bug_reports"
        self._lock = threading.Lock()

    def _ensure_dir(self) -> None:
        """Create bug reports directory if it doesn't exist."""
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, report: BugReport) -> Path:
        """Save a bug report and return the file path."""
        with self._lock:
            self._ensure_dir()
            timestamp_str = report.timestamp.strftime("%Y%m%d_%H%M%S")
            run_key_safe = (report.run_key or "unknown").replace("/", "_").replace("\\", "_")
            filename = f"{timestamp_str}_{run_key_safe}.json"
            file_path = self._reports_dir / filename

            with file_path.open("w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)

            return file_path

    def list_reports(self, limit: int | None = None) -> list[BugReport]:
        """List all bug reports, newest first."""
        reports: list[BugReport] = []
        with self._lock:
            if not self._reports_dir.exists():
                return reports

            files = sorted(self._reports_dir.glob("*.json"), reverse=True)
            for file_path in files:
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        reports.append(BugReport.from_dict(data))
                        if limit and len(reports) >= limit:
                            break
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        return reports

    def get_report_count(self) -> int:
        """Count total number of bug reports."""
        with self._lock:
            if not self._reports_dir.exists():
                return 0
            return len(list(self._reports_dir.glob("*.json")))
