"""Append pipeline run details to the build log."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.application.config.paths import StoragePaths


def append_build_log(message: str, log_path: Path | None = None) -> None:
    """Append a log entry with UTC timestamp to the build log."""
    resolved_log_path = _resolve_log_path(log_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"- {message} ({timestamp})\n"
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(line)


def _resolve_log_path(log_path: Path | None) -> Path:
    if log_path is not None:
        return log_path
    storage_paths = StoragePaths.load()
    return storage_paths.logs_root / "BUILD_LOG.md"
