"""Build log helper for recording build events."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def append_build_log(message: str) -> None:
    """Append a timestamped entry to docs/BUILD_LOG.md."""
    repo_root = Path(__file__).resolve().parents[2]
    log_path = repo_root / "docs" / "BUILD_LOG.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    entry = f"- {timestamp} | {message}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
