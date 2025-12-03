from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

BUILD_LOG_PATH = Path(__file__).resolve().parents[2] / "docs" / "BUILD_LOG.md"


def append_build_log(message: str) -> None:
    """Append a single strict-mode status line to docs/BUILD_LOG.md."""
    BUILD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"- {message} ({timestamp})\n"
    with BUILD_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
