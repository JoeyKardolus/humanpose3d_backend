"""DTO for bug report data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class BugReport:
    """Represents a user-submitted bug report."""

    timestamp: datetime
    run_key: str | None
    error_summary: str
    stderr_snippet: str | None = None
    user_comment: str | None = None
    contact_email: str | None = None
    pipeline_settings: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "run_key": self.run_key,
            "error_summary": self.error_summary,
            "stderr_snippet": self.stderr_snippet,
            "user_comment": self.user_comment,
            "contact_email": self.contact_email,
            "pipeline_settings": self.pipeline_settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BugReport":
        """Create a BugReport from a dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            run_key=data.get("run_key"),
            error_summary=data["error_summary"],
            stderr_snippet=data.get("stderr_snippet"),
            user_comment=data.get("user_comment"),
            contact_email=data.get("contact_email"),
            pipeline_settings=data.get("pipeline_settings"),
        )
