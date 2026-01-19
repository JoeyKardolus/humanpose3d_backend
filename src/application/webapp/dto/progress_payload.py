"""DTO for progress responses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressPayload:
    """Serializable progress state for a running pipeline."""

    run_key: str
    progress: float
    stage: str | None
    done: bool
    error: str | None
    results_url: str | None
