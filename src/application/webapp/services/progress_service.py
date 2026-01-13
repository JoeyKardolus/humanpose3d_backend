from __future__ import annotations

"""Service for composing progress response payloads."""

import time

from src.application.webapp.dto.progress_payload import ProgressPayload
from src.application.webapp.repositories.run_status_repository import RunStatusRepository


class ProgressService:
    """Read status data and shape progress responses."""

    def __init__(self, status_repo: RunStatusRepository) -> None:
        self._status_repo = status_repo
        self._progress_init_cap = 12.0

    def build_payload(self, run_key: str) -> ProgressPayload | None:
        """Build a progress payload for a run key."""
        status = self._status_repo.get_status(run_key)
        if status is None:
            return None
        payload = ProgressPayload(
            run_key=run_key,
            progress=float(status.get("progress", 0.0)),
            stage=status.get("stage", "Running"),
            done=bool(status.get("done", False)),
            error=status.get("error"),
            results_url=status.get("results_url"),
        )
        if payload.done:
            return payload

        started_at = status.get("started_at")
        if isinstance(started_at, (int, float)) and payload.progress < self._progress_init_cap:
            elapsed = max(time.monotonic() - started_at, 0.0)
            warm_progress = min(self._progress_init_cap, elapsed * 0.4)
            if warm_progress > payload.progress:
                self._status_repo.set_status(
                    run_key,
                    {
                        "progress": warm_progress,
                        "stage": payload.stage or "Extracting landmarks",
                        "last_update": time.monotonic(),
                    },
                )
                payload = ProgressPayload(
                    run_key=run_key,
                    progress=warm_progress,
                    stage=payload.stage or "Extracting landmarks",
                    done=payload.done,
                    error=payload.error,
                    results_url=payload.results_url,
                )
        return payload
