from __future__ import annotations

"""In-memory repository for tracking pipeline run status."""

import threading
from typing import Mapping


class RunStatusRepository:
    """Thread-safe store for run status metadata."""

    def __init__(self) -> None:
        self._status: dict[str, dict[str, object]] = {}
        self._lock = threading.Lock()

    def set_status(self, run_key: str, updates: Mapping[str, object]) -> None:
        """Merge updates into an existing run status entry."""
        with self._lock:
            status = self._status.setdefault(run_key, {})
            status.update(updates)

    def get_status(self, run_key: str) -> dict[str, object] | None:
        """Return a copy of the status map for a run."""
        with self._lock:
            status = self._status.get(run_key)
            return dict(status) if status is not None else None
