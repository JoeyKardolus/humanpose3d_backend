from __future__ import annotations

"""Service for generating safe run identifiers."""

import uuid
from pathlib import Path


class RunIdFactory:
    """Builds run ids that are safe for filesystem usage."""

    def create(self, filename: str) -> str:
        """Return a sanitized run id derived from a filename."""
        run_id = f"{Path(filename).stem}-{uuid.uuid4().hex[:8]}"
        safe_run_id = "".join(
            ch for ch in run_id if ch.isalnum() or ch in {"-", "_"}
        ).strip("-_")
        if not safe_run_id:
            safe_run_id = uuid.uuid4().hex[:12]
        return safe_run_id
