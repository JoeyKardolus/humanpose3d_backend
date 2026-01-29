"""Validation helpers for filesystem paths."""

from __future__ import annotations

from pathlib import Path

from django.http import Http404


class PathValidator:
    """Centralizes path validation for user-provided values."""

    def safe_relative_path(self, raw_path: str | None) -> Path | None:
        """Return a safe relative path or None for invalid input."""
        if not raw_path:
            return None
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return None
        if ".." in candidate.parts:
            return None
        return candidate

    def resolve_output_dir(self, output_root: Path, run_key: str) -> Path:
        """Resolve and validate output directories for a run key."""
        run_dir = (output_root / run_key).resolve()
        if output_root not in run_dir.parents and run_dir != output_root:
            raise Http404("Invalid output path.")
        return run_dir
