from __future__ import annotations

"""Service for removing stored pipeline data."""

import shutil
from pathlib import Path

from django.http import Http404

from src.application.webapp.validators.path_validator import PathValidator


class RunCleanupService:
    """Delete output runs and uploaded input data."""

    def __init__(
        self,
        output_root: Path,
        upload_root: Path,
        path_validator: PathValidator,
    ) -> None:
        self._output_root = output_root
        self._upload_root = upload_root
        self._path_validator = path_validator

    def delete_run(self, run_key: str) -> bool:
        """Remove output and upload directories for a run key."""
        run_dir = self._path_validator.resolve_output_dir(self._output_root, run_key)
        deleted = False
        if run_dir.exists():
            shutil.rmtree(run_dir)
            deleted = True

        upload_dir = (self._upload_root / run_dir.name).resolve()
        if self._upload_root not in upload_dir.parents and upload_dir != self._upload_root:
            raise Http404("Invalid upload path.")
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            deleted = True

        return deleted
