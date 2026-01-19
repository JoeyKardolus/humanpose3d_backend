"""Service for resolving media file paths."""

from __future__ import annotations

from pathlib import Path

from django.http import Http404

from src.application.webapp.validators.path_validator import PathValidator


class MediaService:
    """Resolve and validate file paths for downloads and previews."""

    def __init__(
        self, output_root: Path, upload_root: Path, path_validator: PathValidator
    ) -> None:
        self._output_root = output_root
        self._upload_root = upload_root
        self._path_validator = path_validator

    def resolve_output_file(self, run_key: str, file_path: str) -> Path:
        """Resolve a file within the output directory tree."""
        run_dir = self._path_validator.resolve_output_dir(self._output_root, run_key)
        target = (run_dir / file_path).resolve()
        if run_dir not in target.parents and target != run_dir:
            raise Http404("Invalid file path.")
        if not target.exists() or not target.is_file():
            raise Http404("File not found.")
        return target

    def resolve_upload_file(self, run_key: str, file_path: str) -> Path:
        """Resolve a file within the upload directory tree."""
        safe_run_id = Path(run_key).name
        upload_dir = (self._upload_root / safe_run_id).resolve()
        target = (upload_dir / file_path).resolve()
        if upload_dir not in target.parents and target != upload_dir:
            raise Http404("Invalid file path.")
        if not target.exists() or not target.is_file():
            raise Http404("File not found.")
        return target
