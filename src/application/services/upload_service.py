"""Service for persisting uploaded videos."""

from __future__ import annotations

import shutil
from pathlib import Path

from django.core.files.uploadedfile import UploadedFile


class UploadService:
    """Persist incoming uploads to the configured directory."""

    def __init__(self, upload_root: Path) -> None:
        self._upload_root = upload_root

    def save_upload(self, uploaded: UploadedFile, safe_run_id: str) -> Path:
        """Write the uploaded file to disk and return its path."""
        upload_dir = self._upload_root / safe_run_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / f"{safe_run_id}{Path(uploaded.name).suffix}"
        with upload_path.open("wb") as handle:
            for chunk in uploaded.chunks():
                handle.write(chunk)
        return upload_path

    def remove_upload(self, safe_run_id: str) -> None:
        """Remove persisted uploads for a run if they exist."""
        upload_dir = self._upload_root / safe_run_id
        if not upload_dir.exists():
            return
        try:
            shutil.rmtree(upload_dir)
        except OSError:
            return
