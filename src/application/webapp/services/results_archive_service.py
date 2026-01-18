from __future__ import annotations

"""Service for building downloadable archives of pipeline outputs."""

import tempfile
import zipfile
from pathlib import Path


class ResultsArchiveService:
    """Create zip archives for output runs."""

    def build_archive(self, run_dir: Path, run_key: str) -> tempfile.NamedTemporaryFile:
        """Create a zip file containing all output files for a run."""
        archive = tempfile.NamedTemporaryFile(suffix=f"_{run_key}.zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for path in sorted(run_dir.rglob("*")):
                if path.is_file():
                    zip_file.write(path, arcname=path.relative_to(run_dir).as_posix())
        archive.seek(0)
        return archive
