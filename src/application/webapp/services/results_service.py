from __future__ import annotations

"""Service for enumerating pipeline output files."""

from pathlib import Path


class ResultsService:
    """Build file listings for output browsing."""

    def list_files(self, run_dir: Path) -> list[dict[str, object]]:
        """Return metadata for files within the run directory."""
        files: list[dict[str, object]] = []
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "name": path.name,
                        "relative_path": path.relative_to(run_dir).as_posix(),
                        "size_kb": max(path.stat().st_size // 1024, 1),
                    }
                )
        return files
