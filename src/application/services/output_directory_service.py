"""Service for preparing pipeline output directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputDirectories:
    """Container for pipeline output directory paths."""

    pipeline_run_dir: Path
    output_dir: Path


class OutputDirectoryService:
    """Create and validate output paths for pipeline runs."""

    def __init__(self, output_root: Path, repo_root: Path) -> None:
        self._output_root = output_root
        self._repo_root = repo_root

    def prepare_directories(
        self, safe_run_id: str, run_key: str
    ) -> tuple[OutputDirectories, list[str]]:
        """Create run output directories and return validation errors."""
        pipeline_run_dir = self._output_root / safe_run_id
        output_dir = self._output_root / run_key
        pipeline_run_dir.mkdir(parents=True, exist_ok=True)

        errors: list[str] = []
        if output_dir != pipeline_run_dir and output_dir.exists():
            errors.append(
                "Output location already exists. Choose a different output folder."
            )
            errors.append(f"Output path: {output_dir}")

        return OutputDirectories(
            pipeline_run_dir=pipeline_run_dir,
            output_dir=output_dir,
        ), errors
