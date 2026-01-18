from __future__ import annotations

"""Service for listing previous pipeline run outputs."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputRunEntry:
    """Metadata describing a stored pipeline run."""

    run_key: str
    display_name: str
    modified_time: float


class OutputHistoryService:
    """Enumerate run output directories for the UI."""

    _FINAL_TRC_SUFFIX = "_final.trc"
    _ERROR_LOG_NAME = "pipeline_error.log"

    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root

    def list_runs(self) -> list[OutputRunEntry]:
        """Return stored run metadata sorted by most recent."""
        if not self._output_root.exists():
            return []

        run_dirs = self._find_run_dirs()
        entries = [
            OutputRunEntry(
                run_key=run_dir.relative_to(self._output_root).as_posix(),
                display_name=self._format_display_name(run_dir),
                modified_time=run_dir.stat().st_mtime,
            )
            for run_dir in run_dirs
        ]
        return sorted(entries, key=lambda entry: entry.modified_time, reverse=True)

    def _find_run_dirs(self) -> set[Path]:
        run_dirs: set[Path] = set()
        for candidate in self._output_root.rglob("*"):
            if not candidate.is_dir():
                continue
            if self._contains_run_marker(candidate):
                run_dirs.add(candidate)
        return run_dirs

    def _contains_run_marker(self, candidate: Path) -> bool:
        try:
            for item in candidate.iterdir():
                if not item.is_file():
                    continue
                if item.name == self._ERROR_LOG_NAME:
                    return True
                if item.name.endswith(self._FINAL_TRC_SUFFIX):
                    return True
        except OSError:
            return False
        return False

    def _format_display_name(self, run_dir: Path) -> str:
        run_key = run_dir.relative_to(self._output_root).as_posix()
        return run_key.replace("/", " / ")
