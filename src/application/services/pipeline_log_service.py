"""Service for writing pipeline execution logs."""

from __future__ import annotations

from pathlib import Path


class PipelineLogService:
    """Persist stdout/stderr logs to disk."""

    def write_log(self, log_path: Path, stdout_text: str, stderr_text: str) -> None:
        """Write a structured log file with stdout and stderr sections."""
        log_path.write_text(
            "\n".join(
                [
                    "[stdout]",
                    stdout_text.strip(),
                    "",
                    "[stderr]",
                    stderr_text.strip(),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
