from __future__ import annotations

"""DTO capturing pipeline execution output."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineExecutionResult:
    """Return code and collected output from a pipeline run."""

    return_code: int
    stdout_text: str
    stderr_text: str
