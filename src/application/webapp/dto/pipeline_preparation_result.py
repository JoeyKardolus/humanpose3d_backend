from __future__ import annotations

"""DTO for pipeline preparation outcomes."""

from dataclasses import dataclass

from src.application.webapp.dto.pipeline_run_spec import PipelineRunSpec


@dataclass(frozen=True)
class PipelinePreparationResult:
    """Aggregates validation errors and a prepared run spec."""

    errors: list[str]
    prepared: PipelineRunSpec | None
