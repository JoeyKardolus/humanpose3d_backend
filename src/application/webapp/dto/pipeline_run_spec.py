"""DTO that defines a prepared pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineRunSpec:
    """Immutable specification for a pipeline run."""

    run_key: str
    safe_run_id: str
    upload_path: Path
    pipeline_run_dir: Path
    output_dir: Path
