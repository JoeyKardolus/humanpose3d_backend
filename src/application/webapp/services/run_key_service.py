"""Service for building pipeline run keys."""

from __future__ import annotations


from src.application.webapp.validators.path_validator import PathValidator


class RunKeyService:
    """Compose run keys using output location and run ids."""

    def __init__(self, path_validator: PathValidator) -> None:
        self._path_validator = path_validator

    def build_run_key(self, output_location: str | None, safe_run_id: str) -> str:
        """Create the run key for a given output location."""
        if not output_location:
            return safe_run_id
        output_path = self._path_validator.safe_relative_path(output_location)
        if output_path is None:
            return safe_run_id
        return (output_path / safe_run_id).as_posix()
