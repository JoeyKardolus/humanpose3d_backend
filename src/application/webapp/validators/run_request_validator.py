"""Validation rules for pipeline run requests."""

from __future__ import annotations

from typing import Mapping

from src.application.webapp.validators.path_validator import PathValidator


class RunRequestValidator:
    """Applies request-level validation for pipeline runs."""

    def __init__(self, path_validator: PathValidator) -> None:
        self._path_validator = path_validator

    def validate(
        self,
        form_data: Mapping[str, str],
        has_upload: bool,
    ) -> tuple[list[str], str, str | None]:
        """Validate incoming form data and return errors and normalized values."""
        errors: list[str] = []
        if not has_upload:
            errors.append("Please upload a video file before submitting.")

        if form_data.get("consent") != "accepted":
            errors.append(
                "You must confirm participant consent before running an analysis."
            )

        output_location = form_data.get("output_location", "").strip()
        output_path = self._path_validator.safe_relative_path(output_location)
        if output_location and not output_path:
            errors.append("Output location must be a relative path (no .. segments).")

        sex_raw = form_data.get("sex", "").strip().lower()
        if sex_raw and sex_raw not in {"male", "female"}:
            errors.append("Sex must be Male or Female for the current pipeline.")

        return errors, sex_raw, output_location or None
