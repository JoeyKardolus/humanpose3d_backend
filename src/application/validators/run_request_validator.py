"""Validation rules for pipeline run requests."""

from __future__ import annotations

from typing import Mapping

from src.application.validators.path_validator import PathValidator


class RunRequestValidator:
    """Applies request-level validation for pipeline runs."""

    def __init__(self, path_validator: PathValidator) -> None:
        self._path_validator = path_validator

    def validate(
        self,
        form_data: Mapping[str, str],
        has_upload: bool,
    ) -> tuple[list[str], str | None]:
        """Validate incoming form data and return errors and normalized values."""
        errors: list[str] = []
        if not has_upload:
            errors.append("Please upload a video file before submitting.")

        if form_data.get("consent") != "accepted":
            errors.append(
                "You must confirm participant consent before running an analysis."
            )

        # Validate height (optional but must be in range if provided)
        height_str = form_data.get("height", "").strip()
        if height_str:
            try:
                height = float(height_str)
                if height < 0.5 or height > 2.5:
                    errors.append("Height must be between 0.5 and 2.5 meters.")
            except ValueError:
                errors.append("Height must be a valid number.")

        # Validate weight (optional but must be in range if provided)
        weight_str = form_data.get("weight", "").strip()
        if weight_str:
            try:
                weight = float(weight_str)
                if weight < 10 or weight > 500:
                    errors.append("Weight must be between 10 and 500 kg.")
            except ValueError:
                errors.append("Weight must be a valid number.")

        # Validate visibility threshold
        visibility_str = form_data.get("visibility_min", "").strip()
        if visibility_str:
            try:
                visibility = float(visibility_str)
                if visibility < 0 or visibility > 1:
                    errors.append("Visibility threshold must be between 0 and 1.")
            except ValueError:
                errors.append("Visibility threshold must be a valid number.")

        # Validate augmentation cycles
        cycles_str = form_data.get("augmentation_cycles", "").strip()
        if cycles_str:
            try:
                cycles = int(cycles_str)
                if cycles < 1:
                    errors.append("Augmentation cycles must be at least 1.")
            except ValueError:
                errors.append("Augmentation cycles must be a valid integer.")

        output_location = form_data.get("output_location", "").strip()
        output_path = self._path_validator.safe_relative_path(output_location)
        if output_location and not output_path:
            errors.append("Output location must be a relative path (no .. segments).")

        return errors, output_location or None
