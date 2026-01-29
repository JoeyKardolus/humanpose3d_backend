"""Service for constructing the CLI command used to run the pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping


class PipelineCommandBuilder:
    """Builds the pipeline command based on user input."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root

    def build(
        self,
        upload_path: Path,
        form_data: Mapping[str, str],
    ) -> list[str]:
        """Translate form data into a pipeline CLI invocation."""
        manage_path = self._repo_root / "manage.py"
        command = [
            sys.executable,
            "-u",
            str(manage_path),
            "run_pipeline",
            "--video",
            str(upload_path),
        ]

        def _add_flag(name: str, value: str | None, default: str | None = None) -> None:
            if value is None or value == "":
                if default is not None:
                    command.extend([name, default])
                return
            command.extend([name, value])

        # Core parameters
        _add_flag("--height", form_data.get("height"), default="1.78")
        _add_flag("--weight", form_data.get("weight"), default="75.0")
        _add_flag("--visibility-min", form_data.get("visibility_min"), default="0.1")
        _add_flag("--pof-model-path", form_data.get("pof_model_path"))
        _add_flag("--joint-model-path", form_data.get("joint_model_path"))
        _add_flag(
            "--augmentation-cycles", form_data.get("augmentation_cycles"), default="20"
        )
        _add_flag(
            "--joint-angle-smooth-window",
            form_data.get("joint_angle_smooth_window"),
            default="9",
        )

        # Boolean flags
        if self._coerce_bool(form_data.get("estimate_missing")):
            command.append("--estimate-missing")
        if self._coerce_bool(form_data.get("force_complete")):
            command.append("--force-complete")
        if self._coerce_bool(form_data.get("camera_pof")):
            command.append("--camera-pof")
        if self._coerce_bool(form_data.get("joint_refinement")):
            command.append("--joint-refinement")
        if self._coerce_bool(form_data.get("compute_all_joint_angles")):
            command.append("--compute-all-joint-angles")
        if self._coerce_bool(form_data.get("plot_all_joint_angles")):
            command.append("--plot-all-joint-angles")

        command.append("--export-preview")
        return command

    def _coerce_bool(self, value: str | None) -> bool:
        """Interpret checkbox form values as booleans."""
        return value is not None
