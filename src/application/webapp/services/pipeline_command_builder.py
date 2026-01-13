from __future__ import annotations

"""Service for constructing the CLI command used to run the pipeline."""

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
        sex_raw: str,
    ) -> list[str]:
        """Translate form data into a pipeline CLI invocation."""
        main_path = self._repo_root / "main.py"
        command = [
            sys.executable,
            "-u",
            str(main_path),
            "--video",
            str(upload_path),
        ]

        def _add_flag(name: str, value: str | None, default: str | None = None) -> None:
            if value is None or value == "":
                if default is not None:
                    command.extend([name, default])
                return
            command.extend([name, value])

        _add_flag("--height", form_data.get("height"), default="1.78")
        _add_flag("--mass", form_data.get("weight"), default="75.0")
        _add_flag("--age", form_data.get("age"), default="30")
        if sex_raw in {"male", "female"}:
            _add_flag("--sex", sex_raw)
        _add_flag("--visibility-min", form_data.get("visibility_min"), default="0.3")
        _add_flag(
            "--augmentation-cycles", form_data.get("augmentation_cycles"), default="20"
        )
        _add_flag(
            "--joint-angle-smooth-window",
            form_data.get("joint_angle_smooth_window"),
            default="9",
        )
        _add_flag("--bone-smooth-window", form_data.get("bone_smooth_window"), default="21")
        _add_flag("--ground-percentile", form_data.get("ground_percentile"), default="5.0")
        _add_flag("--ground-margin", form_data.get("ground_margin"), default="0.02")
        _add_flag(
            "--bone-length-tolerance",
            form_data.get("bone_length_tolerance"),
            default="0.15",
        )
        _add_flag("--bone-depth-weight", form_data.get("bone_depth_weight"), default="0.8")
        _add_flag(
            "--bone-length-iterations", form_data.get("bone_length_iterations"), default="3"
        )
        _add_flag(
            "--multi-constraint-iterations",
            form_data.get("multi_constraint_iterations"),
            default="10",
        )

        if self._coerce_bool(form_data.get("estimate_missing")):
            command.append("--estimate-missing")
        if self._coerce_bool(form_data.get("force_complete")):
            command.append("--force-complete")
        if self._coerce_bool(form_data.get("anatomical_constraints")):
            command.append("--anatomical-constraints")
        if self._coerce_bool(form_data.get("bone_length_constraints")):
            command.append("--bone-length-constraints")
        if self._coerce_bool(form_data.get("multi_constraint_optimization")):
            command.append("--multi-constraint-optimization")
        if self._coerce_bool(form_data.get("compute_all_joint_angles")):
            command.append("--compute-all-joint-angles")
        return command

    def _coerce_bool(self, value: str | None) -> bool:
        """Interpret checkbox form values as booleans."""
        return value is not None
