"""Path configuration helpers for the webapp domain."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_STORAGE_ENV_VAR = "HUMANPOSE3D_HOME"
_DEFAULT_STORAGE_DIRNAME = ".humanpose3d"


@dataclass(frozen=True)
class StoragePaths:
    """Resolved filesystem roots for data, models, and logs."""

    root: Path
    input_root: Path
    output_root: Path
    models_root: Path
    checkpoints_root: Path
    logs_root: Path
    training_root: Path

    @classmethod
    def load(cls) -> "StoragePaths":
        """Resolve storage roots from environment or defaults."""
        root_value = os.environ.get(_STORAGE_ENV_VAR)
        root = Path(root_value).expanduser() if root_value else Path.home() / _DEFAULT_STORAGE_DIRNAME
        root = root.resolve()
        models_root = root / "models"
        return cls(
            root=root,
            input_root=root / "input",
            output_root=root / "output",
            models_root=models_root,
            checkpoints_root=models_root / "checkpoints",
            logs_root=root / "logs",
            training_root=root / "training",
        )


@dataclass(frozen=True)
class AppPaths:
    """Resolved filesystem roots used by the webapp."""

    repo_root: Path
    storage_root: Path
    output_root: Path
    upload_root: Path

    @classmethod
    def from_anchor(cls, anchor: Path) -> "AppPaths":
        """Resolve application roots relative to a module anchor."""
        repo_root = None
        for parent in anchor.resolve().parents:
            if parent.name == "src":
                repo_root = parent.parent
                break
        if repo_root is None:
            repo_root = anchor.resolve().parents[3]
        storage_paths = StoragePaths.load()
        output_root = storage_paths.output_root
        upload_root = storage_paths.input_root
        return cls(
            repo_root=repo_root,
            storage_root=storage_paths.root,
            output_root=output_root,
            upload_root=upload_root,
        )
