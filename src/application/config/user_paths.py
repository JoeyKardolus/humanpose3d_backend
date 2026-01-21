"""User-specific path configuration for ~/.humanpose3d directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UserPaths:
    """Paths within the user's home directory (~/.humanpose3d)."""

    base: Path
    models: Path
    models_checkpoints: Path
    data_input: Path
    data_output: Path

    @classmethod
    def default(cls) -> "UserPaths":
        """Create UserPaths with default ~/.humanpose3d location."""
        base = Path.home() / ".humanpose3d"
        return cls(
            base=base,
            models=base / "models",
            models_checkpoints=base / "models" / "checkpoints",
            data_input=base / "data" / "input",
            data_output=base / "data" / "output",
        )

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        self.base.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        self.models_checkpoints.mkdir(parents=True, exist_ok=True)
        self.data_input.mkdir(parents=True, exist_ok=True)
        self.data_output.mkdir(parents=True, exist_ok=True)

    def models_exist(self) -> bool:
        """Check if all required model files exist."""
        required_models = [
            self.models_checkpoints / "best_depth_model.pth",
            self.models_checkpoints / "best_joint_model.pth",
            self.models_checkpoints / "best_main_refiner.pth",
            self.models / "pose_landmarker_heavy.task",
            self.models / "GRU.h5",
        ]
        return all(model.exists() for model in required_models)
