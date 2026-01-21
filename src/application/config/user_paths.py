"""User-specific path configuration for ~/.humanpose3d directory.

All user data (models, input videos, output results) is stored in a platform-independent
location in the user's home directory. This ensures:
- Portability across operating systems (Windows, macOS, Linux)
- Separation of application code and user data
- PyInstaller compatibility (data persists after application updates)
- No write permissions needed in application directory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UserPaths:
    """Paths within the user's home directory (~/.humanpose3d).

    Platform-independent path configuration:
    - Windows: C:\\Users\\<username>\\.humanpose3d
    - macOS: /Users/<username>/.humanpose3d
    - Linux: /home/<username>/.humanpose3d

    All paths use pathlib.Path for cross-platform compatibility.
    """

    base: Path
    models: Path
    models_checkpoints: Path
    data_input: Path
    data_output: Path

    @classmethod
    def default(cls) -> "UserPaths":
        """Create UserPaths with default ~/.humanpose3d location.

        Uses Path.home() which automatically resolves to the correct home
        directory on all platforms without hardcoded separators.
        """
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
