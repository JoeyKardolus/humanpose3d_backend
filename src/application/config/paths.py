"""Path configuration helpers for the webapp domain."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .user_paths import UserPaths


def get_application_root() -> Path:
    """
    Get the application root directory (platform-independent, PyInstaller-compatible).

    Returns:
        Application root path:
        - When frozen (PyInstaller): directory containing the executable
        - When running from source: repository root directory
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        # sys._MEIPASS is the temporary folder where PyInstaller extracts files
        if hasattr(sys, '_MEIPASS'):
            # Use the directory containing the executable as application root
            return Path(sys.executable).parent
        return Path(sys.executable).parent
    else:
        # Running from source - find repository root
        anchor = Path(__file__).resolve()
        for parent in anchor.parents:
            if parent.name == "src":
                return parent.parent
        # Fallback: go up 3 levels from config module
        return anchor.parents[3]


@dataclass(frozen=True)
class AppPaths:
    """Resolved filesystem roots used by the webapp.

    All data paths now use UserPaths (~/.humanpose3d) for platform independence
    and proper separation between application code and user data.
    """

    repo_root: Path
    output_root: Path
    upload_root: Path
    models_root: Path
    models_checkpoints: Path

    @classmethod
    def from_anchor(cls, anchor: Path) -> "AppPaths":
        """Resolve application roots using UserPaths for data directories.

        IMPORTANT: This now uses ~/.humanpose3d for all data, making the
        application portable and PyInstaller-friendly.
        """
        repo_root = get_application_root()
        user_paths = UserPaths.default()

        return cls(
            repo_root=repo_root,
            output_root=user_paths.data_output,
            upload_root=user_paths.data_input,
            models_root=user_paths.models,
            models_checkpoints=user_paths.models_checkpoints,
        )

    @classmethod
    def default(cls) -> "AppPaths":
        """Create AppPaths with default configuration."""
        return cls.from_anchor(Path(__file__))
