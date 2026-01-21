"""Resource path resolution for PyInstaller compatibility.

This module provides utilities for accessing bundled resources (like model files)
in both development and PyInstaller-bundled environments.
"""

from __future__ import annotations

import sys
from pathlib import Path


def get_resource_path(relative_path: str | Path) -> Path:
    """
    Get absolute path to resource, works for dev and for PyInstaller bundles.

    When running from source, returns path relative to repository root.
    When frozen with PyInstaller, returns path to extracted temporary directory.

    Args:
        relative_path: Path relative to application root (e.g., "models/pose_landmarker_heavy.task")

    Returns:
        Absolute path to the resource

    Example:
        >>> model_path = get_resource_path("models/pose_landmarker_heavy.task")
        >>> print(model_path)
        /tmp/_MEIxxxxxx/models/pose_landmarker_heavy.task  # When bundled
        /home/user/project/models/pose_landmarker_heavy.task  # When in dev
    """
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = Path(sys._MEIPASS)
        else:
            # Fallback: use executable directory
            base_path = Path(sys.executable).parent
    else:
        # Running in development mode
        # Find repository root (contains 'src' directory)
        anchor = Path(__file__).resolve()
        for parent in anchor.parents:
            if (parent / "src").exists():
                base_path = parent
                break
        else:
            # Fallback: go up to grandparent
            base_path = anchor.parent.parent.parent

    return (base_path / relative_path).resolve()


def is_frozen() -> bool:
    """
    Check if the application is running as a PyInstaller bundle.

    Returns:
        True if running from PyInstaller bundle, False if running from source
    """
    return getattr(sys, 'frozen', False)


def get_temp_extraction_path() -> Path | None:
    """
    Get PyInstaller's temporary extraction directory if running frozen.

    Returns:
        Path to PyInstaller temp directory, or None if not frozen
    """
    if is_frozen() and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)
    return None
