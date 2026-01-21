"""Service for downloading pre-trained model files from git repository.

Platform-independent implementation using pathlib and subprocess.
"""

from __future__ import annotations

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Callable


def _check_git_available() -> tuple[bool, str]:
    """Check if git is available on the system (platform-independent).

    Returns:
        (is_available, git_path_or_error_message)
    """
    git_command = "git"
    try:
        result = subprocess.run(
            [git_command, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, git_command
        return False, "Git is installed but not functioning correctly"
    except FileNotFoundError:
        return False, "Git is not installed or not in PATH"
    except subprocess.TimeoutExpired:
        return False, "Git command timed out"
    except Exception as e:
        return False, f"Error checking git: {str(e)}"


class ModelDownloadService:
    """Downloads model files from the git repository's models branch."""

    REPO_URL = "https://github.com/JoeyKardolus/humanpose3d_backend.git"
    MODELS_BRANCH = "models"

    def __init__(self, target_dir: Path):
        """
        Initialize the download service.

        Args:
            target_dir: Directory where models should be downloaded (~/.humanpose3d)
        """
        self.target_dir = target_dir

    def download_models(
        self, progress_callback: Callable[[str], None] | None = None
    ) -> tuple[bool, str]:
        """
        Download model files from the git repository (platform-independent).

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (success: bool, message: str)
        """
        # Check if git is available
        git_available, git_msg = _check_git_available()
        if not git_available:
            return False, f"Cannot download models: {git_msg}"

        try:
            self._update_progress(progress_callback, "Creating temporary directory...")

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                self._update_progress(
                    progress_callback,
                    f"Cloning models from {self.REPO_URL}..."
                )

                # Clone only the models branch with minimal depth
                result = subprocess.run(
                    [
                        "git",
                        "clone",
                        "--branch",
                        self.MODELS_BRANCH,
                        "--depth",
                        "1",
                        "--single-branch",
                        self.REPO_URL,
                        str(tmp_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes timeout
                )

                if result.returncode != 0:
                    return False, f"Git clone failed: {result.stderr}"

                self._update_progress(progress_callback, "Copying model files...")

                # Copy models directory
                src_models = tmp_path / "models"
                if not src_models.exists():
                    return False, "Models directory not found in repository"

                # Ensure target directory exists
                self.target_dir.mkdir(parents=True, exist_ok=True)

                # Copy models directory
                dst_models = self.target_dir / "models"
                if dst_models.exists():
                    shutil.rmtree(dst_models)
                shutil.copytree(src_models, dst_models)

                self._update_progress(progress_callback, "Download complete!")

                return True, "Models downloaded successfully"

        except subprocess.TimeoutExpired:
            return False, "Download timed out. Please check your internet connection."
        except (OSError, IOError) as e:
            return False, f"File system error during download: {str(e)}"
        except subprocess.SubprocessError as e:
            return False, f"Git operation failed: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error during download: {str(e)}"

    def _update_progress(
        self, callback: Callable[[str], None] | None, message: str
    ) -> None:
        """Send progress update if callback is provided."""
        if callback:
            callback(message)
