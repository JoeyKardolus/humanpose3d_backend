"""Test suite for platform-independent path configuration."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

from src.application.config.user_paths import UserPaths
from src.application.config.paths import AppPaths, get_application_root


class TestUserPaths:
    """Test user data paths in home directory."""

    def test_default_paths_use_home_directory(self):
        """Verify all paths are under user home directory."""
        user_paths = UserPaths.default()

        # All paths should be under home directory
        assert user_paths.base.is_relative_to(Path.home())
        assert user_paths.models.is_relative_to(Path.home())
        assert user_paths.models_checkpoints.is_relative_to(Path.home())
        assert user_paths.data_input.is_relative_to(Path.home())
        assert user_paths.data_output.is_relative_to(Path.home())

    def test_paths_are_absolute(self):
        """Verify all paths are absolute, not relative."""
        user_paths = UserPaths.default()

        assert user_paths.base.is_absolute()
        assert user_paths.models.is_absolute()
        assert user_paths.models_checkpoints.is_absolute()
        assert user_paths.data_input.is_absolute()
        assert user_paths.data_output.is_absolute()

    def test_path_structure(self):
        """Verify correct path hierarchy."""
        user_paths = UserPaths.default()
        base = Path.home() / ".humanpose3d"

        assert user_paths.base == base
        assert user_paths.models == base / "models"
        assert user_paths.models_checkpoints == base / "models" / "checkpoints"
        assert user_paths.data_input == base / "data" / "input"
        assert user_paths.data_output == base / "data" / "output"

    def test_ensure_directories_creates_all_paths(self, tmp_path, monkeypatch):
        """Verify ensure_directories creates all required directories."""
        # Use tmp_path instead of real home directory
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()

        def fake_home_func():
            return fake_home

        monkeypatch.setattr(Path, "home", fake_home_func)

        user_paths = UserPaths.default()
        user_paths.ensure_directories()

        # All directories should exist
        assert user_paths.base.exists()
        assert user_paths.models.exists()
        assert user_paths.models_checkpoints.exists()
        assert user_paths.data_input.exists()
        assert user_paths.data_output.exists()

    def test_models_exist_check(self, tmp_path, monkeypatch):
        """Verify models_exist correctly checks for required files."""
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()

        def fake_home_func():
            return fake_home

        monkeypatch.setattr(Path, "home", fake_home_func)

        user_paths = UserPaths.default()
        user_paths.ensure_directories()

        # Initially no models
        assert not user_paths.models_exist()

        # Create model files
        model_files = [
            user_paths.models_checkpoints / "best_depth_model.pth",
            user_paths.models_checkpoints / "best_joint_model.pth",
            user_paths.models_checkpoints / "best_main_refiner.pth",
            user_paths.models / "pose_landmarker_heavy.task",
            user_paths.models / "GRU.h5",
        ]

        for model_file in model_files:
            model_file.touch()

        # Now all models should exist
        assert user_paths.models_exist()


class TestAppPaths:
    """Test application root and data paths."""

    def test_app_paths_uses_user_paths_for_data(self):
        """Verify AppPaths uses UserPaths for all data directories."""
        app_paths = AppPaths.default()
        user_paths = UserPaths.default()

        # Data paths should match UserPaths
        assert app_paths.output_root == user_paths.data_output
        assert app_paths.upload_root == user_paths.data_input
        assert app_paths.models_root == user_paths.models
        assert app_paths.models_checkpoints == user_paths.models_checkpoints

    def test_repo_root_is_absolute(self):
        """Verify repo_root is an absolute path."""
        app_paths = AppPaths.default()
        assert app_paths.repo_root.is_absolute()

    def test_get_application_root_in_development(self):
        """Verify application root detection when running from source."""
        # Should not be frozen in tests
        assert not getattr(sys, 'frozen', False)

        root = get_application_root()
        assert root.is_absolute()
        # Should find src directory in parents
        assert (root / "src").exists()


class TestPlatformIndependence:
    """Test cross-platform compatibility."""

    def test_no_hardcoded_separators(self):
        """Verify paths use pathlib, not string concatenation."""
        user_paths = UserPaths.default()

        # Convert to string and check - should work on all platforms
        base_str = str(user_paths.base)
        models_str = str(user_paths.models)

        # Path should contain platform separator
        if sys.platform == "win32":
            assert "\\" in base_str or "/" in base_str
        else:
            assert "/" in base_str

        # Models path should be longer than base (has subdirectory)
        assert len(models_str) > len(base_str)

    def test_pathlib_operations_work(self):
        """Verify pathlib operations are platform-independent."""
        user_paths = UserPaths.default()

        # Test common operations
        test_path = user_paths.data_input / "test_video.mp4"
        assert test_path.parent == user_paths.data_input
        assert test_path.name == "test_video.mp4"
        assert test_path.suffix == ".mp4"
        assert test_path.stem == "test_video"

    def test_path_resolution_is_platform_independent(self):
        """Verify path resolution works on all platforms."""
        user_paths = UserPaths.default()

        # Resolve should work
        resolved = user_paths.base.resolve()
        assert resolved.is_absolute()

        # relative_to should work
        try:
            rel = user_paths.models.relative_to(user_paths.base)
            assert str(rel) == "models"
        except ValueError:
            pytest.fail("relative_to failed - paths not properly related")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
