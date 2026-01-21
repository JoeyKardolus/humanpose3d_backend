"""Management command to download pre-trained model files."""

from __future__ import annotations

import sys
from django.core.management.base import BaseCommand

from src.application.config.user_paths import UserPaths
from src.application.services.model_download_service import ModelDownloadService


class Command(BaseCommand):
    help = "Download pre-trained model files from GitHub to ~/.humanpose3d/models"  # noqa: A003

    def add_arguments(self, parser) -> None:  # type: ignore[override]
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-download even if models already exist",
        )
        parser.add_argument(
            "--check-only",
            action="store_true",
            help="Only check if models exist, don't download",
        )

    def handle(self, *args, **options) -> None:  # type: ignore[override]
        user_paths = UserPaths.default()

        self.stdout.write("=" * 70)
        self.stdout.write(self.style.SUCCESS("HumanPose3D Model Downloader"))
        self.stdout.write("=" * 70)
        self.stdout.write(f"\nTarget directory: {user_paths.base}")
        self.stdout.write(f"Models directory: {user_paths.models}\n")

        # Check if models already exist
        models_exist = user_paths.models_exist()

        if options["check_only"]:
            self._check_models(user_paths, models_exist)
            return

        if models_exist and not options["force"]:
            self.stdout.write(self.style.SUCCESS("\n✓ All required models are already present!"))
            self.stdout.write("\nModel files:")
            self._list_models(user_paths)
            self.stdout.write(
                "\nTo re-download models, use: python manage.py download_models --force"
            )
            return

        if models_exist and options["force"]:
            self.stdout.write(
                self.style.WARNING("\n⚠ Models exist but --force specified, re-downloading...")
            )

        # Ensure directory exists
        user_paths.ensure_directories()

        # Create download service
        service = ModelDownloadService(user_paths.base)

        # Download models with progress
        self.stdout.write("\nStarting download...\n")

        def progress_callback(message: str) -> None:
            self.stdout.write(f"  {message}")

        success, message = service.download_models(progress_callback)

        self.stdout.write("")  # Blank line

        if success:
            self.stdout.write(self.style.SUCCESS(f"✓ {message}"))
            self.stdout.write("\nDownloaded files:")
            self._list_models(user_paths)
            self.stdout.write(
                "\n" + self.style.SUCCESS("Models are ready! You can now run the pipeline.")
            )
        else:
            self.stdout.write(self.style.ERROR(f"✗ {message}"))
            self.stdout.write(
                "\n" + self.style.ERROR("Download failed. Please check the error message above.")
            )
            sys.exit(1)

    def _check_models(self, user_paths: UserPaths, models_exist: bool) -> None:
        """Check and report model status without downloading."""
        self.stdout.write("\nChecking model files...\n")

        required_models = [
            ("Depth refinement model", user_paths.models_checkpoints / "best_depth_model.pth"),
            ("Joint refinement model", user_paths.models_checkpoints / "best_joint_model.pth"),
            ("Main refiner model", user_paths.models_checkpoints / "best_main_refiner.pth"),
            ("MediaPipe pose model", user_paths.models / "pose_landmarker_heavy.task"),
            ("Pose2Sim LSTM model", user_paths.models / "GRU.h5"),
        ]

        all_present = True
        for name, path in required_models:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                self.stdout.write(
                    self.style.SUCCESS(f"  ✓ {name}: {path.name} ({size_mb:.2f} MB)")
                )
            else:
                self.stdout.write(self.style.ERROR(f"  ✗ {name}: {path.name} (missing)"))
                all_present = False

        self.stdout.write("")  # Blank line

        if all_present:
            self.stdout.write(self.style.SUCCESS("All required models are present."))
        else:
            self.stdout.write(
                self.style.WARNING("Some models are missing. Run without --check-only to download.")
            )
            sys.exit(1)

    def _list_models(self, user_paths: UserPaths) -> None:
        """List all downloaded model files with sizes."""
        models = [
            user_paths.models_checkpoints / "best_depth_model.pth",
            user_paths.models_checkpoints / "best_joint_model.pth",
            user_paths.models_checkpoints / "best_main_refiner.pth",
            user_paths.models / "pose_landmarker_heavy.task",
            user_paths.models / "GRU.h5",
        ]

        for model_path in models:
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                self.stdout.write(f"  - {model_path.name} ({size_mb:.2f} MB)")
