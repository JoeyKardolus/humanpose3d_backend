#!/usr/bin/env python3
"""Standalone script to download pre-trained model files.

This script downloads all required model files from GitHub to ~/.humanpose3d/models/
and works without requiring Django or other dependencies.

Usage:
    python scripts/download_models.py              # Download models
    python scripts/download_models.py --check      # Check if models exist
    python scripts/download_models.py --force      # Re-download even if exist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.config.user_paths import UserPaths
from src.application.services.model_download_service import ModelDownloadService


def print_header() -> None:
    """Print script header."""
    print("=" * 70)
    print("HumanPose3D Model Downloader")
    print("=" * 70)


def list_models(user_paths: UserPaths) -> None:
    """List all model files with sizes."""
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
            print(f"  - {model_path.name} ({size_mb:.2f} MB)")


def check_models(user_paths: UserPaths) -> bool:
    """Check if all required models exist and print status.

    Returns:
        True if all models exist, False otherwise
    """
    print("\nChecking model files...\n")

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
            print(f"  ✓ {name}: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {name}: {path.name} (missing)")
            all_present = False

    print()  # Blank line
    return all_present


def download_models(user_paths: UserPaths, force: bool = False) -> bool:
    """Download model files from GitHub.

    Args:
        user_paths: User paths configuration
        force: Force re-download even if models exist

    Returns:
        True if successful, False otherwise
    """
    # Check if models already exist
    models_exist = user_paths.models_exist()

    if models_exist and not force:
        print("\n✓ All required models are already present!")
        print("\nModel files:")
        list_models(user_paths)
        print("\nTo re-download models, use: --force")
        return True

    if models_exist and force:
        print("\n⚠ Models exist but --force specified, re-downloading...")

    # Ensure directories exist
    user_paths.ensure_directories()

    # Create download service
    service = ModelDownloadService(user_paths.base)

    # Download with progress
    print("\nStarting download...\n")

    def progress_callback(message: str) -> None:
        print(f"  {message}")

    success, message = service.download_models(progress_callback)

    print()  # Blank line

    if success:
        print(f"✓ {message}")
        print("\nDownloaded files:")
        list_models(user_paths)
        print("\n✓ Models are ready! You can now run the pipeline.")
        return True
    else:
        print(f"✗ {message}")
        print("\n✗ Download failed. Please check the error message above.")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download pre-trained model files for HumanPose3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                Download models if not present
  %(prog)s --check        Check model status without downloading
  %(prog)s --force        Force re-download all models

Models are downloaded to: ~/.humanpose3d/models/
  Windows: C:\\Users\\<username>\\.humanpose3d\\models
  macOS:   /Users/<username>/.humanpose3d/models
  Linux:   /home/<username>/.humanpose3d/models

Total download size: ~121 MB (5 model files)
        """,
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if models exist, don't download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if models already exist",
    )

    args = parser.parse_args()

    # Print header
    print_header()

    # Get user paths
    user_paths = UserPaths.default()

    print(f"\nTarget directory: {user_paths.base}")
    print(f"Models directory: {user_paths.models}")

    # Check-only mode
    if args.check:
        all_present = check_models(user_paths)
        if all_present:
            print("All required models are present.")
            return 0
        else:
            print("Some models are missing. Run without --check to download.")
            return 1

    # Download mode
    success = download_models(user_paths, force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
