"""
Setup script for VideoPose3D pretrained model integration.

Downloads pretrained weights from Facebook Research and sets up the model
for 2D-to-3D pose lifting with temporal convolutions.

Reference: https://github.com/facebookresearch/VideoPose3D
"""

from pathlib import Path
import urllib.request
import hashlib

# Model URLs from Facebook Research
PRETRAINED_MODELS = {
    "h36m_cpn": {
        "url": "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin",
        "sha256": "b3865c7eff38ff182b8a269259436728487e802edcdb65a3eb4f843341e2d692",
        "description": "Human3.6M model with CPN 2D detections (243-frame receptive field)",
        "receptive_field": 243,
        "num_joints": 17,
    },
    "h36m_detectron": {
        "url": "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin",
        "sha256": None,  # Optional verification
        "description": "Human3.6M model with Detectron COCO 2D detections",
        "receptive_field": 81,
        "num_joints": 17,
    },
}


def download_file_with_progress(url: str, dest_path: Path):
    """Download file with progress bar."""

    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}%", end="", flush=True)

    print(f"Downloading from: {url}")
    print(f"Saving to: {dest_path}")

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print("\nDownload complete!")


def verify_checksum(file_path: Path, expected_sha256: str | None) -> bool:
    """Verify file SHA256 checksum."""
    if expected_sha256 is None:
        print("No checksum provided, skipping verification")
        return True

    print("Verifying checksum...")
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    actual = sha256.hexdigest()
    if actual == expected_sha256:
        print("✅ Checksum verified")
        return True
    else:
        print(f"❌ Checksum mismatch!")
        print(f"   Expected: {expected_sha256}")
        print(f"   Got:      {actual}")
        return False


def setup_videopose3d(model_name: str = "h36m_cpn", force_download: bool = False):
    """
    Download and setup VideoPose3D pretrained model.

    Args:
        model_name: Model to download ('h36m_cpn' or 'h36m_detectron')
        force_download: Re-download even if file exists
    """
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(PRETRAINED_MODELS.keys())}")

    model_config = PRETRAINED_MODELS[model_name]

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoint")
    checkpoint_dir.mkdir(exist_ok=True)

    # Determine output path
    model_path = checkpoint_dir / f"{model_name}.bin"

    print("=" * 80)
    print("VideoPose3D Model Setup")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Description: {model_config['description']}")
    print(f"Receptive field: {model_config['receptive_field']} frames")
    print(f"Joints: {model_config['num_joints']}")
    print()

    # Check if already exists
    if model_path.exists() and not force_download:
        print(f"✅ Model already exists at: {model_path}")
        print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        print()
        print("Use --force to re-download")
        return model_path

    # Download model
    try:
        download_file_with_progress(model_config["url"], model_path)

        # Verify checksum if provided
        if model_config["sha256"]:
            if not verify_checksum(model_path, model_config["sha256"]):
                print("Warning: Checksum verification failed!")
                print("The downloaded file may be corrupted.")
                return None

        print()
        print("=" * 80)
        print("✅ Setup Complete!")
        print("=" * 80)
        print(f"Model saved to: {model_path}")
        print(f"Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        print()
        print("Next steps:")
        print("1. Install PyTorch: pip install torch torchvision")
        print("2. Enable VideoPose3D in main.py with --use-videopose3d flag")
        print("3. See docs/VIDEOPOSE3D_SETUP.md for integration details")

        return model_path

    except Exception as e:
        print(f"❌ Download failed: {e}")
        if model_path.exists():
            model_path.unlink()  # Clean up partial download
        return None


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download VideoPose3D pretrained models")
    parser.add_argument(
        "--model",
        choices=list(PRETRAINED_MODELS.keys()),
        default="h36m_cpn",
        help="Model to download (default: h36m_cpn for best quality)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists",
    )

    args = parser.parse_args()

    model_path = setup_videopose3d(args.model, args.force)

    if model_path:
        print(f"\n✅ Success! Model ready at: {model_path}")
    else:
        print("\n❌ Setup failed")
        exit(1)


if __name__ == "__main__":
    main()
