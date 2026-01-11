#!/usr/bin/env python3
"""
Download Human3.6M pre-processed dataset using Python.
"""

import urllib.request
import os
from pathlib import Path

def download_file(url, output_path):
    """Download file with progress."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    try:
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rProgress: {percent:.1f}%", end='', flush=True)

        urllib.request.urlretrieve(url, output_path, progress)
        print("\n✓ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def main():
    print("="*80)
    print("Human3.6M Dataset Download")
    print("="*80)
    print()

    # Create directory
    data_dir = Path("data/human36m")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Try multiple sources
    sources = [
        {
            "name": "VideoPose3D 3D Data",
            "url": "https://dl.fbaipublicfiles.com/video-pose-3d/data_3d_h36m.npz",
            "filename": "h36m_3d.npz"
        },
        {
            "name": "VideoPose3D 2D GT",
            "url": "https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_gt.npz",
            "filename": "h36m_2d_gt.npz"
        },
        {
            "name": "GitHub Release (Alternative)",
            "url": "https://github.com/facebookresearch/VideoPose3D/releases/download/v1.0/data_3d_h36m.npz",
            "filename": "h36m_3d_github.npz"
        }
    ]

    success = False
    for source in sources:
        print(f"\nTrying: {source['name']}")
        output_path = data_dir / source['filename']

        if output_path.exists():
            print(f"✓ Already exists: {output_path}")
            success = True
            continue

        if download_file(source['url'], output_path):
            success = True
        print()

    if success:
        print("="*80)
        print("✓ Download successful!")
        print("="*80)
        print()
        print("Downloaded files:")
        for f in data_dir.glob("*.npz"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
    else:
        print("="*80)
        print("❌ All downloads failed")
        print("="*80)
        print()
        print("Manual download instructions:")
        print("1. Go to: https://github.com/facebookresearch/VideoPose3D")
        print("2. Click 'Releases' → Download data_3d_h36m.npz")
        print("3. Move to: data/human36m/")
        print()
        print("Or register for official dataset:")
        print("http://vision.imar.ro/human3.6m/")


if __name__ == "__main__":
    main()
