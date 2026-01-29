#!/usr/bin/env python3
"""Add camera_R to existing training data NPZ files.

Loads camera rotation matrix from AIST++ and adds it to each NPZ file.
Much faster than regenerating everything.

Usage:
    uv run python scripts/data/add_camera_R.py --workers 8
"""

import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from scipy.spatial.transform import Rotation

# AIST++ paths
AIST_ROOT = Path("data/AIST++")
CAMERAS_DIR = AIST_ROOT / "annotations" / "cameras"
MAPPING_FILE = CAMERAS_DIR / "mapping.txt"
OUTPUT_DIR = Path("data/training/aistpp_converted")


def load_sequence_mapping() -> dict:
    """Load sequence -> setting mapping."""
    mapping = {}
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1]
    return mapping


def load_all_cameras() -> dict:
    """Load all camera settings into memory."""
    all_cameras = {}
    for setting_file in CAMERAS_DIR.glob("setting*.json"):
        setting_name = setting_file.stem
        with open(setting_file, 'r') as f:
            cameras = json.load(f)

        all_cameras[setting_name] = {}
        for cam in cameras:
            # Rodrigues to rotation matrix
            rvec = np.array(cam['rotation'])
            R = Rotation.from_rotvec(rvec).as_matrix().astype(np.float32)
            all_cameras[setting_name][cam['name']] = R

    return all_cameras


# Global cache (loaded once per worker)
_sequence_mapping = None
_all_cameras = None


def init_worker():
    """Initialize worker with cached data."""
    global _sequence_mapping, _all_cameras
    _sequence_mapping = load_sequence_mapping()
    _all_cameras = load_all_cameras()


def get_sequence_name(filename: str) -> str:
    """Extract sequence name from filename (without camera and frame)."""
    # Format: gBR_sBM_cAll_d04_mBR0_ch01_c03_f000021.npz
    # Sequence: gBR_sBM_cAll_d04_mBR0_ch01
    parts = filename.split('_')
    # Find index of camera part (c01, c02, etc.)
    for i, part in enumerate(parts):
        if part.startswith('c') and part[1:].isdigit() and len(part) == 3:
            return '_'.join(parts[:i])
    return '_'.join(parts[:-1])  # Fallback: everything except frame


def get_camera_name(filename: str) -> str:
    """Extract camera name (c01, c02, etc.) from filename."""
    parts = filename.split('_')
    for part in parts:
        if part.startswith('c') and part[1:].isdigit() and len(part) == 3:
            return part
    return None


def process_file(npz_path: Path) -> tuple:
    """Add camera_R to a single NPZ file."""
    global _sequence_mapping, _all_cameras

    try:
        filename = npz_path.stem

        # Get camera name
        camera_name = get_camera_name(filename)
        if camera_name is None:
            return (npz_path.name, "no camera in filename")

        # Get sequence name and find setting
        seq_name = get_sequence_name(filename)
        setting = _sequence_mapping.get(seq_name, 'setting1')

        # Get rotation matrix
        if setting not in _all_cameras:
            return (npz_path.name, f"setting {setting} not found")
        if camera_name not in _all_cameras[setting]:
            return (npz_path.name, f"camera {camera_name} not in {setting}")

        camera_R = _all_cameras[setting][camera_name]

        # Load existing data
        data = dict(np.load(npz_path))

        # Check if already has camera_R
        if 'camera_R' in data:
            return (npz_path.name, "skip")

        # Add camera_R
        data['camera_R'] = camera_R

        # Save back
        np.savez_compressed(npz_path, **data)

        return (npz_path.name, "ok")
    except Exception as e:
        return (npz_path.name, str(e))


def main():
    parser = argparse.ArgumentParser(description='Add camera_R to existing NPZ files')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--data-dir', type=str, default=str(OUTPUT_DIR), help='Data directory')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    npz_files = list(data_dir.glob('*.npz'))

    print(f"Found {len(npz_files)} NPZ files in {data_dir}")
    print(f"Loading camera data from {CAMERAS_DIR}")

    if not npz_files:
        print("No files to process")
        return

    # Check camera data exists
    if not CAMERAS_DIR.exists():
        print(f"ERROR: Camera directory not found: {CAMERAS_DIR}")
        return

    # Process files
    ok_count = 0
    skip_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
        futures = {executor.submit(process_file, f): f for f in npz_files}

        for i, future in enumerate(as_completed(futures)):
            name, status = future.result()
            if status == "ok":
                ok_count += 1
            elif status == "skip":
                skip_count += 1
            else:
                error_count += 1
                if error_count <= 10:
                    print(f"  Error: {name}: {status}")

            if (i + 1) % 50000 == 0:
                print(f"  Processed {i+1}/{len(npz_files)} (ok={ok_count}, skip={skip_count}, err={error_count})")

    print(f"\nDone: {ok_count} updated, {skip_count} skipped, {error_count} errors")


if __name__ == '__main__':
    main()
