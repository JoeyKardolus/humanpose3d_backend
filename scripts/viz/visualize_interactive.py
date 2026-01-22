#!/usr/bin/env python3
"""Interactive visualization script for augmented TRC files."""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.application.config.paths import StoragePaths
from src.visualizedata.visualize_data import VisualizeData

def main():
    if len(sys.argv) > 1:
        trc_path = Path(sys.argv[1])
    else:
        # Default to complete TRC with corrected HJC
        storage_paths = StoragePaths.load()
        trc_path = storage_paths.output_root / "joey" / "joey_LSTM_complete.trc"

    if not trc_path.exists():
        print(f"Error: TRC file not found: {trc_path}")
        sys.exit(1)

    print(f"Loading TRC file: {trc_path}")
    viz = VisualizeData()
    marker_names, frames = viz.load_trc_frames(trc_path)

    print(f"Loaded {len(marker_names)} markers, {len(frames)} frames")
    print(f"Markers with data: {sum(1 for f in [frames[0]] for i in range(len(marker_names)) if i < len(f) and not any(map(lambda x: x != x, f[i])))}")

    print("\nOpening interactive 3D viewer...")
    print("Use the slider to navigate frames")
    print("Use the Play/Pause button to animate")
    print("Drag with mouse to rotate the view")

    viz.plot_landmarks(frames, export_path=None, block=True)

if __name__ == "__main__":
    main()
