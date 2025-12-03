# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

 uv run python main.py \
    --video data/input/joey.mp4 \
    --height 1.78 \
    --mass 75 \
    --age 30 \
    --sex male \
    --gaussian-smooth 2.5 \
    --flk-filter \
    --anatomical-constraints \
    --estimate-missing \
    --augmentation-cycles 25

## Project Overview

HumanPose is a 3D human pose estimation pipeline that uses MediaPipe for landmark detection and Pose2Sim for marker augmentation. The pipeline processes video input through multiple stages: capture/detection, CSV export, TRC conversion, and augmentation to produce biomechanics-compatible output files.

## Build and Development Commands

### Environment Setup
- `uv sync` - Install Python 3.12 toolchain and dependencies from pyproject.toml/uv.lock

### Running the Pipeline
```bash
uv run python main.py \
  --video data/input/<name>.mp4 \
  --height <meters> \
  --mass <kg> \
  --age <years> \
  --sex <male|female>
```

Outputs are written to `data/output/pose-3d/<basename>/`

**Note**: The pipeline now defaults to 20 augmentation cycles with 0.3 visibility threshold for best quality/speed balance. Adjust with `--augmentation-cycles N` as needed.

### Testing
- `uv run pytest` - Run all tests
- `uv run pytest -k <module_name>` - Run tests for specific module
- Test files mirror source structure under `tests/`

### Development Tools
- `uv run jupyter lab notebooks/` - Launch Jupyter notebooks with repo packages on path
- `uv run python -m black src tests` - Format code before committing

## Pipeline Architecture
   https://spotify.link/6xdMvgL2IYb 
The pipeline follows a strict 5-step orchestration model in `main.py`:

1. **Step 1 - Extraction** (`src/posedetector/`): MediaPipe Pose processes video frames and extracts world landmarks (33 points mapped to 22 Pose2Sim-aligned markers via `POSE_NAME_MAP`). Outputs CSV with columns: timestamp_s, landmark, x_m, y_m, z_m, visibility.

2. **Step 1.5 - ISB Optimization** (optional, `--isb-opt`): External script can rotate/optimize CSV coordinates before TRC conversion.

3. **Step 2 - TRC Conversion** (`src/datastream/`): Converts CSV to TRC format using `ORDER_22` marker ordering. Derives synthetic markers (Hip, Neck) from parent landmarks (Hip = mean(LHip, RHip), Neck = mean(LShoulder, RShoulder)) per `DERIVED_PARENTS`. Missing landmarks remain empty - no placeholders or duplicates.

4. **Step 3 - Augmentation** (`src/markeraugmentation/`): Invokes Pose2Sim's `augment_markers_all` multiple times (default 30 cycles) via CLI or Python shim to add full OpenCap marker set using LSTM prediction. Multi-cycle averaging with small perturbations (1mm Gaussian noise per cycle) significantly improves marker completion rates. This adds **43 additional anatomical markers** (shoulder clusters, thigh clusters, medial/lateral joint markers, hip joint centers) to the original 22 markers, creating a 65-marker dataset. Outputs `<basename>_LSTM.trc` with 65 markers in data but only 22 listed in header. Intermediate cycle files are automatically cleaned up after averaging.

5. **Step 4 - Header Fix** (optional, `--fix-header`): Adjusts TRC metadata (NumMarkers, marker labels) to match actual data without inventing markers.

## Key Implementation Details

### Strict Mode Rules
- All landmark processing follows "strict mode": no placeholder values, no duplicate markers, explicit visibility thresholds (default 0.3)
- Derived markers require both parents present; missing data stays blank
- CSV rows are sorted by (timestamp, landmark) for deterministic output
- Each run appends to `docs/BUILD_LOG.md` via `append_build_log()`

### Marker Estimation Pipeline
**Problem**: MediaPipe may miss landmarks (right arm occluded, head out of frame, etc.), causing Pose2Sim augmentation to fail completely.

**Solution**: Two-stage estimation approach:

1. **Pre-augmentation** (`--estimate-missing`): Fills missing INPUT markers before Pose2Sim
   - Mirrors missing right arm from left arm (using body symmetry)
   - Extrapolates Head from Nose and Neck vectors
   - Estimates SmallToes from BigToe-Heel geometry
   - Implemented in `src/datastream/marker_estimation.py`
   - Significantly improves LSTM augmentation success (0% → 81% with incomplete data)

2. **Post-augmentation** (`--force-complete`): Fills missing OUTPUT markers after Pose2Sim
   - Estimates shoulder clusters (r_sh1-3, L_sh1-3) from shoulder-elbow vectors
   - Calculates hip joint centers (RHJC, LHJC) using Bell et al. 1990 regression
   - Implemented in `src/datastream/post_augmentation_estimation.py`
   - Targets the 8 markers LSTM often skips (81% → 100%)

**Recommended**: Always use `--estimate-missing` for incomplete pose detection. Use `--force-complete` when you need all 43 augmented markers for biomechanical analysis.

### Module Responsibilities
- **`mediastream/`** - Video I/O using OpenCV, returns RGB frames + FPS
- **`posedetector/`** - MediaPipe inference, maps 33 landmarks → 22 markers via `POSE_NAME_MAP`
- **`datastream/`** - CSV/TRC writers, marker estimation:
  - Implements ORDER_22, DERIVED_PARENTS logic for TRC conversion
  - `marker_estimation.py` - Pre-augmentation estimation using anatomical symmetry
  - `post_augmentation_estimation.py` - Post-augmentation completion for shoulder clusters and HJC
- **`markeraugmentation/`** - Pose2Sim integration, creates temp project structure, resolves CLI via POSE2SIM_CMD env var or local .venv/bin/pose2sim
- **`visualizedata/`** - 3D Matplotlib plotting for landmarks/TRC with auto-detection of marker sets:
  - Reads ALL markers from TRC data (not just header - critical for augmented files where header only lists 22 but data contains 65)
  - Auto-selects skeleton connections: `OPENCAP_CONNECTIONS` (33 connections) for 65-marker augmented data, `MEDIAPIPE_CONNECTIONS` for 22-33 marker data
  - OpenCap connections based on actual marker availability (not all 43 augmented markers may have data)
  - Uses anatomically correct connections: full limb chains (shoulder→elbow→wrist→toes), pelvis structure, augmented joint markers
  - Exports MP4 (ffmpeg) or GIF (Pillow) fallback
  - Interactive mode enabled by default (TkAgg backend)
- **`application/`** - Build logging and step scripts (mostly superseded by main.py orchestration)

### Data Flow
```
video → frames (mediastream)
      → landmarks (posedetector)
      → CSV records (datastream)
      → TRC (datastream)
      → augmented TRC (markeraugmentation)
      → [fixed TRC] (datastream)
```

### Important Flags
- `--show-video` - Renders MediaPipe preview window + exports `<name>_preview.mp4` (requires Qt/XCB + ffmpeg)
- `--plot-landmarks` - Displays extracted CSV landmarks in 3D Matplotlib viewer
- `--plot-augmented` - Visualizes augmented TRC and exports `<name>_LSTM_preview.mp4`
- `--visibility-min` - Threshold for landmark export (default 0.3 for better coverage)
- `--estimate-missing` - Estimates missing markers using anatomical symmetry before augmentation (recommended for incomplete poses)
- `--force-complete` - Post-processes augmented TRC to estimate shoulder clusters and hip joint centers (optional)
- `--augmentation-cycles` - Number of augmentation cycles to run and average (default 20 for best results)

## ORDER_22 Marker Set
The pipeline uses a fixed 22-marker layout aligned with Pose2Sim's OpenCap expectations:

Neck, RShoulder, LShoulder, RHip, LHip, RKnee, LKnee, RAnkle, LAnkle, RHeel, LHeel, RSmallToe, LSmallToe, RBigToe, LBigToe, RElbow, LElbow, RWrist, LWrist, Hip, Head, Nose

MediaPipe provides direct mappings for most landmarks (shoulders, elbows, wrists, hips, knees, ankles, heels, foot_index→BigToe, nose). Neck and Hip are derived from shoulder/hip pairs. Some markers (SmallToe, Head) may remain empty depending on visibility.

## Testing Guidelines
- Use `tmp_path` fixtures for file I/O tests
- Prefer deterministic fixtures from `data/input/tests/` for integration tests
- Mock MediaPipe/OpenCV calls for heavy operations
- Golden CSVs/TRCs should be documented in PR descriptions
- Coverage priorities: parsing, error handling, visualization helpers

## Interactive Visualization

### Quick Start
```bash
uv run python visualize_interactive.py [path/to/file.trc]
```

If no path provided, defaults to `data/output/pose-3d/joey/joey_LSTM.trc`

### Controls
- **Mouse drag**: Rotate 3D view
- **Slider**: Navigate through frames
- **Play/Pause button**: Animate playback
- **Close window**: Exit viewer

### Marker Count Detection
- Automatically detects 65-marker (augmented) vs 22-marker (non-augmented) data
- Uses `OPENCAP_CONNECTIONS` for augmented files with proper anatomical skeleton
- Uses `MEDIAPIPE_CONNECTIONS` for non-augmented files

## Recommended Workflow

### For Best Augmentation Results (Default)
```bash
uv run python main.py \
  --video data/input/<video>.mp4 \
  --height <meters> \
  --mass <kg> \
  --age <years> \
  --sex <male|female> \
  --estimate-missing \
  --force-complete
```

This pipeline (with default 20 cycles):
1. Extracts landmarks with MediaPipe (visibility threshold 0.3)
2. Estimates any missing markers using anatomical symmetry (optional)
3. Converts to TRC with ORDER_22
4. Runs 20 augmentation cycles with small perturbations (0.5mm)
5. Averages all cycles for smooth, complete markers
6. Post-processes remaining gaps if `--force-complete` is used
7. Outputs `*_LSTM.trc` (or `*_complete.trc` if force-complete)
8. Auto-cleans intermediate cycle files

**Performance**: ~30-45 seconds total on modern hardware

### For Quick Testing (Single Cycle)
```bash
uv run python main.py \
  --video data/input/<video>.mp4 \
  --height 1.78 \
  --mass 75.0 \
  --age 30 \
  --sex male \
  --augmentation-cycles 1 \
  --estimate-missing
```

**Performance**: ~10-15 seconds

### For Visualization Only
```bash
uv run python main.py \
  --video data/input/<video>.mp4 \
  --height 1.78 \
  --mass 75.0 \
  --age 30 \
  --sex male \
  --plot-augmented
```

### For Quick Checks
```bash
# Check marker counts
python3 -c "
from pathlib import Path
import numpy as np
from src.visualizedata.visualize_data import VisualizeData
viz = VisualizeData()
_, frames = viz.load_trc_frames(Path('your_file.trc'))
frame = frames[0]
augmented = sum(1 for i in range(22, 65) if not np.isnan(frame[i]).any())
print(f'Augmented: {augmented}/43')
"
```

## Known Constraints & Troubleshooting

### Pose2Sim Augmentation Improvements
- **Multi-cycle averaging**: Default 20 cycles with small perturbations (0.5mm) and averaging for consistent results
- **Visibility threshold**: 0.3 default (lower = more data, more guessing allowed)
- **Typical results**: 100% completion of augmented markers that Pose2Sim generates (typically 35/43 markers with estimation)
- **Performance**: 20 cycles completes in ~30-45 seconds on modern hardware
- **Use `--estimate-missing`** for incomplete poses (occluded limbs, partial views)

### TRC File Structure Quirk
- Augmented TRC files have **header mismatch**: header lists 22 markers but data contains 65 markers
- This is normal Pose2Sim behavior - augmented markers are appended to data but not header
- `VisualizeData.load_trc_frames()` handles this by reading actual data columns, not header
- Custom TRC readers must count data columns (197 = 2 + 65*3) not header entries (68 = 2 + 22*3)

### Visualization
- **Interactive mode**: Default TkAgg backend opens GUI window (drag to rotate, slider to navigate frames)
- **Skeleton structure**: Clean connections - toe→ankle→knee→hip→spine→shoulders, plus arms
- **Auto-detection**: Uses OpenCap connections for 50+ markers, MediaPipe connections for <50 markers
- **Headless mode**: Set `MPLBACKEND=Agg` env var for non-interactive rendering
- **MP4 export**: Requires ffmpeg; falls back to GIF (larger files, lower quality)
- **GUI features** (--show-video, --plot-*): Require Qt/XCB libs on Linux

### Common Issues
- **"No trc files found"**: Pose2Sim can't find input - check project_dir structure in Config.toml
- **Right arm missing**: Camera angle issue - MediaPipe can't detect occluded limbs; use `--estimate-missing` to mirror from left
- **Hip appears stuck**: Derived Hip marker averages LHip/RHip which dampens movement; use individual hip markers or RHJC/LHJC for analysis
- **Head marker very far**: Nose-Neck extrapolation unreliable when head tilted; Head marker often empty in MediaPipe output

## Code Style
- Follow PEP 8 with 4-space indentation, snake_case names
- Type-hint function signatures (see `main.py`, `data_stream.py`)
- Modules expose single public class (MediaStream, PoseDetector, VisualizeData)
- Concise docstrings for public methods; inline comments for non-obvious logic
- Raise clear exceptions on invalid input

## Commit Guidelines
- Imperative, scoped messages: `mediastream: validate fps metadata`
- Keep unrelated changes in separate commits
- PRs need summary, validation notes (commands run), issue refs, screenshots/GIFs for viz changes
- CI (pytest + lint) must pass
