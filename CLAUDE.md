# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Recommended Command (Best Quality - Multi-Constraint Optimization + Joint Angles)

```bash
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 \
  --mass 75 \
  --age 30 \
  --sex male \
  --anatomical-constraints \
  --bone-length-constraints \
  --estimate-missing \
  --force-complete \
  --augmentation-cycles 20 \
  --multi-constraint-optimization \
  --compute-all-joint-angles \
  --plot-all-joint-angles
```

**Results** (tested on joey.mp4, 535 frames, 20 cycles):
- ⏱️ Processing Time: **~45 seconds** (with GPU acceleration)
- 📏 Bone Length Improvement: **68.0%** (0.113 → 0.036 CV)
- 🎯 Marker Quality: **59/65 markers** (4 unreliable markers auto-filtered)
- 📊 Joint Angles: **12 joint groups** computed (pelvis, hip, knee, ankle, trunk, shoulder, elbow)
- ✓ No scattered markers - unreliable augmented markers filtered by temporal variance
- ✓ Stable medial markers - distance constraints prevent optimization-induced noise
- ✓ Organized output - automatic cleanup into clean directory structure
- ✓ GPU acceleration - 3-10x speedup on augmentation (automatic CPU fallback if unavailable)

## Joint Angle Computation (ISB-Compliant)

### Comprehensive Joint Angles (ALL Joints - RECOMMENDED)

Compute **ALL** joint angles (pelvis, lower body, trunk, upper body) using ISB-compliant anatomical coordinate systems:

```bash
# Comprehensive: ALL joints (pelvis + lower + trunk + upper body)
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 --mass 75 --age 30 --sex male \
  --anatomical-constraints --bone-length-constraints \
  --estimate-missing --force-complete \
  --augmentation-cycles 20 \
  --multi-constraint-optimization \
  --compute-all-joint-angles \
  --plot-all-joint-angles \
  --save-angle-comparison
```

**Outputs** (automatically organized in `joint_angles/` subdirectory):
- `joint_angles/<video>_angles_pelvis.csv` - Pelvis angles (flex, abd, rot)
- `joint_angles/<video>_angles_hip_{R|L}.csv` - Hip angles (flex, abd, rot)
- `joint_angles/<video>_angles_knee_{R|L}.csv` - Knee angles (flex, abd, rot)
- `joint_angles/<video>_angles_ankle_{R|L}.csv` - Ankle angles (flex, abd, rot)
- `joint_angles/<video>_angles_trunk.csv` - Trunk angles (flex, abd, rot)
- `joint_angles/<video>_angles_shoulder_{R|L}.csv` - Shoulder angles (flex, abd, rot)
- `joint_angles/<video>_angles_elbow_{R|L}.csv` - Elbow flexion angle
- `joint_angles/<video>_all_joint_angles.png` - Comprehensive multi-panel visualization (ALL joints)
- `joint_angles/<video>_joint_angles_comparison.png` - Side-by-side comparison (right vs left, if --save-angle-comparison used)

**Universal Column Naming**: All joints use consistent naming: `{joint}_flex_deg`, `{joint}_abd_deg`, `{joint}_rot_deg`

**Joint Coverage**: 12 joint groups computed using ISB standards
- **Pelvis**: Global orientation relative to world frame
- **Lower Body**: Hip, Knee, Ankle (both sides, 3-DOF each)
- **Trunk**: Thorax relative to pelvis
- **Upper Body**: Shoulder (3-DOF), Elbow (1-DOF), both sides

**Important**: Requires `--force-complete` to estimate Hip Joint Centers (RHJC/LHJC) and shoulder clusters.

### Individual Joint Angle Computation

For specific joints only:

```bash
# Lower limb only (hip, knee, ankle - one side)
uv run python main.py --video data/input/joey.mp4 --force-complete --compute-joint-angles --plot-joint-angles --joint-angle-side R

# Upper body only (trunk, shoulder, elbow - one side)
uv run python main.py --video data/input/joey.mp4 --force-complete --compute-upper-body-angles --plot-upper-body-angles --upper-body-side R
```

**Outputs** (individual):
- `<video>_angles_{R|L}.csv` - Lower limb angles (9 DOF: 3 per joint)
- `<video>_upper_angles_{R|L}.csv` - Upper body angles (7 DOF: trunk + shoulder + elbow)

## Neural Depth Refinement (Advanced/Work in Progress)

For research applications requiring maximum depth accuracy, the pipeline includes tools to generate training data from AIST++ dataset with REAL MediaPipe errors.

### AIST++ Dataset Setup

AIST++ is a large-scale dance motion dataset with synchronized video and 3D keypoints (10.1M annotated frames).

```bash
# 1. Download annotations (2.3GB total, or 834MB for 3D keypoints only):
#    https://google.github.io/aistplusplus_dataset/download.html
#    Extract to data/AIST++/annotations/

# 2. Download videos from AIST Dance Database:
#    https://aistdancedb.ongaaccel.jp/database_download/
#    Place in data/AIST++/videos/
```

**Dataset structure:**
```
data/AIST++/
├── annotations/
│   ├── keypoints3d/     # 3D keypoints (.pkl files)
│   ├── keypoints2d/     # 2D keypoints
│   ├── cameras/         # Camera calibration
│   └── splits/          # Train/val/test splits
└── videos/              # Dance videos (.mp4, 60fps)
```

### Generating Training Data

```bash
# Convert AIST++ to training pairs (REAL MediaPipe errors!)
uv run python scripts/convert_aistpp_to_training.py
```

**Output:** `data/training/aistpp_converted/` containing NPZ files with:
- `corrupted`: MediaPipe 3D pose (17, 3) - noisy depth from video
- `ground_truth`: AIST++ 3D keypoints (17, 3) - accurate depth
- `visibility`: Per-joint visibility (17,) from MediaPipe
- `azimuth`: Horizontal view angle 0-90° (from camera calibration)
- `elevation`: Vertical view angle -90 to +90° (from camera calibration)

### View Angle Computation

View angles are computed from **actual camera positions** in AIST++ calibration data, not from torso orientation (which fails when subject bends).

```python
# Camera position from extrinsics: C = -R^T @ t
# Azimuth: horizontal angle (0° = frontal, 90° = profile)
# Elevation: vertical angle (+ve = camera above, -ve = below)
azimuth, elevation = compute_view_angles(gt_pose, camera_pos)
```

### Model Architecture (Joint End-to-End Training)

```
┌────────────────────────────────────────────────────────────┐
│              JOINT DEPTH REFINEMENT MODEL                  │
│                                                            │
│  pose (17,3) ──┬──→ View Angle Head ──→ (azimuth, elev)   │
│  visibility ───┘          │                    │          │
│                           ▼                    ▼          │
│                    ┌─────────────────────────────────┐    │
│                    │     Depth Refinement Head       │    │
│                    │  Transformer + View-Conditioned │    │
│                    └──────────────┬──────────────────┘    │
│                                   ▼                       │
│                            Δz (17,) per joint             │
└────────────────────────────────────────────────────────────┘

Loss = L_depth + λ * L_viewangle  (both heads train together)
```

At inference: single forward pass, no camera calibration needed.

### Why AIST++ ?

| Feature | AIST++ | HumanEva | CMU Mocap |
|---------|--------|----------|-----------|
| Annotated frames | **10.1M** | ~2K | 0 (no video) |
| Subjects | **30** | 3 | 144 |
| Joint format | **COCO 17** | Custom 15 | Custom 31 |
| Video | 60fps multi-view | 30fps | None |
| MediaPipe errors | **REAL** | REAL | Simulated |
| Download | Python API | Manual | Git clone |

### Technical Details

**Joint format (17 COCO keypoints):**
```
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle
```

All 17 joints map directly to MediaPipe landmarks - no interpolation needed.

**Coordinate system:**
- Both MediaPipe and AIST++ centered on pelvis (hip midpoint)
- Y-up coordinate system after MediaPipe Y-flip

**Resources:**
- [AIST++ Dataset](https://google.github.io/aistplusplus_dataset/)
- [AIST++ API](https://github.com/google/aistplusplus_api)
- Converter: `scripts/convert_aistpp_to_training.py`
- Visualizer: `scripts/visualize_aistpp_training.py`

## Project Overview

HumanPose is a 3D human pose estimation pipeline that uses MediaPipe for landmark detection and Pose2Sim for marker augmentation. The pipeline processes video input through multiple stages: capture/detection, CSV export, TRC conversion, and augmentation to produce biomechanics-compatible output files.

## Build and Development Commands

### Environment Setup
- `uv sync` - Install Python 3.12 toolchain and dependencies from pyproject.toml/uv.lock

### GPU Acceleration (Optional)

The pipeline uses GPU acceleration **only for Pose2Sim LSTM inference** (marker augmentation), providing **3-10x speedup** on multi-cycle augmentation.

**Important:** MediaPipe always uses CPU - this is intentional. MediaPipe's CPU path uses XNNPACK which is highly optimized and faster than GPU for desktop systems (GPU has initialization overhead and CPU↔GPU data transfer costs).

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x
- cuDNN 9 for CUDA 12

**Installation (Ubuntu/WSL2):**
```bash
# Add NVIDIA CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA Toolkit and cuDNN
sudo apt install -y cuda-toolkit-12-6 libcudnn9-cuda-12

# Reinstall onnxruntime-gpu to detect new CUDA libraries
uv pip uninstall onnxruntime-gpu
uv pip install onnxruntime-gpu --force-reinstall
```

**Verify GPU is available:**
```bash
uv run python -c "import onnxruntime as ort; print('Available providers:', ort.get_available_providers())"
```

You should see `CUDAExecutionProvider` in the output.

**Automatic Fallback:**
- If GPU is not available, the pipeline **automatically falls back to CPU** with no user intervention
- CPU-only systems work without any code changes
- The GPU patch gracefully degrades: `[GPU] Warning: CUDA provider not available, using CPU`

**Implementation:**
- `src/markeraugmentation/gpu_config.py` - GPU configuration module
- `main.py` calls `patch_pose2sim_gpu()` **after MediaPipe extraction** (step 1) but before Pose2Sim augmentation (step 3)
- ONNX Runtime provider order: `CUDA → CPU` (automatic fallback)
- MediaPipe is unaffected - uses CPU with XNNPACK optimization

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

### Output Organization

The pipeline automatically organizes outputs into a clean directory structure:

```
data/output/pose-3d/<video_name>/
├── <video>_final.trc               # Final optimized skeleton (59-65 markers, some filtered)
├── <video>_initial.trc             # Initial TRC from MediaPipe (22 markers)
├── <video>_raw_landmarks.csv       # Raw MediaPipe landmarks
└── joint_angles/                   # Joint angle analysis (if computed)
    ├── <video>_all_joint_angles.png         # Comprehensive visualization
    ├── <video>_angles_pelvis.csv            # Pelvis angles
    ├── <video>_angles_hip_{R|L}.csv         # Hip angles (both sides)
    ├── <video>_angles_knee_{R|L}.csv        # Knee angles
    ├── <video>_angles_ankle_{R|L}.csv       # Ankle angles
    ├── <video>_angles_trunk.csv             # Trunk angles
    ├── <video>_angles_shoulder_{R|L}.csv    # Shoulder angles
    └── <video>_angles_elbow_{R|L}.csv       # Elbow angles
```

**Automatic Cleanup**: Intermediate files (augmentation cycles, Config files, pose2sim projects) are automatically removed. See [OUTPUT_ORGANIZATION.md](docs/OUTPUT_ORGANIZATION.md) for details.

### Testing
- `uv run pytest` - Run all tests
- `uv run pytest -k <module_name>` - Run tests for specific module
- Test files mirror source structure under `tests/`

### Development Tools
- `uv run jupyter lab notebooks/` - Launch Jupyter notebooks with repo packages on path
- `uv run python -m black src tests` - Format code before committing

## Pipeline Architecture

The pipeline follows a 7-step orchestration model in `main.py`:

1. **Step 1 - Extraction** (`src/posedetector/`): MediaPipe Pose processes video frames and extracts world landmarks (33 points mapped to 22 Pose2Sim-aligned markers via `POSE_NAME_MAP`). Outputs CSV with columns: timestamp_s, landmark, x_m, y_m, z_m, visibility.

2. **Step 2 - TRC Conversion** (`src/datastream/`): Converts CSV to TRC format using `ORDER_22` marker ordering. Derives synthetic markers (Hip, Neck) from parent landmarks (Hip = mean(LHip, RHip), Neck = mean(LShoulder, RShoulder)) per `DERIVED_PARENTS`. Missing landmarks remain empty - no placeholders or duplicates.

3. **Step 3 - Augmentation** (`src/markeraugmentation/`): Invokes Pose2Sim's `augment_markers_all` multiple times (default 20 cycles) via CLI or Python shim to add full OpenCap marker set using LSTM prediction. Multi-cycle averaging with small perturbations (1mm Gaussian noise per cycle) significantly improves marker completion rates. This adds **43 additional anatomical markers** (shoulder clusters, thigh clusters, medial/lateral joint markers, hip joint centers) to the original 22 markers, creating a 65-marker dataset. Intermediate cycle files are automatically cleaned up.

4. **Step 3.5 - Force Complete** (optional, `--force-complete`): Estimates shoulder clusters and hip joint centers using Bell et al. 1990 regression. Required for joint angle computation.

5. **Step 4 - Multi-Constraint Optimization** (optional, `--multi-constraint-optimization`): Applies biomechanical constraints in 3 phases: (0) Filter unreliable augmented markers, (1) Stabilize bone lengths, (2) Apply ground plane and hip width constraints. Results in 59-65 markers (4-6 typically filtered).

6. **Step 5 - Joint Angles** (optional, `--compute-all-joint-angles`): Computes ISB-compliant joint angles for all 12 joint groups. Automatically organized into `joint_angles/` subdirectory.

7. **Step 6 - Cleanup & Organization**: Automatically renames files (`<basename>_final.trc`, `<basename>_initial.trc`, `<basename>_raw_landmarks.csv`), organizes joint angles into subdirectory, and removes intermediate files.

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
- **`anatomical/`** - Anatomical constraints and corrections:
  - `anatomical_constraints.py` - Bone length smoothing, pelvis depth filtering, ground plane detection
  - `bone_length_constraints.py` - Temporal bone length consistency enforcement to reduce depth noise
  - `ground_plane_refinement.py` - Enhanced ground plane with stance detection and depth propagation
  - `joint_angle_depth_correction.py` - Joint angle constraint enforcement with depth adjustments
  - `multi_constraint_optimization.py` - Iterative multi-constraint optimization preventing cascading violations
- **`kinematics/`** - Joint angle computation and analysis using ISB standards:
  - `segment_coordinate_systems.py` - ISB-compliant anatomical coordinate systems (pelvis, femur, tibia, foot, trunk, humerus). Includes `ensure_continuity()` for axis flip prevention.
  - `angle_processing.py` - Unwrapping, smoothing, zeroing, clamping utilities for angle time series
  - `joint_angles_euler.py` - Lower limb angles (hip/knee/ankle) using Euler XYZ decomposition
  - `joint_angles_upper_body.py` - Upper body angles (trunk, shoulder, elbow) with XYZ and ZXY Euler
  - `comprehensive_joint_angles.py` - ALL joints (pelvis + lower + trunk + upper) in single unified call. Tracks frame-to-frame continuity for all segments.
  - `visualize_angles.py` - 3-panel time-series plots for individual joint groups
  - `visualize_comprehensive_angles.py` - Multi-panel grid plots for all 12 joint groups + side-by-side comparison
- **`markeraugmentation/`** - Pose2Sim integration with GPU acceleration:
  - Creates temp project structure, resolves CLI via POSE2SIM_CMD env var or local .venv/bin/pose2sim
  - `gpu_config.py` - Monkey-patches ONNX Runtime to use CUDA for 3-10x LSTM inference speedup
  - Automatic CPU fallback when GPU unavailable
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
      → [optimized TRC] (multi-constraint optimization)
      → [joint angles] (kinematics) → CSV + PNG in joint_angles/
      → [organized output] (automatic cleanup)
```

### Important Flags
- `--show-video` - Renders MediaPipe preview window + exports `<name>_preview.mp4` (requires Qt/XCB + ffmpeg)
- `--plot-landmarks` - Displays extracted CSV landmarks in 3D Matplotlib viewer
- `--plot-augmented` - Visualizes augmented TRC and exports `<name>_LSTM_preview.mp4`
- `--visibility-min` - Threshold for landmark export (default 0.3 for better coverage)
- `--estimate-missing` - Estimates missing markers using anatomical symmetry before augmentation (recommended for incomplete poses)
- `--force-complete` - Post-processes augmented TRC to estimate shoulder clusters and hip joint centers (optional)
- `--augmentation-cycles` - Number of augmentation cycles to run and average (default 20 for best results)
- `--bone-length-constraints` - Enforces consistent bone lengths across frames to reduce depth noise (recommended for better accuracy)
- `--bone-length-tolerance` - Acceptable bone length deviation (default 0.15 = 15%)
- `--bone-depth-weight` - Weight for depth vs xy correction, 0-1 (default 0.8 focuses corrections on noisy z-axis)
- `--bone-length-report` - Print detailed statistics on bone length consistency improvements
- `--ground-plane-refinement` - Enhanced ground plane with stance detection and depth propagation up kinematic chains
- `--ground-contact-threshold` - Max distance from ground for foot contact detection (default 0.03m)
- `--min-contact-frames` - Minimum consecutive frames for valid stance phase (default 3)
- `--depth-propagation-weight` - Weight decay for depth correction propagation (default 0.7)
- `--compute-joint-angles` - Compute hip/knee/ankle joint angles using Euler decomposition (requires --force-complete)
- `--joint-angle-side` - Side for joint angle computation: R or L (default R)
- `--joint-angle-smooth-window` - Smoothing window for angle computation (default 9, 0=no smoothing)
- `--plot-joint-angles` - Generate 3-panel visualization of joint angles (Hip/Knee/Ankle)
- `--check-joint-constraints` - Check for biomechanical constraint violations in computed angles
- `--compute-upper-body-angles` - Compute trunk/shoulder/elbow angles (requires --force-complete)
- `--upper-body-side` - Side for upper body computation: R or L (default R)
- `--plot-upper-body-angles` - Visualize upper body angles (3-panel: trunk/shoulder/elbow)
- `--compute-all-joint-angles` - Compute ALL joints (pelvis + lower + trunk + upper) using ISB standards (RECOMMENDED)
- `--plot-all-joint-angles` - Generate comprehensive multi-panel visualization for all 12 joint groups
- `--save-angle-comparison` - Save side-by-side comparison plot (right vs left)
- `--multi-constraint-optimization` - Apply iterative multi-constraint optimization (RECOMMENDED for best quality)
- `--multi-constraint-iterations` - Max iterations for multi-constraint optimization (default 10)

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

If no path provided, defaults to `data/output/pose-3d/joey/joey_final.trc`

### Controls
- **Mouse drag**: Rotate 3D view
- **Slider**: Navigate through frames
- **Play/Pause button**: Animate playback
- **Close window**: Exit viewer

### Marker Count Detection
- Automatically detects 65-marker (augmented) vs 22-marker (non-augmented) data
- Uses `OPENCAP_CONNECTIONS` for augmented files with proper anatomical skeleton
- Uses `MEDIAPIPE_CONNECTIONS` for non-augmented files

## Multi-Constraint Optimization (RECOMMENDED)

The `--multi-constraint-optimization` flag enables iterative refinement of pose data through multiple biomechanical constraints, preventing the "cascading violation" problem where fixing one constraint breaks another.

### How It Works

The optimizer applies biomechanical constraints in 3 sequential phases:

**Phase 0: Augmented Marker Quality Filtering**
- Automatically filters unreliable augmented markers with high temporal variance (threshold: 0.05)
- Removes noisy LSTM predictions (typically 4-6 markers: heels, some medial markers) before optimization
- Preserves all MediaPipe markers (high confidence: 95%+)
- Typical result: 59/65 markers retained (4-6 noisy markers filtered)

**Phase 1: Bone Length Stabilization**
- Restores consistent segment lengths in main kinematic chain (hip→knee→ankle)
- Uses distance constraints (not rigid movement) for augmented markers
- Medial knee/ankle markers maintain fixed distance from lateral parent markers
- Prevents optimization-induced noise in augmented markers
- Aggressive stabilization: 5 iterations, 8% tolerance, 95% depth weight

**Phase 2: Finalization**
- Ground plane alignment (feet above ground, threshold 0.03m)
- Hip width anthropometric constraint (0.20 × height ±15%)
- Heel Z-axis temporal smoothing for stability

**Key Innovation:** Augmented markers (medial knee/ankle, heels) use **distance constraints** instead of rigid kinematic chains. This prevents them from introducing noise during optimization while maintaining anatomical relationships.

### Results vs Standard Pipeline

| Metric | Standard | Multi-Constraint | Improvement |
|--------|----------|------------------|-------------|
| Bone length CV | 0.113 | 0.036 | **68.0%** |
| Markers with data | 65/65 | 59/65 | -4-6 unreliable filtered |
| Scattered markers | Common | None | **100%** |
| Ground violations | Variable | 0 | **100%** |
| Processing time | ~32s | ~45s | +13s |

### When to Use

- **RECOMMENDED** for all biomechanical analysis and motion capture workflows
- Automatically removes unreliable augmented markers (heels, noisy medial markers)
- Prevents scattered augmented markers during optimization
- Essential for accurate joint angle computation and inverse dynamics
- Use for any workflow requiring stable, biomechanically valid skeleton data

## Recommended Workflow

### For Best Quality (RECOMMENDED - Multi-Constraint Optimization + Joint Angles)
```bash
uv run python main.py \
  --video data/input/<video>.mp4 \
  --height <meters> \
  --mass <kg> \
  --age <years> \
  --sex <male|female> \
  --anatomical-constraints \
  --bone-length-constraints \
  --estimate-missing \
  --force-complete \
  --augmentation-cycles 20 \
  --multi-constraint-optimization \
  --compute-all-joint-angles \
  --plot-all-joint-angles
```

**Results** (tested on joey.mp4, 535 frames, 20 cycles):
- ⏱️ Processing Time: **~45 seconds** (with GPU acceleration)
- 📏 Bone Length Improvement: **68.0%** (0.113 → 0.036 CV)
- 🎯 Marker Quality: **59/65 markers** (4 unreliable auto-filtered)
- 📊 Joint Angles: **12 joint groups** computed (pelvis, hip, knee, ankle, trunk, shoulder, elbow)
- ✓ No scattered markers - unreliable augmented markers filtered by variance
- ✓ Stable medial markers - distance constraints prevent noise
- ✓ Organized output - automatic cleanup into clean directory structure
- ✓ GPU acceleration - 3-10x speedup on augmentation (automatic CPU fallback)

**What it does**:
1. Extracts landmarks with MediaPipe (visibility threshold 0.3)
2. Estimates missing markers using anatomical symmetry
3. Applies pre-augmentation constraints (bone length, ground plane)
4. Runs 20 augmentation cycles with averaging for smooth markers
5. Estimates shoulder clusters and hip joint centers (`--force-complete`)
6. **Applies multi-constraint optimization** (3 phases):
   - **Phase 0**: Filters unreliable augmented markers (variance > 0.05)
   - **Phase 1**: Bone length stabilization with distance constraints for augmented markers
   - **Phase 2**: Ground plane alignment and hip width anthropometric constraints
7. **Computes comprehensive joint angles**: All 12 joint groups using ISB standards
8. **Automatic cleanup**: Organizes output into clean directory structure
9. Outputs: `<video>_final.trc`, `<video>_initial.trc`, `<video>_raw_landmarks.csv`, `joint_angles/` (13 files)

---

### For Standard Processing (No Multi-Constraint)
```bash
uv run python main.py \
  --video data/input/<video>.mp4 \
  --height <meters> \
  --mass <kg> \
  --age <years> \
  --sex <male|female> \
  --anatomical-constraints \
  --bone-length-constraints \
  --estimate-missing \
  --augmentation-cycles 20
```

**Performance**: ~32 seconds
**Use when**: Simple visualization or tracking tasks where joint angle accuracy isn't critical

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

### GPU Acceleration
- **GPU is for Pose2Sim only**: MediaPipe always uses CPU (XNNPACK optimized, faster than GPU for desktop)
- **Check GPU availability**: Run `uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"` - should see `CUDAExecutionProvider`
- **Missing CUDA libraries**: If GPU detected but not working, install CUDA Toolkit 12.x and cuDNN 9 (see GPU Acceleration section)
- **CPU fallback works automatically**: Pipeline prints `[GPU] Warning: CUDA provider not available, using CPU` and continues normally
- **Performance**: GPU provides 3-10x speedup on Pose2Sim multi-cycle augmentation; CPU-only still works fine, just slower
- **Verify GPU usage**: Pipeline prints `[GPU] Pose2Sim GPU acceleration enabled (CUDA)` before augmentation step
- **WSL2 GPU support**: Requires Windows 11 with WSL2 GPU passthrough enabled and NVIDIA drivers installed on Windows host
- **Do NOT add GPU delegate to MediaPipe**: MediaPipe's CPU path is faster than GPU for desktop systems

## Pelvis Angle Calculation (Validated)

The pelvis angle computation has been **validated against a reference implementation** in `use/scripts/compute_pelvis_global_angles.py`. The main codebase produces identical results (max diff < 0.001 deg).

### Pelvis Coordinate System (ISB-Compliant)

```python
# From ASIS/PSIS markers (augmented by Pose2Sim):
Z = normalize(RASIS - LASIS)       # Right (medial-lateral)
Y_temp = normalize(ASIS_mid - PSIS_mid)  # Up (superior)
X = normalize(Y_temp × Z)          # Forward (anterior)
Y = normalize(Z × X)               # Orthogonalized up
```

### Euler Angle Sequence
- **ZXY sequence** (clinical convention) via `scipy.spatial.transform.Rotation.as_euler('ZXY')`
- Returns: `[flex, abd, rot]` in degrees (universal naming)
- **pelvis_flex_deg**: Rotation around Z (right) axis - sagittal plane tilt
- **pelvis_abd_deg**: Rotation around X (anterior) axis - frontal plane tilt
- **pelvis_rot_deg**: Rotation around Y (superior) axis - axial rotation

### Key Implementation Details
- **Smoothing**: Window size = 9 (applied to marker coordinates, not angles)
- **Zeroing**: `global_mean` mode (subtracts mean of entire trial)
- **No median filtering** for pelvis (only coordinate smoothing)
- **No clamping** for pelvis (global angles can exceed joint limits)
- **Continuity check**: Flips all axes if score < 0 to prevent 180° discontinuities

### Validation Data Location
```
use/output/
├── pose2sim_input_exact_LSTM_fixed.trc              # Reference TRC (65 markers, 709 frames)
├── pose2sim_input_exact_LSTM_fixed_pelvis_global_ZXY.csv  # Reference pelvis angles (ground truth)
└── pelvis_angles_plot.png                            # Reference visualization

Validation outputs (project root):
├── pelvis_validation_sidebyside.png                  # Side-by-side comparison (recommended)
├── pelvis_validation_comparison.png                  # Overlay + difference plot
└── pelvis_angles_main_codebase.png                   # Main codebase output only
```

### Running Validation
```bash
# Compare main codebase output with reference
uv run python scripts/compare_pelvis_output.py
```

**Expected output:**
```
pelvis_flex_deg: Max diff: 0.0005 deg  [PASS]
pelvis_abd_deg:  Max diff: 0.0005 deg  [PASS]
pelvis_rot_deg:  Max diff: 0.0005 deg  [PASS]
RESULT: ALL TESTS PASSED
```

### Critical Implementation Notes
1. **ASIS/PSIS markers required** - No fallback to Hip markers (anatomically different points)
2. **Marker names**: `r.ASIS_study`, `L.ASIS_study`, `r.PSIS_study`, `L.PSIS_study` from Pose2Sim augmentation
3. **TRC header fix**: Pose2Sim outputs TRC with header mismatch (22 in header, 65 in data) - handled automatically by `read_trc()`

## Coordinate System Implementation

### Axis Continuity (Flip Prevention)
All segment coordinate systems use `ensure_continuity()` to prevent 180° axis flips between frames:
- Computes score = sum of dot products between current and previous frame axes
- If score < 0, flips all 3 axes simultaneously (preserves right-handedness)
- Prevents massive discontinuities in Euler angle output

### Foot Coordinate System (Validated)
The foot coordinate system matches the reference implementation:
```python
X = normalize(toe - calcaneus)      # Anterior (heel -> toe)
Z_hint = fifth_meta - toe           # Lateral hint
Z_temp = normalize(Z_hint)          # Aligned with pelvis
Y = normalize(cross(Z_temp, X))     # Dorsal/Superior
Z = normalize(cross(X, Y))          # Re-orthogonalized lateral
```

**Key**: Y is computed from `cross(Z, X)`, then Z is re-orthogonalized. This matches the professor's implementation and produces correct ankle angles where flexion shows the largest ROM during running.

### Frame-to-Frame Continuity Tracking
`comprehensive_joint_angles.py` tracks previous coordinate systems for all segments:
- `prev_pelvis`, `prev_trunk`, `prev_femur_r/l`, `prev_tibia_r/l`, `prev_foot_r/l`, `prev_humerus_r/l`
- Each segment's axes are passed to `ensure_continuity()` to prevent discontinuities

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
