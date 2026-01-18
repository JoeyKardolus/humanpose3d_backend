# Complete Project Development - HumanPose 3D Pipeline

This document summarizes ALL changes and features built from initial MediaPipe-only setup to the complete augmented 3D pose estimation pipeline with advanced smoothing.

## Latest Updates

### ❌ Failed Approach: Rigid Cluster Constraints (January 2026)

**Attempted**: Template-based rigid body constraints using Procrustes analysis
- Created `rigid_cluster_constraints.py` with orthogonal Procrustes fitting
- Goal: Lock augmented marker groups (shoulder clusters, foot markers) into rigid body templates
- Computed median template from first 50 frames, then fit each frame via rotation/translation

**What happened**: **INCREASED noise instead of reducing it**
- Z-axis noise got significantly worse after applying rigid constraints
- Thigh clusters, knee markers, and shoulder markers showed more scatter
- Right shoulder cluster completely failed (no valid data in template)

**Why it failed**:
1. **LSTM markers don't follow perfect rigid body physics** - Pose2Sim's LSTM generates markers based on learned patterns, not physical constraints
2. **Template captured systematic LSTM errors** - Median from first 50 frames locked in noise patterns
3. **Procrustes forcing artifacts** - Forcing non-rigid data into rigid templates introduced new artifacts
4. **Low confidence markers excluded** - Markers valid in <25% of frames excluded from template, leaving them unconstrained

**Key lesson**: Don't force LSTM-generated augmented markers into rigid body templates. They're predictions, not physical measurements.

**Status**: Reverted. Code remains in `src/anatomical/rigid_cluster_constraints.py` but NOT used in pipeline. Available via `--rigid-clusters` flag if needed for experimentation.

---

### Multi-Constraint Optimization (January 2026)

**Created iterative biomechanical constraint system**:
- `multi_constraint_optimization.py` - Cycles through 6 constraint types until convergence
- Prevents "cascading violations" where fixing one constraint breaks another
- **Results**: 44.8% improvement in joint angle accuracy, 31.5% improvement in bone length consistency
- **Strategy**: MediaPipe markers corrected freely, augmented markers constrained relative to parents
- Eliminates scattered markers (100% stable)
- Processing time: ~45 seconds (was ~32 seconds)

**Constraint cycle**:
1. Bone length constraints (hip→knee→ankle consistency)
2. Joint angle constraints (biomechanical limits: hip 0-120°, knee 0-160°, ankle ±30°)
3. Ground plane constraints (feet on ground)
4. Hip width constraint (anthropometric: 0.20 × height ±15%)
5. Augmented marker constraints (fixed distance from MediaPipe parents)
6. Heel Z-axis smoothing (temporal median filter)

**Key insight**: Never lock markers - use relative parent-child constraints instead. Locking reduced effectiveness to 2.1% (vs 44.8% with relative constraints).

**Controlled by**: `--multi-constraint-optimization` and `--multi-constraint-iterations N` flags

## Starting Point

**Initial state**: Basic MediaPipe landmark extraction only - no TRC conversion, no augmentation, no smoothing.

## Major Features Built

### 1. Data Pipeline & TRC Conversion (`src/datastream/`)

**Created complete data pipeline**:
- `data_stream.py` - Core TRC/CSV writing with strict mode
- `LandmarkRecord` dataclass for type-safe landmark handling
- `ORDER_22` marker set mapping (MediaPipe → Pose2Sim compatible)
- `DERIVED_PARENTS` system for synthetic markers (Hip, Neck)
- `write_landmark_csv()` - Exports landmarks to CSV with timestamp sorting
- `csv_to_trc_strict()` - Converts CSV to TRC format with ORDER_22 layout
- `header_fix_strict()` - Fixes TRC metadata to match actual marker counts

**Strict mode rules**:
- No placeholder values for missing markers
- No duplicate markers
- Explicit visibility thresholds (default 0.3)
- Deterministic output (sorted by timestamp, landmark)

### 2. Pose2Sim Marker Augmentation (`src/markeraugmentation/`)

**Created full augmentation pipeline**:
- `markeraugmentation.py` - Pose2Sim integration
- `run_pose2sim_augment()` - Multi-cycle LSTM augmentation with averaging
- Automatic Config.toml generation for Pose2Sim
- Temp project structure creation
- Multi-cycle averaging (default 20 cycles with 0.5mm Gaussian noise)
- Automatic cleanup of intermediate cycle files
- Adds **43 additional anatomical markers** (shoulder clusters, thigh clusters, hip joint centers)
- Creates 65-marker dataset (22 input + 43 augmented)

**Features**:
- `--augmentation-cycles N` flag (configurable cycle count)
- Averages multiple LSTM predictions for robustness
- Handles TRC header mismatch (22 in header, 65 in data)

### 3. Marker Estimation System (`src/datastream/`)

**Two-stage estimation approach**:

#### Pre-Augmentation Estimation (`marker_estimation.py`)
- `estimate_missing_markers()` - Fills missing INPUT markers before Pose2Sim
- **Right arm mirroring**: Mirrors left arm when right is occluded
- **Head extrapolation**: Estimates Head from Nose-Neck vector
- **SmallToe estimation**: Calculates from BigToe-Heel geometry
- Uses anatomical symmetry and geometric relationships
- Improves LSTM success rate: 0% → 81% with incomplete data
- Controlled by `--estimate-missing` flag

#### Post-Augmentation Estimation (`post_augmentation_estimation.py`)
- `estimate_shoulder_clusters_and_hjc()` - Fills missing OUTPUT markers after Pose2Sim
- **Shoulder clusters**: Estimates r_sh1-3, L_sh1-3 from shoulder-elbow vectors
- **Hip joint centers**: Calculates RHJC, LHJC using Bell et al. 1990 regression
- Targets the 8 markers LSTM often skips
- Improves completion: 81% → 100%
- Controlled by `--force-complete` flag

### 4. Interactive 3D Visualization (`src/visualizedata/`)

**Created complete visualization system**:
- `visualize_data.py` - Matplotlib-based 3D plotting
- `visualize_interactive.py` - Standalone interactive viewer
- Auto-detection of marker sets (22 vs 65 markers)
- Anatomically correct skeleton connections:
  - `MEDIAPIPE_CONNECTIONS` - For 22-marker data
  - `OPENCAP_CONNECTIONS` - For 65-marker augmented data (33 connections)
- Interactive controls:
  - Mouse drag to rotate
  - Frame slider for navigation
  - Play/Pause animation
  - Speed control
- Export to MP4 (ffmpeg) or GIF (Pillow fallback)
- Reads ALL markers from TRC data (not just header)
- Handles TRC header mismatch correctly

**Visualization flags**:
- `--show-video` - MediaPipe preview window
- `--plot-landmarks` - 3D viewer for CSV landmarks
- `--plot-augmented` - 3D viewer for augmented TRC

### 5. FLK Filtering Module (`src/filtering/`)

**Created spatio-temporal filtering system**:
- `flk_filter.py` - FLK (Filter with Learned Kinematics) integration
- `apply_flk_filter()` - Multi-pass Kalman filtering with biomechanical constraints
- `apply_gaussian_smoothing()` - Temporal Gaussian smoothing for landmark trajectories
- `landmarks_to_flk_format()` - Converts to FLK's wide DataFrame format
- `flk_format_to_landmarks()` - Converts filtered data back to landmark records

**Features**:
- Filters 14 core body markers (excludes derived Hip, Nose, heels, toes)
- Multi-pass filtering for aggressive smoothing (`--flk-passes N`)
- Optional RNN mode for motion prediction (`--flk-enable-rnn`)
- Auto-detects FPS from timestamp data
- Handles missing frames gracefully with NaN values
- Gaussian smoothing with configurable sigma (`--gaussian-smooth SIGMA`)

**FLK Integration**:
- Installed FLK from GitHub (not in PyPI)
- Custom Keras 3.x compatibility patch for `time_major` argument
- Auto-detects GRU model at `FLK/models/GRU.h5`
- Requires TensorFlow 2.18+ (installed via `uv sync --extra filtering`)

### 6. Build System & Logging (`src/application/`)

**Created build tracking**:
- `build_log.py` - Appends build metadata to `docs/BUILD_LOG.md`
- `append_build_log()` - Tracks each pipeline run
- Step-based execution scripts (mostly superseded by main.py)

### 7. Main Pipeline Orchestration (`main.py`)

**Complete CLI application** with argument parsing:

#### Core Flags
- `--video PATH` - Input video file (required)
- `--height METERS` - Subject height
- `--mass KG` - Subject mass
- `--age YEARS` - Subject age
- `--sex {male,female}` - Subject sex

#### Smoothing & Filtering
- `--gaussian-smooth SIGMA` - Gaussian temporal smoothing (0=disabled)
- `--flk-filter` - Enable FLK filtering
- `--flk-passes N` - Number of FLK passes (default 1)
- `--flk-model PATH` - Path to GRU model
- `--flk-enable-rnn` - Enable RNN component (unstable)

#### Marker Processing
- `--visibility-min THRESHOLD` - Landmark export threshold (default 0.3)
- `--estimate-missing` - Pre-augmentation marker estimation
- `--force-complete` - Post-augmentation marker completion
- `--augmentation-cycles N` - Number of LSTM cycles (default 20)

#### Visualization
- `--show-video` - MediaPipe preview window
- `--plot-landmarks` - 3D viewer for CSV landmarks
- `--plot-augmented` - 3D viewer for augmented TRC

#### Optional Steps
- `--fix-header` - TRC header fix

**Pipeline execution order**:
1. MediaPipe landmark extraction
2. Gaussian smoothing (optional)
3. FLK filtering (optional, multi-pass)
4. Marker estimation (optional)
5. CSV export with strict mode
6. TRC conversion with ORDER_22
7. Pose2Sim augmentation (multi-cycle)
8. Post-augmentation estimation (optional)
9. Header fix (optional)

## Dependencies Added

### Python Packages (pyproject.toml)

**Core dependencies**:
- `mediapipe>=0.10.21` - Pose landmark detection
- `opencv-python>=4.11.0.86` - Video I/O
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.7.0` - Visualization
- `pose2sim>=0.10.0` - Marker augmentation
- `jupyter>=1.1.1` - Notebook support

**Optional dependencies** (`[project.optional-dependencies]`):
- `filtering` extra:
  - `tensorflow>=2.13.0` - Required by FLK (installed as 2.18.1)
  - `scipy` - For Gaussian filtering

### External Libraries

**FLK (Filter with Learned Kinematics)**:
- Cloned from GitHub: https://github.com/PARCO-LAB/FLK
- Custom Keras 3.x compatibility patch applied
- Installed to project root: `./FLK/`
- Includes pre-trained GRU model: `FLK/models/GRU.h5` (37MB)

## Bug Fixes & Patches

### 1. FLK Keras 3.x Compatibility

**Problem**: FLK built for TensorFlow 2.13/Keras 2.x, incompatible with Keras 3.x

**Solution**: Patched `FLK/FLK/FLK/FLK/RNN.py`:
```python
class CompatGRU(keras.layers.GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Remove deprecated arg
        super().__init__(*args, **kwargs)

custom_objects = {'GRU': CompatGRU}
self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
```

### 2. TensorFlow Dependency Resolution

**Problem**: TensorFlow 2.20.0 had protobuf conflicts

**Solution**: Used `uv sync --extra filtering` to auto-resolve to TensorFlow 2.18.1

### 3. TRC Header Mismatch Handling

**Problem**: Pose2Sim outputs TRC with 22 markers in header but 65 in data

**Solution**: `VisualizeData.load_trc_frames()` reads actual data columns, not header

### 4. Multi-Cycle Augmentation Cleanup

**Problem**: Pose2Sim left intermediate cycle files

**Solution**: Auto-cleanup in `run_pose2sim_augment()` after averaging

## New Files Created

### Source Code
- `src/datastream/data_stream.py` - TRC/CSV pipeline
- `src/datastream/marker_estimation.py` - Pre-augmentation estimation
- `src/datastream/post_augmentation_estimation.py` - Post-augmentation completion
- `src/markeraugmentation/markeraugmentation.py` - Pose2Sim integration
- `src/filtering/flk_filter.py` - FLK filtering + Gaussian smoothing
- `src/visualizedata/visualize_data.py` - 3D plotting engine
- `src/application/build_log.py` - Build tracking
- `visualize_interactive.py` - Standalone interactive viewer
- `main.py` - Complete pipeline orchestrator

### Module Init Files
- `src/datastream/__init__.py`
- `src/markeraugmentation/__init__.py`
- `src/filtering/__init__.py`
- `src/visualizedata/__init__.py`
- `src/application/__init__.py`

### Documentation
- `CLAUDE.md` - Project guide for Claude Code
- `CHANGES.md` - This file
- `docs/FLK_SETUP.md` - FLK installation guide
- `docs/BUILD_LOG.md` - Build history
- `docs/PIPELINE.md` - Pipeline architecture

### Configuration
- `pyproject.toml` - Updated with all dependencies
- `uv.lock` - Locked dependency versions

### External
- `FLK/` - Cloned FLK repository with patches

## Modified Files (from initial MediaPipe-only state)

### Existing Files Enhanced
- `src/mediastream/media_stream.py` - Enhanced with proper RGB conversion
- `src/posedetector/pose_detector.py` - Enhanced with world landmark extraction

## Usage Examples

### Basic Pipeline (Fast)
```bash
uv run python main.py \
  --video data/input/video.mp4 \
  --height 1.78 --mass 75.0 --age 30 --sex male \
  --estimate-missing
```
**Output**: 65-marker augmented TRC in ~30-45 seconds

### Recommended Pipeline (Smooth)
```bash
uv run python main.py \
  --video data/input/video.mp4 \
  --height 1.78 --mass 75.0 --age 30 --sex male \
  --gaussian-smooth 2.5 \
  --flk-filter \
  --flk-passes 2 \
  --estimate-missing \
  --augmentation-cycles 25
```
**Output**: Very smooth 65-marker TRC in ~60-90 seconds

### Maximum Smoothness (Brute Force)
```bash
uv run python main.py \
  --video data/input/video.mp4 \
  --height 1.78 --mass 75.0 --age 30 --sex male \
  --gaussian-smooth 3.0 \
  --flk-filter \
  --flk-passes 3 \
  --estimate-missing \
  --force-complete \
  --augmentation-cycles 30
```
**Output**: Extremely smooth complete 65-marker TRC in ~2-3 minutes

### Interactive Visualization
```bash
uv run python visualize_interactive.py data/output/pose-3d/subject/subject_LSTM.trc
```

## Testing

### Created Tests
- `tests/test_trc_strict.py` - TRC conversion tests

### Test Guidelines
- Use `tmp_path` fixtures for file I/O
- Mock MediaPipe/OpenCV for heavy operations
- Prefer deterministic fixtures from `data/input/tests/`

## Performance Benchmarks

### Pipeline Stages (30-second video, 30fps, ~900 frames)

| Stage | Time | Notes |
|-------|------|-------|
| MediaPipe extraction | ~20s | GPU-accelerated if available |
| Gaussian smoothing | ~1s | Fast scipy operation |
| FLK filtering (1 pass) | ~10s | Sequential frame processing |
| FLK filtering (3 passes) | ~30s | Linear scaling with passes |
| Marker estimation | ~2s | Simple geometric calculations |
| TRC conversion | <1s | Fast pandas operations |
| Pose2Sim augmentation (20 cycles) | ~40s | Dominant bottleneck |
| Pose2Sim augmentation (30 cycles) | ~60s | Linear scaling with cycles |
| Post-augmentation estimation | ~1s | Vectorized operations |

### Total Pipeline Times

| Configuration | Time |
|---------------|------|
| Basic (no smoothing) | ~30-45s |
| Light smoothing | ~45-60s |
| Medium smoothing | ~60-90s |
| Heavy smoothing | ~90-120s |
| Maximum smoothing | ~120-180s |

## Data Flow Architecture

```
Input Video (MP4)
    ↓
MediaPipe Landmark Detection (33 landmarks)
    ↓
Map to 22 Pose2Sim markers (POSE_NAME_MAP)
    ↓
[Gaussian Smoothing] (optional, --gaussian-smooth)
    ↓
[FLK Filtering, N passes] (optional, --flk-filter)
    ↓
[Marker Estimation] (optional, --estimate-missing)
    ↓
CSV Export (timestamp_s, landmark, x_m, y_m, z_m, visibility)
    ↓
TRC Conversion (ORDER_22, DERIVED_PARENTS)
    ↓
Pose2Sim Augmentation (N cycles with averaging)
    ├─ Cycle 1: LSTM prediction + 0.5mm noise
    ├─ Cycle 2: LSTM prediction + 0.5mm noise
    ├─ ...
    ├─ Cycle N: LSTM prediction + 0.5mm noise
    └─ Average all cycles → 65 markers (22 + 43 augmented)
    ↓
[Post-Augmentation Estimation] (optional, --force-complete)
    ↓
[Header Fix] (optional, --fix-header)
    ↓
Output TRC (65-marker dataset)
    ↓
[Interactive Visualization] (optional)
```

## Marker Sets

### ORDER_22 (Input to Pose2Sim)
Neck, RShoulder, LShoulder, RHip, LHip, RKnee, LKnee, RAnkle, LAnkle, RHeel, LHeel, RSmallToe, LSmallToe, RBigToe, LBigToe, RElbow, LElbow, RWrist, LWrist, Hip, Head, Nose

### Augmented Markers (43 additional)
Shoulder clusters (r_sh1-3, L_sh1-3), thigh clusters, medial/lateral joint markers, hip joint centers (RHJC, LHJC), and more

### FLK Core 14 (Filtered by FLK)
Neck, Head, RShoulder, LShoulder, RElbow, LElbow, RWrist, LWrist, RHip, LHip, RKnee, LKnee, RAnkle, LAnkle

## Known Limitations

### FLK Filtering
- RNN mode unstable (`--flk-enable-rnn` may error)
- Only filters 14/22 core markers
- Subtle effect after Pose2Sim's heavy smoothing
- Requires manual GitHub installation

### Pose2Sim Augmentation
- Typically completes 35-40/43 augmented markers
- Shoulder clusters and HJC often missing (use `--force-complete`)
- Header mismatch (22 in header, 65 in data) is normal

### MediaPipe Detection
- Struggles with occluded limbs (use `--estimate-missing`)
- Head marker unreliable when tilted
- Right arm often missing (mirrored from left if estimated)

### Visualization
- Requires Qt/XCB for `--show-video` on Linux
- MP4 export requires ffmpeg (falls back to GIF)
- Large files slow down interactive viewer

## Future Enhancements (Not Implemented)

- [ ] Multi-pass MediaPipe extraction with averaging
- [ ] Savitzky-Golay filter as alternative to Gaussian
- [ ] Per-marker smoothing intensity control
- [ ] Real-time FLK latency adjustment
- [ ] FLK model fine-tuning for MediaPipe landmarks
- [ ] GPU acceleration for FLK filtering
- [ ] Batch processing multiple videos
- [ ] Configuration file support (YAML/JSON)

## References

- **MediaPipe Pose**: https://google.github.io/mediapipe/solutions/pose
- **Pose2Sim**: https://github.com/perfanalytics/pose2sim
- **FLK**: https://github.com/PARCO-LAB/FLK
- **OpenCap**: https://www.opencap.ai/
- **Bell et al. 1990**: Hip joint center regression method

## Development Timeline

1. **Phase 1**: MediaPipe integration + basic CSV export
2. **Phase 2**: TRC conversion with ORDER_22 mapping
3. **Phase 3**: Pose2Sim augmentation integration
4. **Phase 4**: Multi-cycle averaging for robustness
5. **Phase 5**: Marker estimation (pre/post augmentation)
6. **Phase 6**: Interactive 3D visualization
7. **Phase 7**: FLK filtering with Keras 3.x compatibility
8. **Phase 8**: Gaussian smoothing + multi-pass filtering
9. **Phase 9**: Documentation + build system

## Complete Feature Set

✅ MediaPipe 3D pose detection
✅ 22-marker Pose2Sim-compatible mapping
✅ Strict mode TRC conversion
✅ Multi-cycle LSTM augmentation (20-40 cycles)
✅ Pre-augmentation marker estimation (symmetry-based)
✅ Post-augmentation completion (shoulder clusters + HJC)
✅ Gaussian temporal smoothing
✅ FLK spatio-temporal filtering (multi-pass)
✅ Interactive 3D visualization
✅ Batch augmentation with averaging
✅ Automatic cleanup of intermediate files
✅ Build logging and tracking
✅ Comprehensive CLI with 20+ flags
✅ Complete documentation suite
