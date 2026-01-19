# CLAUDE.md

3D human pose estimation pipeline using MediaPipe + Pose2Sim. Outputs biomechanics-compatible TRC files with ISB-compliant joint angles.

## Quick Start

```bash
# Full pipeline with neural refinement (RECOMMENDED)
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 --mass 75 \
  --estimate-missing --force-complete \
  --augmentation-cycles 20 \
  --plot-all-joint-angles \
  --visibility-min 0.1 \
  --main-refiner
```

**Results**: ~60s processing, neural depth + joint refinement, 59/64 markers, 12 joint groups computed.

The `--main-refiner` flag enables the full neural refinement pipeline, replacing traditional constraint-based methods with learned models trained on AIST++ motion capture data.

## Pipeline Steps

1. **MediaPipe extraction** → 33 landmarks → 22 Pose2Sim markers
2. **Neural depth refinement** → corrects MediaPipe depth errors (17 COCO joints)
3. **TRC conversion** with derived markers (Hip, Neck)
4. **Pose2Sim augmentation** → 64 markers (43 added via LSTM)
5. **Joint angle computation** → 12 ISB-compliant joint groups
6. **Neural joint refinement** → corrects joint angles using learned constraints
7. **Automatic cleanup** → organized output structure

## Output Structure

```
data/output/pose-3d/<video>/
├── <video>_final.trc           # Optimized 59-64 markers
├── <video>_initial.trc         # Initial 22 markers
├── <video>_raw_landmarks.csv
└── joint_angles/               # 13 files (CSV + PNG per joint)
```

## Key Flags

| Flag | Description |
|------|-------------|
| `--main-refiner` | **Recommended**: Full neural pipeline (depth + joint refinement) |
| `--estimate-missing` | Mirror occluded limbs from visible side |
| `--force-complete` | Estimate shoulder clusters + hip joint centers |
| `--augmentation-cycles N` | Multi-cycle averaging (default 20) |
| `--plot-all-joint-angles` | Multi-panel visualization |
| `--visibility-min 0.1` | Landmark confidence threshold (default 0.3, use 0.1 to prevent marker dropout) |

**Note**: Legacy flags (`--neural-depth-refinement`, `--joint-constraint-refinement`, `--multi-constraint-optimization`, `--anatomical-constraints`, `--bone-length-constraints`, `--age`, `--sex`) have been removed. Use `--main-refiner` for the recommended neural pipeline.

## Module Structure

| Module | Purpose |
|--------|---------|
| `mediastream/` | Video I/O (OpenCV) |
| `posedetector/` | MediaPipe inference, landmark mapping |
| `datastream/` | CSV/TRC conversion, marker estimation |
| `markeraugmentation/` | Pose2Sim integration, GPU acceleration |
| `kinematics/` | ISB joint angles, Euler decomposition, visualization |
| `visualizedata/` | 3D plotting, skeleton connections |
| `depth_refinement/` | Neural depth correction model |
| `joint_refinement/` | Neural joint constraint model |
| `main_refinement/` | Fusion model combining depth + joint |
| `pipeline/` | Orchestration (refinement, cleanup) |
| `application/` | Django web interface |

## GPU Acceleration

GPU accelerates Pose2Sim LSTM only (3-10x speedup). MediaPipe uses CPU (faster with XNNPACK).

```bash
# Check GPU
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Install CUDA (if needed)
sudo apt install -y cuda-toolkit-12-6 libcudnn9-cuda-12
uv pip install onnxruntime-gpu --force-reinstall
```

Automatic CPU fallback if GPU unavailable.

## Common Issues

| Issue | Solution |
|-------|----------|
| Right arm missing | Use `--estimate-missing` to mirror from left |
| "No trc files found" | Check Pose2Sim project structure |
| Depth errors / front-back confusion | Use `--main-refiner` (neural depth correction) |
| Joint angle spikes | Use `--main-refiner` (learned joint constraints) |
| Markers disappear mid-video | Use `--visibility-min 0.1` (MediaPipe confidence drops below default 0.3 threshold) |

## Setup

### Prerequisites

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env  # or restart shell

# Clone and install
git clone <repo-url>
cd humanpose3d_mediapipe
uv sync
```

### Environment (headless/WSL)

```bash
# Required for headless environments (no GUI)
export MPLBACKEND=Agg

# Or use direnv with included .envrc
direnv allow
```

### Verify Installation

```bash
# Check imports work
uv run python -c "import mediapipe; from Pose2Sim import Pose2Sim; print('OK')"

# Check GPU (optional)
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Development

```bash
uv sync                           # Install dependencies
uv sync --group neural            # Install neural training deps
uv run pytest                     # Run tests
uv run python -m black src tests  # Format code
uv run python scripts/viz/visualize_interactive.py [file.trc]  # 3D viewer
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: tkinter` | `export MPLBACKEND=Agg` (headless) or `apt install python3-tk` (GUI) |
| TensorFlow CUDA warnings | Ignore - TF/ONNX runtime conflict, doesn't affect results |
| `Pose2Sim` import fails | Use `from Pose2Sim import Pose2Sim` (capitalized) |

## Neural Refinement

### Depth Refinement
Corrects MediaPipe depth errors using camera viewpoint prediction. Trains on AIST++ dataset (1.2M frames) with real MediaPipe errors paired with motion capture ground truth.

**Architecture**: Transformer with Part Orientation Fields (POF)
- Based on MonocularTotalCapture (CVPR 2019) Part Orientation Fields
- Predicts 14 per-limb 3D unit vectors (local depth info per body part)
- Uses camera direction vector for front/back disambiguation
- Optional MTC-style least-squares solver for geometric consistency
- Key insight: 2D foreshortening directly encodes 3D limb orientation

**Model**: ~3M params (d_model=128, 6 layers, 8 heads)

**Key components**:
- **LimbOrientationPredictor**: Predicts 3D unit vectors for 14 limbs from 2D foreshortening
- **LeastSquaresDepthSolver**: MTC-style solver ensuring 3D is consistent with 2D observations
- **DirectAnglePredictor**: Predicts global azimuth/elevation for disambiguation
- **Camera Direction Vector**: Explicit (x,y,z) direction toward camera for front/back
- **CrossJointAttention**: Transformer for inter-joint depth reasoning

**14 Limbs** (COCO-17 indices):
| Limb | Parent → Child | Limb | Parent → Child |
|------|----------------|------|----------------|
| L upper arm | 5→7 | R upper arm | 6→8 |
| L forearm | 7→9 | R forearm | 8→10 |
| L thigh | 11→13 | R thigh | 12→14 |
| L shin | 13→15 | R shin | 14→16 |
| Shoulder width | 5↔6 | Hip width | 11↔12 |
| L torso | 5→11 | R torso | 6→12 |
| L cross-body | 5→12 | R cross-body | 6→11 |

**Performance**:
| Metric | Value |
|--------|-------|
| Azimuth error | 7.1° |
| Elevation error | 4.4° |
| Depth error | 11.2 cm (45% improvement) |
| Bone variance | 1.23 cm std (75% reduction) |

**Bone Length Consistency**: Training includes bone variance loss to encourage temporally consistent bone lengths. At inference, bone locking computes median bone lengths from first 50 frames and projects all frames to those lengths. Result: more consistent than ground truth (0.57x GT variance).

```bash
# Train model with POF (recommended)
uv run --group neural python scripts/train/depth_model.py \
  --data "data/training/aistpp_converted,data/training/mtc_converted" \
  --epochs 50 --batch-size 256 --workers 8 --bf16 \
  --use-limb-orientations --limb-orientation-weight 0.5 \
  --d-model 128 --num-layers 6 --num-heads 8

# Train with MTC-style least-squares solver (experimental)
uv run --group neural python scripts/train/depth_model.py \
  --epochs 50 --batch-size 256 --workers 8 --bf16 \
  --use-limb-orientations --limb-orientation-weight 0.5 \
  --use-least-squares --projection-loss-weight 0.3

# Diagnose POF predictions
uv run --group neural python scripts/debug/diagnose_pof.py \
  --checkpoint models/checkpoints/best_depth_model.pth

# Use trained model
--neural-depth-refinement --depth-model-path models/checkpoints/best_depth_model.pth
```

**Training flags**:
| Flag | Description |
|------|-------------|
| `--use-limb-orientations` | Enable POF (Part Orientation Fields) prediction |
| `--limb-orientation-weight 0.5` | Weight for limb orientation loss |
| `--use-least-squares` | Enable MTC-style least-squares depth solver (experimental) |
| `--projection-loss-weight 0.3` | Weight for projection consistency loss |
| `--optimizer adamw` | Optimizer: adamw, lion, ademamix, sophia, schedule_free, soap |
| `--d-model 128` | Model hidden dimension (recommended) |
| `--num-layers 6` | Number of transformer layers (recommended) |
| `--num-heads 8` | Number of attention heads (recommended) |
| `--checkpoint PATH` | Resume from checkpoint |

**Least-Squares Solver** (MTC-style, experimental):
- Solves for joint depths hierarchically: hips → torso → arms/legs
- Uses input 3D X,Y as "2D" positions (MTC insight: under orthographic projection, 3D[:2] ≈ 2D)
- This ensures coordinate consistency between delta_2d and orientation vectors
- Scale factor regularization penalizes negative scales (wrong limb direction)
- Note: Experimental feature, may require further tuning for stability

**Training Data Generation** (`scripts/data/convert_aistpp.py`):

Processes AIST++ dataset (1400 sequences × 6 camera views) to create ~1.5M training pairs with REAL MediaPipe errors.

```bash
# Parallel (recommended, ~2-4 hours)
uv run python scripts/data/convert_aistpp.py --workers 8

# Single-threaded (verbose output)
uv run python scripts/data/convert_aistpp.py

# Verify alignment (saves to data/training/viz/)
uv run python scripts/viz/training_sample_viz.py --num-samples 40
```

**Coordinate transformations**:
1. **Y/Z flip**: MediaPipe (Y-down, Z-toward-camera) → AIST++ (Y-up, Z-away)
2. **Body frame alignment**: `align_body_frames()` rotates MediaPipe to match GT world orientation
3. **Scale normalization**: Both poses normalized to unit torso scale
4. **2D projection**: GT 3D → 2D via camera calibration (for POF foreshortening)
5. **c05 flip**: Camera c05 videos are horizontally flipped in AIST++; script compensates

**Expected alignment quality** (from viz stats):
- Body Frame Error: ~0°
- Torso Orientation: ~6°
- Arms/Legs Orientation: ~20-30° (actual MediaPipe depth errors)

Training sample fields (`.npz`):
| Field | Shape | Description |
|-------|-------|-------------|
| `corrupted` | (17, 3) | MediaPipe 3D, normalized |
| `ground_truth` | (17, 3) | AIST++ GT 3D, normalized |
| `pose_2d` | (17, 2) | MediaPipe 2D detection |
| `projected_2d` | (17, 2) | GT 3D projected to 2D (for POF) |
| `azimuth` | scalar | Camera angle 0-360° |
| `elevation` | scalar | Camera angle -90 to +90° |
| `visibility` | (17,) | Per-joint visibility |

**CMU MTC Dataset** (`scripts/data/convert_cmu_mtc.py`):

Additional training data from CMU Panoptic MTC dataset (~28K frames × 31 cameras). Provides diverse multi-view poses with high-quality motion capture ground truth.

```bash
# Extract dataset (290GB archive, use pigz for speed)
cd data/mtc
pigz -dc mtc_dataset.tar.gz | tar -xf -
# Or standard tar (slower):
tar -xzf mtc_dataset.tar.gz

# Explore dataset structure
uv run python scripts/data/convert_cmu_mtc.py --explore --mtc-dir data/mtc/a4_release

# Convert to training format
uv run python scripts/data/convert_cmu_mtc.py \
  --mtc-dir data/mtc/a4_release \
  --output-dir data/training/mtc_converted \
  --frame-skip 3 --workers 4

# Train with combined AIST++ and MTC data (recommended)
uv run --group neural python scripts/train/depth_model.py \
  --data "data/training/aistpp_converted,data/training/mtc_converted" \
  --epochs 50 --batch-size 256 --workers 8 --bf16 \
  --use-limb-orientations --limb-orientation-weight 0.5 \
  --d-model 128 --num-layers 6 --num-heads 8
```

**MTC coordinate transforms** (different from AIST++):
- MTC uses Y-down, Z-toward convention; script flips both Y and Z
- Camera params (t) are in cm; projection converts accordingly

### Joint Constraint Refinement
Learns soft joint constraints from AIST++ motion capture data. Transformer-based model corrects joint angles using cross-joint attention.

**Model**: 916K params, d_model=128, 4 heads, 4 layers
**Training**: 660K samples from AIST++ (6 camera views), 100 epochs
**Performance**: Mean correction 3.47°, handles errors up to 73°

**Data Quality**:
- **Gimbal lock filter**: Dataset skips samples with angles >170° (Euler singularities at ±180°)
- **Body-local foot estimation**: Foot markers use hip→shoulder "up" vector (not world Y), handles variable pose orientations after body frame alignment

```bash
# Generate training data (processes all AIST++ sequences)
uv run python scripts/data/generate_joint_angles.py --max-sequences 10000 --workers 12

# Train model (recommended settings)
uv run --group neural python scripts/train/joint_model.py \
  --epochs 100 --batch-size 1024 --workers 8 --fp16 \
  --d-model 128 --n-layers 4

# Use in pipeline
--joint-constraint-refinement

# Visualize refinement on real video (3D skeleton with color-coded corrections)
uv run --group neural python visualize_joint_refinement.py

# Interactive comparison against ground truth (AIST++ training samples)
uv run python compare_joint_interactive.py
```

**Training flags**:
| Flag | Description |
|------|-------------|
| `--d-model 128` | Model hidden dimension |
| `--n-layers 4` | Number of transformer layers |
| `--max-samples N` | Limit training samples (for debugging) |
| `--checkpoint PATH` | Resume from checkpoint |

**Visualization**: Red dashed = raw MediaPipe, Blue solid = refined, Joint colors = correction magnitude (green=small, red=large)

### MainRefiner (Fusion Model)

Combines depth and joint constraint models into a unified two-stage pipeline:

**Stage 1 - Pre-augmentation (17 COCO joints)**:
- Depth refinement corrects MediaPipe 3D errors
- Runs before marker augmentation

**Stage 2 - Post-augmentation (64 markers)**:
- Joint angles computed from augmented markers (ASIS, PSIS, etc.)
- Joint constraint model refines computed angles

**Model**: 1.2M params (d_model=128, 4 heads, 2 layers)
- Learns optimal fusion of depth + joint outputs via gating
- Cross-attention between depth and joint features
- Per-joint confidence estimation

**Training**:
```bash
# Train MainRefiner (requires pre-trained depth + joint models)
uv run --group neural python scripts/train/main_refiner.py \
  --data "data/training/aistpp_converted,data/training/mtc_converted" \
  --depth-checkpoint models/checkpoints/best_depth_model.pth \
  --joint-checkpoint models/checkpoints/best_joint_model.pth \
  --epochs 50 --batch-size 256 --workers 8 --bf16
```

**Inference** (total ~4.8M params, <10ms per frame on CPU):
```bash
# Full pipeline with main-refiner
uv run python main.py --video input.mp4 --main-refiner \
  --estimate-missing --force-complete --augmentation-cycles 20
```

The `--main-refiner` flag auto-enables:
- `--neural-depth-refinement` (early stage)
- `--compute-all-joint-angles` (required for joint model)
- `--joint-constraint-refinement` (late stage)

## Technical Details

### Marker Sets

**Original 21 markers** (from MediaPipe):
Neck, RShoulder, LShoulder, RHip, LHip, RKnee, LKnee, RAnkle, LAnkle, RHeel, LHeel, RSmallToe, LSmallToe, RBigToe, LBigToe, RElbow, LElbow, RWrist, LWrist, Hip, Nose

**Augmented 64 markers** (after Pose2Sim LSTM):
- Original 21 markers
- Lower body (35): ASIS, PSIS, medial knee/ankle, toe markers, thigh clusters, shoulder clusters, HJC
- Upper body (8): r_lelbow_study, r_melbow_study, r_lwrist_study, r_mwrist_study, L_lelbow_study, L_melbow_study, L_lwrist_study, L_mwrist_study

### Joint Angle Output
All joints use: `{joint}_flex_deg`, `{joint}_abd_deg`, `{joint}_rot_deg`
- Pelvis: ZXY Euler (global orientation)
- Lower body: Hip, Knee, Ankle (3-DOF each, both sides)
- Upper body: Trunk, Shoulder (3-DOF), Elbow (1-DOF)

### Coordinate Systems
- ISB-compliant anatomical axes for all segments
- `ensure_continuity()` prevents 180° axis flips between frames
- Pelvis from ASIS/PSIS markers, validated against reference implementation

## Code Style

- PEP 8, 4-space indent, snake_case
- Type hints on function signatures
- Single public class per module
- Imperative commit messages: `module: description`
