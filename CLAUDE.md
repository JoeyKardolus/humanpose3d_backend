# CLAUDE.md

3D human pose estimation pipeline using MediaPipe + Pose2Sim. Outputs biomechanics-compatible TRC files with ISB-compliant joint angles.

## Quick Start

```bash
# Baseline pipeline (stable)
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 --mass 75 \
  --estimate-missing \
  --augmentation-cycles 20 \
  --compute-all-joint-angles \
  --plot-all-joint-angles \
  --visibility-min 0.1
```

**Results**: ~60s processing, 59/64 markers, 12 joint groups computed.

**Experimental neural options** (independent, off by default):
- `--camera-pof`: POF 3D reconstruction from 2D keypoints (replaces MediaPipe depth)
- `--joint-refinement`: Neural joint constraint correction (requires `--compute-all-joint-angles`)

## Pipeline Steps

1. **MediaPipe extraction** → 33 landmarks → 22 Pose2Sim markers
2. **POF 3D reconstruction** → reconstructs 3D from 2D using Part Orientation Fields (17 COCO joints)
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
| `--height 1.78` | Subject height in meters (enables true metric scale output) |
| `--mass 75` | Subject mass in kg (used for Pose2Sim biomechanics) |
| `--estimate-missing` | Mirror occluded limbs from visible side |
| `--force-complete` | Estimate shoulder clusters + hip joint centers |
| `--augmentation-cycles N` | Multi-cycle averaging (default 20) |
| `--compute-all-joint-angles` | Compute 12 ISB-compliant joint groups |
| `--plot-all-joint-angles` | Multi-panel visualization |
| `--visibility-min 0.1` | Landmark confidence threshold (default 0.3, use 0.1 to prevent marker dropout) |
| `--camera-pof` | Experimental: POF 3D reconstruction from 2D keypoints |
| `--joint-refinement` | Experimental: Neural joint constraint correction |

**Note**: Legacy flags (`--neural-depth-refinement`, `--multi-constraint-optimization`, `--anatomical-constraints`, `--bone-length-constraints`, `--age`, `--sex`, `--main-refiner`) have been removed. Neural options are now independent.

## Module Structure

| Module | Purpose |
|--------|---------|
| `mediastream/` | Video I/O (OpenCV) |
| `posedetector/` | MediaPipe inference, landmark mapping |
| `datastream/` | CSV/TRC conversion, marker estimation |
| `markeraugmentation/` | Pose2Sim integration, GPU acceleration |
| `kinematics/` | ISB joint angles, Euler decomposition, visualization |
| `visualizedata/` | 3D plotting, skeleton connections |
| `pof/` | POF models (Transformer, GCN, SemGCN-Temporal), LS solver, metric scale |
| `joint_refinement/` | Neural joint constraint model |
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
| Depth errors / front-back confusion | Try `--camera-pof` (experimental POF 3D reconstruction) |
| Joint angle spikes | Try `--joint-refinement` (experimental learned constraints) |
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

### POF (Part Orientation Fields) 3D Reconstruction
Reconstructs 3D poses from 2D keypoints using Part Orientation Fields. Based on MonocularTotalCapture (CVPR 2019). Trains on AIST++ dataset (1.2M frames) with real MediaPipe 2D detections paired with motion capture ground truth.

**Architecture**: Neural network predicting 14 per-limb 3D unit vectors
- **Learns** full 3D orientation from 2D appearance (cannot be derived geometrically due to perspective)
- MTC-style least-squares solver ensures 3D is consistent with 2D observations
- Bypasses MediaPipe's broken depth estimation entirely

**Model Options**:
- Transformer: ~3M params (d_model=128, 6 layers, 8 heads) - ~11° error
- **SemGCN-Temporal**: ~1.7M params (d_model=256, 4 GCN layers) - **~7° error** (recommended)

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

**POF Angular Error Comparison**:
| Source | POF Error (limb orientation) |
|--------|------------------------------|
| MediaPipe 3D | 16.3° mean (11.9° median) |
| Transformer POF (from 2D) | ~11° |
| **SemGCN-Temporal POF** | **~7°** (best) |

**Metric Scale Recovery**:
The POF model works in normalized space (unit torso scale) internally. True metric output is recovered using known subject height:

```
metric_torso = height / HEIGHT_TO_TORSO_RATIO  (where ratio ≈ 3.4)
```

This solves the monocular scale ambiguity without camera intrinsics:
- `--height 1.78` → torso = 0.524m → all bone lengths in true meters
- Training uses normalized space (scale-invariant POF directions)
- Inference denormalizes using height-derived metric scale

```bash
# Train Transformer POF model (original)
uv run --group neural python scripts/train/pof_model.py \
  --data "data/training/aistpp_converted,data/training/mtc_converted" \
  --epochs 50 --batch-size 256 --workers 8 --bf16 \
  --d-model 128 --num-layers 6 --num-heads 8

# Use trained model
--camera-pof --pof-model-path models/checkpoints/best_pof_model.pth
```

### SemGCN-Temporal POF (Recommended)

Graph Neural Network that **learns full 3D orientation** from 2D appearance. Achieves **~7° error** vs 11° for transformer, 16° for MediaPipe.

**Core insight**: The model must learn to predict the full 3D orientation vector from 2D keypoint appearance - this cannot be derived geometrically due to perspective distortion. When a limb appears foreshortened in 2D, its depth direction is ambiguous, but the model learns to resolve this from visual context.

**Architecture**:

1. **Semantic Graph Convolution**: Uses skeleton structure as inductive bias
   - Joint-sharing edges: limbs connected at same joint
   - Kinematic edges: parent→child limb dependencies
   - Symmetry edges: left↔right limb pairs

2. **POF Prediction Head**: Predicts full 3D unit vectors for each limb
   - Main output: 14 × 3 unit orientation vectors
   - Trained with cosine similarity loss against ground truth

3. **Z-Sign Classification Head** (auxiliary task)
   - Predicts P(Z > 0) for all 14 limbs
   - Provides explicit depth direction supervision
   - Achieves **94% accuracy** - helps disambiguate foreshortening
   - Optional post-processing: correct POF.z if it disagrees with z_sign prediction

4. **Temporal Context**: Previous frame's POF informs current prediction
   - "Arm was forward in frame N → probably still forward in frame N+1"
   - Arms don't teleport between frames

**Model**: ~1.7M params (d_model=256, 4 GCN layers)
- 3 parallel GCN stacks (joint/kinematic/symmetry edges)
- Fusion layer combines edge-type outputs
- Temporal encoder for previous frame's POF
- POF head (main) + Z-sign head (auxiliary)

**Training**:
```bash
# Train SemGCN-Temporal (recommended)
uv run --group neural python scripts/train/pof_gnn_model.py \
  --data data/training/aistpp_rtmpose \
  --model-type semgcn-temporal \
  --d-model 256 --num-layers 4 \
  --z-sign-weight 0.2 --temporal \
  --epochs 50 --batch-size 256 --workers 8 --bf16
```

**Training Results** (1.15M samples, 6867 sequences):
| Epoch | Val Error | Z-Sign Acc | Notes |
|-------|-----------|------------|-------|
| 1 | 7.62° | 93.8% | Already beats transformer |
| 5 | ~7.0° | 94.5% | Rapid convergence |
| 50 | ~6-7° | ~95% | Final |

**Model Variants**:
| Model | Params | Error | Use Case |
|-------|--------|-------|----------|
| `gcn` | ~350K | ~10° | Lightweight |
| `semgcn` | ~500K | ~9° | Single-frame |
| `semgcn-temporal` | ~1.7M | **~7°** | Video (recommended) |

**Training Data Generation** (`scripts/data/convert_aistpp.py`):

Processes AIST++ dataset (1400 sequences × 6 camera views) to create ~1.5M training pairs.

```bash
# Parallel (recommended, ~2-4 hours)
uv run python scripts/data/convert_aistpp.py --workers 8

# Single-threaded (verbose output)
uv run python scripts/data/convert_aistpp.py
```

Training sample fields (`.npz`):
| Field | Shape | Description |
|-------|-------|-------------|
| `corrupted` | (17, 3) | MediaPipe 3D, normalized |
| `ground_truth` | (17, 3) | AIST++ GT 3D, normalized |
| `pose_2d` | (17, 2) | MediaPipe 2D detection |
| `projected_2d` | (17, 2) | GT 3D projected to 2D (for POF) |
| `camera_R` | (3, 3) | Camera rotation matrix |
| `visibility` | (17,) | Per-joint visibility |
| `gt_scale` | scalar | Ground truth torso length in meters |

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
--joint-refinement --compute-all-joint-angles

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

### Using Both Models Together

The POF and joint refinement models can be used together or separately:

```bash
# POF 3D reconstruction only
uv run python main.py --video input.mp4 --camera-pof \
  --height 1.78 --estimate-missing --augmentation-cycles 20

# Joint refinement only (requires joint angles)
uv run python main.py --video input.mp4 --joint-refinement \
  --compute-all-joint-angles --height 1.78 --estimate-missing

# Both models together
uv run python main.py --video input.mp4 --camera-pof --joint-refinement \
  --compute-all-joint-angles --height 1.78 --estimate-missing --augmentation-cycles 20
```

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
