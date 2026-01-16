# CLAUDE.md

3D human pose estimation pipeline using MediaPipe + Pose2Sim. Outputs biomechanics-compatible TRC files with ISB-compliant joint angles.

## Quick Start

```bash
# Full pipeline with joint angles (RECOMMENDED)
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 --mass 75 --age 30 --sex male \
  --anatomical-constraints --bone-length-constraints \
  --estimate-missing --force-complete \
  --augmentation-cycles 20 \
  --multi-constraint-optimization \
  --compute-all-joint-angles --plot-all-joint-angles \
  --visibility-min 0.1
```

**Results**: ~45s processing, 68% bone length improvement, 59/64 markers, 12 joint groups computed.

## Pipeline Steps

1. **MediaPipe extraction** → 33 landmarks → 22 Pose2Sim markers
2. **TRC conversion** with derived markers (Hip, Neck)
3. **Pose2Sim augmentation** → 64 markers (43 added via LSTM)
4. **Multi-constraint optimization** → filters noisy markers, stabilizes bones
5. **Joint angle computation** → 12 ISB-compliant joint groups
6. **Automatic cleanup** → organized output structure

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
| `--estimate-missing` | Mirror occluded limbs from visible side |
| `--force-complete` | Estimate shoulder clusters + hip joint centers |
| `--augmentation-cycles N` | Multi-cycle averaging (default 20) |
| `--multi-constraint-optimization` | 3-phase biomechanical refinement |
| `--compute-all-joint-angles` | All 12 joint groups (pelvis, hip, knee, ankle, trunk, shoulder, elbow) |
| `--plot-all-joint-angles` | Multi-panel visualization |
| `--bone-length-constraints` | Temporal bone length consistency |
| `--visibility-min 0.1` | Landmark confidence threshold (default 0.3, use 0.1 to prevent marker dropout) |

## Module Structure

| Module | Purpose |
|--------|---------|
| `mediastream/` | Video I/O (OpenCV) |
| `posedetector/` | MediaPipe inference, landmark mapping |
| `datastream/` | CSV/TRC conversion, marker estimation |
| `markeraugmentation/` | Pose2Sim integration, GPU acceleration |
| `anatomical/` | Bone constraints, ground plane, multi-constraint optimization |
| `kinematics/` | ISB joint angles, Euler decomposition, visualization |
| `visualizedata/` | 3D plotting, skeleton connections |

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
| Scattered markers | Use `--multi-constraint-optimization` |
| Markers disappear mid-video | Use `--visibility-min 0.1` (MediaPipe confidence drops below default 0.3 threshold) |

## Development

```bash
uv sync                           # Install dependencies
uv run pytest                     # Run tests
uv run python -m black src tests  # Format code
uv run python visualize_interactive.py [file.trc]  # 3D viewer
```

## Neural Refinement

### Depth Refinement
Corrects MediaPipe depth errors using camera viewpoint prediction. Trains on AIST++ dataset (1.2M frames) with real MediaPipe errors paired with motion capture ground truth.

**Architecture**: Transformer with ElePose-style direct angle prediction
- Uses ElePose backbone (CVPR 2022) for deep 2D foreshortening features
- Predicts camera azimuth/elevation directly from 2D+3D pose features
- Avoids body-frame mismatch between training (GT pose) and inference (corrupted pose)
- Key insight: 2D foreshortening (shoulder width ratio, limb lengths) encodes viewpoint

**Model**: 7.3M params with ElePose (d_model=128, 6 layers, 8 heads, elepose_hidden_dim=1024)

**Key components**:
- **ElePose backbone**: ResNet-style 2D pose encoder (1024-dim features)
- **DirectAnglePredictor**: Predicts azimuth/elevation directly (avoids body-frame mismatch)
- **ViewAngleEncoder**: Fourier features for periodic angle patterns
- **CrossJointAttention**: Transformer for inter-joint depth reasoning

**Performance**:
| Metric | Value |
|--------|-------|
| Azimuth error | 7.1° |
| Elevation error | 4.4° |
| Depth error | 11.2 cm (45% improvement) |
| Bone variance | 1.23 cm std (75% reduction) |

**Bone Length Consistency**: Training includes bone variance loss to encourage temporally consistent bone lengths. At inference, bone locking computes median bone lengths from first 50 frames and projects all frames to those lengths. Result: more consistent than ground truth (0.57x GT variance).

```bash
# Generate training data (parallel, ~2-4 hours)
bash scripts/run_parallel_conversion.sh

# Train model (recommended settings with ElePose)
uv run --group neural python scripts/train_depth_model.py \
  --epochs 50 --batch-size 256 --workers 8 --fp16 \
  --d-model 128 --num-layers 6 --num-heads 8 \
  --elepose --elepose-hidden-dim 1024

# Use trained model
--neural-depth-refinement --depth-model-path models/checkpoints/best_depth_model.pth
```

**Training flags**:
| Flag | Description |
|------|-------------|
| `--elepose` | Use ElePose backbone for camera angle prediction (recommended) |
| `--elepose-hidden-dim 1024` | Hidden dimension for ElePose backbone |
| `--d-model 128` | Model hidden dimension |
| `--num-layers 6` | Number of transformer layers |
| `--num-heads 8` | Number of attention heads |
| `--checkpoint PATH` | Resume from checkpoint |

### Joint Constraint Refinement
Learns soft joint constraints from AIST++ motion capture data. Transformer-based model corrects joint angles using cross-joint attention.

**Model**: 916K params, d_model=128, 4 heads, 4 layers
**Training**: 660K samples from AIST++ (6 camera views), 100 epochs
**Performance**: Mean correction 3.47°, handles errors up to 73°

```bash
# Generate training data (processes all AIST++ sequences)
uv run python scripts/generate_joint_angle_training.py --max-sequences 10000 --workers 4

# Train model (recommended settings)
uv run --group neural python scripts/train_joint_model.py \
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
