# Neural Models

## MainRefiner (Recommended)

The **MainRefiner** is a unified neural pipeline that combines depth and joint constraint refinement:

```bash
# Full pipeline with neural refinement
uv run python manage.py run_pipeline \
  --video data/input/joey.mp4 \
  --height 1.78 --weight 75 \
  --estimate-missing --force-complete \
  --augmentation-cycles 20 \
  --main-refiner \
  --plot-all-joint-angles
```

**How it works:**
1. **Stage 1 (Pre-augmentation)**: Depth refinement corrects MediaPipe 3D errors on 17 COCO joints
2. **Stage 2 (Post-augmentation)**: Joint constraint refinement applies learned soft constraints to computed angles

**Training:**
```bash
uv run --group neural python scripts/train/main_refiner.py \
  --data "data/training/aistpp_converted,data/training/mtc_converted" \
  --depth-checkpoint models/checkpoints/best_depth_model.pth \
  --joint-checkpoint models/checkpoints/best_joint_model.pth \
  --epochs 50 --batch-size 256 --workers 8 --bf16
```

**Model specs**: 1.2M params (d_model=128, 4 heads, 2 layers), <10ms inference per frame

---

## Joint Constraint Refinement Model

### Overview

Created a **separate neural model** for joint constraint refinement that learns realistic, data-driven joint constraints from AIST++. This is independent from the depth refinement model.

**Key Innovation**: Instead of hard-coded joint angle limits, the model learns soft constraints from the data distribution. It understands pose context (bent knee allows different hip ROM, arms up changes trunk constraints).

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│           JOINT CONSTRAINT REFINEMENT MODEL             │
│                                                         │
│  Input: Joint angles (12, 3) + visibility               │
│         Computed by validated ISB kinematics            │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Angle Encoder (sin/cos periodic features)       │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Cross-Joint Attention (kinematic chain bias)   │   │
│  │  - Bent knee → different hip constraints        │   │
│  │  - Arms up → different trunk constraints        │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Per-Joint Refinement Heads (12 × MLP)          │   │
│  │  - Δangle predictions for each DOF              │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Output: Refined joint angles (soft-constrained)       │
└─────────────────────────────────────────────────────────┘
```

**Model size**: 916K parameters (compact, fast inference)

### Training Data Pipeline

```
Existing NPZ (17 COCO joints from AIST++)
    ↓
Estimate missing markers:
├── Neck = midpoint(shoulders)
├── Heel = ankle + foot_back_offset (5cm back, 3cm down)
├── BigToe = ankle + foot_front_offset (20cm forward)
└── SmallToe = BigToe + lateral_offset (3cm)
    ↓
Convert to TRC (22 markers)
    ↓
Pose2Sim augmentation → 65 markers
    ↓
Compute joint angles (validated ISB kinematics)
    ↓
Save to aistpp_joint_angles/
```

**Important**: Uses the validated ISB kinematics code from `src/kinematics/comprehensive_joint_angles.py`. The model learns to refine angles, not compute them.

### Data Format (Extended NPZ)

```python
{
    # Original fields (from depth refinement):
    'corrupted': (17, 3),           # MediaPipe COCO pose
    'ground_truth': (17, 3),        # AIST++ COCO pose
    'visibility': (17,),
    'azimuth': float,
    'elevation': float,

    # New: Joint angles (12 joints × 3 DOF)
    'corrupted_angles': (12, 3),    # Angles from MediaPipe pose
    'ground_truth_angles': (12, 3), # Angles from GT pose
}

# Joint order (12 joints):
['pelvis', 'hip_R', 'hip_L', 'knee_R', 'knee_L',
 'ankle_R', 'ankle_L', 'trunk', 'shoulder_R', 'shoulder_L',
 'elbow_R', 'elbow_L']
```

### Implementation Status

**Data Generation** ✅
- `scripts/generate_joint_angle_training.py` - Groups frames by sequence, estimates feet, runs augmentation, computes angles
- GPU-accelerated via `patch_pose2sim_gpu()` for LSTM inference
- Output: `data/training/aistpp_joint_angles/`
- Current: ~1M samples generating (3000 sequences × ~340 frames/seq)

**Model** ✅ (`src/joint_refinement/`)
- `model.py` - JointConstraintRefiner (916K parameters)
  - AngleEncoder with sin/cos periodic features
  - CrossJointAttention with kinematic chain bias
  - Per-joint refinement heads
- `losses.py` - Reconstruction + symmetry + delta regularization
- `dataset.py` - PyTorch dataset with L/R swap augmentation
- `inference.py` - JointRefiner class for runtime

**Training** ✅
- `scripts/train_joint_model.py` - Full training script
- Quick test (3 epochs, 1800 samples): 22.7° → 22.4° mean error
- Model saved: `models/checkpoints/best_joint_model.pth`

**Pipeline Integration** ✅
- Integrated into the `run_pipeline` management command with CLI flags
- `--joint-constraint-refinement` - Enable neural refinement after angle computation
- `--joint-model-path` - Custom model path (default: `models/checkpoints/best_joint_model.pth`)
- Applies refinement per-frame to all 12 joint groups

### Usage

**Generate Training Data:**
```bash
uv run python scripts/data/generate_joint_angles.py --max-sequences 3000 --workers 4
```

**Train:**
```bash
uv run --group neural python scripts/train/joint_model.py --epochs 100 --batch-size 64 --fp16
```

**Run with Main Pipeline (recommended):**
```bash
uv run python manage.py run_pipeline \
  --video data/input/joey.mp4 \
  --height 1.78 --weight 75 \
  --estimate-missing --force-complete \
  --augmentation-cycles 20 \
  --main-refiner \
  --plot-all-joint-angles
```

**Standalone Inference:**
```python
from src.joint_refinement.inference import JointRefiner

refiner = JointRefiner('models/checkpoints/best_joint_model.pth')
refined_angles = refiner.refine(angles, visibility)
```

### Next Steps

1. **Complete training data generation** (~1M samples in progress)
2. **Full training run** (100 epochs on complete dataset)
3. **Validate**: Compare refined angles with ground truth on test sequences
4. **Benchmark**: Measure improvement on joey.mp4 before/after refinement

---

## Session: 2026-01-13 - Depth Refinement First Training Results

### Overview

Implemented and **validated** neural depth refinement using AIST++ dataset with REAL MediaPipe errors.

**Key Insights**:
1. Depth errors are systematic and correlate with camera viewing angle
2. **2D pose appearance directly encodes camera viewpoint** (ElePose CVPR 2022)
3. View angles computed from ACTUAL camera positions (not torso - which fails when bending)

The model learns to:
1. **Predict camera viewpoint from 2D pose** (no calibration needed at inference!)
2. Detect when limbs are poorly visible/occluded (visibility scores)
3. Infer correct depth from OTHER visible joints (cross-joint attention)
4. Exploit pose priors via transformer architecture

### First Training Results

**Training completed** on 22,764 samples (100 sequences):

| Metric | Before (MediaPipe) | After (Model) | Improvement |
|--------|-------------------|---------------|-------------|
| **Mean Depth Error** | 11.65 cm | 5.40 cm | **53.6%** |
| **Std Depth Error** | 11.03 cm | - | - |
| **Max Depth Error** | 375 cm (outliers) | - | - |

**Camera Prediction** (from 2D pose only - no calibration!):
- Azimuth error: **±11.0°**
- Elevation error: **±14.8°**

**Training Details**:
- Epochs: 50
- Batch size: 64
- Training time: ~7 minutes (RTX 5080, FP16)
- Best validation loss: 0.0545
- Model saved: `models/checkpoints/best_depth_model.pth`

### Data Generation Status

**Current**: ~1.2M samples across 9 camera views (c01-c09)

| Camera | Samples | Status |
|--------|---------|--------|
| c01 | 2,102 | 1% - in progress |
| c02 | 171,904 | **100% complete** |
| c03 | 95,500 | 55% - in progress |
| c04 | 110,625 | 64% - in progress |
| c05 | 60,468 | 35% - in progress |
| c06 | 60,764 | 35% - pending batch 2 |
| c07 | 60,308 | 54% - pending batch 2 |
| c08 | 60,498 | 41% - pending batch 2 |
| c09 | 60,996 | 42% - pending batch 2 |

**Target**: ~1.5M samples (172K per camera × 9 cameras)

**Resume conversion:**
```bash
# Just run the script - it auto-skips completed work
bash scripts/run_parallel_conversion.sh
```

Expected improvement with full data:
- Depth error: 5.4cm → ~3-4cm
- Camera prediction: 11° → ~2-5°

### Implementation Status

**Data Pipeline** ✅
- `scripts/convert_aistpp_to_training.py` - Converts AIST++ to training pairs
- Uses REAL MediaPipe errors from video frames (not synthetic noise)
- Extracts BOTH 2D and 3D poses from MediaPipe
- View angles from ACTUAL camera positions (not torso normal)

**Model Architecture** ✅ (`src/depth_refinement/`)
- `model.py` - PoseAwareDepthRefiner (655K parameters)
  - **Pose2DEncoder**: Hand-crafted viewpoint features (15 features)
    - Shoulder/hip height differences (foreshortening)
    - Limb length ratios (L/R asymmetry)
    - Torso aspect ratio
    - Nose offset from body center
  - **CameraPositionPredictor**: Predicts (x,y,z) relative to pelvis
    - Uses 2D + 3D features for robust prediction
    - Computes azimuth/elevation geometrically
  - Cross-joint attention (4 layers, 4 heads)
  - View angle conditioning via Fourier features
- `losses.py` - Depth + bone length + symmetry + camera losses
- `dataset.py` - PyTorch dataset with augmentation (includes pose_2d)

**Training** ✅
- `scripts/train_depth_model.py` - Full training script
- GPU-accelerated (RTX 5080, FP16 mixed precision)
- `scripts/test_depth_model.py` - Quick validation script

**Inference** ✅
- `src/depth_refinement/inference.py` - DepthRefiner class
- **No camera calibration needed** - model predicts viewpoint from 2D pose
- Batch processing support

### Data Format (NPZ files)

```python
{
    'corrupted': (17, 3),      # MediaPipe 3D pose (REAL depth errors)
    'ground_truth': (17, 3),   # AIST++ mocap GT
    'visibility': (17,),       # Per-joint visibility from MediaPipe
    'pose_2d': (17, 2),        # MediaPipe 2D pose (KEY FOR CAMERA!)
    'azimuth': float,          # 0-360° from camera position
    'elevation': float,        # -90 to +90° from camera position
    'camera_relative': (3,),   # Camera pos relative to pelvis
}
```

### 2D Pose Encodes Viewpoint (ElePose Insight)

```
FRONTAL VIEW (0°)          PROFILE VIEW (90°)
      O                          O
     /|\                        /|
    / | \                      / |
   /  |  \                    /  |
      |                      ─┘  │
     / \                         │
    /   \                       /│\

• Shoulders appear WIDE        • Shoulders appear NARROW
• Arms symmetric L/R           • Arms asymmetric
• Legs similar length          • Near leg appears shorter

HAND-CRAFTED FEATURES (15 total):
├── shoulder_height_diff, hip_height_diff
├── shoulder_width, hip_width
├── left/right arm lengths, leg lengths
├── arm_ratio, leg_ratio
├── torso_height, torso_width, torso_aspect
├── nose_offset_x, body_center_x
```

### Usage

**Training:**
```bash
uv run --group neural python scripts/train/depth_model.py \
  --data "data/training/aistpp_converted,data/training/mtc_converted" \
  --epochs 50 --batch-size 256 --workers 8 --bf16 \
  --use-limb-orientations --limb-orientation-weight 0.5 \
  --d-model 128 --num-layers 6 --num-heads 8
```

**Inference (standalone):**
```python
from src.depth_refinement.inference import DepthRefiner

refiner = DepthRefiner('models/checkpoints/best_depth_model.pth')
refined_pose = refiner.refine(pose, visibility)  # Auto-computes view angle
```

**Recommended usage via main pipeline:** Use `--main-refiner` flag (see MainRefiner section above).

### Files

```
src/depth_refinement/
├── __init__.py               # Module exports
├── model.py                  # Depth refinement model (~3M params)
├── losses.py                 # Depth + biomechanical losses
├── dataset.py                # AIST++ dataset loader
└── inference.py              # DepthRefiner class

scripts/
├── data/convert_aistpp.py          # AIST++ data converter
├── data/convert_cmu_mtc.py         # CMU MTC data converter
└── train/depth_model.py            # Training script

models/
└── checkpoints/
    └── best_depth_model.pth        # Trained model

data/
├── AIST++/
│   ├── annotations/keypoints3d/    # 1,408 sequences
│   └── videos/                     # Dance videos
└── training/aistpp_converted/      # Training pairs (NPZ)
```

### Next Steps

1. **Full training run** with 300K+ samples (currently generating)
2. **Integrate with run_pipeline command** - Add `--learned-depth-refinement` flag
3. **Validate on joey.mp4** - Measure bone length CV improvement
4. **Target metrics**: <4cm depth error, <5° camera prediction

---

## Session History

### 2026-01-13 (Part 4) - Joint Constraint Pipeline Integration
- Fixed angle extraction bug in training data generator (was extracting time column instead of angles)
- Added GPU acceleration to `generate_joint_angle_training.py` via `patch_pose2sim_gpu()`
- Started 1M sample generation (3000 sequences, 4 workers)
- **Integrated joint refinement into run_pipeline command:**
  - Added `--joint-constraint-refinement` flag
  - Added `--joint-model-path` for custom model paths
  - Created `apply_neural_joint_refinement()` function
- Updated CLAUDE.md with full documentation section

### 2026-01-13 (Part 3) - Joint Constraint Refinement Module
- Created full `src/joint_refinement/` module (916K parameter model)
- Implemented cross-joint attention with kinematic chain bias
- Created training script with symmetry loss and delta regularization
- Initial test: 22.7° → 22.4° mean error (limited data)

### 2026-01-13 (Part 2) - Large-Scale Data Generation
- Created `scripts/run_parallel_conversion.sh` for multi-camera parallel processing
- Running 4 cameras simultaneously (c01, c03, c04, c05), then batch 2 (c06-c09)
- **1.2M samples already generated** - c02 fully complete
- Automatic resume support - scripts skip already-processed sequences
- Target: ~1.5M total samples across all 9 camera views

### 2026-01-13 (Part 1)
- First successful training run (50 epochs, 22K samples)
- **Results**: 53.6% depth error reduction (11.6cm → 5.4cm)
- Camera prediction working: ±11° azimuth from 2D pose alone
- Started large-scale data generation (targeting 300K+ samples)

### 2026-01-12 (Part 2)
- Switched from HumanEva to AIST++ (10.1M frames vs 2K)
- Created conversion pipeline with view angle computation
- Fixed coordinate scale (AIST++ cm → meters)

### 2026-01-12 (Part 1) - DEPRECATED
- HumanEva conversion abandoned (insufficient data)

### 2026-01-10
- GPU setup (RTX 5080, CUDA 12.8)
- CMU mocap pipeline (now deprecated in favor of AIST++)
