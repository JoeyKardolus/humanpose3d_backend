# Neural Models

## MainRefiner (Recommended)

The **MainRefiner** is a unified neural pipeline that combines POF 3D reconstruction and joint constraint refinement:

```bash
# Full pipeline with neural refinement
uv run python manage.py run_pipeline \
  --video ~/.humanpose3d/input/joey.mp4 \
  --height 1.78 --weight 75 \
  --estimate-missing --force-complete \
  --augmentation-cycles 20 \
  --main-refiner \
  --plot-all-joint-angles
```

**How it works:**
1. **Stage 1 (Pre-augmentation)**: POF reconstructs 3D from 2D keypoints (17 COCO joints)
2. **Stage 2 (Post-augmentation)**: Joint constraint refinement applies learned soft constraints to computed angles

**Training:**
```bash
uv run --group neural python scripts/train/main_refiner.py \
  --data "data/training/aistpp_converted,data/training/mtc_converted" \
  --pof-checkpoint models/checkpoints/best_pof_model.pth \
  --joint-checkpoint models/checkpoints/best_joint_model.pth \
  --epochs 50 --batch-size 256 --workers 8 --bf16
```

**Model specs**: 1.2M params (d_model=128, 4 heads, 2 layers), <10ms inference per frame

---

## POF (Part Orientation Fields) Model

### Overview

POF reconstructs 3D poses from 2D keypoints by predicting per-limb 3D unit vectors. Based on MonocularTotalCapture (CVPR 2019).

**Key Insight**: 2D foreshortening directly encodes 3D limb orientation. A foreshortened limb in 2D is pointing toward/away from the camera.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              POF (Part Orientation Fields)              │
│                                                         │
│  Input: 2D keypoints (17, 2) + visibility (17,)         │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Pose2D Encoder (joint positions + foreshorten) │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Transformer (6 layers, 8 heads, d_model=128)   │   │
│  │  - Cross-limb attention                          │   │
│  │  - Learns depth correlations between limbs       │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Per-Limb Orientation Heads (14 × MLP)           │   │
│  │  - Outputs unit vectors (x, y, z)                │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Least-Squares Solver (MTC-style)                │   │
│  │  - Reconstructs 3D joints from limb vectors      │   │
│  │  - Ensures consistency with 2D observations      │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Output: 3D pose (17, 3) in camera space              │
└─────────────────────────────────────────────────────────┘
```

**Model size**: ~3M parameters

### 14 Limbs (COCO-17 indices)

| Limb | Parent → Child | Limb | Parent → Child |
|------|----------------|------|----------------|
| L upper arm | 5→7 | R upper arm | 6→8 |
| L forearm | 7→9 | R forearm | 8→10 |
| L thigh | 11→13 | R thigh | 12→14 |
| L shin | 13→15 | R shin | 14→16 |
| Shoulder width | 5↔6 | Hip width | 11↔12 |
| L torso | 5→11 | R torso | 6→12 |
| L cross-body | 5→12 | R cross-body | 6→11 |

### Performance

| Source | POF Error (limb orientation) |
|--------|------------------------------|
| MediaPipe 3D | 16.3° mean (11.9° median) |
| POF model (from 2D) | ~11° (beats MP 3D!) |

The POF model learns 2D→3D mapping directly, bypassing MediaPipe's broken depth estimation.

### Metric Scale Recovery

The POF model works in normalized space (unit torso scale) internally. True metric output is recovered using known subject height:

```
metric_torso = height / HEIGHT_TO_TORSO_RATIO  (where ratio ≈ 3.4)
```

- `--height 1.78` → torso = 0.524m → all bone lengths in true meters
- Training uses normalized space (scale-invariant POF directions)
- Inference denormalizes using height-derived metric scale

### Training

```bash
# Train POF model
uv run --group neural python scripts/train/pof_model.py \
  --data "data/training/aistpp_converted,data/training/mtc_converted" \
  --epochs 50 --batch-size 256 --workers 8 --bf16 \
  --d-model 128 --num-layers 6 --num-heads 8

# Use trained model explicitly
--camera-pof --pof-model-path models/checkpoints/best_pof_model.pth
```

### Data Format (NPZ files)

```python
{
    'corrupted': (17, 3),      # MediaPipe 3D pose (reference only)
    'ground_truth': (17, 3),   # AIST++ mocap GT
    'visibility': (17,),       # Per-joint visibility
    'pose_2d': (17, 2),        # MediaPipe 2D detection (PRIMARY INPUT)
    'projected_2d': (17, 2),   # GT 3D projected to 2D (for POF)
    'camera_R': (3, 3),        # Camera rotation matrix
    'gt_scale': float,         # Ground truth torso length in meters
}
```

### Files

```
src/pof/
├── __init__.py               # Module exports
├── model.py                  # CameraPOFModel (~3M params)
├── inference.py              # CameraPOFInference class
├── reconstruction.py         # Least-squares solver
└── dataset.py                # AIST++ dataset loader

scripts/
├── data/convert_aistpp.py    # AIST++ data converter
├── data/convert_cmu_mtc.py   # CMU MTC data converter
└── train/pof_model.py        # Training script

models/checkpoints/
└── best_pof_model.pth        # Trained model
```

---

## Joint Constraint Refinement Model

### Overview

Learns soft joint constraints from AIST++ motion capture data. Transformer-based model corrects joint angles using cross-joint attention.

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

### Performance

- **Training**: 660K samples from AIST++ (6 camera views), 100 epochs
- **Mean correction**: 3.47°, handles errors up to 73°

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
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 --mass 75 \
  --main-refiner \
  --plot-all-joint-angles
```

**Standalone Inference:**
```python
from src.joint_refinement.inference import JointRefiner

refiner = JointRefiner('~/.humanpose3d/models/checkpoints/best_joint_model.pth')
refined_angles = refiner.refine(angles, visibility)
```

### Files

```
src/joint_refinement/
├── __init__.py               # Module exports
├── model.py                  # JointConstraintRefiner (916K parameters)
├── losses.py                 # Reconstruction + symmetry + delta regularization
├── dataset.py                # PyTorch dataset with L/R swap augmentation
└── inference.py              # JointRefiner class

scripts/
├── data/generate_joint_angles.py  # Training data generator
└── train/joint_model.py           # Training script

models/checkpoints/
└── best_joint_model.pth           # Trained model
```

---

## Model Summary

| Model | Params | Purpose | Input | Output |
|-------|--------|---------|-------|--------|
| POF | ~3M | 3D reconstruction from 2D | 2D keypoints | 3D pose |
| Joint Refiner | ~916K | Joint angle correction | Joint angles | Refined angles |
| MainRefiner | ~1.2M | Fusion gating (POF + joint) | Both above | Fused output |

**Total inference**: ~4.8M params, <10ms per frame on CPU
