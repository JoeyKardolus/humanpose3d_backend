# Neural Depth Refinement - Progress Tracker

## Session: 2026-01-13 - 2D Pose Camera Prediction + Full Data Generation

### Overview

Implemented neural depth refinement using AIST++ dataset with REAL MediaPipe errors.

**Key Insights**:
1. Depth errors are systematic and correlate with camera viewing angle
2. **2D pose appearance directly encodes camera viewpoint** (ElePose CVPR 2022)
3. View angles computed from ACTUAL camera positions (not torso - which fails when bending)

The model learns to:
1. **Predict camera viewpoint from 2D pose** (no calibration needed at inference!)
2. Detect when limbs are poorly visible/occluded (visibility scores)
3. Infer correct depth from OTHER visible joints (cross-joint attention)
4. Exploit pose priors via transformer architecture

### Implementation Status

**Data Pipeline** ✅
- `scripts/convert_aistpp_to_training.py` - Converts AIST++ to training pairs
- Uses REAL MediaPipe errors from video frames (not synthetic noise)
- Extracts BOTH 2D and 3D poses from MediaPipe
- View angles from ACTUAL camera positions (not torso normal)
- Currently: **Regenerating ~900K+ training pairs** from 3119 videos

**Model Architecture** ✅ (`src/depth_refinement/`)
- `model.py` - PoseAwareDepthRefiner (226K parameters)
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
- GPU-accelerated (RTX 5080)
- Camera prediction verified: **1.4° test error** on held-out data
- Cross-camera generalization: Model predicts unseen camera angles within 4°

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
uv run --group neural python scripts/train_depth_model.py \
  --epochs 50 \
  --batch-size 64 \
  --fp16
```

**Inference:**
```python
from src.depth_refinement.inference import DepthRefiner

refiner = DepthRefiner('models/checkpoints/best_model.pth')
refined_pose = refiner.refine(pose, visibility)  # Auto-computes view angle
```

### Files

```
src/depth_refinement/
├── __init__.py               # Module exports
├── model.py                  # PoseAwareDepthRefiner (226K params)
├── losses.py                 # Depth + biomechanical losses
├── dataset.py                # AIST++ dataset loader
└── inference.py              # DepthRefiner class

scripts/
├── convert_aistpp_to_training.py   # Data converter
├── visualize_aistpp_training.py    # Visualization
└── train_depth_model.py            # Training script

data/
├── AIST++/
│   ├── annotations/keypoints3d/    # 1,408 sequences
│   ├── videos/                     # ~12,670 videos (downloading)
│   └── api/                        # Dataset API
└── training/aistpp_converted/      # Training pairs (NPZ)
```

### In Progress

- **Video download**: ~750/12,670 videos (~6%)
- **Training data**: 7K+ pairs generated

### Next Steps

1. **Full training run** once more data is available (~10K+ pairs recommended)
2. **Integrate with main.py** - Add `--learned-depth-refinement` flag
3. **Validate on joey.mp4** - Measure bone length CV improvement

---

## Session History

### 2026-01-12 (Part 2)
- Switched from HumanEva to AIST++ (10.1M frames vs 2K)
- Created conversion pipeline with view angle computation
- Fixed coordinate scale (AIST++ cm → meters)

### 2026-01-12 (Part 1) - DEPRECATED
- HumanEva conversion abandoned (insufficient data)

### 2026-01-10
- GPU setup (RTX 5080, CUDA 12.8)
- CMU mocap pipeline (now deprecated in favor of AIST++)
