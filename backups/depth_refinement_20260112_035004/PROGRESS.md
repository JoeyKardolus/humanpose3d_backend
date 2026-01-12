# Neural Depth Refinement - Progress Tracker

## Session: 2026-01-12 (Part 3) - AIST++ Implementation Complete

### Overview

Implemented neural depth refinement using AIST++ dataset with REAL MediaPipe errors.

**Key Insight**: Depth errors are systematic and correlate with view angle (computed from torso orientation).

The model learns to:
1. Detect when limbs are poorly visible/occluded (visibility scores)
2. Infer correct depth from OTHER visible joints (cross-joint attention)
3. Exploit pose priors via transformer architecture

### Implementation Status

**Data Pipeline** ✅
- `scripts/convert_aistpp_to_training.py` - Converts AIST++ to training pairs
- Uses REAL MediaPipe errors from video frames (not synthetic noise)
- Computes view angle from GT torso plane (cross product of shoulder × hip vectors)
- Currently: 7K+ training pairs generated, ~750 videos downloaded

**Model Architecture** ✅ (`src/depth_refinement/`)
- `model.py` - PoseAwareDepthRefiner (226K parameters)
  - Cross-joint attention (4 layers, 4 heads)
  - View angle conditioning via Fourier features
  - Per-joint confidence prediction
- `losses.py` - Depth + bone length + symmetry losses
- `dataset.py` - PyTorch dataset with augmentation

**Training** ✅
- `scripts/train_depth_model.py` - Full training script
- GPU-accelerated (RTX 5080)
- Quick test: 14.78cm → 9.32cm depth error in 2 epochs

**Inference** ✅
- `src/depth_refinement/inference.py` - DepthRefiner class
- Auto-computes view angle from pose
- Batch processing support

### Data Format (NPZ files)

```python
{
    'corrupted': (17, 3),      # MediaPipe pose (REAL depth errors)
    'ground_truth': (17, 3),   # AIST++ mocap GT
    'view_angle': float,       # From torso (0-90°)
    'visibility': (17,),       # Per-joint visibility
}
```

### View Angle Computation

```python
def compute_view_angle_from_torso(pose_3d):
    # Torso plane from shoulders and hips
    shoulder_vec = right_shoulder - left_shoulder
    hip_vec = right_hip - left_hip

    # Normal points forward from body
    torso_normal = normalize(cross(shoulder_vec, hip_vec))

    # Angle between torso normal and camera Z-axis
    camera_dir = [0, 0, 1]
    cos_angle = abs(dot(torso_normal, camera_dir))

    return degrees(arccos(cos_angle))  # 0°=frontal, 90°=profile
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
