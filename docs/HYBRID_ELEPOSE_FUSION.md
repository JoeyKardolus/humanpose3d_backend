# Hybrid ElePose Fusion for Angle Prediction

## Overview

Combines ElePose deep 2D foreshortening features with existing transformer context for improved camera angle prediction (both azimuth and elevation).

## Architecture

```
2D pose ──→ ElePoseBackbone ──→ elepose_features (1024)
                                       ↓
                                   [concat]
                                       ↓
joint_features + 3D + vis ──→ existing_features (~1536)
                                       ↓
                                 fusion_head
                                       ↓
                              ┌────────┴────────┐
                              ↓                  ↓
                         az (sin,cos)        elevation
```

### Components

| Branch | Input | Output | Info Source |
|--------|-------|--------|-------------|
| ElePose | 2D pose (34) | 1024-dim features | Deep 2D foreshortening patterns |
| Existing | joint_features + 3D + vis | ~1536-dim features | Transformer context, 3D structure |
| Fusion | concat(elepose, existing) | 3 values | Combined prediction |

### Why Hybrid Fusion

| Method | Azimuth Info | Elevation Info |
|--------|--------------|----------------|
| Existing only | 3D pose, transformer context | Same |
| ElePose only | Deep 2D foreshortening | Deep 2D foreshortening |
| **Hybrid** | Both signals combined | Both signals combined |

## Parameter Count

| Model | Params |
|-------|--------|
| Base (no ElePose) | 2.2M |
| Hybrid (with ElePose) | 7.0M |
| ElePose adds | +4.8M |

## Key Classes

### ElePoseBackbone
Shared feature extractor based on ElePose (CVPR 2022):
- Input: 2D pose (batch, 17, 2)
- Upscale: 34 → 1024 with BatchNorm + LeakyReLU
- 2 ResidualBlocks for feature refinement
- Output: (batch, 1024) deep 2D features

### DirectAnglePredictor (updated)
Hybrid fusion for angle prediction:
- `use_elepose=False`: Uses existing features only
- `use_elepose=True`: Fuses ElePose + existing features

## Usage

### Training
```bash
uv run python scripts/train_depth_model.py \
  --epochs 50 --batch-size 384 --fp16 \
  --d-model 128 --num-layers 6 --num-heads 8 \
  --elepose \
  --angle-noise-std 5.0 --angle-noise-warmup 10
```

### CLI Flags
| Flag | Description |
|------|-------------|
| `--elepose` | Enable hybrid fusion (ElePose + existing) |
| `--elepose-hidden-dim` | ElePose backbone dimension (default: 1024) |

### Code
```python
from src.depth_refinement.model import create_model

# Without ElePose (2.2M params)
model = create_model(d_model=128, use_elepose=False)

# With ElePose hybrid fusion (7.0M params)
model = create_model(d_model=128, use_elepose=True)
```

## References

- [ElePose: Unsupervised 3D Human Pose Estimation by Predicting Camera Elevation and Learning Normalizing Flows on 2D Poses (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Wandt_ElePose_Unsupervised_3D_Human_Pose_Estimation_by_Predicting_Camera_Elevation_CVPR_2022_paper.pdf)
- Key insight: 2D pose foreshortening directly encodes camera viewpoint
