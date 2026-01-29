# GNN-Based POF Model

Graph Neural Network alternative to the transformer-based POF model for predicting Part Orientation Fields from 2D keypoints.

## Overview

The skeleton is naturally a graph structure (joints = nodes, bones = edges), making GNNs a strong fit for pose estimation. Instead of dense all-to-all attention (transformer), GNNs propagate information only along anatomical connections.

**Key advantages over transformer:**
- Sparse skeletal graph vs dense attention
- Fewer parameters (~855K vs ~601K)
- Faster inference (O(n×k) vs O(n²))
- Built-in anatomical inductive bias
- Better interpretability

## Architecture

### Model Variants

| Model | Description | Params (d=192, L=4) |
|-------|-------------|---------------------|
| `gcn` | Basic GCN with combined adjacency | ~300K |
| `semgcn` | Semantic GCN with 3 edge types | ~855K |

### SemGCN (Recommended)

Uses three separate GCN stacks for different relationship types:

```
Input: 2D pose + visibility + foreshortening features
    ↓
┌─────────────────────────────────────────────────────┐
│  Per-limb feature encoding (9 features → d_model)   │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Joint-Share │  │  Kinematic  │  │  Symmetry   │
│    GCN      │  │     GCN     │  │     GCN     │
│  (4 layers) │  │  (4 layers) │  │  (4 layers) │
└─────────────┘  └─────────────┘  └─────────────┘
    ↓                  ↓                ↓
┌─────────────────────────────────────────────────────┐
│           Fusion (concat → linear → d_model)         │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│    Global context + POF head → 14 unit vectors       │
└─────────────────────────────────────────────────────┘
```

## Graph Structure

### Nodes: 14 Limbs

| Index | Limb | Parent → Child |
|-------|------|----------------|
| 0 | L_upper_arm | 5 → 7 |
| 1 | L_forearm | 7 → 9 |
| 2 | R_upper_arm | 6 → 8 |
| 3 | R_forearm | 8 → 10 |
| 4 | L_thigh | 11 → 13 |
| 5 | L_shin | 13 → 15 |
| 6 | R_thigh | 12 → 14 |
| 7 | R_shin | 14 → 16 |
| 8 | shoulder_width | 5 → 6 |
| 9 | hip_width | 11 → 12 |
| 10 | L_torso | 5 → 11 |
| 11 | R_torso | 6 → 12 |
| 12 | L_cross | 5 → 12 |
| 13 | R_cross | 6 → 11 |

### Edge Types

**1. Joint-Sharing (56 edges)**
Limbs connected if they share at least one joint:
```
L_upper_arm ↔ L_forearm     (share elbow)
L_upper_arm ↔ L_torso       (share L shoulder)
L_upper_arm ↔ shoulder_width (share L shoulder)
```

**2. Kinematic (32 edges)**
Parent-child dependencies in reconstruction order:
```
L_torso → L_upper_arm → L_forearm
hip_width → L_thigh → L_shin
```

**3. Symmetry (12 edges)**
Left-right pairs for bilateral constraints:
```
L_upper_arm ↔ R_upper_arm
L_forearm ↔ R_forearm
L_thigh ↔ R_thigh
L_shin ↔ R_shin
L_torso ↔ R_torso
L_cross ↔ R_cross
```

## Model Sizes

| d_model | layers | Parameters |
|---------|--------|------------|
| 128 | 4 | 385K |
| 128 | 6 | 486K |
| **192** | **4** | **855K** |
| 192 | 6 | 1.08M |
| 256 | 4 | 1.5M |
| 256 | 6 | 1.9M |

**Recommended**: d_model=192, layers=4 (~855K params)

## Training

### Basic Training

```bash
uv run --group neural python scripts/train/pof_gnn_model.py \
  --data data/training/aistpp_converted \
  --model-type semgcn \
  --d-model 192 \
  --num-layers 4 \
  --epochs 50 \
  --batch-size 256 \
  --workers 8 \
  --bf16
```

### With Multiple Data Sources

```bash
uv run --group neural python scripts/train/pof_gnn_model.py \
  --data "data/training/aistpp_converted,data/training/mtc_converted" \
  --model-type semgcn \
  --d-model 192 \
  --num-layers 4 \
  --epochs 50 \
  --batch-size 256 \
  --workers 8 \
  --bf16
```

### Resume from Checkpoint

```bash
uv run --group neural python scripts/train/pof_gnn_model.py \
  --data data/training/aistpp_converted \
  --model-type semgcn \
  --checkpoint models/checkpoints/pof_semgcn_epoch_25.pth \
  --epochs 50
```

### Training Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-type` | semgcn | Model: `gcn` or `semgcn` |
| `--d-model` | 128 | Hidden dimension |
| `--num-layers` | 4 | Number of GCN layers per edge type |
| `--dropout` | 0.1 | Dropout rate |
| `--use-gat` | false | Use Graph Attention (SemGCN only) |
| `--lr` | 1e-4 | Learning rate |
| `--batch-size` | 256 | Batch size |
| `--epochs` | 50 | Training epochs |
| `--bf16` | false | Use BF16 mixed precision |
| `--save-every` | 10 | Save checkpoint every N epochs |

## Checkpoints

Saved to `models/checkpoints/`:

| File | Description |
|------|-------------|
| `best_pof_semgcn_model.pth` | Best validation loss |
| `pof_semgcn_epoch_N.pth` | Periodic checkpoint |
| `pof_semgcn_final.pth` | Final epoch |

Note: GNN checkpoints are separate from transformer (`best_pof_model.pth`).

## Usage in Code

### Create Model

```python
from src.pof import create_gnn_pof_model, SemGCNPOFModel

# Create SemGCN model
model = create_gnn_pof_model(
    model_type="semgcn",
    d_model=192,
    num_layers=4,
)

# Or create directly
model = SemGCNPOFModel(d_model=192, num_layers=4)
```

### Load Trained Model

```python
from src.pof import load_gnn_pof_model

model = load_gnn_pof_model(
    "models/checkpoints/best_pof_semgcn_model.pth",
    device="cuda"
)
```

### Forward Pass

```python
import torch

# Same interface as transformer model
pose_2d = torch.randn(batch_size, 17, 2)        # Normalized 2D keypoints
visibility = torch.ones(batch_size, 17)          # Joint visibility
limb_delta_2d = torch.randn(batch_size, 14, 2)  # 2D limb directions
limb_length_2d = torch.rand(batch_size, 14)     # 2D limb lengths

pof = model(pose_2d, visibility, limb_delta_2d, limb_length_2d)
# pof: (batch_size, 14, 3) unit vectors
```

### Inspect Graph Structure

```python
from src.pof import (
    build_joint_sharing_adj,
    build_kinematic_adj,
    build_symmetry_adj,
)

adj_joint = build_joint_sharing_adj()    # (14, 14)
adj_kin = build_kinematic_adj()          # (14, 14)
adj_sym = build_symmetry_adj()           # (14, 14)

print(f"Joint-sharing edges: {int(adj_joint.sum())}")
print(f"Kinematic edges: {int(adj_kin.sum())}")
print(f"Symmetry edges: {int(adj_sym.sum())}")
```

## Comparison: Transformer vs GNN

| Aspect | Transformer | SemGCN |
|--------|-------------|--------|
| Parameters | 601K | 855K |
| Attention | Dense (all-to-all) | Sparse (graph neighbors) |
| Inductive bias | Learned | Anatomical structure |
| Edge types | None | 3 (joint, kinematic, symmetry) |
| Interpretability | Harder | Easier |

## Files

| File | Description |
|------|-------------|
| `src/pof/gnn_model.py` | GCN and SemGCN model classes |
| `src/pof/graph_utils.py` | Adjacency matrix builders |
| `scripts/train/pof_gnn_model.py` | Training script |

## Expected Results

Training should show:
- Initial angular error: ~50°
- After 50 epochs: ~15-20° (similar to transformer)
- Potentially better bone length consistency due to graph structure
