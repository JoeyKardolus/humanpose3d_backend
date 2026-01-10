# Neural Depth Refinement - Progress Tracker

## Session: 2026-01-10 (Continued)

### Completed

**GPU Setup**
- ✅ PyTorch 2.9.1+cu128 installed with CUDA support
- ✅ RTX 5080 verified (16GB VRAM, CUDA 12.8)
- ✅ Command: `uv run --group neural python`

**Data Pipeline**
- ✅ CMU Motion Capture mapping created (31 joints → 59/65 markers)
- ✅ BVH parser with forward kinematics working
- ✅ Training data generator written (corrupted + ground truth pairs)
- ✅ Supports multiple camera angles (0°-75°) and noise levels (30-80mm)

### In Progress

**Data Download**
- 🔄 Re-downloading CMU mocap dataset (2554 BVH files)
- Location: `data/training/cmu_mocap/cmu-mocap/`
- Note: Excluded from git (too large), needs re-download after clone

### Next Steps

1. **Generate Training Examples** (~30 min)
   - Run `training/generate_training_data.py`
   - Start with Subject 001 (14 sequences)
   - Verify data quality and format

2. **Implement PoseFormer** (~2 hours)
   - Transformer architecture with temporal attention
   - 8 heads, 6 layers, 512 hidden dim
   - Input: 65 markers × 11 frames × features
   - Output: Δz corrections per marker

3. **Biomechanical Loss Functions** (~1 hour)
   - Bone length consistency
   - Joint angle ROM constraints
   - Ground plane contact
   - Left/right symmetry
   - Temporal smoothness

4. **Training Script** (~1 hour)
   - FP16 mixed precision
   - Batch size 256
   - DataLoader with 12 workers
   - Tensorboard logging

5. **Train Model** (~5-8 min on RTX 5080)
   - 50 epochs
   - Early stopping on validation loss
   - Save checkpoints

6. **Integration** (~1 hour)
   - Add `--learned-depth-refinement` flag to main.py
   - Lazy import (only when flag used)
   - Inference module with GPU support

7. **Testing** (~30 min)
   - Run on joey.mp4, MicrosoftTeams-video.mp4
   - Measure bone length CV improvement
   - Compare joint angle quality

### Expected Results

- Bone length CV: 0.036 → 0.015 (58% improvement)
- Pelvis ROM: 18.4° → 12.1° (34% reduction)
- Ground violations: 3.2% → 0.4% (87% reduction)
- Depth jitter: 12.3mm → 4.1mm (67% reduction)

### Files Structure

```
training/
├── bvh_to_positions.py           # BVH parser + forward kinematics
├── cmu_to_opencap_mapping.py     # CMU → OpenCap marker mapping
└── generate_training_data.py     # Training data generator

data/training/
├── cmu_mocap/                    # CMU dataset (not in git)
│   └── cmu-mocap/
│       └── data/
│           ├── 001/              # Subject 001
│           ├── 002/              # Subject 002
│           └── ...               # Subjects 003-143
└── cmu_converted/                # Training pairs (NPZ files)
    ├── 01_01_f0000_a00_n030.npz
    ├── 01_01_f0000_a00_n050.npz
    └── ...

src/anatomical/
├── depth_model.py                # PoseFormer (to implement)
├── learned_depth_refinement.py   # Inference module (to implement)
└── biomechanical_losses.py       # Loss functions (to implement)

scripts/
└── train_depth_model.py          # Training script (to implement)
```

### Commands

**Download CMU Mocap**:
```bash
mkdir -p data/training/cmu_mocap
cd data/training/cmu_mocap
git clone https://github.com/una-dinosauria/cmu-mocap.git
```

**Generate Training Data**:
```bash
python training/generate_training_data.py
```

**Train Model** (after implementation):
```bash
uv run --group neural python scripts/train_depth_model.py \
  --data data/training/cmu_converted \
  --epochs 50 \
  --batch-size 256 \
  --fp16
```

**Test Pipeline**:
```bash
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 --mass 75 --age 30 --sex male \
  --force-complete \
  --learned-depth-refinement \
  --compute-all-joint-angles
```

### Time Estimates

- Training data generation (Subject 001): ~5 minutes
- PoseFormer implementation: ~2 hours
- Training on RTX 5080: ~5-8 minutes
- Integration + testing: ~1.5 hours

**Total remaining: ~4 hours** to working prototype
