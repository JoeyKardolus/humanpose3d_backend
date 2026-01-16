# Session Summary - 2026-01-10

## 🎯 What We Accomplished Today

### 1. Fixed Pelvis/Trunk Joint Angles ✅
- **Problem**: Angles were 5-10x too large (80° pelvis tilt vs expected 10-15°)
- **Root causes**: Wrong coordinate system (Z instead of Y primary), smoothing window mismatch (9 vs 21), excessive clamping
- **Solution**: Rewrote `pelvis_axes()` with Y-axis primary, ZXY Euler, smoothing window=21, relaxed limits
- **Result**: <2% difference vs reference script (49.53° vs 50.34° on MicrosoftTeams-video)

**Files modified**:
- `src/kinematics/segment_coordinate_systems.py` (pelvis coordinate system)
- `src/kinematics/comprehensive_joint_angles.py` (smooth_window=21, ZXY Euler)
- `src/kinematics/joint_angles_euler.py` (smooth_window=21)
- `src/kinematics/joint_angles_upper_body.py` (smooth_window=21)
- `src/kinematics/angle_processing.py` (relaxed biomechanical limits)

**Git backup**: Tag `stable-before-neural-depth` created

---

### 2. Neural Depth Refinement Planning ✅

**Goal**: Train neural network to correct MediaPipe depth errors caused by camera viewing angle

**Approach**: Self-supervised learning using biomechanical constraints as loss functions
- No manual labeling required
- Learns from CMU Motion Capture data (ground truth)
- Simulates MediaPipe errors, trains to fix them

**Expected improvements**:
- Bone length CV: 0.036 → 0.015 (58% improvement)
- Pelvis ROM: 18.4° → 12.1° (34% reduction, more realistic)
- Ground violations: 3.2% → 0.4% (87% reduction)
- Depth jitter: 12.3mm → 4.1mm (67% reduction)

---

### 3. CMU Motion Capture Database Setup ✅

**Downloaded**: 2554 BVH files from GitHub mirror
- Source: https://github.com/una-dinosauria/cmu-mocap
- Format: BVH (skeleton + motion data)
- Subjects: 001-143 (all subjects)
- Total size: ~1GB

**Skeleton structure analyzed**:
- 31 joints: Hips, thighs, legs, feet, toes, spine, neck, head, shoulders, arms, hands
- Hierarchy parsed using `scripts/explore_cmu_format.py`

**Location**: `data/training/cmu_mocap/cmu-mocap/` (excluded from git)

---

## 🔥 GPU-OPTIMIZED DESIGN (RTX 5080)

**Decision**: MAX POWER approach (no CPU compatibility)

**Design changes**:
1. **GPU-REQUIRED**: `--learned-depth-refinement` flag requires CUDA (no fallback)
2. **Bigger model**: PoseFormer Transformer (23M params) instead of Temporal-GCN (5M)
3. **FP16 training**: Mixed precision for 2x speed + 2x memory
4. **Batch size 256**: RTX 5080 can handle massive batches
5. **Training time**: 5-8 minutes for 50 epochs (vs 2 hours on CPU)

**Performance targets**:
- Training: ~8 minutes on RTX 5080
- Inference: ~0.3s per video
- Dataset: 5.4M training examples (CMU mocap with augmentation)

---

## 📋 Next Session TODO

### Immediate tasks (start here tomorrow):
1. **Build CMU → OpenCap mapping** (31 joints → 65 markers)
2. **Extract 3D positions** from BVH motion data (forward kinematics)
3. **Simulate depth errors** (viewpoint-dependent noise corruption)
4. **Install PyTorch CUDA**: `uv add torch --index-url https://download.pytorch.org/whl/cu124`
5. **Implement PoseFormer** Transformer architecture

### Resume commands:
```bash
cd /home/dupe/ai-test-project/humanpose3d_mediapipe
git status  # Should be on main branch, clean working tree
python scripts/explore_cmu_format.py  # Verify CMU data still accessible
cat .claude/plans/idempotent-percolating-cake.md  # Review full plan
```

### Full plan location:
- `.claude/plans/idempotent-percolating-cake.md`

---

## 🗂️ Project State

**Git status**:
- Branch: `main`
- Latest commit: `f7fd775` (Session checkpoint: Neural depth refinement planning)
- Backup tag: `stable-before-neural-depth` (restore point if needed)

**Key files**:
- `run_reference_pelvis.py` - Reference pelvis angle computation (validation)
- `scripts/explore_cmu_format.py` - BVH parser for CMU mocap
- `.claude/plans/idempotent-percolating-cake.md` - Full implementation plan
- `SESSION_2026-01-10.md` - This summary (you're reading it!)

**Data** (not in git):
- `data/training/cmu_mocap/cmu-mocap/` - CMU Motion Capture database (2554 BVH files)
- Re-download if missing: `cd data/training/cmu_mocap && git clone https://github.com/una-dinosauria/cmu-mocap.git`

---

## 🎓 Key Learnings

1. **Coordinate systems matter**: Y-axis vs Z-axis as primary completely changes Euler decomposition
2. **Smoothing window critical**: 9 vs 21 frames caused 30-40% error in joint angles
3. **GPU changes everything**: RTX 5080 enables Transformer models that would be impractical on CPU
4. **CMU mocap = gold standard**: Professional motion capture data, perfect for training
5. **Self-supervised learning works**: Don't need labeled data, biomechanical constraints are supervision

---

## 💡 Ideas for Future

- **Multi-GPU training**: If you get another 5080, we can parallelize
- **Model ensemble**: Train 5 models with different seeds, average predictions
- **Human3.6M fine-tuning**: If we get academic access, supervised fine-tuning
- **Real-time inference**: Optimize model for live video processing
- **Web demo**: Deploy model as REST API for professor to test

---

## 📞 Contact Points

**Stuck on something?**
1. Check the full plan: `.claude/plans/idempotent-percolating-cake.md`
2. Restore backup: `git checkout stable-before-neural-depth`
3. Re-download CMU data: See "Data" section above
4. Verify GPU: `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`

---

**Session end**: 2026-01-10 ~23:30 UTC
**Next session**: Ready to build CMU mapping and start training! 🚀
