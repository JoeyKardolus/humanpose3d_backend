# Multi-Constraint Optimization

## Overview

Multi-constraint optimization is an iterative refinement approach that applies multiple biomechanical constraints cyclically until convergence, preventing the "cascading violation" problem where fixing one constraint breaks another.

## The Problem

When applying biomechanical constraints independently:
1. Fixing bone lengths may violate joint angle limits
2. Correcting joint angles may change bone lengths
3. Adjusting depth for one marker may scatter connected markers
4. Each fix creates new violations elsewhere

**Result**: Constraints fight each other, limiting improvement to only 2-5%.

## The Solution

Apply constraints **iteratively in cycles** until all violations minimize together:

```
for iteration in 1..10:
    1. Fix bone lengths → may violate joint angles
    2. Fix joint angles → may violate bone lengths
    3. Fix ground plane → may violate everything
    4. Fix hip width → maintain proportions
    5. Re-constrain augmented markers → prevent scattering
    6. Smooth heel depth → reduce noise

    if improvement < 0.01%:
        break  # converged
```

## Architecture

### Constraint Hierarchy

**MediaPipe markers** (stable, from pose detection):
- Main joints: RKnee, LKnee, RAnkle, LAnkle
- Shoulders: RShoulder, LShoulder
- Corrected by: bone length, joint angle, ground plane, hip width constraints

**Augmented markers** (LSTM-generated, prone to scatter):
- Medial markers: r_mknee_study, L_mknee_study, r_mankle_study, L_mankle_study
- Heels: r_calc_study, L_calc_study
- Shoulder clusters: r_sh1_study, r_sh2_study, r_sh3_study, etc.
- **Strategy**: Maintain fixed distance from MediaPipe parent markers

### Parent-Child Relationships

```python
parent_child_pairs = [
    # Lower body
    ("r_knee_study", "r_mknee_study"),      # Knee → medial knee
    ("L_knee_study", "L_mknee_study"),
    ("r_ankle_study", "r_mankle_study"),    # Ankle → medial ankle
    ("L_ankle_study", "L_mankle_study"),
    ("r_ankle_study", "r_calc_study"),      # Ankle → heel
    ("L_ankle_study", "L_calc_study"),

    # Upper body
    ("RShoulder", "r_sh1_study"),           # Shoulder → cluster
    ("RShoulder", "r_sh2_study"),
    ("RShoulder", "r_sh3_study"),
    ("LShoulder", "L_sh1_study"),
    ("LShoulder", "L_sh2_study"),
    ("LShoulder", "L_sh3_study"),
]
```

## Implementation

### File: `src/anatomical/multi_constraint_optimization.py`

**Key function**: `multi_constraint_optimization()`

**Constraint cycle**:
1. **Bone Length** (`apply_bone_length_constraints_numpy`) - Restore consistent lengths in hip→knee→ankle chains
2. **Joint Angles** (`refine_depth_with_joint_constraints`) - Enforce biomechanical limits with depth adjustments
3. **Ground Plane** (`apply_ground_plane_to_coords`) - Clamp feet to estimated ground level
4. **Hip Width** (`constrain_hip_width`) - Maintain 0.20 × height ±15%
5. **Augmented Markers** - Reposition to fixed distance from parents
6. **Heel Smoothing** - Median filter on Z-axis (window=5)

**Convergence**: Stops when improvement < 0.01% or max iterations reached (default 10)

## Results

### Benchmark (joey.mp4, 615 frames, 20-cycle average)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Joint angle violations** | 1602 | 884 | **44.8%** |
| **Bone length CV** | 0.1291 | 0.0884 | **31.5%** |
| **Scattered markers** | Common | 0 | **100%** |
| **Processing time** | 32s | 45s | +13s |

### Comparison with Locked Markers Approach

Early attempts used "locking" (freezing marker positions):
- ❌ Locked heels: Joint angle improvement **2.1%** (constraints blocked)
- ❌ Locked ankles: Joint angle improvement **2.1%** (constraints blocked)
- ✅ **Parent-child constraints**: Joint angle improvement **44.8%** (constraints work freely!)

**Lesson**: Never lock markers - use relative constraints instead.

## Usage

### Command Line

```bash
uv run python main.py \
  --video data/input/video.mp4 \
  --height 1.78 \
  --mass 75 \
  --age 30 \
  --sex male \
  --anatomical-constraints \
  --bone-length-constraints \
  --estimate-missing \
  --force-complete \
  --augmentation-cycles 20 \
  --multi-constraint-optimization \
  --multi-constraint-iterations 10
```

### Python API

```python
from src.anatomical.multi_constraint_optimization import multi_constraint_optimization

coords_refined, stats = multi_constraint_optimization(
    coords,                    # (num_frames, num_markers, 3)
    marker_index,             # Dict[str, int]
    subject_height=1.78,      # meters
    max_iterations=10,
    bone_length_weight=1.0,   # Enable all constraints
    joint_angle_weight=1.0,
    ground_plane_weight=1.0,
    verbose=True
)

print(f"Joint angles: {stats['improvement']['joint_angles']:.1f}% better")
print(f"Bone lengths: {stats['improvement']['bone_lengths']:.1f}% better")
```

## When to Use

### ✅ **Use multi-constraint optimization when:**
- Performing biomechanical analysis (joint angles, kinetics)
- Computing ground reaction forces or inverse dynamics
- Augmented markers appear scattered in visualization
- Accuracy matters more than processing time

### ❌ **Skip multi-constraint optimization when:**
- Simple tracking or visualization tasks
- Real-time processing required
- Computing only gross motion parameters (center of mass, stride length)

## Limitations

1. **Temporal smoothness**: May decrease by 10-50% due to frame-by-frame optimization
   - **Mitigation**: Apply post-smoothing if needed for trajectory analysis

2. **Heel depth noise**: Heels inherit ankle depth variations
   - **Mitigation**: Temporal median filter applied (window=5)

3. **Processing time**: Adds ~13 seconds per video (32s → 45s)
   - **Acceptable**: Trade-off for 44.8% better joint angles

## Future Work

- [ ] Add temporal constraint to optimization cycle (smooth trajectories)
- [ ] Adaptive learning rates per constraint type
- [ ] GPU acceleration for parallel frame processing
- [ ] Constraint weights tunable per use-case (e.g., emphasize joint angles for gait analysis)

## References

- BioPose (2025): Multi-objective optimization for pose refinement
- MANIKIN (2024): Biomechanical constraint enforcement
- Bell et al. (1990): Hip joint center regression equations
- Davis et al. (1991): Pelvis anthropometry standards
