# Augmented Marker Quality Filtering

**Date**: 2026-01-09
**Status**: Implemented and tested

## Problem

Pose2Sim's LSTM augmentation adds 43 anatomical markers to the original 22 MediaPipe markers. However, some augmented markers (particularly heels, medial knees/ankles) showed high temporal variance, indicating unreliable predictions. These noisy markers caused scattered visualization and introduced errors during multi-constraint optimization.

### Root Causes

1. **Rigid Kinematic Chains**: Augmented markers (medial knee, medial ankle, heels) were incorrectly included in rigid kinematic chains. When bone length constraints adjusted parent markers, children moved by the same rigid delta.

2. **Constraint Fighting**: Biomechanical angle computation uses BOTH lateral and medial markers to build segment coordinate systems. Moving them rigidly changed their relative positions, corrupting angle computations and creating feedback loops.

3. **Optimization-Induced Noise**: Markers with low initial variance (0.05) became very noisy (0.10+) AFTER optimization due to cascading constraint violations.

## Solution

### 1. Distance Constraints (Not Rigid Movement)

Changed augmented markers from rigid kinematic chains to distance constraints:

```python
# OLD (incorrect): Rigid movement
kinematic_chains = {
    "r_knee_study": ["r_ankle_study", "r_mknee_study"],  # Medial moves rigidly!
}

# NEW (correct): Distance constraints
kinematic_chains = {
    "r_knee_study": ["r_ankle_study"],  # Only main chain
}
augmented_constraints = {
    "r_knee_study": ["r_mknee_study"],  # Maintains fixed distance
}
```

**Implementation**: `apply_augmented_marker_distance_constraints()` in `multi_constraint_optimization.py`
- Computes median distance between parent and child across all frames
- Projects child marker onto sphere of that radius around parent
- Preserves direction (medial stays medial), adjusts only distance

### 2. Temporal Variance Filtering

Automatically filters unreliable augmented markers before optimization:

```python
def filter_unreliable_augmented_markers(
    coords: np.ndarray,
    marker_index: Dict[str, int],
    variance_threshold: float = 0.05,  # Tunable threshold
    verbose: bool = True,
) -> np.ndarray:
```

**Algorithm**:
1. For each augmented marker (not MediaPipe originals):
   - Compute temporal variance: `mean(var(coords, axis=0))`
   - If variance > threshold: set all frames to NaN
2. MediaPipe markers always preserved (confidence 95%+)
3. Typical result: 4-6 markers filtered (heels, some medial/lateral markers)

**Filtered Markers** (joey.mp4 example):
- `r_calc_study` (right heel, variance=0.057)
- `L_knee_study` (left lateral knee, variance=0.056)
- `r_toe_study` (right toe, variance=0.055)
- `r_mknee_study` (right medial knee, variance=0.052)

## Results

### Before Fix
- Bone length CV: 0.129 → 0.088 (31.5% improvement)
- Scattered augmented markers visible in visualization
- Heels extremely noisy (variance 0.10+)
- Medial knee markers unstable

### After Fix
- Bone length CV: 0.129 → 0.049 (**62.2% improvement**)
- No scattered markers - clean visualization
- 59/65 markers retained (4 unreliable filtered)
- All remaining markers stable

## Configuration

### Tuning Variance Threshold

Edit `src/anatomical/multi_constraint_optimization.py`:

```python
coords = filter_unreliable_augmented_markers(
    coords,
    marker_index,
    variance_threshold=0.05,  # Lower = more strict filtering
    verbose=verbose,
)
```

**Guidelines**:
- **0.03-0.04**: Very strict - filters ~8-10 markers
- **0.05** (default): Balanced - filters ~4-6 markers
- **0.08**: Lenient - filters ~2-3 markers
- **0.10+**: Minimal filtering - may allow noisy markers

### Disabling Filtering

To disable filtering (not recommended):

```python
# Comment out the filter call
# coords = filter_unreliable_augmented_markers(...)
```

## Technical Details

### Variance Computation

```python
# Get marker coordinates across all frames
marker_coords = coords[:, marker_idx, :]  # (num_frames, 3)

# Filter valid frames (not NaN)
valid_frames = ~np.isnan(marker_coords).any(axis=1)
valid_coords = marker_coords[valid_frames]

# Compute variance per axis, then mean
temporal_var = np.mean(np.var(valid_coords, axis=0))
# temporal_var = mean([var(x), var(y), var(z)])
```

**Interpretation**:
- Variance measures position stability across time
- Lower variance = more stable prediction
- High variance = jittery/scattered marker

### Distance Constraint Implementation

```python
def apply_augmented_marker_distance_constraints(
    coords: np.ndarray,
    marker_index: Dict[str, int],
    augmented_constraints: Dict[str, List[str]],
) -> np.ndarray:
    # For each parent → child pair
    for parent_name, child_names in augmented_constraints.items():
        # 1. Compute median distance (robust to outliers)
        distances = []
        for frame in frames:
            dist = norm(child_pos - parent_pos)
            distances.append(dist)
        target_distance = median(distances)

        # 2. Project child onto sphere around parent
        for frame in frames:
            vec = child_pos - parent_pos
            direction = vec / norm(vec)
            child_pos_new = parent_pos + direction * target_distance
```

**Why median?**: Robust to outliers from failed LSTM predictions

## Integration

### Pipeline Flow

```
1. MediaPipe extraction → CSV
2. TRC conversion
3. Pose2Sim augmentation (20 cycles)
4. Force-complete estimation
5. Multi-constraint optimization:
   ├─ Phase 0: Filter unreliable markers (NEW)
   ├─ Phase 1: Joint angle optimization
   ├─ Phase 2: Bone length + distance constraints (FIXED)
   └─ Phase 3: Ground plane + hip width
6. Output: Stable skeleton with 59-61 high-quality markers
```

### Multi-Constraint Optimization Updates

**Kinematic chains** (rigid movement only):
```python
kinematic_chains = {
    # Main chains only
    "RHJC_study": ["r_knee_study"],
    "r_knee_study": ["r_ankle_study"],
    "L_knee_study": ["L_ankle_study"],

    # Toes are children of heel
    "r_calc_study": ["r_toe_study", "r_5meta_study"],
}
```

**Augmented constraints** (distance only):
```python
augmented_constraints = {
    "r_knee_study": ["r_mknee_study"],  # Medial knee
    "L_knee_study": ["L_mknee_study"],
    "r_ankle_study": ["r_mankle_study", "r_calc_study"],  # Medial ankle + heel
    "L_ankle_study": ["L_mankle_study", "L_calc_study"],
}
```

## References

- `src/anatomical/multi_constraint_optimization.py` - Core implementation
- `CLAUDE.md` - User-facing documentation
- `docs/MULTI_CONSTRAINT_OPTIMIZATION.md` - Original optimization docs

## Future Work

- Adaptive variance thresholds per marker type
- Machine learning-based quality prediction
- Real-time confidence scores during augmentation
