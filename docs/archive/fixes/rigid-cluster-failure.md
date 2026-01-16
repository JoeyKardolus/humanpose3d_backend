# Rigid Cluster Constraints - Failed Approach Analysis

**Date**: January 2026
**Status**: ❌ Failed - Reverted from pipeline

## Summary

Attempted to use Procrustes-based rigid body constraints to reduce noise in augmented markers. **Result: Increased noise** instead of reducing it, particularly on Z-axis.

## What We Tried

### Approach
- Created `src/anatomical/rigid_cluster_constraints.py`
- Used Orthogonal Procrustes Analysis (Kabsch algorithm)
- Computed median template from first 50 frames for each marker cluster
- Applied rigid body transformation (rotation + translation) to each frame to match template

### Marker Clusters
- Shoulder clusters (3 markers each)
- Thigh clusters (3 markers each)
- Foot clusters (heel + toes)
- Elbow/wrist/knee/ankle clusters (2 markers each)

### Implementation Details
- Median template from 50 frames (robust to single-frame noise)
- Markers valid in >25% of frames included in template
- Blend weight 0.9 (90% constraint, 10% original)
- Maximum movement threshold: 15cm per frame
- SVD-based rotation estimation with reflection check

## What Actually Happened

### Noise Increased
- **Z-axis noise got significantly worse** after applying constraints
- Thigh clusters showed more scatter, not less
- Left knee augmented markers became noisier
- Right arm markers had extreme noise

### Specific Failures
1. **Right shoulder cluster**: Completely skipped - all 3 markers had no valid data in template
2. **Thigh markers**: Applied but introduced Z-axis artifacts
3. **Knee markers**: More scattered than before constraint application

### Visual Comparison
```
Before rigid constraints: Moderate Z-axis noise (~50-100mm variance)
After rigid constraints:  High Z-axis noise (~150-300mm variance)
```

## Why It Failed

### 1. LSTM Markers ≠ Physical Measurements
- Pose2Sim's LSTM **predicts** marker positions based on learned patterns
- These predictions don't perfectly follow rigid body physics
- Forcing them into rigid templates conflicts with LSTM's learned behavior

### 2. Template Captures Systematic Errors
- Median from first 50 frames **locked in LSTM's systematic biases**
- If LSTM consistently misplaces markers in certain poses, template captures those errors
- Propagates errors to all frames instead of correcting them

### 3. Procrustes Introduces Artifacts
- Forcing non-rigid data into rigid templates creates **artificial corrections**
- Z-axis (depth) is least constrained by LSTM, most affected by rigid constraints
- Small rotational errors amplified across marker clusters

### 4. Low Confidence Markers Excluded
- Markers valid in <25% of frames excluded from template
- These are often the noisiest markers that need constraints most
- Left unconstrained while "good" markers get over-constrained

## Key Lessons

### ❌ Don't Do This
- **Don't force LSTM predictions into rigid body templates**
- **Don't use Procrustes on marker clusters from learned models**
- **Don't compute templates from early frames only** (may capture initialization artifacts)

### ✅ What Works Instead
- **Relative parent-child constraints** (e.g., heel stays fixed distance from ankle)
- **Temporal smoothing** (median filters on Z-axis)
- **Biomechanical constraints** (joint angle limits, bone length consistency)
- **Let MediaPipe markers correct freely**, constrain augmented markers relative to them

## Technical Details

### Procrustes Algorithm Used
```python
# Cross-covariance matrix
H = source.T @ target

# SVD decomposition
U, S, Vt = np.linalg.svd(H)

# Optimal rotation (ensure proper rotation, not reflection)
R = Vt.T @ U.T
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = Vt.T @ U.T
```

### Diagnostic Output Example
```
[r_shoulder] SKIP: no valid markers
  Excluded: r_sh1_study, r_sh2_study, r_sh3_study

[r_thigh] Template: 3 markers
  Included: r_thigh1_study, r_thigh2_study, r_thigh3_study
  Skipped: 47/615 frames (7.6%) - movement >15cm
```

## Current Status

- Code preserved in `src/anatomical/rigid_cluster_constraints.py`
- **NOT used in main pipeline**
- Available via `--rigid-clusters` flag for experimentation
- Multi-constraint optimization does NOT use rigid clusters

## References

- MOSHFIT (2013): Occlusion-tolerant rigid body fitting - works for motion capture, NOT for LSTM predictions
- End-to-End Motion Capture (2025): Rigid body marker constraints - assumes physical measurements, not predictions
- Our finding: These classical approaches fail when applied to learned model outputs

## Related Files

- `src/anatomical/rigid_cluster_constraints.py` - Implementation (unused)
- `CHANGES.md` - Summary of failed approach
- `main.py` - Rigid cluster integration removed (lines 500-544 reverted)
