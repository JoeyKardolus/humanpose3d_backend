# Part Orientation Fields (POF) - Math Explanation

## What We Were Doing (Global Camera Angles)

```
From torso (4 points: shoulders + hips):
  → Compute body coordinate frame (forward, right, up axes)
  → Camera position relative to this frame
  → Single azimuth (0-360°) + elevation (-90 to +90°)

This gives ONE viewing direction for the ENTIRE skeleton.
```

The problem: When arm points toward camera and leg points sideways, a single global angle can't capture both depth ambiguities.

## POFs (Per-Limb Version)

For each limb (e.g., upper arm from shoulder→elbow):

```
Ground truth orientation:
  vec = elbow_3d - shoulder_3d        # 3D vector
  unit_vec = vec / ||vec||            # Normalize to unit length

  Result: (x, y, z) where x² + y² + z² = 1
```

So instead of 2 global angles, you get **14 limbs × 3D = 42 values**.

## The Key Insight

Each limb's unit vector encodes its **direction in camera space**:

```
Limb pointing straight at camera:    (0, 0, -1)  ← Z toward viewer
Limb pointing right:                 (1, 0, 0)   ← X axis
Limb pointing down:                  (0, -1, 0)  ← Y axis
Limb at 45° diagonal:                (0.7, 0, -0.7)
```

The **2D foreshortening** of each limb directly reveals its Z component:
- Limb looks short in 2D → pointing toward/away from camera (large |Z|)
- Limb looks long in 2D → perpendicular to camera (small |Z|)

## The Front/Back Ambiguity Problem

2D foreshortening tells us |Z| (magnitude) but NOT the sign!
- A foreshortened limb could be pointing AT the camera (Z < 0) or AWAY from it (Z > 0)
- Both look identical in 2D

**Solution**: Camera direction vector

```python
# Azimuth convention: 0°=front, 90°=right, 180°=back, 270°=left
# Compute explicit "toward camera" direction in world coords:

cos_el = cos(elevation)
cam_dir_x = sin(azimuth) * cos_el   # +X when camera is to the right
cam_dir_y = sin(elevation)          # +Y when camera is above
cam_dir_z = cos(azimuth) * cos_el   # +Z when camera is in front (az=0°)

# This tells the model: "toward camera is THIS direction"
# Now model can disambiguate: foreshortened limb pointing in cam_dir → Z < 0
```

The model combines:
1. 2D foreshortening → tells magnitude |Z|
2. Camera direction → tells sign of Z (toward vs away)

## Loss Function

Cosine similarity between predicted and GT orientations:

```python
cos_sim = dot(pred_unit_vec, gt_unit_vec)  # Range: -1 to +1
loss = 1 - cos_sim                          # Range: 0 to 2

# Perfect alignment: cos_sim = 1, loss = 0
# Perpendicular:     cos_sim = 0, loss = 1
# Opposite:          cos_sim = -1, loss = 2
```

## Comparison

| Approach | Frame | Output | Depth Info |
|----------|-------|--------|------------|
| Global (torso) | Whole body | 2 angles | Same for all joints |
| POF (per-limb) | Each segment | 14 × 3D vectors | Independent per limb |

It's the same geometric principle (orientation encodes depth) but **localized**. Each limb segment gets its own "mini camera angle" that can differ from its neighbors.

## Why This Helps

Self-occlusion example - person reaching toward camera with right arm:

```
Global approach:
  Torso facing ~30° right
  → Applies same rotation to all joints
  → Right arm correction is wrong (it's at ~90° to torso)

POF approach:
  Torso limbs: ~30° azimuth equivalent
  Right upper arm: ~80° (pointing more toward camera)
  Right forearm: ~90° (pointing directly at camera)
  → Each segment gets correct depth correction
```

The network learns that "short-looking upper arm in 2D + these joint features → Z component is large negative (toward camera)".

## 14 Limbs Defined

Using COCO-17 joint indices:

| Limb | Parent → Child | Description |
|------|----------------|-------------|
| 0 | 5 → 7 | L shoulder → elbow |
| 1 | 7 → 9 | L elbow → wrist |
| 2 | 6 → 8 | R shoulder → elbow |
| 3 | 8 → 10 | R elbow → wrist |
| 4 | 11 → 13 | L hip → knee |
| 5 | 13 → 15 | L knee → ankle |
| 6 | 12 → 14 | R hip → knee |
| 7 | 14 → 16 | R knee → ankle |
| 8 | 5 ↔ 6 | Shoulder width |
| 9 | 11 ↔ 12 | Hip width |
| 10 | 5 → 11 | L torso |
| 11 | 6 → 12 | R torso |
| 12 | 5 → 12 | L cross-body (L shoulder → R hip) |
| 13 | 6 → 11 | R cross-body (R shoulder → L hip) |

The cross-body limbs (12, 13) help capture torso rotation and twisting motions.

## Reference

Based on Part Orientation Fields from MonocularTotalCapture (CVPR 2019).
