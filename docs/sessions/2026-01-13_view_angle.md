# Session: View Angle Computation Refinement
**Date**: 2026-01-13

## Summary
Refined the view angle computation for AIST++ training data to use a simpler, more robust approach.

## Key Changes

### View Angle Computation (Simplified)
Changed from complex body-facing direction calculation to simple **camera ray hitting torso plane**:

```python
# Torso plane normal
v1 = right_shoulder - left_shoulder
v2 = left_hip - left_shoulder
normal = np.cross(v1, v2)

# Camera ray
cam_ray = torso_center - camera_pos

# Angle from plane (not from normal!)
dot = np.dot(cam_ray, normal)
angle_from_normal = arccos(|dot|)
angle_from_plane = 90° - angle_from_normal

# Front/back detection via nose
nose_side = dot(nose - torso_center, normal)
cam_side = -dot
viewing_front = (nose_side * cam_side) > 0
if not viewing_front:
    angle = -angle  # Negative = back view
```

### Angle Semantics
- **0°** = profile view (camera ray parallel to torso plane)
- **90°** = frontal view (camera ray perpendicular to torso plane)
- **Negative** = viewing from back (nose on opposite side from camera)

### Files Modified
- `scripts/convert_aistpp_to_training.py` - Updated `compute_view_angles()` function
- `scripts/visualize_view_angle_simple.py` - New clean visualization script

### Files Removed (cleanup)
- `scripts/visualize_view_angle_on_frame.py` - Old cluttered version
- `scripts/visualize_computed_view_angles.py` - Old version
- `scripts/visualize_inference_vs_training.py` - Old version
- `scripts/visualize_view_angle.py` - Old version

## Training Data Status
- **15,856 samples** generated in `data/training/aistpp_converted/`
- Samples from camera c01 (frontal camera)
- Azimuth range in current data: ~50-80° (mostly frontal views)
- Need to regenerate with updated angle computation

## Visualization Output
Clean 3-panel visualization showing:
1. Top-down view with torso plane, camera, and ray
2. Angle diagram showing ray hitting plane
3. Result with big angle number and FRONT/BACK indicator

Example files: `view_angle_gBR_sBM_c01_d04_mBR0_ch01_f0000.png`

## Next Steps
1. Regenerate training data with new angle computation
2. Verify angle distribution across different camera views
3. Proceed with model training (Phase 2 of plan)

## Plan Reference
See `/home/dupe/.claude/plans/witty-wibbling-brook.md` for full implementation plan.
