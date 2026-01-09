# Constraint Fighting Bug - Root Cause Analysis

**Date**: 2026-01-09
**Status**: CRITICAL BUG - Blocking multi-constraint optimization

---

## The Problem

Bone length constraints and joint angle constraints are **fighting each other** when they should be mathematically independent:

- **Expected**: Adjusting bone length (distance between joints) should NOT change joint angles (angle between bones)
- **Actual**: Bone length corrections ARE changing joint angles, and vice versa

### Evidence

Sequential optimization results:
- Phase 1 (angles first): CV 0.129 → 0.188 (WORSE!)
- Phase 2 (lengths second): CV 0.188 → 0.115, but angles degrade

OR

- Phase 1 (lengths first): CV 0.129 → 0.070 (GOOD!)
- Phase 2 (angles second): CV 0.070 → 0.095 (DEGRADES!)

**No matter what order we apply them, they undo each other's work.**

---

## Why This Should NOT Happen (Mathematical Proof)

### Joint Angle Definition

A joint angle is determined by **3 points** (e.g., hip-knee-ankle):

```
hip_vec = knee - hip
ankle_vec = ankle - knee

angle = arccos(dot(hip_vec, ankle_vec) / (||hip_vec|| * ||ankle_vec||))
```

**Key insight**: The angle depends on the **DIRECTION** of vectors, not their **LENGTH**.

### Bone Length Adjustment (Should Preserve Angles)

When we adjust bone length, we do:

```python
# Current length
current_length = ||child - parent||

# Compute direction
direction = (child - parent) / current_length

# New position (same direction, different length)
child_new = parent + direction * target_length
```

**This preserves direction → should preserve all joint angles!**

### Conclusion

If bone length adjustments are changing joint angles, our joint angle code must be:
1. Computing angles incorrectly
2. Applying corrections that change BOTH direction AND length
3. Not respecting kinematic chains properly

---

## User's Excellent Insight

> "Let's say the knee has too much angle on the z-axis so will be in violation. When the knee gets moved closer below the hips to correct the violation, the bone length will shorten and could be in violation because of that. But when the bone length then gets adjusted, it shouldn't change the angle or rotation of the joint at all."

**This is exactly right!** Our joint angle constraint must be doing something wrong.

---

## ROOT CAUSE IDENTIFIED ✓

Location: `src/anatomical/joint_angle_depth_correction.py:202-234`

### The Bug (Lines 202-234)

```python
# Current code (WRONG):
delta_z = -learning_rate * angle_error / gradient
if adjust_child:
    child[2] += delta_z  # ← BUG: Only changing Z!
```

**Problem**: We adjust **only the Z-coordinate** while keeping X,Y fixed. This:
1. Changes the **DIRECTION** of vector (child - joint)
2. Changes the **LENGTH** of vector (child - joint)
3. Changes **ALL ANGLES** involving this vector!

### Mathematical Proof of Bug

**Example:**
```
Original:
  joint = (0, 0, 0)
  child = (1.0, 0.5, 0.3)
  length = √(1² + 0.5² + 0.3²) = 1.12m
  direction = [0.89, 0.45, 0.27]

After Z adjustment (+0.1m):
  child = (1.0, 0.5, 0.4)  # Only Z changed!
  length = √(1² + 0.5² + 0.4²) = 1.14m  ← CHANGED!
  direction = [0.88, 0.44, 0.35]  ← CHANGED!
```

**Consequence**: Bone length increased from 1.12m to 1.14m (+1.8% error). When bone length constraint tries to restore the original 1.12m, it moves the child marker back, which changes the joint angle again!

**This is why constraints fight each other.**

---

## How Others Do It (Research Results)

### Key Finding: Bone Lengths Are INHERENT to the Rigid Body Structure

From research on inverse kinematics and biomechanical modeling:

> "The segment lengths in IK are inherently preserved - they are properties of the rigid body structure itself. Joint limits are constraints on the rotational or translational parameters at the joints, not on the segment lengths."
>
> Source: [Chapter 6. Inverse Kinematics - Illinois CS](https://motion.cs.illinois.edu/RoboticSystems/InverseKinematics.html)

**Translation**: Proper IK solvers work in **JOINT SPACE** (rotation angles), not **CARTESIAN SPACE** (X,Y,Z positions). Bone lengths are constants, not variables!

### MANIKIN (2024): State-of-the-Art Approach

> "Recent 2024 paper introduces joint angles based on local coordinate systems consistent with ISB standards, featuring a bijective mapping so that one can recover the skeleton data from its angle representation (assuming knowledge of the length of individual bones)."
>
> Source: [MANIKIN: Biomechanically Accurate Neural IK](https://link.springer.com/chapter/10.1007/978-3-031-72627-9_8)

**Key insight**: "Bijective mapping" means bone lengths are preserved BY DESIGN. The representation guarantees you can't change bone length without explicitly wanting to.

### OpenSim: Professional Implementation

> "OpenSim joints can mix different multibody system computational implementations to satisfy musculoskeletal modeling needs, with joints being responsible for creating both degrees-of-freedom through internal coordinate mobilizers and constraint equations."
>
> "Most joints in OpenSim models are custom joints, which provide the most generic joint representation for modeling both conventional joints and complex biomechanical joints."
>
> Source: [OpenSim: Musculoskeletal Modeling Framework - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4397580/)

**Key insight**: OpenSim uses "mobilizers" (internal coordinates) that represent joints as ROTATIONS, not Cartesian translations. Bone lengths are implicit in the model structure.

### Conclusion

**We're doing it backwards!** Modern biomechanical IK solvers:
1. Work in JOINT SPACE (rotation angles) as primary representation
2. Bone lengths are CONSTANTS in the model
3. Cartesian positions are DERIVED from joint angles + bone lengths

We're trying to:
1. Work in CARTESIAN SPACE (X,Y,Z positions) as primary representation
2. Adjust X,Y,Z to fix angles (which breaks bone lengths!)
3. Then fix bone lengths (which breaks angles!)

This is fundamentally incompatible.

---

## Proposed Fix

### Option 1: Correct Cartesian Adjustment (Preserve Bone Length by Design)

Instead of adjusting only Z, adjust in the **RADIAL DIRECTION** to preserve bone length:

```python
# CORRECT approach:
def adjust_for_angle_preserve_length(parent, joint, child, target_angle):
    # 1. Compute current bone length (THIS IS CONSTANT!)
    bone_length = ||child - joint||

    # 2. Find the 3D direction that achieves target_angle
    # This requires solving: angle = arccos(dot(v1, v2))
    # We want to rotate child around an axis perpendicular to the plane

    # 3. Place child at: joint + new_direction * bone_length
    # Bone length preserved by construction!
```

**Advantages**:
- Preserves bone length by construction (not by checking/rejecting)
- Works in 3D Cartesian space (compatible with our data structure)
- Mathematically guaranteed not to fight bone length constraints

**Disadvantages**:
- More complex math (need to compute rotation axis and angle)
- May not always have a solution (sometimes angle can't be achieved while keeping XY)

### Option 2: Proper IK (Work in Joint Space)

Convert to joint-space representation:

```python
# 1. Convert Cartesian coords to joint angles + bone lengths
joint_angles, bone_lengths = cartesian_to_joint_space(coords)

# 2. Apply joint angle constraints (bone lengths are constants!)
joint_angles_constrained = enforce_joint_limits(joint_angles)

# 3. Convert back to Cartesian
coords_refined = joint_space_to_cartesian(joint_angles_constrained, bone_lengths)
# Bone lengths guaranteed unchanged!
```

**Advantages**:
- Mathematically correct (how all modern IK solvers work)
- Bone lengths preserved by design (they're not even variables!)
- Aligns with ISB standards and biomechanical literature

**Disadvantages**:
- Requires implementing forward/inverse kinematics
- More complex engineering (but would pay off long-term)

### Recommendation

**Short-term (immediate fix)**: Implement Option 1 - adjust markers in radial direction to preserve bone length by construction.

**Long-term (proper solution)**: Migrate to Option 2 - proper joint-space IK. This would also unlock:
- Better integration with biomechanical analysis tools (OpenSim, etc.)
- More robust constraint enforcement
- Clearer separation of concerns (angles vs positions)

## Action Items

1. ✅ **Web search**: Research how biomechanical IK solvers apply joint constraints
2. ✅ **Code review**: Examine `joint_angle_depth_correction.py` for directional bugs
3. ✅ **Fix**: Implement rotation-based adjustment using Rodrigues' formula
4. ✅ **Test**: Verify bone length preserved and constraints don't fight
5. 📋 **Long-term**: Consider migrating to proper joint-space IK

---

## THE FIX (IMPLEMENTED 2026-01-09)

### What We Changed

**File**: `src/anatomical/joint_angle_depth_correction.py:155-278`

**Before** (BROKEN):
```python
# Adjust only Z-coordinate
delta_z = -learning_rate * angle_error / gradient
child[2] += delta_z  # Changes BOTH direction AND length!
```

**After** (FIXED):
```python
# Rotate marker around joint using Rodrigues' formula
rotation_axis = cross(v_parent, v_child)  # Perpendicular to plane
rotation_angle = +learning_rate * angle_error  # Corrected sign!

# Rodrigues' rotation formula (preserves distance)
v_rotated = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
child = joint + v_rotated  # Bone length preserved by construction!
```

### Results

**Before fix**:
- Joint angles: Improved then degraded (iterative fighting)
- Bone lengths: 0.129 → 0.159 CV (23% WORSE!)
- Constraints completely incompatible

**After fix**:
- Joint angles: **+20.9% improvement** (1602 → 1267 violations)
- Bone lengths: **+19.1% improvement** (0.129 → 0.105 CV)
- Minimal constraint interaction

### Mathematical Proof It Works

```python
# Test case
Initial bone length: 0.5099m
After rotation: 0.5099m
Difference: 0.000000m (perfect preservation!)

Initial angle: 11.31°
Target: 10.00°
After rotation: 10.65° (moved toward target!)
```

**Bone length preserved to numerical precision** while **angle adjusts correctly**.

---

## Remaining Issues

**Minor constraint interaction still occurs**:
- Rotating ankle to fix knee angle → preserves knee→ankle length ✓
- But changes ankle→heel angle → affects downstream joints ✗

**Why acceptable**:
- OLD BUG: Bone length degraded 23% (completely broken)
- NEW FIX: Both constraints improve ~20% (working well!)

**Future improvement**: Solve all joints simultaneously (global optimization) instead of sequentially. This would require proper joint-space IK (Option 2 from earlier analysis).

---

## Expected Behavior After Fix

```
Initial state:
  Joint angles: 1602 violations
  Bone lengths: 0.1291 CV

Phase 1 - Joint angle constraints:
  Joint angles: 1602 → ~800 (50% improvement)
  Bone lengths: 0.1291 → 0.1291 (NO CHANGE!)  ← This should happen!

Phase 2 - Bone length constraints:
  Joint angles: ~800 → ~800 (NO CHANGE!)      ← This should happen!
  Bone lengths: 0.1291 → 0.088 (31% improvement)

Final result:
  Joint angles: 44.8% improvement
  Bone lengths: 31.5% improvement
  NO FIGHTING!
```

---

## Notes

- Base augmented skeleton is stable (CV 0.127)
- Bone length constraints alone work fine (0.127 → 0.070)
- Joint angle constraints alone partially work (1602 → 1520 violations)
- **The bug appears when combining both constraints**

This suggests the joint angle code is changing bone lengths as a side effect.
