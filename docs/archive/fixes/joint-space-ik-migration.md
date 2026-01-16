# Joint-Space IK Migration - Impact Analysis

**Date**: 2026-01-09
**Question**: Will migrating to proper joint-space IK break existing implementations?

---

## Current Pipeline Architecture

```
MediaPipe (Cartesian XYZ)
    ↓
Anatomical Constraints (Cartesian)
    ↓
Bone Length Constraints (Cartesian)
    ↓
TRC Export (Cartesian)
    ↓
Pose2Sim Augmentation (Cartesian)
    ↓
Joint Angle Constraints (Cartesian) ← BROKEN!
    ↓
Visualization (Cartesian)
```

**Key point**: Everything currently works in **Cartesian space** (X, Y, Z coordinates).

---

## Option 2 Architecture (Joint-Space IK)

```
MediaPipe (Cartesian XYZ)
    ↓
Convert: Cartesian → Joint Space
    ↓ (angles + bone lengths)
Anatomical Constraints (Joint Space)
    ↓
Joint Angle Constraints (Joint Space) ← PROPER IK!
    ↓
Convert: Joint Space → Cartesian
    ↓
TRC Export (Cartesian)
    ↓
Pose2Sim Augmentation (Cartesian)
    ↓
Visualization (Cartesian)
```

---

## What Would Break?

### ✅ **SAFE** - External Interfaces

These would **NOT** break because we convert at boundaries:

1. **MediaPipe Input**: Still receives Cartesian coords → convert to joint space internally
2. **TRC Export**: Convert back to Cartesian before writing TRC
3. **Visualization**: Already uses Cartesian coords from TRC files
4. **Pose2Sim Integration**: Works with TRC (Cartesian) - no change needed
5. **Joint Angle Computation** (kinematics module): Already converts Cartesian→angles - would become redundant

### ⚠️ **AFFECTED** - Internal Constraints

These would need **refactoring**:

1. **Anatomical Constraints** (`anatomical_constraints.py`)
   - Currently: Adjust Cartesian positions directly
   - After: Would work in joint space OR become redundant
   - **Impact**: Medium - some constraints map naturally to joint space (angles), others don't (ground plane)

2. **Bone Length Constraints** (`bone_length_constraints.py`)
   - Currently: Enforce consistent bone lengths across frames
   - After: **COMPLETELY REDUNDANT** - bone lengths are constants in joint space!
   - **Impact**: High - entire module becomes unnecessary (this is GOOD!)

3. **Ground Plane Constraints** (`ground_plane_refinement.py`)
   - Currently: Adjust Y-coordinates (Cartesian)
   - After: Would need to work on forward kinematics output (Cartesian)
   - **Impact**: Medium - still doable, but less direct

### ❌ **COMPLEX** - New Requirements

These would need **new implementation**:

1. **Forward Kinematics** (NEW)
   ```python
   def joint_space_to_cartesian(joint_angles, bone_lengths, root_position):
       """Convert joint angles → marker positions"""
       # For each joint in kinematic chain:
       #   position = parent_position + rotation_matrix @ bone_vector
   ```
   - **Complexity**: Medium - standard FK algorithm
   - **Challenge**: Need to handle multiple kinematic chains (legs, arms, spine)

2. **Inverse Kinematics** (NEW)
   ```python
   def cartesian_to_joint_space(marker_positions):
       """Convert marker positions → joint angles + bone lengths"""
       # For each joint:
       #   angles = compute_euler_angles(parent, joint, child)
       #   bone_length = ||child - parent||
   ```
   - **Complexity**: Medium - we already do this in kinematics module!
   - **Challenge**: Handling missing/occluded markers

3. **Root Position Handling** (NEW)
   - Joint space gives **relative** positions
   - Need to track **absolute** root position (pelvis/hip center)
   - **Complexity**: Low - just store root separately

### 🚧 **CRITICAL REALIZATION** - Augmented Markers ARE REQUIRED

**USER CAUGHT THE BUG IN MY ANALYSIS!**

**The Problem**: I said "augmented markers don't have joint angles" - THIS IS WRONG!

**The Truth**: We **CANNOT** compute proper 3-DOF joint angles (flexion, abduction, rotation) WITHOUT augmented markers!

**Evidence** from `src/kinematics/segment_coordinate_systems.py`:

```python
# To compute HIP joint angles, we need:
def femur_axes(
    hjc,           # Hip joint center = RHJC_study ← AUGMENTED!
    lateral_knee,  # r_knee_study (MediaPipe)
    medial_knee,   # r_mknee_study ← AUGMENTED!
    ...
)

# To compute KNEE joint angles, we need:
def tibia_axes(
    lateral_knee,  # r_knee_study (MediaPipe)
    medial_knee,   # r_mknee_study ← AUGMENTED!
    lateral_ankle, # r_ankle_study (MediaPipe)
    medial_ankle,  # r_mankle_study ← AUGMENTED!
    ...
)
```

**Without augmented markers, we can only compute simple flexion angles** from 3 points (parent-joint-child).

**With augmented markers, we can compute full 3-DOF angles** (flexion + abduction + rotation) using proper anatomical coordinate systems.

**This means**:
- Augmented markers (medial knee, medial ankle, hip joint centers) are **ESSENTIAL** for biomechanical analysis
- They're not "decorations" - they're required to define anatomical reference frames
- Joint-space IK **REQUIRES** these markers to exist first!

**The Chicken-and-Egg Problem**:
1. To compute joint angles → need segment coordinate systems
2. To build segment coordinate systems → need medial markers (augmented)
3. To generate augmented markers → need Pose2Sim augmentation
4. Pose2Sim works in Cartesian space!

**Conclusion**: We MUST stay in Cartesian space for the augmentation pipeline, THEN convert to joint space for analysis/constraints.

---

## Migration Path (If We Do It)

### Phase 1: Implement Conversion Functions (No Breaking Changes)
```python
# Add new module: src/kinematics/forward_inverse_kinematics.py
def cartesian_to_joint_angles(coords, marker_index) -> JointAngles
def joint_angles_to_cartesian(angles, bone_lengths, root_pos) -> Coords
```
- **Risk**: Low - just new functions, don't touch existing code
- **Benefit**: Can test conversions independently

### Phase 2: Refactor Joint Angle Constraints (Isolated Change)
```python
# In multi_constraint_optimization.py:
angles = cartesian_to_joint_angles(coords)
angles_constrained = enforce_joint_limits_proper(angles)  # NEW
coords = joint_angles_to_cartesian(angles_constrained, bone_lengths, root)
```
- **Risk**: Medium - replaces buggy constraint, but isolated to one function
- **Benefit**: Fixes constraint fighting bug without touching rest of pipeline

### Phase 3: Decide on Augmented Marker Strategy
- **Option A**: Hybrid (main chain in joint space, augmented as offsets)
- **Option B**: Keep everything Cartesian except joint angle constraint step
- **Recommendation**: **Option B** for now - least disruptive

### Phase 4: Gradual Optimization (Future)
- Migrate more constraints to joint space as makes sense
- Keep external interfaces Cartesian for compatibility

---

## Recommendation

### Short-term (THIS WEEK): ❌ **DON'T do full joint-space migration**

**Why**: Too risky, too complex, affects too many systems

**Instead**: Use **Option 1** from CONSTRAINT_FIGHTING_BUG.md:
- Fix `adjust_depth_for_angle_violation()` to adjust in **RADIAL DIRECTION**
- Preserves bone length by construction
- Minimal code changes (~10 lines)
- No impact on rest of pipeline
- Fixes the constraint fighting bug

### Long-term (FUTURE): ✅ **Consider hybrid approach**

**When**: After current multi-constraint optimization is working
**What**:
- Keep external interfaces Cartesian (MediaPipe, TRC, Pose2Sim)
- Add conversion functions for forward/inverse kinematics
- Use joint-space ONLY for joint angle constraints (Phase 2 above)
- Keep augmented markers as Cartesian offsets from parents

**Why**:
- Minimal disruption
- Fixes constraint fighting properly
- Unlocks future improvements
- Compatible with existing tools

---

## Answer to Original Question

**Q: Will Option 2 (joint-space IK) break existing implementations?**

**A: Yes and No.**

**External interfaces**: ✅ **NO** - MediaPipe, TRC, Pose2Sim, visualization all stay Cartesian

**Internal constraints**: ⚠️ **PARTIALLY** - Would need refactoring, but doable

**Augmented markers**: ❌ **UNSOLVED** - No clear joint-space representation for 43 augmented markers

**Overall verdict**:
- **Full migration to joint-space**: Too risky, too complex, unsolved problems with augmented markers
- **Hybrid approach** (joint-space only for constraints, Cartesian everywhere else): Feasible, but still significant work
- **Recommended**: Fix current Cartesian approach first (Option 1), consider hybrid later

---

## Implementation Priority

1. **IMMEDIATE** (this week): Fix radial direction adjustment (Option 1)
2. **SHORT-TERM** (this month): Test and validate fixed constraints
3. **MEDIUM-TERM** (next quarter): Add forward/inverse kinematics functions for future use
4. **LONG-TERM** (future): Consider hybrid joint-space approach if needed

**Don't let perfect be the enemy of good. Fix the bug first, optimize later.**
