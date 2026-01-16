# Cleanup Analysis - Redundant and Deprecated Features

**Date**: 2026-01-09
**Goal**: Identify removable code when using recommended workflow with multi-constraint optimization + comprehensive joint angles

## Recommended Workflow

```bash
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 --mass 75 --age 30 --sex male \
  --anatomical-constraints \
  --bone-length-constraints \
  --estimate-missing \
  --force-complete \
  --augmentation-cycles 20 \
  --multi-constraint-optimization \
  --compute-all-joint-angles \
  --plot-all-joint-angles \
  --save-angle-comparison
```

**Flags Used**: 12 total
**Flags Available**: 49 total
**Redundancy**: 37 flags (75%) potentially removable

---

## Category 1: DEPRECATED - Can Be Removed

### 1.1 Old Joint Angle Correction (Superseded by Multi-Constraint)

**REMOVE THESE**:
- `--joint-angle-depth-correction` - Old single-pass depth correction
- `--joint-angle-correction-iterations` - Iterations for old method
- `--joint-angle-correction-lr` - Learning rate for old method

**Reason**: Multi-constraint optimization (`--multi-constraint-optimization`) handles this better with:
- Iterative convergence
- Multiple constraint types
- No learning rate tuning needed
- Better results (62% bone length improvement vs 30-40%)

**Files to remove**:
- `src/anatomical/joint_angle_depth_correction.py` (deprecated)
- Related code in `main.py` lines 550-615

---

### 1.2 Individual Joint Angle Computation (Superseded by Comprehensive)

**DEPRECATE** (keep for backward compatibility, but don't recommend):
- `--compute-joint-angles` - Single side lower limb only
- `--joint-angle-side` - Side selection for single computation
- `--compute-upper-body-angles` - Single side upper body only
- `--upper-body-side` - Side selection for upper body

**Reason**: `--compute-all-joint-angles` does all of this in one call:
- Computes BOTH sides simultaneously
- Includes pelvis and trunk
- Better consistency (shared smoothing, zeroing)
- Single comprehensive visualization

**Keep for now**: Legacy users may rely on these, but mark as deprecated in docs

---

### 1.3 Rigid Cluster Constraints (Experimental, Never Recommended)

**REMOVE THESE**:
- `--rigid-clusters` - Experimental cluster constraints
- Related imports and code

**Reason**:
- Never mentioned in recommended workflow
- Experimental feature that wasn't finalized
- Multi-constraint optimization handles marker stability better
- Check if `src/anatomical/rigid_cluster_constraints.py` is even used

**Files to check**:
- `src/anatomical/rigid_cluster_constraints.py`
- `docs/RIGID_CLUSTER_FAILURE.md` - Document why it failed

---

### 1.4 Individual Constraint Flags (Superseded by Multi-Constraint)

**CONSIDER REMOVING** (or make them no-ops that warn):
- `--anatomical-constraints` - Old single-pass constraints
- `--bone-length-constraints` - Old single-pass bone length
- `--ground-plane-refinement` - Old ground plane method

**Reason**: `--multi-constraint-optimization` does ALL of these:
- Bone length constraints (phase 2)
- Ground plane constraints (phase 3)
- Anatomical filtering (phase 0)
- Better: Prevents constraint fighting

**BUT**: Users might want granular control for debugging.

**Recommendation**: Keep but document as "advanced/debugging only"

---

## Category 2: RARELY USED - Consider Removing

### 2.1 FLK Filter (Unused in Recommended Workflow)

**REMOVE THESE**:
- `--flk-filter` - Apply FLK temporal filtering
- `--flk-model` - FLK model variant
- `--flk-enable-rnn` - Enable RNN for FLK
- `--flk-passes` - Number of FLK passes
- `--gaussian-smooth` - Gaussian smoothing (alternative)

**Reason**:
- Not mentioned in any recommended workflow
- Multi-constraint optimization handles smoothness
- Adds complexity without clear benefit
- FLK documentation is minimal

**Files to check**:
- `src/filtering/flk_filter.py` - Is this even used?
- Related tests

---

### 2.2 Low-Level Constraint Tuning (Expert Only)

**CONSIDER HIDING** (keep code, remove from main help):
- `--bone-length-tolerance` - Fine-tune tolerance (default 0.15 works)
- `--bone-length-iterations` - Iterations (multi-constraint handles this)
- `--bone-depth-weight` - Depth vs XY weight (default 0.8 works)
- `--ground-contact-threshold` - Ground contact threshold (default 0.03 works)
- `--min-contact-frames` - Min frames for stance (default 3 works)
- `--depth-propagation-weight` - Propagation weight (default 0.7 works)
- `--bone-smooth-window` - Bone smoothing window
- `--ground-percentile` - Ground plane percentile
- `--ground-margin` - Ground plane margin

**Reason**:
- Expert parameters that users shouldn't touch
- Defaults are well-tuned
- Clutters help output (49 flags is overwhelming!)

**Recommendation**: Move to environment variables or config file

---

### 2.3 Header Fix (Automated Now)

**REMOVE THIS**:
- `--fix-header` - Manual header correction

**Reason**:
- TRC loading is now robust (reads actual data columns, not header)
- `VisualizeData.load_trc_frames()` handles mismatched headers
- Post-augmentation estimation adds markers to data, not header (expected behavior)
- No user reports of header issues

**Files to check**:
- `src/datastream/data_stream.py::header_fix_strict()` - Still needed?

---

### 2.4 Visibility Threshold (Advanced)

**CONSIDER REMOVING**:
- `--visibility-min` - MediaPipe visibility threshold

**Reason**:
- Default 0.3 works well across all test videos
- Users don't understand what this does
- `--estimate-missing` handles low visibility better

**Recommendation**: Keep as environment variable for experts

---

## Category 3: KEEP - Essential Features

### 3.1 Core Input (REQUIRED)
✅ `--video` - Input video file
✅ `--height` - Subject height
✅ `--mass` - Subject mass
✅ `--age` - Subject age
✅ `--sex` - Subject sex

### 3.2 Pipeline Control (RECOMMENDED)
✅ `--estimate-missing` - Pre-augmentation estimation
✅ `--force-complete` - Post-augmentation estimation
✅ `--augmentation-cycles` - Multi-cycle averaging (default 20)
✅ `--multi-constraint-optimization` - **THE** optimization method
✅ `--multi-constraint-iterations` - Max iterations (default 10)

### 3.3 Joint Angles (RECOMMENDED)
✅ `--compute-all-joint-angles` - Comprehensive ISB-compliant angles
✅ `--plot-all-joint-angles` - Multi-panel visualization
✅ `--save-angle-comparison` - R vs L comparison
✅ `--joint-angle-smooth-window` - Smoothing window (default 9)
✅ `--check-joint-constraints` - Validate biomechanics

### 3.4 Visualization (USEFUL)
✅ `--show-video` - MediaPipe preview during processing
✅ `--plot-landmarks` - Debug: View raw MediaPipe output
✅ `--plot-augmented` - View final augmented skeleton

---

## Redundant Visualizations

### Old Individual Plots (Superseded)
- `--plot-joint-angles` - Single side 3-panel (hip/knee/ankle only)
- `--plot-upper-body-angles` - Single side 3-panel (trunk/shoulder/elbow only)

**Reason**: `--plot-all-joint-angles` generates comprehensive 7x2 grid with ALL joints

**Recommendation**: Deprecate, keep for backward compat

---

## Proposed Cleanup Actions

### Phase 1: IMMEDIATE REMOVAL (High Confidence)

1. **Remove deprecated joint angle correction**:
   - Delete `src/anatomical/joint_angle_depth_correction.py`
   - Remove `--joint-angle-depth-correction` flag
   - Remove related code in `main.py`

2. **Remove experimental rigid clusters**:
   - Delete `src/anatomical/rigid_cluster_constraints.py` (if unused)
   - Remove `--rigid-clusters` flag

3. **Remove FLK filter** (if never used):
   - Check usage: `grep -r "flk_filter" --exclude-dir=.git`
   - If unused, delete `src/filtering/flk_filter.py`
   - Remove all FLK-related flags

4. **Remove header fix**:
   - Remove `--fix-header` flag
   - Keep `header_fix_strict()` function if used internally

### Phase 2: DEPRECATION (Mark as Deprecated)

1. **Individual joint angle computation**:
   - Add deprecation warnings to `--compute-joint-angles`
   - Add deprecation warnings to `--compute-upper-body-angles`
   - Update docs to recommend `--compute-all-joint-angles`

2. **Individual constraint flags**:
   - Add warnings: "Use --multi-constraint-optimization instead"
   - Keep functional for backward compatibility

### Phase 3: SIMPLIFICATION (Reduce CLI Clutter)

1. **Move expert parameters to config file**:
   - Create `config.yaml` or environment variables
   - Remove from CLI help (keep in code)
   - Parameters: bone-length-tolerance, ground-contact-threshold, etc.

2. **Consolidate visualization flags**:
   - Keep: `--plot-all-joint-angles` (primary)
   - Keep: `--plot-augmented` (skeleton)
   - Keep: `--show-video` (preview)
   - Deprecate: `--plot-joint-angles`, `--plot-upper-body-angles`
   - Remove: `--no-plot` (never used)

---

## Expected Impact

### Before Cleanup
- **49 CLI flags** (overwhelming for new users)
- **~15,000 lines** of code
- **Multiple deprecated code paths**
- **Confusing documentation** (too many options)

### After Cleanup
- **~20-25 CLI flags** (focused, clear)
- **~12,000 lines** of code (20% reduction)
- **Single recommended path**: Multi-constraint + comprehensive angles
- **Clear documentation**: Simple examples that work

### Removed Files (Estimated)
1. `src/anatomical/joint_angle_depth_correction.py` (~300 lines)
2. `src/anatomical/rigid_cluster_constraints.py` (~400 lines)
3. `src/filtering/flk_filter.py` (~500 lines?) - if unused
4. Old visualization code (~200 lines)
5. Deprecated tests (~500 lines)

**Total removal**: ~1,900 lines + deprecated flags

---

## Testing Required Before Removal

1. **Check FLK filter usage**:
   ```bash
   grep -r "flk_filter" src/ tests/ --exclude-dir=.git
   grep -r "apply_flk" src/ tests/
   ```

2. **Check rigid cluster usage**:
   ```bash
   grep -r "rigid_cluster" src/ tests/ --exclude-dir=.git
   ```

3. **Check joint angle depth correction usage**:
   ```bash
   grep -r "joint_angle_depth" src/ tests/ --exclude-dir=.git
   grep -r "refine_depth_with" src/
   ```

4. **Check header fix usage**:
   ```bash
   grep -r "header_fix" src/ tests/
   ```

5. **Run all tests**:
   ```bash
   uv run pytest -v
   ```

---

## Recommendation Priority

### 🔴 HIGH PRIORITY - Remove Now
1. Joint angle depth correction (superseded)
2. Rigid cluster constraints (experimental failure)
3. FLK filter (if unused)

### 🟡 MEDIUM PRIORITY - Deprecate
1. Individual joint angle flags (mark as legacy)
2. Individual constraint flags (warn to use multi-constraint)
3. Old visualization flags (superseded by comprehensive)

### 🟢 LOW PRIORITY - Simplify Later
1. Move expert parameters to config
2. Consolidate similar flags
3. Improve help text organization

---

## Questions for User

1. **FLK Filter**: Is this used anywhere? Can we remove it?
2. **Rigid Clusters**: This failed - can we delete it?
3. **Backward Compatibility**: Do you need to support old CLI flags for existing users?
4. **Config File**: Would you prefer YAML config for expert parameters?

