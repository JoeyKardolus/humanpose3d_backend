# Changelog

Development history for HumanPose3D MediaPipe pipeline.

## 2026-01-13 - Neural Refinement Systems

### Joint Constraint Refinement
- **New**: `src/joint_refinement/` module (916K parameters)
- Learns soft joint constraints from AIST++ motion data
- Cross-joint attention with kinematic chain bias
- Integrated via `--joint-constraint-refinement` flag

### Depth Refinement First Results
- **First training run**: 53.6% depth error reduction (11.6cm -> 5.4cm)
- Camera prediction from 2D pose: ±11° accuracy (no calibration needed)
- Training on AIST++ dataset (10.1M frames, 9 cameras)

### View Angle Computation
- Simplified to camera-ray-to-torso-plane approach
- More robust than body-facing direction calculation

## 2026-01-12 - AIST++ Dataset Integration

### Dataset Switch
- Switched from HumanEva/CMU to AIST++ (10.1M annotated frames)
- Real MediaPipe errors from actual video frames
- 9 camera views for viewpoint diversity

### Training Data Pipeline
- `scripts/convert_aistpp_to_training.py` - Data converter
- `scripts/run_parallel_conversion.sh` - Multi-camera parallel processing
- Target: 1.5M+ training samples

## 2026-01-10 - Pelvis Angle Validation

### Fixes
- Fixed pelvis coordinate system (Y-up primary axis)
- Changed to ZXY Euler sequence (clinical convention)
- Added axis continuity check (prevents 180° flips)
- Removed over-tight clamping on pelvis angles
- Updated default smooth_window from 9 to 21

### Validation
- Created reference comparison script
- Results match reference implementation within 2%
- See `PELVIS_ANGLE_FIXES.md` for details

## 2026-01-09 - Code Cleanup & Organization

### Removed Deprecated Features
- `joint_angle_depth_correction.py` (~350 lines)
- `rigid_cluster_constraints.py` (~450 lines)
- `flk_filter.py` (~500 lines)
- 11 CLI flags removed (49 -> 38 flags)

### Output Organization
- Automatic cleanup of intermediate files
- `joint_angles/` subdirectory for angle outputs
- Clear file naming: `_initial.trc`, `_final.trc`, `_raw_landmarks.csv`

### Multi-Constraint Optimization
- 3-phase approach: Filter -> Stabilize -> Finalize
- 68% bone length improvement (CV: 0.113 -> 0.036)
- Automatic filtering of unreliable augmented markers

## 2025-12 - Biomechanical Constraints

### Features Added
- ISB-compliant joint angle computation (12 joint groups)
- Multi-constraint optimization pipeline
- Ground plane refinement with stance detection
- Bone length consistency enforcement

### Architecture
- `src/kinematics/` - Joint angle computation
- `src/anatomical/` - Biomechanical constraints
- Comprehensive visualization tools

## 2025-11-29 - Initial Pipeline

### Foundation
- MediaPipe pose extraction (33 landmarks -> 22 markers)
- Pose2Sim LSTM augmentation (22 -> 65 markers)
- TRC format output for biomechanics tools
- GPU acceleration for LSTM inference

### Strict Mode
- No placeholder values
- Explicit visibility thresholds
- Clean marker estimation pipeline

---

*For verbose run logs, see `archive/BUILD_LOG_runs.md`*
