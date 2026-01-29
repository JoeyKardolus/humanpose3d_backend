# Output Directory Organization

**Date**: 2026-01-09
**Status**: Clean and organized structure

---

## Directory Structure

```
~/.humanpose3d/output/<video_name>/
├── <video>_final.trc               # Final multi-constraint optimized skeleton (65 markers)
├── <video>_initial.trc             # Initial TRC from MediaPipe (22 markers)
├── <video>_raw_landmarks.csv       # Raw MediaPipe landmarks (CSV format)
├── joint_angles/                   # Joint angle analysis
│   ├── <video>_all_joint_angles.png         # Comprehensive visualization (7x2 grid)
│   ├── <video>_angles_pelvis.csv            # Pelvis tilt/obliquity/rotation
│   ├── <video>_angles_hip_R.csv             # Right hip flexion/abduction/rotation
│   ├── <video>_angles_hip_L.csv             # Left hip flexion/abduction/rotation
│   ├── <video>_angles_knee_R.csv            # Right knee flexion
│   ├── <video>_angles_knee_L.csv            # Left knee flexion
│   ├── <video>_angles_ankle_R.csv           # Right ankle dorsiflexion/inversion/rotation
│   ├── <video>_angles_ankle_L.csv           # Left ankle dorsiflexion/inversion/rotation
│   ├── <video>_angles_trunk.csv             # Trunk flexion/lateral bend/rotation
│   ├── <video>_angles_shoulder_R.csv        # Right shoulder angles
│   ├── <video>_angles_shoulder_L.csv        # Left shoulder angles
│   ├── <video>_angles_elbow_R.csv           # Right elbow flexion
│   └── <video>_angles_elbow_L.csv           # Left elbow flexion
└── old_runs/                       # Archived files from previous runs
    └── (deprecated files from experimental features)
```

---

## File Descriptions

### Primary Outputs

**`<video>_final.trc`**
- Final optimized skeleton
- 59-64 markers (22 MediaPipe + augmented, some filtered)
- Neural refinement applied if `--main-refiner` used:
  - Depth correction (pre-augmentation)
  - Joint constraint refinement (post-augmentation)
- Ready for biomechanical analysis, inverse dynamics, or visualization

**`<video>_initial.trc`**
- Initial TRC from MediaPipe pose detection
- 22 markers: Neck, RShoulder, LShoulder, RHip, LHip, RKnee, LKnee, RAnkle, LAnkle, RHeel, LHeel, RSmallToe, LSmallToe, RBigToe, LBigToe, RElbow, LElbow, RWrist, LWrist, Hip, Head, Nose
- Pre-augmentation estimation applied if `--estimate-missing` used

**`<video>_raw_landmarks.csv`**
- Raw MediaPipe pose landmarks
- CSV format: timestamp_s, landmark, x_m, y_m, z_m, visibility
- No constraints or filtering applied
- Useful for debugging or custom processing

### Joint Angles

**`joint_angles/<video>_all_joint_angles.png`**
- Comprehensive 7x2 grid visualization
- Shows all joint angles over time with color-coded DOFs
- 7 joint groups: pelvis, hip, knee, ankle, trunk, shoulder, elbow
- 2 sides: right (R) and left (L)

**`joint_angles/<video>_angles_*.csv`**
- Individual CSV files for each joint group
- Columns: time_s, flexion_deg, abduction_deg, rotation_deg (as applicable)
- ISB-compliant coordinate systems
- Smoothed (default window=9), unwrapped, zeroed to first 0.5s
- Clamped to biomechanical ranges (prevents unrealistic spikes)

---

## Cleanup Policy

### Automatic Cleanup

The pipeline automatically removes:
- **Intermediate TRC files** from augmentation cycles (e.g., `*_cycle0.trc`, `*_complete.trc`)
- **Temporary Pose2Sim projects** (`pose2sim_project_cycle*/`)
- **Config files** (`Config_cycle*.toml`)
- **Zone.Identifier files** (Windows WSL metadata)

### Archived Files

Old runs and deprecated feature outputs are moved to `old_runs/`:
- Files from deprecated features (rigid clusters, old joint angle depth correction)
- Previous runs with different parameters
- Old single-side angle visualizations

**Note**: If `old_runs/` gets too large (>1GB), manually delete it.

---

## File Sizes (Typical)

| File | Size | Description |
|------|------|-------------|
| `*_raw_landmarks.csv` | ~500KB | Raw MediaPipe data (535 frames × 22 markers) |
| `*_initial.trc` | ~300KB | Initial TRC (22 markers) |
| `*_final.trc` | ~900KB | Final optimized (65 markers, some filtered) |
| `*_all_joint_angles.png` | ~800KB | Comprehensive visualization |
| `*_angles_*.csv` | 5-15KB | Individual joint angle time series |

---

## Integration with Pipeline

The cleanup happens **after** the main pipeline completes:

```bash
uv run python manage.py run_pipeline \
  --video ~/.humanpose3d/input/<video>.mp4 \
  --height 1.78 --weight 75 \
  --estimate-missing --force-complete \
  --augmentation-cycles 20 \
  --main-refiner \
  --plot-all-joint-angles

# Output automatically organized:
# → joey_final.trc
# → joey_initial.trc
# → joey_raw_landmarks.csv
# → joint_angles/ (13 files)
```

---

## Manual Cleanup

To clean up output directory manually:

```bash
cd ~/.humanpose3d/output/<video_name>/
./cleanup.sh  # If script exists

# Or manually:
rm -rf pose2sim_project_cycle*/ Config_cycle*.toml
rm *.Zone.Identifier
mkdir -p old_runs joint_angles
mv <old_files> old_runs/
mv *_angles_*.csv *_all_joint_angles.png joint_angles/
```

---

## Recommended Workflow

1. **Run pipeline** with recommended flags
2. **Check `<video>_final.trc`** for marker quality
3. **View `joint_angles/<video>_all_joint_angles.png`** for biomechanics
4. **Import CSVs** into analysis software (OpenSim, MATLAB, Python)
5. **Archive or delete `old_runs/`** when done

---

## Notes

- Joint angles require `--force-complete` flag (ensures hip joint centers are estimated)
- Right-side data may have fewer frames due to marker visibility (check CSV for NaN)
- Some unreliable augmented markers may be filtered (typical: 4-6 markers removed)
- Old runs are preserved for 1 session, then manually cleaned
