# 3D Pose Estimation Pipeline Report

**Date:** 2026-01-11
**Input:** MicrosoftTeams-video.mp4
**Duration:** ~12 seconds (710 frames @ ~60 fps)

---

## Overview

This report documents the complete pipeline for extracting 3D pose landmarks from video and computing pelvis kinematics using MediaPipe, Pose2Sim marker augmentation, and custom angle computation scripts.

---

## Pipeline Steps

### Step 1: Video to CSV (MediaPipe 3D Pose)

**Script:** `run_augment_pipeline.py` → `video_to_csv()` function
**Tool:** MediaPipe Pose (model_complexity=2)
**Output:** `pose_world_landmarks.csv`

**Process:**
1. Load video with OpenCV (`cv2.VideoCapture`)
2. Process each frame through MediaPipe Pose with world landmark estimation
3. Extract 33 pose landmarks per frame in world coordinates (meters)
4. Write to CSV with columns: `timestamp_s`, `landmark`, `x_m`, `y_m`, `z_m`, `visibility`

**Result:** 710 frames processed, 33 landmarks per frame = 23,430 landmark rows

---

### Step 2: CSV to TRC Conversion (22 Input Markers)

**Script:** `run_augment_pipeline.py` → `csv_to_trc_exact()` function
**Output:** `pose2sim_input_exact.trc`

**Process:**
1. Read MediaPipe CSV and group by timestamp
2. Map MediaPipe landmark names to biomechanical marker names:
   ```
   LEFT_SHOULDER  → LShoulder    RIGHT_SHOULDER → RShoulder
   LEFT_ELBOW     → LElbow       RIGHT_ELBOW    → RElbow
   LEFT_WRIST     → LWrist       RIGHT_WRIST    → RWrist
   LEFT_HIP       → LHip         RIGHT_HIP      → RHip
   LEFT_KNEE      → LKnee        RIGHT_KNEE     → RKnee
   LEFT_ANKLE     → LAnkle       RIGHT_ANKLE    → RAnkle
   LEFT_HEEL      → LHeel        RIGHT_HEEL     → RHeel
   LEFT_FOOT_INDEX → LBigToe     RIGHT_FOOT_INDEX → RBigToe
   NOSE           → Nose
   ```
3. Compute derived markers:
   - `Neck` = midpoint of LShoulder and RShoulder
   - `Hip` = midpoint of LHip and RHip
   - `LSmallToe`/`RSmallToe` = copy of BigToe (if not available)
4. Write TRC file with 22 markers in exact order required by augmenter:
   ```
   Hip, RHip, RKnee, RAnkle, RBigToe, RSmallToe, RHeel,
   LHip, LKnee, LAnkle, LBigToe, LSmallToe, LHeel,
   Neck, Head, Nose, RShoulder, RElbow, RWrist,
   LShoulder, LElbow, LWrist
   ```

---

### Step 3: Marker Augmentation (Pose2Sim LSTM v0.3)

**Script:** `run_augment_pipeline.py` → `run_augmenter()` function
**Tool:** Pose2Sim MarkerAugmentation (LSTM neural network)
**Output:** `pose2sim_input_exact_LSTM.trc`

**Configuration:**
```toml
[participant]
height = 1.78  # meters
mass = 75.0    # kg
age = 30
sex = "male"

[markerAugmentation]
feet_on_floor = false
use_lower_limb = true
use_upper_limb = true
```

**Process:**
1. Copy input TRC to `pose-3d/` directory
2. Generate `Config.toml` with participant parameters and frame range
3. Run Pose2Sim `markerAugmentation.augment_markers_all()`
4. LSTM model predicts additional anatomical markers from the 22 input markers

**Result:** Augmented from 22 markers to 65 markers including:
- ASIS markers (r.ASIS_study, L.ASIS_study)
- PSIS markers (r.PSIS_study, L.PSIS_study)
- Additional joint centers and anatomical landmarks

---

### Step 4: TRC Header Fix

**Script:** `run_augment_pipeline.py` → `header_fix()` function
**Output:** `pose2sim_input_exact_LSTM_fixed.trc`

**Process:**
1. Read LSTM output TRC
2. Count actual marker triplets in data rows
3. Retrieve official response marker names from Pose2Sim
4. Update header to include all `*_study` marker names
5. Fix NumMarkers count in header line 2

**Result:** 65 markers with proper names in header

---

### Step 5: Pelvis Global Angles Computation

**Script:** `compute_pelvis_global_angles.py`
**Output:** `pose2sim_input_exact_LSTM_fixed_pelvis_global_ZXY.csv`

**Required Markers:**
- `r.ASIS_study` (Right Anterior Superior Iliac Spine)
- `L.ASIS_study` (Left Anterior Superior Iliac Spine)
- `r.PSIS_study` (Right Posterior Superior Iliac Spine)
- `L.PSIS_study` (Left Posterior Superior Iliac Spine)

**Process:**
1. Read augmented TRC file
2. Apply smoothing (moving average, window=9)
3. For each frame, construct pelvis coordinate system:
   - Compute ASIS midpoint and PSIS midpoint
   - Z-axis (Zp): Right direction (RASIS → LASIS)
   - Y-axis (Yp): Up direction (perpendicular, PSIS → ASIS plane)
   - X-axis (Xp): Forward direction (cross product)
4. Build rotation matrix from pelvis axes
5. Extract Euler angles (ZXY sequence):
   - **Flexion/Extension** (Z): Sagittal plane rotation
   - **Abduction/Adduction** (X): Frontal plane rotation
   - **Axial Rotation** (Y): Transverse plane rotation
6. Apply unwrap to remove ±180° discontinuities
7. Zero to global mean (subtract mean from each angle series)

**Settings:**
```python
SMOOTH_WINDOW = 9
UNWRAP = True
ZERO_MODE = "global_mean"
EULER_SEQ = "ZXY"
```

---

## Results Summary

### Pelvis Angle Statistics (709 frames, ~12 seconds)

| Angle | Min | Max | Range | Std Dev |
|-------|-----|-----|-------|---------|
| Flexion/Extension (Z) | -7.01° | +6.38° | 13.39° | 2.83° |
| Abduction/Adduction (X) | -11.11° | +6.90° | 18.01° | 3.61° |
| Axial Rotation (Y) | -6.51° | +5.66° | 12.17° | 3.27° |

The rhythmic oscillation pattern in all three planes indicates cyclic movement (likely walking or similar activity).

---

## Environment & Dependencies

**Python Environment:** Conda `mmpose` (Python 3.9.25)

**Key Packages:**
- mediapipe 0.10.9 (with solutions API)
- opencv-python 4.12.0.88
- Pose2Sim (local, with markerAugmentation)
- onnxruntime 1.19.2
- scipy 1.13.1
- numpy 2.0.2
- PyQt5 5.15.11

**Note:** MediaPipe 0.10.9 is required for the `mp.solutions.pose` API. Newer versions (0.10.30+) use a different tasks-based API that is incompatible with this pipeline.

---

## Output Files

| File | Description | Size |
|------|-------------|------|
| `pose_world_landmarks.csv` | Raw MediaPipe 3D landmarks | 2.3 MB |
| `pose2sim_input_exact.trc` | 22-marker TRC input | 425 KB |
| `pose2sim_input_exact_LSTM.trc` | 65-marker augmented output | 2.2 MB |
| `pose2sim_input_exact_LSTM_fixed.trc` | Header-fixed final TRC | 2.2 MB |
| `pose2sim_input_exact_LSTM_fixed_pelvis_global_ZXY.csv` | Pelvis angles | 21 KB |
| `pelvis_angles_plot.png` | Visualization | 246 KB |

---

## Directory Structure

```
use/
├── input/
│   └── MicrosoftTeams-video.mp4      # Source video
├── output/
│   ├── pose_world_landmarks.csv      # Step 1 output
│   ├── pose2sim_input_exact.trc      # Step 2 output
│   ├── pose2sim_input_exact_LSTM.trc # Step 3 output
│   ├── pose2sim_input_exact_LSTM_fixed.trc  # Step 4 output
│   ├── pose2sim_input_exact_LSTM_fixed_pelvis_global_ZXY.csv  # Step 5 output
│   └── pelvis_angles_plot.png        # Visualization
├── scripts/
│   ├── run_augment_pipeline.py       # Main pipeline script
│   ├── compute_pelvis_global_angles.py  # Pelvis angle computation
│   ├── pose3d_test.py                # Basic pose extraction test
│   └── rebuild_trc_header_strict.py  # TRC header utility
└── PIPELINE_REPORT.md                # This report
```

---

## How to Reproduce

```bash
# Activate environment
conda activate mmpose

# Run full pipeline (bypassing GUI)
cd /home/dupe/ai-test-project/pose3d_project_jaap/use

python3 << 'EOF'
import sys
sys.path.insert(0, '..')
from pathlib import Path
exec(open('scripts/run_augment_pipeline.py').read().split('def main():')[0])

video_path = Path("input/MicrosoftTeams-video.mp4")
PARTICIPANT = {"mass": 75.0, "height": 1.78, "age": 30, "sex": "male"}

csv_path = video_to_csv(video_path)
trc_exact = csv_to_trc_exact(csv_path)
lstm_trc = run_augmenter(trc_exact, PARTICIPANT)
fixed_trc = header_fix(lstm_trc)
print("Done:", fixed_trc)
EOF

# Then run pelvis angles on the output
python3 scripts/compute_pelvis_global_angles.py
```

---

## Notes

- The pipeline bypasses GUI dialogs for headless/WSL environments
- Participant height/mass affect the LSTM marker augmentation scaling
- The pelvis angle computation requires the `*_study` markers from augmentation
- Zero-centering to global mean removes static offset but preserves dynamics
