# 3D Pose Estimation & Pelvis Kinematics Pipeline

Extract 3D pose landmarks from video using MediaPipe, augment markers with Pose2Sim LSTM, and compute pelvis kinematics.

---

## Requirements

- **OS:** Linux (tested on WSL2)
- **Python:** 3.9.x (required for MediaPipe compatibility)
- **Conda:** Miniconda or Anaconda

---

## Installation

### 1. Create Conda Environment

```bash
conda create -n pose3d python=3.9 -y
conda activate pose3d
```

### 2. Install Dependencies

```bash
# Core packages
pip install mediapipe==0.10.9
pip install opencv-python
pip install numpy pandas scipy

# For Pose2Sim marker augmentation
pip install onnxruntime
pip install toml
pip install c3d
pip install anytree
pip install tensorflow
pip install PyQt5

# For visualization
pip install matplotlib
```

**Important:** Use `mediapipe==0.10.9` specifically. Newer versions (0.10.30+) use a different API that is incompatible with this pipeline.

### 3. Verify Installation

```bash
python -c "import mediapipe as mp; print('solutions' in dir(mp))"
# Should print: True
```

---

## Directory Structure

```
use/
├── input/          # Place your video files here
├── output/         # Generated outputs go here
├── scripts/        # Pipeline scripts
├── README.md       # This file
└── PIPELINE_REPORT.md  # Detailed technical documentation
```

---

## Usage

### Quick Start (Command Line - No GUI)

```bash
conda activate pose3d
cd /path/to/use

python3 << 'EOF'
import os, sys
sys.path.insert(0, '..')
from pathlib import Path

os.chdir('/path/to/use')
exec(open('scripts/run_augment_pipeline.py').read().split('def main():')[0])

# Configure
video_path = Path("input/YOUR_VIDEO.mp4")
PARTICIPANT = {
    "mass": 75.0,      # kg
    "height": 1.78,    # meters
    "age": 30,
    "sex": "male"      # or "female"
}

# Run pipeline
csv_path = video_to_csv(video_path)
trc_exact = csv_to_trc_exact(csv_path)
lstm_trc = run_augmenter(trc_exact, PARTICIPANT)
fixed_trc = header_fix(lstm_trc)

print("Done! Output:", fixed_trc)
EOF
```

### With GUI (Desktop Environment)

If you have a display available:

```bash
conda activate pose3d
cd /path/to/use
python scripts/run_augment_pipeline.py
```

This will open dialogs to:
1. Enter participant data (height, mass, age, sex)
2. Select video file

### Compute Pelvis Angles

After running the main pipeline:

```bash
python3 << 'EOF'
import os, sys
sys.path.insert(0, '..')
from pathlib import Path
import math, csv
import numpy as np

os.chdir('/path/to/use')
exec(open('scripts/compute_pelvis_global_angles.py').read().replace('if __name__ == "__main__":', 'if False:'))

trc = Path("output/pose2sim_input_exact_LSTM_fixed.trc")
names, idx, frames, times, coords = read_trc(trc)
coords = smooth_coords(coords, SMOOTH_WINDOW)
F = coords.shape[0]

flex = np.full(F, np.nan)
abd = np.full(F, np.nan)
rot = np.full(F, np.nan)

prev = None
for i in range(F):
    row = coords[i]
    pel = pelvis_axes(row, idx, prev=prev)
    if pel is None: continue
    prev = pel
    Rp = R_from_axes(pel["Xp"], pel["Yp"], pel["Zp"])
    ez, ex, ey = euler_ZXY(Rp)
    flex[i] = ez * SIGNS["flex"]
    abd[i] = ex * SIGNS["abd"]
    rot[i] = ey * SIGNS["rot"]

if UNWRAP:
    flex[:] = unwrap_series_deg(flex)
    abd[:] = unwrap_series_deg(abd)
    rot[:] = unwrap_series_deg(rot)

if ZERO_MODE == "global_mean":
    m = np.isfinite(flex) & np.isfinite(abd) & np.isfinite(rot)
    if m.any():
        flex -= np.nanmean(flex[m])
        abd -= np.nanmean(abd[m])
        rot -= np.nanmean(rot[m])

out = Path("output") / (trc.stem + "_pelvis_global_ZXY.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["time_s","pelvis_flex_deg(Z)","pelvis_abd_deg(X)","pelvis_rot_deg(Y)"])
    for i in range(F):
        w.writerow([
            f"{times[i]:.6f}",
            "" if not math.isfinite(flex[i]) else f"{flex[i]:.3f}",
            "" if not math.isfinite(abd[i]) else f"{abd[i]:.3f}",
            "" if not math.isfinite(rot[i]) else f"{rot[i]:.3f}",
        ])
print("Output:", out)
EOF
```

---

## Pipeline Steps

| Step | Input | Output | Description |
|------|-------|--------|-------------|
| 1 | Video (.mp4) | pose_world_landmarks.csv | MediaPipe 3D pose extraction |
| 2 | CSV | pose2sim_input_exact.trc | Convert to TRC format (22 markers) |
| 3 | TRC | *_LSTM.trc | Pose2Sim marker augmentation (65 markers) |
| 4 | TRC | *_LSTM_fixed.trc | Fix TRC header with marker names |
| 5 | TRC | *_pelvis_global_ZXY.csv | Compute pelvis angles |

---

## Output Files

| File | Description |
|------|-------------|
| `pose_world_landmarks.csv` | Raw 3D landmarks from MediaPipe (33 per frame) |
| `pose2sim_input_exact.trc` | TRC with 22 input markers for augmenter |
| `pose2sim_input_exact_LSTM.trc` | Augmented TRC with 65 markers |
| `pose2sim_input_exact_LSTM_fixed.trc` | Final TRC with proper header |
| `*_pelvis_global_ZXY.csv` | Pelvis angles: flex/ext, abd/add, rotation |

---

## Participant Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `height` | Height in meters | 1.78 |
| `mass` | Body mass in kg | 75.0 |
| `age` | Age in years | 30 |
| `sex` | "male" or "female" | "male" |

These parameters affect the LSTM marker augmentation scaling.

---

## Pelvis Angles Explained

| Angle | Axis | Description |
|-------|------|-------------|
| Flexion/Extension | Z | Forward/backward pelvic tilt |
| Abduction/Adduction | X | Side-to-side pelvic tilt |
| Axial Rotation | Y | Left/right pelvic rotation |

Euler sequence: ZXY (clinical convention)

---

## Troubleshooting

### "No module named 'mediapipe'"
```bash
pip install mediapipe==0.10.9
```

### "module 'mediapipe' has no attribute 'solutions'"
You have a newer incompatible version. Downgrade:
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.9
```

### "No module named 'onnxruntime'" (or toml, c3d, anytree)
```bash
pip install onnxruntime toml c3d anytree
```

### "Failed to import Qt binding modules"
```bash
pip install PyQt5
```

### GUI dialogs don't appear (WSL/headless)
Use the command-line method shown above to bypass GUI dialogs.

### "Kon video niet openen" (Could not open video)
- Check video file path is correct
- Ensure video is not corrupted
- Try converting to standard MP4: `ffmpeg -i input.mov -c:v libx264 output.mp4`

### Missing pelvis markers in output
The LSTM augmenter must complete successfully to generate `*_study` markers (ASIS, PSIS) needed for pelvis angle computation.

---

## Supported Video Formats

- MP4 (recommended)
- AVI
- MOV
- MKV

For best results, use good lighting and ensure the full body is visible in frame.

---

## References

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [Pose2Sim](https://github.com/perfanalytics/pose2sim)
- ISB recommendations for pelvis coordinate systems

---

## License

Scripts provided as-is for research and educational purposes.
