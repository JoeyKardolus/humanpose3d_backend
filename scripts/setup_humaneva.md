# HumanEva Dataset Setup Guide

## Overview
HumanEva-I: Synchronized video + 3D mocap dataset
- 7 subjects, 6 actions (walking, jogging, throwing, gestures, box, combo)
- 3-4 calibrated cameras per sequence
- 3D mocap ground truth (15 body joints)
- **Size**: ~4GB total
- **License**: Free for research (registration required)

## Step 1: Register and Download

**Official site**: http://humaneva.is.tue.mpg.de/datasets_human_1

**Or Max Planck Institute**: https://is.mpg.de/ps/code/humaneva-datasets-i-and-ii

1. Click "Download" or "Register"
2. Fill in registration form (academic/research use)
3. Accept license agreement
4. Usually **instant or same-day approval**
5. Download HumanEva-I dataset

**Choose download format:**
- `.zip` (for Windows/Mac)
- `.tar.gz` (for Linux)

## Step 2: Dataset Structure

After extraction:
```
HumanEva/
├── S1/           # Subject 1
│   ├── Walking_1/
│   │   ├── frame_000001.jpg  # Video frames
│   │   ├── mocap.txt         # 3D mocap positions
│   │   └── camera.cal        # Camera calibration
│   ├── Jogging_1/
│   └── ...
├── S2/
└── ...
```

## Step 3: Quick Download Script

```bash
#!/bin/bash
# After registration, download HumanEva-I

# Create directory
mkdir -p data/humaneva
cd data/humaneva

# Download (replace URL with your download link from email)
wget -O humaneva_i.tar.gz "YOUR_DOWNLOAD_LINK_HERE"

# Extract
tar -xzf humaneva_i.tar.gz

# Cleanup
rm humaneva_i.tar.gz

echo "✓ HumanEva-I dataset ready!"
```

## Step 4: Convert to Training Data

We'll create a script that:
1. Loads video frames
2. Runs MediaPipe on them → Get noisy 3D poses
3. Loads mocap ground truth → Clean 3D poses
4. Creates training pairs: (MediaPipe output, Mocap truth)

## Advantages over Human3.6M

| Feature | HumanEva | Human3.6M |
|---------|----------|-----------|
| Registration | Quick (hours) | Slow (1-2 days) |
| Size | 4GB | 100GB |
| Video format | JPEG frames | MP4 files |
| Joints | 15 body | 32 joints |
| Access | Open | Restricted |

## Advantages over CMU Mocap

| Feature | HumanEva | CMU Mocap |
|---------|----------|-----------|
| Video footage | ✅ Real videos | ❌ No video |
| MediaPipe errors | ✅ Real corruption | ❌ Simulated |
| Multi-view | ✅ 3-4 cameras | ❌ Single view |
| Quality | ✅ Real-world | ⚠️ Synthetic |

## Next Steps

1. Register at http://humaneva.is.tue.mpg.de/datasets_human_1
2. Download dataset (choose 1-2 subjects for quick test)
3. Run: `python scripts/convert_humaneva_to_training.py`
4. Train depth refinement model on REAL MediaPipe data!

## Sources
- [HumanEva Dataset](http://humaneva.is.tue.mpg.de/datasets_human_1)
- [MPI-IS Download Page](https://is.mpg.de/ps/code/humaneva-datasets-i-and-ii)
- [GitHub Support Code](https://github.com/emredog/humaneva)
