# Human3.6M Dataset Setup Guide

## Overview
Human3.6M: 3.6 million 3D human poses with video footage
- 11 subjects performing 15 actions
- 4 camera views per action
- 3D mocap ground truth (32 joints)
- Video footage at 50 FPS

## Step 1: Request Access

**Official website**: http://vision.imar.ro/human3.6m/

1. Go to the website
2. Click "Register" or "Download"
3. Fill out the agreement form (academic use)
4. Wait for approval email (usually 1-2 days)

## Step 2: Download Options

### Option A: Official Download (Recommended)
Once approved, you get credentials to download from their server.

**What to download:**
- **Videos**: Subject 1, 5, 6, 7, 8 (training), Subject 9, 11 (validation)
- **Poses**: 3D joint positions (D3 Positions)
- **Camera parameters**: For projection

**Size**: ~100GB for videos + poses

### Option B: Pre-processed Version (Faster)

Use existing pre-processed datasets:

```bash
# Clone repository with download scripts
git clone https://github.com/una-dinosauria/3d-pose-baseline
cd 3d-pose-baseline/data

# Download pre-processed poses (no videos, just 3D data)
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
```

**This gives you:**
- 3D joint positions (already extracted)
- Train/test splits
- NO videos (just poses)

### Option C: PyTorch Dataset (Easiest for Training)

```bash
pip install human36m-dataset

# In Python:
from human36m_dataset import Human36M
dataset = Human36M(root='./data', download=True)
```

## Step 3: What We Actually Need

For depth refinement training, we need:
1. **Video frames** → Run MediaPipe → Get noisy 3D poses
2. **Ground truth 3D poses** → Target for training

**Recommended approach:**
1. Download pre-processed poses (Option B)
2. Optionally download videos for 1-2 subjects to test MediaPipe
3. Use existing MediaPipe detections if available

## Step 4: Data Structure

```
data/human36m/
├── S1/           # Subject 1
│   ├── Videos/
│   │   ├── Directions.54138969.mp4
│   │   └── ...
│   └── Poses/
│       ├── Directions.h5
│       └── ...
├── S5/
├── ...
```

## Alternative: Synthetic Data (Faster)

If Human3.6M approval takes too long:

**AMASS Dataset** (Immediate access):
- 15 hours of mocap data
- 300+ subjects
- Can render synthetic videos
- No approval needed

```bash
# Download AMASS
wget https://amass.is.tue.mpg.de/download.php
# Sign up, get immediate download link
```

## Next Steps After Download

1. Extract 3D poses
2. Run MediaPipe on videos → Get corrupted poses
3. Create training pairs: (MediaPipe output, Ground truth)
4. Train depth refinement model

