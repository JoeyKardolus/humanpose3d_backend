# FLK Filtering Setup

This guide explains how to install and use FLK (Filter with Learned Kinematics) for spatio-temporal smoothing of pose landmarks in the HumanPose pipeline.

## What is FLK?

FLK is a real-time filter that combines:
- **Adaptive Kalman filtering** with RNN-learned motion models
- **Biomechanical constraints** for anatomical coherency
- **Low-pass filtering** for temporal smoothing

It effectively reduces jitter, handles missing frames, and smooths noisy pose detections from MediaPipe.

## Installation

### 1. Install FLK from GitHub

```bash
# Clone the FLK repository
git clone https://github.com/PARCO-LAB/FLK.git
cd FLK

# Install dependencies
pip install -r requirements.txt

# Build and install the package
python3 -m pip install --upgrade build
python3 -m build
pip3 install dist/flk-0.0.1-py3-none-any.whl

# Return to your project
cd ..
```

### 2. Download Pre-trained Model (Optional)

FLK includes a pre-trained GRU model for motion prediction. If you cloned the repository, the model should be at `FLK/models/GRU.h5`.

To make it accessible to the pipeline, either:
- Copy it to your project: `cp FLK/models/GRU.h5 models/`
- Use `--flk-model` flag to point to the FLK directory

**Note**: FLK works without the RNN model (Kalman filter only), but the RNN improves motion prediction for complex movements.

## Usage

### Basic Filtering (Kalman + Biomechanical Constraints)

```bash
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 \
  --mass 75.0 \
  --age 30 \
  --sex male \
  --flk-filter \
  --estimate-missing
```

This applies FLK filtering to the 14 core body markers after MediaPipe extraction, before TRC conversion.

### Advanced Filtering (with RNN)

```bash
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 \
  --mass 75.0 \
  --age 30 \
  --sex male \
  --flk-filter \
  --flk-enable-rnn \
  --flk-model FLK/models/GRU.h5 \
  --estimate-missing
```

**Warning**: RNN mode is slower but provides better motion prediction for complex movements.

## How It Works in the Pipeline

The FLK filter is applied after MediaPipe landmark extraction (Step 1) and before marker estimation:

```
1. Video → MediaPipe landmarks (CSV with 22 markers)
2. [FLK Filter] → Smooth 14 core markers (if --flk-filter)
3. [Estimation] → Fill missing markers (if --estimate-missing)
4. TRC conversion
5. Pose2Sim augmentation
```

### Filtered Markers (14 Core)

FLK filters these anatomically important markers:
- **Upper body**: Neck, Head, RShoulder, LShoulder
- **Arms**: RElbow, LElbow, RWrist, LWrist
- **Torso/Legs**: RHip, LHip, RKnee, LKnee, RAnkle, LAnkle

**Excluded markers** (not filtered, passed through unchanged):
- Hip (derived marker, redundant)
- Nose (facial, not body kinematics)
- Foot details: RHeel, LHeel, RBigToe, LBigToe, RSmallToe, LSmallToe

These excluded markers are often noisy in MediaPipe and are better handled by the `--estimate-missing` flag or Pose2Sim augmentation.

## Performance Impact

- **Without RNN**: ~1-2 seconds overhead for typical 30-second video
- **With RNN**: ~3-5 seconds overhead (slower but more accurate)

FLK processes each frame sequentially, so longer videos take proportionally longer.

## Troubleshooting

### ImportError: No module named 'FLK'

FLK is not installed. Follow the installation steps above.

### WARNING: No FLK model found, will run without RNN

The GRU model file wasn't found. Either:
- Copy the model to `models/GRU.h5` in your project
- Use `--flk-model FLK/models/GRU.h5` to point to the FLK directory
- Continue without RNN (Kalman filter still works)

### TensorFlow errors

FLK requires TensorFlow for the RNN component. Install it:
```bash
pip install tensorflow>=2.13.0
```

Or disable RNN by not using `--flk-enable-rnn` flag.

## When to Use FLK Filtering

**Use FLK when**:
- MediaPipe detections are jittery or unstable
- Subject movements are smooth (walking, running, exercise)
- You need temporally consistent trajectories
- Video has good lighting and minimal occlusions

**Skip FLK when**:
- Video has many dropped frames or heavy occlusions (use `--estimate-missing` instead)
- Subject makes rapid, jerky movements (filter may over-smooth)
- You want maximum processing speed (FLK adds overhead)

## Combining with Other Filters

FLK works well with other pipeline features:

```bash
# Recommended: FLK + estimation + augmentation
uv run python main.py \
  --video data/input/video.mp4 \
  --height 1.78 --mass 75.0 --age 30 --sex male \
  --flk-filter \
  --estimate-missing \
  --force-complete \
  --augmentation-cycles 20
```

This gives you:
1. Smooth core landmarks from FLK
2. Estimated missing markers from symmetry
3. Complete augmented marker set from Pose2Sim

## References

- **FLK Paper**: Meli et al., "FLK: Filter with Learned Kinematics for 3D Human Pose Estimation", Signal Processing (2024)
- **GitHub**: https://github.com/PARCO-LAB/FLK
- **License**: BSD-3-Clause
