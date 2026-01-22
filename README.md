# HumanPose3D - MediaPipe to Biomechanics Pipeline

3D human pose estimation pipeline using MediaPipe for detection, POF (Part Orientation Fields) for 3D reconstruction, and Pose2Sim for marker augmentation.

## Prerequisites

### 1. Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env  # Add to PATH (or restart shell)
```

### 2. Clone and setup

```bash
git clone <repo-url>
cd humanpose3d_mediapipe
uv sync  # Creates .venv and installs all dependencies
```

### 3. Environment setup (headless/WSL)

For headless environments (servers, WSL, Docker), set the matplotlib backend:

```bash
export MPLBACKEND=Agg  # Add to .bashrc for persistence
```

Or use the included `.envrc` with [direnv](https://direnv.net/):

```bash
direnv allow  # Automatically sets MPLBACKEND=Agg
```

## Quick Start

```bash
# Run with neural refinement + joint angles (RECOMMENDED)
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 \
  --mass 75 \
  --estimate-missing \
  --force-complete \
  --augmentation-cycles 20 \
  --main-refiner \
  --plot-all-joint-angles \
  --visibility-min 0.1

# Visualize results
uv run python scripts/viz/visualize_interactive.py data/output/pose-3d/joey/joey_final.trc
```

## Features

- ✅ **MediaPipe Pose Detection** - 33 landmarks → 22 Pose2Sim markers
- ✅ **POF 3D Reconstruction** - Part Orientation Fields reconstruct 3D from 2D (bypasses MediaPipe depth errors)
- ✅ **GPU-Accelerated LSTM Augmentation** - 22 → 64 markers (full OpenCap set) with 3-10x speedup
- ✅ **Neural Joint Refinement** - Learned soft joint constraints from motion capture data
- ✅ **MainRefiner Pipeline** - Unified neural pipeline combining POF + joint refinement
- ✅ **Marker Estimation** - Fill missing data using anatomical symmetry
- ✅ **Comprehensive Joint Angles** - ISB-compliant computation for all joints (pelvis, lower body, trunk, upper body)
- ✅ **Interactive Visualization** - 3D skeleton viewer with playback controls
- ✅ **Automatic Output Organization** - Clean directory structure with organized angle files
- ✅ **Automatic CPU Fallback** - Works on any system, GPU optional

## Results

**Benchmark** (joey.mp4, 535 frames, 20 augmentation cycles, neural refinement):
- **Processing time**: ~60 seconds
- **POF accuracy**: ~11° limb orientation error (beats MediaPipe 3D at 16.3°)
- **Marker quality**: 59/64 markers
- **Joint angles**: 12 joint groups computed (pelvis, hip, knee, ankle, trunk, shoulder, elbow)
- **Output**: Clean directory structure with automatic organization
- **GPU**: Automatic CUDA acceleration for LSTM inference, graceful CPU fallback

## Documentation

### User Guides
- [CLAUDE.md](CLAUDE.md) - Full pipeline documentation and usage guide
- [docs/OUTPUT_ORGANIZATION.md](docs/OUTPUT_ORGANIZATION.md) - Output directory structure and file descriptions

### Technical Details
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and module structure
- [docs/NEURAL_MODELS.md](docs/NEURAL_MODELS.md) - Neural refinement models (depth + joint)

### Development History
- [docs/CHANGELOG.md](docs/CHANGELOG.md) - Development history and milestones
- [docs/BUILD_LOG.md](docs/BUILD_LOG.md) - Run logs and testing notes

## Pipeline Overview

```
Video → MediaPipe 2D → POF 3D Reconstruction → TRC → GPU-Accelerated LSTM → Joint Angles → Neural Joint Refinement → Output
```

1. **Extraction**: MediaPipe detects 33 landmarks, mapped to 22 Pose2Sim markers
2. **POF 3D Reconstruction**: Part Orientation Fields reconstruct 3D from 2D keypoints (with `--main-refiner`)
3. **Augmentation**: GPU-accelerated Pose2Sim LSTM adds 43 markers (medial, shoulder clusters, HJC)
4. **Joint Angles**: ISB-compliant computation for all 12 joint groups
5. **Joint Refinement**: Neural model applies learned soft constraints (with `--main-refiner`)
6. **Output**: Organized directory with final TRC, initial TRC, raw CSV, and joint angles

**GPU Acceleration**: Automatic CUDA support for 3-10x speedup on augmentation. CPU fallback if GPU unavailable.

### Output Structure
```
data/output/pose-3d/<video>/
├── <video>_final.trc               # Final optimized skeleton (59-64 markers)
├── <video>_initial.trc             # Initial MediaPipe output (22 markers)
├── <video>_raw_landmarks.csv       # Raw landmark data
└── joint_angles/                   # Joint angle analysis
    ├── <video>_all_joint_angles.png
    └── 12 CSV files (one per joint group)
```

## Neural Refinement (Optional)

The pipeline includes **Part Orientation Fields (POF)** for 3D reconstruction and **neural joint refinement** for biomechanical constraints.

### Quick Start (Recommended)

```bash
# Full pipeline with POF + joint refinement
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 --mass 75 \
  --estimate-missing --force-complete \
  --augmentation-cycles 20 \
  --main-refiner \
  --plot-all-joint-angles
```

The `--main-refiner` flag enables:
- **POF 3D reconstruction**: Predicts 14 per-limb 3D unit vectors from 2D keypoints
- **Joint constraint refinement**: Applies learned soft constraints to computed angles

### Training POF Model

```bash
# 1. Install neural dependencies
uv sync --group neural

# 2. Generate training data from AIST++ dataset
uv run python scripts/data/convert_aistpp.py --workers 8

# 3. Train POF model
uv run --group neural python scripts/train/pof_model.py \
  --data data/training/aistpp_converted \
  --epochs 50 --batch-size 256 --workers 8 --bf16 \
  --d-model 128 --num-layers 6 --num-heads 8
```

**Model details:**
- **Architecture**: Transformer predicting 14 per-limb 3D unit vectors
- **Training data**: AIST++ dataset (1.2M frames, 9 camera views)
- **Performance**: ~11° limb orientation error (beats MediaPipe 3D at 16.3°)

## Requirements

### System Requirements

- **Python**: 3.12+ (tested with 3.12.3)
- **OS**: Linux, macOS, Windows (WSL2 recommended for Windows)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~5GB for dependencies

### Optional System Dependencies

```bash
# For GUI matplotlib (interactive plots) - not needed for headless/WSL
sudo apt install python3-tk

# For GPU acceleration (see GPU section below)
sudo apt install cuda-toolkit-12-6 libcudnn9-cuda-12
```

### Core Dependencies

Managed via `pyproject.toml` and installed with `uv sync`:

| Package | Version | Purpose |
|---------|---------|---------|
| mediapipe | ≥0.10.21 | Pose detection (33 landmarks) |
| pose2sim | ≥0.10.0 | Marker augmentation (LSTM) |
| opencv-python | ≥4.11.0 | Video I/O |
| numpy | ≥1.24.0 | Array operations |
| pandas | ≥2.0.0 | Data manipulation |
| matplotlib | ≥3.7.0 | Visualization |
| tensorflow | ≥2.13.0 | LSTM backend |
| onnxruntime-gpu | ≥1.23.0 | GPU-accelerated inference |
| torch | ≥2.9.1 | Neural models |
| django | ≥6.0.1 | Web API (optional) |

### Neural Refinement Dependencies (Optional)

Install with `uv sync --group neural` for training neural depth/joint models:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.9.1 | Neural network training |
| torchvision | ≥0.24.1 | Vision utilities |
| einops | ≥0.8.1 | Tensor operations |
| ezc3d | ≥1.5.0 | C3D file support |
| tensorboard | ≥2.18.0 | Training visualization |
| tqdm | ≥4.66.0 | Progress bars |

### GPU Acceleration (Optional)

For **3-10x speedup** on LSTM augmentation. **Not required** - pipeline automatically falls back to CPU.

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x
- cuDNN 9 for CUDA 12

**Installation (Ubuntu/WSL2):**
```bash
# Add NVIDIA CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA Toolkit and cuDNN
sudo apt install -y cuda-toolkit-12-6 libcudnn9-cuda-12

# Reinstall onnxruntime-gpu to detect CUDA
uv pip install onnxruntime-gpu --force-reinstall
```

**Verify GPU:**
```bash
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should see: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv: command not found` | Run `curl -LsSf https://astral.sh/uv/install.sh \| sh` then restart shell |
| `ModuleNotFoundError: tkinter` | Set `export MPLBACKEND=Agg` or install `python3-tk` |
| TensorFlow CUDA warnings | Safe to ignore - TF/ONNX conflict, doesn't affect functionality |
| `CUDA not available` | Install CUDA toolkit or use CPU (automatic fallback) |
| Permission denied on video | Check file path and permissions |

## Citation

If you use this pipeline in your research, please cite:
- MediaPipe: [Lugaresi et al. 2019](https://arxiv.org/abs/1906.08172)
- Pose2Sim: [Pagnon et al. 2022](https://joss.theoj.org/papers/10.21105/joss.04362)

## License

See [LICENSE](LICENSE) file for details
