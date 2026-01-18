# HumanPose3D - MediaPipe to Biomechanics Pipeline

3D human pose estimation pipeline using MediaPipe for detection and Pose2Sim for marker augmentation, with advanced biomechanical constraint optimization.

## Prerequisites

### 1. Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env  # Add to PATH (or restart shell)
```

### 2. Clone and setup

```bash
git clone <repo-url>
cd humanpose3d_backend
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
# Run with multi-constraint optimization + joint angles (RECOMMENDED)
uv run python main.py \
  --video data/input/joey.mp4 \
  --height 1.78 \
  --mass 75 \
  --age 30 \
  --sex male \
  --anatomical-constraints \
  --bone-length-constraints \
  --estimate-missing \
  --force-complete \
  --augmentation-cycles 20 \
  --multi-constraint-optimization \
  --compute-all-joint-angles \
  --plot-all-joint-angles \
  --visibility-min 0.1

# Visualize results
uv run python visualize_interactive.py data/output/pose-3d/joey/joey_final.trc
```

## Features

- ✅ **MediaPipe Pose Detection** - 33 landmarks → 22 Pose2Sim markers
- ✅ **GPU-Accelerated LSTM Augmentation** - 21 → 64 markers (full OpenCap set) with 3-10x speedup
- ✅ **Multi-Constraint Optimization** - 68% bone length improvement, automatic unreliable marker filtering
- ✅ **Neural Depth Refinement** - PoseFormer-based depth correction using biomechanical constraints
- ✅ **Biomechanical Constraints** - Enforce realistic bone lengths, ground contact, and hip width
- ✅ **Marker Estimation** - Fill missing data using anatomical symmetry
- ✅ **Comprehensive Joint Angles** - ISB-compliant computation for all joints (pelvis, lower body, trunk, upper body)
- ✅ **Interactive Visualization** - 3D skeleton viewer with playback controls
- ✅ **Automatic Output Organization** - Clean directory structure with organized angle files
- ✅ **Automatic CPU Fallback** - Works on any system, GPU optional

## Results

**Benchmark** (joey.mp4, 535 frames, 20 augmentation cycles, GPU acceleration):
- **Processing time**: ~45 seconds (3-10x faster with GPU vs CPU)
- **Bone length consistency**: 68% improvement (0.113 → 0.036 CV)
- **Marker quality**: 59/64 markers (unreliable markers auto-filtered)
- **Joint angles**: 12 joint groups computed (pelvis, hip, knee, ankle, trunk, shoulder, elbow)
- **Output**: Clean directory structure with automatic organization
- **GPU**: Automatic CUDA acceleration for LSTM inference, graceful CPU fallback

## Documentation

### User Guides
- [CLAUDE.md](CLAUDE.md) - Full pipeline documentation and usage guide
- [docs/OUTPUT_ORGANIZATION.md](docs/OUTPUT_ORGANIZATION.md) - Output directory structure and file descriptions

### Technical Details
- [docs/MULTI_CONSTRAINT_OPTIMIZATION.md](docs/MULTI_CONSTRAINT_OPTIMIZATION.md) - Multi-constraint optimization algorithm
- [docs/CLEANUP_COMPLETE_2026-01-09.md](docs/CLEANUP_COMPLETE_2026-01-09.md) - Code cleanup report

### Development History
- [docs/BUILD_LOG.md](docs/BUILD_LOG.md) - Development history and decisions
- [docs/SESSION_SUMMARY_2026-01-09.md](docs/SESSION_SUMMARY_2026-01-09.md) - Latest session notes

## Pipeline Overview

```
Video → MediaPipe → CSV → TRC → GPU-Accelerated LSTM → Multi-Constraint Optimization → Joint Angles → Clean Output
```

1. **Extraction**: MediaPipe detects 33 landmarks, mapped to 22 Pose2Sim markers
2. **Augmentation**: GPU-accelerated Pose2Sim LSTM adds 43 augmented markers (medial, shoulder clusters, HJC)
3. **Optimization**: Multi-constraint refinement with unreliable marker filtering
4. **Joint Angles**: ISB-compliant computation for all 12 joint groups
5. **Output**: Organized directory with final TRC, initial TRC, raw CSV, and joint angles

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

## Neural Depth Refinement (Optional)

For advanced users, the pipeline includes a **PoseFormer-based neural depth refinement** model that learns to correct depth errors using biomechanical constraints.

### Training the Model

Train on CMU Motion Capture data (professional mocap as ground truth):

```bash
# 1. Install neural dependencies (PyTorch, etc.)
uv sync --group neural

# 2. Download CMU Motion Capture dataset (~2GB)
cd data/training/cmu_mocap
git clone https://github.com/una-dinosauria/cmu-mocap.git
cd ../../..

# 3. Generate training data from CMU mocap (~500MB, ~5 minutes)
uv run --group neural python training/generate_training_data.py

# 4. Train PoseFormer model (~9 hours on RTX 5080)
uv run --group neural python scripts/train_depth_model.py \
  --batch-size 64 \
  --epochs 50 \
  --workers 8 \
  --fp16
```

**Training details:**
- **Dataset**: 208K temporal sequences from CMU mocap with simulated depth errors
- **Architecture**: PoseFormer (25.5M parameters, Transformer-based)
- **Training**: 6 camera angles (0-75°), 3 noise levels (30-80mm)
- **Losses**: Bone length, ground plane, symmetry, smoothness, joint angles
- **Output**: `models/checkpoints/best_depth_model.pth`

### Applying Depth Refinement

Once trained, apply the model to refine TRC files:

```bash
# Refine depth in final TRC output
uv run --group neural python scripts/apply_depth_refinement.py \
  --input data/output/pose-3d/joey/joey_final.trc \
  --model models/checkpoints/best_depth_model.pth \
  --output data/output/pose-3d/joey/joey_refined.trc
```

**What it does:**
- Automatically estimates camera viewing angle from torso orientation
- Applies learned depth corrections using 11-frame temporal context
- Outputs refined TRC with improved Z-axis (depth) accuracy

**When to use:**
- Research applications requiring high depth accuracy
- Motion capture validation and quality improvement
- Comparative biomechanics studies

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
