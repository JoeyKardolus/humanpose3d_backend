# HumanPose3D - MediaPipe to Biomechanics Pipeline

3D human pose estimation pipeline using MediaPipe for detection and Pose2Sim for marker augmentation, with advanced biomechanical constraint optimization.

## Table of Contents

- [Prerequisites](#prerequisites)
  - [1. Install uv (Python package manager)](#1-install-uv-python-package-manager)
  - [2. Clone and setup](#2-clone-and-setup)
  - [3. Environment setup (headless/WSL)](#3-environment-setup-headlesswsl)
- [Usage](#usage)
  - [GUI (Web App)](#gui-web-app)
  - [CLI (Management Command)](#cli-management-command)
  - [API (cURL)](#api-curl)
- [Standalone App (PyInstaller)](#standalone-app-pyinstaller)
  - [Building the Executable](#building-the-executable)
  - [Features](#features-1)
  - [About the Spec Files](#about-the-spec-files)
- [Features](#features)
- [Results](#results)
- [Documentation](#documentation)
  - [User Guides](#user-guides)
  - [Technical Details](#technical-details)
  - [Development](#development)
- [Pipeline Overview](#pipeline-overview)
  - [Processing Steps](#processing-steps)
  - [Output Structure](#output-structure)
- [Neural Depth Refinement (Optional)](#neural-depth-refinement-optional)
  - [Training the Model](#training-the-model)
  - [Applying Depth Refinement](#applying-depth-refinement)
- [Requirements](#requirements)
  - [System Requirements](#system-requirements)
  - [Optional System Dependencies](#optional-system-dependencies)
  - [Core Dependencies](#core-dependencies)
  - [Neural Refinement Dependencies (Optional)](#neural-refinement-dependencies-optional)
  - [GPU Acceleration (Optional)](#gpu-acceleration-optional)
- [Common Issues](#common-issues)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## Prerequisites

### 1. Install uv (Python package manager)

Windows (PowerShell):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env  # Add to PATH (or restart shell)
```

Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env  # Add to PATH (or restart shell)
```

### 2. Clone and setup

```bash
git clone git@github.com:JoeyKardolus/humanpose3d_backend.git
cd humanpose3d_backend

uv python install 3.12  # Installs Python version 3.12
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

Install the system `tesseract` binary if your preview videos are rotated and the source metadata is missing. OCR-based rotation detection uses it when generating preview videos.

Windows:
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
(Add to PATH during installation)

macOS:
Download and install from: https://github.com/tesseract-ocr/tesseract/wiki/Downloads
Or compile from source: https://tesseract-ocr.github.io/tessdoc/Compiling.html

Linux:
```bash
sudo apt-get install tesseract-ocr
```

## Usage

### GUI (Web App)

1. Start the server:
   ```bash
   uv run python manage.py runserver
   ```
2. Open `http://127.0.0.1:8000/`.
3. Upload a video, set subject details, and run the pipeline.
4. View results, download outputs, and inspect statistics from the results page.

### CLI (Management Command)

Run with neural refinement + joint angles (recommended):
```bash
uv run python manage.py run_pipeline \
  --video PATH/TO/VIDEO.mp4 \
  --height 1.78 \
  --weight 75 \
  --estimate-missing \
  --force-complete \
  --augmentation-cycles 20 \
  --main-refiner \
  --plot-all-joint-angles \
  --visibility-min 0.1
```

#### Key CLI Flags

| Flag | Description |
|------|-------------|
| `--main-refiner` | **Recommended**: Full neural pipeline (depth + joint refinement) |
| `--estimate-missing` | Mirror occluded limbs from visible side |
| `--force-complete` | Estimate shoulder clusters + hip joint centers |
| `--augmentation-cycles N` | Multi-cycle averaging (default 20) |
| `--plot-all-joint-angles` | Multi-panel visualization |
| `--visibility-min 0.1` | Landmark confidence threshold (default 0.3, use 0.1 to prevent marker dropout) |
| `--show-video` | Display MediaPipe preview during processing |
| `--export-preview` | Save preview video with landmarks overlay |

Visualize results:
```bash
uv run python scripts/viz/visualize_interactive.py data/output/joey/joey_final.trc
```

### API (cURL)

Start a run asynchronously (multipart form upload):
```bash
curl -F "video=@data/input/joey.mp4" \
  -F "height=1.78" \
  -F "weight=75" \
  -F "consent=accepted" \
  -F "estimate_missing=on" \
  -F "force_complete=on" \
  -F "augmentation_cycles=20" \
  -F "main_refiner=on" \
  -F "plot_all_joint_angles=on" \
  -F "save_angle_comparison=on" \
  -F "show_all_markers=on" \
  -F "show_video=on" \
  -F "export_preview=on" \
  -F "plot_landmarks=on" \
  -F "plot_augmented=on" \
  -F "visibility_min=0.1" \
  -F "temporal_smoothing=3" \
  -F "depth_model_path=models/checkpoints/best_depth_model.pth" \
  -F "joint_model_path=models/checkpoints/best_joint_model.pth" \
  -F "main_refiner_path=models/checkpoints/best_main_refiner.pth" \
  http://127.0.0.1:8000/api/runs/
```

Poll progress:
```bash
curl http://127.0.0.1:8000/api/runs/<run_key>/progress/
```

Fetch results list:
```bash
curl http://127.0.0.1:8000/api/runs/<run_key>/
```

## Standalone App (PyInstaller)

Build a standalone executable for distribution (Windows/macOS/Linux). The executable bundles the Django server and automatically opens a web browser when launched.

### Building the Executable

1. Install PyInstaller:
   ```bash
   uv pip install pyinstaller
   ```

2. Build using the build script **(recommended)**:
   ```bash
   ./scripts/packaging/build.sh linux   # or macos, windows
   ```

   This automatically handles all flags and paths:
   - Outputs to `bin/HumanPose3D-<platform>/`
   - Build artifacts to `bin/build/`
   - Auto-confirms overwrites with `-y`

   **Or build directly with PyInstaller:**
   ```bash
   uv run pyinstaller scripts/packaging/HumanPose3D-linux.spec -y --distpath bin --workpath bin/build
   ```

3. Run the executable:
   - Linux/macOS:
     ```bash
     ./bin/HumanPose3D-linux/HumanPose3D-linux
     ```
   - Windows:
     ```powershell
     .\bin\HumanPose3D-windows\HumanPose3D-windows.exe
     ```

4. **(Optional) Create launcher script** (Linux only):
   ```bash
   ./scripts/packaging/create_launcher.sh
   ```
   This creates `HumanPose3D.sh` which opens the app in a terminal window. Useful for double-clicking from file managers.

### Features

- Starts Django server at `http://127.0.0.1:8000/`
- Automatically opens default web browser when launched
- Shows clear terminal output with server status and stop instructions (Ctrl+C)
- Optional launcher script for terminal window on double-click
- No Python installation required on target system
- Bundles all dependencies, templates, static files, and neural models
- Supports CLI commands: `./HumanPose3D-linux run_pipeline --video ...`

### About the Spec Files

Spec files in `scripts/packaging/` (e.g., `HumanPose3D-linux.spec`) contain build configuration:
- Entry point: `scripts/packaging/pyinstaller_entry.py` (handles server startup and browser opening)
- Bundled data: templates, static files, neural models
- Hidden imports: all Django apps and Python modules

The spec files should be committed to version control for reproducible builds.

## Features

- ✅ **MediaPipe Pose Detection** - 33 landmarks → 22 Pose2Sim markers
- ✅ **GPU-Accelerated LSTM Augmentation** - 22 → 64 markers (full OpenCap set) with 3-10x speedup
- ✅ **Neural Depth Refinement** - Transformer-based depth correction trained on AIST++ motion capture
- ✅ **Neural Joint Refinement** - Learned soft joint constraints from motion capture data
- ✅ **MainRefiner Pipeline** - Unified neural pipeline combining depth + joint refinement
- ✅ **Marker Estimation** - Fill missing data using anatomical symmetry
- ✅ **Comprehensive Joint Angles** - ISB-compliant computation for all joints (pelvis, lower body, trunk, upper body)
- ✅ **Interactive Visualization** - 3D skeleton viewer with playback controls
- ✅ **Automatic Output Organization** - Clean directory structure with organized angle files
- ✅ **Automatic CPU Fallback** - Works on any system, GPU optional

## Results

**Benchmark** (joey.mp4, 535 frames, 20 augmentation cycles, neural refinement):
- **Processing time**: ~60 seconds
- **Depth accuracy**: 45% improvement with neural refinement
- **Marker quality**: 59/64 markers
- **Joint angles**: 12 joint groups computed (pelvis, hip, knee, ankle, trunk, shoulder, elbow)
- **Output**: Clean directory structure with automatic organization
- **GPU**: Automatic CUDA acceleration for LSTM inference, graceful CPU fallback

## Documentation

### User Guides
- [docs/OUTPUT_ORGANIZATION.md](docs/OUTPUT_ORGANIZATION.md) - Output directory structure and file descriptions

### Technical Details
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and module structure
- [docs/NEURAL_MODELS.md](docs/NEURAL_MODELS.md) - Neural refinement models (depth + joint)

### Development
- [CLAUDE.md](CLAUDE.md) - AI assistant instructions and codebase guidelines
- [docs/AGENTS.md](docs/AGENTS.md) - Development principles and architectural guidelines
- [docs/CHANGELOG.md](docs/CHANGELOG.md) - Development history and milestones
- [docs/BUILD_LOG.md](docs/BUILD_LOG.md) - Run logs and testing notes

## Pipeline Overview

```
Video → MediaPipe → Neural Depth Refinement → TRC → GPU-Accelerated LSTM → Joint Angles → Neural Joint Refinement → Output
```

### Processing Steps

1. **MediaPipe extraction** → 33 landmarks → 22 Pose2Sim markers
2. **Neural depth refinement** → corrects MediaPipe depth errors (17 COCO joints)
3. **TRC conversion** with derived markers (Hip, Neck)
4. **Pose2Sim augmentation** → 64 markers (43 added via LSTM)
5. **Joint angle computation** → 12 ISB-compliant joint groups
6. **Neural joint refinement** → corrects joint angles using learned constraints
7. **Automatic cleanup** → organized output structure

**Neural Refinement Pipeline** (`--main-refiner` flag):
- **Stage 1 (Pre-augmentation)**: Depth refinement corrects MediaPipe 3D errors on 17 COCO joints
- **Stage 2 (Post-augmentation)**: Joint constraint model refines computed angles on 64 markers
- **Performance**: ~60s processing, 45% depth improvement, <10ms per frame on CPU

**GPU Acceleration**: Automatic CUDA support for 3-10x speedup on augmentation. CPU fallback if GPU unavailable.

### Output Structure
```
data/output/<video>/
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
  --input data/output/joey/joey_final.trc \
  --model models/checkpoints/best_depth_model.pth \
  --output data/output/joey/joey_refined.trc
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

## Common Issues

| Issue | Solution |
|-------|----------|
| Right arm missing | Use `--estimate-missing` to mirror from left |
| "No trc files found" | Check Pose2Sim project structure |
| Depth errors / front-back confusion | Use `--main-refiner` (neural depth correction) |
| Joint angle spikes | Use `--main-refiner` (learned joint constraints) |
| Markers disappear mid-video | Use `--visibility-min 0.1` (MediaPipe confidence drops below default 0.3 threshold) |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv: command not found` | Run `curl -LsSf https://astral.sh/uv/install.sh \| sh` then restart shell |
| `ModuleNotFoundError: tkinter` | Set `export MPLBACKEND=Agg` or install `python3-tk` |
| TensorFlow CUDA warnings | Safe to ignore - TF/ONNX conflict, doesn't affect functionality |
| `CUDA not available` | Install CUDA toolkit or use CPU (automatic fallback) |
| Permission denied on video | Check file path and permissions |
| `Pose2Sim` import fails | Use `from Pose2Sim import Pose2Sim` (capitalized) |

## Citation

If you use this pipeline in your research, please cite:
- MediaPipe: [Lugaresi et al. 2019](https://arxiv.org/abs/1906.08172)
- Pose2Sim: [Pagnon et al. 2022](https://joss.theoj.org/papers/10.21105/joss.04362)

## License

See [LICENSE](LICENSE) file for details
