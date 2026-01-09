# HumanPose3D - MediaPipe to Biomechanics Pipeline

3D human pose estimation pipeline using MediaPipe for detection and Pose2Sim for marker augmentation, with advanced biomechanical constraint optimization.

## Quick Start

```bash
# Install dependencies
uv sync

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
  --plot-all-joint-angles

# Visualize results
uv run python visualize_interactive.py data/output/pose-3d/joey/joey_final.trc
```

## Features

- ✅ **MediaPipe Pose Detection** - 33 landmarks → 22 Pose2Sim markers
- ✅ **Pose2Sim LSTM Augmentation** - 22 → 65 markers (full OpenCap set)
- ✅ **Multi-Constraint Optimization** - 68% bone length improvement, automatic unreliable marker filtering
- ✅ **Biomechanical Constraints** - Enforce realistic bone lengths, ground contact, and hip width
- ✅ **Marker Estimation** - Fill missing data using anatomical symmetry
- ✅ **Comprehensive Joint Angles** - ISB-compliant computation for all joints (pelvis, lower body, trunk, upper body)
- ✅ **Interactive Visualization** - 3D skeleton viewer with playback controls
- ✅ **Automatic Output Organization** - Clean directory structure with organized angle files

## Results

**Benchmark** (joey.mp4, 535 frames, 20 augmentation cycles):
- **Processing time**: ~45 seconds
- **Bone length consistency**: 68% improvement (0.113 → 0.036 CV)
- **Marker quality**: 59/65 markers (4 unreliable markers auto-filtered)
- **Joint angles**: 12 joint groups computed (pelvis, hip, knee, ankle, trunk, shoulder, elbow)
- **Output**: Clean directory structure with automatic organization

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
Video → MediaPipe → CSV → TRC → Pose2Sim LSTM → Multi-Constraint Optimization → Joint Angles → Clean Output
```

1. **Extraction**: MediaPipe detects 33 landmarks, mapped to 22 Pose2Sim markers
2. **Augmentation**: Pose2Sim LSTM adds 43 augmented markers (medial, shoulder clusters, HJC)
3. **Optimization**: Multi-constraint refinement with unreliable marker filtering
4. **Joint Angles**: ISB-compliant computation for all 12 joint groups
5. **Output**: Organized directory with final TRC, initial TRC, raw CSV, and joint angles

### Output Structure
```
data/output/pose-3d/<video>/
├── <video>_final.trc               # Final optimized skeleton (59-65 markers)
├── <video>_initial.trc             # Initial MediaPipe output (22 markers)
├── <video>_raw_landmarks.csv       # Raw landmark data
└── joint_angles/                   # Joint angle analysis
    ├── <video>_all_joint_angles.png
    └── 12 CSV files (one per joint group)
```

## Requirements

- Python 3.12+
- MediaPipe
- Pose2Sim
- NumPy, SciPy, Matplotlib
- OpenCV (for video I/O)

All dependencies managed via `uv` (see `pyproject.toml`).

## Citation

If you use this pipeline in your research, please cite:
- MediaPipe: [Lugaresi et al. 2019](https://arxiv.org/abs/1906.08172)
- Pose2Sim: [Pagnon et al. 2022](https://joss.theoj.org/papers/10.21105/joss.04362)

## License

See [LICENSE](LICENSE) file for details
