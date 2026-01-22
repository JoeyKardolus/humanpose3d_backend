# System Architecture

## Current Modules

| Module | Purpose |
|--------|---------|
| `mediastream/` | Video I/O (`read_video_rgb`, `probe_video_rotation`) |
| `posedetector/` | MediaPipe inference (33 landmarks → 22 Pose2Sim markers) |
| `datastream/` | CSV/TRC conversion, marker estimation |
| `markeraugmentation/` | Pose2Sim LSTM integration (22 → 64 markers), GPU acceleration |
| `pof/` | Part Orientation Fields (3D reconstruction from 2D) |
| `joint_refinement/` | Neural joint constraints (cross-joint attention) |
| `main_refinement/` | Fusion model (POF + joint gating) |
| `kinematics/` | ISB joint angles (12 joint groups, Euler decomposition) |
| `pipeline/` | Orchestration (`refinement.py`, `cleanup.py`) |
| `visualizedata/` | 3D Matplotlib plotting, skeleton connections |
| `application/webapp/` | Django web interface |

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT                                         │
│  Video (.mp4) + Subject params (height, mass)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1: POSE EXTRACTION (mediastream + posedetector)                   │
│  - OpenCV video capture → RGB frames                                    │
│  - MediaPipe Pose → 33 landmarks (world coords)                         │
│  - POSE_NAME_MAP renames to 22 Pose2Sim-compatible marker names         │
│  Output: CSV (timestamp, landmark, x, y, z, visibility)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 2: POF 3D RECONSTRUCTION (pof) [--main-refiner or --camera-pof]   │
│  - Part Orientation Fields predict 14 per-limb 3D unit vectors          │
│  - Least-squares solver reconstructs 3D joints from limb vectors        │
│  - Bypasses MediaPipe depth errors entirely                             │
│  Output: Reconstructed 17 COCO joints in 3D                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 3: TRC CONVERSION (datastream)                                    │
│  - CSV → TRC format (ORDER_22 marker layout)                            │
│  - Derive synthetic markers (Hip, Neck)                                 │
│  - Pre-augmentation estimation (--estimate-missing)                     │
│  Output: Initial TRC (22 markers)                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 4: MARKER AUGMENTATION (markeraugmentation)                       │
│  - Pose2Sim LSTM (GPU-accelerated via ONNX Runtime)                     │
│  - 20-cycle averaging with 1mm perturbation                             │
│  - Adds 43 anatomical markers (ASIS, PSIS, medial, clusters, HJC)       │
│  - Post-augmentation completion (--force-complete)                      │
│  Output: Augmented TRC (64 markers)                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 5: JOINT ANGLE COMPUTATION (kinematics)                           │
│  - ISB-compliant segment coordinate systems                             │
│  - Euler decomposition (12 joint groups × 3 DOF)                        │
│  - ensure_continuity() prevents 180° axis flips                         │
│  Output: Joint angle CSVs + visualization PNG                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 6: NEURAL JOINT REFINEMENT (joint_refinement) [--main-refiner]    │
│  - Cross-joint attention with kinematic chain bias                      │
│  - Learned soft constraints from AIST++ motion capture                  │
│  - Per-joint delta corrections                                          │
│  Output: Refined joint angles                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 7: OUTPUT ORGANIZATION (pipeline/cleanup)                         │
│  - Rename files: _final.trc, _initial.trc, _raw_landmarks.csv           │
│  - Organize joint angles into subdirectory                              │
│  - Remove intermediate files                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                        │
│  data/output/pose-3d/<video>/                                           │
│  ├── <video>_final.trc           (optimized 59-64 markers)              │
│  ├── <video>_initial.trc         (initial 22 markers)                   │
│  ├── <video>_raw_landmarks.csv   (raw MediaPipe data)                   │
│  └── joint_angles/               (12 CSV + 1 PNG)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Entry Points

| File | Purpose |
|------|---------|
| `main.py` | Full pipeline orchestrator |
| `scripts/train/pof_model.py` | POF model training |
| `scripts/train/joint_model.py` | Joint constraint model training |
| `scripts/train/main_refiner.py` | MainRefiner fusion model training |
| `scripts/data/convert_aistpp.py` | AIST++ training data conversion |
| `scripts/viz/visualize_interactive.py` | Interactive TRC viewer |

## Marker Count Progression

```
MediaPipe: 33 landmarks
    ↓ POSE_NAME_MAP
Pose2Sim input: 22 markers
    ↓ LSTM augmentation (20 cycles)
Augmented: 64 markers (+43 anatomical)
    ↓ Quality filtering
Final: 59-64 markers (unreliable removed)
```

## Neural Models

| Model | Params | Purpose |
|-------|--------|---------|
| POF | ~3M | 3D reconstruction from 2D via Part Orientation Fields |
| Joint Refiner | ~916K | Cross-joint attention → angle corrections |
| MainRefiner | ~1.2M | Fusion gating (POF + joint) |

Training data: AIST++ (1.2M frames) + CMU MTC (~28K frames × 31 cameras)

---

*Last updated: 2026-01-21*
