# HumanPose3D - Complete System Architecture

## Unified Architecture Overview

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    HUMANPOSE3D COMPLETE SYSTEM                                                    ║
║                                                                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                         INPUT LAYER                                                         │  ║
║  │                                                                                                             │  ║
║  │    Video File (.mp4)              Subject Parameters                                                        │  ║
║  │         │                         ├── height (m)                                                            │  ║
║  │         │                         ├── mass (kg)                                                             │  ║
║  │         │                         ├── age                                                                   │  ║
║  │         │                         └── sex                                                                   │  ║
║  └─────────┼───────────────────────────────┼───────────────────────────────────────────────────────────────────┘  ║
║            │                               │                                                                      ║
║            ▼                               │                                                                      ║
║  ┌─────────────────────────────────────────┼───────────────────────────────────────────────────────────────────┐  ║
║  │                              POSE EXTRACTION LAYER                                                          │  ║
║  │                                                                                                             │  ║
║  │   ┌──────────────────────────────────┐                                                                      │  ║
║  │   │      src/mediastream/            │                                                                      │  ║
║  │   │      MediaStream                 │                                                                      │  ║
║  │   │                                  │                                                                      │  ║
║  │   │   • OpenCV video capture         │                                                                      │  ║
║  │   │   • Frame extraction (RGB)       │                                                                      │  ║
║  │   │   • FPS metadata                 │                                                                      │  ║
║  │   └────────────┬─────────────────────┘                                                                      │  ║
║  │                │ frames                                                                                     │  ║
║  │                ▼                                                                                            │  ║
║  │   ┌──────────────────────────────────┐         ┌─────────────────────────────────────────────────────────┐  │  ║
║  │   │      src/posedetector/           │         │                  OUTPUT                                 │  │  ║
║  │   │      PoseDetector                │         │                                                         │  │  ║
║  │   │                                  │         │   33 MediaPipe landmarks                                │  │  ║
║  │   │   • MediaPipe Pose (CPU/XNNPACK) │────────▶│         │                                               │  │  ║
║  │   │   • 33 landmarks detected        │         │         ▼ POSE_NAME_MAP                                 │  │  ║
║  │   │   • World + image coordinates    │         │   22 Pose2Sim markers (ORDER_22)                        │  │  ║
║  │   │   • Per-landmark visibility      │         │         │                                               │  │  ║
║  │   └──────────────────────────────────┘         │         ├── 3D world coords (x, y, z)                   │  │  ║
║  │                                                │         ├── 2D image coords (u, v) ◄── FOR NEURAL NET   │  │  ║
║  │                                                │         └── visibility scores                           │  │  ║
║  │                                                └─────────────────────────────────────────────────────────┘  │  ║
║  └──────────────────────────────────────────────────────────┬──────────────────────────────────────────────────┘  ║
║                                                             │                                                     ║
║            ┌────────────────────────────────────────────────┼────────────────────────────────────────┐            ║
║            │                                                │                                        │            ║
║            ▼                                                ▼                                        ▼            ║
║  ┌─────────────────────────┐              ┌─────────────────────────────────────────────────────────────────────┐ ║
║  │   src/datastream/       │              │                    NEURAL DEPTH REFINEMENT                         │ ║
║  │   DataStream            │              │                    src/depth_refinement/                           │ ║
║  │                         │              │                    (OPTIONAL - FUTURE)                             │ ║
║  │ • CSV export            │              │  ┌───────────────────────────────────────────────────────────────┐ │ ║
║  │ • TRC conversion        │              │  │                                                               │ │ ║
║  │ • ORDER_22 layout       │              │  │   pose_3d (17,3) ──┬──▶ Joint Encoder ──▶ joint_features     │ │ ║
║  │ • Derived markers:      │              │  │   visibility (17,)─┘              │                          │ │ ║
║  │   - Hip (avg L/R)       │              │  │                                   │                          │ │ ║
║  │   - Neck (avg shoulders)│              │  │   pose_2d (17,2) ──▶ Pose2D Encoder ──▶ viewpoint_features   │ │ ║
║  └───────────┬─────────────┘              │  │                     (15 hand-crafted features)         │     │ │ ║
║              │                            │  │                     • shoulder/hip heights             │     │ │ ║
║              │                            │  │                     • limb length ratios               │     │ │ ║
║              │                            │  │                     • torso aspect ratio               │     │ │ ║
║              ▼                            │  │                     • L/R asymmetry                    │     │ │ ║
║  ┌─────────────────────────┐              │  │                                                        │     │ │ ║
║  │ marker_estimation.py    │              │  │                         ┌──────────────────────────────┘     │ │ ║
║  │ (--estimate-missing)    │              │  │                         │                                    │ │ ║
║  │                         │              │  │                         ▼                                    │ │ ║
║  │ PRE-AUGMENTATION:       │              │  │   ┌─────────────────────────────────────────────────────┐    │ │ ║
║  │ • Mirror L↔R arms       │              │  │   │      DIRECT ANGLE PREDICTOR (ElePose)              │    │ │ ║
║  │ • Extrapolate Head      │              │  │   │                                                     │    │ │ ║
║  │ • Estimate SmallToes    │              │  │   │   pose_2d ──▶ ElePose Backbone (1024-dim)          │    │ │ ║
║  └───────────┬─────────────┘              │  │   │              │                                      │    │ │ ║
║              │                            │  │   │              ▼                                      │    │ │ ║
║              │ 22 markers                 │  │   │   Fusion: [joint + 3D + vis + elepose] → MLP       │    │ │ ║
║              ▼                            │  │   │              │                                      │    │ │ ║
║  ┌─────────────────────────────────────┐  │  │   │              ▼                                      │    │ │ ║
║  │      MARKER AUGMENTATION LAYER      │  │  │   │   [az_sin, az_cos, elevation]                      │    │ │ ║
║  │      src/markeraugmentation/        │  │  │   │              │                                      │    │ │ ║
║  │                                     │  │  │   │              ▼                                      │    │ │ ║
║  │  ┌───────────────────────────────┐  │  │  │   │   azimuth = atan2(sin,cos)  0-360°                 │    │ │ ║
║  │  │     Pose2Sim LSTM             │  │  │  │   │   elevation = tanh(e)×90    -90 to +90°            │    │ │ ║
║  │  │     (GPU accelerated)         │  │  │  │   │                                                     │    │ │ ║
║  │  │                               │  │  │  │   │   NO CAMERA CALIBRATION NEEDED AT INFERENCE!       │    │ │ ║
║  │  │  • 20 cycles (default)        │  │  │  │   └──────────────────────────┬──────────────────────────┘    │ │ ║
║  │  │  • Multi-cycle averaging      │  │  │  │                              │                               │ │ ║
║  │  │  • 1mm Gaussian perturbation  │  │  │  │                              ▼                               │ │ ║
║  │  │                               │  │  │  │   ┌─────────────────────────────────────────────────────┐    │ │ ║
║  │  │  INPUT:  22 markers           │  │  │  │   │           VIEW ANGLE ENCODER                       │    │ │ ║
║  │  │  OUTPUT: 65 markers           │  │  │  │   │                                                     │    │ │ ║
║  │  │                               │  │  │  │   │   Fourier features: sin/cos at multiple freqs      │    │ │ ║
║  │  │  +43 augmented markers:       │  │  │  │   │   azimuth → (sin, cos) × 4 frequencies             │    │ │ ║
║  │  │  • Shoulder clusters (6)      │  │  │  │   │   elevation → (sin, cos) × 4 frequencies           │    │ │ ║
║  │  │  • Thigh clusters (8)         │  │  │  │   │                                                     │    │ │ ║
║  │  │  • Medial joints (8)          │  │  │  │   └──────────────────────────┬──────────────────────────┘    │ │ ║
║  │  │  • Hip joint centers (2)      │  │  │  │                              │ view_encoding                 │ │ ║
║  │  │  • ASIS/PSIS markers (4)      │  │  │  │                              │                               │ │ ║
║  │  │  • Heel markers (2)           │  │  │  │                              ▼                               │ │ ║
║  │  │  • Other anatomical (13)      │  │  │  │   ┌─────────────────────────────────────────────────────┐    │ │ ║
║  │  └───────────────────────────────┘  │  │  │   │           DEPTH REFINEMENT HEAD                    │    │ │ ║
║  │              │                      │  │  │   │                                                     │    │ │ ║
║  │  ┌───────────────────────────────┐  │  │  │   │   joint_features + view_encoding                   │    │ │ ║
║  │  │     gpu_config.py             │  │  │  │   │              │                                      │    │ │ ║
║  │  │                               │  │  │  │   │              ▼                                      │    │ │ ║
║  │  │  • CUDA acceleration patch    │  │  │  │   │   Transformer (4 layers, 4 heads, d=64)            │    │ │ ║
║  │  │  • 3-10x speedup on GPU       │  │  │  │   │   Cross-joint attention:                           │    │ │ ║
║  │  │  • Auto CPU fallback          │  │  │  │   │   "Which joints inform depth of others?"           │    │ │ ║
║  │  └───────────────────────────────┘  │  │  │   │              │                                      │    │ │ ║
║  └─────────────────┬───────────────────┘  │  │   │              ▼                                      │    │ │ ║
║                    │                      │  │   │   depth_delta (17,) per-joint Z correction         │    │ │ ║
║                    │ 65 markers           │  │   │                                                     │    │ │ ║
║                    ▼                      │  │   │   refined_z = original_z + depth_delta             │    │ │ ║
║  ┌─────────────────────────────────────┐  │  │   └─────────────────────────────────────────────────────┘    │ │ ║
║  │ post_augmentation_estimation.py     │  │  │                                                               │ │ ║
║  │ (--force-complete)                  │  │  └───────────────────────────────────────────────────────────────┘ │ ║
║  │                                     │  │                                                                    │ ║
║  │ POST-AUGMENTATION:                  │  │  TRAINING DATA (scripts/convert_aistpp_to_training.py):           │ ║
║  │ • Shoulder clusters (Bell 1990)     │  │  ┌───────────────────────────────────────────────────────────────┐ │ ║
║  │ • Hip Joint Centers (regression)    │  │  │  AIST++ Dataset (10.1M frames, 30 subjects, 9 cameras)       │ │ ║
║  │                                     │  │  │         │                                                     │ │ ║
║  └─────────────────┬───────────────────┘  │  │         ▼                                                     │ │ ║
║                    │                      │  │  ┌─────────────┐    ┌─────────────┐                           │ │ ║
║                    │ 65 markers           │  │  │ Video Frame │    │ AIST++ GT   │                           │ │ ║
║                    ▼                      │  │  │ (1080p 60fps)│    │ keypoints3d │                           │ │ ║
║  ┌─────────────────────────────────────────────┤  └──────┬──────┘    └──────┬──────┘                           │ │ ║
║  │      BIOMECHANICAL OPTIMIZATION LAYER      ││         │                  │                                 │ │ ║
║  │      src/anatomical/                       ││         ▼                  │                                 │ │ ║
║  │                                            ││  ┌─────────────┐           │                                 │ │ ║
║  │  ┌──────────────────────────────────────┐  ││  │  MediaPipe  │           │                                 │ │ ║
║  │  │  multi_constraint_optimization.py    │  ││  │  Pose       │           │                                 │ │ ║
║  │  │  (--multi-constraint-optimization)   │  ││  └──────┬──────┘           │                                 │ │ ║
║  │  │                                      │  ││         │                  │                                 │ │ ║
║  │  │  PHASE 0: Quality Filtering          │  ││         ▼                  ▼                                 │ │ ║
║  │  │  ├── Temporal variance analysis      │  ││  ┌─────────────────────────────────────┐                     │ │ ║
║  │  │  ├── Filter unreliable markers       │  ││  │         TRAINING PAIR (NPZ)        │                     │ │ ║
║  │  │  └── Typically removes 4-6 markers   │  ││  │                                     │                     │ │ ║
║  │  │                                      │  ││  │  corrupted: (17,3)  ← MediaPipe    │                     │ │ ║
║  │  │  PHASE 1: Bone Length Stabilization  │  ││  │  ground_truth: (17,3) ← AIST++    │                     │ │ ║
║  │  │  ├── Median bone length targets      │  ││  │  visibility: (17,)                 │                     │ │ ║
║  │  │  ├── 95% depth-weighted correction   │  ││  │  pose_2d: (17,2)    ← KEY!        │                     │ │ ║
║  │  │  ├── Distance constraints for        │  ││  │  azimuth: 0-360°   ← from camera  │                     │ │ ║
║  │  │  │   augmented markers               │  ││  │  elevation: -90-90° ← from camera │                     │ │ ║
║  │  │  └── Prevents scattered markers      │  ││  │  camera_relative: (3,)             │                     │ │ ║
║  │  │                                      │  ││  └─────────────────────────────────────┘                     │ │ ║
║  │  │  PHASE 2: Ground + Anthropometrics   │  │└─────────────────────────────────────────────────────────────┘ │ ║
║  │  │  ├── Ground plane detection          │  │                                                                │ ║
║  │  │  ├── No foot penetration             │  │  TRAINING (scripts/train_depth_model.py):                     │ ║
║  │  │  ├── Hip width = 0.20 × height       │  │  • Loss = L_depth + λ_bone × L_bone + λ_cam × L_camera        │ ║
║  │  │  └── Heel Z smoothing                │  │  • GPU accelerated (RTX 5080)                                 │ ║
║  │  │                                      │  │  • Camera prediction: 1.4° test error                         │ ║
║  │  │  RESULT: 68% bone length improvement │  │                                                                │ ║
║  │  │          (CV: 0.113 → 0.036)         │  └────────────────────────────────────────────────────────────────┘ ║
║  │  └──────────────────────────────────────┘                                                                     ║
║  │                                            │                                                                   ║
║  │  ┌──────────────────────────────────────┐  │                                                                   ║
║  │  │  bone_length_constraints.py          │  │                                                                   ║
║  │  │  anatomical_constraints.py           │  │                                                                   ║
║  │  │  ground_plane_refinement.py          │  │                                                                   ║
║  │  │  joint_constraints.py                │  │                                                                   ║
║  │  │                                      │  │                                                                   ║
║  │  │  (Used by multi-constraint optimizer │  │                                                                   ║
║  │  │   and joint angle computation)       │  │                                                                   ║
║  │  └──────────────────────────────────────┘  │                                                                   ║
║  └─────────────────┬──────────────────────────┘                                                                   ║
║                    │                                                                                              ║
║                    │ 59-65 markers (optimized)                                                                    ║
║                    ▼                                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                    JOINT ANGLE COMPUTATION LAYER                                           │  ║
║  │                                    src/kinematics/                                                         │  ║
║  │                                                                                                            │  ║
║  │  ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐   │  ║
║  │  │  segment_coordinate_systems.py                                                                     │   │  ║
║  │  │                                                                                                    │   │  ║
║  │  │  ISB-COMPLIANT ANATOMICAL COORDINATE SYSTEMS:                                                      │   │  ║
║  │  │                                                                                                    │   │  ║
║  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │  ║
║  │  │  │   PELVIS    │  │   FEMUR     │  │   TIBIA     │  │   FOOT      │  │   TRUNK     │  ...         │   │  ║
║  │  │  │             │  │             │  │             │  │             │  │             │              │   │  ║
║  │  │  │ ASIS/PSIS   │  │ HJC→Knee    │  │ Knee→Ankle  │  │ Heel→Toe    │  │ Pelvis→C7   │              │   │  ║
║  │  │  │ markers     │  │ epicondyles │  │ malleoli    │  │ 5th meta    │  │ shoulders   │              │   │  ║
║  │  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘              │   │  ║
║  │  │                                                                                                    │   │  ║
║  │  │  Each returns rotation matrix R ∈ SO(3) with ensure_continuity() to prevent 180° axis flips       │   │  ║
║  │  └────────────────────────────────────────────────────────────────────────────────────────────────────┘   │  ║
║  │                                                          │                                                │  ║
║  │                                                          ▼                                                │  ║
║  │  ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐   │  ║
║  │  │  comprehensive_joint_angles.py (--compute-all-joint-angles)                                        │   │  ║
║  │  │                                                                                                    │   │  ║
║  │  │  EULER ANGLE DECOMPOSITION:  R_joint = R_distal.T @ R_proximal                                     │   │  ║
║  │  │                                                                                                    │   │  ║
║  │  │  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐  │   │  ║
║  │  │  │  12 JOINT GROUPS COMPUTED:                                                                   │  │   │  ║
║  │  │  │                                                                                              │  │   │  ║
║  │  │  │  PELVIS ──── Global orientation (ZXY Euler) ──────────────────────────────────────────────▶ │  │   │  ║
║  │  │  │     │                                                                                        │  │   │  ║
║  │  │  │     ├── HIP_R ──── Pelvis→Femur (XYZ) ── flex/abd/rot ──────────────────────────────────▶   │  │   │  ║
║  │  │  │     │     │                                                                                  │  │   │  ║
║  │  │  │     │     └── KNEE_R ──── Femur→Tibia (XYZ) ── flex/abd/rot ────────────────────────────▶   │  │   │  ║
║  │  │  │     │           │                                                                            │  │   │  ║
║  │  │  │     │           └── ANKLE_R ──── Tibia→Foot (XYZ) ── flex/abd/rot ──────────────────────▶   │  │   │  ║
║  │  │  │     │                                                                                        │  │   │  ║
║  │  │  │     ├── HIP_L ──── (same as right side) ────────────────────────────────────────────────▶   │  │   │  ║
║  │  │  │     │     └── KNEE_L ── ANKLE_L                                                              │  │   │  ║
║  │  │  │     │                                                                                        │  │   │  ║
║  │  │  │     └── TRUNK ──── Pelvis→Thorax (ZXY) ── flex/abd/rot ─────────────────────────────────▶   │  │   │  ║
║  │  │  │           │                                                                                  │  │   │  ║
║  │  │  │           ├── SHOULDER_R ──── Trunk→Humerus (XYZ) ── flex/abd/rot ──────────────────────▶   │  │   │  ║
║  │  │  │           │     │                                                                            │  │   │  ║
║  │  │  │           │     └── ELBOW_R ──── Humerus→Forearm ── flexion only ───────────────────────▶   │  │   │  ║
║  │  │  │           │                                                                                  │  │   │  ║
║  │  │  │           └── SHOULDER_L ── ELBOW_L                                                          │  │   │  ║
║  │  │  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │   │  ║
║  │  │                                                                                                    │   │  ║
║  │  │  POST-PROCESSING: unwrap → smooth (Savgol, window=9) → zero to mean → clamp (optional)            │   │  ║
║  │  └────────────────────────────────────────────────────────────────────────────────────────────────────┘   │  ║
║  │                                                          │                                                │  ║
║  │  ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐   │  ║
║  │  │  joint_constraints.py (--check-joint-constraints) [from src/anatomical/]                           │   │  ║
║  │  │                                                                                                    │   │  ║
║  │  │  ANATOMICAL JOINT LIMITS:                                                                          │   │  ║
║  │  │  • Hip: flex [-30,130]°, abd [-30,50]°, rot [-45,45]°                                              │   │  ║
║  │  │  • Knee: flex [0,160]°, abd [-15,15]°, rot [-40,40]°                                               │   │  ║
║  │  │  • Ankle: flex [-45,30]°, abd [-30,30]°, rot [-30,30]°                                             │   │  ║
║  │  │  • Shoulder: flex [-60,180]°, abd [-30,180]°, rot [-90,90]°                                        │   │  ║
║  │  │  • Elbow: flex [0,150]°, abd [-10,10]°, rot [-90,90]°                                              │   │  ║
║  │  │                                                                                                    │   │  ║
║  │  │  FUNCTIONS: check_angle_violations() → soft_clamp_angles() → print_violation_summary()            │   │  ║
║  │  └────────────────────────────────────────────────────────────────────────────────────────────────────┘   │  ║
║  │                                                          │                                                │  ║
║  │  ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐   │  ║
║  │  │  visualize_comprehensive_angles.py (--plot-all-joint-angles)                                       │   │  ║
║  │  │                                                                                                    │   │  ║
║  │  │  Multi-panel time series visualization for all 12 joint groups                                     │   │  ║
║  │  └────────────────────────────────────────────────────────────────────────────────────────────────────┘   │  ║
║  └────────────────────────────────────────────────────────────┬───────────────────────────────────────────────┘  ║
║                                                               │                                                   ║
║                                                               ▼                                                   ║
║  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  ║
║  │                                          OUTPUT LAYER                                                      │  ║
║  │                                                                                                            │  ║
║  │   data/output/pose-3d/<video>/                                                                             │  ║
║  │   │                                                                                                        │  ║
║  │   ├── <video>_final.trc ◄─────────────── Optimized 59-65 marker skeleton                                  │  ║
║  │   ├── <video>_initial.trc ◄───────────── Original 22 MediaPipe markers                                    │  ║
║  │   ├── <video>_raw_landmarks.csv ◄─────── Raw landmark data with visibility                                │  ║
║  │   │                                                                                                        │  ║
║  │   └── joint_angles/                                                                                        │  ║
║  │       ├── <video>_all_joint_angles.png ◄─ Multi-panel visualization                                       │  ║
║  │       ├── <video>_angles_pelvis.csv                                                                        │  ║
║  │       ├── <video>_angles_hip_R.csv        All CSV files have columns:                                     │  ║
║  │       ├── <video>_angles_hip_L.csv        • {joint}_flex_deg                                              │  ║
║  │       ├── <video>_angles_knee_R.csv       • {joint}_abd_deg                                               │  ║
║  │       ├── <video>_angles_knee_L.csv       • {joint}_rot_deg                                               │  ║
║  │       ├── <video>_angles_ankle_R.csv                                                                       │  ║
║  │       ├── <video>_angles_ankle_L.csv                                                                       │  ║
║  │       ├── <video>_angles_trunk.csv                                                                         │  ║
║  │       ├── <video>_angles_shoulder_R.csv                                                                    │  ║
║  │       ├── <video>_angles_shoulder_L.csv                                                                    │  ║
║  │       ├── <video>_angles_elbow_R.csv                                                                       │  ║
║  │       └── <video>_angles_elbow_L.csv                                                                       │  ║
║  │                                                                                                            │  ║
║  │   src/visualizedata/visualize_data.py ─── 3D Matplotlib skeleton viewer                                   │  ║
║  │   visualize_interactive.py ───────────── Interactive TRC viewer with playback                             │  ║
║  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


LEGEND:
═══════
  ──▶   Data flow
  │     Containment/hierarchy
  ◄──   Output file

MODULES:
════════
  src/mediastream/      Video I/O (OpenCV)
  src/posedetector/     MediaPipe inference (33→22 landmarks)
  src/datastream/       CSV/TRC conversion, marker estimation
  src/markeraugmentation/ Pose2Sim LSTM (22→65 markers), GPU acceleration
  src/anatomical/       Biomechanical constraints, multi-constraint optimization, joint angle limits
  src/kinematics/       ISB joint angle computation (12 joints)
  src/depth_refinement/ Neural depth correction (transformer + camera prediction)
  src/visualizedata/    3D visualization

ENTRY POINTS:
═════════════
  main.py               Full pipeline orchestrator (7 steps)
  visualize_interactive.py  Interactive TRC viewer
  scripts/train_depth_model.py  Neural model training
  scripts/convert_aistpp_to_training.py  Training data generation
```

## Quick Reference

### Pipeline Steps (main.py)

| Step | Module | Input | Output | Flag |
|------|--------|-------|--------|------|
| 1 | posedetector | video.mp4 | 33 landmarks | (always) |
| 2 | datastream | landmarks | TRC (22 markers) | (always) |
| 3 | markeraugmentation | TRC | TRC (65 markers) | (always) |
| 3.5 | post_augmentation | TRC | TRC + HJC/shoulders | --force-complete |
| 4 | multi_constraint | TRC | TRC (59-65 markers) | --multi-constraint-optimization |
| 5 | kinematics | TRC | CSV + PNG | --compute-all-joint-angles |
| 5.5 | joint_constraints | angles | violations + clamped | --check-joint-constraints |
| 6 | (cleanup) | files | organized output | (always) |

### Marker Count Progression

```
MediaPipe: 33 landmarks
    ↓ POSE_NAME_MAP
Pose2Sim input: 22 markers
    ↓ LSTM augmentation (20 cycles)
Augmented: 65 markers (+43 anatomical)
    ↓ Quality filtering
Final: 59-65 markers (unreliable removed)
```

### Neural Depth Refinement Status

- **Training data**: 1.2M frames from AIST++ (30 subjects, 9 cameras)
- **Model**: 7.3M parameters (ElePose + Transformer)
  - ElePose backbone: 1024-dim 2D foreshortening features
  - DirectAnglePredictor: azimuth/elevation without body-frame mismatch
  - CrossJointAttention: 6 layers, 8 heads, d_model=128
- **Camera prediction**: 7.1° azimuth, 4.4° elevation error
- **Depth error**: 11.2 cm (45% improvement over raw MediaPipe)
- **Integration**: `--neural-depth-refinement --depth-model-path PATH`

---

*Last updated: 2026-01-16*
