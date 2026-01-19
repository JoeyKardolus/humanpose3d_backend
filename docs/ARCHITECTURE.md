# System Architecture

## Directory Structure

```
src/
├── application/          # Django web interface
│   ├── webapp/
│   │   ├── controllers/  # View functions (thin entry points)
│   │   ├── services/     # Business logic
│   │   ├── use_cases/    # Orchestration (multi-service workflows)
│   │   ├── repositories/ # Data access layer
│   │   ├── validators/   # Request validation
│   │   ├── dto/          # Data transfer objects
│   │   └── config/       # Configuration
│   ├── static/           # CSS, JavaScript
│   └── templates/        # HTML templates
├── api/                  # REST API endpoints
├── cli/                  # Django management commands
├── mediastream/          # Video I/O (OpenCV)
├── posedetector/         # MediaPipe inference
├── datastream/           # CSV/TRC conversion
├── markeraugmentation/   # Pose2Sim LSTM (GPU-accelerated)
├── depth_refinement/     # Neural depth correction
├── joint_refinement/     # Neural joint constraints
├── main_refinement/      # Fusion model (depth + joint)
├── kinematics/           # ISB joint angle computation
├── pipeline/             # Pipeline orchestration
├── postprocessing/       # Temporal smoothing
└── visualizedata/        # 3D visualization
```

## Core Modules

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `mediastream/` | Video I/O | `read_video_rgb()`, `probe_video_rotation()` |
| `posedetector/` | MediaPipe inference | 33 landmarks → 22 Pose2Sim markers via `POSE_NAME_MAP` |
| `datastream/` | Data conversion | CSV/TRC conversion, marker estimation |
| `markeraugmentation/` | LSTM augmentation | Pose2Sim integration (22 → 64 markers), GPU acceleration |
| `depth_refinement/` | Neural depth model | POF + view angle transformer (~3M params) |
| `joint_refinement/` | Neural joint model | Cross-joint attention (~916K params) |
| `main_refinement/` | Fusion model | Depth + joint gating (~1.2M params) |
| `kinematics/` | Joint angles | ISB-compliant, 12 joint groups, Euler decomposition |
| `pipeline/` | Orchestration | `runner.py`, `refinement.py`, `cleanup.py` |
| `postprocessing/` | Post-processing | Temporal smoothing |
| `visualizedata/` | Visualization | 3D Matplotlib plotting, skeleton connections |

## Application Layer (Django)

Following **strict separation of concerns** per AGENTS.md principles:

| Component | Purpose | Rule |
|-----------|---------|------|
| `application/webapp/controllers/` | HTTP entry points | Thin, request → response only |
| `application/webapp/use_cases/` | Orchestration | Coordinates multiple services |
| `application/webapp/services/` | Business logic | Domain operations |
| `application/webapp/repositories/` | Data access | File I/O, state management |
| `application/webapp/validators/` | Validation | Input validation, path checking |
| `application/webapp/dto/` | Data transfer | Request/response objects |
| `application/templates/` | Presentation | HTML only (no inline JS/CSS) |
| `application/static/` | Assets | Dedicated CSS/JS files |
| `api/` | REST API | JSON endpoints for programmatic access |
| `cli/` | Management commands | `run_pipeline` Django command |

**Architectural rule**: All application logic stays in `src/application/`. Pipeline logic stays outside.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT                                         │
│  Video (.mp4) + Subject params (height, weight)                         │
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
│  Step 2: NEURAL DEPTH REFINEMENT (depth_refinement) [--main-refiner]    │
│  - Part Orientation Fields predict limb 3D vectors                      │
│  - Camera view angle prediction from 2D pose                            │
│  - Transformer corrects MediaPipe depth errors                          │
│  Output: Refined 17 COCO joints                                         │
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

### Main Entry Points

| File | Purpose | Usage |
|------|---------|-------|
| `manage.py` | Django management (recommended) | `uv run python manage.py run_pipeline --video ...` |
| `main.py` | Direct pipeline runner (legacy) | `uv run python main.py --video ...` |

Both entry points run the same pipeline via `src/pipeline/runner.py`.

### Scripts

| Script | Purpose |
|--------|---------|
| **Training Scripts** | |
| `scripts/train/depth_model.py` | Train depth refinement model (POF transformer) |
| `scripts/train/joint_model.py` | Train joint constraint model |
| `scripts/train/main_refiner.py` | Train MainRefiner fusion model |
| **Data Processing** | |
| `scripts/data/convert_aistpp.py` | Convert AIST++ dataset to training format |
| `scripts/data/convert_cmu_mtc.py` | Convert CMU Panoptic MTC dataset |
| **Visualization** | |
| `scripts/viz/visualize_interactive.py` | Interactive 3D TRC viewer |
| `scripts/viz/visualize_depth_comparison.py` | Compare depth refinement results |
| `scripts/viz/visualize_joint_refinement.py` | Visualize joint refinement corrections |

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

| Model | Params | Input | Output | Stage |
|-------|--------|-------|--------|-------|
| **Depth Refiner** | ~3M | 17 COCO joints (3D) + 2D pose | Refined 3D positions | Pre-augmentation |
| **Joint Refiner** | ~916K | 12 joint groups (angles) | Refined angles | Post-augmentation |
| **MainRefiner** | ~1.2M | Combined features | Gated fusion output | Both stages |

**Training data**: AIST++ (1.2M frames, 6 camera views) + CMU Panoptic MTC (~28K frames, 31 cameras)

### Model Architecture Details

**Depth Refiner (POF Transformer)**:
- Part Orientation Fields predict 14 limb 3D unit vectors
- Camera direction vector for front/back disambiguation
- Cross-joint attention for inter-joint depth reasoning
- Performance: 45% depth improvement, 75% bone variance reduction

**Joint Refiner (Cross-joint Attention)**:
- Learns soft biomechanical constraints from motion capture
- Kinematic chain bias (parent-child joint relationships)
- Per-joint delta corrections
- Performance: Mean correction 3.47°, handles errors up to 73°

**MainRefiner (Fusion Model)**:
- Gating mechanism combines depth and joint refinements
- Two-stage pipeline integration
- Per-joint confidence estimation
- Total inference: <10ms per frame on CPU

## Data Flow

```
External Dependencies:
├── OpenCV → Video frames
├── MediaPipe → 33 landmarks
├── Pose2Sim LSTM → Marker augmentation
└── PyTorch/ONNX → Neural inference

Internal Communication:
├── Pipeline modules → Coordinate via file I/O (CSV/TRC)
├── Neural models → Loaded once, applied per-frame
├── Application layer → Calls pipeline via use_cases
└── Services → Pure business logic, no I/O
```

**Design principle**: Modules communicate via **files** (CSV/TRC), not in-memory. This enables:
- Clear module boundaries
- Easy debugging (inspect intermediate files)
- Resumable pipeline (start from any stage)
- Testability (mock file I/O)

## Module Communication Patterns

| From | To | Via | Purpose |
|------|-----|-----|---------|
| `mediastream` | `posedetector` | RGB frames | Video decoding |
| `posedetector` | `datastream` | CSV | Landmark export |
| `datastream` | `markeraugmentation` | TRC | Format conversion |
| `markeraugmentation` | `kinematics` | TRC | Augmented markers |
| `kinematics` | Application | CSV + PNG | Joint angles + plots |
| Application | `pipeline/runner` | CLI args / DTO | Pipeline invocation |

**Web interface flow**:
1. User uploads video → `upload_service`
2. Request validated → `run_request_validator`
3. Pipeline prepared → `prepare_pipeline_run` use case
4. Async execution → `run_pipeline_async` use case → subprocess
5. Progress tracked → `pipeline_progress_tracker` → JSON file
6. Results served → `results_service` → HTTP response

## Design Principles

Following **AGENTS.md** architectural guidelines:

### General Principles
- **KISS and SRP**: Single Responsibility Principle everywhere
- **Explicit over clever**: Structure beats speed, readability beats cleverness
- **Many small files**: Better than few large ones. Split when files grow.
- **No convenience shortcuts**: If something feels too convenient, double-check it.

### Application Layer
- **Strict OOP**: All backend code follows object-oriented design
- **Django MVT**: Models (data), Views (entry points), Templates (presentation)
- **Service layer**: Business logic lives in services, not views
- **Use cases**: Orchestrate multiple services for complex workflows
- **No overloading**: Split `views.py`, `models.py` into dedicated modules

### Front-end
- **Clean HTML**: No inline JavaScript or CSS (except dynamic Bootstrap values)
- **Bootstrap first**: Use Bootstrap utilities before custom CSS
- **Dedicated files**: All JS in `.js` files, all CSS in `.css` files
- **Business logic**: Never in front-end, always in backend services

### Testing Strategy
- **Unit tests**: Test modules in isolation with mocked dependencies
- **Integration tests**: Test pipeline end-to-end on sample videos
- **Fixtures**: Prefer deterministic test data
- **Mock heavy dependencies**: MediaPipe, OpenCV, Pose2Sim
- **Test structure**: Mirrors `src/` directory structure

### Code Style
- **PEP 8**: 4-space indent, snake_case, type hints on public interfaces
- **Docstrings**: Module, class, and public method documentation
- **No wildcard imports**: Explicit imports only
- **No circular dependencies**: Clear module hierarchy
- **Imperative commits**: `module: description` format

## Performance Characteristics

| Operation | Time (joey.mp4, 535 frames) | Bottleneck |
|-----------|----------------------------|------------|
| MediaPipe extraction | ~5s | CPU (single-threaded) |
| Depth refinement | ~3s | Neural inference (CPU) |
| TRC conversion | <1s | I/O |
| LSTM augmentation (GPU) | ~8s | GPU memory bandwidth |
| LSTM augmentation (CPU) | ~50s | CPU compute |
| Joint angle computation | ~2s | NumPy operations |
| Joint refinement | ~2s | Neural inference (CPU) |
| **Total (GPU)** | **~60s** | |
| **Total (CPU)** | **~100s** | |

**Optimization notes**:
- GPU provides 3-10x speedup for LSTM augmentation only
- MediaPipe is CPU-only (faster with XNNPACK than GPU)
- Neural models (<10ms per frame) are negligible overhead
- 20-cycle augmentation averages multiple LSTM runs for stability

---

*Last updated: 2026-01-19*
