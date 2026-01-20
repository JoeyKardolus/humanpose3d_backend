# CLAUDE.md - AI Assistant Instructions

Instructions for AI assistants (Claude Code) working on this codebase.

## Project Overview

3D human pose estimation pipeline: MediaPipe detection → Neural depth refinement → Pose2Sim LSTM augmentation → ISB-compliant joint angles → Neural joint constraint refinement.

**Tech stack**: Python 3.12, Django 6, MediaPipe, Pose2Sim, PyTorch, ONNX Runtime

## Architectural Principles

### General Principles
- **KISS and SRP**: Keep it simple. Single Responsibility Principle everywhere.
- **Explicit over clever**: Structure beats speed. Readability beats cleverness.
- **Many small files**: Better than few large ones. Split when files grow.
- **No convenience shortcuts**: If something feels too convenient, double-check it.

### Front-end Guidelines

#### Markup
- Use **plain HTML** - clean and declarative
- ❌ No inline JavaScript
- ❌ No inline CSS (except dynamic values like `style="width: {{ value }}%"`)
- ❌ No `<script>` tags inside HTML files

HTML is for structure only.

#### Styling
- **Primary framework**: Bootstrap
- **Custom styling**: Plain CSS in dedicated `.css` files
- ❌ No Tailwind, no inline styles, no `<style>` blocks

CSS rules:
- CSS lives in dedicated files under `src/application/static/`
- Prefer Bootstrap utilities before adding custom CSS
- Custom CSS must stay minimal and scoped
- Split into multiple files if CSS grows large

#### Front-end Logic
- Use **JavaScript (ES6+)** in dedicated `.js` files
- ❌ No inline JavaScript
- All front-end logic lives in `src/application/static/*.js`
- HTML may only reference bundled or compiled assets

**Business logic never belongs in the front-end.**

### Back-end Guidelines

#### Architecture
- Backend code follows **strict Object-Oriented Programming**
- Use **Django MVT with app-based domain structure**
- Think in **apps as domains**, not layers

Django principles:
- **Models** → data & domain rules (per app)
- **Views/Controllers** → thin entry points (request in, response out)
- **Templates** → presentation only
- **Services / use_cases** → business logic

**Views orchestrate. Services decide. Apps communicate explicitly.**

#### Django / Python Rules
- Do **not** overload framework files (`views.py`, `models.py`, `signals.py`)
- Split logic into dedicated modules:
  - `controllers/` - view functions (thin entry points)
  - `services/` - business logic
  - `use_cases/` - orchestration (coordinates multiple services)
  - `repositories/` - data access
  - `validators/` - validation logic
  - `dto/` - data transfer objects
  - `config/` - configuration

**If a file grows, it must be split. No exceptions.**

#### Application Boundary (Strict)
- All application logic lives in: **`src/application/`**
- When referring to "the application", this folder is meant
- ❌ No business logic outside `src/application/`
- ❌ No side effects leaking into framework or infrastructure layers

**Framework = shell. Application = core.**

## Codebase Structure

```
src/
├── application/          # Django app - web interface (all UI logic here)
│   ├── controllers/      # View functions (thin entry points)
│   ├── services/         # Business logic
│   ├── use_cases/        # Orchestration (coordinates services)
│   ├── repositories/     # Data access
│   ├── validators/       # Validation logic
│   ├── dto/              # Data transfer objects
│   ├── config/           # Configuration
│   ├── static/           # CSS, JS
│   └── templates/        # HTML
├── cli/                  # Management commands
├── mediastream/          # Video I/O (OpenCV)
├── posedetector/         # MediaPipe inference, landmark mapping
├── datastream/           # CSV/TRC conversion, marker estimation
├── markeraugmentation/   # Pose2Sim integration, GPU acceleration
├── kinematics/           # ISB joint angles, Euler decomposition
├── visualizedata/        # 3D plotting, skeleton connections
├── depth_refinement/     # Neural depth correction model
├── joint_refinement/     # Neural joint constraint model
├── main_refinement/      # Fusion model (depth + joint)
├── pipeline/             # Pipeline orchestration
└── postprocessing/       # Output organization
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `mediastream/` | Video I/O (OpenCV) |
| `posedetector/` | MediaPipe inference, landmark mapping |
| `datastream/` | CSV/TRC conversion, marker estimation |
| `markeraugmentation/` | Pose2Sim integration, GPU acceleration |
| `kinematics/` | ISB joint angles, Euler decomposition, visualization |
| `visualizedata/` | 3D plotting, skeleton connections |
| `depth_refinement/` | Neural depth correction model |
| `joint_refinement/` | Neural joint constraint model |
| `main_refinement/` | Fusion model combining depth + joint |
| `pipeline/` | Orchestration (refinement, cleanup) |
| `application/` | Django web interface |

## Development Workflow

### Running the Pipeline

```bash
# CLI (recommended for development)
uv run python manage.py run_pipeline \
  --video data/input/video.mp4 \
  --height 1.78 --weight 75 \
  --main-refiner \
  --estimate-missing --force-complete \
  --augmentation-cycles 20 \
  --plot-all-joint-angles \
  --visibility-min 0.1

# Web interface
uv run python manage.py runserver
# Visit http://127.0.0.1:8000/

# Tests
uv run pytest

# Format code before committing
uv run python -m black src tests

# Build standalone executable (recommended method)
./scripts/packaging/build.sh linux
```

### Building Standalone Executables

The project uses PyInstaller to create standalone executables:

**Build process:**
```bash
uv pip install pyinstaller

# Recommended: Use the build script
./scripts/packaging/build.sh [linux|macos|windows]

# Alternative: Direct PyInstaller (requires all flags)
uv run pyinstaller scripts/packaging/HumanPose3D-<platform>.spec -y --distpath bin --workpath bin/build
```

**Build script features:**
- Automatically adds all required flags (`-y`, `--distpath`, `--workpath`)
- No manual flag management needed
- Clear output messages

**Output paths:**
- Executables: `bin/HumanPose3D-<platform>/`
- Build artifacts: `bin/build/` (temporary files, ignored by git)

**Entry point:** `scripts/packaging/pyinstaller_entry.py`
- Detects frozen (bundled) vs development environment
- Starts Django server on 127.0.0.1:8000 with `--noreload`
- Automatically opens web browser when server is ready
- Shows clear terminal banner with server status and stop instructions
- Supports CLI commands in executable: `./HumanPose3D-linux run_pipeline ...`

**Launcher script** (optional): `scripts/packaging/create_launcher.sh`
- Generates `HumanPose3D.sh` wrapper script
- Opens executable in a terminal window (gnome-terminal, konsole, xfce4-terminal, xterm)
- Useful for double-clicking from file managers
- Keeps terminal open after app exits

**Spec files:** `scripts/packaging/HumanPose3D-<platform>.spec` (commit to version control)
- Bundles: templates, static files, neural models
- Hidden imports: all Django apps and Python modules
- Output: `bin/HumanPose3D-<platform>/`

### Key Pipeline Flags

| Flag | Description | When to use |
|------|-------------|-------------|
| `--main-refiner` | Full neural pipeline (depth + joint) | **Always recommend** for production runs |
| `--estimate-missing` | Mirror occluded limbs from visible side | When limbs are occluded |
| `--force-complete` | Estimate shoulder clusters + hip joint centers | For full 64-marker output |
| `--augmentation-cycles N` | Multi-cycle averaging (default 20) | Higher = more stable but slower |
| `--visibility-min 0.1` | Landmark confidence threshold | Use 0.1 to prevent marker dropout |
| `--plot-all-joint-angles` | Multi-panel joint angle visualization | For analysis and validation |

### Pipeline Flow

1. **MediaPipe extraction** → 33 landmarks → 22 Pose2Sim markers
2. **Neural depth refinement** (`--main-refiner`) → corrects depth errors on 17 COCO joints
3. **TRC conversion** → with derived markers (Hip, Neck)
4. **Pose2Sim augmentation** → GPU-accelerated LSTM → 64 markers (43 added)
5. **Joint angle computation** → 12 ISB-compliant joint groups
6. **Neural joint refinement** (`--main-refiner`) → learned soft constraints
7. **Automatic cleanup** → organized output structure

### Output Structure

```
data/output/pose-3d/<video>/
├── <video>_final.trc           # Optimized 59-64 markers
├── <video>_initial.trc         # Initial 22 markers
├── <video>_raw_landmarks.csv   # Raw MediaPipe data
└── joint_angles/               # 13 files (CSV + PNG per joint)
```

## Technical Reference

### Marker Sets

**Original 22 markers** (from MediaPipe):
Neck, RShoulder, LShoulder, RHip, LHip, RKnee, LKnee, RAnkle, LAnkle, RHeel, LHeel, RSmallToe, LSmallToe, RBigToe, LBigToe, RElbow, LElbow, RWrist, LWrist, Hip, Nose

**Augmented 64 markers** (after Pose2Sim LSTM):
- Original 22 markers
- Lower body (35): ASIS, PSIS, medial knee/ankle, toe markers, thigh clusters, shoulder clusters, HJC
- Upper body (8): elbow/wrist study markers (medial/lateral)

### Joint Angles

All joints use: `{joint}_flex_deg`, `{joint}_abd_deg`, `{joint}_rot_deg`
- **Pelvis**: ZXY Euler (global orientation)
- **Lower body**: Hip, Knee, Ankle (3-DOF each, both sides)
- **Upper body**: Trunk, Shoulder (3-DOF), Elbow (1-DOF)
- **Standard**: ISB-compliant anatomical axes

### Neural Models

**Depth Refinement** (~3M params):
- Transformer with Part Orientation Fields (POF)
- Based on MonocularTotalCapture (CVPR 2019)
- Predicts 14 per-limb 3D unit vectors
- Uses camera direction vector for front/back disambiguation
- Performance: 45% depth improvement, 75% bone variance reduction

**Joint Constraint Refinement** (~916K params):
- Transformer-based soft constraint learning
- Trained on AIST++ motion capture data
- Mean correction: 3.47°, handles errors up to 73°

**MainRefiner** (~1.2M params):
- Fusion model combining depth + joint outputs
- Two-stage: pre-augmentation (depth) + post-augmentation (joint)
- Total inference: <10ms per frame on CPU

### GPU Acceleration

- GPU accelerates **Pose2Sim LSTM only** (3-10x speedup)
- MediaPipe uses CPU (faster with XNNPACK)
- Automatic CPU fallback if GPU unavailable
- Check GPU: `uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"`

## Code Style

### Python Style
- Follow **PEP 8** strictly
- Use **4-space indentation**, **snake_case**
- **Type hints** on all public interfaces
- **One responsibility per file**
- Add **docstrings** to modules, classes, and public methods
- Inline comments only where logic is non-obvious; keep them short and factual
- ❌ No wildcard imports
- ❌ No circular dependencies

### Commit Messages
- Use **imperative, scoped messages**: `module: description`
- Examples:
  - `depth_refinement: Add POF limb orientation prediction`
  - `webapp: Split pipeline_views into controllers module`
  - `kinematics: Fix ankle angle computation for ISB compliance`

### Testing
- Use **pytest**
- Test files mirror `src/` structure
- Prefer **deterministic fixtures**
- **Mock heavy dependencies** (OpenCV, MediaPipe)
- Document new test data in PRs

### Pull Requests
- Keep unrelated changes separate
- PRs require:
  - Short summary (what/why)
  - Validation notes (how tested)
  - Visuals for visualization changes
- CI must be green before merge

## Common Issues

| Issue | Solution |
|-------|----------|
| Right arm missing | Use `--estimate-missing` to mirror from left |
| Depth errors / front-back confusion | Use `--main-refiner` (neural depth correction) |
| Joint angle spikes | Use `--main-refiner` (learned joint constraints) |
| Markers disappear mid-video | Use `--visibility-min 0.1` (MediaPipe confidence threshold) |
| "No trc files found" | Check Pose2Sim project structure |

## Key Insights for Development

1. **Application boundary is strict**: All Django/web logic in `src/application/`, all pipeline logic outside
2. **Services orchestrate modules**: Use services/use_cases for complex workflows
3. **Split files proactively**: Better to have many small focused files than few large ones
4. **Neural refinement is two-stage**: Pre-augmentation (depth on 17 joints) + post-augmentation (joint angles on 64 markers)
5. **GPU is optional**: Pipeline works on any system, GPU only speeds up LSTM
6. **ISB compliance matters**: Joint angles must follow ISB standards for biomechanics compatibility

## Documentation References

- User guide: [README.md](README.md)
- Architecture principles: [docs/AGENTS.md](docs/AGENTS.md)
- System architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Neural models: [docs/NEURAL_MODELS.md](docs/NEURAL_MODELS.md)
- Development log: [docs/BUILD_LOG.md](docs/BUILD_LOG.md)

---

**Remember**: Explicit > Implicit. Structure > Speed. Many small files > Few large files.
