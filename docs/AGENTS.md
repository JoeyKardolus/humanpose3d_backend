follow## Application Guidelines

### General Principles
- Keep it simple and readable.
- KISS and Single Responsibility Principle apply everywhere.
- Prefer explicit code over clever code.
- Structure always beats speed.
- Many small files are better than a few large ones.
- If something feels convenient, double-check it.

---

## Front-end Guidelines

### Markup
- Use **plain HTML**
- HTML must stay **clean and declarative**
- ❌ No inline JavaScript
- ❌ No inline CSS
- ❌ No `<script>` tags inside HTML files

HTML is for structure only. Nothing else.

---

### Styling
- **Primary framework:** Bootstrap
- **Custom styling:** Plain CSS
- ❌ No Tailwind
- ❌ No inline styles
- ❌ No `<style>` blocks in HTML

CSS rules:
- CSS lives in dedicated `.css` files
- Prefer Bootstrap utilities before adding custom CSS
- Custom CSS must stay minimal and scoped

If CSS grows large, split it into multiple files.

---

### Front-end Logic
- Use **JavaScript (ES6+)**
- ❌ No inline JavaScript
- All front-end logic lives in **`.js` files**
- HTML may only reference bundled or compiled assets

Business logic never belongs in the front-end.

---

## Back-end Guidelines

### Architecture
- Backend code must follow **strict Object-Oriented Programming**
- Use **Django MVT with app-based domain structure**
- Think in **apps as domains**, not layers

Django principles:
- **Models** → data & domain rules (per app)
- **Views** → thin entry points (request in, response out)
- **Templates** → presentation only
- **Services / use_cases** → business logic

Views orchestrate.  
Services decide.  
Apps communicate explicitly.

---

### Django / Python Rules
- Do **not** overload framework files:
  - `views.py`
  - `models.py`
  - `signals.py`

Split logic into dedicated modules, for example:
- `services/`
- `repositories/`
- `use_cases/`
- `validators/`
- `dto/`

If a file grows, it must be split. No exceptions.

---

### Application Boundary (Strict)
- All application logic lives in:
  **`src/application/`**
- When referring to “the application”, this folder is meant
- ❌ No business logic outside `src/application`
- ❌ No side effects leaking into framework or infrastructure layers

Framework = shell  
Application = core

---

## Project Structure & Module Organization
- `src/` holds production code split by responsibility:
  - `mediastream` (I/O)
  - `posedetector` (MediaPipe inference)
  - `datastream` (CSV export)
  - `visualizedata` (3D plotting)
  - helper packages for augmentation and streaming utilities
- `src/pipeline/runner.py` wires modules together; keep it thin and orchestration-only
- `data/input/` and `data/output/` store raw videos and generated artifacts
- `models/` ships with `pose_landmarker_heavy.task`
- Tests live under `tests/` and mirror the structure of `src/`
- The project uses **UV** for running and packaging

---

## Build, Test, and Development Commands
- `uv sync` installs the Python 3.12 toolchain
- `uv run python manage.py run_pipeline --video data/input/<name>.mp4 --height <m> --weight <kg> --main-refiner`
- `uv run pytest`

---

## Pipeline Flags & Visualization
- `--main-refiner` enables neural depth + joint refinement (recommended)
- `--show-video` renders MediaPipe preview and exports preview video
- `--plot-landmarks` replays extracted CSV landmarks
- `--plot-augmented` visualizes Pose2Sim augmented TRC
- `--plot-all-joint-angles` generates comprehensive joint angle visualization

---

## Coding Style & Naming Conventions
- Follow PEP 8
- Use descriptive `snake_case`
- Type-hint all public interfaces
- One responsibility per file
- Add docstrings to modules, classes, and public methods.
- Use inline comments only where logic is non-obvious; keep them short and factual.
- ❌ No wildcard imports
- ❌ No circular dependencies
- Format code before committing:
  `uv run python -m black src tests`

Strict OOP applies.  
Functions do one thing. Files stay small.

---

## Testing Guidelines
- Use `pytest`
- Test files mirror `src/` structure
- Prefer deterministic fixtures
- Mock heavy OpenCV / MediaPipe dependencies
- Document new test data in PRs

---

## Commit & Pull Request Guidelines
- Use imperative, scoped commit messages
- Keep unrelated changes separate
- PRs require:
  - short summary
  - validation notes
  - visuals for visualization changes
- CI must be green before merge

---

## Final Notes
- Avoid framework shortcuts
- Make unclear things explicit
- Readability beats cleverness

Explicit > Implicit  
Structure > Speed
