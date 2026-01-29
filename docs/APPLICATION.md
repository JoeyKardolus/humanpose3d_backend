# Web Application (KinetIQ)

Django web interface for the 3D human pose estimation pipeline. Provides video upload, async processing, real-time progress tracking, and interactive results visualization.

## Quick Start

```bash
# Development server
uv run python manage.py runserver

# Production (with ngrok for external access)
uv run python manage.py runserver 0.0.0.0:8000
ngrok http 8000
```

Access at `http://localhost:8000/` or your ngrok URL.

## Architecture Overview

```
src/application/
├── config/                 # Configuration (paths, settings)
├── controllers/            # Django views (HTTP handlers)
├── dto/                    # Data Transfer Objects
├── repositories/           # In-memory state storage
├── services/               # Business logic (16 services)
├── use_cases/              # Workflow orchestration (3 use cases)
├── validators/             # Input validation
├── static/                 # Frontend assets (JS, CSS, images)
└── templates/              # Django HTML templates
```

**Design Pattern**: Clean Architecture with layered separation:
- **Controllers** (views): HTTP request/response handling only
- **Use Cases**: Orchestrate multiple services for workflows
- **Services**: Pure business logic, no HTTP concerns
- **Repositories**: Thread-safe in-memory state
- **DTOs**: Immutable data transfer between layers

## URL Routes

| Route | View | Purpose |
|-------|------|---------|
| `/` | HomeView | Upload form + results history |
| `/run/` | RunPipelineView | Async pipeline trigger (JSON) |
| `/progress/<run_key>/` | PipelineProgressView | Progress polling (JSON) |
| `/results/<run_key>/` | ResultsView | Results listing page |
| `/results/<run_key>/statistics/` | StatisticsView | Joint angle charts |
| `/results/<run_key>/download/<path>/` | DownloadView | File download |
| `/results/<run_key>/media/<path>/` | MediaView | Video streaming |
| `/results/<run_key>/delete/` | DeleteRunView | Cleanup run |
| `/results/<run_key>/download-all/` | DownloadAllView | ZIP archive |

## Features

### Video Upload & Validation

- Supports MP4, MOV, AVI, MKV formats
- Duration limit: 60 seconds (configurable)
- Video metadata extracted via ffprobe
- Client-side duration warning for long videos

### Async Pipeline Execution

```
Upload → Validate → Prepare → Background Thread → Progress Tracking → Results
```

1. **Preparation**: Validate form, save upload, create directories
2. **Execution**: Spawn daemon thread running CLI pipeline
3. **Progress**: Parse log output → update in-memory status → JSON API
4. **Completion**: Move outputs, notify frontend

### Real-Time Progress Tracking

Frontend polls `/progress/<run_key>/` every 500ms:

```json
{
  "run_key": "video-abc123",
  "progress": 45,
  "stage": "Running marker augmentation",
  "done": false,
  "error": null,
  "results_url": null
}
```

**Pipeline Stages** (detected from log parsing):
1. Extracting landmarks (0-20%)
2. Converting to TRC (20-30%)
3. Running marker augmentation (30-70%)
4. Computing joint angles (70-90%)
5. Finalizing output (90-100%)

### Model Download System

On first launch, required models are downloaded automatically:

- `pose_landmarker_heavy.task` (MediaPipe pose model)
- `MarkerAugmenter_LSTM.onnx` (Pose2Sim LSTM)
- POF checkpoints (if neural options enabled)

**API Endpoints**:
- `GET /api/models/status/` - Check model availability
- `POST /api/models/download/` - Trigger download
- `GET /api/models/download/<job_id>/progress/` - Download progress

### Results Visualization

**Results Page** (`/results/<run_key>/`):
- File listing with sizes
- Download individual files or ZIP archive
- Delete run button

**Note**: Input videos are not saved—only marker data (TRC, CSV, joint angles) is kept. Videos are temporarily stored during processing and automatically deleted after pipeline completion.

**Statistics Page** (`/results/<run_key>/statistics/`):
- Interactive Chart.js time-series plots
- All 12 joint angle groups
- Multi-axis view (flexion, abduction, rotation)
- Marker trajectory visualization

### Dark Mode Support

Automatic dark/light mode based on system preference via `color-scheme` meta tag. Bootstrap 5.3 color modes with custom CSS variables for consistent styling.

## Configuration

### Pipeline Settings (Form)

| Field | Default | Description |
|-------|---------|-------------|
| Height (m) | - | Subject height (required for metric scale) |
| Weight (kg) | - | Subject mass (for biomechanics) |
| Visibility threshold | 0.1 | MediaPipe confidence cutoff (0-1) |
| Augmentation cycles | 20 | Multi-cycle LSTM averaging |
| Joint angle smooth window | 9 | Savitzky-Golay filter size |
| Estimate missing | checked | Mirror occluded limbs |
| Force complete | unchecked | Estimate shoulder/hip markers |
| Compute all angles | checked | Calculate 12 joint groups |
| Plot all angles | checked | Generate PNG visualizations |
| POF 3D reconstruction | unchecked | Experimental: POF instead of MediaPipe depth |
| Joint refinement | unchecked | Experimental: Neural angle correction |

### Storage Paths

Default: `~/.humanpose3d/` (override with `HUMANPOSE3D_HOME` environment variable)

```
~/.humanpose3d/
├── input/          # Uploaded videos (temporary, deleted after processing)
├── output/         # Processing results (marker data only, no videos)
├── models/         # Downloaded model files
│   └── checkpoints/
├── logs/           # Application logs
└── training/       # Training data (optional)
```

**Privacy**: Input videos are never persisted. They are temporarily stored during processing and automatically deleted upon pipeline completion.

### Django Settings

Key settings in `humanpose3d/settings.py`:

```python
# No persistent database (in-memory state only)
DATABASES = {"default": {"ENGINE": "django.db.backends.dummy"}}

# CSRF trusted origins for ngrok
CSRF_TRUSTED_ORIGINS = [
    "https://*.ngrok.io",
    "https://*.ngrok-free.app",
    "http://localhost:*",
    "http://127.0.0.1:*",
]
```

## Services Reference

### Pipeline Services

| Service | Purpose |
|---------|---------|
| `PipelineRunner` | Subprocess execution with streaming I/O |
| `PipelineCommandBuilder` | Form data → CLI arguments |
| `PipelineProgressTracker` | Log parsing → progress updates |
| `PipelineResultService` | Output organization, video metadata |

### Upload & Storage

| Service | Purpose |
|---------|---------|
| `UploadService` | File persistence, duration validation |
| `OutputDirectoryService` | Directory creation/validation |
| `RunIdFactory` | Safe filesystem identifiers |
| `RunKeyService` | Composite key composition |

### Results & Display

| Service | Purpose |
|---------|---------|
| `ResultsService` | Output file enumeration |
| `StatisticsService` | Joint angle data assembly |
| `OutputHistoryService` | Previous run listing |
| `ResultsArchiveService` | ZIP file creation |
| `MediaService` | File path resolution |

### Validation & Security

| Service | Purpose |
|---------|---------|
| `RunRequestValidator` | Form data validation |
| `PathValidator` | Directory traversal prevention |
| `RunCleanupService` | Safe file deletion |

## Frontend

### JavaScript (`main.js`)

- Form submission with async pipeline trigger
- Model download progress modal
- Pipeline progress overlay with polling
- Instructions and privacy consent modals
- Video duration validation (client-side warning)
- Collapse toggle animations

### Statistics (`statistics.js`)

- Chart.js time-series rendering
- Joint angle selector (12 groups)
- Multi-axis visualization
- Responsive canvas sizing

### Styles (`styles.css`)

- Bootstrap 5.3.3 customization
- Dark mode variables
- Card-based panel layout
- Progress overlay styling
- Responsive grid system

## Error Handling

Pipeline errors are parsed into user-friendly messages:

| Error Pattern | User Message |
|---------------|--------------|
| `video is too long` | Video is too long. Please upload a video under 1 minute. |
| `ffprobe not found` | Video processing tools are not installed on the server. |
| `out of memory` | Server ran out of memory. Try a shorter video. |
| `no pose detected` | Could not detect a person in the video. |
| `cuda error` | GPU processing failed. The server will retry with CPU. |
| `permission denied` | Server file permission error. |
| `invalid video` | The video file appears to be invalid or corrupted. |

Technical details (last 20 log lines) available in collapsible section.

## Security

- **CSRF Protection**: Django middleware + template tags
- **Path Validation**: Prevents `../` directory traversal
- **Safe Run IDs**: Sanitized to alphanumeric + dash/underscore
- **File Access Control**: Validates paths within allowed directories
- **CORS Whitelist**: ngrok domains + localhost only
- **Subprocess Isolation**: Pipeline runs with repo_root as cwd

## Development

### Running Tests

```bash
# All tests
uv run pytest tests/

# Application tests only
uv run pytest tests/application/

# With coverage
uv run pytest --cov=src/application tests/application/
```

### Adding a New Service

1. Create service class in `services/`
2. Define interface (type hints, docstrings)
3. Wire up in `controllers/pipeline_views.py` (module-level)
4. Add unit tests in `tests/application/services/`

### Template Structure

Templates use Django template inheritance with Bootstrap 5:

```html
{% load static %}
<!doctype html>
<html lang="en">
<head>
  <meta name="color-scheme" content="light dark">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="{% static 'styles.css' %}">
</head>
<body class="page-body">
  <!-- Modals -->
  <!-- Content -->
</body>
</html>
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CSRF verification failed | Add domain to `CSRF_TRUSTED_ORIGINS` in settings |
| Models not downloading | Check network access, verify `~/.humanpose3d/models/` permissions |
| Progress stuck at 0% | Check subprocess is running, review server logs |
| Video not playing | Ensure correct MIME type, check rotation metadata |
| Results page 404 | Run may have been deleted, check output directory |

## API Integration

REST API available at `/api/` for programmatic access:

```bash
# List runs
curl http://localhost:8000/api/runs/

# Run sync pipeline
curl -X POST -F "video=@input.mp4" -F "height=1.78" \
  http://localhost:8000/api/runs/sync/

# Check progress
curl http://localhost:8000/api/runs/<run_key>/progress/
```

See `src/api/` for full API implementation.

---

*Last updated: 2026-01-29* (videos no longer saved, marker data only)
