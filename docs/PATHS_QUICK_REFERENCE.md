# Path Configuration - Quick Reference

## Where Files Are Stored

### User Data (All Platforms)
```
~/.humanpose3d/
├── models/                          # Neural network models
│   ├── checkpoints/
│   │   ├── best_depth_model.pth
│   │   ├── best_joint_model.pth
│   │   └── best_main_refiner.pth
│   ├── pose_landmarker_heavy.task
│   └── GRU.h5
└── data/
    ├── input/                       # User uploads
    └── output/                      # Pipeline results
        └── <video_name>/
```

**Platform-specific paths**:
- Windows: `C:\Users\<username>\.humanpose3d`
- macOS: `/Users/<username>/.humanpose3d`
- Linux: `/home/<username>/.humanpose3d`

## Quick Code Examples

### Get User Data Paths
```python
from src.application.config.user_paths import UserPaths

user_paths = UserPaths.default()

# Access paths
models_dir = user_paths.models
input_dir = user_paths.data_input
output_dir = user_paths.data_output
```

### Get Application Paths
```python
from src.application.config.paths import AppPaths

app_paths = AppPaths.default()

# Application root (for code, not data)
repo_root = app_paths.repo_root

# Data directories (same as UserPaths)
output_dir = app_paths.output_root
```

### Access Bundled Resources (PyInstaller)
```python
from src.application.config.resource_paths import get_resource_path

# Get path to bundled file (works in dev and PyInstaller)
config_file = get_resource_path("config/settings.json")
```

### Check If Running as PyInstaller Bundle
```python
from src.application.config.resource_paths import is_frozen

if is_frozen():
    print("Running as bundled application")
else:
    print("Running from source")
```

## Common Operations

### Create User Data Directories
```python
from src.application.config.user_paths import UserPaths

user_paths = UserPaths.default()
user_paths.ensure_directories()  # Creates all required directories
```

### Check If Models Exist
```python
from src.application.config.user_paths import UserPaths

user_paths = UserPaths.default()
if user_paths.models_exist():
    print("All models are available")
else:
    print("Need to download models")
```

### Build Output Path for Video
```python
from pathlib import Path
from src.application.config.user_paths import UserPaths

video_path = Path("input_video.mp4")
user_paths = UserPaths.default()

# Output directory for this video
output_dir = user_paths.data_output / video_path.stem
output_dir.mkdir(parents=True, exist_ok=True)

# Output files
trc_file = output_dir / f"{video_path.stem}_final.trc"
csv_file = output_dir / f"{video_path.stem}_raw_landmarks.csv"
```

## Platform-Independent Path Operations

### Always Use pathlib.Path
```python
from pathlib import Path

# ✅ GOOD: Platform-independent
path = Path.home() / ".humanpose3d" / "data" / "output"

# ❌ BAD: Platform-specific
path = "/home/user/.humanpose3d/data/output"  # Unix only
path = "C:\\Users\\user\\.humanpose3d\\data\\output"  # Windows only
```

### Join Paths
```python
from pathlib import Path

# ✅ GOOD: Use / operator
base = Path.home() / ".humanpose3d"
models = base / "models"
checkpoint = models / "checkpoints" / "model.pth"

# ❌ BAD: String concatenation
path = str(base) + "/models/checkpoints/model.pth"
```

### Convert Between Path and String
```python
from pathlib import Path

# Path to string
path = Path.home() / ".humanpose3d"
path_str = str(path)  # For subprocess, file operations

# String to Path
path_str = "/home/user/.humanpose3d"
path = Path(path_str)
```

### Check Path Properties
```python
from pathlib import Path

path = Path.home() / ".humanpose3d" / "models" / "model.pth"

# Check existence
if path.exists():
    print("File exists")

# Check if directory
if path.is_dir():
    print("Is directory")

# Get parent, name, stem, suffix
parent = path.parent  # Path.home() / ".humanpose3d" / "models"
name = path.name      # "model.pth"
stem = path.stem      # "model"
suffix = path.suffix  # ".pth"
```

## Configuration Patterns

### Service That Needs Paths
```python
from pathlib import Path
from src.application.config.user_paths import UserPaths

class MyService:
    def __init__(self):
        self.user_paths = UserPaths.default()

    def process_video(self, video_path: Path) -> Path:
        # Input validation
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Create output directory
        output_dir = self.user_paths.data_output / "runs" / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process...
        result_file = output_dir / f"{video_path.stem}_result.trc"

        return result_file
```

### Django View That Needs Paths
```python
from django.views import View
from src.application.config.paths import AppPaths

class MyView(View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app_paths = AppPaths.default()

    def get(self, request):
        # Access output directory
        output_dir = self.app_paths.output_root
        # ... rest of view logic
```

### CLI Command That Needs Paths
```python
import argparse
from pathlib import Path
from src.application.config.user_paths import UserPaths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()

    user_paths = UserPaths.default()
    user_paths.ensure_directories()

    # Video path from user input
    video_path = args.video.resolve()  # Make absolute

    # Output path in user data directory
    output_dir = user_paths.data_output / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
```

## Migration Checklist

When updating code to use new path system:

- [ ] Import `UserPaths` or `AppPaths` instead of hardcoded paths
- [ ] Use `pathlib.Path` instead of strings
- [ ] Use `/` operator for path joining
- [ ] Use `Path.home()` for home directory
- [ ] Convert to string only for subprocess/external tools
- [ ] Make paths absolute with `.resolve()` when needed
- [ ] Create directories with `mkdir(parents=True, exist_ok=True)`

## Debugging Path Issues

### Print All Paths
```python
from src.application.config.user_paths import UserPaths
from src.application.config.paths import AppPaths

user_paths = UserPaths.default()
app_paths = AppPaths.default()

print("UserPaths:")
print(f"  base: {user_paths.base}")
print(f"  models: {user_paths.models}")
print(f"  data_input: {user_paths.data_input}")
print(f"  data_output: {user_paths.data_output}")

print("\nAppPaths:")
print(f"  repo_root: {app_paths.repo_root}")
print(f"  output_root: {app_paths.output_root}")
print(f"  upload_root: {app_paths.upload_root}")
```

### Verify Platform Independence
```python
import sys
from pathlib import Path

print(f"Platform: {sys.platform}")
print(f"Home directory: {Path.home()}")
print(f"Executable: {sys.executable}")
print(f"Frozen: {getattr(sys, 'frozen', False)}")
```

## Resources

- Full documentation: [PATH_CENTRALIZATION.md](PATH_CENTRALIZATION.md)
- PyInstaller guide: [PYINSTALLER_BUNDLING.md](PYINSTALLER_BUNDLING.md)
- pathlib documentation: https://docs.python.org/3/library/pathlib.html
