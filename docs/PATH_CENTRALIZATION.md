# Path Centralization and Platform Independence

This document describes the path centralization changes made to HumanPose3D for platform independence.

## Overview

All file paths have been centralized to use the user's home directory (`~/.humanpose3d`) with full platform independence. This ensures the application works identically on Windows, macOS, and Linux.

## Changes Made

### 1. Core Path Configuration

#### `src/application/config/user_paths.py`
**Purpose**: Central configuration for all user data paths

**Key Features**:
- Uses `Path.home()` for platform-independent home directory resolution
- All paths use `pathlib.Path` (no string concatenation)
- Provides `ensure_directories()` to create directory structure
- Provides `models_exist()` to check for required model files

**Paths**:
```python
~/.humanpose3d/
├── models/                          # Downloaded model files
│   ├── checkpoints/
│   │   ├── best_depth_model.pth     # Depth refinement model
│   │   ├── best_joint_model.pth     # Joint constraint model
│   │   └── best_main_refiner.pth    # Main refiner model
│   ├── pose_landmarker_heavy.task   # MediaPipe model
│   └── GRU.h5                       # Pose2Sim LSTM model
└── data/
    ├── input/                       # User-uploaded videos
    └── output/                      # Pipeline results
        └── <video_name>/
```

**Platform-Specific Resolution**:
- **Windows**: `C:\Users\<username>\.humanpose3d`
- **macOS**: `/Users/<username>/.humanpose3d`
- **Linux**: `/home/<username>/.humanpose3d`

#### `src/application/config/paths.py`
**Purpose**: Application root detection and integration with UserPaths

**Changes**:
- Added `get_application_root()` for application root detection
- Updated `AppPaths` to use `UserPaths` for all data directories
- Added `models_root` and `models_checkpoints` to `AppPaths`
- Maintains `repo_root` only for application code (not data)

### 2. Module Updates

#### `src/markeraugmentation/markeraugmentation.py`
**Changes**:
- Updated `_resolve_pose2sim_command()` for platform-independent virtualenv paths
- Handles Windows (`Scripts/python.exe`) and Unix (`bin/python`) correctly
- Uses `sys.platform` check for proper path resolution

**Before**:
```python
local_py = repo_root / ".venv" / "bin" / "python"  # Unix only
```

**After**:
```python
if sys.platform == "win32":
    local_py = venv_dir / "Scripts" / "python.exe"
else:
    local_py = venv_dir / "bin" / "python"
```

#### `src/application/services/model_download_service.py`
**Changes**:
- Added `_check_git_available()` for platform-independent git detection
- Downloads models directly to `UserPaths.default().models`
- Improved error handling for git operations

### 3. Path Usage Verification

**Audit Results**:
- ✅ No hardcoded path separators (`/` or `\` in strings)
- ✅ No `os.path` usage (all use `pathlib.Path`)
- ✅ All data paths use `UserPaths`
- ✅ All path operations are platform-independent

## Benefits

### 1. Platform Independence
- **Works on Windows, macOS, Linux** without code changes
- Uses `pathlib.Path` for all operations
- No hardcoded separators or platform-specific assumptions

### 2. User Experience
- **No admin rights required**: Everything in user's home directory
- **Clean uninstall**: Delete `~/.humanpose3d` to remove all data
- **Portable configuration**: Same location on all platforms

### 3. Development Experience
- **Clear separation**: Code vs data
- **Easy testing**: Just delete `~/.humanpose3d` to reset
- **Version control friendly**: User data never in repository

## File Locations Reference

### Application Files (Read-Only)
Location depends on installation method:
- **Development**: Repository root

Content:
- Python source code (`src/`)
- Django templates and static files
- Tests

### User Data Files (Read-Write)
Location: `~/.humanpose3d` (all platforms)

Content:
- Downloaded model files
- Input videos
- Pipeline output (TRC files, joint angles, plots)
- Temporary files

## Migration Notes

### For Existing Users

Old location: `<repo_root>/data/`
New location: `~/.humanpose3d/data/`

**Migration steps**:
1. Models are automatically downloaded to new location
2. Old data in repo can be manually copied if needed:
   ```bash
   # Copy old data to new location (optional)
   cp -r data/input ~/.humanpose3d/data/
   cp -r data/output ~/.humanpose3d/data/
   ```

### For Developers

**Before**:
```python
from src.application.config.paths import AppPaths
app_paths = AppPaths.from_anchor(Path(__file__))
output_dir = app_paths.output_root  # Was repo_root/data/output
```

**After**:
```python
from src.application.config.user_paths import UserPaths
user_paths = UserPaths.default()
output_dir = user_paths.data_output  # Now ~/.humanpose3d/data/output
```

Or use `AppPaths` which now integrates `UserPaths`:
```python
from src.application.config.paths import AppPaths
app_paths = AppPaths.default()
output_dir = app_paths.output_root  # Now ~/.humanpose3d/data/output
```

## Testing

### Manual Testing
```bash
# Test path resolution
python -c "
from src.application.config.user_paths import UserPaths
from src.application.config.paths import AppPaths

user_paths = UserPaths.default()
app_paths = AppPaths.default()

print('User data:', user_paths.base)
print('Application root:', app_paths.repo_root)
print('Output directory:', app_paths.output_root)
"
```

### Automated Testing
```bash
# Run path configuration tests
pytest tests/test_platform_paths.py -v
```

### Cross-Platform Testing
Test on multiple platforms:
1. **Windows**: WSL, native Windows, or VM
2. **macOS**: Intel or Apple Silicon
3. **Linux**: Various distributions

## Troubleshooting

### Issue: "Models not found"
**Solution**: Run model download or check `~/.humanpose3d/models/`

### Issue: "Permission denied"
**Solution**: Ensure user has write access to home directory (should always work)

### Issue: Different paths on Windows
**Solution**: This is expected. Use `Path.home()` in code, never hardcode paths.

## Future Improvements

### Potential Enhancements
1. **Configuration file**: Store user preferences in `~/.humanpose3d/config.json`
2. **Cache management**: Automatic cleanup of old output files
3. **Custom paths**: Allow users to override default locations via environment variables
4. **Logging**: Centralized logs in `~/.humanpose3d/logs/`

### Environment Variable Support (Future)
```bash
# Optional: Override default location
export HUMANPOSE3D_HOME="/custom/path"
```

## Summary

All file paths in HumanPose3D now:
- ✅ Use `~/.humanpose3d` for all user data
- ✅ Are platform-independent (Windows, macOS, Linux)
- ✅ Use `pathlib.Path` exclusively
- ✅ Separate code from data
- ✅ Require no admin rights

The application is now portable and user-friendly on all major platforms.
