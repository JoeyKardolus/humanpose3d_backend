# PyInstaller Bundling Guide

This guide explains how to create standalone executables of HumanPose3D for Windows, macOS, and Linux using PyInstaller.

## Architecture Overview

### Path Management (Platform-Independent)

All file paths in HumanPose3D are platform-independent and follow this architecture:

#### User Data Directory: `~/.humanpose3d`

All user data (models, input videos, output results) is stored in the user's home directory:

- **Windows**: `C:\Users\<username>\.humanpose3d`
- **macOS**: `/Users/<username>/.humanpose3d`
- **Linux**: `/home/<username>/.humanpose3d`

**Structure:**
```
~/.humanpose3d/
├── models/                          # Downloaded model files
│   ├── checkpoints/
│   │   ├── best_depth_model.pth
│   │   ├── best_joint_model.pth
│   │   └── best_main_refiner.pth
│   ├── pose_landmarker_heavy.task
│   └── GRU.h5
└── data/
    ├── input/                       # User-uploaded videos
    └── output/                      # Pipeline results
        └── <video_name>/
```

#### Application Code Directory

The application code location is determined automatically:
- **PyInstaller bundle**: Directory containing the executable
- **Development mode**: Repository root

**Key modules:**
- `src/application/config/user_paths.py`: User data paths (~/.humanpose3d)
- `src/application/config/paths.py`: Application root detection
- `src/application/config/resource_paths.py`: PyInstaller resource resolution

### Benefits of This Architecture

1. **Platform Independence**: Uses `pathlib.Path` and `Path.home()` - no hardcoded separators
2. **PyInstaller Compatible**: Application updates don't affect user data
3. **No Admin Rights**: No write permissions needed in application directory
4. **Portable**: User data persists across application reinstalls
5. **Clean Uninstall**: User can delete `~/.humanpose3d` to remove all data

## Prerequisites

```bash
# Install PyInstaller
pip install pyinstaller

# Or with uv (recommended)
uv pip install pyinstaller
```

## Building the Executable

### Option 1: Quick Build (Recommended for Testing)

```bash
# Build for current platform
pyinstaller --name HumanPose3D \
    --onedir \
    --console \
    manage.py

# Output will be in dist/HumanPose3D/
```

### Option 2: Using the Spec File (Recommended for Production)

```bash
# Copy the template
cp humanpose3d.spec.template humanpose3d.spec

# Edit the spec file if needed (entry point, hidden imports, etc.)
nano humanpose3d.spec

# Build
pyinstaller humanpose3d.spec

# Output will be in dist/HumanPose3D/
```

### Platform-Specific Builds

#### Windows

```bash
# Single directory (faster startup)
pyinstaller --name HumanPose3D --onedir --console manage.py

# Single executable (slower startup, easier distribution)
pyinstaller --name HumanPose3D --onefile --console manage.py

# With icon
pyinstaller --name HumanPose3D --onefile --console --icon=icon.ico manage.py
```

Output: `dist/HumanPose3D.exe` or `dist/HumanPose3D/HumanPose3D.exe`

#### macOS

```bash
# App bundle
pyinstaller --name HumanPose3D --windowed manage.py

# With icon (.icns required)
pyinstaller --name HumanPose3D --windowed --icon=icon.icns manage.py
```

Output: `dist/HumanPose3D.app`

#### Linux

```bash
# Standard executable
pyinstaller --name HumanPose3D --onedir manage.py

# The executable will work on similar Linux distributions
```

Output: `dist/HumanPose3D/HumanPose3D`

## Advanced Configuration

### Hidden Imports

If PyInstaller misses some imports, add them to the spec file:

```python
hiddenimports = [
    'django',
    'mediapipe',
    'torch',
    'cv2',
    'Pose2Sim',
    # Add more as needed
]
```

### Excluding Modules (Reduce Size)

```python
excludes = [
    'pytest',
    'unittest',
    'test',
    'tkinter',  # If not using GUI
]
```

### Data Files (Usually Not Needed)

Since models are downloaded to `~/.humanpose3d`, you typically don't need to bundle data files. If needed:

```python
datas = [
    ('path/to/config.json', '.'),
]
```

## First Run After Bundling

When users run the bundled application for the first time:

1. Application detects missing models in `~/.humanpose3d/models/`
2. User is prompted to download models (requires git)
3. Models are downloaded to `~/.humanpose3d/models/`
4. Application is ready to use

## Testing the Bundle

### 1. Test on Current Platform

```bash
# Run the bundled application
./dist/HumanPose3D/HumanPose3D

# Or on Windows
dist\HumanPose3D\HumanPose3D.exe

# Check paths are correct
./dist/HumanPose3D/HumanPose3D --help
```

### 2. Test Model Download

```bash
# Remove models directory to test download
rm -rf ~/.humanpose3d/models

# Run application and download models
./dist/HumanPose3D/HumanPose3D
```

### 3. Test Pipeline

```bash
# Place a test video in ~/.humanpose3d/data/input/
cp test_video.mp4 ~/.humanpose3d/data/input/

# Run pipeline
./dist/HumanPose3D/HumanPose3D run_pipeline \
    --video ~/.humanpose3d/data/input/test_video.mp4 \
    --height 1.78 --weight 75
```

## Common Issues

### Issue: "No module named 'X'"

**Solution**: Add the module to `hiddenimports` in the spec file.

```python
hiddenimports = ['missing_module']
```

### Issue: "Models not found"

**Solution**: Ensure `~/.humanpose3d/models/` exists and contains all required files:
- `checkpoints/best_depth_model.pth`
- `checkpoints/best_joint_model.pth`
- `checkpoints/best_main_refiner.pth`
- `pose_landmarker_heavy.task`
- `GRU.h5`

### Issue: "Permission denied" on macOS

**Solution**: Sign the application or allow in System Preferences > Security & Privacy.

```bash
# Remove quarantine attribute
xattr -cr dist/HumanPose3D.app
```

### Issue: Large executable size

**Solutions**:
1. Use `--onedir` instead of `--onefile` (faster, similar size)
2. Exclude unnecessary modules in spec file
3. Use UPX compression (enabled by default)

### Issue: Slow startup with --onefile

**Solution**: Use `--onedir` instead. The startup time is much faster because PyInstaller doesn't need to extract everything to a temp directory.

## Distribution

### Windows

**Recommended**: Create an installer with Inno Setup or NSIS
- Bundle: `dist/HumanPose3D/`
- Installer creates shortcut, adds to PATH
- Uninstaller included

### macOS

**Recommended**: Create a .dmg disk image

```bash
# Install create-dmg
brew install create-dmg

# Create DMG
create-dmg \
    --volname "HumanPose3D" \
    --window-pos 200 120 \
    --window-size 600 400 \
    dist/HumanPose3D.dmg \
    dist/HumanPose3D.app
```

### Linux

**Recommended**: Create a .deb or .rpm package, or distribute as tarball

```bash
# Create tarball
cd dist
tar -czf HumanPose3D-linux-x64.tar.gz HumanPose3D/

# Or create AppImage (more complex but universal)
```

## Automated Build Script

Create `build.py` to automate builds:

```python
#!/usr/bin/env python3
"""Build script for HumanPose3D."""

import sys
import subprocess
from pathlib import Path

def build():
    """Build executable for current platform."""
    spec_file = Path("humanpose3d.spec")

    if not spec_file.exists():
        print("Error: humanpose3d.spec not found")
        return 1

    cmd = ["pyinstaller", "--clean", str(spec_file)]

    print(f"Building for {sys.platform}...")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\nBuild successful!")
        print(f"Output: dist/HumanPose3D")
    else:
        print("\nBuild failed!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(build())
```

## Troubleshooting

### Enable Debug Mode

Edit spec file and set `debug=True`:

```python
exe = EXE(
    # ...
    debug=True,
    # ...
)
```

### Check Import Errors

```bash
# Run with Python to see full traceback
python -m PyInstaller.utils.run_tests --log-level DEBUG
```

### Verify All Dependencies

```bash
# List all imports
pipdeptree

# Check for missing modules
python -c "import modulename"
```

## Best Practices

1. **Test Early**: Build executables frequently during development
2. **Version Control**: Keep spec file in git
3. **Clean Builds**: Use `--clean` flag to avoid stale files
4. **Test Downloads**: Verify model download works in bundled version
5. **Cross-Platform**: Test on all target platforms
6. **Size Optimization**: Only bundle what's needed
7. **User Data**: Never bundle user data - always use `~/.humanpose3d`

## Resources

- [PyInstaller Documentation](https://pyinstaller.org/)
- [PyInstaller Spec Files](https://pyinstaller.org/en/stable/spec-files.html)
- [Common Issues](https://github.com/pyinstaller/pyinstaller/wiki/Common-Issues)
