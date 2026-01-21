# Model Download Guide

HumanPose3D requires 5 pre-trained model files (~121 MB total) to run. This guide explains how to download them.

## Quick Start

### Option 1: Standalone Script (Recommended)

```bash
# Download all models
python scripts/download_models.py

# Check if models exist
python scripts/download_models.py --check

# Force re-download
python scripts/download_models.py --force
```

### Option 2: Django Management Command

```bash
# Download all models
python manage.py download_models

# Check if models exist
python manage.py download_models --check-only

# Force re-download
python manage.py download_models --force
```

## What Gets Downloaded

All models are downloaded to `~/.humanpose3d/models/`:

| File | Size | Purpose |
|------|------|---------|
| `checkpoints/best_depth_model.pth` | ~31 MB | Neural depth refinement |
| `checkpoints/best_joint_model.pth` | ~11 MB | Joint constraint refinement |
| `checkpoints/best_main_refiner.pth` | ~14 MB | Main refiner fusion model |
| `pose_landmarker_heavy.task` | ~29 MB | MediaPipe pose detection |
| `GRU.h5` | ~37 MB | Pose2Sim LSTM augmentation |
| **Total** | **~121 MB** | |

## Download Location (Platform-Independent)

Models are stored in your home directory:

- **Windows**: `C:\Users\<username>\.humanpose3d\models`
- **macOS**: `/Users/<username>/.humanpose3d\models`
- **Linux**: `/home/<username>/.humanpose3d\models`

## Requirements

- **Git**: Must be installed and in PATH
- **Internet connection**: Models are cloned from GitHub
- **Disk space**: ~200 MB free (models + temporary files)

### Check Git Installation

```bash
git --version
```

If git is not installed:
- **Windows**: Download from https://git-scm.com/download/win
- **macOS**: `brew install git` or `xcode-select --install`
- **Linux**: `sudo apt-get install git` or `sudo yum install git`

## Usage Examples

### First-Time Setup

```bash
# Clone the repository
git clone https://github.com/YourUsername/HumanPose3D.git
cd HumanPose3D

# Download models
python scripts/download_models.py
```

Output:
```
======================================================================
HumanPose3D Model Downloader
======================================================================

Target directory: /home/user/.humanpose3d
Models directory: /home/user/.humanpose3d/models

Starting download...

  Creating temporary directory...
  Cloning models from https://github.com/JoeyKardolus/humanpose3d_backend.git...
  Copying model files...
  Download complete!

✓ Models downloaded successfully

Downloaded files:
  - best_depth_model.pth (31.07 MB)
  - best_joint_model.pth (10.63 MB)
  - best_main_refiner.pth (14.01 MB)
  - pose_landmarker_heavy.task (29.24 MB)
  - GRU.h5 (36.63 MB)

✓ Models are ready! You can now run the pipeline.
```

### Check Model Status

```bash
python scripts/download_models.py --check
```

Output when models are present:
```
======================================================================
HumanPose3D Model Downloader
======================================================================

Target directory: /home/user/.humanpose3d
Models directory: /home/user/.humanpose3d/models

Checking model files...

  ✓ Depth refinement model: best_depth_model.pth (31.07 MB)
  ✓ Joint refinement model: best_joint_model.pth (10.63 MB)
  ✓ Main refiner model: best_main_refiner.pth (14.01 MB)
  ✓ MediaPipe pose model: pose_landmarker_heavy.task (29.24 MB)
  ✓ Pose2Sim LSTM model: GRU.h5 (36.63 MB)

All required models are present.
```

### Re-download Models

If models are corrupted or you want to update them:

```bash
python scripts/download_models.py --force
```

## How It Works

1. **Git Clone**: Clones only the `models` branch from GitHub (shallow clone)
2. **Extract**: Copies model files to `~/.humanpose3d/models/`
3. **Verify**: Checks that all 5 required files exist
4. **Cleanup**: Removes temporary clone directory

The download uses:
- `--branch models`: Only clone the models branch
- `--depth 1`: Shallow clone (faster, smaller)
- `--single-branch`: Don't fetch other branches

## Automatic Download (Future)

The pipeline will automatically prompt to download models if they're missing:

```bash
python manage.py run_pipeline --video test.mp4 --height 1.78 --weight 75
```

Output if models are missing:
```
[main] Error: Models not found in ~/.humanpose3d/models/
[main] Please download models first:
[main]   python scripts/download_models.py
```

## Troubleshooting

### Issue: "Git is not installed or not in PATH"

**Solution**: Install git and ensure it's in your system PATH.

```bash
# Verify git is accessible
git --version

# If not found, add git to PATH or install it
```

### Issue: "Git clone failed: Permission denied"

**Possible causes**:
1. **Firewall blocking git**: Allow git through firewall
2. **Corporate proxy**: Configure git proxy settings
3. **GitHub access**: Ensure you can access github.com

**Solutions**:

```bash
# Test GitHub access
curl -I https://github.com

# Configure proxy (if needed)
git config --global http.proxy http://proxy.example.com:8080
git config --global https.proxy https://proxy.example.com:8080
```

### Issue: "Download timed out"

**Solution**: Check internet connection and try again. The timeout is 5 minutes.

```bash
# Retry download
python scripts/download_models.py --force
```

### Issue: "Models directory not found in repository"

**Cause**: The models branch doesn't have the expected structure.

**Solution**: Report this issue on GitHub. The repository structure may have changed.

### Issue: "File system error during download"

**Possible causes**:
1. **Insufficient disk space**: Free up ~200 MB
2. **Permission denied**: Check write access to `~/.humanpose3d/`
3. **Disk full**: Check available space

**Solutions**:

```bash
# Check disk space
df -h ~

# Check permissions
ls -la ~/.humanpose3d/

# Try with elevated permissions (not recommended)
sudo python scripts/download_models.py  # Last resort
```

### Issue: Models exist but pipeline says they're missing

**Solution**: Verify all 5 files exist and are complete:

```bash
# Check models
python scripts/download_models.py --check

# Force re-download if any are corrupt
python scripts/download_models.py --force
```

## Manual Download (Alternative)

If automated download fails, you can manually download models:

1. Visit: https://github.com/JoeyKardolus/humanpose3d_backend/tree/models
2. Download the `models` directory
3. Place it in `~/.humanpose3d/models/`

```bash
# Create directory
mkdir -p ~/.humanpose3d/models/checkpoints

# Download each file manually (replace URLs as needed)
wget -O ~/.humanpose3d/models/checkpoints/best_depth_model.pth <URL>
wget -O ~/.humanpose3d/models/checkpoints/best_joint_model.pth <URL>
wget -O ~/.humanpose3d/models/checkpoints/best_main_refiner.pth <URL>
wget -O ~/.humanpose3d/models/pose_landmarker_heavy.task <URL>
wget -O ~/.humanpose3d/models/GRU.h5 <URL>

# Verify
python scripts/download_models.py --check
```

## Updating Models

To update to newer model versions:

```bash
# Re-download all models
python scripts/download_models.py --force
```

Old models will be replaced with new versions from GitHub.

## Programmatic Usage

You can use the model download service in your own scripts:

```python
from pathlib import Path
from src.application.config.user_paths import UserPaths
from src.application.services.model_download_service import ModelDownloadService

# Initialize
user_paths = UserPaths.default()
service = ModelDownloadService(user_paths.base)

# Check if models exist
if not user_paths.models_exist():
    print("Downloading models...")

    # Download with progress callback
    def progress(msg):
        print(f"  {msg}")

    success, message = service.download_models(progress)

    if success:
        print("Models downloaded successfully!")
    else:
        print(f"Download failed: {message}")
else:
    print("Models already present.")
```

## Security Considerations

- Models are downloaded from GitHub over HTTPS
- Only official repository is used (hardcoded URL)
- No executable code is downloaded (only model files)
- Files are validated to ensure correct structure

## Storage Management

### Disk Usage

```bash
# Check model directory size
du -sh ~/.humanpose3d/models
```

### Cleanup

To remove downloaded models:

```bash
# Remove all models
rm -rf ~/.humanpose3d/models

# Or remove entire user data
rm -rf ~/.humanpose3d

# Re-download when needed
python scripts/download_models.py
```

## Related Documentation

- [Path Centralization](PATH_CENTRALIZATION.md): Understanding where files are stored
- [Quick Reference](PATHS_QUICK_REFERENCE.md): Path configuration examples

## Support

If you encounter issues not covered here:

1. Check that git is installed and accessible
2. Verify internet connection and GitHub access
3. Try manual download as fallback
4. Report issues on GitHub with error messages
