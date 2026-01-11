#!/bin/bash
# Quick setup for Human3.6M pre-processed dataset

set -e

echo "========================================"
echo "Human3.6M Quick Setup"
echo "========================================"
echo ""

# Create data directory
mkdir -p data/human36m
cd data/human36m

echo "Downloading Human3.6M pre-processed data..."
echo ""

# Try alternative sources
echo "Attempting download from GitHub mirror..."

# Option 1: Try VideoPose3D preprocessed data
wget -c https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_gt.npz -O h36m_2d_gt.npz 2>&1 || true

if [ -f "h36m_2d_gt.npz" ]; then
    echo "✓ Downloaded 2D ground truth data"
else
    echo "❌ Download failed"
fi

# Option 2: Try downloading 3D positions
wget -c https://dl.fbaipublicfiles.com/video-pose-3d/data_3d_h36m.npz -O h36m_3d.npz 2>&1 || true

if [ -f "h36m_3d.npz" ]; then
    echo "✓ Downloaded 3D pose data"
else
    echo "❌ Download failed"
fi

echo ""
echo "========================================"
echo "Downloaded files:"
ls -lh *.npz 2>/dev/null || echo "No files downloaded"
echo "========================================"
echo ""
echo "If downloads failed, try manual download:"
echo "1. Go to: https://github.com/facebookresearch/VideoPose3D"
echo "2. Download data_3d_h36m.npz from their releases"
echo "3. Place in data/human36m/"
