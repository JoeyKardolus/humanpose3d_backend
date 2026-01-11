#!/bin/bash
# Quick setup for Human3.6M pre-processed dataset (no approval needed)

set -e

echo "========================================"
echo "Human3.6M Quick Setup"
echo "========================================"
echo ""

# Create data directory
mkdir -p data/human36m
cd data/human36m

echo "Downloading pre-processed Human3.6M poses..."
echo "(This version has 3D poses but no videos)"
echo ""

# Option 1: Download from Stanford (Ashesh Jain's preprocessed version)
wget -c http://www.cs.stanford.edu/people/ashesh/h3.6m.zip

echo ""
echo "Extracting..."
unzip -q h3.6m.zip

echo ""
echo "Cleaning up..."
rm h3.6m.zip

echo ""
echo "========================================"
echo "✓ Human3.6M poses downloaded!"
echo "========================================"
echo ""
echo "Data structure:"
ls -lh

echo ""
echo "Next steps:"
echo "1. This gives you 3D ground truth poses"
echo "2. To get videos, you need to register at: http://vision.imar.ro/human3.6m/"
echo "3. Or use AMASS dataset for immediate synthetic data"
echo ""
echo "Alternative (with videos):"
echo "  pip install human36m-dataset"
echo "  # Then download will happen automatically in Python"
