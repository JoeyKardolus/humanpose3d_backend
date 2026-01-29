#!/bin/bash
cd "$(dirname "$0")/../.."

HUMANPOSE_HOME="${HUMANPOSE3D_HOME:-$HOME/.humanpose3d}"
TRAINING_ROOT="${HUMANPOSE_HOME}/training"

echo "Waiting for data generation to complete..."
while pgrep -f "convert_aistpp_parallel" > /dev/null; do
    sleep 60
    FILES=$(find "${TRAINING_ROOT}/aistpp_converted" -name "*.npz" | wc -l)
    echo "$(date): $FILES files generated, still running..."
done

echo ""
echo "Data generation complete!"
FILES=$(find "${TRAINING_ROOT}/aistpp_converted" -name "*.npz" | wc -l)
echo "Total files: $FILES"
echo ""
echo "Starting training..."

uv run python scripts/train/depth_model.py --epochs 50 --batch-size 64 --fp16
