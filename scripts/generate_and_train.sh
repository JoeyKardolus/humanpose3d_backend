#!/bin/bash
# Generate joint angle training data, then train model

set -e
cd /home/dupy/projects/humanpose3d_backend
export PATH="$HOME/.local/bin:$PATH"

echo "=== Starting data generation ==="
uv run python scripts/data/generate_joint_angles.py --workers 12

echo ""
echo "=== Data generation complete, starting training ==="
uv run --group neural python scripts/train/joint_model.py \
  --epochs 100 --batch-size 1024 --workers 8 --fp16 \
  --d-model 128 --n-layers 4

echo "=== Training complete ==="
