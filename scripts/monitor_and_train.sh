#!/bin/bash
# Monitor data generation and start training at 1M samples

TARGET=1000000
HUMANPOSE_HOME="${HUMANPOSE3D_HOME:-$HOME/.humanpose3d}"
DATA_DIR="${HUMANPOSE_HOME}/training/aistpp_converted"
LOG_FILE="${HUMANPOSE_HOME}/logs/monitor_training.log"

mkdir -p "${HUMANPOSE_HOME}/logs"

echo "$(date): Starting monitor - waiting for ${TARGET} samples..." | tee -a $LOG_FILE

while true; do
    COUNT=$(find "$DATA_DIR" -type f -name "*.npz" 2>/dev/null | wc -l)
    echo "$(date): Current sample count: $COUNT / $TARGET" | tee -a $LOG_FILE

    if [ "$COUNT" -ge "$TARGET" ]; then
        echo "$(date): Reached $TARGET samples! Validating files..." | tee -a $LOG_FILE

        # Wait for data generation to finish (check if process is still running)
        while pgrep -f "convert_aistpp" > /dev/null; do
            echo "$(date): Waiting for data generation to complete..." | tee -a $LOG_FILE
            sleep 60
        done

        echo "$(date): Data generation complete. Removing any corrupted files..." | tee -a $LOG_FILE

        # Remove corrupted files (check for 'corrupted' key which dataset expects)
        python3 -c "
import numpy as np
from pathlib import Path
import os

data_dir = Path('$DATA_DIR')
files = list(data_dir.glob('*.npz'))
print(f'Validating {len(files)} files...')

removed = 0
for f in files:
    try:
        with np.load(f) as data:
            _ = data['corrupted']  # Key the dataset actually uses
    except:
        os.remove(f)
        removed += 1

print(f'Removed {removed} corrupted files')
"

        FINAL_COUNT=$(find "$DATA_DIR" -type f -name "*.npz" 2>/dev/null | wc -l)
        echo "$(date): Final sample count: $FINAL_COUNT. Starting training..." | tee -a $LOG_FILE

        # Start training with recommended settings from CLAUDE.md
        uv run --group neural python scripts/train/depth_model.py \
            --epochs 50 --batch-size 256 --workers 8 --bf16 \
            --use-limb-orientations --limb-orientation-weight 0.5 \
            2>&1 | tee -a $LOG_FILE

        echo "$(date): Training completed!" | tee -a $LOG_FILE
        exit 0
    fi

    # Check every 5 minutes
    sleep 300
done
