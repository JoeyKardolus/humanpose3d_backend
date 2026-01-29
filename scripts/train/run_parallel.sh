#!/bin/bash
# Run AIST++ conversion for incomplete cameras in parallel (batches of 4)
# Each camera runs in its own process, with automatic skip for already-processed sequences

cd "$(dirname "$0")/../.."

echo "============================================="
echo "AIST++ Parallel Conversion - Resumable"
echo "============================================="
echo "Skipping c02 (already complete)"
echo "Running 4 cameras at a time"
echo ""

# Log file
HUMANPOSE_HOME="${HUMANPOSE3D_HOME:-$HOME/.humanpose3d}"
LOG_DIR="${HUMANPOSE_HOME}/logs"
mkdir -p "$LOG_DIR"

# Cameras that need processing (c02 is complete)
INCOMPLETE_CAMS=(c01 c03 c04 c05 c06 c07 c08 c09)

# Batch size
BATCH=4

echo "=== Batch 1: c01 c03 c04 c05 ==="
for cam in c01 c03 c04 c05; do
    echo "Starting $cam (log: $LOG_DIR/${cam}.log)"
    uv run python scripts/data/convert_aistpp_parallel.py $cam > "$LOG_DIR/${cam}.log" 2>&1 &
done

echo ""
echo "Batch 1 running. Monitor with: tail -f ${LOG_DIR}/c01.log"
echo "Waiting for batch 1 to complete..."
wait

echo ""
echo "=== Batch 2: c06 c07 c08 c09 ==="
for cam in c06 c07 c08 c09; do
    echo "Starting $cam (log: $LOG_DIR/${cam}.log)"
    uv run python scripts/data/convert_aistpp_parallel.py $cam > "$LOG_DIR/${cam}.log" 2>&1 &
done

echo ""
echo "Batch 2 running. Monitor with: tail -f ${LOG_DIR}/c06.log"
echo "Waiting for batch 2 to complete..."
wait

echo ""
echo "============================================="
echo "ALL CAMERAS COMPLETE!"
echo "============================================="

# Final count
echo ""
echo "Final sample counts:"
for cam in c01 c02 c03 c04 c05 c06 c07 c08 c09; do
    count=$(find "${HUMANPOSE_HOME}/training/aistpp_converted" -name "*_${cam}_f*.npz" 2>/dev/null | wc -l)
    echo "  $cam: $count"
done
