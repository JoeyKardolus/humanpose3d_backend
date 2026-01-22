#!/bin/bash
# =============================================================================
# Auto Training Monitor
# Monitors sample generation and automatically starts training when targets hit
# =============================================================================

cd "$(dirname "$0")/../.."

DEPTH_TARGET=1500000      # 1.5M samples
JOINT_TARGET=750000       # 750K samples
CHECK_INTERVAL=60         # Check every 60 seconds

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

HUMANPOSE_HOME="${HUMANPOSE3D_HOME:-$HOME/.humanpose3d}"
LOG_DIR="${HUMANPOSE_HOME}/logs"
TRAINING_ROOT="${HUMANPOSE_HOME}/training"

count_samples() {
    find "$1" -maxdepth 1 -name "*.npz" 2>/dev/null | wc -l
}

mkdir -p "$LOG_DIR"

# =============================================================================
# PHASE 1: Monitor depth sample generation
# =============================================================================
log "=== AUTO TRAINING MONITOR STARTED ==="
log "Depth target: $DEPTH_TARGET samples"
log "Joint angle target: $JOINT_TARGET samples"
log ""

log "Phase 1: Monitoring depth sample generation..."

while true; do
    DEPTH_COUNT=$(count_samples "${TRAINING_ROOT}/aistpp_converted")
    log "Depth samples: $DEPTH_COUNT / $DEPTH_TARGET"
    
    if [ "$DEPTH_COUNT" -ge "$DEPTH_TARGET" ]; then
        log "DEPTH TARGET REACHED!"
        break
    fi
    
    sleep $CHECK_INTERVAL
done

# Kill conversion processes
log "Stopping conversion processes..."
pkill -f "convert_aistpp" || true
sleep 5

# =============================================================================
# PHASE 2: Train depth model
# =============================================================================
log ""
log "Phase 2: Starting depth model training..."

uv run python scripts/train/depth_model.py \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    2>&1 | tee "$LOG_DIR/depth_training.log"

log "Depth training complete!"

# =============================================================================
# PHASE 3: Generate joint angle samples
# =============================================================================
log ""
log "Phase 3: Starting joint angle generation..."

# Run joint angle generation in background
nohup uv run python scripts/data/generate_joint_angles.py \
    --max-sequences 3000 \
    --workers 1 \
    > "$LOG_DIR/joint_angle_generation.log" 2>&1 &

JOINT_PID=$!
log "Joint angle generation started (PID: $JOINT_PID)"

# Monitor joint angle generation
while true; do
    JOINT_COUNT=$(count_samples "${TRAINING_ROOT}/aistpp_joint_angles")
    log "Joint angle samples: $JOINT_COUNT / $JOINT_TARGET"
    
    if [ "$JOINT_COUNT" -ge "$JOINT_TARGET" ]; then
        log "JOINT ANGLE TARGET REACHED!"
        break
    fi
    
    # Check if process is still running
    if ! kill -0 $JOINT_PID 2>/dev/null; then
        log "Joint angle generation process ended"
        JOINT_COUNT=$(count_samples "${TRAINING_ROOT}/aistpp_joint_angles")
        log "Final joint angle count: $JOINT_COUNT"
        break
    fi
    
    sleep $CHECK_INTERVAL
done

# Kill joint angle generation if still running
kill $JOINT_PID 2>/dev/null || true

# =============================================================================
# PHASE 4: Train joint angle model
# =============================================================================
log ""
log "Phase 4: Starting joint angle model training..."

uv run python scripts/train/joint_model.py \
    --epochs 100 \
    --batch-size 256 \
    2>&1 | tee "$LOG_DIR/joint_training.log"

log "Joint angle training complete!"

# =============================================================================
# Done!
# =============================================================================
log ""
log "=== ALL TRAINING COMPLETE ==="
log "Depth samples: $(ls "${TRAINING_ROOT}/aistpp_converted"/*.npz | wc -l)"
log "Joint samples: $(ls "${TRAINING_ROOT}/aistpp_joint_angles"/*.npz | wc -l)"
log "Models saved in: ~/.humanpose3d/models/checkpoints/"
ls -la ~/.humanpose3d/models/checkpoints/

# Send notification (optional - beep)
echo -e '\a'
