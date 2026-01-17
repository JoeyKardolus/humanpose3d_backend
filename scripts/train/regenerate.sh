#!/bin/bash
# =============================================================================
# Regenerate Training Data for Depth Refinement and Joint Angle Models
# =============================================================================
#
# This script generates:
#   - ~1.5M depth refinement samples (from AIST++ with real MediaPipe errors)
#   - ~750K joint angle constraint samples (subset with computed joint angles)
#
# Prerequisites:
#   - AIST++ dataset downloaded to data/AIST++/
#   - uv package manager installed
#
# Usage:
#   ./scripts/regenerate_training_data.sh [--depth-only] [--joint-only]
#
# Estimated time: ~24-48 hours total on a decent GPU machine
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

echo "=============================================================="
echo "TRAINING DATA REGENERATION"
echo "=============================================================="
echo "Project root: $PROJECT_ROOT"
echo "Target: 1.5M depth samples, 750K joint angle samples"
echo ""

# Parse arguments
DEPTH_ONLY=false
JOINT_ONLY=false

for arg in "$@"; do
    case $arg in
        --depth-only)
            DEPTH_ONLY=true
            shift
            ;;
        --joint-only)
            JOINT_ONLY=true
            shift
            ;;
    esac
done

# =============================================================================
# PHASE 1: Generate Depth Refinement Training Data (~1.5M samples)
# =============================================================================
if [ "$JOINT_ONLY" = false ]; then
    echo ""
    echo "=============================================================="
    echo "PHASE 1: Depth Refinement Data (target: 1.5M samples)"
    echo "=============================================================="
    echo ""
    echo "Strategy:"
    echo "  - Use 6 camera views (c01-c06) for viewpoint diversity"
    echo "  - Process ~250 frames per video (skip every 2nd frame)"
    echo "  - Scale-normalized for proper depth learning"
    echo ""

    # Create output directory
    mkdir -p data/training/aistpp_converted

    # Run the conversion script
    # The script is already configured with proper parameters
    echo "Starting depth data generation..."
    echo "This will take several hours. Progress will be logged."
    echo ""

    uv run python scripts/data/convert_aistpp.py 2>&1 | tee logs/depth_conversion.log

    # Count samples
    DEPTH_COUNT=$(ls data/training/aistpp_converted/*.npz 2>/dev/null | wc -l)
    echo ""
    echo "Depth samples generated: $DEPTH_COUNT"
    echo ""
fi

# =============================================================================
# PHASE 2: Generate Joint Angle Training Data (~750K samples)
# =============================================================================
if [ "$DEPTH_ONLY" = false ]; then
    echo ""
    echo "=============================================================="
    echo "PHASE 2: Joint Angle Data (target: 750K samples)"
    echo "=============================================================="
    echo ""
    echo "Strategy:"
    echo "  - Process ~50% of depth samples"
    echo "  - Run Pose2Sim augmentation + joint angle computation"
    echo "  - Groups frames by sequence for temporal coherence"
    echo ""

    # Check if depth data exists
    DEPTH_COUNT=$(ls data/training/aistpp_converted/*.npz 2>/dev/null | wc -l || echo "0")
    if [ "$DEPTH_COUNT" -eq 0 ]; then
        echo "ERROR: No depth samples found. Run depth generation first."
        echo "Use: $0 --depth-only"
        exit 1
    fi

    echo "Found $DEPTH_COUNT depth samples"

    # Calculate how many sequences to process for 750K samples
    # Each sequence has ~250 frames, so need ~3000 sequences for 750K
    # But sequences vary in length, so we'll process based on target output

    mkdir -p data/training/aistpp_joint_angles

    echo "Starting joint angle data generation..."
    echo "This will take many hours. GPU acceleration enabled for Pose2Sim."
    echo ""

    # Calculate sequences needed for ~750K samples
    # Each sequence-camera has ~180 frames, so need ~4200 sequence-cameras
    # With ~2-3 cameras per sequence on average, that's ~1500-2000 sequences
    # We'll process ~2500 sequences to be safe and let caching handle duplicates

    # Process sequences - the script handles grouping and caching
    uv run python scripts/data/generate_joint_angles.py \
        --max-sequences 2500 \
        --workers 1 \
        2>&1 | tee logs/joint_angle_conversion.log

    # Count samples
    JOINT_COUNT=$(ls data/training/aistpp_joint_angles/*.npz 2>/dev/null | wc -l)
    echo ""
    echo "Joint angle samples generated: $JOINT_COUNT"
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================================="
echo "GENERATION COMPLETE"
echo "=============================================================="

if [ "$JOINT_ONLY" = false ]; then
    DEPTH_COUNT=$(ls data/training/aistpp_converted/*.npz 2>/dev/null | wc -l || echo "0")
    DEPTH_SIZE=$(du -sh data/training/aistpp_converted 2>/dev/null | cut -f1 || echo "0")
    echo "Depth samples:       $DEPTH_COUNT ($DEPTH_SIZE)"
fi

if [ "$DEPTH_ONLY" = false ]; then
    JOINT_COUNT=$(ls data/training/aistpp_joint_angles/*.npz 2>/dev/null | wc -l || echo "0")
    JOINT_SIZE=$(du -sh data/training/aistpp_joint_angles 2>/dev/null | cut -f1 || echo "0")
    echo "Joint angle samples: $JOINT_COUNT ($JOINT_SIZE)"
fi

echo ""
echo "Training data ready at: data/training/"
echo ""
echo "Next steps:"
echo "  1. Train depth model:  uv run python scripts/train/depth_model.py --epochs 100"
echo "  2. Train joint model:  uv run python scripts/train/joint_model.py --epochs 100"
echo ""
