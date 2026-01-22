#!/bin/bash
#
# Safe MTC dataset extraction script
# Extracts the archive sequence-by-sequence to avoid overwhelming WSL
#
# Usage:
#   ./extract_mtc_safe.sh                    # Extract all sequences
#   ./extract_mtc_safe.sh 171026_pose1       # Extract specific sequence
#   ./extract_mtc_safe.sh --list             # List sequences in archive
#   ./extract_mtc_safe.sh --skip-images      # Only extract pkl files (no images)
#

set -e

HUMANPOSE_HOME="${HUMANPOSE3D_HOME:-$HOME/.humanpose3d}"
MTC_DIR="${HUMANPOSE_HOME}/training/mtc"
ARCHIVE="$MTC_DIR/mtc_dataset.tar.gz"
EXTRACT_DIR="$MTC_DIR"

# Known sequences in the dataset (from archive exploration)
SEQUENCES=(
    "171026_pose1"
    "171026_pose2"
    "171026_pose3"
    "171204_pose1"
    "171204_pose2"
    "171204_pose3"
    "171204_pose4"
)

# Parse arguments
SKIP_IMAGES=false
LIST_ONLY=false
SPECIFIC_SEQ=""

for arg in "$@"; do
    case $arg in
        --skip-images)
            SKIP_IMAGES=true
            ;;
        --list)
            LIST_ONLY=true
            ;;
        171*)
            SPECIFIC_SEQ="$arg"
            ;;
    esac
done

# Check archive exists
if [ ! -f "$ARCHIVE" ]; then
    echo "ERROR: Archive not found: $ARCHIVE"
    exit 1
fi

# List mode
if [ "$LIST_ONLY" = true ]; then
    echo "Sequences in archive:"
    for seq in "${SEQUENCES[@]}"; do
        echo "  - $seq"
    done
    echo ""
    echo "Already extracted:"
    for seq in "${SEQUENCES[@]}"; do
        seq_dir="$EXTRACT_DIR/a4_release/hdImgs/$seq"
        if [ -d "$seq_dir" ]; then
            count=$(find "$seq_dir" -maxdepth 1 -type d | wc -l)
            echo "  - $seq: $((count-1)) frames"
        fi
    done
    exit 0
fi

# Function to extract a single sequence
extract_sequence() {
    local seq=$1
    local seq_dir="$EXTRACT_DIR/a4_release/hdImgs/$seq"

    echo ""
    echo "========================================"
    echo "EXTRACTING: $seq"
    echo "========================================"

    # Check if already extracted
    if [ -d "$seq_dir" ]; then
        frame_count=$(find "$seq_dir" -maxdepth 1 -type d 2>/dev/null | wc -l)
        if [ "$frame_count" -gt 100 ]; then
            echo "  Already extracted ($((frame_count-1)) frames), skipping..."
            return 0
        fi
    fi

    # Extract just this sequence
    echo "  Extracting images for $seq..."

    # Use pigz for faster decompression, extract only this sequence
    # The --skip-old-files flag prevents re-extracting existing files
    pigz -dc "$ARCHIVE" 2>/dev/null | tar -xf - \
        --skip-old-files \
        -C "$EXTRACT_DIR" \
        "a4_release/hdImgs/$seq/" 2>/dev/null || true

    # Force filesystem sync to prevent WSL crash
    echo "  Syncing filesystem..."
    sync

    # Brief pause to let WSL catch up
    sleep 2

    # Report progress
    if [ -d "$seq_dir" ]; then
        frame_count=$(find "$seq_dir" -maxdepth 1 -type d 2>/dev/null | wc -l)
        echo "  Done: $((frame_count-1)) frames extracted"
    else
        echo "  WARNING: Directory not created"
    fi
}

# Extract pickle files first (small, essential)
extract_pickles() {
    echo ""
    echo "========================================"
    echo "EXTRACTING PICKLE FILES"
    echo "========================================"

    if [ -f "$EXTRACT_DIR/a4_release/annotation.pkl" ] && [ -f "$EXTRACT_DIR/a4_release/camera_data.pkl" ]; then
        echo "  Pickle files already extracted"
        return 0
    fi

    pigz -dc "$ARCHIVE" 2>/dev/null | tar -xf - \
        --skip-old-files \
        -C "$EXTRACT_DIR" \
        "a4_release/annotation.pkl" \
        "a4_release/camera_data.pkl" \
        "a4_release/README.txt" 2>/dev/null || true

    sync
    echo "  Done"
}

# Main extraction
echo "MTC Dataset Safe Extraction"
echo "Archive: $ARCHIVE"
echo "Output: $EXTRACT_DIR"
echo ""

# Always extract pickle files
extract_pickles

if [ "$SKIP_IMAGES" = true ]; then
    echo ""
    echo "Skipping image extraction (--skip-images)"
    exit 0
fi

# Determine which sequences to extract
if [ -n "$SPECIFIC_SEQ" ]; then
    sequences_to_extract=("$SPECIFIC_SEQ")
else
    sequences_to_extract=("${SEQUENCES[@]}")
fi

# Extract sequences
for seq in "${sequences_to_extract[@]}"; do
    extract_sequence "$seq"

    # Extra sync and pause between sequences
    sync
    sleep 3

    # Check memory pressure (WSL specific)
    free_mem=$(free -m | awk '/^Mem:/{print $4}')
    if [ "$free_mem" -lt 1000 ]; then
        echo ""
        echo "WARNING: Low memory ($free_mem MB free). Pausing 10s..."
        sleep 10
        # Drop caches if we have permissions
        echo 3 | sudo tee /proc/sys/vm/drop_caches 2>/dev/null || true
    fi
done

echo ""
echo "========================================"
echo "EXTRACTION COMPLETE"
echo "========================================"

# Final summary
echo ""
echo "Extracted sequences:"
for seq in "${SEQUENCES[@]}"; do
    seq_dir="$EXTRACT_DIR/a4_release/hdImgs/$seq"
    if [ -d "$seq_dir" ]; then
        count=$(find "$seq_dir" -maxdepth 1 -type d 2>/dev/null | wc -l)
        echo "  - $seq: $((count-1)) frames"
    else
        echo "  - $seq: NOT EXTRACTED"
    fi
done

echo ""
echo "Total disk usage:"
du -sh "$EXTRACT_DIR/a4_release/" 2>/dev/null || true
