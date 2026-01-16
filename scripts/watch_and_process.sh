#!/bin/bash
# Watch for new camera videos and start processing when available

VIDEO_DIR="data/AIST++/videos"
LOG_DIR="/tmp"

# Cameras to monitor
CAMERAS="c05 c06 c07 c08 c09"

# Minimum videos before starting processing
MIN_VIDEOS=100

while true; do
    for cam in $CAMERAS; do
        # Count videos for this camera
        count=$(ls $VIDEO_DIR/*_${cam}_*.mp4 2>/dev/null | wc -l)

        # Check if process is already running
        running=$(ps aux | grep "convert_aistpp_parallel.py $cam" | grep -v grep | wc -l)

        if [ $count -ge $MIN_VIDEOS ] && [ $running -eq 0 ]; then
            echo "[$(date)] Starting $cam processing ($count videos available)"
            nohup uv run python scripts/convert_aistpp_parallel.py $cam > $LOG_DIR/aistpp_${cam}.log 2>&1 &
        fi
    done

    # Check progress every 30 seconds
    sleep 30

    # Show status
    echo "[$(date)] Status:"
    for cam in c02 c03 c04 c05 c06 c07 c08 c09; do
        videos=$(find $VIDEO_DIR -name "*_${cam}_*.mp4" 2>/dev/null | wc -l)
        samples=$(ls data/training/aistpp_converted/ 2>/dev/null | grep "_${cam}_" | wc -l)
        running=$(ps aux | grep "convert_aistpp_parallel.py $cam" | grep python3 | grep -v grep | wc -l)
        status="stopped"
        [ $running -gt 0 ] && status="RUNNING"
        echo "  $cam: $videos videos, $samples samples [$status]"
    done
    echo ""
done
