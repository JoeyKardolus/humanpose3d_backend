#!/bin/bash
# Track file changes as they happen during the session

WORK_LOG="$CLAUDE_PROJECT_DIR/.claude/work-log.tmp"
TIMESTAMP=$(date +%H:%M:%S)

# Extract file path from tool input (JSON)
if command -v jq &> /dev/null; then
    FILE_PATH=$(echo "$1" | jq -r '.file_path // empty' 2>/dev/null)
else
    # Fallback: simple grep for file_path
    FILE_PATH=$(echo "$1" | grep -oP '"file_path"\s*:\s*"\K[^"]+' 2>/dev/null)
fi

if [[ -n "$FILE_PATH" ]]; then
    echo "[$TIMESTAMP] Modified: $FILE_PATH" >> "$WORK_LOG"
fi

exit 0
