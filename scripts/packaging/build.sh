#!/bin/bash
# Simple build wrapper that adds the -y flag automatically
# Usage: ./scripts/packaging/build.sh [platform]
# Platform: linux (default), macos, or windows

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Detect platform if not specified
PLATFORM="${1:-linux}"

case "$PLATFORM" in
    linux)
        SPEC_FILE="HumanPose3D-linux.spec"
        ;;
    macos)
        SPEC_FILE="HumanPose3D-macos.spec"
        ;;
    windows)
        SPEC_FILE="HumanPose3D-windows.spec"
        ;;
    *)
        echo "Error: Unknown platform '$PLATFORM'"
        echo "Usage: $0 [linux|macos|windows]"
        exit 1
        ;;
esac

cd "$REPO_ROOT"

echo "Building HumanPose3D for $PLATFORM..."
echo "Using spec file: scripts/packaging/$SPEC_FILE"
echo "Output: bin/HumanPose3D-$PLATFORM/"
echo "Build artifacts: bin/build/"
echo ""

# Run PyInstaller with:
# -y: auto-confirm overwrites
# --distpath bin: output to bin/ directory
# --workpath bin/build: build artifacts to bin/build/
uv run pyinstaller "scripts/packaging/$SPEC_FILE" -y --distpath bin --workpath bin/build

echo ""
echo "Build complete! Executable: bin/HumanPose3D-$PLATFORM/"
