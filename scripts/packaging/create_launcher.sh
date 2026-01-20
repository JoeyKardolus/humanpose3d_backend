#!/bin/bash
# Create a desktop launcher that opens the app in a terminal
# This ensures users can see the server output and stop it with Ctrl+C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cat > "$REPO_ROOT/HumanPose3D.sh" << 'EOF'
#!/bin/bash
# HumanPose3D Launcher
# Opens the application in a terminal window

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTABLE="$SCRIPT_DIR/bin/HumanPose3D-linux/HumanPose3D-linux"

# Detect available terminal emulator
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal -- bash -c "$EXECUTABLE; echo -e '\nPress Enter to close...'; read"
elif command -v konsole &> /dev/null; then
    konsole -e bash -c "$EXECUTABLE; echo -e '\nPress Enter to close...'; read"
elif command -v xfce4-terminal &> /dev/null; then
    xfce4-terminal -e "bash -c '$EXECUTABLE; echo -e \"\\nPress Enter to close...\"; read'"
elif command -v xterm &> /dev/null; then
    xterm -e bash -c "$EXECUTABLE; echo -e '\nPress Enter to close...'; read"
elif command -v x-terminal-emulator &> /dev/null; then
    x-terminal-emulator -e bash -c "$EXECUTABLE; echo -e '\nPress Enter to close...'; read"
else
    # Fallback: run directly (requires running from terminal)
    echo "No terminal emulator found. Running directly..."
    "$EXECUTABLE"
fi
EOF

chmod +x "$REPO_ROOT/HumanPose3D.sh"

echo "Launcher script created: $REPO_ROOT/HumanPose3D.sh"
echo "You can double-click this script to open HumanPose3D in a terminal."
