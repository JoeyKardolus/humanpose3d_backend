#!/usr/bin/env python3
"""Interactive visualization for camera-space TRC files (from POF reconstruction).

Camera-space coordinates have Y-down convention (like images).
This script flips Y for proper 3D visualization with Y-up.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# COCO-17 skeleton connections for visualization
SKELETON_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
]

# OpenCap marker connections (subset that maps to COCO-17)
OPENCAP_CONNECTIONS = [
    ('Nose', 'Neck'),
    ('Neck', 'LShoulder'), ('Neck', 'RShoulder'),
    ('LShoulder', 'RShoulder'),
    ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    ('LShoulder', 'LHip'), ('RShoulder', 'RHip'),
    ('LHip', 'RHip'),
    ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
    ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
]


def load_trc(filepath: Path) -> tuple:
    """Load TRC file and return marker names and frame data."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse header
    marker_line = lines[3].strip().split('\t')
    marker_names = []
    for i in range(2, len(marker_line), 3):
        if marker_line[i]:
            marker_names.append(marker_line[i])

    n_markers = len(marker_names)

    # Parse data
    frames = []
    for line in lines[6:]:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue

        # Initialize frame with NaN for all markers
        frame_data = [[np.nan, np.nan, np.nan] for _ in range(n_markers)]

        for m_idx in range(n_markers):
            i = 2 + m_idx * 3
            if i + 2 < len(parts):
                try:
                    x = float(parts[i]) if parts[i] else np.nan
                    y = float(parts[i + 1]) if parts[i + 1] else np.nan
                    z = float(parts[i + 2]) if parts[i + 2] else np.nan
                    frame_data[m_idx] = [x, y, z]
                except ValueError:
                    pass  # Keep NaN

        frames.append(frame_data)

    return marker_names, np.array(frames, dtype=np.float32)


def visualize_camera_space(trc_path: Path):
    """Visualize TRC file in camera-space coordinates."""
    marker_names, frames = load_trc(trc_path)

    print(f"Loaded {len(marker_names)} markers, {len(frames)} frames")
    print(f"Markers: {marker_names[:10]}...")

    # Create marker name to index mapping
    name_to_idx = {name: i for i, name in enumerate(marker_names)}

    # Convert to camera-space display coordinates
    # Camera space: X-right, Y-down, Z-forward
    # Display: X-right, Y-up (flip), Z-forward
    display_frames = frames.copy()
    display_frames[:, :, 1] = -frames[:, :, 1]  # Flip Y for Y-up display

    # Compute bounds
    valid_mask = ~np.isnan(display_frames)
    if valid_mask.any():
        x_min, x_max = np.nanmin(display_frames[:, :, 0]), np.nanmax(display_frames[:, :, 0])
        y_min, y_max = np.nanmin(display_frames[:, :, 1]), np.nanmax(display_frames[:, :, 1])
        z_min, z_max = np.nanmin(display_frames[:, :, 2]), np.nanmax(display_frames[:, :, 2])

        # Add padding
        pad = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)

        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2

        x_min, x_max = x_mid - max_range / 2 - pad, x_mid + max_range / 2 + pad
        y_min, y_max = y_mid - max_range / 2 - pad, y_mid + max_range / 2 + pad
        z_min, z_max = z_mid - max_range / 2 - pad, z_mid + max_range / 2 + pad
    else:
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
        z_min, z_max = 0, 3

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2)

    # Initial view for camera-space (looking from camera position)
    ax.view_init(elev=20, azim=-60)

    # Animation state
    state = {'playing': False, 'frame_idx': 0}

    def update_plot(frame_idx):
        ax.clear()
        frame = display_frames[frame_idx]

        # Plot markers
        valid = ~np.isnan(frame[:, 0])
        ax.scatter(frame[valid, 0], frame[valid, 2], frame[valid, 1],
                   c='blue', marker='o', s=30)

        # Plot skeleton connections
        for name_a, name_b in OPENCAP_CONNECTIONS:
            if name_a in name_to_idx and name_b in name_to_idx:
                idx_a = name_to_idx[name_a]
                idx_b = name_to_idx[name_b]
                if idx_a < len(frame) and idx_b < len(frame):
                    if not np.isnan(frame[idx_a]).any() and not np.isnan(frame[idx_b]).any():
                        ax.plot([frame[idx_a, 0], frame[idx_b, 0]],
                                [frame[idx_a, 2], frame[idx_b, 2]],
                                [frame[idx_a, 1], frame[idx_b, 1]],
                                'r-', linewidth=2)

        ax.set_xlabel('X (right)')
        ax.set_ylabel('Z (depth)')
        ax.set_zlabel('Y (up)')
        ax.set_title(f'Camera-Space View - Frame {frame_idx + 1}/{len(frames)}')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_zlim(y_min, y_max)

        fig.canvas.draw_idle()

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(frames) - 1, valinit=0, valstep=1)

    def on_slider(val):
        state['frame_idx'] = int(val)
        update_plot(state['frame_idx'])

    slider.on_changed(on_slider)

    # Play/Pause button
    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
    button = Button(ax_button, 'Play')

    def on_button(event):
        state['playing'] = not state['playing']
        button.label.set_text('Pause' if state['playing'] else 'Play')

    button.on_clicked(on_button)

    # Animation timer
    def animate(event):
        if state['playing']:
            state['frame_idx'] = (state['frame_idx'] + 1) % len(frames)
            slider.set_val(state['frame_idx'])

    timer = fig.canvas.new_timer(interval=40)  # ~25 fps
    timer.add_callback(animate, None)
    timer.start()

    # Initial plot
    update_plot(0)

    plt.show()


def main():
    if len(sys.argv) > 1:
        trc_path = Path(sys.argv[1])
    else:
        trc_path = Path('data/output/PushUp/PushUp_LSTM.trc')

    if not trc_path.exists():
        print(f"Error: TRC file not found: {trc_path}")
        sys.exit(1)

    print(f"Loading camera-space TRC: {trc_path}")
    visualize_camera_space(trc_path)


if __name__ == "__main__":
    main()
