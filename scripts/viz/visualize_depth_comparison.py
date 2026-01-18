#!/usr/bin/env python3
"""
Interactive 3D viewer comparing original MediaPipe poses vs depth-refined poses.

Usage:
    uv run python visualize_depth_comparison.py [video_name]

Controls:
    - Mouse drag: Rotate 3D view
    - Slider: Navigate through frames
    - Play/Pause: Animate playback
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.datastream.data_stream import LandmarkRecord
from src.depth_refinement.inference import DepthRefiner

# Skeleton connections using landmark names
SKELETON_CONNECTIONS = [
    ('Nose', 'RShoulder'), ('Nose', 'LShoulder'),
    ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    ('RShoulder', 'LShoulder'),
    ('RShoulder', 'RHip'), ('LShoulder', 'LHip'),
    ('RHip', 'LHip'),
    ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
    ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
    ('RAnkle', 'RHeel'), ('RAnkle', 'RBigToe'),
    ('LAnkle', 'LHeel'), ('LAnkle', 'LBigToe'),
]

# Mapping for depth refinement (COCO 17 format)
OPENCAP_TO_COCO = {
    'Nose': 0, 'LShoulder': 5, 'RShoulder': 6, 'LElbow': 7, 'RElbow': 8,
    'LWrist': 9, 'RWrist': 10, 'LHip': 11, 'RHip': 12, 'LKnee': 13,
    'RKnee': 14, 'LAnkle': 15, 'RAnkle': 16,
}


def load_landmarks_from_csv(csv_path: Path) -> tuple[list[float], dict]:
    """Load landmarks from CSV file."""
    frames_data = defaultdict(dict)

    with open(csv_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                ts = float(parts[0])
                name = parts[1]
                x, y, z, vis = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                frames_data[ts][name] = {'x': x, 'y': y, 'z': z, 'vis': vis}

    timestamps = sorted(frames_data.keys())
    return timestamps, frames_data


def apply_depth_refinement(timestamps, frames_data, model_path):
    """Apply neural depth refinement and return refined frames.

    IMPORTANT: The model was trained on pelvis-centered, Y-up data (AIST++ convention).
    MediaPipe outputs Y-down (image coordinates). We must transform coordinates to
    match training convention before inference, then transform back.
    """
    n_frames = len(timestamps)

    # Build COCO format arrays
    pose_3d = np.zeros((n_frames, 17, 3), dtype=np.float32)
    visibility = np.zeros((n_frames, 17), dtype=np.float32)
    pose_2d = np.zeros((n_frames, 17, 2), dtype=np.float32)

    for fi, ts in enumerate(timestamps):
        landmarks = frames_data[ts]
        for name, coco_idx in OPENCAP_TO_COCO.items():
            if name in landmarks:
                lm = landmarks[name]
                pose_3d[fi, coco_idx] = [lm['x'], lm['y'], lm['z']]
                visibility[fi, coco_idx] = lm['vis']
                pose_2d[fi, coco_idx] = [lm['x'], lm['y']]

    # === TRANSFORM TO TRAINING CONVENTION ===
    # Training data was: pelvis-centered, Y-up, Z-away from camera
    # MediaPipe outputs: pelvis-centered (by our pipeline), Y-down, Z-toward camera

    # 1. Compute pelvis position per frame (midpoint of hips)
    # COCO: left_hip=11, right_hip=12
    pelvis = (pose_3d[:, 11, :] + pose_3d[:, 12, :]) / 2  # (n_frames, 3)

    # 2. Center 3D pose on pelvis
    pose_centered = pose_3d - pelvis[:, np.newaxis, :]  # (n_frames, 17, 3)

    # 3. Flip Y and Z to match training convention
    # Y: down -> up (negate)
    # Z: toward camera -> away (negate)
    pose_centered[:, :, 1] = -pose_centered[:, :, 1]
    pose_centered[:, :, 2] = -pose_centered[:, :, 2]

    # NOTE: pose_2d should be RAW MediaPipe coordinates (not centered, not flipped)
    # Training used raw 2D pose for camera angle prediction

    # Load model and refine
    print("Loading depth refinement model...")
    refiner = DepthRefiner(model_path)
    print("Applying depth refinement...")
    refined_centered = refiner.refine_sequence(pose_centered, visibility, pose_2d)

    # === TRANSFORM BACK TO ORIGINAL CONVENTION ===
    # 1. Flip Y and Z back
    refined_centered[:, :, 1] = -refined_centered[:, :, 1]
    refined_centered[:, :, 2] = -refined_centered[:, :, 2]

    # 2. Un-center (add pelvis back)
    refined_poses = refined_centered + pelvis[:, np.newaxis, :]

    # Compute corrections for propagating to foot markers
    corrections = refined_poses - pose_3d  # (n_frames, 17, 3)

    # Foot markers that should follow ankle corrections
    # COCO: LAnkle=15, RAnkle=16
    LEFT_FOOT_MARKERS = {'LHeel', 'LBigToe', 'LSmallToe'}
    RIGHT_FOOT_MARKERS = {'RHeel', 'RBigToe', 'RSmallToe'}

    # Create refined frames data
    refined_frames = {}
    for fi, ts in enumerate(timestamps):
        refined_frames[ts] = {}
        for name, data in frames_data[ts].items():
            refined_frames[ts][name] = data.copy()
            # Update x, y, z for COCO joints (full 3D correction)
            if name in OPENCAP_TO_COCO:
                coco_idx = OPENCAP_TO_COCO[name]
                refined_frames[ts][name]['x'] = float(refined_poses[fi, coco_idx, 0])
                refined_frames[ts][name]['y'] = float(refined_poses[fi, coco_idx, 1])
                refined_frames[ts][name]['z'] = float(refined_poses[fi, coco_idx, 2])
            elif name in LEFT_FOOT_MARKERS:
                # Propagate LAnkle correction to left foot markers
                ankle_correction = corrections[fi, 15]  # LAnkle = COCO index 15
                refined_frames[ts][name]['x'] = data['x'] + float(ankle_correction[0])
                refined_frames[ts][name]['y'] = data['y'] + float(ankle_correction[1])
                refined_frames[ts][name]['z'] = data['z'] + float(ankle_correction[2])
            elif name in RIGHT_FOOT_MARKERS:
                # Propagate RAnkle correction to right foot markers
                ankle_correction = corrections[fi, 16]  # RAnkle = COCO index 16
                refined_frames[ts][name]['x'] = data['x'] + float(ankle_correction[0])
                refined_frames[ts][name]['y'] = data['y'] + float(ankle_correction[1])
                refined_frames[ts][name]['z'] = data['z'] + float(ankle_correction[2])

    # Compute stats for all axes
    diffs_x, diffs_y, diffs_z = [], [], []
    for fi, ts in enumerate(timestamps):
        for name, coco_idx in OPENCAP_TO_COCO.items():
            if name in frames_data[ts]:
                orig = frames_data[ts][name]
                ref = refined_frames[ts][name]
                diffs_x.append(abs(ref['x'] - orig['x']))
                diffs_y.append(abs(ref['y'] - orig['y']))
                diffs_z.append(abs(ref['z'] - orig['z']))

    print(f"Mean 3D corrections:")
    print(f"  X: {np.mean(diffs_x)*100:.2f} cm")
    print(f"  Y: {np.mean(diffs_y)*100:.2f} cm")
    print(f"  Z: {np.mean(diffs_z)*100:.2f} cm (depth)")
    total_3d = np.sqrt(np.array(diffs_x)**2 + np.array(diffs_y)**2 + np.array(diffs_z)**2)
    print(f"  Total 3D: {np.mean(total_3d)*100:.2f} cm")

    return refined_frames


class DepthComparisonViewer:
    def __init__(self, timestamps, original_frames, refined_frames):
        self.timestamps = timestamps
        self.original = original_frames
        self.refined = refined_frames
        self.n_frames = len(timestamps)
        self.current_frame = 0
        self.playing = False

        # Create figure
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Adjust layout for controls
        self.fig.subplots_adjust(bottom=0.2)

        # Slider
        ax_slider = self.fig.add_axes([0.2, 0.08, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, self.n_frames - 1,
                            valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        # Play button
        ax_play = self.fig.add_axes([0.4, 0.02, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.toggle_play)

        # Animation timer
        self.timer = self.fig.canvas.new_timer(interval=40)  # ~25 fps
        self.timer.add_callback(self.advance_frame)

        # Initial draw
        self.draw_frame(0)

        # Set title
        self.fig.suptitle('Depth Comparison: Original (Blue) vs Refined (Red)', fontsize=14)

    def draw_frame(self, frame_idx):
        self.ax.clear()

        ts = self.timestamps[frame_idx]
        orig = self.original[ts]
        ref = self.refined[ts]

        # Draw original (blue)
        self._draw_skeleton(orig, color='blue', alpha=0.7, label='Original')

        # Draw refined (red)
        self._draw_skeleton(ref, color='red', alpha=0.7, label='Refined')

        # Set axis properties
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Z (depth, m)')
        self.ax.set_zlabel('Y (m)')

        # Auto-scale based on data
        all_x, all_y, all_z = [], [], []
        for name, data in orig.items():
            if data['vis'] > 0.3:
                all_x.append(data['x'])
                all_y.append(-data['y'])
                all_z.append(data['z'])

        if all_x:
            x_center = np.mean(all_x)
            y_center = np.mean(all_y)
            z_center = np.mean(all_z)
            span = 1.5  # Larger span = smaller skeletons in view

            self.ax.set_xlim(x_center - span/2, x_center + span/2)
            self.ax.set_ylim(z_center - span/2, z_center + span/2)
            self.ax.set_zlim(y_center - span/2, y_center + span/2)
            self.ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

        self.ax.set_title(f'Frame {frame_idx} | t = {ts:.2f}s', fontsize=12)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, label='Original MediaPipe'),
            Line2D([0], [0], color='red', linewidth=2, label='Depth Refined'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

        self.fig.canvas.draw_idle()

    def _draw_skeleton(self, landmarks, color, alpha, label):
        # Draw joints
        for name, data in landmarks.items():
            if data['vis'] > 0.3:
                # Note: -y to flip (head up), z as depth
                self.ax.scatter(data['x'], data['z'], -data['y'],
                              c=color, s=40, alpha=alpha)

        # Draw bones
        for n1, n2 in SKELETON_CONNECTIONS:
            if n1 in landmarks and n2 in landmarks:
                d1, d2 = landmarks[n1], landmarks[n2]
                if d1['vis'] > 0.3 and d2['vis'] > 0.3:
                    self.ax.plot([d1['x'], d2['x']],
                               [d1['z'], d2['z']],
                               [-d1['y'], -d2['y']],
                               color=color, linewidth=2, alpha=alpha)

    def on_slider_change(self, val):
        self.current_frame = int(val)
        self.draw_frame(self.current_frame)

    def toggle_play(self, event):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.label.set_text('Pause')
            self.timer.start()
        else:
            self.btn_play.label.set_text('Play')
            self.timer.stop()

    def advance_frame(self):
        if self.playing:
            self.current_frame = (self.current_frame + 1) % self.n_frames
            self.slider.set_val(self.current_frame)

    def show(self):
        plt.show()


def main():
    # Default to joey if no argument
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    else:
        video_name = 'joey'

    # Paths
    csv_path = Path(f'data/output/pose-3d/{video_name}/{video_name}_raw_landmarks.csv')
    model_path = Path('models/checkpoints/best_depth_model.pth')

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        print("Run the pipeline first to generate landmarks")
        sys.exit(1)

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print("Train the depth model first")
        sys.exit(1)

    print(f"Loading landmarks from {csv_path}...")
    timestamps, original_frames = load_landmarks_from_csv(csv_path)
    print(f"Loaded {len(timestamps)} frames")

    # Apply depth refinement
    refined_frames = apply_depth_refinement(timestamps, original_frames, model_path)

    # Launch viewer
    print("\nLaunching interactive viewer...")
    print("Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Slider: Navigate frames")
    print("  - Play/Pause: Animate")
    print("\nBlue = Original MediaPipe")
    print("Red = Depth Refined")

    viewer = DepthComparisonViewer(timestamps, original_frames, refined_frames)
    viewer.show()


if __name__ == '__main__':
    main()
