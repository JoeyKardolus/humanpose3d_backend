#!/usr/bin/env python3
"""
3D Skeleton visualization showing joint angle refinement effect.

Shows two overlapped skeletons:
- Red: Raw MediaPipe output (before refinement)
- Blue: Refined skeleton (after Pose2Sim + neural refinement)

Usage:
    uv run --group neural python visualize_joint_refinement.py [video_name]
"""

import sys
sys.path.insert(0, '.')

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from src.kinematics.comprehensive_joint_angles import compute_all_joint_angles
from src.kinematics.trc_utils import read_trc
from src.joint_refinement.model import create_model


# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    # Spine
    ('Hip', 'Neck'),
    # Right leg
    ('Hip', 'RHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
    ('RAnkle', 'RHeel'), ('RAnkle', 'RBigToe'),
    # Left leg
    ('Hip', 'LHip'), ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
    ('LAnkle', 'LHeel'), ('LAnkle', 'LBigToe'),
    # Right arm
    ('Neck', 'RShoulder'), ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    # Left arm
    ('Neck', 'LShoulder'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    # Shoulders
    ('RShoulder', 'LShoulder'),
    # Head
    ('Neck', 'Nose'),
]

# Map joint angles to marker positions for visualization
JOINT_TO_MARKERS = {
    'pelvis': 'Hip',
    'hip_R': 'RHip',
    'hip_L': 'LHip',
    'knee_R': 'RKnee',
    'knee_L': 'LKnee',
    'ankle_R': 'RAnkle',
    'ankle_L': 'LAnkle',
    'trunk': 'Neck',
    'shoulder_R': 'RShoulder',
    'shoulder_L': 'LShoulder',
    'elbow_R': 'RElbow',
    'elbow_L': 'LElbow',
}

JOINT_ORDER = ['pelvis', 'hip_R', 'hip_L', 'knee_R', 'knee_L', 'ankle_R', 'ankle_L',
               'trunk', 'shoulder_R', 'shoulder_L', 'elbow_R', 'elbow_L']


def load_raw_mediapipe(csv_path: Path, times: np.ndarray) -> dict:
    """Load raw MediaPipe landmarks from CSV."""
    print(f"Loading raw MediaPipe from: {csv_path.name}")

    df = pd.read_csv(csv_path)

    # Get unique timestamps and landmarks
    timestamps = df['timestamp_s'].unique()
    landmarks = list(df['landmark'].unique())

    # Build marker index
    marker_idx = {name: i for i, name in enumerate(landmarks)}

    # Build coords array (n_frames, n_markers, 3)
    n_frames = len(timestamps)
    n_markers = len(landmarks)
    coords = np.full((n_frames, n_markers, 3), np.nan)

    # Map timestamps to frame indices
    time_to_frame = {t: i for i, t in enumerate(timestamps)}

    for _, row in df.iterrows():
        fi = time_to_frame.get(row['timestamp_s'])
        if fi is not None:
            mi = marker_idx[row['landmark']]
            coords[fi, mi] = [row['x_m'], row['y_m'], row['z_m']]

    # Compute derived markers (Hip center, Neck) if not present
    if 'Hip' not in marker_idx and 'LHip' in marker_idx and 'RHip' in marker_idx:
        hip_idx = len(landmarks)
        landmarks.append('Hip')
        marker_idx['Hip'] = hip_idx
        new_coords = np.full((n_frames, len(landmarks), 3), np.nan)
        new_coords[:, :n_markers, :] = coords
        lhip = coords[:, marker_idx['LHip'], :]
        rhip = coords[:, marker_idx['RHip'], :]
        new_coords[:, hip_idx, :] = (lhip + rhip) / 2
        coords = new_coords
        n_markers = len(landmarks)

    if 'Neck' not in marker_idx and 'LShoulder' in marker_idx and 'RShoulder' in marker_idx:
        neck_idx = len(landmarks)
        landmarks.append('Neck')
        marker_idx['Neck'] = neck_idx
        new_coords = np.full((n_frames, len(landmarks), 3), np.nan)
        new_coords[:, :n_markers, :] = coords
        lsho = coords[:, marker_idx['LShoulder'], :]
        rsho = coords[:, marker_idx['RShoulder'], :]
        new_coords[:, neck_idx, :] = (lsho + rsho) / 2
        coords = new_coords
        n_markers = len(landmarks)

    # Match frames by index (frame counts are nearly identical)
    # This ensures perfect sync between raw and refined skeletons
    n_refined = len(times)
    n_raw = n_frames

    # Use frame index matching - scale raw indices to match refined count
    coords_matched = np.full((n_refined, n_markers, 3), np.nan)

    for ri in range(n_refined):
        # Map refined frame to nearest raw frame
        raw_idx = int(ri * n_raw / n_refined)
        raw_idx = min(raw_idx, n_raw - 1)
        coords_matched[ri] = coords[raw_idx]

    print(f"  Loaded {len(landmarks)} landmarks, {n_raw} raw frames -> matched to {n_refined} refined frames")
    return marker_idx, coords_matched


def load_skeleton_and_angles(trc_path: Path, model_path: Path, raw_csv_path: Path = None):
    """Load 3D skeleton positions and compute/refine joint angles."""
    print(f"Loading from: {trc_path.name}")

    # Load 3D marker positions
    marker_idx, frames, times, coords = read_trc(trc_path)
    n_frames = len(times)
    print(f"Loaded {n_frames} frames, {len(marker_idx)} markers")

    # Compute ISB joint angles
    angles = compute_all_joint_angles(
        trc_path, smooth_window=9, unwrap=True,
        zero_mode='first_n_seconds', zero_window_s=0.5, verbose=True
    )

    # Convert to array
    original_angles = np.zeros((n_frames, 12, 3))
    for ji, joint in enumerate(JOINT_ORDER):
        df = angles[joint]
        angle_cols = df.iloc[:, 1:].values
        if angle_cols.shape[1] == 1:
            original_angles[:, ji, 0] = angle_cols[:, 0]
        else:
            original_angles[:, ji, :] = angle_cols[:, :3]

    # Load and apply model
    print(f"Loading model: {model_path}")
    model = create_model()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Handle NaN
    nan_mask = np.isnan(original_angles)
    angles_clean = np.nan_to_num(original_angles, nan=0.0)
    visibility = torch.tensor((~nan_mask).any(axis=2).astype(np.float32))

    # Refine
    print("Applying neural refinement...")
    with torch.no_grad():
        input_tensor = torch.tensor(angles_clean, dtype=torch.float32)
        refined_tensor, delta_tensor = model(input_tensor, visibility)

    corrections = delta_tensor.numpy()
    corrections[nan_mask] = 0  # Zero out invalid

    # Stats
    valid_mask = ~nan_mask
    valid_corr = corrections[valid_mask]
    print(f"\nMean |correction|: {np.abs(valid_corr).mean():.2f}°")
    print(f"Max |correction|: {np.abs(valid_corr).max():.2f}°")

    # Load raw MediaPipe if available
    raw_marker_idx, raw_coords = None, None
    if raw_csv_path and raw_csv_path.exists():
        raw_marker_idx, raw_coords = load_raw_mediapipe(raw_csv_path, times)

    return marker_idx, times, coords, original_angles, corrections, raw_marker_idx, raw_coords


class SkeletonViewer:
    """3D skeleton viewer with joint angle corrections."""

    def __init__(self, marker_idx, times, coords, angles, corrections,
                 raw_marker_idx=None, raw_coords=None):
        self.marker_idx = marker_idx
        self.times = times
        self.coords = coords
        self.angles = angles
        self.corrections = corrections
        self.raw_marker_idx = raw_marker_idx
        self.raw_coords = raw_coords
        self.n_frames = len(times)
        self.current_frame = 0
        self.playing = False

        # Create figure
        self.fig = plt.figure(figsize=(16, 10))

        # 3D skeleton plot
        self.ax3d = self.fig.add_subplot(121, projection='3d')

        # Correction bar chart
        self.ax_bar = self.fig.add_subplot(122)

        # Controls
        self.fig.subplots_adjust(bottom=0.15)
        ax_slider = self.fig.add_axes([0.15, 0.05, 0.5, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, self.n_frames - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        ax_play = self.fig.add_axes([0.7, 0.05, 0.08, 0.03])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.toggle_play)

        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self.advance_frame)

        self.update_plot()

    def get_marker_pos(self, fi, name):
        """Get marker position for frame from refined skeleton."""
        if name not in self.marker_idx:
            return None
        idx = self.marker_idx[name]
        pos = self.coords[fi, idx]
        if np.isfinite(pos).all():
            return pos
        return None

    def get_raw_marker_pos(self, fi, name):
        """Get marker position for frame from raw MediaPipe."""
        if self.raw_marker_idx is None or self.raw_coords is None:
            return None
        if name not in self.raw_marker_idx:
            return None
        idx = self.raw_marker_idx[name]
        pos = self.raw_coords[fi, idx]
        if np.isfinite(pos).all():
            return pos
        return None

    def update_plot(self):
        fi = self.current_frame
        t = self.times[fi]

        # Clear
        self.ax3d.clear()
        self.ax_bar.clear()

        # Get correction magnitudes per joint
        corr_magnitude = np.abs(self.corrections[fi]).mean(axis=1)  # (12,)

        # Normalize for color mapping (0-10 degrees -> 0-1)
        corr_normalized = np.clip(corr_magnitude / 10.0, 0, 1)

        # Draw RAW MediaPipe skeleton (red, transparent) FIRST (behind)
        if self.raw_marker_idx is not None:
            for m1, m2 in SKELETON_CONNECTIONS:
                p1 = self.get_raw_marker_pos(fi, m1)
                p2 = self.get_raw_marker_pos(fi, m2)
                if p1 is not None and p2 is not None:
                    self.ax3d.plot(
                        [p1[0], p2[0]],
                        [p1[2], p2[2]],  # Z as depth
                        [-p1[1], -p2[1]],  # -Y for head up
                        color='red', linewidth=2, alpha=0.4, linestyle='--'
                    )

            # Draw raw joints
            for name in self.raw_marker_idx:
                pos = self.get_raw_marker_pos(fi, name)
                if pos is not None:
                    self.ax3d.scatter(
                        pos[0], pos[2], -pos[1],
                        c='red', s=30, alpha=0.4, marker='o'
                    )

        # Draw REFINED skeleton bones (blue, solid)
        for m1, m2 in SKELETON_CONNECTIONS:
            p1 = self.get_marker_pos(fi, m1)
            p2 = self.get_marker_pos(fi, m2)
            if p1 is not None and p2 is not None:
                self.ax3d.plot(
                    [p1[0], p2[0]],
                    [p1[2], p2[2]],  # Z as depth
                    [-p1[1], -p2[1]],  # -Y for head up
                    color='blue', linewidth=2.5, alpha=0.9
                )

        # Draw refined joints with color based on correction magnitude
        cmap = plt.cm.RdYlGn_r  # Red = high correction, Green = low
        for ji, joint in enumerate(JOINT_ORDER):
            marker_name = JOINT_TO_MARKERS.get(joint)
            if marker_name:
                pos = self.get_marker_pos(fi, marker_name)
                if pos is not None:
                    color = cmap(corr_normalized[ji])
                    size = 100 + corr_magnitude[ji] * 20  # Bigger = more correction
                    self.ax3d.scatter(
                        pos[0], pos[2], -pos[1],
                        c=[color], s=size, edgecolors='black', linewidths=1,
                        zorder=10
                    )

        # Set axis properties
        self.ax3d.set_xlabel('X (m)')
        self.ax3d.set_ylabel('Z (depth)')
        self.ax3d.set_zlabel('Y (m)')

        # Auto-scale
        all_pos = []
        for name in self.marker_idx:
            pos = self.get_marker_pos(fi, name)
            if pos is not None:
                all_pos.append(pos)

        if all_pos:
            all_pos = np.array(all_pos)
            center = all_pos.mean(axis=0)
            span = 1.2

            self.ax3d.set_xlim(center[0] - span/2, center[0] + span/2)
            self.ax3d.set_ylim(center[2] - span/2, center[2] + span/2)
            self.ax3d.set_zlim(-center[1] - span/2, -center[1] + span/2)

        # Add legend for skeleton colors
        legend_elements = [
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, alpha=0.5,
                   label='Raw MediaPipe'),
            Line2D([0], [0], color='blue', linestyle='-', linewidth=2.5,
                   label='Refined (Pose2Sim + Neural)'),
        ]
        self.ax3d.legend(handles=legend_elements, loc='upper left', fontsize=8)

        self.ax3d.set_title(f'3D Skeleton | Frame {fi} | t={t:.2f}s\n'
                           f'Joint colors: Green=small correction, Red=large')

        # Bar chart of corrections
        colors = [cmap(c) for c in corr_normalized]
        bars = self.ax_bar.barh(JOINT_ORDER, corr_magnitude, color=colors)
        self.ax_bar.set_xlabel('|Correction| (degrees)')
        self.ax_bar.set_title('Joint Angle Corrections')
        self.ax_bar.set_xlim(0, max(10, corr_magnitude.max() * 1.2))
        self.ax_bar.grid(axis='x', alpha=0.3)

        # Value labels
        for bar, val in zip(bars, corr_magnitude):
            if val > 0.1:
                self.ax_bar.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                               f'{val:.1f}°', va='center', fontsize=9)

        # Legend
        self.fig.suptitle(
            f'Joint Angle Neural Refinement\n'
            f'Mean correction: {np.abs(self.corrections[fi]).mean():.1f}° | '
            f'Max: {np.abs(self.corrections[fi]).max():.1f}°',
            fontsize=12, fontweight='bold'
        )

        plt.tight_layout(rect=[0, 0.1, 1, 0.93])
        self.fig.canvas.draw_idle()

    def on_slider_change(self, val):
        self.current_frame = int(val)
        self.update_plot()

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
    video_name = sys.argv[1] if len(sys.argv) > 1 else 'joey'
    run_dir = Path(f'data/output/pose-3d/{video_name}')

    # Find TRC file
    trc_candidates = [
        run_dir / f'{video_name}_LSTM_complete.trc',
        run_dir / f'{video_name}_final.trc',
        run_dir / f'{video_name}_LSTM.trc',
    ]

    trc_path = None
    for candidate in trc_candidates:
        if candidate.exists():
            trc_path = candidate
            break

    if trc_path is None:
        print(f"Error: No TRC file found in {run_dir}")
        sys.exit(1)

    model_path = Path('models/checkpoints/best_joint_model.pth')
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Find raw MediaPipe CSV
    raw_csv_path = run_dir / f'{video_name}_raw_landmarks.csv'
    if not raw_csv_path.exists():
        print(f"Warning: Raw MediaPipe CSV not found: {raw_csv_path}")
        print("  Will only show refined skeleton")
        raw_csv_path = None

    # Load data
    result = load_skeleton_and_angles(trc_path, model_path, raw_csv_path)
    marker_idx, times, coords, angles, corrections, raw_marker_idx, raw_coords = result

    # Launch viewer
    print("\nLaunching 3D skeleton viewer...")
    print("  - Red dashed: Raw MediaPipe output")
    print("  - Blue solid: Refined (Pose2Sim + Neural)")
    print("  - Joint colors: Green=small correction, Red=large correction")
    print("  - Drag to rotate, slider to navigate, Play to animate")

    viewer = SkeletonViewer(marker_idx, times, coords, angles, corrections,
                            raw_marker_idx, raw_coords)
    viewer.show()


if __name__ == '__main__':
    main()
