from __future__ import annotations

import os
from pathlib import Path

# Default to interactive backend, but allow override via environment
os.environ.setdefault("MPLBACKEND", "TkAgg")

import numpy as np
import matplotlib

matplotlib.use(os.environ["MPLBACKEND"])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from typing import List, Tuple


class VisualizeData:
    # MediaPipe 33-landmark connections (legacy - for non-augmented data)
    MEDIAPIPE_CONNECTIONS = [
        (8, 6),
        (6, 5),
        (5, 4),
        (4, 0),
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),  # Face
        (12, 11),
        (11, 23),
        (23, 24),
        (24, 12),  # Torso
        (12, 14),
        (14, 16),
        (16, 18),
        (18, 20),
        (20, 16),
        (16, 22),  # Left Arm
        (11, 13),
        (13, 15),
        (15, 17),
        (17, 19),
        (19, 15),
        (15, 21),  # Right Arm
        (24, 26),
        (26, 28),
        (28, 30),
        (30, 32),
        (32, 28),  # Left Leg
        (23, 25),
        (25, 27),
        (27, 29),
        (29, 31),
        (31, 27),  # Right Leg
    ]

    # OpenCap 65-marker anatomical connections (augmented data from Pose2Sim)
    # Minimal clean skeleton as requested
    OPENCAP_CONNECTIONS = [
        # Right leg: toe → ankle → knee → hip
        (13, 7),     # RBigToe → RAnkle
        (7, 5),      # RAnkle → RKnee
        (5, 3),      # RKnee → RHip

        # Left leg: toe → ankle → knee → hip
        (14, 8),     # LBigToe → LAnkle
        (8, 6),      # LAnkle → LKnee
        (6, 4),      # LKnee → LHip

        # Pelvis: connect hips
        (3, 4),      # RHip ↔ LHip

        # Spine: hip center to shoulder center (neck)
        (19, 0),     # Hip center → Neck

        # Shoulders: connect shoulders
        (1, 2),      # RShoulder ↔ LShoulder

        # Arms
        (1, 15),     # RShoulder → RElbow
        (15, 17),    # RElbow → RWrist
        (2, 16),     # LShoulder → LElbow
        (16, 18),    # LElbow → LWrist

        # Head: leave as dot (no connections)
    ]

    # Alias for backward compatibility
    POSE_CONNECTIONS = MEDIAPIPE_CONNECTIONS

    def __init__(self) -> None:
        pass

    def convert_landmarks_to_numpy(self, frame) -> np.ndarray:
        """Convert parsed MediaPipe or TRC frame landmarks into XYZ coordinates."""
        if frame is None:
            return np.zeros((0, 3))
        if isinstance(frame, np.ndarray):
            return frame
        if hasattr(frame, "x"):
            iterator = frame
        else:
            iterator = frame if frame and hasattr(frame[0], "x") else []
        return np.array([[lm.x, lm.y, lm.z] for lm in iterator], dtype=float)

    def load_trc_frames(self, trc_path: Path) -> Tuple[List[str], List[np.ndarray]]:
        """Load TRC file and return marker names and frame data.

        Note: For Pose2Sim-augmented files, the header may only list the initial markers
        (e.g., 22), but the data columns contain all markers including augmented ones (e.g., 65).
        This function reads ALL markers from the actual data, not just those in the header.
        """
        lines = trc_path.read_text(encoding="utf-8").splitlines()
        name_line_idx = next(
            (idx for idx, line in enumerate(lines) if line.startswith("Frame#")), None
        )
        if name_line_idx is None:
            raise ValueError(f"Could not locate marker header in {trc_path}")
        axis_line_idx = name_line_idx + 1
        data_start_idx = axis_line_idx + 1
        while data_start_idx < len(lines) and not lines[data_start_idx].strip():
            data_start_idx += 1

        # Read marker names from header
        name_tokens = lines[name_line_idx].split("\t")
        header_markers = []
        for idx in range(2, len(name_tokens), 3):
            label = name_tokens[idx].strip()
            if not label:
                label = f"Marker{len(header_markers)+1}"
            header_markers.append(label)

        # Determine actual number of markers from first data line
        first_data_line = None
        for line in lines[data_start_idx:]:
            if line.strip():
                first_data_line = line.rstrip("\n").split("\t")
                break

        if first_data_line is None:
            raise ValueError(f"No data rows found in {trc_path}")

        # Calculate actual marker count from data columns (skip Frame# and Time)
        data_columns = len(first_data_line) - 2
        actual_marker_count = data_columns // 3

        # If we have more markers in data than in header, generate names for augmented markers
        marker_names = header_markers.copy()
        if actual_marker_count > len(header_markers):
            # These are augmented markers added by Pose2Sim
            # Use standard OpenCap marker names
            augmented_names = [
                "C7_study", "r_shoulder_study", "L_shoulder_study",
                "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study",
                "r_knee_study", "L_knee_study", "r_mknee_study", "L_mknee_study",
                "r_ankle_study", "L_ankle_study", "r_mankle_study", "L_mankle_study",
                "r_calc_study", "L_calc_study", "r_toe_study", "L_toe_study",
                "r_5meta_study", "L_5meta_study",
                "r_lelbow_study", "L_lelbow_study", "r_melbow_study", "L_melbow_study",
                "r_lwrist_study", "L_lwrist_study", "r_mwrist_study", "L_mwrist_study",
                "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
                "r_sh1_study", "r_sh2_study", "r_sh3_study",
                "L_sh1_study", "L_sh2_study", "L_sh3_study",
                "RHJC_study", "LHJC_study",
            ]
            num_augmented_needed = actual_marker_count - len(header_markers)
            marker_names.extend(augmented_names[:num_augmented_needed])

            # Fill any remaining with generic names if we somehow need more
            while len(marker_names) < actual_marker_count:
                marker_names.append(f"AugMarker{len(marker_names)+1}")

        # Load all frame data using the actual marker count
        frames: List[np.ndarray] = []
        for line in lines[data_start_idx:]:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            data = parts[2:]
            coords = []
            for marker_idx in range(actual_marker_count):
                base = marker_idx * 3
                triple = data[base : base + 3] if base + 2 < len(data) else ["", "", ""]
                if len(triple) < 3:
                    triple = ["", "", ""]
                try:
                    coords.append(
                        [
                            float(triple[0]) if triple[0] else np.nan,
                            float(triple[1]) if triple[1] else np.nan,
                            float(triple[2]) if triple[2] else np.nan,
                        ]
                    )
                except ValueError:
                    coords.append([np.nan, np.nan, np.nan])
            frames.append(np.array(coords, dtype=float))
        return marker_names, frames

    def plot_landmarks(
        self,
        landmarks: List,
        export_path: Path | None = None,
        block: bool = True,
    ) -> None:
        frames_data = [self.convert_landmarks_to_numpy(frame) for frame in landmarks]
        if not frames_data:
            print("[visualize] no landmarks available to plot.")
            return

        stacked = np.vstack(frames_data)
        mask = ~np.isnan(stacked).any(axis=1)
        valid_points = stacked[mask]
        if valid_points.size == 0:
            print("[visualize] landmarks only contain NaNs; skipping plot.")
            return

        margin = 0.1
        x_min, x_max = valid_points[:, 0].min() - margin, valid_points[:, 0].max() + margin
        y_min, y_max = valid_points[:, 1].min() - margin, valid_points[:, 1].max() + margin
        z_min, z_max = valid_points[:, 2].min() - margin, valid_points[:, 2].max() + margin

        fig = plt.figure(figsize=(10, 8))
        ax: Axes3D = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=-80, azim=-90)
        plt.subplots_adjust(bottom=0.15)

        anim = {"animation": None, "playing": True}

        # Auto-detect which connections to use based on number of markers
        # 50+ markers = OpenCap augmented, <50 markers = MediaPipe
        num_markers = len(frames_data[0]) if frames_data else 0
        connections = self.OPENCAP_CONNECTIONS if num_markers >= 50 else self.MEDIAPIPE_CONNECTIONS

        def update(frame_idx):
            ax.clear()
            idx = min(max(int(frame_idx), 0), len(frames_data) - 1)
            frame = frames_data[idx]
            x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]

            ax.scatter(x, y, z, c='blue', marker='o', s=20)

            for a, b in connections:
                if (
                    a >= len(frame)
                    or b >= len(frame)
                    or np.isnan(frame[a]).any()
                    or np.isnan(frame[b]).any()
                ):
                    continue
                ax.plot([x[a], x[b]], [y[a], y[b]], [z[a], z[b]], 'r-', linewidth=1.5)

            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.set_zlabel("Z axis")
            ax.set_title(f"Frame {idx}")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            fig.canvas.draw_idle()

        def animate(frame):
            slider.set_val(frame % len(frames_data))

        def play_pause(event):
            if anim["playing"]:
                anim["animation"].event_source.stop()
                anim["playing"] = False
                btn_play.label.set_text("Play")
            else:
                anim["animation"].event_source.start()
                anim["playing"] = True
                btn_play.label.set_text("Pause")

        ax_slider = plt.axes([0.2, 0.05, 0.5, 0.03])
        slider = Slider(
            ax_slider, "Frame", 0, len(frames_data) - 1, valinit=0, valstep=1
        )
        slider.on_changed(update)

        btn_play = None
        if len(frames_data) > 1:
            ax_button = plt.axes([0.75, 0.05, 0.1, 0.03])
            btn_play = Button(ax_button, "Pause")
            btn_play.on_clicked(play_pause)

        update(0)

        anim["animation"] = FuncAnimation(
            fig, animate, frames=range(len(frames_data)), interval=50, repeat=True
        )

        if export_path:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            writer: PillowWriter | FFMpegWriter
            chosen_writer = "pillow"
            try:
                writer = FFMpegWriter(fps=20)
                anim["animation"].save(str(export_path), writer=writer)
                chosen_writer = "ffmpeg"
            except (RuntimeError, FileNotFoundError):
                # PillowWriter can't save MP4, use GIF instead
                export_path = export_path.with_suffix(".gif")
                writer = PillowWriter(fps=10)
                anim["animation"].save(str(export_path), writer=writer)
            print(f"[visualize] saved animation ({chosen_writer}) to {export_path}")

        backend = matplotlib.get_backend().lower()
        interactive = backend not in {"agg", "module://matplotlib_inline.backend_inline"}
        if interactive and block:
            plt.show()
        else:
            plt.close(fig)

    def plot_trc_file(
        self, trc_path: Path, export_path: Path | None = None, block: bool = True
    ) -> None:
        _, frames = self.load_trc_frames(trc_path)
        self.plot_landmarks(frames, export_path=export_path, block=block)
