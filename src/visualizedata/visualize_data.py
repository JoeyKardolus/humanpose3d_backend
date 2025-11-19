from cv2 import invert
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation


class VisualizeData:
    POSE_CONNECTIONS = [
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

    def __init__(self) -> None:
        pass

    def convert_landmarks_to_numpy(self, landmarks) -> np.ndarray:
        """Convert landmarks to numpy array more efficiently."""
        return np.array([[lm.x, lm.y, lm.z] for frame in landmarks for lm in frame])

    def plot_landmarks(self, landmarks: list) -> None:
        frames_data = [self.convert_landmarks_to_numpy(frame) for frame in landmarks]

        all_points = np.vstack(frames_data)
        margin = 0.1
        x_min, x_max = all_points[:, 0].min() - margin, all_points[:, 0].max() + margin
        y_min, y_max = all_points[:, 1].min() - margin, all_points[:, 1].max() + margin
        z_min, z_max = all_points[:, 2].min() - margin, all_points[:, 2].max() + margin

        fig = plt.figure(figsize=(10, 8))
        ax: Axes3D = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=-80, azim=-90)
        plt.subplots_adjust(bottom=0.15)

        anim = {"animation": None, "playing": True}

        def update(frame_idx):
            ax.clear()
            frame = frames_data[int(frame_idx)]
            x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]

            ax.scatter(x, y, z)

            for a, b in self.POSE_CONNECTIONS:
                ax.plot([x[a], x[b]], [y[a], y[b]], [z[a], z[b]])

            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.set_zlabel("Z axis")
            ax.set_title(f"Frame {int(frame_idx)}")
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

        ax_button = plt.axes([0.75, 0.05, 0.1, 0.03])
        btn_play = Button(ax_button, "Pause")
        btn_play.on_clicked(play_pause)

        update(0)

        anim["animation"] = FuncAnimation(
            fig, animate, frames=range(len(frames_data)), interval=0.1, repeat=True
        )

        plt.show()
