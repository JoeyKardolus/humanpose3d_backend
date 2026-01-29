#!/usr/bin/env python3
"""Z-Sign Diagnostic Visualization for POF Model.

Visualizes the Z-sign classification head behavior to debug depth ambiguity
issues that cause marker L/R swapping.

Usage:
    uv run python scripts/viz/visualize_zsign_diagnostic.py data/input/video.mp4 --height 1.78
    uv run python scripts/viz/visualize_zsign_diagnostic.py video.mp4 --focus-limbs 9,8
    uv run python scripts/viz/visualize_zsign_diagnostic.py video.mp4 --export-csv zsign.csv
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mediastream.media_stream import read_video_rgb
from src.posedetector.rtmpose_detector import RTMPoseDetector
from src.pof.inference import CameraPOFInference
from src.pof.constants import LIMB_NAMES, LIMB_DEFINITIONS, NUM_LIMBS

# COCO-17 skeleton connections for visualization
COCO_CONNECTIONS = [
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Critical limbs for monitoring (solved early, control L/R structure)
CRITICAL_LIMBS = [9, 8, 12, 13]  # hip_width, shoulder_width, cross-body


def classify_zsign_state(pof_z: np.ndarray, zsign_prob: np.ndarray,
                         z_thresh: float = 0.15, prob_thresh: float = 0.1):
    """Classify each limb's Z-sign state.

    Returns:
        states: (14,) array where:
            0 = facing camera (Z < 0, confident)
            1 = facing away (Z > 0, confident)
            2 = parallel/ambiguous
    """
    states = np.zeros(len(pof_z), dtype=int)

    for i in range(len(pof_z)):
        z_mag = abs(pof_z[i])
        prob_confidence = abs(zsign_prob[i] - 0.5)

        if z_mag < z_thresh or prob_confidence < prob_thresh:
            states[i] = 2  # parallel/ambiguous
        elif pof_z[i] < 0:
            states[i] = 0  # facing camera
        else:
            states[i] = 1  # facing away

    return states


def get_pose_2d_and_visibility(frames, fps):
    """Extract 2D poses using RTMPose."""
    detector = RTMPoseDetector(model_size="m", device="cpu")
    result = detector.detect(frames, fps, visibility_min=0.1)
    return result.keypoints_2d, result.visibility


def run_inference_with_zsign(video_path: Path, height_m: float, max_frames: int):
    """Run POF inference and collect Z-sign diagnostic data."""
    print(f"Loading video: {video_path}")
    frames, fps = read_video_rgb(video_path)

    if len(frames) > max_frames:
        print(f"Using first {max_frames} frames")
        frames = frames[:max_frames]

    print(f"Extracting 2D poses from {len(frames)} frames...")
    pose_2d, visibility = get_pose_2d_and_visibility(frames, fps)

    print("Loading POF model...")
    # Use SemGCN-Temporal model which has Z-sign classification head
    repo_root = Path(__file__).parent.parent.parent
    model_path = repo_root / "models/checkpoints/best_pof_semgcn-temporal_model.pth"
    if not model_path.exists():
        # Fallback to unused folder
        model_path = repo_root / "models/checkpoints/unused/best_pof_semgcn-temporal_model.pth"
    pof_inference = CameraPOFInference(str(model_path))

    print("Running POF inference with Z-sign diagnostics...")
    pof, zsign_info = pof_inference.predict_pof_with_zsign(pose_2d, visibility)

    print("Reconstructing 3D poses...")
    aspect_ratio = frames.shape[2] / frames.shape[1]
    poses_3d = pof_inference.reconstruct_3d(
        pose_2d, visibility, height_m,
        use_meter_coords=True,
        aspect_ratio=aspect_ratio,
    )

    return poses_3d, pof, zsign_info, visibility


def visualize_zsign_diagnostic(poses_3d, pof, zsign_info, visibility,
                               focus_limbs=None, title="Z-Sign Diagnostic"):
    """Interactive multi-panel Z-sign visualization."""
    n_frames = len(poses_3d)

    if focus_limbs is None:
        focus_limbs = CRITICAL_LIMBS

    # Prepare display poses (center on pelvis, flip Y)
    pelvis = (poses_3d[:, 11] + poses_3d[:, 12]) / 2
    centered_poses = poses_3d - pelvis[:, np.newaxis, :]
    display_poses = centered_poses.copy()
    display_poses[:, :, 1] = -centered_poses[:, :, 1]

    # Compute bounds
    pad = 0.2
    max_range = max(
        display_poses[:, :, 0].max() - display_poses[:, :, 0].min(),
        display_poses[:, :, 1].max() - display_poses[:, :, 1].min(),
        display_poses[:, :, 2].max() - display_poses[:, :, 2].min(),
    ) / 2 + pad

    # Create figure with 4 panels
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.25, wspace=0.25)

    ax_skeleton = fig.add_subplot(gs[0, 0], projection='3d')
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_temporal = fig.add_subplot(gs[1, 1])

    plt.subplots_adjust(bottom=0.15)

    # State
    state = {'playing': False, 'frame_idx': 0}

    # Color maps
    zsign_colors = {0: 'blue', 1: 'red', 2: 'gold'}  # toward, away, parallel

    def update_plot(frame_idx):
        # Clear all axes
        ax_skeleton.clear()
        ax_bars.clear()
        ax_heatmap.clear()
        ax_temporal.clear()

        pose = display_poses[frame_idx]
        vis = visibility[frame_idx]
        probs = zsign_info['probs'][frame_idx]
        pof_z = pof[frame_idx, :, 2]
        corrections = zsign_info['corrections'][frame_idx]

        states = classify_zsign_state(pof_z, probs)

        # ===== Panel 1: 3D Skeleton with Z-sign colored limbs =====
        ax_skeleton.view_init(elev=10, azim=-80)

        # Draw limbs with Z-sign coloring
        for limb_idx, (p_idx, c_idx) in enumerate(LIMB_DEFINITIONS):
            if vis[p_idx] > 0.3 and vis[c_idx] > 0.3:
                color = zsign_colors[states[limb_idx]]
                lw = 4 if limb_idx in CRITICAL_LIMBS else 2
                ax_skeleton.plot(
                    [pose[p_idx, 0], pose[c_idx, 0]],
                    [pose[p_idx, 2], pose[c_idx, 2]],
                    [pose[p_idx, 1], pose[c_idx, 1]],
                    color=color, linewidth=lw, alpha=0.9
                )

        # Draw joints
        for i in range(17):
            if vis[i] > 0.3:
                ax_skeleton.scatter(
                    pose[i, 0], pose[i, 2], pose[i, 1],
                    c='black', s=40, edgecolor='white', linewidth=0.5
                )

        ax_skeleton.set_xlabel('X')
        ax_skeleton.set_ylabel('Z (depth)')
        ax_skeleton.set_zlabel('Y')
        ax_skeleton.set_xlim(-max_range, max_range)
        ax_skeleton.set_ylim(-max_range, max_range)
        ax_skeleton.set_zlim(-max_range, max_range)
        ax_skeleton.set_title(f'3D Skeleton (Frame {frame_idx + 1}/{n_frames})\n'
                              'Blue=toward cam, Red=away, Gold=parallel')

        # ===== Panel 2: Per-limb confidence bars =====
        bar_positions = np.arange(NUM_LIMBS)
        bar_values = probs - 0.5  # Center at 0 (left=toward, right=away)

        colors = [zsign_colors[s] for s in states]
        edge_colors = ['red' if c else 'none' for c in corrections]
        edge_widths = [2 if c else 0 for c in corrections]

        bars = ax_bars.barh(bar_positions, bar_values, color=colors,
                            edgecolor=edge_colors, linewidth=edge_widths)

        # Mark critical limbs
        for limb_idx in CRITICAL_LIMBS:
            ax_bars.axhline(limb_idx, color='purple', linestyle='--', alpha=0.3)

        ax_bars.axvline(0, color='black', linewidth=1)
        ax_bars.set_yticks(bar_positions)
        ax_bars.set_yticklabels([f'{i}: {LIMB_NAMES[i]}' for i in range(NUM_LIMBS)],
                                 fontsize=8)
        ax_bars.set_xlim(-0.5, 0.5)
        ax_bars.set_xlabel('Z-Sign Probability (left=toward cam, right=away)')
        ax_bars.set_title('Per-Limb Z-Sign Confidence\n(Red edge = correction applied)')

        # ===== Panel 3: Temporal heatmap =====
        heatmap_data = zsign_info['probs'].T  # (14, N)
        im = ax_heatmap.imshow(heatmap_data, aspect='auto', cmap='coolwarm',
                                vmin=0, vmax=1, interpolation='nearest')

        # Mark current frame
        ax_heatmap.axvline(frame_idx, color='lime', linewidth=2)

        # Mark correction events
        correction_frames = np.where(zsign_info['corrections'].T)
        ax_heatmap.scatter(correction_frames[1], correction_frames[0],
                           marker='x', c='black', s=10, alpha=0.5)

        ax_heatmap.set_yticks(range(NUM_LIMBS))
        ax_heatmap.set_yticklabels([f'{i}' for i in range(NUM_LIMBS)], fontsize=8)
        ax_heatmap.set_xlabel('Frame')
        ax_heatmap.set_ylabel('Limb Index')
        ax_heatmap.set_title('Z-Sign Probability Over Time\n(Blue=toward, Red=away, X=correction)')

        # ===== Panel 4: Focus limbs temporal plot =====
        window = 50
        start = max(0, frame_idx - window)
        end = min(n_frames, frame_idx + window)
        x_range = np.arange(start, end)

        for limb_idx in focus_limbs:
            label = f'{limb_idx}: {LIMB_NAMES[limb_idx]}'
            ax_temporal.plot(x_range, zsign_info['probs'][start:end, limb_idx],
                             label=label, linewidth=2)

        # Threshold line
        ax_temporal.axhline(0.5, color='black', linestyle='--', alpha=0.5)

        # Low confidence region
        ax_temporal.axhspan(0.4, 0.6, color='yellow', alpha=0.2)

        # Current frame marker
        ax_temporal.axvline(frame_idx, color='lime', linewidth=2, label='Current')

        ax_temporal.set_xlim(start, end)
        ax_temporal.set_ylim(0, 1)
        ax_temporal.set_xlabel('Frame')
        ax_temporal.set_ylabel('P(Z > 0)')
        ax_temporal.set_title(f'Focus Limbs Z-Sign Probability\n(Yellow = low confidence zone)')
        ax_temporal.legend(loc='upper right', fontsize=8)
        ax_temporal.grid(True, alpha=0.3)

        fig.canvas.draw_idle()

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)

    def on_slider(val):
        state['frame_idx'] = int(val)
        update_plot(state['frame_idx'])

    slider.on_changed(on_slider)

    # Play/Pause button
    ax_button = plt.axes([0.4, 0.01, 0.2, 0.03])
    button = Button(ax_button, 'Play')

    def on_button(event):
        state['playing'] = not state['playing']
        button.label.set_text('Pause' if state['playing'] else 'Play')

    button.on_clicked(on_button)

    # Animation
    def animate(event):
        if state['playing']:
            state['frame_idx'] = (state['frame_idx'] + 1) % n_frames
            slider.set_val(state['frame_idx'])

    timer = fig.canvas.new_timer(interval=80)  # ~12.5 fps for analysis
    timer.add_callback(animate, None)
    timer.start()

    update_plot(0)
    plt.show()


def export_zsign_csv(zsign_info, output_path: Path):
    """Export Z-sign data to CSV for external analysis."""
    n_frames = len(zsign_info['probs'])

    with open(output_path, 'w') as f:
        # Header
        header = ['frame']
        for limb_idx in range(NUM_LIMBS):
            name = LIMB_NAMES[limb_idx]
            header.extend([f'{name}_prob', f'{name}_correction'])
        f.write(','.join(header) + '\n')

        # Data
        for i in range(n_frames):
            row = [str(i)]
            for limb_idx in range(NUM_LIMBS):
                row.append(f'{zsign_info["probs"][i, limb_idx]:.4f}')
                row.append('1' if zsign_info['corrections'][i, limb_idx] else '0')
            f.write(','.join(row) + '\n')

    print(f"Exported Z-sign data to {output_path}")


def print_zsign_summary(zsign_info):
    """Print summary statistics about Z-sign predictions."""
    n_frames = len(zsign_info['probs'])
    probs = zsign_info['probs']
    corrections = zsign_info['corrections']

    print("\n" + "=" * 60)
    print("Z-SIGN DIAGNOSTIC SUMMARY")
    print("=" * 60)

    print(f"\nTotal frames: {n_frames}")

    print("\nPer-limb statistics:")
    print("-" * 60)
    print(f"{'Limb':<20} {'Mean Prob':>10} {'Std':>8} {'Corrections':>12} {'Flips':>8}")
    print("-" * 60)

    for limb_idx in range(NUM_LIMBS):
        name = LIMB_NAMES[limb_idx]
        mean_prob = probs[:, limb_idx].mean()
        std_prob = probs[:, limb_idx].std()
        n_corrections = corrections[:, limb_idx].sum()

        # Count flips (changes in binary prediction)
        binary_preds = (probs[:, limb_idx] > 0.5).astype(int)
        n_flips = np.sum(np.abs(np.diff(binary_preds)))

        marker = " ***" if limb_idx in CRITICAL_LIMBS else ""
        print(f"{limb_idx}: {name:<15} {mean_prob:>10.3f} {std_prob:>8.3f} "
              f"{n_corrections:>12} {n_flips:>8}{marker}")

    print("-" * 60)
    print("*** = Critical limbs (hip_width, shoulder_width, cross-body)")

    # Low confidence frames
    low_conf_mask = np.abs(probs - 0.5) < 0.1  # Within 0.4-0.6
    low_conf_frames = low_conf_mask.any(axis=1).sum()
    print(f"\nFrames with low-confidence limbs (prob 0.4-0.6): {low_conf_frames}/{n_frames}")

    # Hip width stability
    hip_probs = probs[:, 9]  # limb 9 = hip_width
    hip_flips = np.sum(np.abs(np.diff((hip_probs > 0.5).astype(int))))
    print(f"\nHip width (limb 9) - CRITICAL:")
    print(f"  Mean probability: {hip_probs.mean():.3f}")
    print(f"  Std deviation: {hip_probs.std():.3f}")
    print(f"  Z-sign flips: {hip_flips}")
    print(f"  Frames in low-conf zone: {np.sum((hip_probs > 0.4) & (hip_probs < 0.6))}")


def main():
    parser = argparse.ArgumentParser(description="Z-Sign Diagnostic Visualization")
    parser.add_argument("video", type=Path, help="Video file to analyze")
    parser.add_argument("--height", type=float, default=1.78,
                        help="Subject height in meters")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="Maximum frames to process")
    parser.add_argument("--focus-limbs", type=str, default="9,8,12,13",
                        help="Comma-separated limb indices to focus on")
    parser.add_argument("--export-csv", type=Path,
                        help="Export Z-sign data to CSV")
    parser.add_argument("--no-gui", action="store_true",
                        help="Print summary only, no visualization")

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}")
        sys.exit(1)

    focus_limbs = [int(x) for x in args.focus_limbs.split(',')]

    # Run inference
    poses_3d, pof, zsign_info, visibility = run_inference_with_zsign(
        args.video, args.height, args.max_frames
    )

    # Print summary
    print_zsign_summary(zsign_info)

    # Export if requested
    if args.export_csv:
        export_zsign_csv(zsign_info, args.export_csv)

    # Visualize
    if not args.no_gui:
        visualize_zsign_diagnostic(
            poses_3d, pof, zsign_info, visibility,
            focus_limbs=focus_limbs,
            title=args.video.stem
        )


if __name__ == "__main__":
    main()
