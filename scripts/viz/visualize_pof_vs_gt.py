#!/usr/bin/env python3
"""Interactive visualization comparing POF reconstruction to ground truth on training samples."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from glob import glob
from collections import defaultdict
import re
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pof.inference import CameraPOFInference
from src.pof.reconstruction import reconstruct_skeleton_least_squares
from src.pof.dataset import (
    normalize_pose_2d, compute_gt_pof_from_3d, world_to_camera_space
)
from src.pof.constants import NUM_LIMBS, LIMB_DEFINITIONS

# COCO-17 skeleton connections
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

JOINT_NAMES = [
    'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
    'L_shldr', 'R_shldr', 'L_elbow', 'R_elbow',
    'L_wrist', 'R_wrist', 'L_hip', 'R_hip',
    'L_knee', 'R_knee', 'L_ankle', 'R_ankle'
]


def normalize_3d(pose_3d):
    """Normalize 3D pose to pelvis-centered, unit-torso."""
    pelvis = (pose_3d[11] + pose_3d[12]) / 2
    centered = pose_3d - pelvis
    l_torso = np.linalg.norm(pose_3d[5] - pose_3d[11])
    r_torso = np.linalg.norm(pose_3d[6] - pose_3d[12])
    scale = max((l_torso + r_torso) / 2, 1e-6)
    return centered / scale


def load_sequence_data(seq_name, sequences, max_frames=100):
    """Load a sequence and run POF reconstruction."""
    files = sorted(sequences[seq_name])[:max_frames]

    pose_2d_list = []
    gt_3d_camera_list = []
    visibility_list = []

    for f in files:
        sample = np.load(f)
        pose_2d_list.append(sample['pose_2d'])
        gt_world = sample['ground_truth']
        camera_R = sample['camera_R']
        gt_camera = world_to_camera_space(gt_world, camera_R)
        gt_3d_camera_list.append(gt_camera)
        visibility_list.append(sample['visibility'])

    pose_2d = np.array(pose_2d_list)
    gt_3d_camera = np.array(gt_3d_camera_list)
    visibility = np.array(visibility_list)

    # Normalize GT
    gt_3d_norm = np.array([normalize_3d(gt) for gt in gt_3d_camera])

    return pose_2d, gt_3d_norm, visibility


def compute_frame_errors(recon, gt):
    """Compute per-frame MPJPE and POF error."""
    n_frames = len(recon)
    mpjpe = np.zeros(n_frames)
    pof_err = np.zeros(n_frames)

    for i in range(n_frames):
        mpjpe[i] = np.mean(np.linalg.norm(recon[i] - gt[i], axis=1))

        # POF error
        errs = []
        for parent, child in LIMB_DEFINITIONS:
            pred_vec = recon[i, child] - recon[i, parent]
            gt_vec = gt[i, child] - gt[i, parent]
            pred_norm = np.linalg.norm(pred_vec)
            gt_norm = np.linalg.norm(gt_vec)
            if pred_norm > 1e-6 and gt_norm > 1e-6:
                cos = np.clip(np.dot(pred_vec/pred_norm, gt_vec/gt_norm), -1, 1)
                errs.append(np.degrees(np.arccos(cos)))
        pof_err[i] = np.mean(errs) if errs else 0

    return mpjpe, pof_err


def visualize_comparison(recon_3d, gt_3d, pose_2d, mpjpe, pof_err, seq_name):
    """Interactive visualization comparing reconstruction to GT."""
    n_frames = len(recon_3d)

    fig = plt.figure(figsize=(16, 8))

    # 3D subplot for GT
    ax_gt = fig.add_subplot(131, projection='3d')
    ax_gt.set_title('Ground Truth')

    # 3D subplot for reconstruction
    ax_recon = fig.add_subplot(132, projection='3d')
    ax_recon.set_title('POF Reconstruction')

    # 2D input
    ax_2d = fig.add_subplot(133)
    ax_2d.set_title('2D Input')

    plt.subplots_adjust(bottom=0.25)

    state = {'frame_idx': 0, 'playing': False}

    def draw_skeleton_3d(ax, pose, color, alpha=1.0):
        """Draw 3D skeleton."""
        ax.clear()

        # Draw bones
        for i, j in COCO_CONNECTIONS:
            ax.plot([pose[i, 0], pose[j, 0]],
                   [pose[i, 2], pose[j, 2]],
                   [pose[i, 1], pose[j, 1]],
                   color=color, linewidth=2, alpha=alpha)

        # Draw joints
        ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1],
                  c=color, s=30, edgecolor='black', linewidth=0.5)

        # Set consistent axis limits
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Z (depth)')
        ax.set_zlabel('Y (up)')
        ax.view_init(elev=15, azim=-60)

    def draw_skeleton_2d(ax, pose_2d):
        """Draw 2D skeleton."""
        ax.clear()

        # Draw bones
        for i, j in COCO_CONNECTIONS:
            ax.plot([pose_2d[i, 0], pose_2d[j, 0]],
                   [pose_2d[i, 1], pose_2d[j, 1]],
                   'b-', linewidth=2)

        # Draw joints
        ax.scatter(pose_2d[:, 0], pose_2d[:, 1], c='blue', s=30, zorder=5)

        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Flip Y for image coords
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def update_plot(frame_idx):
        draw_skeleton_3d(ax_gt, gt_3d[frame_idx], 'green')
        ax_gt.set_title(f'Ground Truth')

        draw_skeleton_3d(ax_recon, recon_3d[frame_idx], 'blue')
        ax_recon.set_title(f'POF Reconstruction\nMPJPE: {mpjpe[frame_idx]*500:.0f}mm, POF: {pof_err[frame_idx]:.1f}°')

        draw_skeleton_2d(ax_2d, pose_2d[frame_idx])
        ax_2d.set_title(f'2D Input - Frame {frame_idx+1}/{n_frames}')

        fig.suptitle(f'{seq_name}\nOverall: MPJPE={mpjpe.mean()*500:.0f}mm, POF={pof_err.mean():.1f}°', fontsize=12)
        fig.canvas.draw_idle()

    # Frame slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)

    def on_slider(val):
        state['frame_idx'] = int(val)
        update_plot(state['frame_idx'])

    slider.on_changed(on_slider)

    # Play/Pause button
    ax_play = plt.axes([0.4, 0.02, 0.1, 0.04])
    btn_play = Button(ax_play, 'Play')

    # Prev/Next buttons
    ax_prev = plt.axes([0.25, 0.02, 0.1, 0.04])
    btn_prev = Button(ax_prev, '< Prev')
    ax_next = plt.axes([0.55, 0.02, 0.1, 0.04])
    btn_next = Button(ax_next, 'Next >')

    def on_play(event):
        state['playing'] = not state['playing']
        btn_play.label.set_text('Pause' if state['playing'] else 'Play')

    def on_prev(event):
        state['frame_idx'] = max(0, state['frame_idx'] - 1)
        slider.set_val(state['frame_idx'])

    def on_next(event):
        state['frame_idx'] = min(n_frames - 1, state['frame_idx'] + 1)
        slider.set_val(state['frame_idx'])

    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    # Animation timer
    def animate(event):
        if state['playing']:
            state['frame_idx'] = (state['frame_idx'] + 1) % n_frames
            slider.set_val(state['frame_idx'])

    timer = fig.canvas.new_timer(interval=100)  # 10 fps
    timer.add_callback(animate, None)
    timer.start()

    update_plot(0)
    plt.show()


def save_comparison_frames(recon_3d, gt_3d, pose_2d, mpjpe, pof_err, seq_name, output_dir='data/output/pof_comparison'):
    """Save comparison frames as images (for headless environments)."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    n_frames = len(recon_3d)

    # Save every 10th frame
    for frame_idx in range(0, n_frames, 10):
        fig = plt.figure(figsize=(16, 6))

        # GT
        ax_gt = fig.add_subplot(131, projection='3d')
        for i, j in COCO_CONNECTIONS:
            ax_gt.plot([gt_3d[frame_idx, i, 0], gt_3d[frame_idx, j, 0]],
                      [gt_3d[frame_idx, i, 2], gt_3d[frame_idx, j, 2]],
                      [gt_3d[frame_idx, i, 1], gt_3d[frame_idx, j, 1]],
                      'g-', linewidth=2)
        ax_gt.scatter(gt_3d[frame_idx, :, 0], gt_3d[frame_idx, :, 2], gt_3d[frame_idx, :, 1], c='green', s=30)
        ax_gt.set_xlim(-1.5, 1.5); ax_gt.set_ylim(-1.5, 1.5); ax_gt.set_zlim(-1.5, 1.5)
        ax_gt.set_title('Ground Truth')
        ax_gt.view_init(elev=15, azim=-60)

        # Reconstruction
        ax_recon = fig.add_subplot(132, projection='3d')
        for i, j in COCO_CONNECTIONS:
            ax_recon.plot([recon_3d[frame_idx, i, 0], recon_3d[frame_idx, j, 0]],
                         [recon_3d[frame_idx, i, 2], recon_3d[frame_idx, j, 2]],
                         [recon_3d[frame_idx, i, 1], recon_3d[frame_idx, j, 1]],
                         'b-', linewidth=2)
        ax_recon.scatter(recon_3d[frame_idx, :, 0], recon_3d[frame_idx, :, 2], recon_3d[frame_idx, :, 1], c='blue', s=30)
        ax_recon.set_xlim(-1.5, 1.5); ax_recon.set_ylim(-1.5, 1.5); ax_recon.set_zlim(-1.5, 1.5)
        ax_recon.set_title(f'POF Recon\nMPJPE: {mpjpe[frame_idx]*500:.0f}mm, POF: {pof_err[frame_idx]:.1f}°')
        ax_recon.view_init(elev=15, azim=-60)

        # 2D
        ax_2d = fig.add_subplot(133)
        for i, j in COCO_CONNECTIONS:
            ax_2d.plot([pose_2d[frame_idx, i, 0], pose_2d[frame_idx, j, 0]],
                      [pose_2d[frame_idx, i, 1], pose_2d[frame_idx, j, 1]], 'b-', linewidth=2)
        ax_2d.scatter(pose_2d[frame_idx, :, 0], pose_2d[frame_idx, :, 1], c='blue', s=30)
        ax_2d.set_xlim(0, 1); ax_2d.set_ylim(1, 0)
        ax_2d.set_title(f'2D Input - Frame {frame_idx+1}/{n_frames}')

        fig.suptitle(f'{seq_name} | Overall: MPJPE={mpjpe.mean()*500:.0f}mm, POF={pof_err.mean():.1f}°')

        out_path = f'{output_dir}/{seq_name}_frame{frame_idx:04d}.png'
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {out_path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize POF reconstruction vs ground truth')
    parser.add_argument('--save', action='store_true', help='Save frames as images instead of interactive display')
    parser.add_argument('--num-sequences', type=int, default=5, help='Number of sequences to process (for --save)')
    args = parser.parse_args()

    print("Loading training data...")
    all_files = sorted(glob('data/training/aistpp_rtmpose/*.npz'))

    def get_sequence(path):
        stem = path.split('/')[-1].replace('.npz', '')
        match = re.match(r'^(.+)_f\d+', stem)
        return match.group(1) if match else stem

    sequences = defaultdict(list)
    for f in all_files:
        sequences[get_sequence(f)].append(f)

    # Get val sequences
    all_seq_names = sorted(sequences.keys())
    rng = random.Random(42)
    rng.shuffle(all_seq_names)
    val_seqs = all_seq_names[:int(len(all_seq_names) * 0.1)]

    # Filter for sequences with consistent frame spacing
    def get_frame_idx(p):
        m = re.search(r'_f(\d+)\.npz', p)
        return int(m.group(1)) if m else 0

    good_seqs = []
    for seq in val_seqs:
        files = sorted(sequences[seq])
        if len(files) >= 50:
            idxs = [get_frame_idx(f) for f in files]
            gaps = [idxs[i+1] - idxs[i] for i in range(len(idxs)-1)]
            if len(set(gaps)) == 1:
                good_seqs.append(seq)

    print(f"Found {len(good_seqs)} validation sequences with consistent frame spacing")

    # Load POF model
    print("Loading POF model...")
    pof_inf = CameraPOFInference('models/checkpoints/best_pof_semgcn-temporal_model.pth', verbose=True)

    if args.save:
        # Save mode: process multiple sequences and save frames as images
        for seq_idx in range(min(args.num_sequences, len(good_seqs))):
            seq_name = good_seqs[seq_idx]
            print(f"\nProcessing sequence {seq_idx+1}/{args.num_sequences}: {seq_name}")

            pose_2d, gt_3d, visibility = load_sequence_data(seq_name, sequences)
            pof_pred = pof_inf.predict_pof(pose_2d, visibility)
            recon_3d = reconstruct_skeleton_least_squares(
                pof_pred, pose_2d, None, pelvis_depth=0.0, denormalize=False
            )
            mpjpe, pof_err = compute_frame_errors(recon_3d, gt_3d)
            print(f"  MPJPE: {mpjpe.mean()*500:.0f}mm, POF Error: {pof_err.mean():.1f}°")

            save_comparison_frames(recon_3d, gt_3d, pose_2d, mpjpe, pof_err, seq_name)

        print(f"\nDone! Images saved to data/output/pof_comparison/")
    else:
        # Interactive mode
        seq_idx = 0

        while True:
            seq_name = good_seqs[seq_idx % len(good_seqs)]
            print(f"\nLoading sequence {seq_idx+1}/{len(good_seqs)}: {seq_name}")

            # Load data
            pose_2d, gt_3d, visibility = load_sequence_data(seq_name, sequences)

            # Run POF reconstruction
            print("Running POF reconstruction...")
            pof_pred = pof_inf.predict_pof(pose_2d, visibility)
            recon_3d = reconstruct_skeleton_least_squares(
                pof_pred, pose_2d, None, pelvis_depth=0.0, denormalize=False
            )

            # Compute errors
            mpjpe, pof_err = compute_frame_errors(recon_3d, gt_3d)
            print(f"MPJPE: {mpjpe.mean()*500:.0f}mm, POF Error: {pof_err.mean():.1f}°")

            # Visualize
            visualize_comparison(recon_3d, gt_3d, pose_2d, mpjpe, pof_err, seq_name)

            # Ask to continue
            choice = input("\n[n]ext sequence, [p]rev, [q]uit, or enter sequence number: ").strip().lower()
            if choice == 'q':
                break
            elif choice == 'p':
                seq_idx = max(0, seq_idx - 1)
            elif choice == 'n' or choice == '':
                seq_idx += 1
            elif choice.isdigit():
                seq_idx = int(choice) - 1


if __name__ == "__main__":
    main()
