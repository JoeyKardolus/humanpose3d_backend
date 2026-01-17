#!/usr/bin/env python3
"""
Explain and visualize how camera angle prediction works.

The key insight (from ElePose, CVPR 2022):
- The 2D/3D appearance of a pose ENCODES the camera viewing angle
- A neural network can learn to "decode" this from pose features

This script creates a visualization showing:
1. How the same pose looks different from different angles
2. Why this means we can predict camera angle from pose
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.environ['MPLBACKEND'] = 'Agg'

# COCO 17 joint connections for visualization
CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

def create_standing_pose():
    """Create a simple standing pose in 3D (COCO 17 format)."""
    # Approximate standing pose centered at origin
    pose = np.array([
        [0.0, 1.7, 0.0],    # 0: nose
        [-0.05, 1.72, 0.0], # 1: left_eye
        [0.05, 1.72, 0.0],  # 2: right_eye
        [-0.1, 1.7, 0.0],   # 3: left_ear
        [0.1, 1.7, 0.0],    # 4: right_ear
        [-0.2, 1.5, 0.0],   # 5: left_shoulder
        [0.2, 1.5, 0.0],    # 6: right_shoulder
        [-0.35, 1.2, 0.0],  # 7: left_elbow
        [0.35, 1.2, 0.0],   # 8: right_elbow
        [-0.35, 0.9, 0.0],  # 9: left_wrist
        [0.35, 0.9, 0.0],   # 10: right_wrist
        [-0.1, 1.0, 0.0],   # 11: left_hip
        [0.1, 1.0, 0.0],    # 12: right_hip
        [-0.1, 0.5, 0.0],   # 13: left_knee
        [0.1, 0.5, 0.0],    # 14: right_knee
        [-0.1, 0.0, 0.0],   # 15: left_ankle
        [0.1, 0.0, 0.0],    # 16: right_ankle
    ])
    return pose

def project_pose(pose_3d, azimuth_deg, elevation_deg=15, distance=3.0):
    """
    Project 3D pose to 2D from a given camera angle.

    Returns 2D coordinates as if viewed from the camera.
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    # Camera position on a sphere around the subject
    cam_x = distance * np.cos(el) * np.sin(az)
    cam_y = distance * np.sin(el) + 1.0  # Offset to look at center of body
    cam_z = distance * np.cos(el) * np.cos(az)
    cam_pos = np.array([cam_x, cam_y, cam_z])

    # Camera looks at center of body
    target = np.array([0, 1.0, 0])

    # Build camera coordinate system
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0, 1, 0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)

    # Project each joint
    pose_2d = []
    for joint in pose_3d:
        # Vector from camera to joint
        to_joint = joint - cam_pos

        # Project onto camera plane
        x = np.dot(to_joint, right)
        y = np.dot(to_joint, up)
        z = np.dot(to_joint, forward)  # Depth for perspective

        # Perspective divide (simple)
        if z > 0.1:
            x = x / z
            y = y / z

        pose_2d.append([x, y])

    return np.array(pose_2d)

def draw_pose_2d(ax, pose_2d, title, color='blue'):
    """Draw 2D pose on matplotlib axes."""
    ax.scatter(pose_2d[:, 0], pose_2d[:, 1], c=color, s=30, zorder=3)

    for i, j in CONNECTIONS:
        ax.plot([pose_2d[i, 0], pose_2d[j, 0]],
                [pose_2d[i, 1], pose_2d[j, 1]],
                c=color, linewidth=1.5, zorder=2)

    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal')
    ax.axis('off')

def main():
    print("Creating camera angle prediction explanation...")

    pose_3d = create_standing_pose()

    # Create figure with explanation
    fig = plt.figure(figsize=(16, 12))

    # Top: Concept diagram
    ax_concept = fig.add_axes([0.05, 0.65, 0.9, 0.3])
    ax_concept.axis('off')

    concept_text = """
    CAMERA ANGLE PREDICTION: How it Works
    ═══════════════════════════════════════════════════════════════════════════════

    KEY INSIGHT: The same 3D pose looks DIFFERENT from different camera angles!

    Training (with AIST++ data):
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Ground Truth Camera Position  ────►  Compute azimuth (0-360°) & elevation  │
    │  (from AIST++ calibration)                                                  │
    └─────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  MediaPipe 3D Pose  ──►  Transformer  ──►  Predict angles  ──►  Loss with GT │
    │  (17 joints × 3D)       (cross-joint)    (azimuth, elev)                    │
    └─────────────────────────────────────────────────────────────────────────────┘

    Inference (real-world video):
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  MediaPipe 3D Pose  ──►  Transformer  ──►  Predict angles  ──►  Use predicted │
    │  (no camera info!)      (cross-joint)    (learned patterns)   for depth fix  │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
    ax_concept.text(0.5, 0.5, concept_text, fontsize=10, fontfamily='monospace',
                    ha='center', va='center', transform=ax_concept.transAxes)

    # Bottom: Visual demonstration - same pose from 4 angles
    angles = [
        (0, "0° (Front)"),
        (90, "90° (Right Profile)"),
        (180, "180° (Back)"),
        (270, "270° (Left Profile)"),
    ]

    for idx, (az, label) in enumerate(angles):
        ax = fig.add_subplot(2, 4, 5 + idx)
        pose_2d = project_pose(pose_3d, az, elevation_deg=15)
        draw_pose_2d(ax, pose_2d, label)

    # Add annotation below the 4 views
    fig.text(0.5, 0.02,
             "Same 3D pose → Different 2D appearances → Network learns to decode viewing angle from pose patterns",
             ha='center', fontsize=12, style='italic')

    fig.text(0.5, 0.35,
             "Notice: Shoulder widths, limb foreshortening, and joint positions all change with viewing angle.\n"
             "The neural network learns these patterns to predict camera angle without calibration data!",
             ha='center', fontsize=11)

    plt.savefig('camera_prediction_explained.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: camera_prediction_explained.png")

    # Also create a more detailed architecture diagram
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.axis('off')

    architecture_text = """
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                    DEPTH REFINEMENT MODEL ARCHITECTURE                                ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                       ║
    ║   INPUT                                                                               ║
    ║   ─────                                                                               ║
    ║   • pose: (batch, 17, 3) - MediaPipe 3D joint positions                              ║
    ║   • visibility: (batch, 17) - per-joint confidence scores                            ║
    ║   • azimuth: (batch,) - GT angle 0-360° [TRAINING ONLY]                              ║
    ║   • elevation: (batch,) - GT angle -90 to +90° [TRAINING ONLY]                       ║
    ║                                                                                       ║
    ║   ┌──────────────────────────────────────────────────────────────────────────────┐   ║
    ║   │                          JOINT ENCODER                                        │   ║
    ║   │   pose + visibility → MLP → (batch, 17, 64) joint features                   │   ║
    ║   │   + Positional encoding (joint identity: nose=0, left_eye=1, ...)            │   ║
    ║   └──────────────────────────────────────────────────────────────────────────────┘   ║
    ║                                       │                                               ║
    ║                                       ▼                                               ║
    ║   ┌──────────────────────────────────────────────────────────────────────────────┐   ║
    ║   │                    CAMERA ANGLE PREDICTOR (NEW!)                              │   ║
    ║   │   joint_features → flatten → MLP → (azimuth, elevation)                      │   ║
    ║   │                                                                               │   ║
    ║   │   Training: Compare with GT → angle_loss = MSE(pred, GT)                     │   ║
    ║   │   Inference: Use predicted angles (no GT available)                          │   ║
    ║   └──────────────────────────────────────────────────────────────────────────────┘   ║
    ║                                       │                                               ║
    ║                                       ▼                                               ║
    ║   ┌──────────────────────────────────────────────────────────────────────────────┐   ║
    ║   │                     CROSS-JOINT TRANSFORMER                                   │   ║
    ║   │   4-layer transformer encoder with 4 attention heads                         │   ║
    ║   │   Learns: "which joints inform depth of other joints?"                       │   ║
    ║   │   Example: wrist depth inferred from elbow + shoulder configuration          │   ║
    ║   └──────────────────────────────────────────────────────────────────────────────┘   ║
    ║                                       │                                               ║
    ║                                       ▼                                               ║
    ║   ┌──────────────────────────────────────────────────────────────────────────────┐   ║
    ║   │                      VIEW ANGLE ENCODER                                       │   ║
    ║   │   azimuth (0-360°) + elevation (-90 to +90°)                                 │   ║
    ║   │   → Fourier features (sin/cos at multiple frequencies)                       │   ║
    ║   │   → MLP → (batch, 64) view embedding                                         │   ║
    ║   │                                                                               │   ║
    ║   │   Why Fourier? Angles are PERIODIC (0° ≈ 360°), standard MLPs don't capture  │   ║
    ║   └──────────────────────────────────────────────────────────────────────────────┘   ║
    ║                                       │                                               ║
    ║                                       ▼                                               ║
    ║   ┌──────────────────────────────────────────────────────────────────────────────┐   ║
    ║   │                       DEPTH CORRECTION HEAD                                   │   ║
    ║   │   [attended_features || view_embedding] → MLP → delta_z (batch, 17)          │   ║
    ║   │                                                                               │   ║
    ║   │   Output: Per-joint depth correction to ADD to MediaPipe z-coordinate        │   ║
    ║   │   corrected_z = mediapipe_z + delta_z                                        │   ║
    ║   └──────────────────────────────────────────────────────────────────────────────┘   ║
    ║                                                                                       ║
    ║   OUTPUT                                                                              ║
    ║   ──────                                                                              ║
    ║   • delta_z: (batch, 17) - depth corrections per joint                               ║
    ║   • confidence: (batch, 17) - how confident is each correction                       ║
    ║   • pred_azimuth: (batch,) - predicted camera azimuth                                ║
    ║   • pred_elevation: (batch,) - predicted camera elevation                            ║
    ║                                                                                       ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║   TRAINING LOSS                                                                       ║
    ║   ─────────────                                                                       ║
    ║   L_total = L_depth + λ_bone × L_bone + λ_sym × L_symmetry + λ_angle × L_angle       ║
    ║                                                                                       ║
    ║   • L_depth: MSE between corrected pose and ground truth                             ║
    ║   • L_bone: Bone length consistency (anatomical constraint)                          ║
    ║   • L_symmetry: Left-right symmetry constraint                                       ║
    ║   • L_angle: MSE between predicted and GT camera angles (NEW!)                       ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """

    ax2.text(0.5, 0.5, architecture_text, fontsize=9, fontfamily='monospace',
             ha='center', va='center', transform=ax2.transAxes)

    plt.savefig('model_architecture_explained.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: model_architecture_explained.png")

    print("\n" + "="*70)
    print("SUMMARY: Camera Angle Prediction")
    print("="*70)
    print("""
The network learns to predict camera angle from the pose itself because:

1. PERSPECTIVE DISTORTION: When camera is to the right, the right shoulder
   appears larger and closer than the left shoulder.

2. FORESHORTENING: Arms/legs pointing toward camera appear shorter than
   arms/legs perpendicular to the camera.

3. OCCLUSION PATTERNS: Certain joints are more likely to be occluded from
   certain angles (e.g., back view can't see face joints well).

4. DEPTH CUES: MediaPipe's z-coordinates have angle-specific error patterns
   that the network learns to recognize.

At training: We use GT camera angles from AIST++ to supervise the predictor.
At inference: The network predicts angles from pose → uses them for depth correction.

This is based on ElePose (CVPR 2022) which showed this approach works well!
""")

if __name__ == '__main__':
    main()
