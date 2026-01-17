#!/usr/bin/env python3
"""
Explain the camera POSITION prediction approach.

Key insight: Predict camera position (x,y,z) → compute angles from position.
This ensures training and inference use the SAME angle calculation.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['MPLBACKEND'] = 'Agg'

def main():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Training pipeline
    ax1 = axes[0]
    ax1.axis('off')
    ax1.set_title('TRAINING (with AIST++ data)', fontsize=14, fontweight='bold')

    training_text = """
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  AIST++ provides:                                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • Ground truth 3D pose (from mocap)                 │   │
│  │  • Camera calibration → camera position (x,y,z)      │   │
│  │  • Multi-view video                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Run MediaPipe on video → noisy 3D pose           │   │
│  │  2. Align to GT pose (same pelvis center)            │   │
│  │  3. Compute view angles from camera position:        │   │
│  │                                                      │   │
│  │     camera_relative = camera_pos - pelvis_center     │   │
│  │     azimuth, elevation = compute_angles(pose, cam)   │   │
│  │                                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Training Data (NPZ files):                          │   │
│  │  • corrupted: MediaPipe 3D pose (17, 3)              │   │
│  │  • ground_truth: AIST++ pose (17, 3)                 │   │
│  │  • camera_relative: (3,) camera pos relative to pelvis│  │
│  │  • azimuth: 0-360° (computed from camera pos)        │   │
│  │  • elevation: -90 to +90°                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MODEL TRAINING:                                     │   │
│  │                                                      │   │
│  │  Input: MediaPipe pose + visibility                  │   │
│  │         ↓                                            │   │
│  │  CameraPositionPredictor → pred_camera_pos (3,)      │   │
│  │         ↓                                            │   │
│  │  compute_angles(pose, pred_camera_pos)               │   │
│  │         ↓                                            │   │
│  │  Loss = MSE(pred_camera_pos, GT_camera_pos)          │   │
│  │       + MSE(corrected_depth, GT_depth)               │   │
│  │                                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
    """
    ax1.text(0.5, 0.5, training_text, fontsize=9, fontfamily='monospace',
             ha='center', va='center', transform=ax1.transAxes)

    # Right: Inference pipeline
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('INFERENCE (real-world video)', fontsize=14, fontweight='bold')

    inference_text = """
┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Real-world video (NO camera calibration!):                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • Single monocular video                            │   │
│  │  • Unknown camera position                           │   │
│  │  • Unknown camera intrinsics                         │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Run MediaPipe → noisy 3D pose (17, 3)            │   │
│  │  2. Run trained model:                               │   │
│  │                                                      │   │
│  │     pred_camera_pos = model.predict_camera(pose)     │   │
│  │                  ↓                                   │   │
│  │     azimuth, elevation = compute_angles(             │   │
│  │         pose, pred_camera_pos  ← SAME FUNCTION!      │   │
│  │     )                                                │   │
│  │                  ↓                                   │   │
│  │     delta_z = model.predict_depth(pose, angles)      │   │
│  │                  ↓                                   │   │
│  │     corrected_pose = pose + [0, 0, delta_z]          │   │
│  │                                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Output: Corrected 3D pose with better depth         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ═══════════════════════════════════════════════════════════│
│                                                              │
│  KEY INSIGHT:                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Training and inference use the SAME function:       │   │
│  │                                                      │   │
│  │    compute_angles_from_camera_position(pose, cam)    │   │
│  │                                                      │   │
│  │  This ensures consistency! The model learns to       │   │
│  │  predict camera position, and the geometric          │   │
│  │  angle calculation is deterministic (not learned).   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
    """
    ax2.text(0.5, 0.5, inference_text, fontsize=9, fontfamily='monospace',
             ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig('camera_position_prediction_explained.png', dpi=150,
                facecolor='white', edgecolor='none', bbox_inches='tight')
    print("Saved: camera_position_prediction_explained.png")

    # Print summary
    print("\n" + "="*70)
    print("CAMERA POSITION PREDICTION - How It Works")
    print("="*70)
    print("""
TRAINING:
  1. AIST++ provides GT camera position from calibration
  2. We compute azimuth/elevation from camera position using geometry
  3. Model learns to predict camera position from pose
  4. Loss supervises both camera position AND depth correction

INFERENCE:
  1. MediaPipe extracts noisy 3D pose from video
  2. Model predicts camera position (x, y, z) from pose
  3. Same geometric function computes azimuth/elevation
  4. Angles condition the depth correction head

WHY THIS APPROACH?
  - Consistent: Same angle computation for training & inference
  - Interpretable: Camera position is a physical quantity
  - Learnable: Position encodes both direction AND distance
  - Robust: Network learns from diverse AIST++ viewpoints

WHAT THE NETWORK LEARNS:
  - Shoulder size ratio → left/right camera direction
  - Limb foreshortening → front/back camera direction
  - Joint visibility patterns → camera elevation
  - Perspective distortion → camera distance
""")

if __name__ == '__main__':
    main()
