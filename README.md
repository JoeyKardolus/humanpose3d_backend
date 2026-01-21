HumanPose3D Models Branch
=========================

This is a separate git branch that stores pre-trained model files for HumanPose3D.
The models are kept in a separate branch to avoid bloating the main repository
with large binary files (~125 MB total).

This directory stores models for HumanPose3D.

Directory Structure:
  models/           - Neural network model files
  models/checkpoints/  - PyTorch checkpoint files

Models (downloaded automatically on first run):
  - best_depth_model.pth      (32 MB) - Depth refinement model
  - best_joint_model.pth      (11 MB) - Joint constraint model
  - best_main_refiner.pth     (15 MB) - Main refiner fusion model
  - pose_landmarker_heavy.task (30 MB) - MediaPipe pose detection
  - GRU.h5                    (37 MB) - Pose2Sim LSTM augmentation

You can safely delete this directory to reset the application.
Models will be re-downloaded on next run.

Usage:
  To use pre-downloaded models, clone this branch into ~/.humanpose3d/:
    git clone -b models https://github.com/yourusername/HumanPose3D.git ~/.humanpose3d

For more information, visit:
  https://github.com/yourusername/HumanPose3D
