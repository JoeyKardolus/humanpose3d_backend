"""Camera-space Part Orientation Fields (POF) for 3D pose reconstruction.

This module provides a camera-space approach to 3D pose estimation from
2D keypoints. Unlike world-space methods, it operates entirely in camera
coordinates without requiring azimuth/elevation prediction.

Key components:
- FacingDirection: Detect person's facing direction from ear/nose visibility
- AnatomicalProportions: Estimate bone lengths from body height
- CameraPOFModel: Neural network to predict POF unit vectors
- CameraPOFInference: Easy-to-use inference wrapper

Basic usage:
    from src.pof import CameraPOFInference

    # Load trained model
    pof = CameraPOFInference("models/checkpoints/best_pof_model.pth")

    # Reconstruct 3D from 2D
    pose_3d = pof.reconstruct_3d(keypoints_2d, visibility, height_m=1.78)

    # Or refine existing 3D pose
    refined = pof.refine_pose(pose_3d, keypoints_2d, visibility)
"""

# Constants
from .constants import (
    COCO_JOINT_NAMES,
    COCO_JOINT_INDICES,
    LIMB_DEFINITIONS,
    LIMB_NAMES,
    NUM_JOINTS,
    NUM_LIMBS,
    KINEMATIC_CHAINS,
    RECONSTRUCTION_ORDER,
    HEIGHT_TO_TORSO_RATIO,
)

# Facing direction detection
from .facing import (
    FacingDirection,
    detect_facing_direction,
    detect_facing_from_pose,
    facing_to_one_hot,
    detect_facing_direction_batch,
    facing_batch_to_one_hot,
    interpret_forward_direction,
)

# Bone length estimation
from .bone_lengths import (
    AnatomicalProportions,
    estimate_bone_lengths_from_height,
    estimate_bone_lengths_array,
    bone_lengths_to_array,
    compute_bone_lengths_from_pose,
)

# Neural network model
from .model import (
    CameraPOFModel,
    create_pof_model,
    load_pof_model,
)

# Dataset and training
from .dataset import (
    CameraPOFDataset,
    create_pof_dataloaders,
    compute_gt_pof_from_3d,
    compute_gt_pof_from_3d_torch,
    normalize_pose_2d,
    compute_limb_features_2d,
)

# Loss functions
from .losses import (
    CameraPOFLoss,
    LeastSquaresPOFLoss,
    pof_cosine_loss,
    pof_angular_error,
    symmetry_loss,
    projection_consistency_loss,
    scale_factor_regularization,
    solved_depth_loss,
    full_pose_loss,
)

# Least-squares solver
from .least_squares import (
    solve_depth_least_squares_pof,
    normalize_2d_for_pof,
    denormalize_pose_3d,
    solve_with_denormalization,
)

# Reconstruction
from .reconstruction import (
    reconstruct_skeleton_from_pof,
    reconstruct_skeleton_batch,
    reconstruct_skeleton_least_squares,
    reconstruct_skeleton_least_squares_batch,
    apply_pof_to_pose,
    apply_pof_to_pose_batch,
    compute_reconstruction_error,
)

# Inference
from .inference import (
    CameraPOFInference,
    create_pof_inference,
)

# Visualization
from .visualization import (
    plot_pof_vectors,
    plot_skeleton_3d,
    plot_reconstruction_comparison,
    plot_pof_error_distribution,
    create_debug_visualization,
)

__all__ = [
    # Constants
    "COCO_JOINT_NAMES",
    "COCO_JOINT_INDICES",
    "LIMB_DEFINITIONS",
    "LIMB_NAMES",
    "NUM_JOINTS",
    "NUM_LIMBS",
    "KINEMATIC_CHAINS",
    "RECONSTRUCTION_ORDER",
    "HEIGHT_TO_TORSO_RATIO",
    # Facing
    "FacingDirection",
    "detect_facing_direction",
    "detect_facing_from_pose",
    "facing_to_one_hot",
    "detect_facing_direction_batch",
    "facing_batch_to_one_hot",
    "interpret_forward_direction",
    # Bone lengths
    "AnatomicalProportions",
    "estimate_bone_lengths_from_height",
    "estimate_bone_lengths_array",
    "bone_lengths_to_array",
    "compute_bone_lengths_from_pose",
    # Model
    "CameraPOFModel",
    "create_pof_model",
    "load_pof_model",
    # Dataset
    "CameraPOFDataset",
    "create_pof_dataloaders",
    "compute_gt_pof_from_3d",
    "compute_gt_pof_from_3d_torch",
    "normalize_pose_2d",
    "compute_limb_features_2d",
    # Loss
    "CameraPOFLoss",
    "LeastSquaresPOFLoss",
    "pof_cosine_loss",
    "pof_angular_error",
    "symmetry_loss",
    "projection_consistency_loss",
    "scale_factor_regularization",
    "solved_depth_loss",
    "full_pose_loss",
    # Least-squares solver
    "solve_depth_least_squares_pof",
    "normalize_2d_for_pof",
    "denormalize_pose_3d",
    "solve_with_denormalization",
    # Reconstruction
    "reconstruct_skeleton_from_pof",
    "reconstruct_skeleton_batch",
    "reconstruct_skeleton_least_squares",
    "reconstruct_skeleton_least_squares_batch",
    "apply_pof_to_pose",
    "apply_pof_to_pose_batch",
    "compute_reconstruction_error",
    # Inference
    "CameraPOFInference",
    "create_pof_inference",
    # Visualization
    "plot_pof_vectors",
    "plot_skeleton_3d",
    "plot_reconstruction_comparison",
    "plot_pof_error_distribution",
    "create_debug_visualization",
]
