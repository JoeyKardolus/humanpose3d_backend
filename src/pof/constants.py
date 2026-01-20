"""Constants for camera-space POF module.

Defines joint indices, limb definitions, and kinematic chains for
Part Orientation Fields (POF) based 3D pose reconstruction.
"""

from typing import Dict, List, Tuple

# MediaPipe COCO-17 joint names (used in this module)
COCO_JOINT_NAMES: List[str] = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Reverse mapping: name -> index
COCO_JOINT_INDICES: Dict[str, int] = {
    name: i for i, name in enumerate(COCO_JOINT_NAMES)
}

# Number of joints in COCO-17 format
NUM_JOINTS: int = 17

# 14 limb definitions: (parent_idx, child_idx) using COCO-17 indices
# POF unit vector points from parent to child
LIMB_DEFINITIONS: List[Tuple[int, int]] = [
    (5, 7),    # 0: L shoulder -> L elbow (L upper arm)
    (7, 9),    # 1: L elbow -> L wrist (L forearm)
    (6, 8),    # 2: R shoulder -> R elbow (R upper arm)
    (8, 10),   # 3: R elbow -> R wrist (R forearm)
    (11, 13),  # 4: L hip -> L knee (L thigh)
    (13, 15),  # 5: L knee -> L ankle (L shin)
    (12, 14),  # 6: R hip -> R knee (R thigh)
    (14, 16),  # 7: R knee -> R ankle (R shin)
    (5, 6),    # 8: L shoulder -> R shoulder (shoulder width)
    (11, 12),  # 9: L hip -> R hip (hip width)
    (5, 11),   # 10: L shoulder -> L hip (L torso)
    (6, 12),   # 11: R shoulder -> R hip (R torso)
    (5, 12),   # 12: L shoulder -> R hip (L cross-body diagonal)
    (6, 11),   # 13: R shoulder -> L hip (R cross-body diagonal)
]

# Human-readable limb names
LIMB_NAMES: List[str] = [
    "L_upper_arm",    # 0
    "L_forearm",      # 1
    "R_upper_arm",    # 2
    "R_forearm",      # 3
    "L_thigh",        # 4
    "L_shin",         # 5
    "R_thigh",        # 6
    "R_shin",         # 7
    "shoulder_width", # 8
    "hip_width",      # 9
    "L_torso",        # 10
    "R_torso",        # 11
    "L_cross",        # 12
    "R_cross",        # 13
]

# Number of limbs
NUM_LIMBS: int = 14

# Left/right limb swap pairs for augmentation (indices into LIMB_DEFINITIONS)
LIMB_SWAP_PAIRS: List[Tuple[int, int]] = [
    (0, 2),   # L/R upper arm
    (1, 3),   # L/R forearm
    (4, 6),   # L/R thigh
    (5, 7),   # L/R shin
    (10, 11), # L/R torso
    (12, 13), # L/R cross-body
]

# Left/right joint swap pairs for augmentation (COCO-17 indices)
JOINT_SWAP_PAIRS: List[Tuple[int, int]] = [
    (1, 2),   # L/R eye
    (3, 4),   # L/R ear
    (5, 6),   # L/R shoulder
    (7, 8),   # L/R elbow
    (9, 10),  # L/R wrist
    (11, 12), # L/R hip
    (13, 14), # L/R knee
    (15, 16), # L/R ankle
]

# Kinematic chain hierarchy: maps child joint -> parent joint
# Used for forward kinematics reconstruction from pelvis (root)
# Pelvis is the midpoint of hips (11, 12) and serves as root
KINEMATIC_CHAINS: Dict[int, int] = {
    # Torso chain (shoulders from hips via torso)
    5: 11,   # L_shoulder <- L_hip (via L_torso limb)
    6: 12,   # R_shoulder <- R_hip (via R_torso limb)
    # Left arm chain
    7: 5,    # L_elbow <- L_shoulder
    9: 7,    # L_wrist <- L_elbow
    # Right arm chain
    8: 6,    # R_elbow <- R_shoulder
    10: 8,   # R_wrist <- R_elbow
    # Left leg chain
    13: 11,  # L_knee <- L_hip
    15: 13,  # L_ankle <- L_knee
    # Right leg chain
    14: 12,  # R_knee <- R_hip
    16: 14,  # R_ankle <- R_knee
}

# Topologically sorted reconstruction order (from root to leaves)
# Hips (11, 12) are initialized first as root, then reconstructed in this order
RECONSTRUCTION_ORDER: List[int] = [
    # Start from hips (root)
    11, 12,  # Hips (initialized directly)
    # Torso (shoulders from hips)
    5, 6,    # Shoulders
    # Arms
    7, 8,    # Elbows
    9, 10,   # Wrists
    # Legs
    13, 14,  # Knees
    15, 16,  # Ankles
]

# Mapping from joint index to the limb index used for reconstruction
# This tells us which limb POF to use when reconstructing each joint
JOINT_TO_LIMB: Dict[int, int] = {
    5: 10,   # L_shoulder uses L_torso (limb 10)
    6: 11,   # R_shoulder uses R_torso (limb 11)
    7: 0,    # L_elbow uses L_upper_arm (limb 0)
    9: 1,    # L_wrist uses L_forearm (limb 1)
    8: 2,    # R_elbow uses R_upper_arm (limb 2)
    10: 3,   # R_wrist uses R_forearm (limb 3)
    13: 4,   # L_knee uses L_thigh (limb 4)
    15: 5,   # L_ankle uses L_shin (limb 5)
    14: 6,   # R_knee uses R_thigh (limb 6)
    16: 7,   # R_ankle uses R_shin (limb 7)
}

# Mapping from limb index to bone name (for bone_lengths.py)
LIMB_TO_BONE: Dict[int, str] = {
    0: "upper_arm",      # L_upper_arm
    1: "forearm",        # L_forearm
    2: "upper_arm",      # R_upper_arm (same as L)
    3: "forearm",        # R_forearm (same as L)
    4: "thigh",          # L_thigh
    5: "shin",           # L_shin
    6: "thigh",          # R_thigh (same as L)
    7: "shin",           # R_shin (same as L)
    8: "shoulder_width", # shoulder width
    9: "hip_width",      # hip width
    10: "torso_side",    # L_torso
    11: "torso_side",    # R_torso (same as L)
    12: "cross_torso",   # L cross-body diagonal
    13: "cross_torso",   # R cross-body diagonal (same as L)
}

# Indices for facing direction detection
NOSE_IDX: int = 0
LEFT_EAR_IDX: int = 3
RIGHT_EAR_IDX: int = 4
LEFT_SHOULDER_IDX: int = 5
RIGHT_SHOULDER_IDX: int = 6
LEFT_HIP_IDX: int = 11
RIGHT_HIP_IDX: int = 12

# Human body proportions for metric scale recovery
# Torso = shoulder-to-hip distance (average of left and right sides)
# Height / Torso ratio is approximately 3.4 for adults
# This allows recovering true metric scale from known subject height
HEIGHT_TO_TORSO_RATIO: float = 3.4
