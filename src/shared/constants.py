"""Shared constants for COCO skeleton and visualization.

Contains skeleton topology and color definitions used across
visualization scripts.
"""

# COCO-17 skeleton connections (parent_idx, child_idx)
COCO_SKELETON_CONNECTIONS = [
    # Head
    (0, 1),   # nose -> L_eye
    (0, 2),   # nose -> R_eye
    (1, 3),   # L_eye -> L_ear
    (2, 4),   # R_eye -> R_ear
    # Shoulders
    (5, 6),   # L_shoulder -> R_shoulder
    # Left arm
    (5, 7),   # L_shoulder -> L_elbow
    (7, 9),   # L_elbow -> L_wrist
    # Right arm
    (6, 8),   # R_shoulder -> R_elbow
    (8, 10),  # R_elbow -> R_wrist
    # Torso
    (5, 11),  # L_shoulder -> L_hip
    (6, 12),  # R_shoulder -> R_hip
    (11, 12), # L_hip -> R_hip
    # Left leg
    (11, 13), # L_hip -> L_knee
    (13, 15), # L_knee -> L_ankle
    # Right leg
    (12, 14), # R_hip -> R_knee
    (14, 16), # R_knee -> R_ankle
]

# COCO-17 joint names
COCO_JOINT_NAMES = [
    "nose",       # 0
    "L_eye",      # 1
    "R_eye",      # 2
    "L_ear",      # 3
    "R_ear",      # 4
    "L_shoulder", # 5
    "R_shoulder", # 6
    "L_elbow",    # 7
    "R_elbow",    # 8
    "L_wrist",    # 9
    "R_wrist",    # 10
    "L_hip",      # 11
    "R_hip",      # 12
    "L_knee",     # 13
    "R_knee",     # 14
    "L_ankle",    # 15
    "R_ankle",    # 16
]

# Short names for compact display
COCO_JOINT_NAMES_SHORT = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_sh", "R_sh", "L_el", "R_el", "L_wr", "R_wr",
    "L_hip", "R_hip", "L_kn", "R_kn", "L_an", "R_an",
]

# Joint colors by body part
JOINT_COLORS = {
    "head": "#FF6B6B",      # Red - nose, eyes, ears (0-4)
    "shoulder": "#4ECDC4",  # Teal - shoulders (5-6)
    "arm": "#45B7D1",       # Blue - elbows, wrists (7-10)
    "hip": "#96CEB4",       # Green - hips (11-12)
    "leg": "#FFEAA7",       # Yellow - knees, ankles (13-16)
}

# Map joint index to color
def get_joint_color(joint_idx: int) -> str:
    """Get color for joint by index."""
    if joint_idx <= 4:
        return JOINT_COLORS["head"]
    elif joint_idx <= 6:
        return JOINT_COLORS["shoulder"]
    elif joint_idx <= 10:
        return JOINT_COLORS["arm"]
    elif joint_idx <= 12:
        return JOINT_COLORS["hip"]
    else:
        return JOINT_COLORS["leg"]

# Limb colors (for skeleton visualization)
LIMB_COLORS = {
    "left": "#3498DB",   # Blue for left side
    "right": "#E74C3C",  # Red for right side
    "center": "#2ECC71", # Green for center (torso, etc.)
}

def get_limb_color(joint_i: int, joint_j: int) -> str:
    """Get color for limb based on connected joints."""
    name_i = COCO_JOINT_NAMES[joint_i]
    name_j = COCO_JOINT_NAMES[joint_j]

    if "L_" in name_i or "L_" in name_j:
        return LIMB_COLORS["left"]
    elif "R_" in name_i or "R_" in name_j:
        return LIMB_COLORS["right"]
    else:
        return LIMB_COLORS["center"]
