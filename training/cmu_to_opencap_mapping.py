#!/usr/bin/env python3
"""
CMU Motion Capture (31 joints) to OpenCap Marker Set (65 markers) mapping.

CMU Skeleton (31 joints):
- Lower body: Hips, LHipJoint, RHipJoint, LeftUpLeg, RightUpLeg, LeftLeg, RightLeg,
              LeftFoot, RightFoot, LeftToeBase, RightToeBase
- Spine: LowerBack, Spine, Spine1
- Neck/Head: Neck, Neck1, Head
- Left arm: LeftShoulder, LeftArm, LeftForeArm, LeftHand, LeftFingerBase, LeftHandIndex1, LThumb
- Right arm: RightShoulder, RightArm, RightForeArm, RightHand, RightFingerBase, RightHandIndex1, RThumb

OpenCap Markers (65 total):
- MediaPipe (22): Neck, shoulders, elbows, wrists, hips, knees, ankles, heels, toes, etc.
- Augmented (43): Shoulder clusters, thigh clusters, medial markers, hip joint centers, etc.
"""

from typing import Dict, Optional, List


# Direct mappings: CMU joint → OpenCap marker (straightforward 1:1)
DIRECT_MAPPING = {
    # Pelvis/Hip
    "Hips": "Hip",

    # Lower body - right
    "RightUpLeg": "RHip",  # Actually ASIS position, close enough
    "RightLeg": "RKnee",
    "RightFoot": "RAnkle",
    "RightToeBase": "RBigToe",

    # Lower body - left
    "LeftUpLeg": "LHip",
    "LeftLeg": "LKnee",
    "LeftFoot": "LAnkle",
    "LeftToeBase": "LBigToe",

    # Spine/Neck
    "Spine1": "C7_study",  # Upper spine → C7
    "Head": "Head",

    # Upper body - right
    "RightShoulder": "RShoulder",  # Actually acromion
    "RightArm": "RElbow",
    "RightForeArm": "RWrist",

    # Upper body - left
    "LeftShoulder": "LShoulder",
    "LeftArm": "LElbow",
    "LeftForeArm": "LWrist",
}


# Estimated markers: OpenCap marker → estimation strategy
ESTIMATED_MARKERS = {
    # Pelvis markers (ASIS/PSIS) - estimate from hip joints and spine
    "r.ASIS_study": {
        "method": "offset_from_joint",
        "base_joint": "RightUpLeg",
        "offset": [0.05, 0.02, 0.0],  # Slightly anterior and superior
        "description": "Right ASIS from right hip joint"
    },
    "L.ASIS_study": {
        "method": "offset_from_joint",
        "base_joint": "LeftUpLeg",
        "offset": [-0.05, 0.02, 0.0],
        "description": "Left ASIS from left hip joint"
    },
    "r.PSIS_study": {
        "method": "offset_from_joint",
        "base_joint": "Hips",
        "offset": [0.05, 0.0, -0.05],  # Posterior
        "description": "Right PSIS from pelvis center"
    },
    "L.PSIS_study": {
        "method": "offset_from_joint",
        "base_joint": "Hips",
        "offset": [-0.05, 0.0, -0.05],
        "description": "Left PSIS from pelvis center"
    },

    # Hip Joint Centers - estimate from pelvis geometry
    "RHJC_study": {
        "method": "bell_regression",
        "description": "Right hip joint center (Bell et al. 1990)"
    },
    "LHJC_study": {
        "method": "bell_regression",
        "description": "Left hip joint center (Bell et al. 1990)"
    },

    # Medial knee markers - offset from lateral knee
    "r_mknee_study": {
        "method": "medial_offset",
        "base_joint": "RightLeg",
        "offset_direction": "medial",
        "offset_distance": 0.08,  # ~8cm medial
        "description": "Right medial knee"
    },
    "L_mknee_study": {
        "method": "medial_offset",
        "base_joint": "LeftLeg",
        "offset_direction": "medial",
        "offset_distance": 0.08,
        "description": "Left medial knee"
    },

    # Medial ankle markers
    "r_mankle_study": {
        "method": "medial_offset",
        "base_joint": "RightFoot",
        "offset_direction": "medial",
        "offset_distance": 0.06,
        "description": "Right medial ankle"
    },
    "L_mankle_study": {
        "method": "medial_offset",
        "base_joint": "LeftFoot",
        "offset_direction": "medial",
        "offset_distance": 0.06,
        "description": "Left medial ankle"
    },

    # Heel markers
    "RHeel": {
        "method": "offset_from_joint",
        "base_joint": "RightFoot",
        "offset": [0.0, 0.0, -0.05],  # Posterior from ankle
        "description": "Right heel"
    },
    "LHeel": {
        "method": "offset_from_joint",
        "base_joint": "LeftFoot",
        "offset": [0.0, 0.0, -0.05],
        "description": "Left heel"
    },

    # Small toe markers
    "RSmallToe": {
        "method": "offset_from_joint",
        "base_joint": "RightToeBase",
        "offset": [-0.02, 0.0, 0.01],  # Lateral from big toe
        "description": "Right small toe"
    },
    "LSmallToe": {
        "method": "offset_from_joint",
        "base_joint": "LeftToeBase",
        "offset": [0.02, 0.0, 0.01],
        "description": "Left small toe"
    },

    # Shoulder clusters (3 markers each) - estimate from shoulder-elbow vector
    "r_sh1_study": {
        "method": "shoulder_cluster",
        "base_joint": "RightShoulder",
        "index": 0,
        "description": "Right shoulder cluster marker 1"
    },
    "r_sh2_study": {
        "method": "shoulder_cluster",
        "base_joint": "RightShoulder",
        "index": 1,
        "description": "Right shoulder cluster marker 2"
    },
    "r_sh3_study": {
        "method": "shoulder_cluster",
        "base_joint": "RightShoulder",
        "index": 2,
        "description": "Right shoulder cluster marker 3"
    },
    "L_sh1_study": {
        "method": "shoulder_cluster",
        "base_joint": "LeftShoulder",
        "index": 0,
        "description": "Left shoulder cluster marker 1"
    },
    "L_sh2_study": {
        "method": "shoulder_cluster",
        "base_joint": "LeftShoulder",
        "index": 1,
        "description": "Left shoulder cluster marker 2"
    },
    "L_sh3_study": {
        "method": "shoulder_cluster",
        "base_joint": "LeftShoulder",
        "index": 2,
        "description": "Left shoulder cluster marker 3"
    },

    # Thigh clusters (3 markers each) - along femur
    "r_thigh1_study": {
        "method": "thigh_cluster",
        "base_joint": "RightUpLeg",
        "target_joint": "RightLeg",
        "position": 0.33,  # 1/3 down femur
        "description": "Right thigh cluster marker 1"
    },
    "r_thigh2_study": {
        "method": "thigh_cluster",
        "base_joint": "RightUpLeg",
        "target_joint": "RightLeg",
        "position": 0.50,  # 1/2 down femur
        "description": "Right thigh cluster marker 2"
    },
    "r_thigh3_study": {
        "method": "thigh_cluster",
        "base_joint": "RightUpLeg",
        "target_joint": "RightLeg",
        "position": 0.67,  # 2/3 down femur
        "description": "Right thigh cluster marker 3"
    },
    "L_thigh1_study": {
        "method": "thigh_cluster",
        "base_joint": "LeftUpLeg",
        "target_joint": "LeftLeg",
        "position": 0.33,
        "description": "Left thigh cluster marker 1"
    },
    "L_thigh2_study": {
        "method": "thigh_cluster",
        "base_joint": "LeftUpLeg",
        "target_joint": "LeftLeg",
        "position": 0.50,
        "description": "Left thigh cluster marker 2"
    },
    "L_thigh3_study": {
        "method": "thigh_cluster",
        "base_joint": "LeftUpLeg",
        "target_joint": "LeftLeg",
        "position": 0.67,
        "description": "Left thigh cluster marker 3"
    },

    # Elbow/wrist medial markers
    "r_lelbow_study": {
        "method": "direct_copy",
        "base_joint": "RightArm",
        "description": "Right lateral elbow (same as elbow)"
    },
    "r_melbow_study": {
        "method": "medial_offset",
        "base_joint": "RightArm",
        "offset_direction": "medial",
        "offset_distance": 0.05,
        "description": "Right medial elbow"
    },
    "L_lelbow_study": {
        "method": "direct_copy",
        "base_joint": "LeftArm",
        "description": "Left lateral elbow"
    },
    "L_melbow_study": {
        "method": "medial_offset",
        "base_joint": "LeftArm",
        "offset_direction": "medial",
        "offset_distance": 0.05,
        "description": "Left medial elbow"
    },

    "r_lwrist_study": {
        "method": "direct_copy",
        "base_joint": "RightForeArm",
        "description": "Right lateral wrist"
    },
    "r_mwrist_study": {
        "method": "medial_offset",
        "base_joint": "RightForeArm",
        "offset_direction": "medial",
        "offset_distance": 0.04,
        "description": "Right medial wrist"
    },
    "L_lwrist_study": {
        "method": "direct_copy",
        "base_joint": "LeftForeArm",
        "description": "Left lateral wrist"
    },
    "L_mwrist_study": {
        "method": "medial_offset",
        "base_joint": "LeftForeArm",
        "offset_direction": "medial",
        "offset_distance": 0.04,
        "description": "Left medial wrist"
    },

    # Calcaneus (heel markers)
    "r_calc_study": {
        "method": "direct_copy",
        "base_joint": "RightFoot",
        "description": "Right calcaneus (use heel)"
    },
    "L_calc_study": {
        "method": "direct_copy",
        "base_joint": "LeftFoot",
        "description": "Left calcaneus"
    },

    # Toe markers
    "r_toe_study": {
        "method": "direct_copy",
        "base_joint": "RightToeBase",
        "description": "Right toe"
    },
    "L_toe_study": {
        "method": "direct_copy",
        "base_joint": "LeftToeBase",
        "description": "Left toe"
    },

    "r_5meta_study": {
        "method": "offset_from_joint",
        "base_joint": "RightToeBase",
        "offset": [-0.03, 0.0, 0.0],  # Lateral from big toe
        "description": "Right 5th metatarsal"
    },
    "L_5meta_study": {
        "method": "offset_from_joint",
        "base_joint": "LeftToeBase",
        "offset": [0.03, 0.0, 0.0],
        "description": "Left 5th metatarsal"
    },

    # Upper body
    "Neck": {
        "method": "midpoint",
        "joints": ["LeftShoulder", "RightShoulder"],
        "description": "Neck as midpoint of shoulders"
    },

    "Nose": {
        "method": "offset_from_joint",
        "base_joint": "Head",
        "offset": [0.0, 0.05, 0.08],  # Anterior and superior
        "description": "Nose from head"
    },
}


def get_all_opencap_markers() -> List[str]:
    """Get complete list of 65 OpenCap markers."""
    direct = list(DIRECT_MAPPING.values())
    estimated = list(ESTIMATED_MARKERS.keys())
    return sorted(set(direct + estimated))


def print_mapping_summary():
    """Print summary of mapping strategy."""
    print("=" * 80)
    print("CMU (31 joints) → OpenCap (65 markers) Mapping Strategy")
    print("=" * 80)
    print()

    print(f"Direct mappings: {len(DIRECT_MAPPING)}")
    print(f"Estimated markers: {len(ESTIMATED_MARKERS)}")
    print(f"Total OpenCap markers: {len(get_all_opencap_markers())}")
    print()

    print("Direct Mappings:")
    print("-" * 80)
    for cmu_joint, opencap_marker in sorted(DIRECT_MAPPING.items()):
        print(f"  {cmu_joint:20s} → {opencap_marker}")
    print()

    print("Estimation Methods:")
    print("-" * 80)
    methods = {}
    for marker, info in ESTIMATED_MARKERS.items():
        method = info["method"]
        methods[method] = methods.get(method, 0) + 1

    for method, count in sorted(methods.items()):
        print(f"  {method:20s}: {count} markers")
    print()


if __name__ == "__main__":
    print_mapping_summary()
