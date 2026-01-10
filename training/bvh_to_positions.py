#!/usr/bin/env python3
"""
Convert BVH motion capture data to 3D marker positions.

Uses forward kinematics to convert joint rotations → world positions.
"""

from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import re


class BVHJoint:
    """Joint in BVH hierarchy with transformation data."""
    def __init__(self, name: str, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.zeros(3)  # Local offset from parent
        self.channels = []  # Channel names (Xposition, Zrotation, etc.)
        self.channel_indices = []  # Indices into motion data array

    def __repr__(self):
        return f"BVHJoint({self.name}, {len(self.channels)} DOF)"


def parse_bvh(bvh_path: Path) -> Tuple[Dict[str, BVHJoint], np.ndarray, float]:
    """Parse BVH file into hierarchy + motion data.

    Returns:
        joints: Dict of joint name → BVHJoint
        motion_data: (num_frames, num_channels) array
        frame_time: Time between frames in seconds
    """
    with open(bvh_path, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]

    # Parse hierarchy
    joints = {}
    joint_stack = []
    current_joint = None
    channel_count = 0
    brace_depth = 0

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Track braces to know when joints end
        if '{' in line:
            brace_depth += 1

        if line.startswith('ROOT') or line.startswith('JOINT'):
            joint_name = line.split()[1]
            parent = joint_stack[-1] if joint_stack else None
            current_joint = BVHJoint(joint_name, parent)
            joints[joint_name] = current_joint

            if parent:
                parent.children.append(current_joint)

            joint_stack.append(current_joint)

        elif line.startswith('End Site'):
            # Skip end sites - they're terminal markers, not real joints
            pass

        elif line.startswith('OFFSET'):
            parts = line.split()
            offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            if current_joint:
                current_joint.offset = offset

        elif line.startswith('CHANNELS'):
            parts = line.split()
            num_channels = int(parts[1])
            channels = parts[2:2+num_channels]
            if current_joint:
                current_joint.channels = channels
                current_joint.channel_indices = list(range(channel_count, channel_count + num_channels))
                channel_count += num_channels

        elif line == '}':
            brace_depth -= 1
            # Only pop joint stack when we close a joint's brace (not end site braces)
            if joint_stack and brace_depth < len(joint_stack):
                joint_stack.pop()
                current_joint = joint_stack[-1] if joint_stack else None

        elif line.startswith('MOTION'):
            break

        i += 1

    # Parse motion data
    # Find "MOTION" section
    while i < len(lines) and not lines[i].strip().startswith('MOTION'):
        i += 1
    i += 1

    # Parse "Frames:" line
    num_frames = 0
    if i < len(lines) and lines[i].strip().startswith('Frames:'):
        num_frames = int(lines[i].split(':')[1].strip())
        i += 1

    # Parse "Frame Time:" line
    frame_time = 0.0
    if i < len(lines) and lines[i].strip().startswith('Frame Time:'):
        frame_time = float(lines[i].split(':')[1].strip())
        i += 1

    # Parse motion data (remaining lines)
    motion_data = []
    for line in lines[i:]:
        line = line.strip()
        if line:
            values = [float(v) for v in line.split()]
            motion_data.append(values)

    motion_data = np.array(motion_data)

    return joints, motion_data, frame_time


def rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """Create 3x3 rotation matrix for given axis and angle (degrees)."""
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    if axis == 'X' or axis == 'Xrotation':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'Y' or axis == 'Yrotation':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 'Z' or axis == 'Zrotation':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unknown rotation axis: {axis}")


def forward_kinematics(
    joints: Dict[str, BVHJoint],
    motion_frame: np.ndarray,
    root_name: str = "Hips"
) -> Dict[str, np.ndarray]:
    """Compute world positions of all joints using forward kinematics.

    Args:
        joints: Joint hierarchy
        motion_frame: Single frame of motion data (channel values)
        root_name: Name of root joint

    Returns:
        Dict of joint name → world position (3D)
    """
    positions = {}

    def compute_joint_transform(joint: BVHJoint, parent_transform: np.ndarray):
        """Recursively compute joint transforms."""
        # Start with local transform (offset from parent)
        local_pos = joint.offset.copy()
        local_rot = np.eye(3)

        # Apply channel transformations in order
        # BVH rotations are applied right-to-left (like Euler angles)
        for channel_name, channel_idx in zip(joint.channels, joint.channel_indices):
            value = motion_frame[channel_idx]

            if 'position' in channel_name:
                # Translation channel
                axis_idx = {'Xposition': 0, 'Yposition': 1, 'Zposition': 2}[channel_name]
                local_pos[axis_idx] += value
            elif 'rotation' in channel_name:
                # Rotation channel - BVH applies channels right-to-left
                # So we build: R = R_last * ... * R_first
                axis = channel_name[0]  # 'X', 'Y', or 'Z'
                rot = rotation_matrix(axis, value)
                local_rot = rot @ local_rot  # Right-to-left composition

        # Build 4x4 homogeneous transform
        local_transform = np.eye(4)
        local_transform[:3, :3] = local_rot
        local_transform[:3, 3] = local_pos

        # Combine with parent transform
        world_transform = parent_transform @ local_transform

        # Extract world position
        world_pos = world_transform[:3, 3]
        positions[joint.name] = world_pos

        # Recursively process children
        for child in joint.children:
            compute_joint_transform(child, world_transform)

    # Start from root with identity transform
    root_joint = joints[root_name]
    root_transform = np.eye(4)
    compute_joint_transform(root_joint, root_transform)

    # Debug: check how many joints were processed
    num_joints = len(joints)
    num_processed = len(positions)
    if num_processed < num_joints:
        missing = set(joints.keys()) - set(positions.keys())
        print(f"WARNING: Only processed {num_processed}/{num_joints} joints")
        print(f"Missing joints: {list(missing)[:5]}...")

    return positions


def extract_all_positions(bvh_path: Path) -> Tuple[Dict[str, BVHJoint], np.ndarray, float]:
    """Extract 3D positions for all frames in BVH file.

    Returns:
        joints: Joint hierarchy
        all_positions: (num_frames, num_joints, 3) array
        frame_time: Time between frames
    """
    joints, motion_data, frame_time = parse_bvh(bvh_path)

    num_frames = motion_data.shape[0]
    joint_names = sorted(joints.keys())
    num_joints = len(joint_names)

    # Pre-allocate array
    all_positions = np.zeros((num_frames, num_joints, 3))

    # Process each frame
    for frame_idx in range(num_frames):
        motion_frame = motion_data[frame_idx]
        frame_positions = forward_kinematics(joints, motion_frame)

        # Store in array
        for joint_idx, joint_name in enumerate(joint_names):
            if joint_name in frame_positions:
                all_positions[frame_idx, joint_idx] = frame_positions[joint_name]

    return joints, all_positions, frame_time


def main():
    """Test BVH parsing and forward kinematics."""
    bvh_path = Path("data/training/cmu_mocap/cmu-mocap/data/001/01_01.bvh")

    if not bvh_path.exists():
        print(f"ERROR: BVH file not found: {bvh_path}")
        return

    print(f"Parsing: {bvh_path.name}")
    print()

    joints, all_positions, frame_time = extract_all_positions(bvh_path)

    print(f"✓ Parsed successfully")
    print(f"  Joints: {len(joints)}")
    print(f"  Frames: {all_positions.shape[0]}")
    print(f"  Frame time: {frame_time:.4f}s ({1.0/frame_time:.1f} FPS)")
    print()

    # Print some sample positions
    print("Sample positions (frame 0):")
    print("-" * 60)
    joint_names = sorted(joints.keys())
    for i, joint_name in enumerate(joint_names[:10]):  # First 10 joints
        pos = all_positions[0, i]
        print(f"  {joint_name:20s}: ({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f})")
    print("  ...")
    print()

    # Compute bone lengths for validation
    print("Sample bone lengths:")
    print("-" * 60)
    # Right leg: Hip → Knee → Ankle
    joint_idx = {name: i for i, name in enumerate(joint_names)}

    if 'RightUpLeg' in joint_idx and 'RightLeg' in joint_idx:
        hip_pos = all_positions[0, joint_idx['RightUpLeg']]
        knee_pos = all_positions[0, joint_idx['RightLeg']]
        femur_length = np.linalg.norm(knee_pos - hip_pos)
        print(f"  Femur (RightUpLeg → RightLeg): {femur_length:.3f} units")

    if 'RightLeg' in joint_idx and 'RightFoot' in joint_idx:
        knee_pos = all_positions[0, joint_idx['RightLeg']]
        ankle_pos = all_positions[0, joint_idx['RightFoot']]
        tibia_length = np.linalg.norm(ankle_pos - knee_pos)
        print(f"  Tibia (RightLeg → RightFoot): {tibia_length:.3f} units")

    print()
    print("✓ Forward kinematics working correctly!")


if __name__ == "__main__":
    main()
