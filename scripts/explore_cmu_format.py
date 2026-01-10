#!/usr/bin/env python3
"""Explore CMU Motion Capture BVH files - understand skeleton structure."""

from pathlib import Path
import re
import numpy as np


class BVHJoint:
    """Represents a joint in the BVH hierarchy."""
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.zeros(3)  # Local offset from parent
        self.channels = []  # List of channel names (Xposition, Yrotation, etc.)

    def __repr__(self):
        return f"Joint({self.name}, channels={len(self.channels)})"


def parse_bvh_hierarchy(bvh_path: Path):
    """Parse BVH file hierarchy to understand skeleton structure."""

    with open(bvh_path, 'r') as f:
        lines = f.readlines()

    joints = {}
    joint_stack = []
    current_joint = None

    for line in lines:
        line = line.strip()

        # ROOT or JOINT declaration
        if line.startswith('ROOT') or line.startswith('JOINT'):
            joint_name = line.split()[1]
            parent = joint_stack[-1] if joint_stack else None
            current_joint = BVHJoint(joint_name, parent)
            joints[joint_name] = current_joint

            if parent:
                parent.children.append(current_joint)

            joint_stack.append(current_joint)

        # End Site (terminal joint, no rotation)
        elif line.startswith('End Site'):
            # Skip end sites for now (just markers, no bones)
            pass

        # OFFSET (local position relative to parent)
        elif line.startswith('OFFSET'):
            parts = line.split()
            offset = [float(parts[1]), float(parts[2]), float(parts[3])]
            if current_joint:
                current_joint.offset = np.array(offset)

        # CHANNELS (degrees of freedom)
        elif line.startswith('CHANNELS'):
            parts = line.split()
            num_channels = int(parts[1])
            channels = parts[2:2+num_channels]
            if current_joint:
                current_joint.channels = channels

        # Closing brace (pop from stack)
        elif line == '}':
            if joint_stack:
                joint_stack.pop()
                current_joint = joint_stack[-1] if joint_stack else None

        # MOTION section starts (end of hierarchy)
        elif line.startswith('MOTION'):
            break

    return joints


def print_skeleton_structure(joints):
    """Print the skeleton hierarchy."""

    # Find root joint
    root = None
    for joint in joints.values():
        if joint.parent is None:
            root = joint
            break

    if not root:
        print("ERROR: No root joint found")
        return

    print("=" * 80)
    print("CMU MOTION CAPTURE SKELETON STRUCTURE")
    print("=" * 80)
    print()

    def print_joint(joint, indent=0):
        channels_str = f" [{len(joint.channels)} DOF: {', '.join(joint.channels)}]" if joint.channels else ""
        offset_str = f" offset=({joint.offset[0]:.2f}, {joint.offset[1]:.2f}, {joint.offset[2]:.2f})"
        print(f"{'  ' * indent}{joint.name}{channels_str}{offset_str}")

        for child in joint.children:
            print_joint(child, indent + 1)

    print_joint(root)
    print()
    print(f"Total joints: {len(joints)}")
    print()


def extract_joint_names(joints):
    """Extract all joint names in the skeleton."""
    return sorted(joints.keys())


def main():
    # Find first BVH file
    bvh_path = Path("data/training/cmu_mocap/cmu-mocap/data/001/01_01.bvh")

    if not bvh_path.exists():
        print(f"ERROR: BVH file not found: {bvh_path}")
        print("Please run from project root directory")
        return

    print(f"\nAnalyzing: {bvh_path.name}")
    print()

    # Parse hierarchy
    joints = parse_bvh_hierarchy(bvh_path)

    # Print structure
    print_skeleton_structure(joints)

    # List all joint names
    print("=" * 80)
    print("ALL JOINT NAMES (for mapping to OpenCap markers)")
    print("=" * 80)
    print()

    joint_names = extract_joint_names(joints)
    for i, name in enumerate(joint_names, 1):
        print(f"{i:2d}. {name}")

    print()
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print()
    print("1. Map CMU joints to OpenCap 65-marker set")
    print("2. Extract 3D positions from BVH motion data")
    print("3. Simulate MediaPipe depth errors (viewpoint-dependent)")
    print("4. Save as training pairs (noisy input, ground truth)")
    print()


if __name__ == "__main__":
    main()
