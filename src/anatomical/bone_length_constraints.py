"""
Bone length consistency constraints for 3D pose estimation.

Enforces anatomically plausible bone lengths across temporal sequences
by adjusting marker positions (primarily depth/z-coordinates) to maintain
consistent inter-joint distances.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from src.datastream.data_stream import LandmarkRecord

# Define anatomical bone pairs (parent, child)
BONE_PAIRS = [
    # Arms
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    # Legs
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    # Feet
    ("RAnkle", "RHeel"),
    ("LAnkle", "LHeel"),
    ("RAnkle", "RBigToe"),
    ("LAnkle", "LBigToe"),
    # Torso (will be computed after Hip/Neck derivation)
    ("Hip", "Neck"),
    ("Neck", "Nose"),
]


def organize_records_by_frame(
    records: List[LandmarkRecord],
) -> Dict[float, Dict[str, LandmarkRecord]]:
    """Organize landmark records by timestamp and landmark name."""
    frames: Dict[float, Dict[str, LandmarkRecord]] = defaultdict(dict)
    for record in records:
        frames[record.timestamp_s][record.landmark] = record
    return frames


def calculate_bone_length(
    parent: LandmarkRecord, child: LandmarkRecord
) -> float:
    """Calculate Euclidean distance between two landmarks."""
    dx = child.x_m - parent.x_m
    dy = child.y_m - parent.y_m
    dz = child.z_m - parent.z_m
    return np.sqrt(dx**2 + dy**2 + dz**2)


def compute_reference_bone_lengths(
    frames: Dict[float, Dict[str, LandmarkRecord]],
    bone_pairs: List[Tuple[str, str]],
    percentile: float = 50.0,
) -> Dict[Tuple[str, str], float]:
    """
    Compute reference bone lengths using median across all frames.

    Uses median (robust to outliers) to establish expected bone lengths.
    Only considers frames where both parent and child markers are present.
    """
    bone_lengths: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for frame_markers in frames.values():
        for parent_name, child_name in bone_pairs:
            if parent_name in frame_markers and child_name in frame_markers:
                length = calculate_bone_length(
                    frame_markers[parent_name], frame_markers[child_name]
                )
                if length > 0.01:  # Filter out near-zero lengths (likely errors)
                    bone_lengths[(parent_name, child_name)].append(length)

    # Compute reference length (median) for each bone
    reference_lengths: Dict[Tuple[str, str], float] = {}
    for bone_pair, lengths in bone_lengths.items():
        if lengths:
            reference_lengths[bone_pair] = float(np.percentile(lengths, percentile))

    return reference_lengths


def adjust_child_marker_depth(
    parent: LandmarkRecord,
    child: LandmarkRecord,
    target_length: float,
    depth_weight: float = 0.8,
) -> LandmarkRecord:
    """
    Adjust child marker position to match target bone length.

    Primarily adjusts z-coordinate (depth) as it has highest noise.
    Also makes minor adjustments to x,y for natural correction.

    Args:
        parent: Parent landmark (fixed)
        child: Child landmark (to be adjusted)
        target_length: Desired bone length
        depth_weight: Weight for depth correction (0-1), higher = more z adjustment

    Returns:
        Adjusted child landmark record
    """
    # Current vector and length
    dx = child.x_m - parent.x_m
    dy = child.y_m - parent.y_m
    dz = child.z_m - parent.z_m
    current_length = np.sqrt(dx**2 + dy**2 + dz**2)

    if current_length < 0.01:
        # Avoid division by zero for degenerate cases
        return child

    # Calculate scaling factor
    scale = target_length / current_length

    # Apply weighted correction: more on z (depth), less on x,y
    # This preserves 2D appearance while correcting depth
    xy_weight = 1.0 - depth_weight

    new_x = parent.x_m + dx * (1 + (scale - 1) * xy_weight)
    new_y = parent.y_m + dy * (1 + (scale - 1) * xy_weight)
    new_z = parent.z_m + dz * scale  # Full correction on depth

    # Create adjusted record
    return LandmarkRecord(
        timestamp_s=child.timestamp_s,
        landmark=child.landmark,
        x_m=float(new_x),
        y_m=float(new_y),
        z_m=float(new_z),
        visibility=child.visibility,
    )


def apply_bone_length_constraints(
    records: List[LandmarkRecord],
    tolerance: float = 0.15,
    depth_weight: float = 0.8,
    iterations: int = 3,
) -> List[LandmarkRecord]:
    """
    Apply bone length consistency constraints across temporal sequence.

    Iteratively adjusts marker positions to maintain anatomically plausible
    bone lengths. Focuses on correcting noisy depth (z) estimates while
    preserving 2D pose appearance.

    Args:
        records: Input landmark records
        tolerance: Acceptable deviation from reference length (as fraction, e.g. 0.15 = 15%)
        depth_weight: Weight for depth vs xy correction (0-1), default 0.8
        iterations: Number of constraint enforcement passes (default 3)

    Returns:
        Adjusted landmark records with consistent bone lengths
    """
    if not records:
        return records

    # Organize records by frame
    frames = organize_records_by_frame(records)

    # Compute reference bone lengths (median across all frames)
    reference_lengths = compute_reference_bone_lengths(frames, BONE_PAIRS)

    if not reference_lengths:
        print("[bone_length] WARNING: No valid bone pairs found, skipping constraints")
        return records

    # Iteratively enforce constraints
    for iteration in range(iterations):
        adjustments_made = 0

        for timestamp in sorted(frames.keys()):
            frame_markers = frames[timestamp]

            for parent_name, child_name in BONE_PAIRS:
                # Skip if bone pair not in reference (missing data)
                if (parent_name, child_name) not in reference_lengths:
                    continue

                # Skip if either marker missing in this frame
                if parent_name not in frame_markers or child_name not in frame_markers:
                    continue

                parent = frame_markers[parent_name]
                child = frame_markers[child_name]

                # Calculate current length
                current_length = calculate_bone_length(parent, child)
                target_length = reference_lengths[(parent_name, child_name)]

                # Check if adjustment needed
                length_error = abs(current_length - target_length) / target_length

                if length_error > tolerance:
                    # Adjust child marker to match target length
                    adjusted_child = adjust_child_marker_depth(
                        parent, child, target_length, depth_weight
                    )
                    frame_markers[child_name] = adjusted_child
                    adjustments_made += 1

        if iteration == 0:
            print(f"[bone_length] iteration {iteration + 1}: {adjustments_made} adjustments")

    # Convert back to flat list
    adjusted_records: List[LandmarkRecord] = []
    for timestamp in sorted(frames.keys()):
        for landmark_name in sorted(frames[timestamp].keys()):
            adjusted_records.append(frames[timestamp][landmark_name])

    return adjusted_records


def report_bone_length_statistics(
    original_records: List[LandmarkRecord],
    adjusted_records: List[LandmarkRecord],
) -> None:
    """
    Print statistics comparing bone length consistency before/after adjustment.

    Useful for validating that constraints improved consistency.
    """
    def compute_bone_length_std(records: List[LandmarkRecord]) -> Dict[Tuple[str, str], float]:
        """Compute standard deviation of bone lengths over time."""
        frames = organize_records_by_frame(records)
        bone_lengths: Dict[Tuple[str, str], List[float]] = defaultdict(list)

        for frame_markers in frames.values():
            for parent_name, child_name in BONE_PAIRS:
                if parent_name in frame_markers and child_name in frame_markers:
                    length = calculate_bone_length(
                        frame_markers[parent_name], frame_markers[child_name]
                    )
                    bone_lengths[(parent_name, child_name)].append(length)

        return {
            bone: float(np.std(lengths))
            for bone, lengths in bone_lengths.items()
            if len(lengths) > 1
        }

    print("\n[bone_length] === Bone Length Consistency Report ===")

    original_std = compute_bone_length_std(original_records)
    adjusted_std = compute_bone_length_std(adjusted_records)

    total_improvement = 0.0
    count = 0

    for bone_pair in sorted(original_std.keys()):
        if bone_pair in adjusted_std:
            orig = original_std[bone_pair]
            adj = adjusted_std[bone_pair]
            improvement = (orig - adj) / orig * 100 if orig > 0 else 0
            total_improvement += improvement
            count += 1

            parent, child = bone_pair
            print(f"[bone_length] {parent}-{child}: "
                  f"std {orig:.4f}m → {adj:.4f}m ({improvement:+.1f}%)")

    if count > 0:
        avg_improvement = total_improvement / count
        print(f"[bone_length] Average consistency improvement: {avg_improvement:.1f}%")

    print("[bone_length] ======================================\n")
