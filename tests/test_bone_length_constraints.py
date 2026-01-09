"""
Tests for bone length consistency constraints.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.anatomical.bone_length_constraints import (
    apply_bone_length_constraints,
    calculate_bone_length,
    compute_reference_bone_lengths,
    organize_records_by_frame,
)
from src.datastream.data_stream import LandmarkRecord


def test_calculate_bone_length():
    """Test Euclidean distance calculation between landmarks."""
    parent = LandmarkRecord(
        timestamp_s=0.0,
        landmark="RShoulder",
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        visibility=1.0,
    )
    child = LandmarkRecord(
        timestamp_s=0.0,
        landmark="RElbow",
        x_m=0.3,
        y_m=0.0,
        z_m=0.0,
        visibility=1.0,
    )

    length = calculate_bone_length(parent, child)
    assert abs(length - 0.3) < 1e-6


def test_organize_records_by_frame():
    """Test organizing records into frame dictionary structure."""
    records = [
        LandmarkRecord(0.0, "Nose", 0.0, 0.0, 0.0, 1.0),
        LandmarkRecord(0.0, "RShoulder", 0.1, 0.0, 0.0, 1.0),
        LandmarkRecord(0.033, "Nose", 0.0, 0.0, 0.01, 1.0),
        LandmarkRecord(0.033, "RShoulder", 0.1, 0.0, 0.01, 1.0),
    ]

    frames = organize_records_by_frame(records)

    assert len(frames) == 2
    assert 0.0 in frames
    assert 0.033 in frames
    assert "Nose" in frames[0.0]
    assert "RShoulder" in frames[0.0]


def test_compute_reference_bone_lengths():
    """Test median bone length calculation across frames."""
    # Create test data with consistent bone lengths
    records = []
    for i in range(10):
        t = i * 0.033
        # Shoulder to elbow: consistent 0.3m length
        records.append(LandmarkRecord(t, "RShoulder", 0.0, 0.0, 0.0, 1.0))
        records.append(LandmarkRecord(t, "RElbow", 0.3, 0.0, 0.0, 1.0))

    frames = organize_records_by_frame(records)
    bone_pairs = [("RShoulder", "RElbow")]
    ref_lengths = compute_reference_bone_lengths(frames, bone_pairs)

    assert ("RShoulder", "RElbow") in ref_lengths
    assert abs(ref_lengths[("RShoulder", "RElbow")] - 0.3) < 1e-6


def test_apply_bone_length_constraints_simple():
    """Test that bone length constraints correct inconsistent lengths."""
    # Create test data with noisy z-coordinates
    records = []
    for i in range(5):
        t = i * 0.033
        # Shoulder fixed at origin
        records.append(LandmarkRecord(t, "RShoulder", 0.0, 0.0, 0.0, 1.0))

        # Elbow should be 0.3m away, but add noise to z-coordinate
        z_noise = np.random.uniform(-0.05, 0.05)
        records.append(LandmarkRecord(t, "RElbow", 0.3, 0.0, z_noise, 1.0))

    # Apply constraints
    adjusted_records = apply_bone_length_constraints(
        records, tolerance=0.05, depth_weight=0.8, iterations=3
    )

    # Verify lengths are more consistent after adjustment
    frames = organize_records_by_frame(adjusted_records)
    lengths = []
    for frame_markers in frames.values():
        if "RShoulder" in frame_markers and "RElbow" in frame_markers:
            length = calculate_bone_length(
                frame_markers["RShoulder"], frame_markers["RElbow"]
            )
            lengths.append(length)

    # All lengths should be close to median
    median_length = np.median(lengths)
    max_deviation = max(abs(l - median_length) for l in lengths)

    # After constraints, deviation should be small
    assert max_deviation < 0.05  # Within 5cm tolerance


def test_apply_bone_length_constraints_preserves_count():
    """Test that constraints don't lose or add records."""
    records = []
    for i in range(10):
        t = i * 0.033
        records.append(LandmarkRecord(t, "RShoulder", 0.0, 0.0, 0.0, 1.0))
        records.append(LandmarkRecord(t, "RElbow", 0.3, 0.0, 0.0, 1.0))

    adjusted = apply_bone_length_constraints(records)

    assert len(adjusted) == len(records)


def test_apply_bone_length_constraints_empty_input():
    """Test handling of empty input."""
    result = apply_bone_length_constraints([])
    assert result == []


def test_bone_length_constraints_focuses_on_depth():
    """Test that corrections primarily adjust z-coordinate (depth)."""
    # Create shoulder-elbow pair with incorrect depth
    # Most frames have correct 0.3m length, but one frame has depth error
    records = [
        # Frame 0: Correct length (reference)
        LandmarkRecord(0.0, "RShoulder", 0.0, 0.0, 0.0, 1.0),
        LandmarkRecord(0.0, "RElbow", 0.3, 0.0, 0.0, 1.0),
        # Frame 1: Correct length
        LandmarkRecord(0.033, "RShoulder", 0.0, 0.0, 0.0, 1.0),
        LandmarkRecord(0.033, "RElbow", 0.3, 0.0, 0.0, 1.0),
        # Frame 2: Correct length
        LandmarkRecord(0.066, "RShoulder", 0.0, 0.0, 0.0, 1.0),
        LandmarkRecord(0.066, "RElbow", 0.3, 0.0, 0.0, 1.0),
        # Frame 3: ERROR - excessive depth noise
        LandmarkRecord(0.099, "RShoulder", 0.0, 0.0, 0.0, 1.0),
        LandmarkRecord(0.099, "RElbow", 0.25, 0.0, 0.35, 1.0),  # Total length ~0.43m (43% error)
    ]

    # Original coordinates of the erroneous frame
    original_x = records[7].x_m
    original_y = records[7].y_m
    original_z = records[7].z_m

    # Apply constraints with high depth weight
    adjusted = apply_bone_length_constraints(
        records, tolerance=0.10, depth_weight=0.9, iterations=3
    )

    # Find adjusted elbow position in frame 3
    frames = organize_records_by_frame(adjusted)
    adjusted_elbow = frames[0.099]["RElbow"]

    # Z should change more than x,y due to high depth_weight
    z_change = abs(adjusted_elbow.z_m - original_z)
    x_change = abs(adjusted_elbow.x_m - original_x)
    y_change = abs(adjusted_elbow.y_m - original_y)

    assert z_change > 0.05  # Significant z correction
    assert z_change > x_change * 2  # Z changes much more than x
    assert z_change > y_change * 2  # Z changes much more than y
