"""Marker visibility analysis for hiding low-quality markers."""

from __future__ import annotations

from collections import defaultdict
from typing import List

from src.datastream.data_stream import LandmarkRecord


def calculate_low_visibility_markers(
    records: List[LandmarkRecord], threshold: float = 0.5
) -> List[str]:
    """
    Calculate which markers should be hidden based on visibility.

    Args:
        records: List of landmark records with visibility scores
        threshold: Minimum average visibility to keep marker visible (default: 0.5)

    Returns:
        List of marker names that should be hidden due to low visibility
    """
    # Marker hierarchy: parent -> children
    marker_children = {
        "RShoulder": ["RElbow"],
        "RElbow": ["RWrist"],
        "LShoulder": ["LElbow"],
        "LElbow": ["LWrist"],
        "RKnee": ["RAnkle"],
        "RAnkle": ["RHeel", "RBigToe", "RSmallToe"],
        "LKnee": ["LAnkle"],
        "LAnkle": ["LHeel", "LBigToe", "LSmallToe"],
    }

    def get_all_descendants(marker: str) -> list[str]:
        """Recursively get all descendant markers in the hierarchy."""
        result = [marker]
        if marker in marker_children:
            for child in marker_children[marker]:
                result.extend(get_all_descendants(child))
        return result

    # Calculate average visibility per marker
    vis_sums = defaultdict(float)
    vis_counts = defaultdict(int)
    for rec in records:
        vis_sums[rec.landmark] += rec.visibility
        vis_counts[rec.landmark] += 1

    # Find low visibility markers and their descendants
    low_vis_markers = set()
    for marker, total in vis_sums.items():
        avg_vis = total / vis_counts[marker] if vis_counts[marker] > 0 else 0
        if avg_vis < threshold:
            for marker_name in get_all_descendants(marker):
                if marker_name not in low_vis_markers:
                    low_vis_markers.add(marker_name)

    return sorted(low_vis_markers)
