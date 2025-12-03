"""Estimate missing markers to improve Pose2Sim augmentation."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List
from .data_stream import LandmarkRecord


def estimate_missing_markers(records: List[LandmarkRecord]) -> List[LandmarkRecord]:
    """Fill in missing markers using symmetry and extrapolation.

    This helps Pose2Sim's augmentation work even with incomplete pose detection.
    Uses anatomical symmetry and geometric relationships to estimate missing markers.
    """
    # Group by timestamp
    frames = {}
    for rec in records:
        if rec.timestamp_s not in frames:
            frames[rec.timestamp_s] = {}
        frames[rec.timestamp_s][rec.landmark] = rec

    estimated_records = []

    for timestamp_s, landmarks in frames.items():
        # Add all existing landmarks
        for landmark, rec in landmarks.items():
            estimated_records.append(rec)

        # Estimate missing right arm from left arm (mirror)
        if "RElbow" not in landmarks and "LElbow" in landmarks and "Neck" in landmarks:
            l_elbow = landmarks["LElbow"]
            neck = landmarks["Neck"]
            # Mirror across neck's X position
            x_offset = neck.x_m - l_elbow.x_m
            estimated_records.append(LandmarkRecord(
                timestamp_s=timestamp_s,
                landmark="RElbow",
                x_m=neck.x_m + x_offset,
                y_m=l_elbow.y_m,
                z_m=-l_elbow.z_m,  # Mirror Z
                visibility=0.3  # Low confidence for estimated
            ))

        if "RWrist" not in landmarks and "LWrist" in landmarks and "Neck" in landmarks:
            l_wrist = landmarks["LWrist"]
            neck = landmarks["Neck"]
            x_offset = neck.x_m - l_wrist.x_m
            estimated_records.append(LandmarkRecord(
                timestamp_s=timestamp_s,
                landmark="RWrist",
                x_m=neck.x_m + x_offset,
                y_m=l_wrist.y_m,
                z_m=-l_wrist.z_m,
                visibility=0.3
            ))

        # Estimate Head from Nose and Neck
        if "Head" not in landmarks and "Nose" in landmarks and "Neck" in landmarks:
            nose = landmarks["Nose"]
            neck = landmarks["Neck"]
            # Head is above nose, extrapolated from neck-nose direction
            direction = np.array([nose.x_m - neck.x_m, nose.y_m - neck.y_m, nose.z_m - neck.z_m])
            head_pos = np.array([nose.x_m, nose.y_m, nose.z_m]) + direction * 0.3
            estimated_records.append(LandmarkRecord(
                timestamp_s=timestamp_s,
                landmark="Head",
                x_m=head_pos[0],
                y_m=head_pos[1],
                z_m=head_pos[2],
                visibility=0.4
            ))

        # Estimate SmallToes from BigToes and Heels
        if "RSmallToe" not in landmarks and "RBigToe" in landmarks and "RHeel" in landmarks:
            big_toe = landmarks["RBigToe"]
            heel = landmarks["RHeel"]
            # SmallToe is lateral to BigToe, extrapolated
            direction = np.array([big_toe.x_m - heel.x_m, big_toe.y_m - heel.y_m, big_toe.z_m - heel.z_m])
            small_toe_pos = np.array([big_toe.x_m, big_toe.y_m, big_toe.z_m]) + direction * 0.1
            small_toe_pos[2] += 0.02  # Slightly lateral
            estimated_records.append(LandmarkRecord(
                timestamp_s=timestamp_s,
                landmark="RSmallToe",
                x_m=small_toe_pos[0],
                y_m=small_toe_pos[1],
                z_m=small_toe_pos[2],
                visibility=0.3
            ))

        if "LSmallToe" not in landmarks and "LBigToe" in landmarks and "LHeel" in landmarks:
            big_toe = landmarks["LBigToe"]
            heel = landmarks["LHeel"]
            direction = np.array([big_toe.x_m - heel.x_m, big_toe.y_m - heel.y_m, big_toe.z_m - heel.z_m])
            small_toe_pos = np.array([big_toe.x_m, big_toe.y_m, big_toe.z_m]) + direction * 0.1
            small_toe_pos[2] -= 0.02  # Slightly lateral (opposite side)
            estimated_records.append(LandmarkRecord(
                timestamp_s=timestamp_s,
                landmark="LSmallToe",
                x_m=small_toe_pos[0],
                y_m=small_toe_pos[1],
                z_m=small_toe_pos[2],
                visibility=0.3
            ))

    return estimated_records
