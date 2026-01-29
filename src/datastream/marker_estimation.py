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

        # Derive Neck from shoulders (needed for Nose correction and other estimations)
        # This mirrors what csv_to_trc_strict does, but earlier in the pipeline
        if "Neck" not in landmarks and "LShoulder" in landmarks and "RShoulder" in landmarks:
            l_shoulder = landmarks["LShoulder"]
            r_shoulder = landmarks["RShoulder"]
            neck_rec = LandmarkRecord(
                timestamp_s=timestamp_s,
                landmark="Neck",
                x_m=(l_shoulder.x_m + r_shoulder.x_m) / 2,
                y_m=(l_shoulder.y_m + r_shoulder.y_m) / 2,
                z_m=(l_shoulder.z_m + r_shoulder.z_m) / 2,
                visibility=min(l_shoulder.visibility, r_shoulder.visibility)
            )
            estimated_records.append(neck_rec)
            landmarks["Neck"] = neck_rec

        # Correct Nose Z depth - MediaPipe often places it too far forward
        # Anatomically, nose should be ~12-15cm in front of neck
        if "Nose" in landmarks and "Neck" in landmarks:
            nose = landmarks["Nose"]
            neck = landmarks["Neck"]
            nose_offset = nose.z_m - neck.z_m  # How far nose is from neck in Z

            # Clamp nose to be 10-18cm in front of neck (negative Z = forward)
            max_offset = -0.18  # 18cm forward max
            min_offset = -0.10  # 10cm forward min

            if nose_offset < max_offset or nose_offset > min_offset:
                # Nose Z is outside anatomical range, correct it
                corrected_z = neck.z_m + max(min_offset, min(max_offset, nose_offset))
                corrected_nose = LandmarkRecord(
                    timestamp_s=timestamp_s,
                    landmark="Nose",
                    x_m=nose.x_m,
                    y_m=nose.y_m,
                    z_m=corrected_z,
                    visibility=nose.visibility
                )
                # Replace the nose in estimated_records
                estimated_records = [r for r in estimated_records if not (r.landmark == "Nose" and r.timestamp_s == timestamp_s)]
                estimated_records.append(corrected_nose)
                landmarks["Nose"] = corrected_nose

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

        # Estimate foot markers from ankle positions when not available
        # This enables Pose2Sim LSTM to compute lower body augmented markers
        # when using RTMPose/POF which only outputs ankle positions
        if "RHeel" not in landmarks and "RAnkle" in landmarks:
            ankle = landmarks["RAnkle"]
            # Heel is ~8cm behind ankle (negative X in camera space for right foot)
            # and ~5cm below ankle (positive Y in camera space, Y-down)
            estimated_records.append(LandmarkRecord(
                timestamp_s=timestamp_s,
                landmark="RHeel",
                x_m=ankle.x_m - 0.02,  # Slightly medial
                y_m=ankle.y_m + 0.05,  # Below ankle
                z_m=ankle.z_m + 0.08,  # Behind ankle (positive Z = away from camera)
                visibility=0.3
            ))
            landmarks["RHeel"] = estimated_records[-1]

        if "LHeel" not in landmarks and "LAnkle" in landmarks:
            ankle = landmarks["LAnkle"]
            estimated_records.append(LandmarkRecord(
                timestamp_s=timestamp_s,
                landmark="LHeel",
                x_m=ankle.x_m + 0.02,  # Slightly medial
                y_m=ankle.y_m + 0.05,  # Below ankle
                z_m=ankle.z_m + 0.08,  # Behind ankle
                visibility=0.3
            ))
            landmarks["LHeel"] = estimated_records[-1]

        if "RBigToe" not in landmarks and "RAnkle" in landmarks:
            ankle = landmarks["RAnkle"]
            # BigToe is ~18cm forward from ankle (negative Z = toward camera)
            # and ~10cm below ankle
            estimated_records.append(LandmarkRecord(
                timestamp_s=timestamp_s,
                landmark="RBigToe",
                x_m=ankle.x_m + 0.02,  # Slightly lateral
                y_m=ankle.y_m + 0.10,  # Below ankle
                z_m=ankle.z_m - 0.18,  # In front of ankle (toward camera)
                visibility=0.3
            ))
            landmarks["RBigToe"] = estimated_records[-1]

        if "LBigToe" not in landmarks and "LAnkle" in landmarks:
            ankle = landmarks["LAnkle"]
            estimated_records.append(LandmarkRecord(
                timestamp_s=timestamp_s,
                landmark="LBigToe",
                x_m=ankle.x_m - 0.02,  # Slightly lateral
                y_m=ankle.y_m + 0.10,  # Below ankle
                z_m=ankle.z_m - 0.18,  # In front of ankle
                visibility=0.3
            ))
            landmarks["LBigToe"] = estimated_records[-1]

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
