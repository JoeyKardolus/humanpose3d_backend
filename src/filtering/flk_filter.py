"""FLK (Filter with Learned Kinematics) integration for pose smoothing.

This module provides spatio-temporal filtering of 3D pose landmarks using the
FLK library, which combines adaptive Kalman filtering with biomechanical constraints.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from src.datastream.data_stream import LandmarkRecord

# Core 14 markers that overlap between FLK and our ORDER_22
# Excludes: Hip (derived), Nose, heels, and toe markers
CORE_14_MARKERS = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RHip",
    "LHip",
    "RKnee",
    "LKnee",
    "RAnkle",
    "LAnkle",
    "RElbow",
    "LElbow",
    "RWrist",
    "LWrist",
    "Head",
)


def landmarks_to_flk_format(
    records: List[LandmarkRecord],
) -> Tuple[pd.DataFrame, float]:
    """Convert long-format landmark records to FLK's wide-format DataFrame.

    Args:
        records: List of LandmarkRecord objects (long format: one row per landmark)

    Returns:
        Tuple of (DataFrame, fps):
            - DataFrame with columns: time, Marker1:X, Marker1:Y, Marker1:Z, ...
            - Estimated FPS from timestamps

    Raises:
        ValueError: If no records or insufficient data for FLK processing
    """
    if not records:
        raise ValueError("No landmark records provided")

    # Group records by timestamp
    frames_dict = {}
    for rec in records:
        if rec.landmark not in CORE_14_MARKERS:
            continue  # Skip markers not in core set

        if rec.timestamp_s not in frames_dict:
            frames_dict[rec.timestamp_s] = {}
        frames_dict[rec.timestamp_s][rec.landmark] = (rec.x_m, rec.y_m, rec.z_m)

    if not frames_dict:
        raise ValueError("No core markers found in records")

    # Sort timestamps
    timestamps = sorted(frames_dict.keys())

    # Calculate FPS from timestamps
    if len(timestamps) > 1:
        time_diffs = np.diff(timestamps)
        avg_frame_time = np.mean(time_diffs)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 30.0
    else:
        fps = 30.0  # Default fallback

    # Build wide-format DataFrame
    rows = []
    for ts in timestamps:
        row = {"time": ts}
        frame_data = frames_dict[ts]

        for marker in CORE_14_MARKERS:
            if marker in frame_data:
                x, y, z = frame_data[marker]
                row[f"{marker}:X"] = x
                row[f"{marker}:Y"] = y
                row[f"{marker}:Z"] = z
            else:
                # Missing marker - use NaN (FLK can handle this)
                row[f"{marker}:X"] = np.nan
                row[f"{marker}:Y"] = np.nan
                row[f"{marker}:Z"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, fps


def flk_format_to_landmarks(
    df: pd.DataFrame, original_records: List[LandmarkRecord]
) -> List[LandmarkRecord]:
    """Convert FLK's wide-format DataFrame back to long-format landmark records.

    Args:
        df: FLK output DataFrame with columns: time, Marker1:X, Marker1:Y, Marker1:Z, ...
        original_records: Original landmark records to preserve non-core markers

    Returns:
        List of LandmarkRecord objects with filtered core markers + original non-core markers
    """
    # Build mapping of (timestamp, landmark) -> filtered coordinates
    filtered_coords = {}
    for _, row in df.iterrows():
        ts = row["time"]
        for marker in CORE_14_MARKERS:
            x = row[f"{marker}:X"]
            y = row[f"{marker}:Y"]
            z = row[f"{marker}:Z"]
            if not np.isnan(x):  # Only store valid coordinates
                filtered_coords[(ts, marker)] = (x, y, z)

    # Reconstruct landmark records
    new_records = []
    for rec in original_records:
        key = (rec.timestamp_s, rec.landmark)

        if rec.landmark in CORE_14_MARKERS and key in filtered_coords:
            # Use filtered coordinates for core markers
            x, y, z = filtered_coords[key]
            new_records.append(
                LandmarkRecord(
                    timestamp_s=rec.timestamp_s,
                    landmark=rec.landmark,
                    x_m=x,
                    y_m=y,
                    z_m=z,
                    visibility=rec.visibility,
                )
            )
        else:
            # Preserve original coordinates for non-core markers
            new_records.append(rec)

    return new_records


def apply_flk_filter(
    records: List[LandmarkRecord],
    model_path: Path | None = None,
    enable_rnn: bool = False,
    latency: int = 0,
    num_passes: int = 1,
) -> List[LandmarkRecord]:
    """Apply FLK spatio-temporal filtering to landmark records.

    Args:
        records: List of LandmarkRecord objects to filter
        model_path: Path to FLK's pre-trained GRU model (default: models/GRU.h5)
        enable_rnn: Whether to enable RNN component (slower but better for complex motion)
        latency: Latency frames for prediction (0 = real-time)
        num_passes: Number of times to apply FLK filter (default 1, higher = smoother)

    Returns:
        List of filtered LandmarkRecord objects

    Raises:
        ImportError: If FLK library is not installed
        ValueError: If insufficient data for filtering
    """
    try:
        from FLK import FLK
    except ImportError:
        print(
            "[flk_filter] ERROR: FLK library not installed. Install with:",
            file=sys.stderr,
        )
        print(
            "  git clone https://github.com/PARCO-LAB/FLK.git && cd FLK && pip install .",
            file=sys.stderr,
        )
        raise ImportError("FLK library not available")

    # Convert to FLK format
    df, fps = landmarks_to_flk_format(records)

    if len(df) < 2:
        print(
            f"[flk_filter] WARNING: Only {len(df)} frames, skipping filter (need 2+)",
            file=sys.stderr,
        )
        return records

    # Determine model path
    if model_path is None:
        # Try to find FLK's default model in common locations
        possible_paths = [
            Path("models/GRU.h5"),
            Path("FLK/models/GRU.h5"),
            Path.home() / ".flk/models/GRU.h5",
        ]
        for p in possible_paths:
            if p.exists():
                model_path = p
                break
        else:
            print(
                "[flk_filter] WARNING: No FLK model found, will run without RNN",
                file=sys.stderr,
            )
            enable_rnn = False
            model_path = Path("dummy.h5")  # FLK constructor needs a path

    # Initialize FLK
    keypoints = list(CORE_14_MARKERS)
    print(f"[flk_filter] initializing FLK with {len(keypoints)} markers at {fps:.1f} fps")
    print(f"[flk_filter] applying {num_passes} pass(es) for maximum smoothness")

    # Apply FLK filtering multiple times for extra smoothness
    filtered_df = df.copy()

    for pass_num in range(num_passes):
        first_skeleton = filtered_df.iloc[0, 1:].values  # Skip 'time' column

        flk = FLK(
            fs=int(round(fps)),
            skeleton=first_skeleton,
            keypoints=keypoints,
            model_path=str(model_path),
            latency=latency,
        )
        flk.akf.is_RNN_enabled = enable_rnn

        # Process each frame
        try:
            for i in range(1, len(filtered_df)):
                skeleton = filtered_df.iloc[i, 1:].values

                if np.any(skeleton):
                    # Has data - use correction
                    filtered_skeleton = flk.correct(skeleton)
                else:
                    # Missing frame - use prediction
                    filtered_skeleton = flk.predict()

                filtered_df.iloc[i, 1:] = filtered_skeleton
                flk.reset()

            print(f"[flk_filter] pass {pass_num + 1}/{num_passes} complete - filtered {len(filtered_df)} frames")
        except Exception as e:
            print(f"[flk_filter] ERROR at frame {i}, pass {pass_num + 1}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

    # Convert back to landmark records
    return flk_format_to_landmarks(filtered_df, records)


def apply_gaussian_smoothing(
    records: List[LandmarkRecord], sigma: float = 2.0
) -> List[LandmarkRecord]:
    """Apply Gaussian smoothing to landmark trajectories over time.

    Args:
        records: List of LandmarkRecord objects
        sigma: Gaussian kernel standard deviation (higher = more smoothing)

    Returns:
        List of smoothed LandmarkRecord objects
    """
    if not records:
        return records

    # Group records by landmark
    landmarks_dict = {}
    for rec in records:
        if rec.landmark not in landmarks_dict:
            landmarks_dict[rec.landmark] = []
        landmarks_dict[rec.landmark].append(rec)

    # Sort each landmark group by timestamp
    for landmark in landmarks_dict:
        landmarks_dict[landmark].sort(key=lambda r: r.timestamp_s)

    # Apply Gaussian smoothing to each landmark's trajectory
    smoothed_records = []

    for landmark, landmark_records in landmarks_dict.items():
        if len(landmark_records) < 3:
            # Not enough points to smooth
            smoothed_records.extend(landmark_records)
            continue

        # Extract coordinates and visibility
        x_coords = np.array([r.x_m for r in landmark_records])
        y_coords = np.array([r.y_m for r in landmark_records])
        z_coords = np.array([r.z_m for r in landmark_records])

        # Apply Gaussian filter to each coordinate
        x_smooth = gaussian_filter1d(x_coords, sigma=sigma)
        y_smooth = gaussian_filter1d(y_coords, sigma=sigma)
        z_smooth = gaussian_filter1d(z_coords, sigma=sigma)

        # Create smoothed records
        for i, rec in enumerate(landmark_records):
            smoothed_records.append(
                LandmarkRecord(
                    timestamp_s=rec.timestamp_s,
                    landmark=rec.landmark,
                    x_m=x_smooth[i],
                    y_m=y_smooth[i],
                    z_m=z_smooth[i],
                    visibility=rec.visibility,
                )
            )

    # Sort by (timestamp, landmark) for consistency
    smoothed_records.sort(key=lambda r: (r.timestamp_s, r.landmark))

    print(f"[gaussian_filter] applied sigma={sigma} smoothing to {len(smoothed_records)} records")
    return smoothed_records
