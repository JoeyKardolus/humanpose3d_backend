"""Utilities for processing joint angle time series.

Provides functions for:
- Unwrapping angles to remove discontinuities
- Smoothing with moving average filters
- Zeroing angles to reference configurations
- Euler angle decomposition with multiple sequences
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np
from scipy.ndimage import median_filter as scipy_median_filter
from scipy.spatial.transform import Rotation


# --- Helper Functions ---


def _ensure_odd_window(w: int) -> int:
    """Ensure window size is odd."""
    return w + 1 if w % 2 == 0 else w


def _apply_per_column(func: Callable, arr: np.ndarray, **kwargs) -> np.ndarray:
    """Apply 1D function to each column of 1D or 2D array."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return func(arr, **kwargs)
    if arr.ndim == 2:
        result = np.empty_like(arr)
        for col in range(arr.shape[1]):
            result[:, col] = func(arr[:, col], **kwargs)
        return result
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


# --- Angle Unwrapping ---


def _unwrap_1d(a: np.ndarray) -> np.ndarray:
    """Unwrap 1D angle series."""
    a = a.copy()
    prev = np.nan

    for i in range(len(a)):
        val = a[i]

        if not np.isfinite(val):
            continue

        if not np.isfinite(prev):
            prev = val
            continue

        # Shift val by ±360 to minimize jump from prev
        diff = val - prev

        while diff > 180:
            val -= 360
            diff = val - prev

        while diff < -180:
            val += 360
            diff = val - prev

        a[i] = val
        prev = val

    return a


def unwrap_angles_deg(angles: np.ndarray) -> np.ndarray:
    """Remove 360° discontinuities from angle time series.

    Uses a greedy unwrapping strategy: for each sample, find the
    ±360k representation closest to the previous value.

    Args:
        angles: Angle array (N,) or (N, 3) in degrees

    Returns:
        Unwrapped angles in degrees
    """
    return _apply_per_column(_unwrap_1d, angles)


# --- Smoothing ---


def _smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Smooth 1D signal with NaN-aware moving average."""
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x).astype(float)
    x_filled = np.nan_to_num(x, nan=0.0)

    kernel = np.ones(window, dtype=float)

    # Convolve values and mask
    numerator = np.convolve(x_filled, kernel, mode="same")
    denominator = np.convolve(mask, kernel, mode="same")

    # Avoid division by zero
    return np.where(denominator > 0, numerator / denominator, np.nan)


def smooth_moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average filter with NaN handling.

    Uses convolution with normalized weights to handle missing data.

    Args:
        data: Input array (N,) or (N, M)
        window_size: Filter window size (must be odd, >= 3)

    Returns:
        Smoothed array (same shape as input)
    """
    if window_size <= 1:
        return np.asarray(data, dtype=float).copy()
    return _apply_per_column(_smooth_1d, data, window=_ensure_odd_window(window_size))


# --- Median Filtering ---


def _median_filter_1d(angles: np.ndarray, window: int) -> np.ndarray:
    """Apply median filter to 1D angle array with NaN handling."""
    valid_mask = np.isfinite(angles)
    if not valid_mask.all():
        filled = angles.copy()
        result = scipy_median_filter(filled, size=window, mode='nearest')
        result[~valid_mask] = np.nan
        return result
    return scipy_median_filter(angles, size=window, mode='nearest')


def median_filter_angles(angles: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply median filter to remove angle outliers.

    Robust to isolated spikes caused by gimbal lock or bad marker data.

    Args:
        angles: Angle array (N,) or (N, 3) in degrees
        window_size: Filter window size (must be odd, >= 3)

    Returns:
        Filtered angles in degrees
    """
    if window_size <= 1:
        return np.asarray(angles, dtype=float).copy()
    return _apply_per_column(_median_filter_1d, angles, window=_ensure_odd_window(window_size))


# --- Zeroing ---


def zero_angles(
    angles: np.ndarray,
    times: np.ndarray,
    mode: Literal["first_frame", "first_n_seconds", "global_mean"] = "first_n_seconds",
    window_seconds: float = 0.5,
) -> np.ndarray:
    """Zero angles to a reference configuration.

    Args:
        angles: Angle array (N, 3) for 3-DOF joint or (N,) for 1-DOF
        times: Time stamps (N,)
        mode: Zeroing strategy:
            - "first_frame": Subtract first valid sample
            - "first_n_seconds": Subtract mean of first N seconds
            - "global_mean": Subtract mean of entire trial
        window_seconds: Duration for "first_n_seconds" mode

    Returns:
        Zeroed angles (same shape as input)
    """
    angles = np.asarray(angles, dtype=float).copy()
    times = np.asarray(times, dtype=float)

    if mode == "first_frame":
        # Find first valid sample
        if angles.ndim == 1:
            valid = np.where(np.isfinite(angles))[0]
            if len(valid) == 0:
                return angles
            offset = angles[valid[0]]
        else:
            # Find first row where all columns are valid
            valid_rows = np.all(np.isfinite(angles), axis=1)
            if not valid_rows.any():
                return angles
            offset = angles[np.where(valid_rows)[0][0]]

        return angles - offset

    elif mode == "first_n_seconds":
        t_max = times[0] + window_seconds
        mask = times <= t_max

        if angles.ndim == 1:
            valid = mask & np.isfinite(angles)
            if not valid.any():
                return angles
            offset = np.nanmean(angles[valid])
        else:
            valid = mask[:, None] & np.isfinite(angles)
            offset = np.where(
                valid.any(axis=0),
                np.nanmean(np.where(valid, angles, np.nan), axis=0),
                0.0
            )

        return angles - offset

    elif mode == "global_mean":
        if angles.ndim == 1:
            offset = np.nanmean(angles)
        else:
            offset = np.nanmean(angles, axis=0)

        return angles - offset

    else:
        raise ValueError(f"Unknown mode: {mode}")


# --- Euler Decomposition ---


def _fix_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Project matrix to SO(3) if determinant is wrong."""
    if abs(np.linalg.det(R) - 1.0) > 0.1:
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
    return R


def euler_angles(R: np.ndarray, sequence: str = 'XYZ') -> np.ndarray:
    """Extract Euler angles (degrees) from rotation matrix.

    Args:
        R: 3x3 rotation matrix
        sequence: Euler sequence ('XYZ', 'ZXY', etc.)

    Returns:
        Euler angles in degrees, order matches sequence
    """
    R = _fix_rotation_matrix(R)
    return Rotation.from_matrix(R).as_euler(sequence, degrees=True)


def euler_xyz(rotation_matrix: np.ndarray) -> np.ndarray:
    """Extract Euler angles (XYZ sequence, intrinsic) from rotation matrix.

    Convention: Rotate about X, then Y, then Z (intrinsic/body-fixed).

    Args:
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Euler angles [X_deg, Y_deg, Z_deg]
    """
    return euler_angles(rotation_matrix, 'XYZ')


def euler_zxy(rotation_matrix: np.ndarray) -> np.ndarray:
    """Extract Euler angles (ZXY sequence, intrinsic) from rotation matrix.

    Used for shoulder kinematics (exo/endo, flex/ext, abd/add).

    Args:
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Euler angles [Z_deg, X_deg, Y_deg]
    """
    return euler_angles(rotation_matrix, 'ZXY')


# --- Geometric Angle Computation ---


def geometric_elbow_flexion(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray
) -> float:
    """Compute elbow flexion angle geometrically.

    More robust than Euler decomposition for elbow motion, especially
    during fast movements. 0° = full extension, 180° = full flexion.

    Args:
        shoulder: Shoulder position (3,)
        elbow: Elbow position (3,)
        wrist: Wrist position (3,)

    Returns:
        Elbow flexion angle in degrees (0-180)
    """
    # Vectors from elbow to shoulder and wrist
    v_upper = shoulder - elbow
    v_forearm = wrist - elbow

    # Normalize
    norm_upper = np.linalg.norm(v_upper)
    norm_forearm = np.linalg.norm(v_forearm)

    if norm_upper < 1e-9 or norm_forearm < 1e-9:
        return np.nan

    v_upper = v_upper / norm_upper
    v_forearm = v_forearm / norm_forearm

    # Dot product gives cos(angle)
    cos_angle = np.clip(np.dot(v_upper, v_forearm), -1.0, 1.0)

    # Angle between vectors (0 = aligned, 180 = opposite)
    # For elbow: 0 = extended, 180 = flexed
    angle_rad = math.acos(cos_angle)

    # Convert to flexion: 180 - angle (so extension = 0, flexion = 180)
    flexion_deg = 180.0 - np.degrees(angle_rad)

    return flexion_deg
