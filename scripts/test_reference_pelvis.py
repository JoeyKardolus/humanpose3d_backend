#!/usr/bin/env python3
"""Test reference pelvis angle calculation exactly as in compute_pelvis_global_angles.txt"""

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v/n if n > 1e-12 else np.full(3, np.nan)


def pelvis_axes_reference(rasis, lasis, rpsis, lpsis, prev=None):
    """Exact replication of reference pelvis_axes function"""
    if any(m is None or not np.isfinite(m).all() for m in [rasis, lasis, rpsis, lpsis]):
        return None

    asis_mid = 0.5 * (rasis + lasis)
    psis_mid = 0.5 * (rpsis + lpsis)

    # Reference implementation (lines 102-105)
    Zp = normalize(rasis - lasis)          # rechts (right)
    Ytmp = normalize(asis_mid - psis_mid)  # omhoog (up)
    Xp = normalize(np.cross(Ytmp, Zp))     # voor (anterior)
    Yp = normalize(np.cross(Zp, Xp))       # omhoog (orth.)

    # Continuity check (lines 108-111)
    if prev is not None:
        score = np.dot(Xp, prev["Xp"]) + np.dot(Yp, prev["Yp"]) + np.dot(Zp, prev["Zp"])
        if score < 0:
            Xp, Yp, Zp = -Xp, -Yp, -Zp

    return {"Xp": Xp, "Yp": Yp, "Zp": Zp}


def euler_ZXY(Rp):
    """Exact replication of reference euler_ZXY function"""
    return R.from_matrix(Rp).as_euler('ZXY', degrees=True)


# Read TRC file
trc_path = Path("data/output/pose-3d/hardloop/hardloop_final.trc")
from src.kinematics.joint_angles_euler import read_trc, get_marker

marker_idx, frames, times, coords = read_trc(trc_path)
F = len(times)

# Smooth coordinates (window=21 like reference)
from src.kinematics.angle_processing import smooth_moving_average
for mi in range(coords.shape[1]):
    for axis in range(3):
        coords[:, mi, axis] = smooth_moving_average(coords[:, mi, axis], 21)

# Compute angles using EXACT reference implementation
flex_ref = np.full(F, np.nan)
abd_ref = np.full(F, np.nan)
rot_ref = np.full(F, np.nan)

prev = None
for fi in range(F):
    rasis = get_marker(coords, marker_idx, fi, "r.ASIS_study")
    lasis = get_marker(coords, marker_idx, fi, "L.ASIS_study")
    rpsis = get_marker(coords, marker_idx, fi, "r.PSIS_study")
    lpsis = get_marker(coords, marker_idx, fi, "L.PSIS_study")

    pel = pelvis_axes_reference(rasis, lasis, rpsis, lpsis, prev=prev)
    if pel is None:
        continue
    prev = pel

    Rp = np.column_stack([pel["Xp"], pel["Yp"], pel["Zp"]])
    ez, ex, ey = euler_ZXY(Rp)

    flex_ref[fi] = ez  # No sign flip for now
    abd_ref[fi] = ex
    rot_ref[fi] = ey

# Unwrap
from src.kinematics.angle_processing import unwrap_angles_deg, zero_angles
flex_ref = unwrap_angles_deg(flex_ref)
abd_ref = unwrap_angles_deg(abd_ref)
rot_ref = unwrap_angles_deg(rot_ref)

# Zero with global_mean (like reference default)
valid = np.isfinite(flex_ref) & np.isfinite(abd_ref) & np.isfinite(rot_ref)
if valid.any():
    flex_ref -= np.nanmean(flex_ref[valid])
    abd_ref -= np.nanmean(abd_ref[valid])
    rot_ref -= np.nanmean(rot_ref[valid])

# Print stats
print("=== REFERENCE IMPLEMENTATION (exact replication) ===")
print(f"Flex (tilt):      span={np.nanmax(flex_ref)-np.nanmin(flex_ref):.2f}° (range [{np.nanmin(flex_ref):.2f}, {np.nanmax(flex_ref):.2f}])")
print(f"Abd (obliquity):  span={np.nanmax(abd_ref)-np.nanmin(abd_ref):.2f}° (range [{np.nanmin(abd_ref):.2f}, {np.nanmax(abd_ref):.2f}])")
print(f"Rot (rotation):   span={np.nanmax(rot_ref)-np.nanmin(rot_ref):.2f}° (range [{np.nanmin(rot_ref):.2f}, {np.nanmax(rot_ref):.2f}])")
