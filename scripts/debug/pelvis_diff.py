#!/usr/bin/env python3
"""Debug script to find exact difference in pelvis computation."""

from pathlib import Path
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.kinematics.angle_processing import smooth_moving_average, unwrap_angles_deg, euler_zxy
from src.kinematics.segment_coordinate_systems import pelvis_axes, normalize
from src.kinematics.joint_angles_euler import read_trc, get_marker


def compute_pelvis_raw(trc_path: Path, smooth_window: int = 21):
    """Compute raw pelvis angles (no post-processing) using main codebase."""
    marker_idx, frames, times, coords = read_trc(trc_path)
    num_frames = len(times)

    print(f"TRC has {num_frames} frames, {coords.shape[1]} markers")
    print(f"Available markers: {list(marker_idx.keys())[:30]}...")

    # Check if ASIS/PSIS markers exist
    for name in ["r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study"]:
        if name in marker_idx:
            print(f"  Found: {name} at index {marker_idx[name]}")
        else:
            print(f"  MISSING: {name}")

    # Apply smoothing to coordinates (same as reference)
    if smooth_window > 1:
        print(f"\nApplying coordinate smoothing (window={smooth_window})...")
        smoothed = np.empty_like(coords)
        for mi in range(coords.shape[1]):
            for axis in range(3):
                smoothed[:, mi, axis] = smooth_moving_average(coords[:, mi, axis], smooth_window)
        coords = smoothed

    # Compute raw pelvis angles (NO post-processing)
    raw_angles = np.full((num_frames, 3), np.nan)
    prev_pelvis = None

    for fi in range(num_frames):
        rasis = get_marker(coords, marker_idx, fi, "r.ASIS_study")
        lasis = get_marker(coords, marker_idx, fi, "L.ASIS_study")
        rpsis = get_marker(coords, marker_idx, fi, "r.PSIS_study")
        lpsis = get_marker(coords, marker_idx, fi, "L.PSIS_study")

        pelvis = pelvis_axes(rasis, lasis, rpsis, lpsis, prev_pelvis)
        if pelvis is None:
            continue
        prev_pelvis = pelvis

        # ZXY Euler decomposition (same as reference)
        raw_angles[fi] = euler_zxy(pelvis)

    return times, raw_angles


def compute_pelvis_reference(trc_path: Path, smooth_window: int = 21):
    """Compute pelvis angles using reference implementation (from 'use' directory)."""
    # Re-implement the reference algorithm here for comparison
    lines = trc_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Parse header
    hdr3 = lines[3].split("\t")
    names = []
    k = 2
    while k < len(hdr3):
        nm = hdr3[k].strip()
        if nm:
            names.append(nm)
        k += 3

    idx = {n: i for i, n in enumerate(names)}
    data = [ln for ln in lines[6:] if ln.strip()]
    F = len(data)

    frames_arr = np.zeros(F, int)
    times = np.zeros(F, float)
    coords = np.full((F, len(names), 3), np.nan, float)

    def sfloat(x):
        try:
            return float(x)
        except:
            return np.nan

    for fi, ln in enumerate(data):
        c = ln.split("\t")
        frames_arr[fi] = int(float(c[0]))
        times[fi] = float(c[1])
        for j in range(len(names)):
            cx, cy, cz = 2 + 3*j, 3 + 3*j, 4 + 3*j
            coords[fi, j] = [sfloat(c[cx]), sfloat(c[cy]), sfloat(c[cz])]

    print(f"Reference: {F} frames, {len(names)} markers")

    # Smooth coordinates (reference method)
    def moving_average_nan(x, w):
        if w <= 1:
            return x
        x = np.asarray(x, float)
        m = np.isfinite(x).astype(float)
        x0 = np.nan_to_num(x, nan=0.0)
        ker = np.ones(w, float)
        num = np.convolve(x0, ker, mode="same")
        den = np.convolve(m, ker, mode="same")
        return np.where(den > 0, num / den, np.nan)

    if smooth_window > 1:
        for m in range(len(names)):
            for c in range(3):
                coords[:, m, c] = moving_average_nan(coords[:, m, c], smooth_window)

    def get(row, name):
        if name not in idx:
            return None
        p = row[idx[name]]
        return p if np.isfinite(p).all() else None

    def norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.full(3, np.nan)

    # Reference pelvis axes computation
    raw_angles = np.full((F, 3), np.nan)
    prev = None

    for i in range(F):
        row = coords[i]
        RASIS = get(row, "r.ASIS_study")
        LASIS = get(row, "L.ASIS_study")
        RPSIS = get(row, "r.PSIS_study")
        LPSIS = get(row, "L.PSIS_study")

        if any(v is None for v in (RASIS, LASIS, RPSIS, LPSIS)):
            continue

        ASISmid = 0.5 * (RASIS + LASIS)
        PSISmid = 0.5 * (RPSIS + LPSIS)

        Zp = norm(RASIS - LASIS)
        Ytmp = norm(ASISmid - PSISmid)
        Xp = norm(np.cross(Ytmp, Zp))
        Yp = norm(np.cross(Zp, Xp))

        # Continuity check
        if prev is not None:
            score = np.dot(Xp, prev[0]) + np.dot(Yp, prev[1]) + np.dot(Zp, prev[2])
            if score < 0:
                Xp, Yp, Zp = -Xp, -Yp, -Zp

        prev = (Xp, Yp, Zp)

        # Build rotation matrix
        Rp = np.column_stack([Xp, Yp, Zp])

        # ZXY Euler
        from scipy.spatial.transform import Rotation as R
        ez, ex, ey = R.from_matrix(Rp).as_euler('ZXY', degrees=True)
        raw_angles[i] = [ez, ex, ey]  # flex, abd, rot

    return times, raw_angles


def main():
    trc_path = PROJECT_ROOT / "use" / "output" / "pose2sim_input_exact_LSTM_fixed.trc"

    print("="*60)
    print("COMPUTING RAW PELVIS ANGLES (before unwrap/zeroing)")
    print("="*60)

    print("\n--- Main codebase implementation ---")
    times_main, angles_main = compute_pelvis_raw(trc_path, smooth_window=21)

    print("\n--- Reference implementation ---")
    times_ref, angles_ref = compute_pelvis_reference(trc_path, smooth_window=21)

    print("\n" + "="*60)
    print("RAW ANGLE COMPARISON (before unwrap/zeroing)")
    print("="*60)

    valid = np.isfinite(angles_main).all(axis=1) & np.isfinite(angles_ref).all(axis=1)
    print(f"Valid frames: {valid.sum()}/{len(valid)}")

    for i, name in enumerate(["flex (Z)", "abd (X)", "rot (Y)"]):
        main_vals = angles_main[valid, i]
        ref_vals = angles_ref[valid, i]
        diff = np.abs(main_vals - ref_vals)

        print(f"\n{name}:")
        print(f"  Main: mean={main_vals.mean():.4f}, std={main_vals.std():.4f}")
        print(f"  Ref:  mean={ref_vals.mean():.4f}, std={ref_vals.std():.4f}")
        print(f"  Diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

    # Now apply unwrap and global_mean zeroing to both
    print("\n" + "="*60)
    print("AFTER UNWRAP + GLOBAL_MEAN ZEROING")
    print("="*60)

    # Main codebase post-processing
    angles_main_processed = unwrap_angles_deg(angles_main.copy())
    angles_main_processed -= np.nanmean(angles_main_processed, axis=0)

    # Reference post-processing
    def unwrap_ref(a):
        a = a.copy()
        prev = np.nan
        for i in range(len(a)):
            val = a[i]
            if not np.isfinite(val):
                continue
            if not np.isfinite(prev):
                prev = val
                continue
            while val - prev > 180:
                val -= 360
            while val - prev < -180:
                val += 360
            a[i] = val
            prev = val
        return a

    angles_ref_processed = angles_ref.copy()
    for col in range(3):
        angles_ref_processed[:, col] = unwrap_ref(angles_ref[:, col])
    angles_ref_processed -= np.nanmean(angles_ref_processed, axis=0)

    valid = np.isfinite(angles_main_processed).all(axis=1) & np.isfinite(angles_ref_processed).all(axis=1)

    for i, name in enumerate(["flex (Z)", "abd (X)", "rot (Y)"]):
        main_vals = angles_main_processed[valid, i]
        ref_vals = angles_ref_processed[valid, i]
        diff = np.abs(main_vals - ref_vals)

        print(f"\n{name}:")
        print(f"  Main: mean={main_vals.mean():.4f}, std={main_vals.std():.4f}")
        print(f"  Ref:  mean={ref_vals.mean():.4f}, std={ref_vals.std():.4f}")
        print(f"  Diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

    # Compare with the actual CSV output
    print("\n" + "="*60)
    print("COMPARISON WITH ACTUAL CSV OUTPUT")
    print("="*60)

    csv_path = PROJECT_ROOT / "use" / "output" / "pose2sim_input_exact_LSTM_fixed_pelvis_global_ZXY.csv"
    csv_df = pd.read_csv(csv_path)

    csv_flex = csv_df["pelvis_flex_deg(Z)"].values
    csv_abd = csv_df["pelvis_abd_deg(X)"].values
    csv_rot = csv_df["pelvis_rot_deg(Y)"].values

    for i, (name, csv_vals) in enumerate([("flex (Z)", csv_flex), ("abd (X)", csv_abd), ("rot (Y)", csv_rot)]):
        # Align lengths
        min_len = min(len(csv_vals), len(angles_ref_processed))
        csv_v = csv_vals[:min_len]
        ref_v = angles_ref_processed[:min_len, i]

        valid = np.isfinite(csv_v) & np.isfinite(ref_v)
        diff = np.abs(csv_v[valid] - ref_v[valid])

        print(f"\n{name}:")
        print(f"  CSV:    mean={np.nanmean(csv_v):.4f}, std={np.nanstd(csv_v):.4f}")
        print(f"  RefPy:  mean={np.nanmean(ref_v):.4f}, std={np.nanstd(ref_v):.4f}")
        print(f"  Diff:   max={diff.max():.6f}, mean={diff.mean():.6f}")


if __name__ == "__main__":
    main()
