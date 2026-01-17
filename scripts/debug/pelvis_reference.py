#!/usr/bin/env python3
"""
Exact replication of compute_pelvis_global_angles.txt reference script.
Run on TRC file to get pelvis global angles using ZXY Euler decomposition.
"""

from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Configuration (matching reference script defaults)
SMOOTH_WINDOW = 21          # Reference default
UNWRAP = True
ZERO_MODE = "global_mean"   # Reference default
SIGNS = {"flex": +1, "abd": +1, "rot": +1}  # No sign flips

# ---------- Utils ----------
def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v/n if n > 1e-12 else np.full(3, np.nan)

def moving_average_nan(x, w):
    if w is None or w <= 1: return x
    x = np.asarray(x, float)
    m = np.isfinite(x).astype(float)
    x0 = np.nan_to_num(x, nan=0.0)
    ker = np.ones(w, float)
    num = np.convolve(x0, ker, mode="same")
    den = np.convolve(m,  ker, mode="same")
    return np.where(den>0, num/den, np.nan)

def unwrap_series_deg(a):
    a = np.asarray(a, float).copy()
    prev = np.nan
    for i, val in enumerate(a):
        if not np.isfinite(val):
            continue
        if not np.isfinite(prev):
            prev = val
            continue
        while val - prev > 180:  val -= 360
        while val - prev < -180: val += 360
        a[i] = val
        prev = val
    return a

def read_trc(p: Path):
    """Read TRC file - handles both 22-marker and augmented 65-marker files"""
    L = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    hdr3 = L[3].split("\t")
    names = []
    k = 2
    while k < len(hdr3):
        nm = hdr3[k].strip()
        if nm: names.append(nm)
        k += 3

    # Read data to determine actual number of markers
    data_lines = [ln for ln in L[6:] if ln.strip()]
    if data_lines:
        first_row = data_lines[0].split("\t")
        num_data_cols = len(first_row)
        num_markers_in_data = (num_data_cols - 2) // 3

        # If data has more markers than header (augmented file)
        if num_markers_in_data > len(names):
            augmented_names = [
                "C7_study", "r_shoulder_study", "L_shoulder_study",
                "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study",
                "r_knee_study", "L_knee_study", "r_mknee_study", "L_mknee_study",
                "r_ankle_study", "L_ankle_study", "r_mankle_study", "L_mankle_study",
                "r_calc_study", "L_calc_study", "r_toe_study", "L_toe_study",
                "r_5meta_study", "L_5meta_study",
                "r_lelbow_study", "L_lelbow_study", "r_melbow_study", "L_melbow_study",
                "r_lwrist_study", "L_lwrist_study", "r_mwrist_study", "L_mwrist_study",
                "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
                "r_sh1_study", "r_sh2_study", "r_sh3_study",
                "L_sh1_study", "L_sh2_study", "L_sh3_study",
                "RHJC_study", "LHJC_study",
            ]
            num_augmented_needed = num_markers_in_data - len(names)
            names.extend(augmented_names[:num_augmented_needed])

    idx = {n:i for i,n in enumerate(names)}

    F = len(data_lines)
    frames = np.zeros(F, int)
    times = np.zeros(F, float)
    coords = np.full((F, len(names), 3), np.nan, float)

    for fi, ln in enumerate(data_lines):
        c = ln.split("\t")
        try:
            frames[fi] = int(float(c[0]))
            times[fi] = float(c[1])
        except (ValueError, IndexError):
            continue

        for j in range(len(names)):
            cx, cy, cz = 2+3*j, 3+3*j, 4+3*j
            if cx < len(c) and cy < len(c) and cz < len(c):
                try:
                    coords[fi, j, 0] = float(c[cx])
                    coords[fi, j, 1] = float(c[cy])
                    coords[fi, j, 2] = float(c[cz])
                except ValueError:
                    pass

    return names, idx, frames, times, coords

def smooth_coords(coords, w):
    if w is None or w <= 1: return coords
    F, M, _ = coords.shape
    out = np.empty_like(coords)
    for m in range(M):
        for c in range(3):
            out[:, m, c] = moving_average_nan(coords[:, m, c], w)
    return out

def get(row, idx, name):
    if name not in idx: return None
    p = row[idx[name]]
    return p if np.isfinite(p).all() else None

def pelvis_axes(row, idx, prev=None):
    """Reference pelvis_axes implementation"""
    RASIS = get(row, idx, "r.ASIS_study")
    LASIS = get(row, idx, "L.ASIS_study")
    RPSIS = get(row, idx, "r.PSIS_study")
    LPSIS = get(row, idx, "L.PSIS_study")

    if any(v is None for v in (RASIS, LASIS, RPSIS, LPSIS)):
        return None

    ASISmid = 0.5*(RASIS + LASIS)
    PSISmid = 0.5*(RPSIS + LPSIS)

    # Reference implementation
    Zp = normalize(RASIS - LASIS)          # rechts (right)
    Ytmp = normalize(ASISmid - PSISmid)    # omhoog (up, but in MediaPipe Y=down world)
    Xp = normalize(np.cross(Ytmp, Zp))     # voor (anterior)
    Yp = normalize(np.cross(Zp, Xp))       # omhoog (orthogonalized)

    # Continuity check
    if prev is not None:
        score = np.dot(Xp, prev["Xp"]) + np.dot(Yp, prev["Yp"]) + np.dot(Zp, prev["Zp"])
        if score < 0:
            Xp, Yp, Zp = -Xp, -Yp, -Zp

    return {"Xp": Xp, "Yp": Yp, "Zp": Zp}

def euler_ZXY(Rp):
    """ZXY Euler decomposition"""
    return R.from_matrix(Rp).as_euler('ZXY', degrees=True)

# ---------- Main ----------
def main():
    # Get TRC path from command line or use default
    if len(sys.argv) > 1:
        trc_path = Path(sys.argv[1])
    else:
        trc_path = Path("data/output/pose-3d/MicrosoftTeams-video/MicrosoftTeams-video_final.trc")

    if not trc_path.exists():
        print(f"Error: TRC file not found: {trc_path}")
        print("\nUsage: python run_reference_pelvis.py <trc_file.trc>")
        sys.exit(1)

    print(f"Processing: {trc_path.name}")
    print(f"Settings: SMOOTH_WINDOW={SMOOTH_WINDOW}, ZERO_MODE={ZERO_MODE}\n")

    # Read TRC
    names, idx, frames, times, coords = read_trc(trc_path)
    coords = smooth_coords(coords, SMOOTH_WINDOW)
    F = coords.shape[0]

    # Initialize angle arrays
    flex = np.full(F, np.nan)  # around Z (right) - sagittal tilt
    abd  = np.full(F, np.nan)  # around X (anterior) - frontal tilt
    rot  = np.full(F, np.nan)  # around Y (superior) - axial rotation

    prev = None
    for i in range(F):
        row = coords[i]
        pel = pelvis_axes(row, idx, prev=prev)
        if pel is None: continue
        prev = pel

        Rp = np.column_stack([pel["Xp"], pel["Yp"], pel["Zp"]])
        ez, ex, ey = euler_ZXY(Rp)

        flex[i] = ez * SIGNS["flex"]
        abd[i]  = ex * SIGNS["abd"]
        rot[i]  = ey * SIGNS["rot"]

    # Unwrap
    if UNWRAP:
        flex[:] = unwrap_series_deg(flex)
        abd[:]  = unwrap_series_deg(abd)
        rot[:]  = unwrap_series_deg(rot)

    # Zero
    if ZERO_MODE == "global_mean":
        m = np.isfinite(flex) & np.isfinite(abd) & np.isfinite(rot)
        if m.any():
            flex -= np.nanmean(flex[m])
            abd -= np.nanmean(abd[m])
            rot -= np.nanmean(rot[m])
    elif ZERO_MODE == "first_frame":
        flex -= flex[0]
        abd -= abd[0]
        rot -= rot[0]

    # Report results
    print("=" * 60)
    print("REFERENCE PELVIS GLOBAL ANGLES (ZXY Euler)")
    print("=" * 60)
    print(f"\nFlex/Ext (tilt, around Z):  {np.nanmin(flex):7.2f}° to {np.nanmax(flex):7.2f}°  (span: {np.nanmax(flex)-np.nanmin(flex):5.2f}°)")
    print(f"Abd/Add (obliquity, X):     {np.nanmin(abd):7.2f}° to {np.nanmax(abd):7.2f}°  (span: {np.nanmax(abd)-np.nanmin(abd):5.2f}°)")
    print(f"Rotation (axial, Y):        {np.nanmin(rot):7.2f}° to {np.nanmax(rot):7.2f}°  (span: {np.nanmax(rot)-np.nanmin(rot):5.2f}°)")
    print()

    # Save CSV
    out_csv = trc_path.with_name(trc_path.stem + "_pelvis_global_ZXY_reference.csv")
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s","pelvis_flex_deg(Z)","pelvis_abd_deg(X)","pelvis_rot_deg(Y)"])
        for i in range(F):
            w.writerow([
                f"{times[i]:.6f}",
                "" if not np.isfinite(flex[i]) else f"{flex[i]:.3f}",
                "" if not np.isfinite(abd[i])  else f"{abd[i]:.3f}",
                "" if not np.isfinite(rot[i])  else f"{rot[i]:.3f}",
            ])

    print(f"✓ Saved CSV: {out_csv}")

if __name__ == "__main__":
    main()
