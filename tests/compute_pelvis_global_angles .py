# compute_pelvis_global_angles.py
# Globale bekkenhoeken t.o.v. wereld: Flex/Ext (om Z), Abd/Add (om X), Rot (om Y)
# Met smoothing, assen-continuïteit, unwrap en globale nulstelling.

from pathlib import Path
import math, csv
import numpy as np

# ---------- instellingen ----------
SMOOTH_WINDOW = 21          # 0/1 = geen smoothing; anders oneven getal
UNWRAP = True              # verwijder ±180/360 wrap-sprongen
ZERO_MODE = "global_mean"  # "global_mean" | "first_frame" | None
EULER_SEQ = "ZXY"          # we willen Flex=Z, Abd=X, Rot=Y
SIGNS = {                  # pas zo nodig tekens aan (i.v.m. Y-omlaag wereld-as)
    "flex": +1,            # positieve flexie (voorover) → zo laten; zet -1 als je andersom wilt
    "abd":  +1,            # abductie positief; zet -1 als gewenst
    "rot":  +1,            # axiale rotatie; zet -1 als je yaw-teken om wilt
}

# ---------- utils ----------
def sfloat(x):
    try: return float(x)
    except: return math.nan

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
    L = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    hdr3 = L[3].split("\t")
    names = []; k = 2
    while k < len(hdr3):
        nm = hdr3[k].strip()
        if nm: names.append(nm)
        k += 3
    idx = {n:i for i,n in enumerate(names)}
    data = [ln for ln in L[6:] if ln.strip()]
    F = len(data)
    frames = np.zeros(F, int); times = np.zeros(F, float)
    coords = np.full((F, len(names), 3), np.nan, float)
    for fi, ln in enumerate(data):
        c = ln.split("\t")
        frames[fi] = int(float(c[0])); times[fi] = float(c[1])
        for j in range(len(names)):
            cx, cy, cz = 2+3*j, 3+3*j, 4+3*j
            coords[fi, j] = [sfloat(c[cx]), sfloat(c[cy]), sfloat(c[cz])]
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

# ---------- pelvis axes ----------
def pelvis_axes(row, idx, prev=None):
    RASIS = get(row, idx, "r.ASIS_study")
    LASIS = get(row, idx, "L.ASIS_study")
    RPSIS = get(row, idx, "r.PSIS_study")
    LPSIS = get(row, idx, "L.PSIS_study")
    if any(v is None for v in (RASIS, LASIS, RPSIS, LPSIS)):
        return None

    ASISmid = 0.5*(RASIS + LASIS)
    PSISmid = 0.5*(RPSIS + LPSIS)
    # wereld: X=voor, Z=rechts, Y=omlaag
    Zp = normalize(RASIS - LASIS)          # rechts
    Ytmp = normalize(ASISmid - PSISmid)    # omhoog (tegen wereld +Y in)
    Xp = normalize(np.cross(Ytmp, Zp))     # voor
    Yp = normalize(np.cross(Zp, Xp))       # omhoog (orth.)

    # continuïteit t.o.v. vorige frame
    if prev is not None:
        score = np.dot(Xp, prev["Xp"]) + np.dot(Yp, prev["Yp"]) + np.dot(Zp, prev["Zp"])
        if score < 0:
            Xp, Yp, Zp = -Xp, -Yp, -Zp

    return {"Xp": Xp, "Yp": Yp, "Zp": Zp}

def R_from_axes(X,Y,Z):
    # kolommen zijn lokale assen in wereldcoördinaten
    return np.column_stack([X, Y, Z])

def euler_ZXY(R):
    try:
        from scipy.spatial.transform import Rotation as Rsc
        return Rsc.from_matrix(R).as_euler('ZXY', degrees=True)  # [Z, X, Y] in graden
    except Exception as e:
        raise RuntimeError(
            "SciPy is nodig voor Euler('ZXY'). Installeer met:\n"
            "  conda install scipy\n"
            "of  pip install scipy"
        ) from e

# ---------- main ----------
def main():
    # kies TRC
    try:
        import tkinter as tk
        from tkinter import filedialog
        tk.Tk().withdraw()
        p = filedialog.askopenfilename(
            title="Kies augmented TRC (_LSTM_fixed.trc)",
            filetypes=[("TRC","*.trc")]
        )
        if not p:
            print("Geannuleerd."); return
    except Exception:
        print("GUI niet beschikbaar; defaults.")
        p = "pose-3d/pose2sim_input_exact_LSTM_fixed.trc"

    trc = Path(p)
    names, idx, frames, times, coords = read_trc(trc)
    coords = smooth_coords(coords, SMOOTH_WINDOW)
    F = coords.shape[0]

    flex = np.full(F, np.nan)  # om wereld Z (rechts) — sagittale kanteling
    abd  = np.full(F, np.nan)  # om wereld X (voor)   — laterale kanteling
    rot  = np.full(F, np.nan)  # om wereld Y (omlaag) — axiale rotatie

    prev = None
    for i in range(F):
        row = coords[i]
        pel = pelvis_axes(row, idx, prev=prev)
        if pel is None: continue
        prev = pel

        Rp = R_from_axes(pel["Xp"], pel["Yp"], pel["Zp"])
        # Euler('ZXY') → [Z, X, Y] = [flex, abd, rot] vóór tekens
        ez, ex, ey = euler_ZXY(Rp)
        flex[i] = ez * SIGNS["flex"]
        abd[i]  = ex * SIGNS["abd"]
        rot[i]  = ey * SIGNS["rot"]

    # unwrap (voordat we nulstellen)
    if UNWRAP:
        flex[:] = unwrap_series_deg(flex)
        abd[:]  = unwrap_series_deg(abd)
        rot[:]  = unwrap_series_deg(rot)

    # nulstelling
    if ZERO_MODE == "global_mean":
        m = np.isfinite(flex) & np.isfinite(abd) & np.isfinite(rot)
        if m.any():
            flex -= np.nanmean(flex[m]); abd -= np.nanmean(abd[m]); rot -= np.nanmean(rot[m])
    elif ZERO_MODE == "first_frame":
        flex -= flex[0]; abd -= abd[0]; rot -= rot[0]

    # CSV
    out = trc.with_name(trc.stem + "_pelvis_global_ZXY.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s","pelvis_flex_deg(Z)","pelvis_abd_deg(X)","pelvis_rot_deg(Y)"])
        for i in range(F):
            w.writerow([
                f"{times[i]:.6f}",
                "" if not math.isfinite(flex[i]) else f"{flex[i]:.3f}",
                "" if not math.isfinite(abd[i])  else f"{abd[i]:.3f}",
                "" if not math.isfinite(rot[i])  else f"{rot[i]:.3f}",
            ])
    print("Klaar:", out)

    # plot
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3,1, figsize=(12,6), sharex=True)
        ax[0].plot(times, flex, label="Pelvis Flex/Ext (om Z)")
        ax[1].plot(times, abd,  label="Pelvis Abd/Add (om X)")
        ax[2].plot(times, rot,  label="Pelvis Rot (om Y)")
        for a in ax:
            a.axhline(0, lw=0.6, color="k")
            a.legend(loc="upper right"); a.set_ylabel("deg")
        ax[-1].set_xlabel("tijd (s)")
        fig.suptitle(f"Pelvis global angles (ZXY) — {trc.name}")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("Plot niet gelukt (optioneel):", e)

if __name__ == "__main__":
    main()
