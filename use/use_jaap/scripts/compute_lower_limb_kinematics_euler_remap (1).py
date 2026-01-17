# compute_lower_limb_kinematics_euler_remap.py
# Heup, knie en enkel hoeken (Flex/Ext, Abd/Add, Rot) via Euler 'XYZ' + remap,
# met smoothing en assen-continuïteit (zoals jouw "goede" heup-script).
from pathlib import Path
import math, csv
import numpy as np

# ---------- instellingen ----------
SMOOTH_WINDOW = 9               # 0/1 = geen smoothing; anders oneven getal
ZERO_MODE = "first_n_seconds"       # "global_mean" | "first_n_seconds" | "first_frame" | None
ZERO_WINDOW_S = 0.5
UNWRAP = True  # zet op False als je het niet wil

# Remap en tekens per joint (pas aan als conventies anders moeten)
JOINT_REMAP = {
    "hip":   {"flex":"Z", "abd":"X", "rot":"Y"},
    "knee":  {"flex":"Z", "abd":"X", "rot":"Y"},
    "ankle": {"flex":"Z", "abd":"X", "rot":"Y"},
}
JOINT_SIGNS = {
    "hip":   {"flex":+1, "abd":+1, "rot":+1},
    "knee":  {"flex":+1, "abd":+1, "rot":+1},
    "ankle": {"flex":+1, "abd":+1, "rot":+1},
}

# ---------- utils ----------
def sfloat(x):
    try: return float(x)
    except: return math.nan

def unwrap_series_deg(a):
    """Maak een hoekreeks (in graden) continu door 360°-wraps te verwijderen.
    We kiezen per frame de representatie die het dichtst bij de vorige ligt."""
    a = np.asarray(a, float).copy()
    prev = np.nan
    for i, val in enumerate(a):
        if not np.isfinite(val):
            continue
        if not np.isfinite(prev):
            prev = val
            continue
        # schuif val met ±360 zodat het dicht bij prev komt
        dv = val - prev
        if dv > 180:
            # te grote sprong omhoog -> trek 360 af
            while val - prev > 180:
                val -= 360
        elif dv < -180:
            # te grote sprong omlaag -> tel 360 op
            while val - prev < -180:
                val += 360
        a[i] = val
        prev = val
    return a

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

def read_trc(p: Path):
    L = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    hdr3 = L[3].split("\t")
    names=[]; k=2
    while k < len(hdr3):
        nm = hdr3[k].strip()
        if nm: names.append(nm)
        k += 3
    idx = {n:i for i,n in enumerate(names)}
    data = [ln for ln in L[6:] if ln.strip()]
    F = len(data)
    frames = np.zeros(F,int); times = np.zeros(F,float)
    coords = np.full((F,len(names),3), np.nan, float)
    for fi, ln in enumerate(data):
        c = ln.split("\t")
        frames[fi] = int(float(c[0])); times[fi] = float(c[1])
        for j in range(len(names)):
            cx,cy,cz = 2+3*j, 3+3*j, 4+3*j
            coords[fi,j] = [sfloat(c[cx]), sfloat(c[cy]), sfloat(c[cz])]
    return names, idx, frames, times, coords

def smooth_coords(coords, w):
    if w is None or w <= 1: return coords
    F,M,_ = coords.shape
    out = np.empty_like(coords)
    for m in range(M):
        for c in range(3):
            out[:,m,c] = moving_average_nan(coords[:,m,c], w)
    return out

def get(row, idx, name):
    if name not in idx: return None
    p = row[idx[name]]
    return p if np.isfinite(p).all() else None

# ---------- segment-assen ----------
def pelvis_axes(row, idx, prev=None):
    RASIS = get(row, idx, "r.ASIS_study")
    LASIS = get(row, idx, "L.ASIS_study")
    RPSIS = get(row, idx, "r.PSIS_study")
    LPSIS = get(row, idx, "L.PSIS_study")
    if any(v is None for v in (RASIS, LASIS, RPSIS, LPSIS)): return None
    ASISmid = 0.5*(RASIS + LASIS)
    PSISmid = 0.5*(RPSIS + LPSIS)
    Zp = normalize(RASIS - LASIS)               # rechts
    Yt = normalize(ASISmid - PSISmid)           # omhoog
    Xp = normalize(np.cross(Yt, Zp))            # voor
    Yp = normalize(np.cross(Zp, Xp))            # omhoog (orth.)
    if prev is not None:
        score = np.dot(Xp,prev["Xp"])+np.dot(Yp,prev["Yp"])+np.dot(Zp,prev["Zp"])
        if score < 0: Xp, Yp, Zp = -Xp, -Yp, -Zp
    return {"Xp":Xp, "Yp":Yp, "Zp":Zp}

def femur_axes(row, idx, side, pelvis_Z, prev=None):
    if side=="R":
        HJC=get(row,idx,"RHJC_study"); K1=get(row,idx,"r_knee_study"); K2=get(row,idx,"r_mknee_study")
    else:
        HJC=get(row,idx,"LHJC_study"); K1=get(row,idx,"L_knee_study"); K2=get(row,idx,"L_mknee_study")
    if any(v is None for v in (HJC,K1,K2)): return None
    Kmid = 0.5*(K1 + K2)
    Yf = normalize(Kmid - HJC)                  # prox->dist
    Zf = K1 - K2
    if np.dot(Zf, pelvis_Z) < 0: Zf = -Zf
    Zf = normalize(Zf)
    Xf = normalize(np.cross(Yf, Zf))            # anterior
    Zf = normalize(np.cross(Xf, Yf))            # orth.
    if prev is not None:
        score = np.dot(Xf,prev["Xf"])+np.dot(Yf,prev["Yf"])+np.dot(Zf,prev["Zf"])
        if score < 0: Xf, Yf, Zf = -Xf, -Yf, -Zf
    return {"Xf":Xf, "Yf":Yf, "Zf":Zf}

def tibia_axes(row, idx, side, pelvis_Z, prev=None):
    # long axis: KneeMid -> AnkleMid; width: malleoli
    if side=="R":
        K1=get(row,idx,"r_knee_study"); K2=get(row,idx,"r_mknee_study")
        A1=get(row,idx,"r_ankle_study"); A2=get(row,idx,"r_mankle_study")
    else:
        K1=get(row,idx,"L_knee_study"); K2=get(row,idx,"L_mknee_study")
        A1=get(row,idx,"L_ankle_study"); A2=get(row,idx,"L_mankle_study")
    if any(v is None for v in (K1,K2,A1,A2)): return None
    Kmid = 0.5*(K1 + K2)
    Amid = 0.5*(A1 + A2)
    Yt = normalize(Amid - Kmid)                 # prox->dist (tibia)
    Zt = A1 - A2                                # malleoli breedte
    if np.dot(Zt, pelvis_Z) < 0: Zt = -Zt
    Zt = normalize(Zt)
    Xt = normalize(np.cross(Yt, Zt))            # anterior
    Zt = normalize(np.cross(Xt, Yt))
    if prev is not None:
        score = np.dot(Xt,prev["Xt"])+np.dot(Yt,prev["Yt"])+np.dot(Zt,prev["Zt"])
        if score < 0: Xt, Yt, Zt = -Xt, -Yt, -Zt
    return {"Xt":Xt, "Yt":Yt, "Zt":Zt}

def foot_axes(row, idx, side, pelvis_Z, prev=None):
    # long axis: calc -> toe; width: 5th met vs toe
    if side=="R":
        C=get(row,idx,"r_calc_study"); T=get(row,idx,"r_toe_study"); M5=get(row,idx,"r_5meta_study")
    else:
        C=get(row,idx,"L_calc_study"); T=get(row,idx,"L_toe_study"); M5=get(row,idx,"L_5meta_study")
    if any(v is None for v in (C,T,M5)): return None
    Xf = normalize(T - C)                       # anterior (voorvoet)
    Zf = M5 - T                                 # lateraal
    if np.dot(Zf, pelvis_Z) < 0: Zf = -Zf
    Zf = normalize(Zf)
    Yf = normalize(np.cross(Zf, Xf))            # "op" (dorsaal)
    Zf = normalize(np.cross(Xf, Yf))            # orth.
    if prev is not None:
        score = np.dot(Xf,prev["Xf"])+np.dot(Yf,prev["Yf"])+np.dot(Zf,prev["Zf"])
        if score < 0: Xf, Yf, Zf = -Xf, -Yf, -Zf
    return {"Xf":Xf, "Yf":Yf, "Zf":Zf}

def R_from_axes(X,Y,Z):
    return np.column_stack([X,Y,Z])

def euler_xyz(R):
    try:
        from scipy.spatial.transform import Rotation as Rsc
        return Rsc.from_matrix(R).as_euler('XYZ', degrees=True)
    except Exception:
        c = np.clip(R[0,2], -1.0, 1.0)
        beta  = math.asin(c)                   # Y
        alpha = math.atan2(-R[1,2], R[2,2])    # X
        gamma = math.atan2(-R[0,1], R[0,0])    # Z
        return np.degrees([alpha, beta, gamma])

def remap_angles(eul_xyz, joint, side):
    x,y,z = eul_xyz
    m = {"X":x, "Y":y, "Z":z}
    rm = JOINT_REMAP[joint]; sg = JOINT_SIGNS[joint]
    flex = m[rm["flex"]] * sg["flex"]
    abd  = m[rm["abd"]]  * sg["abd"]
    rot  = m[rm["rot"]]  * sg["rot"]
    # spiegel abd voor links (optioneel; vaak gewenst)
    if side=="L": abd = -abd
    return flex, abd, rot

def main():
    # Kies TRC en zijde
    try:
        import tkinter as tk
        from tkinter import filedialog, simpledialog
        tk.Tk().withdraw()
        p = filedialog.askopenfilename(title="Kies augmented TRC (_LSTM_fixed.trc)",
                                       filetypes=[("TRC","*.trc")])
        if not p:
            print("Geannuleerd.")
            return
        side = (simpledialog.askstring("Zijde", "R of L (default R):", initialvalue="R") or "R").strip().upper()
        if side not in ("R", "L"):
            side = "R"
    except Exception:
        print("GUI niet beschikbaar; defaults gebruikt.")
        p = "pose-3d/pose2sim_input_exact_LSTM_fixed.trc"
        side = "R"

    trc = Path(p)
    names, idx, frames, times, coords = read_trc(trc)
    coords = smooth_coords(coords, SMOOTH_WINDOW)
    F = coords.shape[0]

    # result arrays
    hipF = np.full(F, np.nan); hipA = np.full(F, np.nan); hipR = np.full(F, np.nan)
    kneF = np.full(F, np.nan); kneA = np.full(F, np.nan); kneR = np.full(F, np.nan)
    ankF = np.full(F, np.nan); ankA = np.full(F, np.nan); ankR = np.full(F, np.nan)

    prev_pelvis = None; prev_femur = None; prev_tibia = None; prev_foot = None

    for fi in range(F):
        row = coords[fi]

        pelvis = pelvis_axes(row, idx, prev=prev_pelvis)
        if pelvis is None: 
            continue
        prev_pelvis = pelvis

        femur = femur_axes(row, idx, side, pelvis["Zp"], prev=prev_femur)
        if femur is None:
            continue
        prev_femur = femur

        tibia = tibia_axes(row, idx, side, pelvis["Zp"], prev=prev_tibia)
        if tibia is None:
            continue
        prev_tibia = tibia

        foot = foot_axes(row, idx, side, pelvis["Zp"], prev=prev_foot)
        if foot is None:
            continue
        prev_foot = foot

        Rp = R_from_axes(pelvis["Xp"], pelvis["Yp"], pelvis["Zp"])
        Rf = R_from_axes(femur["Xf"], femur["Yf"], femur["Zf"])
        Rt = R_from_axes(tibia["Xt"], tibia["Yt"], tibia["Zt"])
        Rft= R_from_axes(foot["Xf"],  foot["Yf"],  foot["Zf"])

        # relatieve rotaties
        Rpf = Rp.T @ Rf      # heup
        Rfk = Rf.T @ Rt      # knie
        Rta = Rt.T @ Rft     # enkel

        eHip = euler_xyz(Rpf)
        eKne = euler_xyz(Rfk)
        eAnk = euler_xyz(Rta)

        hipF[fi], hipA[fi], hipR[fi] = remap_angles(eHip, "hip", side)
        kneF[fi], kneA[fi], kneR[fi] = remap_angles(eKne, "knee", side)
        ankF[fi], ankA[fi], ankR[fi] = remap_angles(eAnk, "ankle", side)

    # ---- UNWRAP vóór nulstelling (zorgt dat er geen 300°-sprongen zijn) ----
    if UNWRAP:
        for arr in (hipF, hipA, hipR, kneF, kneA, kneR, ankF, ankA, ankR):
            arr[:] = unwrap_series_deg(arr)

    # ---- nulstelling per joint ----
    def zero_triplet(a, b, c):
        if ZERO_MODE == "global_mean":
            m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            if m.any():
                return a - np.nanmean(a[m]), b - np.nanmean(b[m]), c - np.nanmean(c[m])
        elif ZERO_MODE == "first_n_seconds":
            tmax = times[0] + ZERO_WINDOW_S
            m = (times <= tmax) & np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            if m.any():
                return a - np.nanmean(a[m]), b - np.nanmean(b[m]), c - np.nanmean(c[m])
        elif ZERO_MODE == "first_frame":
            return a - a[0], b - b[0], c - c[0]
        return a, b, c

    hipF, hipA, hipR = zero_triplet(hipF, hipA, hipR)
    kneF, kneA, kneR = zero_triplet(kneF, kneA, kneR)
    ankF, ankA, ankR = zero_triplet(ankF, ankA, ankR)

    # ---- CSV ----
    out = trc.with_name(trc.stem + f"_{side}_hip_knee_ankle_EulerXYZ_remap.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s",
            "hip_flex_deg","hip_abd_deg","hip_rot_deg",
            "knee_flex_deg","knee_abd_deg","knee_rot_deg",
            "ankle_flex_deg","ankle_abd_deg","ankle_rot_deg",
        ])
        for i in range(F):
            w.writerow([
                f"{times[i]:.6f}",
                *(("" if not math.isfinite(v) else f"{v:.3f}") for v in
                   (hipF[i], hipA[i], hipR[i], kneF[i], kneA[i], kneR[i], ankF[i], ankA[i], ankR[i]))
            ])
    print("Klaar:", out)

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3,1, figsize=(12,7), sharex=True)
        axs[0].plot(times, hipF, label="Hip Flex/Ext")
        axs[0].plot(times, hipA, label="Hip Abd/Add")
        axs[0].plot(times, hipR, label="Hip Rot")
        axs[1].plot(times, kneF, label="Knee Flex/Ext")
        axs[1].plot(times, kneA, label="Knee Abd/Add")
        axs[1].plot(times, kneR, label="Knee Rot")
        axs[2].plot(times, ankF, label="Ankle Flex/Ext")
        axs[2].plot(times, ankA, label="Ankle Abd/Add")
        axs[2].plot(times, ankR, label="Ankle Rot")
        for ax in axs:
            ax.axhline(0, lw=0.6); ax.legend(loc="upper right"); ax.set_ylabel("hoek (deg)")
        axs[-1].set_xlabel("tijd (s)")
        fig.suptitle(f"R/L {side} — {trc.name}")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("Plot niet gelukt (optioneel):", e)


if __name__ == "__main__":
    main()
