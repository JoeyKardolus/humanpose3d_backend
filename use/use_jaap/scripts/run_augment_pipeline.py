# run_augment_pipeline.py
# End-to-end pipeline:
# 1) Select video (GUI) -> run MediaPipe Pose world landmarks -> pose_world_landmarks.csv
# 2) Convert CSV -> TRC with EXACT 22 input markers (correct order)
# 3) Copy TRC into pose-3d/, park other TRCs, run Pose2Sim MarkerAugmenter (LSTM v0.3)
# 4) Rebuild header of *_LSTM.trc to include *_study marker names
# Output: pose-3d\<basename>_LSTM_fixed.trc

import os, sys, csv, math, shutil, time
from collections import defaultdict
from pathlib import Path

# ---------- SETTINGS ----------
POSE3D_DIR = Path("pose-3d")
POSE3D_DIR.mkdir(exist_ok=True)

# Participant defaults (worden overschreven door GUI)
PARTICIPANT_DEFAULT = dict(mass=75.0, height=1.78, age=30, sex="male")

def get_participant_via_gui(defaults=PARTICIPANT_DEFAULT):
    """Vraag lengte/gewicht/leeftijd/geslacht via GUI.
       Retourneert dict met keys: mass, height, age, sex."""
    try:
        import tkinter as tk
        from tkinter import simpledialog, messagebox
        root = tk.Tk(); root.withdraw()

        def ask_float(title, prompt, initial):
            val = simpledialog.askstring(title, f"{prompt} (bijv. {initial})", initialvalue=str(initial))
            if val is None: return None
            try:
                v = float(val.replace(",", "."))  # comma -> dot
                if v <= 0: raise ValueError
                return v
            except:
                messagebox.showerror("Ongeldige invoer", f"Geef een positieve waarde op voor: {prompt}")
                return ask_float(title, prompt, initial)

        def ask_int(title, prompt, initial):
            val = simpledialog.askstring(title, f"{prompt} (bijv. {initial})", initialvalue=str(initial))
            if val is None: return None
            try:
                v = int(val)
                if v <= 0: raise ValueError
                return v
            except:
                messagebox.showerror("Ongeldige invoer", f"Geef een positieve geheel getal op voor: {prompt}")
                return ask_int(title, prompt, initial)

        def ask_sex(title, prompt, initial):
            val = simpledialog.askstring(title, f"{prompt} [male/female]", initialvalue=str(initial))
            if val is None: return None
            v = str(val).strip().lower()
            if v not in ("male", "female"):
                messagebox.showerror("Ongeldige invoer", "Kies ‘male’ of ‘female’.")
                return ask_sex(title, prompt, initial)
            return v

        height = ask_float("Persoonsgegevens", "Lengte in meters", defaults["height"])
        if height is None: return defaults
        mass   = ask_float("Persoonsgegevens", "Massa in kg", defaults["mass"])
        if mass is None: return defaults
        age    = ask_int  ("Persoonsgegevens", "Leeftijd in jaren", defaults["age"])
        if age is None: return defaults
        sex    = ask_sex  ("Persoonsgegevens", "Geslacht", defaults["sex"])
        if sex is None: return defaults

        return dict(height=height, mass=mass, age=age, sex=sex)
    except Exception:
        # als tkinter niet werkt (bv. headless), val terug op defaults
        return defaults.copy()

# EXACT input-order (22 markers) for the augmenter (no face/fingers except Nose/Head)
ORDER_22 = [
    "Hip",
    "RHip","RKnee","RAnkle","RBigToe","RSmallToe","RHeel",
    "LHip","LKnee","LAnkle","LBigToe","LSmallToe","LHeel",
    "Neck","Head","Nose",
    "RShoulder","RElbow","RWrist",
    "LShoulder","LElbow","LWrist",
]

# Mapping from MediaPipe CSV labels -> our names (world coordinates in meters)
MP2OUR = {
    "LEFT_SHOULDER": "LShoulder",
    "RIGHT_SHOULDER": "RShoulder",
    "LEFT_ELBOW": "LElbow",
    "RIGHT_ELBOW": "RElbow",
    "LEFT_WRIST": "LWrist",
    "RIGHT_WRIST": "RWrist",
    "LEFT_HIP": "LHip",
    "RIGHT_HIP": "RHip",
    "LEFT_KNEE": "LKnee",
    "RIGHT_KNEE": "RKnee",
    "LEFT_ANKLE": "LAnkle",
    "RIGHT_ANKLE": "RAnkle",
    "LEFT_HEEL": "LHeel",
    "RIGHT_HEEL": "RHeel",
    "LEFT_FOOT_INDEX": "LBigToe",
    "RIGHT_FOOT_INDEX": "RBigToe",
    "NOSE": "Nose",
    "HEAD": "Head",   # soms niet aanwezig; dan blijft 'Head' leeg
}

# ---------- UTILITIES ----------
def sfloat(x):
    try: return float(x)
    except: return math.nan

def choose_file(title, patterns):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        return filedialog.askopenfilename(title=title, filetypes=patterns)
    except Exception as e:
        print("Kon geen bestandskiezer openen:", e)
        return ""

def ensure_ok(ok, msg):
    if not ok:
        print("FOUT:", msg)
        sys.exit(1)

# ---------- STEP 1: VIDEO -> CSV VIA MEDIAPIPE ----------
def video_to_csv(video_path: Path) -> Path:
    print("\n[1/4] MediaPipe 3D draait op:", video_path.name)
    out_csv = video_path.with_name("pose_world_landmarks.csv")

    import cv2
    import mediapipe as mp
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(str(video_path))
    ensure_ok(cap.isOpened(), "Kon video niet openen")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    t0 = time.time()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s","landmark","x_m","y_m","z_m","visibility"])
        with mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False) as pose:
            while True:
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                ts = frame_idx / fps
                if res.pose_world_landmarks:
                    for idx, lm in enumerate(res.pose_world_landmarks.landmark):
                        name = mp_pose.PoseLandmark(idx).name if idx in mp_pose.PoseLandmark.__members__.values() else f"LM_{idx}"
                        writer.writerow([f"{ts:.6f}", name, lm.x, lm.y, lm.z, lm.visibility])
                frame_idx += 1

    cap.release()
    print(f"  -> CSV geschreven: {out_csv}")
    return out_csv

# ---------- STEP 2: CSV -> TRC (EXACT 22 MARKERS) ----------
def csv_to_trc_exact(csv_path: Path) -> Path:
    print("\n[2/4] CSV -> TRC (exacte 22 input-markers)")
    out_trc = csv_path.with_name("pose2sim_input_exact.trc")

    rows_by_t = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows_by_t[r["timestamp_s"]].append(r)

    timestamps = sorted(rows_by_t.keys(), key=lambda s: float(s))
    ensure_ok(len(timestamps) > 0, "Geen data in CSV")

    def estimate_rate(ts):
        if len(ts) < 2: return 30.0
        t0, t1 = float(ts[0]), float(ts[-1])
        dur = max(t1 - t0, 1e-9)
        return (len(ts)-1)/dur

    rate = estimate_rate(timestamps)

    frames = []
    for ts in timestamps:
        d = {}
        for r in rows_by_t[ts]:
            mp_name = r["landmark"]
            if mp_name in MP2OUR:
                name = MP2OUR[mp_name]
                d[name] = (sfloat(r["x_m"]), sfloat(r["y_m"]), sfloat(r["z_m"]))
        # afgeleiden
        if "LShoulder" in d and "RShoulder" in d:
            LS, RS = d["LShoulder"], d["RShoulder"]
            d["Neck"] = ((LS[0]+RS[0])/2, (LS[1]+RS[1])/2, (LS[2]+RS[2])/2)
        if "LHip" in d and "RHip" in d:
            LH, RH = d["LHip"], d["RHip"]
            d["Hip"] = ((LH[0]+RH[0])/2, (LH[1]+RH[1])/2, (LH[2]+RH[2])/2)
        if "RBigToe" in d and "RSmallToe" not in d:
            d["RSmallToe"] = d["RBigToe"]
        if "LBigToe" in d and "LSmallToe" not in d:
            d["LSmallToe"] = d["LBigToe"]
        frames.append(d)

    # TRC schrijven
    with open(out_trc, "w", newline="", encoding="utf-8") as wh:
        w = csv.writer(wh, delimiter="\t")
        w.writerow([f"PathFileType\t4\t(X/Y/Z)\t{out_trc.name}"])
        w.writerow(["DataRate","CameraRate","NumFrames","NumMarkers","Units","OrigDataRate","OrigDataStartFrame","OrigNumFrames"])
        w.writerow([f"{rate:.6f}", f"{rate:.6f}", len(timestamps), len(ORDER_22), "m", f"{rate:.6f}", 1, len(timestamps)])
        # header 3
        hdr3 = ["Frame#", "Time"]
        for nm in ORDER_22: hdr3 += [nm, "", ""]
        w.writerow(hdr3)
        # header 4
        hdr4 = ["", ""]
        for i in range(1, len(ORDER_22)+1): hdr4 += [f"X{i}", f"Y{i}", f"Z{i}"]
        w.writerow(hdr4)
        # lege rij
        w.writerow([])
        # data
        t0 = float(timestamps[0])
        for i, ts in enumerate(timestamps, start=1):
            t = float(ts) - t0
            row = [i, f"{t:.6f}"]
            d = frames[i-1]
            for nm in ORDER_22:
                if nm in d and all(map(math.isfinite, d[nm])):
                    x,y,z = d[nm]
                    row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
                else:
                    row += ["","",""]
            w.writerow(row)

    print(f"  -> TRC geschreven: {out_trc}")
    return out_trc

# ---------- STEP 3: RUN MARKER AUGMENTER ----------
def run_augmenter(trc_path: Path, PARTICIPANT: dict) -> Path:
    print("\n[3/4] Marker Augmenter (LSTM v0.3)")

    # 1) Zet invoer in pose-3d/ en parkeer andere .trc's
    target = POSE3D_DIR / trc_path.name
    shutil.copy2(trc_path, target)
    for p in POSE3D_DIR.glob("*.trc"):
        if p.name != target.name:
            p.rename(p.with_suffix(".park"))

    # Debug: toon wat er nu in pose-3d ligt
    trcs_now = list(POSE3D_DIR.glob("*.trc"))
    print("  -> TRCs in pose-3d:", [t.name for t in trcs_now])

    PROJECT_ROOT = Path.cwd()

    # 2) Bepaal frame_range DIRECT uit de TRC (nooit None)
    #    Pak headerregel 2 (NumFrames), of tel datarijen.
    trc_text = (POSE3D_DIR / trc_path.name).read_text(encoding="utf-8", errors="ignore").splitlines()
    # header: [0]=PathFileType, [1]=DataRate..., [2]=rates/NumFrames-lijn, [3],[4],[5] headers, daarna data
    try:
        hdr2 = trc_text[2].split("\t")
        # "DataRate","CameraRate","NumFrames","NumMarkers","Units","OrigDataRate","OrigDataStartFrame","OrigNumFrames"
        num_frames = int(float(hdr2[2]))
    except Exception:
        # fallback: tel datarijen
        data_rows = [ln for ln in trc_text[6:] if ln.strip()]
        num_frames = len(data_rows)
    frame_range_list = [1, num_frames]  # 1-based gesloten interval

    # 3) Maak Python-config (voor zover markerAugmenter die wel leest)
    config = {
        "project": {
            "project_dir": str(PROJECT_ROOT.resolve()),
            "video_dir":   str(PROJECT_ROOT.resolve()),
            "output_dir":  str(PROJECT_ROOT.resolve()),
            # <<< jouw augmenter leest deze uit 'project'
            "participant_height": float(PARTICIPANT["height"]),
            "participant_mass":   float(PARTICIPANT["mass"]),
            "frame_range":        frame_range_list,
        },
        "data": {
            "trc_dir": str((PROJECT_ROOT / "pose-3d").resolve()),
        },
        "participant": {  # mag blijven; is voor onszelf/consistente TOML
            "mass":   float(PARTICIPANT["mass"]),
            "height": float(PARTICIPANT["height"]),
            "age":    int(PARTICIPANT["age"]),
            "sex":    str(PARTICIPANT["sex"]),
        },
        "markerAugmentation": {
            "feet_on_floor": False,   # springvideo -> uit
            "use_lower_limb": True,
            "use_upper_limb": True,
            "verbose": True,
        },
        "kinematics": {
            "frame_range": frame_range_list,  # extra, kan geen kwaad
            "fastest_frames_to_remove_percent": 0,
        },
    }


    # 4) Schrijf een ROBUUSTE Config.toml met alle varianten
    config_toml = f"""
[project]
project_dir = "."
video_dir   = "."
output_dir  = "."
participant_height = {PARTICIPANT["height"]}
participant_mass   = {PARTICIPANT["mass"]}
frame_range = [{frame_range_list[0]}, {frame_range_list[1]}]

[data]
trc_dir = "pose-3d"

[participant]
height = {PARTICIPANT["height"]}
mass   = {PARTICIPANT["mass"]}
age    = {PARTICIPANT["age"]}
sex    = "{PARTICIPANT["sex"]}"

# Sommige builds lezen dit hieruit:
[subject]
height = {PARTICIPANT["height"]}
mass   = {PARTICIPANT["mass"]}

[markerAugmentation]
feet_on_floor  = false
use_lower_limb = true
use_upper_limb = true
verbose        = true

[kinematics]
frame_range = [{frame_range_list[0]}, {frame_range_list[1]}]
fastest_frames_to_remove_percent = 0
""".strip()


    # schrijf naar project root én kopieën (sommige builds zoeken elders)
    toml_path = PROJECT_ROOT / "Config.toml"
    toml_path.write_text(config_toml, encoding="utf-8")
    for dst in [PROJECT_ROOT/"config.toml", PROJECT_ROOT/"Pose2Sim"/"Config.toml", PROJECT_ROOT/"Pose2Sim"/"config.toml", PROJECT_ROOT/"pose-3d"/"Config.toml", PROJECT_ROOT/"pose-3d"/"config.toml"]:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(config_toml, encoding="utf-8")
        except Exception:
            pass

    print(f"  -> Config.toml(s) geschreven. frame_range = {frame_range_list}, height={PARTICIPANT['height']}, mass={PARTICIPANT['mass']}")

    # 5) Import en run
    try:
        from Pose2Sim import markerAugmentation as MA
    except Exception as e:
        print("Kon Pose2Sim.MarkerAugmenter niet importeren:", e)
        sys.exit(1)

    # Zorg dat cwd exact project root is (waar Config.toml ligt)
    os.chdir(PROJECT_ROOT)

    MA.augment_markers_all(config)

    out = POSE3D_DIR / (trc_path.stem + "_LSTM.trc")
    ensure_ok(out.exists(), f"Augmenter-output niet gevonden: {out}")
    print(f"  -> LSTM-output: {out}")
    return out

# ---------- STEP 4: REBUILD HEADER WITH *_STUDY NAMES ----------
def header_fix(lstm_trc: Path) -> Path:
    print("\n[4/4] Header fixen (voegt *_study namen toe)")
    with open(lstm_trc, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    if len(lines) < 6: 
        raise SystemExit("TRC lijkt onvolledig")

    h0,h1,h2,h3,h4,h5 = lines[:6]
    data = [ln for ln in lines[6:] if ln.strip()]
    first = data[0].split("\t")
    present = (len(first)-2)//3  # #triplets in data

    # bestaande namen tellen
    parts = h3.split("\t")
    names = []
    i = 2
    while i < len(parts):
        nm = parts[i].strip()
        if nm: names.append(nm)
        i += 3
    declared = len(names)

    # haal officiële response markers uit lokale Pose2Sim
    try:
        from Pose2Sim.markerAugmentation import getOpenPoseMarkers_lowerExtremity2, getMarkers_upperExtremity_noPelvis2
        _, resp_lower = getOpenPoseMarkers_lowerExtremity2()
        _, resp_upper = getMarkers_upperExtremity_noPelvis2()
        official = resp_lower + resp_upper   # ~43 namen
    except Exception as e:
        print("Kon response-markers niet ophalen:", e)
        official = []

    extra_needed = max(0, present - declared)
    to_add = official[:extra_needed]
    if extra_needed > len(official):
        # voeg placeholders toe zodat niets verschuift
        to_add += [f"extra_study{i+1}" for i in range(extra_needed - len(official))]

    new_names = names + to_add

    # header[2]: NumMarkers updaten
    h2p = h2.split("\t"); h2p[3] = str(len(new_names))
    new_h2 = "\t".join(h2p)

    # header[3]: namenrij
    h3_list = ["Frame#", "Time"]
    for m in new_names:
        h3_list += [m, "", ""]
    new_h3 = "\t".join(h3_list)

    # header[4]: X/Y/Z labels
    h4_list = ["", ""]
    for i in range(1, len(new_names)+1):
        h4_list += [f"X{i}", f"Y{i}", f"Z{i}"]
    new_h4 = "\t".join(h4_list)

    out = lstm_trc.with_name(lstm_trc.stem + "_fixed.trc")
    with open(out, "w", encoding="utf-8") as w:
        w.write("\n".join([h0, h1, new_h2, new_h3, new_h4, h5] + data))

    print(f"  -> Header gefixt: {out}")
    return out

# ---------- MAIN ----------
def main():
    # Vraag lengte/gewicht/leeftijd/geslacht via GUI
    global PARTICIPANT_DEFAULT
    PARTICIPANT = get_participant_via_gui(PARTICIPANT_DEFAULT)

    # 1) Kies video
    vid = choose_file("Kies een video voor MediaPipe 3D", [("Video", "*.mp4;*.avi;*.mov;*.mkv"), ("Alle bestanden", "*.*")])
    ensure_ok(bool(vid), "Geen video gekozen.")
    video_path = Path(vid)

    # 2) Video -> CSV
    csv_path = video_to_csv(video_path)

    # 3) CSV -> TRC (exacte 22 markers)
    trc_exact = csv_to_trc_exact(csv_path)

    # 4) Augmenter
    lstm_trc = run_augmenter(trc_exact, PARTICIPANT)

    # 5) Header fix
    fixed_trc = header_fix(lstm_trc)

    print("\n✅ Klaar!")
    print("Eindbestand:", fixed_trc)
    print("Je kunt het bekijken met:", 'python view_trc_study.py')
    print("  (zet daar TRC_PATH op:", fixed_trc.as_posix(), ")")

if __name__ == "__main__":
    main()
