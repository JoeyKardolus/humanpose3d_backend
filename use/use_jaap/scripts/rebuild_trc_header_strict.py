# rebuild_trc_header_strict.py
# Herbouwt de TRC-header van jouw *_LSTM.trc robuust en schrijft altijd een *_LSTM_fixed.trc.
# - Leest werkelijke aantal marker-triplets uit de data
# - Haalt de officiële response-markers uit jouw lokale Pose2Sim-code
# - Vult precies zoveel namen aan als er extra triplets zijn
# - Schrijft pose-3d\..._LSTM_fixed.trc en verifieert dat het bestaat

import os, sys

POSE3D = os.path.join(".", "pose-3d")
IN_FILE = os.path.join(POSE3D, "pose2sim_input_exact_LSTM.trc")
OUT_FILE = IN_FILE.replace("_LSTM.trc", "_LSTM_fixed.trc")


def split_tabs(s): return s.rstrip("\n\r").split("\t")
def join_tabs(lst): return "\t".join(lst)

if not os.path.isfile(IN_FILE):
    sys.exit(f"Input TRC niet gevonden: {IN_FILE}")

with open(IN_FILE, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.read().splitlines()
if len(lines) < 6:
    sys.exit("TRC lijkt onvolledig (te weinig headerregels).")

h0,h1,h2,h3,h4,h5 = lines[:6]
data = [ln for ln in lines[6:] if ln.strip()]
if not data:
    sys.exit("Geen data-rijen gevonden.")

# Huidige namen uit header[3]
h3p = split_tabs(h3)
names = []
i = 2
while i < len(h3p):
    nm = h3p[i].strip()
    if nm:
        names.append(nm)
    i += 3

declared = len(names)
present = (len(split_tabs(data[0])) - 2) // 3  # per rij: Frame#, Time + 3*markers

print(f"Header markers (declared): {declared}")
print(f"Markers in data (present) : {present}")

if present == declared:
    print("Header klopt al — er wordt een kopie met '_LSTM_fixed.trc' geschreven (identiek).")
    with open(OUT_FILE, "w", encoding="utf-8") as w:
        w.write("\n".join(lines))
    print(f"OK: {OUT_FILE}")
    sys.exit(0)

if present < declared:
    sys.exit("Data bevat minder marker-triplets dan de header aangeeft — dat is onverwacht.")

# Haal officiële response-markers op uit jouw lokale Pose2Sim
try:
    from Pose2Sim.markerAugmentation import getOpenPoseMarkers_lowerExtremity2, getMarkers_upperExtremity_noPelvis2
except Exception as e:
    sys.exit(f"Kon response-markers niet importeren uit Pose2Sim: {e}")

_, resp_lower = getOpenPoseMarkers_lowerExtremity2()
_, resp_upper = getMarkers_upperExtremity_noPelvis2()
response_markers_all = resp_lower + resp_upper  # totale lijst (orde belangrijk)

needed = present - declared
if needed > len(response_markers_all):
    print(f"Waarschuwing: er zijn {needed} extra triplets, maar slechts {len(response_markers_all)} bekende response-markers.")
to_add = response_markers_all[:needed]

print(f"Voeg {len(to_add)} namen toe: {to_add[:8]}{' ...' if len(to_add)>8 else ''}")

# Herbouw header[2]/[3]/[4]
# header[2]: DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
h2p = split_tabs(h2)
h2p[3] = str(present)  # zet NumMarkers gelijk aan werkelijk aantal triplets in data
new_h2 = join_tabs(h2p)

# Nieuwe namenregel
new_names = names + to_add
h3_list = ["Frame#", "Time"]
for m in new_names:
    h3_list += [m, "", ""]
new_h3 = join_tabs(h3_list)

# Nieuwe X/Y/Z subheader
h4_list = ["", ""]
for i in range(1, present+1):
    h4_list += [f"X{i}", f"Y{i}", f"Z{i}"]
new_h4 = join_tabs(h4_list)

# Schrijf OUT_FILE
with open(OUT_FILE, "w", encoding="utf-8") as w:
    w.write("\n".join([h0, h1, new_h2, new_h3, new_h4, h5] + data))

# Verifieer
if not os.path.isfile(OUT_FILE):
    sys.exit("Schrijven mislukt: output niet gevonden.")
print(f"OK: {OUT_FILE}")
