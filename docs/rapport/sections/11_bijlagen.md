# 11. Bijlagen

## Bijlage A: Volledig Programma van Eisen

### A.1 Frontend Requirements

| ID | Eis | Prioriteit | Rationale |
|----|-----|------------|-----------|
| F1 | Plain HTML, geen inline JavaScript | Must | Scheiding concerns, onderhoudbaarheid |
| F2 | Bootstrap voor styling | Must | Responsive, moderne UI |
| F3 | Responsive design (mobile-friendly) | Should | Toegankelijkheid op tablets |
| F4 | Geen inline CSS, dedicated .css files | Must | Onderhoudbaarheid |
| F5 | Accessibility (WCAG 2.1 AA) | Could | Inclusiviteit |

### A.2 Backend Requirements

| ID | Eis | Prioriteit | Rationale |
|----|-----|------------|-----------|
| B1 | Django MVT architectuur | Must | Bewezen framework, Python ecosystem |
| B2 | Strikte OOP, Single Responsibility Principle | Must | Onderhoudbaarheid, testbaarheid |
| B3 | Modulaire pipeline (losse componenten) | Must | Flexibiliteit, herbruikbaarheid |
| B4 | GPU fallback naar CPU | Must | Brede hardware ondersteuning |
| B5 | ISB-compliant gewrichtshoeken | Must | Standaard in de gezondheidszorg |
| B6 | Applicatielogica alleen in `src/application/` | Must | Scheiding concerns |
| B7 | Type hints op alle functies | Should | Code kwaliteit |
| B8 | Logging op alle pipeline stappen | Should | Debugging, monitoring |

### A.3 Pipeline Requirements

| ID | Eis | Prioriteit | Rationale |
|----|-----|------------|-----------|
| P1 | 33 landmarks → 22 → 64 markers progressie | Must | Volledige biomechanische marker-set |
| P2 | POF 3D reconstructie (<15° error) | Must | Kernfunctionaliteit |
| P3 | Neural joint refinement | Should | Verbeterde nauwkeurigheid |
| P4 | Automatische output organisatie | Must | Gebruiksvriendelijkheid |
| P5 | TRC export voor biomechanische analyse | Must | Interoperabiliteit |
| P6 | CSV export gewrichtshoeken | Must | Analyse in spreadsheets |
| P7 | PNG visualisaties per gewricht | Should | Snelle visuele inspectie |

### A.4 Performance Requirements

| ID | Eis | Prioriteit | Target |
|----|-----|------------|--------|
| R1 | Verwerking per video | Should | <120s (30s video, 30fps) |
| R2 | Memory gebruik | Must | <8GB RAM |
| R3 | Stabiele output | Must | Geen crashes tijdens verwerking |
| R4 | GPU memory | Should | <4GB VRAM |

---

## Bijlage B: Plan van Aanpak

*[In te vullen: oorspronkelijk projectplan met fasering en milestones]*

---

## Bijlage C: Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: Video File                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: MediaPipe Detection                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ Video Read  │───▶│ Pose Detect │───▶│ 33 Landmarks│             │
│  │ (OpenCV)    │    │ (MediaPipe) │    │ 2D + 3D     │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: POF 3D Reconstruction                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ 17 COCO     │───▶│ SemGCN-     │───▶│ 14 Limb     │             │
│  │ Keypoints   │    │ Temporal    │    │ Orientations│             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                              │                      │
│                           ┌──────────────────┘                      │
│                           ▼                                         │
│                    ┌─────────────┐    ┌─────────────┐              │
│                    │ Least-Sq    │───▶│ 17 Joints   │              │
│                    │ Solver      │    │ 3D Metric   │              │
│                    └─────────────┘    └─────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: TRC Conversion                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ 17 COCO     │───▶│ Mapping +   │───▶│ 22 Pose2Sim │             │
│  │ Joints      │    │ Derived     │    │ Markers     │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: Pose2Sim Augmentation                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ 22 Markers  │───▶│ LSTM Model  │───▶│ 64 Markers  │             │
│  │ (TRC)       │    │ (GPU/CPU)   │    │ Augmented   │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 5: Joint Angle Computation                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ 64 Markers  │───▶│ ISB Segment │───▶│ 12 Joint    │             │
│  │             │    │ Coord Sys   │    │ Groups      │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 6: Neural Joint Refinement                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ Raw Angles  │───▶│ Transformer │───▶│ Refined     │             │
│  │             │    │ Model       │    │ Angles      │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT: Organized Files                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ data/output/pose-3d/<video>/                                  │ │
│  │ ├── <video>_final.trc           # 59-64 markers               │ │
│  │ ├── <video>_initial.trc         # Initial 22 markers          │ │
│  │ └── joint_angles/                                             │ │
│  │     ├── all_joint_angles.csv    # Combined angles             │ │
│  │     ├── pelvis_angles.csv/png   # Per-joint files             │ │
│  │     └── ...                     # 12 joint groups             │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Bijlage D: Module Overzicht

| Module | Pad | Verantwoordelijkheid |
|--------|-----|----------------------|
| `mediastream` | `src/mediastream/` | Video I/O via OpenCV |
| `posedetector` | `src/posedetector/` | MediaPipe inference, landmark mapping |
| `datastream` | `src/datastream/` | CSV/TRC conversie, marker estimation |
| `markeraugmentation` | `src/markeraugmentation/` | Pose2Sim integratie, GPU acceleratie |
| `kinematics` | `src/kinematics/` | ISB joint angles, Euler decomposition |
| `visualizedata` | `src/visualizedata/` | 3D plotting, skeleton connections |
| `pof` | `src/pof/` | POF models, LS solver, metric scale |
| `joint_refinement` | `src/joint_refinement/` | Neural joint constraint model |
| `pipeline` | `src/pipeline/` | Orchestratie, cleanup |
| `application` | `src/application/` | Django web interface |

---

## Bijlage E: Voorbeeld Output Files

### E.1 TRC Header (voorbeeld)

```
PathFileType    4    (X/Y/Z)    output.trc
DataRate    CameraRate    NumFrames    NumMarkers    Units    OrigDataRate    OrigDataStartFrame    OrigNumFrames
30.0    30.0    535    59    m    30.0    1    535
Frame#    Time    Nose            RShoulder        LShoulder        ...
            X    Y    Z    X    Y    Z    X    Y    Z    ...
1    0.000    0.123    1.456    0.789    ...
```

### E.2 Joint Angles CSV (voorbeeld)

```csv
frame,time,pelvis_flex_deg,pelvis_abd_deg,pelvis_rot_deg,left_hip_flex_deg,...
0,0.000,5.2,-2.1,3.4,15.7,...
1,0.033,5.3,-2.0,3.5,16.1,...
```

---

## Bijlage F: Development Guidelines

*Zie `AGENTS.md` in de repository voor volledige development richtlijnen.*

Kernprincipes:
- Single Responsibility Principle
- Applicatielogica gescheiden van domeinlogica
- Type hints op alle functies
- Imperative commit messages
