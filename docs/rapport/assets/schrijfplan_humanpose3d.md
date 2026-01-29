    # Schrijfplan: HumanPose3D Rapport

**Minor Zorg en Technologie**  
**Auteurs:** Joey Kardolus, Max Jansen  
**Opdrachtgever:** Jaap Jansen (Hogeschool Utrecht, hoofddocent IBS)

---

## 1. Inleiding

| Onderdeel | Inhoud | Bron |
|-----------|--------|------|
| **Aanleiding** | Huidige motion capture systemen zijn duur (marker-based) of onnauwkeurig (consument markerless). Behoefte aan betaalbare, accurate 3D pose-analyse voor zorgtoepassingen. | README context |
| **Klantwensen** | Markerless motion capture die in gezondheidszorg context bruikbaar is: accurate gewrichtshoeken, geen dure apparatuur, geschikt voor niet-technische gebruikers. | Afstemming met opdrachtgever |
| **Probleemstelling** | MediaPipe levert 2D accurate poses maar de 3D diepte-schatting is onbetrouwbaar (~16° fout). Hoe kunnen we betrouwbare 3D reconstructie bereiken vanuit monoculaire video? | POF_EXPLANATION.md |
| **Opdrachtomschrijving** | Ontwikkeling van een end-to-end pipeline: video → 3D pose → biomechanisch model → gewrichtshoeken, geschikt voor zorgprofessionals. met een makkelijk te gebruiken GUI en lokaal draaiend op gemiddelde laptops | ARCHITECTURE.md |

**Geschatte lengte:** ½–1 pagina

---

## 2. Methode

| Onderdeel | Inhoud |
|-----------|--------|
| **Ontwikkelmethode** | Iteratieve ontwikkeling met continue validatie. Elke module apart ontwikkeld en getest voordat integratie. |
| **Technische aanpak** | Object-georiënteerde architectuur, strikte scheiding concerns (AGENTS.md principes) |
| **Validatie** | Vergelijking met ground truth data (AIST++ dataset), ISB-standaarden voor gewrichtshoeken | en een serie van eigen gemaakte test videos voor inference

**Geschatte lengte:** ¼–½ pagina

---

## 3. Huidige Oplossingen (State of the Art)

| Oplossing | Sterkte | Zwakte |
|-----------|---------|--------|
| **Marker-based MoCap** (Vicon, OptiTrack) | Gouden standaard nauwkeurigheid | Duur (€50K+), markers nodig, lab-setting |
| **MediaPipe standalone** | Gratis, real-time, geen markers | Slechte diepte-schatting |
| **VideoPose3D** (Pavllo et al., 2019) | Temporele consistentie | Vereist multi-view of lifting |
| **ElePose** (Wandt et al., 2022) | Unsupervised | Beperkte anatomische constraints |
| **Pose2Sim** (Pagnon et al., 2022) | Open-source biomechanica workflow | Vereist multi-camera setup |

**Relevante literatuur:** Pavllo 2019, Wandt 2022, Pagnon 2022, Lugaresi 2019

**Geschatte lengte:** ½–1 pagina

---

## 4. Randvoorwaarden

| Categorie | Voorwaarde |
|-----------|------------|
| **Hardware** | Moet werken op standaard laptop (geen GPU vereist, wel aanbevolen) |
| **Input** | Enkele video (smartphone kwaliteit voldoende) |
| **Output** | ISB-compliant gewrichtshoeken, TRC formaat voor interoperabiliteit |
| **Gebruiker** | Geen technische kennis vereist voor basisgebruik |
| **Privacy** | Verwerking lokaal, geen cloud-upload vereist |

**Geschatte lengte:** ¼ pagina

---

## 5. Programma van Eisen (PvE)

Programma van Eisen (kernpunten)
moet op laptop kunnen, met lokale webapp. gewrichthoeken uitgeven etc. maar hou simpel. 

**Bron:** AGENTS.md, ARCHITECTURE.md

**Geschatte lengte:** ½ pagina hoofdtekst + 1–2 pagina bijlage

---

## 6. Bevindingen en Ontwerp/Ontwikkelkeuzes

### 6.1 Architectuurkeuzes

- **Waarom modulair?** → Onderhoudbaarheid, testbaarheid, losse koppeling tussen modules
- **Waarom POF i.p.v. globale camera-hoeken?** → Per-limb diepte lost het "arm naar camera, been opzij" probleem op (zie POF_EXPLANATION.md)
- **Waarom Pose2Sim LSTM?** → Bewezen augmentatie methode, 22→64 markers, GPU-acceleratie mogelijk

### 6.2 Pipeline Evolutie

Chronologisch overzicht van belangrijke beslissingen (uit CHANGELOG.md):

| Datum | Beslissing | Reden |
|-------|------------|-------|
| 2025-11 | MediaPipe + Pose2Sim basis | Snelle prototype |
| 2026-01-10 | Pelvis angle fixes | Validatie toonde 180° flips |
| 2026-01-13 | Neural refinement toegevoegd | 53.6% depth error reductie |
| 2026-01-21 | POF-only architectuur | Depth module verwijderd, POF simpeler en beter |

### 6.3 Technische Afwegingen

| Keuze | Alternatieven | Beslissing | Rationale |
|-------|---------------|------------|-----------|
| POF vs. Depth Refinement | Depth delta correcties | POF | ~11° vs ~16° error, minder parameters |
| GNN vs. Transformer | - | Beide geïmplementeerd | SemGCN heeft anatomische bias, transformer meer flexibel |
| Augmentation cycles | 1, 5, 10, 20 | 20 cycles | Balans tussen nauwkeurigheid en rekentijd |
| ISB standaard | Custom angles | ISB | gezondheidszorg interoperabiliteit |

### 6.4 Visualisaties (toe te voegen)

- [ ] Pipeline flowchart (ARCHITECTURE.md diagram)
- [ ] Joint angle visualisatie voorbeeld (7×2 grid)
- [ ] Marker progressie diagram (33→22→64)
- [ ] Screenshot webapp interface

**Geschatte lengte:** 2–3 pagina's

---

## 7. Resultaten

### 7.1 Kwantitatieve Resultaten

| Metric | Waarde | Referentie |
|--------|--------|------------|
| POF limb orientation error | ~11° | MediaPipe 3D: 16.3° |
| Processing time | ~60s | 535 frames, 20 cycles, neural refinement |
| Marker quality | 59/64 markers | Na filtering onbetrouwbare markers |
| Joint groups computed | 12 | Pelvis, hip, knee, ankle, trunk, shoulder, elbow |
| Depth error reduction | 53.6% | 11.6cm → 5.4cm (eerste training run) |

### 7.2 Visualisaties (toe te voegen)

- [ ] Voorbeeld output van echte video
- [ ] Joint angle plots (voor/na refinement)
- [ ] Before/after POF vergelijking
- [ ] Output directory structuur

**Geschatte lengte:** 1–1½ pagina

---

## 8. Discussie

| Aspect | Inhoud |
|--------|--------|
| **Sterke punten** | Monoculaire input, geen markers, ISB-compliant, open-source stack, automatische CPU fallback |
| **Beperkingen** | Single-view occlusie probleem, geen real-time (nog), validatie beperkt tot AIST++ data |
| **Vergelijking literatuur** | POF aanpak gebaseerd op Xiang et al. 2019 (MonocularTotalCapture), maar vereenvoudigd voor single-view |
| **Methodologische overwegingen** | Ground truth uit MoCap datasets, niet in zorgsetting gevalideerd |
| **Technische schuld** | Sommige modules kunnen verder geoptimaliseerd |

**Geschatte lengte:** ½–1 pagina

---

## 9. Conclusie

Te beantwoorden:
- Hoofdvraag: "Is accurate markerless 3D pose analyse mogelijk vanuit smartphone video?"
- Samenvatting key results (POF ~11° error, 59/64 markers, 12 joint groups)
- Terugkoppeling naar klantwensen (betaalbaar, geen markers,  bruikbaar)
- Mate waarin PvE is gerealiseerd

**Geschatte lengte:** ¼–½ pagina

---

## 10. Aanbevelingen

### 10.1 Doorontwikkeling

| Prioriteit | Aanbeveling | Toelichting |
|------------|-------------|-------------|
| Hoog | Real-time processing | Huidige ~60s per video naar live feedback |
| Middel | Multi-view support | Oplossing voor occlusie probleem |
Vervolg onderzoek naar meer accurate 2d pose prediction (mediapipe heeft mean error van 10 cm vergeleken met ground truth)​

Hieruit POF model verbeteren met doel sub 5 cm mean error​

Uitbreiding training data, met veel variatie camera perspectieven ​

Gewrichts model verbeteren met meer accurate basis​

Start ontwikkeling interpretatie laag (bijv. Veel flexie >90 graden arm, betekend waarschijnlijk hypermobiel)​


### 10.2 Marktintroductie

- **Doelgroep:** Fysiotherapie praktijken, revalidatiecentra, sportanalyse
- **Businessmodel:** SaaS of on-premise licentie
- **Certificering:** Medische device regelgeving verkennen (CE markering, MDR classificatie)
- **Concurrentievoordeel:** Open-source basis, geen hardware lock-in

**Geschatte lengte:** ½ pagina

---

## 11. Literatuurlijst

### Core Methods & Architecture

- Xiang, D., Joo, H., & Sheikh, Y. (2019). Monocular total capture: Posing face, body, and hands in the wild. *CVPR*, 10965-10974.
- Keller, M., et al. (2024). MANIKIN: Biomechanically accurate neural IK. *ECCV*.

### ISB Biomechanics Standards

- Wu, G., et al. (2002). ISB recommendation on definitions of joint coordinate system—Part I: Ankle, hip, and spine. *Journal of Biomechanics*, 35(4), 543-548.
- Wu, G., et al. (2005). ISB recommendation—Part II: Shoulder, elbow, wrist and hand. *Journal of Biomechanics*, 38(5), 981-992.
- Grood, E. S., & Suntay, W. J. (1983). A joint coordinate system for the clinical description of three-dimensional motions. *Journal of Biomechanical Engineering*, 105(2), 136-144.

### Pose Detection & Marker Augmentation

- Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. *arXiv:1906.08172*.
- Pagnon, D., Domalain, M., & Reveret, L. (2022). Pose2Sim: An end-to-end workflow for 3D markerless sports kinematics—Part 2: Accuracy. *Sensors*, 21(19), 6530.
- Pagnon, D. (2022). Pose2Sim: An open-source Python package for multiview markerless kinematics. *JOSS*, 7(77), 4362.

### Monocular 3D Pose Estimation

- Pavllo, D., et al. (2019). 3D human pose estimation in video with temporal convolutions. *CVPR*, 7753-7762.
- Wandt, B., et al. (2022). ElePose: Unsupervised 3D human pose estimation. *CVPR*, 6635-6645.

### Surveys

- Chen, Y., et al. (2024). Enhancing 3D human pose estimation with bone length adjustment. *arXiv:2410.20731*.
- Survey of monocular 3D pose estimation. (2025). *Sensors*, 25(8), 2409.

### Anthropometric References

- Winter, D. A. (2009). *Biomechanics and motor control of human movement* (4th ed.). Wiley.
- NASA. (1978). *NASA anthropometric source book* (NASA Reference Publication 1024).
- Drillis, R., & Contini, R. (1966). Body segment parameters (Technical Report No. 1166-03). NYU.

### Datasets

- AIST++ Dance Video Database. https://google.github.io/aistplusplus_dataset/
- Ionescu, C., et al. (2014). Human3.6M: Large scale datasets for 3D human sensing. *IEEE TPAMI*, 36(7), 1325-1339.
- Joo, H., et al. (2019). Panoptic studio: A massively multiview system. *IEEE TPAMI*, 41(1), 190-204.

---

## 12. Bijlagen

| Bijlage | Inhoud |
|---------|--------|
| A | Volledig Programma van Eisen (frontend/backend/pipeline) |
| B | Plan van Aanpak |
| C | Pipeline flowchart (groot formaat uit ARCHITECTURE.md) |
| D | Code architectuur / module overzicht |
| E | Voorbeeld output files (TRC, CSV, PNG) |
| F | AGENTS.md (development guidelines) |

---

## Geschatte Totale Omvang

| Sectie | Pagina's |
|--------|----------|
| Voorblad + inhoudsopgave | 1–2 |
| Hoofdtekst (secties 1–10) | 8–12 |
| Literatuurlijst | 1–2 |
| Bijlagen | 3–5 |
| **Totaal** | **13–20 pagina's** |

---

## Notities voor Uitwerking

### Te verzamelen materiaal
- [ ] Screenshots webapp interface
- [ ] Joint angle output voorbeeld
- [ ] Pipeline diagram (vector/hoge resolutie)
- [ ] Vergelijkende tabel POF vs MediaPipe

### Aandachtspunten
- Anoniem houden (geen specifieke klantnamen)
- Technisch maar toegankelijk voor zorg-publiek
- Focus op relevantie voor de gezondheidszorg, niet alleen technische prestaties

---

*Laatst bijgewerkt: 2026-01-24*
