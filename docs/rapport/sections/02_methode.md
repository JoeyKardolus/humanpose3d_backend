# 2. Methode

## 2.1 Ontwikkelmethode

Het project is ontwikkeld volgens een iteratieve, pragmatische aanpak. Modules zijn incrementeel gebouwd met directe visuele feedback: 3D skelet-visualisaties maakten fouten direct zichtbaar (verkeerde diepte, gespiegelde ledematen, onrealistische hoeken). Deze snelle visuele feedback-loop stuurde de ontwikkeling.

De ontwikkeling volgde deze kernprincipes:

- **Modulaire architectuur:** Scheiding van concerns tussen modules waar praktisch
- **Visuele validatie:** 3D reconstructies direct beoordeeld op anatomische plausibiliteit
- **Incrementele ontwikkeling:** Werkende code eerst, documentatie volgde

## 2.2 Technische Aanpak

De codebase is modulair opgezet om experimenten mogelijk te maken. Componenten kunnen onafhankelijk worden vervangen (bijv. POF-model wisselen zonder pipeline-aanpassingen). Dit was praktisch noodzakelijk gezien de experimentele aard van het project.

De pipeline is opgebouwd uit componenten die sequentieel data transformeren:

```
Video → MediaPipe → POF 3D → TRC → Pose2Sim → Joint Angles → Output
```

## 2.3 Validatiestrategie

Validatie vond plaats op meerdere niveaus:

| Niveau | Methode | Toelichting |
|--------|---------|-------------|
| **POF model** | Kwantitatief (train/val loss) | ~7° error op AIST++ validatieset |
| **3D reconstructie** | Visuele inspectie | Fouten direct zichtbaar in 3D viewer |
| **Gewrichtshoeken** | Validatie met opdrachtgever | Vergelijking met verwachte bewegingspatronen |
| **End-to-end** | Praktijktest | Verwerking van eigen opgenomen video's |

De primaire validatiemethode was visuele inspectie: fouten in 3D reconstructie (verkeerde diepte, gespiegelde ledematen) zijn met het blote oog waarneembaar. Voor gewrichtshoeken werd de output besproken met de opdrachtgever om te verifiëren dat de waarden overeenkwamen met verwachte bewegingspatronen.

Formele kwantitatieve validatie tegen marker-based ground truth in een klinische setting is niet uitgevoerd. Dit blijft een aanbeveling voor vervolgonderzoek (zie Discussie).

## 2.4 Tooling

| Tool | Doel | Referentie |
|------|------|------------|
| Python 3.11 | Primaire programmeertaal | - |
| PyTorch | Neural network training | - |
| MediaPipe | 2D/3D pose detection | Lugaresi et al. (2019) |
| Pose2Sim | Marker augmentatie | Pagnon et al. (2022) |
| OpenCV | Video I/O | - |
| NumPy/SciPy | Numerieke berekeningen | - |
