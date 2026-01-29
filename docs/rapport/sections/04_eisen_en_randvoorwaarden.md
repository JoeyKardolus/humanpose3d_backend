# 4. Eisen en Randvoorwaarden

## 4.1 Projectcontext

Het systeem is ontwikkeld voor bewegingsanalyse in de gezondheidszorg, waar specifieke randvoorwaarden gelden. Patiëntprivacy vereist dat videodata lokaal verwerkt wordt zonder cloud-upload, conform AVG/GDPR-wetgeving. De doelgroep—zorgprofessionals zonder IT-achtergrond—bepaalt dat het systeem gebruiksvriendelijk moet zijn en moet draaien op standaard hardware die typisch beschikbaar is in zorginstellingen.

Voor toekomstige inzet als medisch hulpmiddel zal certificering volgens de Medical Device Regulation (MDR) onderzocht moeten worden. De huidige versie is gepositioneerd als onderzoekstool en valt daarom niet onder CE-markeringsvereisten.

## 4.2 Kernvereisten

| Vereiste | Specificatie | Status |
|----------|--------------|--------|
| **Lokale verwerking** | Volledig op laptop, geen cloud-upload | ✓ |
| **Webapp interface** | Gebruiksvriendelijke GUI voor niet-technische gebruikers | ✓ |
| **ISB-compliant output** | Gewrichtshoeken volgens Wu et al. (2002, 2005) | ✓ |
| **Standaard hardware** | Werkt op gemiddelde laptop (geen dedicated GPU vereist) | ✓ |
| **Eenvoudige input** | Enkele video van smartphone-kwaliteit (720p+) | ✓ |

## 4.3 Functionele Eisen

| ID | Eis | Prioriteit | Status |
|----|-----|------------|--------|
| F1 | Video upload via webapp | Must | ✓ |
| F2 | 3D pose reconstructie uit 2D keypoints | Must | ✓ |
| F3 | Marker augmentatie (22 → 64 markers) | Must | ✓ |
| F4 | Gewrichtshoeken berekening (12 gewrichtsgroepen) | Must | ✓ |
| F5 | TRC export voor interoperabiliteit met OpenSim/Visual3D | Must | ✓ |
| F6 | Visualisatie van gewrichtshoeken (CSV + grafieken) | Should | ✓ |

## 4.4 Niet-Functionele Eisen

| ID | Eis | Specificatie | Status |
|----|-----|--------------|--------|
| N1 | Verwerkingstijd | < 2 minuten per video | ✓ (~60s) |
| N2 | Geheugengebruik | < 8 GB RAM | ✓ |
| N3 | POF nauwkeurigheid | < 15° limb-oriëntatiefout | ✓ (~7° gerealiseerd) |
| N4 | GPU fallback | Automatische CPU-modus indien geen GPU | ✓ |

## 4.5 Input/Output Specificaties

**Input:**
- Videoformaten: MP4, AVI, MOV (standaard codecs)
- Minimale kwaliteit: 720p (smartphone-camera voldoende)
- Perspectief: frontaal of schuin aanzicht
- Belichting: normale binnenverlichting

**Output:**
- TRC-bestanden (Track Row Column) voor biomechanica-software
- CSV-bestanden met gewrichtshoeken per frame
- PNG-visualisaties van gewrichtshoekverloop
- Ruwe landmark-data voor debugging en verder onderzoek

## 4.6 Realisatie

Alle must-have requirements zijn gerealiseerd. Het systeem draait lokaal op een gemiddelde laptop, biedt een webapp interface, en genereert ISB-compliant gewrichtshoeken. De experimentele POF-module overtreft de gestelde nauwkeurigheidseis ruimschoots (~7° versus de eis van <15°), al vereist volledige productie-integratie nog aanvullend werk zoals beschreven in sectie 5.5.


