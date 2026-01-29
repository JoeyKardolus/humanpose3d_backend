# 3. Huidige Oplossingen (State of the Art)

## 3.1 Overzicht Bestaande Oplossingen

| Oplossing | Type | Sterkte | Zwakte |
|-----------|------|---------|--------|
| **Vicon/OptiTrack** | Marker-based | Gouden standaard nauwkeurigheid (<1mm) | Duur (€50K+), markers nodig, lab-setting |
| **MediaPipe Pose** (Lugaresi et al., 2019) | Markerless, monoculair | Gratis, real-time, 33 landmarks | Onbetrouwbare 3D diepte-schatting |
| **VideoPose3D** (Pavllo et al., 2019) | 2D-to-3D lifting | Temporele consistentie, state-of-the-art | Vereist voorgetraind 2D model |
| **ElePose** (Wandt et al., 2022) | Unsupervised 3D | Geen gelabelde data nodig | Beperkte anatomische constraints |
| **Pose2Sim** (Pagnon et al., 2022) | Multi-view workflow | Open-source, biomechanisch model | Vereist meerdere camera's |

## 3.2 Marker-Based Motion Capture

Systemen zoals Vicon en OptiTrack gebruiken infrarood camera's om reflecterende markers te volgen. Deze aanpak biedt submillimeter nauwkeurigheid en wordt beschouwd als de gouden standaard voor biomechanische analyse.

**Beperkingen voor zorgtoepassingen:**
- Hoge aanschafkosten (€50.000 - €500.000)
- Vereist gespecialiseerde ruimte
- Markers kunnen beweging beïnvloeden
- Niet geschikt voor thuisgebruik of veldmetingen

## 3.3 MediaPipe Pose

Google's MediaPipe (Lugaresi et al., 2019) biedt real-time pose-detectie vanuit enkele camera-beelden. Het systeem detecteert 33 landmarks en levert zowel 2D als 3D coördinaten.

**Sterke punten:**
- Gratis en open-source
- Real-time performance op CPU
- Robuuste 2D detectie

**Zwakke punten:**
- 3D diepte gebaseerd op heuristieken, niet op werkelijke diepte-informatie
- Gemiddelde oriëntatiefout van 16.3° in onze tests
- Geen anatomische constraints

## 3.4 VideoPose3D

Pavllo et al. (2019) presenteren een aanpak die temporele convoluties gebruikt om 2D poses te "liften" naar 3D. Door meerdere frames te analyseren wordt temporele consistentie afgedwongen.

**Relevantie voor ons project:**
- Demonstreert waarde van temporele context
- Getraind op Human3.6M dataset (Ionescu et al., 2014)
- Architectuur inspiratie voor onze POF-aanpak

## 3.5 ElePose

Wandt et al. (2022) introduceren ElePose, dat camera-elevatie voorspelt en normalizing flows leert op 2D poses. Dit maakt unsupervised training mogelijk zonder gelabelde 3D data.

**Beperking:** De globale camera-aanpak lost het per-limb diepte-probleem niet op.

## 3.6 Pose2Sim

Pagnon et al. (2022) bieden een open-source workflow voor multi-camera markerless kinematics:
- Marker augmentatie via LSTM (22 → 64 markers)
- OpenSim integratie voor biomechanische analyse
- ISB-compliant gewrichtshoeken (Wu et al., 2002, 2005)

**Onze integratie:** We gebruiken de Pose2Sim marker-augmentatie module als onderdeel van onze single-camera pipeline.

## 3.7 Kennislacune

Geen van de bestaande oplossingen combineert:
1. Monoculaire input (enkele smartphone-camera)
2. Accurate per-limb 3D reconstructie
3. Volledige biomechanische marker-set
4. ISB-compliant gewrichtshoeken
