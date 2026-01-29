# 6. Resultaten

## 6.1 Baseline Pipeline

De functionele baseline pipeline—MediaPipe pose-detectie gecombineerd met Pose2Sim marker-augmentatie (Pagnon et al., 2021, 2022)—levert consistente resultaten op standaard hardware. Een typische video van 535 frames wordt verwerkt in ongeveer 60 seconden, ruim binnen de gestelde eis van twee minuten. De verwerking vindt volledig lokaal plaats op een laptop zonder dedicated GPU, conform de privacy-vereisten voor zorgtoepassingen.

De marker-augmentatie expandeert de initiële 22 Pose2Sim-compatibele markers naar gemiddeld 59 van de 64 mogelijke biomechanische markers. De ontbrekende worden voornamelijk veroorzaakt door onstabiele COCO markers voorspelt door media pipe of RTM pose.

De gewrichtshoekberekening volgt de ISB-standaard (Wu et al., 2002, 2005) en de Grood & Suntay conventie (1983) voor knie-analyse. De pipeline berekent twaalf gewrichtsgroepen: pelvis, linker en rechter heup, knie en enkel, trunk, linker en rechter schouder, en elleboog. Elk gewricht levert drie Euler-hoeken volgens de anatomische conventies—flexie/extensie, abductie/adductie, en interne/externe rotatie—met uitzondering van de elleboog die als scharniergewricht alleen flexie/extensie kent.

De nauwkeurigheid van de baseline wordt beperkt door MediaPipe's diepte-schatting, die een gemiddelde limb-oriëntatiefout van ongeveer 16° vertoont (Lugaresi et al., 2019). Deze fout is acceptabel voor screening en voortgangsmonitoring, maar onvoldoende voor nauwkeurige diagnostiek.

## 6.2 POF Model Performance

Het experimentele POF-model, gebaseerd op de Part Orientation Fields aanpak van Xiang et al. (2019), toont significante verbeteringen op validatiedata. De SemGCN-Temporal architectuur bereikt een gemiddelde limb-oriëntatiefout van ongeveer 7°—meer dan een halvering ten opzichte van de MediaPipe baseline.

De training op de AIST++ dataset (Li et al., 2021), bestaande uit 1.2 miljoen frames van professionele dansers, verliep efficiënt. Het model bereikte al na één epoch een validatie-error van 7.62°, wat direct beter was dan de transformer-variant met 11° error. Na vijftig epochs stabiliseerde de performance rond 6° met een Z-sign classificatie-nauwkeurigheid van ongeveer 95%. Deze Z-sign voorspelling—of een ledemaat naar de camera of van de camera af wijst—is cruciaal voor het oplossen van de diepte-ambiguïteit.

De modelarchitectuur is compact met ongeveer 1.7 miljoen parameters, waardoor inference snel genoeg is voor praktisch gebruik. De combinatie van Semantic Graph Convolutions met temporele context blijkt effectiever dan pure transformer-architecturen voor deze taak, vermoedelijk door de ingebouwde anatomische structuur van het skelet-graaf.

Een belangrijke kanttekening is dat deze resultaten zijn gemeten op data uit dezelfde distributie als de trainingsdata. Hoe het model presteert op zorg-specifieke bewegingen—langzame revalidatie-oefeningen, bewegingen van ouderen, of pathologische patronen—is nog niet gevalideerd. De AIST++ dataset bevat uitsluitend snelle, expressieve dansbewegingen van jonge, fitte performers.

## 6.3 Joint Constraint Model Status

Het Joint Constraint model is geïmplementeerd en getraind op 660.000 samples uit de AIST++ dataset. Het model leert zachte correcties voor gewrichtshoeken in plaats van harde anatomische limieten toe te passen. Deze aanpak is essentieel voor zorgtoepassingen waar abnormale bewegingspatronen—hypermobiliteit, compensatiestrategieën, pathologische bewegingen—juist gedetecteerd moeten worden.

De effectiviteit van dit model kan echter nog niet worden gevalideerd. Zoals beschreven in sectie 5.5.5, vereist zinvolle training een basis met voldoende lage ruis. De MediaPipe baseline met ~16° error produceert te veel stochastische variatie; het model zou voornamelijk ruis leren voorspellen in plaats van systematische correcties. Pas wanneer de POF-integratie is afgerond en de basis-error naar ~7° is teruggebracht, kan het Joint Constraint model effectief worden getraind en gevalideerd.

## 6.4 Vergelijking met Eisen

De baseline pipeline voldoet aan alle must-have requirements uit sectie 4. De verwerkingstijd van ongeveer 60 seconden blijft ruim onder de gestelde limiet van twee minuten. Het aantal gereconstrueerde markers—59 van 64—overtreft de minimumeis van 50. Alle twaalf gewrichtsgroepen worden berekend volgens ISB-standaard.

De experimentele POF-module overtreft de gestelde nauwkeurigheidseis ruimschoots: ~7° gemeten error versus de eis van maximaal 15°. Deze resultaten zijn echter gemeten op validatiedata en niet in productie-omstandigheden. De volledige integratie van POF in de pipeline, en daarmee de realisatie van deze nauwkeurigheidswinst voor eindgebruikers, vereist nog aanvullend werk aan de camera-space naar world-space transformatie en temporele stabiliteit.

## 6.5 Visualisaties

De pipeline genereert automatisch visualisaties van de gewrichtshoeken over tijd. Voor elk gewricht wordt een grafiek geproduceerd met de drie Euler-componenten, samen met een CSV-bestand voor verdere analyse in externe software. De 3D skelet-reconstructie kan worden geïnspecteerd via een interactieve viewer die deel uitmaakt van de development tools.

### 3D Skelet Reconstructie

![3D Skeleton](../assets/skeleton_3d_example.png)

*Figuur 6.1: 3D skelet-reconstructie met POF-model, gevisualiseerd vanuit meerdere hoeken.*

### MediaPipe vs POF Vergelijking

![POF Comparison](../assets/pof_comparison_example.png)

*Figuur 6.2: Vergelijking van MediaPipe baseline (links) en POF-reconstructie (rechts) tegen ground truth. De POF-reconstructie toont betere diepte-schatting.*

### Output Structuur

![Output Structure](../assets/output_structure.png)

*Figuur 6.3: Overzicht van de automatisch gegenereerde output-bestanden per video.*
