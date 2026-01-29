# HumanPose3D: Markerless 3D Pose Analyse voor de Gezondheidszorg

**Minor Zorg en Technologie**
**Auteurs:** Joey Kardolus, Max Jansen
**Opdrachtgever:** Jaap Jansen (Hogeschool Utrecht)
**Datum:** Januari 2026

---

## Inhoudsopgave

1. [Inleiding](sections/01_inleiding.md)
2. [Methode](sections/02_methode.md)
3. [Huidige Oplossingen (State of the Art)](sections/03_state_of_the_art.md)
4. [Eisen en Randvoorwaarden](sections/04_eisen_en_randvoorwaarden.md)
5. [Bevindingen en Ontwerp/Ontwikkelkeuzes](sections/05_bevindingen_ontwerp.md)
6. [Resultaten](sections/06_resultaten.md)
7. [Discussie](sections/07_discussie.md)
8. [Conclusie](sections/08_conclusie.md)
9. [Aanbevelingen](sections/09_aanbevelingen.md)
10. [Literatuurlijst](sections/10_literatuurlijst.md)
11. [Bijlagen](sections/11_bijlagen.md)

---

## Samenvatting

Dit rapport beschrijft de ontwikkeling van HumanPose3D, een open-source pipeline voor markerless 3D bewegingsanalyse. Het systeem transformeert smartphone-video naar ISB-compliant gewrichtshoeken, zonder dure motion capture apparatuur of fysieke markers.

### Gerealiseerd Product

Een gebruiksvriendelijke desktop-applicatie die:
- Monoculaire video verwerkt naar 3D pose-reconstructie
- 59/64 biomechanische markers genereert via Pose2Sim augmentatie (Pagnon et al., 2021)
- 12 ISB-compliant gewrichtsgroepen berekent (Wu et al., 2002, 2005)
- Volledig lokaal draait op een standaard laptop (~60 seconden per video)

### Experimentele Uitbreiding

Een POF-module (Part Orientation Fields, gebaseerd op Xiang et al., 2019) die:
- ~7° limb-oriëntatiefout bereikt op validatiedata (vs. ~16° voor MediaPipe baseline)
- Nog niet volledig geïntegreerd is in de productie-pipeline (zie sectie 5.5)

### Conclusie

De baseline pipeline is functioneel en voldoet aan de projectdoelstellingen. De POF-uitbreiding toont veelbelovende resultaten maar vereist verdere integratie voordat de nauwkeurigheidswinst beschikbaar is voor eindgebruikers. Het systeem is geschikt voor screening en monitoring; voor kritische beslissingen in de zorg blijft aanvullende validatie noodzakelijk.

---

*Voor technische details zie de individuele secties. De wiskundige formulering van de POF-aanpak staat in sectie 5.4, de onopgeloste uitdagingen in sectie 5.5.*
