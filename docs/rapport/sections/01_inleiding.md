# 1. Inleiding

## 1.1 Aanleiding

Bewegingsanalyse speelt een cruciale rol in de gezondheidszorg, van revalidatie en fysiotherapie tot sportprestatie-optimalisatie. De huidige gouden standaard—marker-based motion capture systemen zoals Vicon en OptiTrack—biedt submillimeter nauwkeurigheid, maar tegen hoge kosten (€50.000+) en met praktische beperkingen: markers moeten fysiek op het lichaam worden geplaatst, en opnames zijn beperkt tot gespecialiseerde laboratoria.

Consumentgerichte markerless oplossingen zoals MediaPipe (Lugaresi et al., 2019) bieden een laagdrempelig alternatief. Deze systemen kunnen pose-landmarks detecteren vanuit reguliere video, maar leveren onbetrouwbare 3D diepte-informatie met een gemiddelde fout van ongeveer 16°. Dit beperkt hun bruikbaarheid in de gezondheidszorg voor toepassingen waar accurate gewrichtshoeken essentieel zijn.

## 1.2 Klantwensen

In afstemming met de opdrachtgever zijn de volgende kernwensen geïdentificeerd:

- **Markerless:** Geen fysieke markers nodig op het lichaam van de patiënt
- **Betaalbaar:** Moet werken met standaard hardware (smartphone en laptop)
- **Toegankelijk:** Gebruiksvriendelijke desktop-applicatie, geschikt voor niet-technische gebruikers
- **Gestandaardiseerd:** Gewrichtshoeken volgens ISB-standaarden (Wu et al., 2002, 2005)
- **Privacy-bewust:** Volledig lokale verwerking, geen cloud-upload vereist

## 1.3 Probleemstelling

MediaPipe Pose levert accurate 2D landmark-detectie, maar de ingebouwde 3D diepte-schatting vertoont systematische fouten. Dit komt vooral voor bij ledematen die naar of van de camera bewegen—het zogenaamde "foreshortening" probleem. Een arm die naar de camera wijst lijkt in 2D kort, maar of deze naar voren of naar achteren wijst is zonder aanvullende informatie niet te bepalen.

**Centrale onderzoeksvraag:**

> Hoe kunnen we betrouwbare 3D pose-reconstructie bereiken vanuit monoculaire smartphone-video, met nauwkeurigheid die geschikt is voor bewegingsanalyse in de gezondheidszorg?

## 1.4 Opdrachtomschrijving

Het doel van dit project is de ontwikkeling van een toegankelijke desktop-applicatie voor markerless bewegingsanalyse. De applicatie moet video-input kunnen verwerken tot biomechanisch bruikbare gewrichtshoeken, zonder dat de gebruiker technische kennis nodig heeft.

De technische aanpak combineert bestaande open-source componenten met eigen ontwikkeling:

- **Pose-detectie:** MediaPipe (Lugaresi et al., 2019) of RTMPose voor 2D/3D landmark-extractie
- **Marker-augmentatie:** Pose2Sim (Pagnon et al., 2021, 2022) voor uitbreiding naar 64 biomechanische markers
- **Gewrichtshoeken:** ISB-compliant berekening volgens Wu et al. (2002, 2005)
- **Diepte-verbetering:** Experimentele POF-module gebaseerd op Xiang et al. (2019) voor toekomstige nauwkeurigheidsverbetering
- **Gewrichtshoek-correctie:** Neuraal Joint Constraint model dat zachte correcties leert in plaats van harde anatomische limieten, essentieel voor detectie van abnormale bewegingspatronen in de gezondheidszorg

De baseline pipeline—zonder POF en Joint Constraint—is functioneel en vormt het kernproduct. De experimentele modules zijn ontwikkeld als basis voor een volgende versie met verbeterde diepte-reconstructie en gewrichtshoek-correctie, maar vereisen nog verdere integratie voordat deze productierijp zijn.
