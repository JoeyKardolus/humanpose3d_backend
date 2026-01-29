# 8. Conclusie

## 8.1 Beantwoording Onderzoeksvraag

De centrale vraag van dit project was of betrouwbare 3D pose-reconstructie mogelijk is vanuit monoculaire smartphone-video, met voldoende nauwkeurigheid voor bewegingsanalyse in de gezondheidszorg.

Het antwoord is genuanceerd. We hebben een functionerende pipeline gebouwd die monoculaire video transformeert naar ISB-compliant gewrichtshoeken (Wu et al., 2002, 2005). De combinatie van open-source 2D/3D detectie via MediaPipe (Lugaresi et al., 2019) en RTMPose, Pose2Sim marker-augmentatie (Pagnon et al., 2021, 2022), en gestandaardiseerde gewrichtshoekberekening levert bruikbare output op een standaard laptop.

Daarnaast is een experimentele POF-module ontwikkeld, gebaseerd op de Part Orientation Fields aanpak van Xiang et al. (2019), als basis voor een volgende versie met verbeterde diepte-reconstructie. Deze module bereikt ~7° error op validatiedata—bijna een halvering ten opzichte van de MediaPipe baseline—maar vereist nog verdere integratie.

## 8.2 Kernresultaten

Het kernresultaat van deze versie is een gebruiksvriendelijke, toegankelijke desktop-applicatie die monoculaire 3D pose-analyse uitvoert inclusief ISB-compliant gewrichtshoekberekeningen. De applicatie is lokaal installeerbaar, vereist geen technische kennis, en levert gestructureerde output die direct bruikbaar is in bestaande biomechanica-workflows.

De huidige nauwkeurigheid met de MediaPipe baseline (~16° limb-oriëntatiefout) is voldoende voor screening en monitoring, maar nog niet voor diagnostiek in de gezondheidszorg. De POF-module legt de technische basis voor een volgende versie met hogere nauwkeurigheid, maar de integratie-uitdagingen beschreven in sectie 5.5 moeten eerst worden opgelost.

## 8.3 Eindconclusie

HumanPose3D overbrugt de kloof tussen dure marker-based systemen en onnauwkeurige consumentenoplossingen. Met een smartphone en laptop kunnen gebruikers nu bewegingsanalyse uitvoeren die voorheen specialistische apparatuur vereiste. De baseline pipeline is functioneel en voldoet aan de initiële projectdoelstellingen.

Het systeem is in de huidige vorm geschikt voor screening, voortgangsmonitoring, thuisgebruik door patiënten, en als open-source basis voor verder onderzoek. Voor kritische beslissingen in de zorg blijft aanvullende validatie noodzakelijk totdat de POF-integratie voltooid is en formele validatiestudies in zorgsettings zijn uitgevoerd.

De volgende stap is het oplossen van de technische uitdagingen rond POF-integratie—met name de camera-naar-world transformatie en temporele stabiliteit—om de volledige nauwkeurigheidswinst ook in productie te realiseren.
