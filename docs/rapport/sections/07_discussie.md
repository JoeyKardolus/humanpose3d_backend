# 7. Discussie

## 7.1 Sterke Punten

Het belangrijkste resultaat van dit project is dat bewegingsanalyse nu toegankelijk wordt voor een breder publiek. Waar traditionele motion capture systemen zoals Vicon investeringen van €50.000 of meer vereisen plus gespecialiseerde ruimtes, draait HumanPose3D op een standaard laptop met als enige input een smartphone-video. Dit verlaagt de drempel voor gebruik in fysiotherapiepraktijken, revalidatiecentra en thuissituaties aanzienlijk.

De technische aanpak biedt concrete voordelen. Door gebruik te maken van Part Orientation Fields (Xiang et al., 2019) wordt het fundamentele probleem van monoculaire diepte-schatting op een nieuwe manier aangepakt: in plaats van één globale camera-hoek te voorspellen, krijgt elk ledemaat zijn eigen oriëntatievector. Dit lost situaties op waar bijvoorbeeld een arm naar de camera wijst terwijl een been opzij staat—iets waar traditionele methoden zoals ElePose (Wandt et al., 2022) moeite mee hebben.

De keuze voor een volledig open-source stack betekent dat gebruikers niet afhankelijk zijn van één leverancier. MediaPipe (Lugaresi et al., 2019), Pose2Sim (Pagnon et al., 2021, 2022) en PyTorch zijn vrij beschikbaar en worden actief onderhouden door grote communities. Dit geeft vertrouwen in de continuïteit van de oplossing.

Praktisch gezien werkt het systeem zonder GPU, hoewel verwerking dan langer duurt. De output is direct bruikbaar in bestaande biomechanica-software doordat ISB-standaarden (Wu et al., 2002, 2005) worden gevolgd voor gewrichtshoekdefinities.

## 7.2 Beperkingen en Huidige Status

### Productstatus

Om een eerlijk beeld te geven van wat wel en niet werkt, is het belangrijk onderscheid te maken tussen productierijpe en experimentele componenten.

De baseline pipeline—MediaPipe (Lugaresi et al., 2019) voor pose-detectie, Pose2Sim (Pagnon et al., 2021) voor marker-augmentatie, en ISB-compliant gewrichtshoekberekening (Wu et al., 2002, 2005)—is stabiel en functioneel. Gebruikers kunnen video's uploaden via de desktop-applicatie en krijgen betrouwbare output. Dit deel van het systeem voldoet aan de initiële projectdoelstellingen.

De POF-uitbreiding (Xiang et al., 2019), bedoeld om de diepte-nauwkeurigheid te verbeteren van ~16° naar ~7° error, bevindt zich nog in experimentele fase. Het model zelf presteert goed op validatiedata, maar de integratie met de rest van de pipeline kent nog onopgeloste problemen. Sectie 5.5 beschrijft deze technische uitdagingen in detail: error-propagatie door de kinematische keten, exposure bias bij temporele context, en de mismatch tussen camera-space en world-space coördinaten.

Het Joint Constraint model vormt een belangrijk onderdeel van de geplande volgende versie. In plaats van harde anatomische limieten—die abnormale bewegingspatronen zouden maskeren—gebruikt dit model zachte, geleerde correcties. Dit is essentieel voor zorgtoepassingen waar juist de afwijkende bewegingen klinisch relevant zijn: hypermobiliteit, compensatiestrategieën bij pijn, of pathologische bewegingspatronen. Het model is geïmplementeerd en getraind, maar kan pas gevalideerd worden wanneer de POF-integratie is afgerond. De theoretische onderbouwing is sterk, maar praktische resultaten moeten nog volgen.

### Inherente Beperkingen

Sommige beperkingen zijn inherent aan de gekozen aanpak. Single-view occlusie blijft een fundamenteel probleem: wanneer een ledemaat achter het lichaam verdwijnt, is er simpelweg geen informatie beschikbaar om de positie te reconstrueren. Multi-view ondersteuning zou dit kunnen oplossen, maar valt buiten de huidige scope.

De verwerkingstijd van ongeveer 60 seconden per video maakt real-time feedback onmogelijk. Voor toepassingen waar directe terugkoppeling essentieel is—zoals biofeedback tijdens oefeningen—is dit een significante beperking.

### Validatiebeperkingen

De validatie kent methodologische beperkingen die eerlijk benoemd moeten worden. Het POF-model is uitsluitend getraind op de AIST++ dataset (Li et al., 2021), die bestaat uit professionele dansers die expressieve, snelle bewegingen uitvoeren. Dit is fundamenteel anders dan de typische bewegingen in zorgsettings: langzame revalidatie-oefeningen, subtiele looppatronen, of bewegingen van ouderen met beperkte mobiliteit.

Formele kwantitatieve validatie tegen marker-based ground truth in een zorgsetting is niet uitgevoerd. De gerapporteerde nauwkeurigheid (~7° error) is gemeten op dezelfde distributie waarop het model is getraind. Hoe het systeem presteert op bewegingen buiten deze distributie is onbekend.

## 7.3 Implicaties voor Gebruik

Gegeven de huidige status is het systeem geschikt voor bepaalde toepassingen, maar niet voor andere. Voor screening en eerste beoordeling van bewegingspatronen biedt het waardevolle informatie tegen lage kosten. Voor het volgen van voortgang over tijd—bijvoorbeeld tijdens een revalidatietraject—kan het trends zichtbaar maken, mits de gebruiker rekening houdt met de inherente meetonzekerheid.

Het systeem is niet geschikt voor situaties waar hoge nauwkeurigheid vereist is. Kritische beslissingen in de zorg moeten niet gebaseerd worden op deze output zonder aanvullende validatie. Sub-millimeter precisie, zoals nodig voor chirurgische planning, is principieel onhaalbaar met deze technologie.

Gebruikers moeten zich bewust zijn van de beperkingen bij interpretatie van de output: een inherente fout van ~16° (baseline) tot ~7° (POF) in limb-oriëntaties, mogelijke frame-dropouts bij snelle bewegingen, en artefacten bij occlusie.
