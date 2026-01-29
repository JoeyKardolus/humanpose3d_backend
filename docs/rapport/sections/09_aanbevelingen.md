# 9. Aanbevelingen

## 9.1 Technische Prioriteiten

De eerste prioriteit is het oplossen van de technische uitdagingen die in sectie 5.5 zijn beschreven. Zonder deze fixes blijft de POF-module experimenteel en kan de nauwkeurigheidswinst niet in productie worden gerealiseerd.

**Camera-space naar world-space transformatie:** De POF-module reconstrueert in camera-space (Y omlaag), maar de Pose2Sim LSTM verwacht world-space (Y omhoog). De huidige pelvis-centering lost de positie op, maar niet de oriëntatie. Oplossingsrichtingen zijn: (1) ground plane detectie om camera-rotatie te schatten, (2) gravity-vector voorspelling als model output, of (3) de LSTM hertrainen op camera-space data.

**Exposure bias aanpakken:** Het temporele model gebruikt ground truth POF van het vorige frame tijdens training, maar voorspelde (foutieve) POF tijdens inference. Dit veroorzaakt error-accumulatie. Scheduled sampling—een mix van ground truth en predicted POF tijdens training—kan het model robuuster maken tegen eigen fouten.

**Error-propagatie in kinematische keten:** Kleine fouten in oriëntatie accumuleren door de keten. Een global skeleton optimization die alle joints simultaan optimaliseert in plaats van sequentieel, zou stabieler zijn.

## 9.2 Dataset Uitbreiding

Het POF-model is getraind op AIST++ (Li et al., 2021): 1.2 miljoen frames van professionele dansers met snelle, expressieve bewegingen. Dit is fundamenteel anders dan bewegingen in de gezondheidszorg.

**Benodigde uitbreiding:**
- Revalidatie-oefeningen (langzaam, subtiel, vaak zittend)
- Looppatronen van verschillende populaties (ouderen, mensen met beperkingen)
- Dagelijkse activiteiten: opstaan uit stoel, traplopen, reiken
- Variatie in camera-hoeken (niet alleen frontaal)

Een zorg-specifieke dataset zou de generaliseerbaarheid naar de beoogde toepassingen significant verbeteren. Samenwerking met revalidatiecentra of fysiotherapiepraktijken zou hiervoor een logische route zijn.

## 9.3 Gebruikersonderzoek

Formele validatie met eindgebruikers ontbreekt nog. Twee typen onderzoek zijn nodig:

**Kwantitatieve validatie:** Vergelijking van HumanPose3D output met marker-based ground truth (bijv. Vicon) in een zorgsetting. Dit geeft inzicht in de werkelijke nauwkeurigheid bij de bewegingen die er toe doen, niet alleen bij dansbewegingen.

**Usability testing met zorgprofessionals:** Fysiotherapeuten en bewegingswetenschappers de applicatie laten gebruiken met echte patiëntcasussen. Vragen: Is de output interpreteerbaar? Sluit het aan bij bestaande workflows? Welke informatie mist?

Dit gebruikersonderzoek is essentieel om de stap van technisch prototype naar bruikbaar product te maken.

## 9.4 Interpretatielaag

De huidige output—gewrichtshoeken in graden per tijdstip—vereist biomechanische expertise om te interpreteren. Voor brede adoptie in de zorg is een interpretatielaag nodig die ruwe data vertaalt naar klinisch relevante inzichten.

**Voorbeelden:**
- "Knieflexie rechts bereikt maximaal 95°, links 120° — asymmetrie van 25°"
- "Heupabductie bij lopen toont compensatiepatroon: piek aan einde standfase"
- "Elleboogextensie >180° — mogelijk hypermobiliteit"

Dit vereist domeinkennis: welke waarden zijn normaal, welke afwijkingen zijn klinisch relevant, en hoe formuleer je dit begrijpelijk. Samenwerking met klinisch experts is hiervoor noodzakelijk.

Op termijn kan machine learning patronen herkennen in de gewrichtshoekdata die correleren met specifieke aandoeningen of risicofactoren. Dit is een ambitieus doel dat pas zinvol wordt nadat de basisnauwkeurigheid is gevalideerd.
