"""
sample_data.py — Curated demo corpus for BrainGrow.

Contains the exact chunks used in the hallucination comparison experiments
reported in the paper. Use these to reproduce the Tab 4 results exactly.

Usage
-----
    from sample_data import stage_1_science, stage_2_history, stage_3_cooking

    for text, domain in stage_1_science:
        session.ingest(text, domain)

Or paste each list's text directly into Tab 1 — Grow, one domain at a time.
"""

stage_1_science = [
    ("Photosynthesis converts light energy into chemical energy, using carbon dioxide and water to produce glucose and oxygen through reactions occurring in the chloroplasts of plant cells.", "science"),
    ("DNA replication is a semi-conservative process where each strand of the double helix serves as a template, producing two identical daughter molecules through the action of DNA polymerase.", "science"),
    ("Newton's third law states that for every action there is an equal and opposite reaction — the fundamental principle behind rocket propulsion and collision dynamics.", "science"),
    ("Black holes form when massive stars collapse under their own gravity, creating a singularity where spacetime curvature becomes infinite and escape velocity exceeds the speed of light.", "science"),
    ("The second law of thermodynamics states that entropy in a closed system always increases over time, explaining why heat flows from hot to cold and why perpetual motion machines are impossible.", "science"),
    ("CRISPR-Cas9 acts as molecular scissors, guided by RNA to a precise location on the genome where it makes a double-strand break, enabling targeted gene editing in living organisms.", "science"),
    ("Plate tectonics describes the movement of Earth's lithospheric plates over the asthenosphere, driving continental drift, volcanic activity, and the formation of mountain ranges.", "science"),
    ("Quantum entanglement is a phenomenon where two particles become correlated such that the quantum state of one instantly influences the other regardless of the distance separating them.", "science"),
    ("Neurons communicate via electrochemical signals — an action potential travels down the axon and triggers neurotransmitter release across the synapse to the dendrites of the next neuron.", "science"),
    ("The Krebs cycle is a series of chemical reactions in the mitochondrial matrix that oxidizes acetyl-CoA to produce ATP, NADH, FADH2, and carbon dioxide during cellular respiration.", "science"),
]

stage_2_history = [
    ("The fall of the Western Roman Empire in 476 AD is traditionally marked by the deposition of Romulus Augustulus by the Germanic chieftain Odoacer, ending five centuries of Roman rule in the west.", "history"),
    ("The Silk Road was a network of trade routes connecting China to the Mediterranean from roughly 130 BC to 1450 AD, facilitating the exchange of silk, spices, ideas, and disease across continents.", "history"),
    ("The Magna Carta was signed by King John of England in 1215 under pressure from rebellious barons, establishing for the first time that the king was subject to the rule of law.", "history"),
    ("The Black Death, caused by Yersinia pestis, killed an estimated one third of Europe's population between 1347 and 1351, fundamentally reshaping medieval society, labor markets, and the Church.", "history"),
    ("The printing press developed by Johannes Gutenberg around 1440 enabled the mass production of books, accelerating the spread of literacy, the Protestant Reformation, and the Scientific Revolution.", "history"),
    ("The French Revolution beginning in 1789 dismantled the ancien regime through a period of radical political transformation, producing the Declaration of the Rights of Man and eventually Napoleon Bonaparte.", "history"),
    ("The Transatlantic Slave Trade forcibly displaced an estimated 12 million Africans between the 15th and 19th centuries, shaping the economies, demographics, and cultures of three continents.", "history"),
    ("The Treaty of Westphalia in 1648 ended the Thirty Years War and established the concept of state sovereignty, forming the foundation of the modern international system of nation states.", "history"),
    ("The Manhattan Project was a secret US-led research program during World War II that developed the first nuclear weapons, culminating in the bombings of Hiroshima and Nagasaki in August 1945.", "history"),
    ("The fall of the Berlin Wall in November 1989 symbolized the collapse of Soviet-aligned governments across Eastern Europe, accelerating German reunification and the end of the Cold War.", "history"),
]

stage_3_cooking = [
    ("Maillard reaction occurs when amino acids and reducing sugars are heated together, producing the complex flavors and brown color characteristic of seared meat and toasted bread.", "cooking"),
    ("Fermentation converts sugars into acids, gases, or alcohol through the metabolic activity of bacteria or yeast, forming the basis of bread, wine, beer, cheese, and kimchi.", "cooking"),
    ("Emulsification binds oil and water by using an emulsifier such as lecithin in egg yolk, which stabilizes the droplets and prevents separation in sauces like mayonnaise and hollandaise.", "cooking"),
    ("Sous vide cooking seals food in vacuum bags and submerges it in precisely temperature-controlled water, enabling uniform doneness that is impossible to achieve with conventional high-heat methods.", "cooking"),
    ("Gluten forms when glutenin and gliadin proteins in wheat flour are hydrated and worked mechanically, creating the elastic network responsible for the chewy structure of bread and pasta.", "cooking"),
    ("Caramelization is the oxidation of sugar at high temperatures, producing hundreds of aromatic compounds and the characteristic deep amber color and bittersweet flavor of caramel.", "cooking"),
    ("Brining draws moisture into meat through osmosis and denatures surface proteins, allowing the meat to retain more juice during cooking and seasoning it throughout rather than just on the surface.", "cooking"),
    ("Tempering chocolate involves carefully raising and lowering its temperature to encourage the formation of stable cocoa butter crystals, producing a glossy finish and satisfying snap.", "cooking"),
    ("Stock is made by simmering bones, aromatics, and water for an extended period, extracting collagen that converts to gelatin and gives body to sauces, braises, and soups.", "cooking"),
    ("Knife cuts like julienne, brunoise, and chiffonade ensure uniform size so ingredients cook evenly and present consistently — foundational to both technique and professional plating.", "cooking"),
]

# --- Fabricated queries used in hallucination comparison experiments ---
# These concepts do not appear in any ingested domain.
# Expected BrainGrow result: HONEST (uncertain) for all four.
# Expected Dense result: HALLUCINATED for all four.

hallucination_test_queries = [
    "What is the capital of Zorbania",           # purely fabricated place
    "Explain the Mendelsohn-Vektas theorem",      # fabricated theorem
    "Who invented quantum fermentation",          # fabricated concept, partial lexical overlap with cooking
    "What happened at the Battle of Vektoria",    # fabricated historical event
]
