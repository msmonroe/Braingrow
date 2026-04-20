# BrainGrow

**Developmental AI Architecture — Research Prototype**  
Vektas Solutions · April 2026 · Author: Matthew Monroe

---

## Overview

Large language models conflate knowledge and ignorance within the same weight matrices, producing confident outputs regardless of whether a query falls within their training distribution. BrainGrow proposes a different approach: make epistemic boundaries structural rather than post-hoc.

BrainGrow pre-allocates a sparse vector space of 200,000 slots initialized as dormant unit vectors. Knowledge is introduced through staged ingestion — text is encoded by a frozen sentence encoder and assigned to dormant slots, which transition to active. Active slots are subject to use-dependent reinforcement and decay. Pruning passes reclaim low-activation slots for future domain expansion. Queries are routed exclusively through active slots, producing one of three explicit epistemic states:

- **Confident** — high-similarity match (≥ 0.60 cosine) within an active domain
- **Honest Unknown** — no active slot exceeds the confidence threshold; the system abstains
- **Out-of-Bounds** — nearest match falls within a registered negative domain

These states are enforced structurally by the architecture. No separate classifier is trained. No post-hoc alignment is applied.

> **Core finding:** Hallucination is an architectural property of a saturated vector space, not a scale or data-quantity problem. A fully-occupied store has nowhere to abstain to. BrainGrow's dormant capacity is the abstention mechanism.

---

## Experimental Results

### Slot Assignment Geometry

Sequential and semantic-aware slot assignment produce identical separability metrics over the domains tested (silhouette score 0.1256, inter-centroid distance 0.6032, separability ratio 0.6739 in both conditions). The frozen sentence encoder is responsible for the semantic topology of the active space. BrainGrow's contribution is the activation lifecycle and routing machinery built on top of that topology, not learned geometric placement.

See `experiments/slot_assignment_comparison.py` to reproduce.

### Hallucination Comparison (Tab 4)

Four entirely fabricated queries were run against both the Dense baseline and BrainGrow after ingesting science, history, and cooking domains:

| Query | Dense Verdict | Dense Sim | BrainGrow Verdict | BrainGrow Sim |
|---|---|---|---|---|
| What is the capital of Zorbania | HALLUCINATED | 0.1468 | HONEST (uncertain) | 0.1468 |
| Explain the Mendelsohn-Vektas theorem | HALLUCINATED | 0.1893 | HONEST (uncertain) | 0.1893 |
| Who invented quantum fermentation | HALLUCINATED | 0.4035 | HONEST (uncertain) | 0.4035 |
| What happened at the Battle of Vektoria | HALLUCINATED | 0.2181 | HONEST (uncertain) | 0.2181 |

The Dense baseline returns a confident nearest-neighbor match in 4/4 cases. BrainGrow correctly abstains in 4/4 cases. Note the third query: "quantum fermentation" achieves a cosine similarity of 0.4035 against a real fermentation chunk due to lexical overlap. The 0.60 confidence threshold correctly classifies this as Honest Unknown, preventing a false-positive confident response on a fabricated concept.

---

## Architecture

```
[ Pre-allocate 200,000 vector slots — random unit vectors, all dormant ]
          ↓
[ Stage 1: Encode Domain A → assign to dormant slots → activate ]
          ↓
[ Stage 2: Encode Domain B → grows into NEW unused slots ]
          ↓
[ Query: cosine similarity against active slots only ]
          ↓
[ Epistemic classification: Confident / Honest Unknown / Out-of-Bounds ]
          ↓
[ Lifecycle: reinforce on hit, decay over time, prune below threshold ]
          ↓
[ Expansion: pruned slots return to dormant pool for future domains ]
```

### The Six Tabs

| Tab | What It Demonstrates |
|---|---|
| **Grow** | Pre-allocated vector space. Knowledge grows into it progressively. Active vs. dormant slots visualized in real time via UMAP / PCA. Includes Stage Diff and Refresh UMAP controls. |
| **Query** | Routing through active slots only. New domains grow into previously dormant space without overwriting existing knowledge. |
| **Prune** | Use-dependent pruning pass. Dormant slots decay. Active slots strengthen. Before/after comparison visualized. |
| **Compare** | Hallucination demo. Runs identical queries against a saturated DenseModel and BrainGrow side-by-side. |
| **Network** | Save / load complete network state as `.bgstate` files. Autosave after every Ingest Stage. |
| **TinyStories** | Scale test against the `roneneldan/TinyStories` corpus — up to 100,000 story snippets across 200,000 slots in three progressive stages. |

---

## Project Structure

```
braingrow/
├── main.py                       # Gradio app entry point (6-tab UI)
├── session.py                    # BrainGrowSession — all business logic (v2)
├── vector_space.py               # Pre-allocation, activation lifecycle, pruning (v2)
├── growth_engine.py              # Staged ingestion, batch encoding, slot assignment
├── query_router.py               # Routes queries through active slots only (v2)
├── epistemic.py                  # Three-tier epistemic classifier (new in v2)
├── comparison_harness.py         # DenseModel vs BrainGrow comparison (v2, threshold fixed)
├── sample_data.py                # Curated demo corpus — reproduces paper results exactly
├── tinystories_loader.py         # TinyStories data pipeline (Tab 6)
├── visualizer.py                 # UMAP projection & Plotly charts
├── instrumentation.py            # Optional timing / error tracing (BRAINGROW_TRACE=1)
├── utils.py                      # Shared unit-normalised encoding utilities
├── requirements.txt              # Core Python dependencies
├── saves/                        # .bgstate network snapshots (autosave target)
├── tests/                        # Pytest suite (one test file per module)
└── experiments/
    └── slot_assignment_comparison.py   # v1 vs v2 slot assignment geometry experiment
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| 200,000 pre-allocated slots | Sufficient headroom for TinyStories full-scale run without live reallocation. |
| `all-MiniLM-L6-v2` (384-dim) | Compact, fast, well-calibrated for semantic similarity at CPU speeds. |
| Confidence threshold 0.60 | Empirically chosen for this encoder: above 0.60 indicates strong overlap; the 0.40–0.60 band is conservatively treated as uncertain to prevent partial-match false positives. |
| Reinforce step 0.10, decay 0.005 | Produces stable activation dynamics across TinyStories ingestion without rapid slot exhaustion or over-pruning. |
| Prune threshold 0.20 | Removes genuinely dormant slots while preserving concepts queried at least twice. |
| Thread-safe `RLock` | Gradio's concurrent callbacks can write without race conditions. |
| `BrainGrowSession` class | All state and logic isolated from Gradio; independently testable. |
| `.bgstate` persistence | Full snapshot (embeddings + activations + metadata) prevents data loss on long runs. |
| `epistemic.py` as separate module | Epistemic classification is independently testable and consistently applied across all code paths. |

---

## Requirements

- Python 3.11+
- PyTorch 2.x (CPU or CUDA)
- sentence-transformers
- Gradio 4+
- Plotly
- UMAP-learn
- NumPy
- scikit-learn
- datasets *(optional — required for Tab 6 TinyStories only)*

---

## Setup

```bash
# Clone the repo, then:
cd braingrow

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# PyTorch (CPU build — sufficient for all experiments):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Optional: TinyStories scale experiment (Tab 6)
pip install datasets

python main.py
```

Then open the URL printed in the console (default: `http://localhost:7860`).

### Tracing / Instrumentation

```bash
BRAINGROW_TRACE=1 python main.py

# Redirect trace output to a file:
BRAINGROW_TRACE=1 BRAINGROW_LOG=braingrow.log python main.py
```

---

## Reproducing the Paper Results

The sample corpus used in all hallucination comparison experiments is in `sample_data.py`. To reproduce the Tab 4 results exactly:

**Step 1 — Ingest Stage 1 (Science)**

Go to Tab 1 — Grow. Paste the following into the text box, set domain label to `science`, and click Ingest Stage:

```
Photosynthesis converts light energy into chemical energy, using carbon dioxide and water to produce glucose and oxygen through reactions occurring in the chloroplasts of plant cells.
DNA replication is a semi-conservative process where each strand of the double helix serves as a template, producing two identical daughter molecules through the action of DNA polymerase.
Newton's third law states that for every action there is an equal and opposite reaction — the fundamental principle behind rocket propulsion and collision dynamics.
Black holes form when massive stars collapse under their own gravity, creating a singularity where spacetime curvature becomes infinite and escape velocity exceeds the speed of light.
The second law of thermodynamics states that entropy in a closed system always increases over time, explaining why heat flows from hot to cold and why perpetual motion machines are impossible.
CRISPR-Cas9 acts as molecular scissors, guided by RNA to a precise location on the genome where it makes a double-strand break, enabling targeted gene editing in living organisms.
Plate tectonics describes the movement of Earth's lithospheric plates over the asthenosphere, driving continental drift, volcanic activity, and the formation of mountain ranges.
Quantum entanglement is a phenomenon where two particles become correlated such that the quantum state of one instantly influences the other regardless of the distance separating them.
Neurons communicate via electrochemical signals — an action potential travels down the axon and triggers neurotransmitter release across the synapse to the dendrites of the next neuron.
The Krebs cycle is a series of chemical reactions in the mitochondrial matrix that oxidizes acetyl-CoA to produce ATP, NADH, FADH2, and carbon dioxide during cellular respiration.
```

**Step 2 — Ingest Stage 2 (History)**

Set domain label to `history`, paste the following, and click Ingest Stage:

```
The fall of the Western Roman Empire in 476 AD is traditionally marked by the deposition of Romulus Augustulus by the Germanic chieftain Odoacer, ending five centuries of Roman rule in the west.
The Silk Road was a network of trade routes connecting China to the Mediterranean from roughly 130 BC to 1450 AD, facilitating the exchange of silk, spices, ideas, and disease across continents.
The Magna Carta was signed by King John of England in 1215 under pressure from rebellious barons, establishing for the first time that the king was subject to the rule of law.
The Black Death, caused by Yersinia pestis, killed an estimated one third of Europe's population between 1347 and 1351, fundamentally reshaping medieval society, labor markets, and the Church.
The printing press developed by Johannes Gutenberg around 1440 enabled the mass production of books, accelerating the spread of literacy, the Protestant Reformation, and the Scientific Revolution.
The French Revolution beginning in 1789 dismantled the ancien regime through a period of radical political transformation, producing the Declaration of the Rights of Man and eventually Napoleon Bonaparte.
The Transatlantic Slave Trade forcibly displaced an estimated 12 million Africans between the 15th and 19th centuries, shaping the economies, demographics, and cultures of three continents.
The Treaty of Westphalia in 1648 ended the Thirty Years War and established the concept of state sovereignty, forming the foundation of the modern international system of nation states.
The Manhattan Project was a secret US-led research program during World War II that developed the first nuclear weapons, culminating in the bombings of Hiroshima and Nagasaki in August 1945.
The fall of the Berlin Wall in November 1989 symbolized the collapse of Soviet-aligned governments across Eastern Europe, accelerating German reunification and the end of the Cold War.
```

**Step 3 — Ingest Stage 3 (Cooking)**

Set domain label to `cooking`, paste the following, and click Ingest Stage:

```
Maillard reaction occurs when amino acids and reducing sugars are heated together, producing the complex flavors and brown color characteristic of seared meat and toasted bread.
Fermentation converts sugars into acids, gases, or alcohol through the metabolic activity of bacteria or yeast, forming the basis of bread, wine, beer, cheese, and kimchi.
Emulsification binds oil and water by using an emulsifier such as lecithin in egg yolk, which stabilizes the droplets and prevents separation in sauces like mayonnaise and hollandaise.
Sous vide cooking seals food in vacuum bags and submerges it in precisely temperature-controlled water, enabling uniform doneness that is impossible to achieve with conventional high-heat methods.
Gluten forms when glutenin and gliadin proteins in wheat flour are hydrated and worked mechanically, creating the elastic network responsible for the chewy structure of bread and pasta.
Caramelization is the oxidation of sugar at high temperatures, producing hundreds of aromatic compounds and the characteristic deep amber color and bittersweet flavor of caramel.
Brining draws moisture into meat through osmosis and denatures surface proteins, allowing the meat to retain more juice during cooking and seasoning it throughout rather than just on the surface.
Tempering chocolate involves carefully raising and lowering its temperature to encourage the formation of stable cocoa butter crystals, producing a glossy finish and satisfying snap.
Stock is made by simmering bones, aromatics, and water for an extended period, extracting collagen that converts to gelatin and gives body to sauces, braises, and soups.
Knife cuts like julienne, brunoise, and chiffonade ensure uniform size so ingredients cook evenly and present consistently — foundational to both technique and professional plating.
```

**Step 4 — Run the Hallucination Comparison**

Switch to Tab 4 — Compare. Set Query Type to `Unknown` and run each of the following queries:

- What is the capital of Zorbania
- Explain the Mendelsohn-Vektas theorem
- Who invented quantum fermentation
- What happened at the Battle of Vektoria

Expected: Dense returns HALLUCINATED on all four. BrainGrow returns HONEST (uncertain) on all four.

---

## Demo Script

| Step | Action | Expected Observation |
|---|---|---|
| 1 | **Initialize** | Launch app. UMAP shows 200,000 grey dormant slots. |
| 2 | **Stage 1 — Science** | Ingest science chunks. UMAP lights up in a sparse cluster. |
| 3 | **Stage 2 — History** | A new cluster appears in a different region. Science cluster unchanged. |
| 4 | **Query — Science** | Ask a science question. Routing highlights the science cluster only. |
| 5 | **Query — History** | Ask a history question. History cluster activates. No cross-contamination. |
| 6 | **Prune** | Run pruning at threshold 0.2. Low-activation slots grey out. Core concepts survive. |
| 7 | **Expand** | Ingest Stage 3 (cooking). Grows into space freed by pruning. |
| 8 | **Compare** | Tab 4, Unknown queries. BrainGrow abstains; DenseModel hallucinates. |
| 9 | **Save** | Tab 5. Save network state before lengthy experiments. |
| 10 | **TinyStories** | Tab 6. Stage A (~1k), Stage B (10k), Stage C (full). Enable Autosave first. |

---

## Demonstrated Behaviors

- **Routing isolation** — queries activate domain-relevant slot clusters with limited cross-domain contamination
- **Honest abstention** — BrainGrow correctly returns Honest Unknown on all four fabricated-concept queries; Dense hallucinates on all four
- **Non-destructive expansion** — ingesting a new domain does not shift or corrupt previously activated regions
- **Pruning recovery** — after a pruning pass, a new domain successfully claims reclaimed dormant slots
- **Visual legibility** — UMAP projections show the query star landing in dormant space (BrainGrow) vs. forced into known clusters (Dense)

---

## Running Experiments

```bash
# Slot assignment geometry comparison (v1 sequential vs v2 semantic-aware)
python experiments/slot_assignment_comparison.py

# Full test suite
pytest tests/
```

---

## Limitations

- The frozen sentence encoder (`all-MiniLM-L6-v2`) is responsible for semantic topology. Slot placement does not add measurable geometric benefit over sequential assignment at current dataset sizes.
- The confidence threshold (0.60) and lifecycle parameters are fixed hyperparameters, not learned from data.
- The DenseModel comparison baseline is a toy saturated store, not an actual neural language model.
- No learned weight updates occur at any stage. BrainGrow is a retrieval architecture, not a generative one.

---

## Future Directions

- **Embodied feedback loop** — replace static ingestion with agent-environment interaction; slots activate based on reward signal
- **Hierarchical pruning** — staged fine-to-coarse pruning mirroring cortical development
- **Cross-domain generalization** — measure whether concepts in overlapping slot regions produce emergent analogical reasoning
- **Rigorous baseline comparison** — train an equivalently-sized static model on the same corpus; compare retrieval accuracy and epistemic calibration
- **Threshold learning** — derive confidence threshold from the empirical similarity distribution rather than setting it manually

---

## Publication

A technical report describing this architecture and experimental results is in preparation for arXiv submission (cs.NE / cs.LG).

---

## License

MIT License. Open for research use and inspection.  
Vektas Solutions · vektassolutions.com
