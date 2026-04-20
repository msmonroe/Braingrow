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
├── session.py                    # BrainGrowSession — all business logic
├── vector_space.py               # Pre-allocation, activation lifecycle, pruning (v2)
├── growth_engine.py              # Staged ingestion, batch encoding, slot assignment
├── query_router.py               # Routes queries through active slots only (v2)
├── epistemic.py                  # Three-tier epistemic classifier (new in v2)
├── comparison_harness.py         # DenseModel vs BrainGrow comparison (v2, bug fixed)
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
| Prune threshold 0.20 | Removes genuinely dormant slots while preserving concepts that have been queried at least twice. |
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

## Demo Script

| Step | Action | Expected Observation |
|---|---|---|
| 1 | **Initialize** | Launch app. UMAP shows 200,000 grey dormant slots. |
| 2 | **Stage 1 — Science** | Ingest science chunks. UMAP lights up in a sparse cluster. Histogram shows a small active fraction. |
| 3 | **Stage 2 — History** | Ingest history chunks. A new cluster appears in a different region. Science cluster unchanged. |
| 4 | **Query — Science** | Ask a science question. Routing highlights the science cluster only. |
| 5 | **Query — History** | Ask a history question. Routing highlights the history cluster. No cross-contamination. |
| 6 | **Prune** | Run pruning at threshold 0.2. Low-activation slots grey out. Core concepts survive. |
| 7 | **Expand** | Ingest Stage 3 (e.g. cooking). Grows into space freed by pruning. |
| 8 | **Compare** | Switch to Tab 4. Run Unknown queries. BrainGrow abstains; DenseModel hallucinates. |
| 9 | **Save** | Switch to Tab 5. Save network state before lengthy experiments. |
| 10 | **TinyStories** | Switch to Tab 6. Run Stage A (~1k chunks), Stage B (10k), Stage C (full scale). Enable Autosave first. |

---

## Demonstrated Behaviors

The following behaviors have been observed and are reproducible:

- **Routing isolation** — queries activate domain-relevant slot clusters with limited cross-domain contamination
- **Honest abstention** — BrainGrow correctly returns Honest Unknown on all four fabricated-concept queries tested; the Dense baseline hallucinates on all four
- **Non-destructive expansion** — ingesting a new domain does not shift or corrupt previously activated slot regions
- **Pruning recovery** — after a pruning pass, a new domain successfully claims reclaimed dormant slots
- **Visual legibility** — UMAP projections show the query embedding landing in dormant space (BrainGrow) vs. forced into known clusters (Dense)

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

- The frozen sentence encoder (`all-MiniLM-L6-v2`) is responsible for semantic topology. Slot placement mechanism does not add measurable geometric benefit over sequential assignment at current dataset sizes.
- The confidence threshold (0.60) and lifecycle parameters are fixed hyperparameters, not learned from data.
- The DenseModel comparison baseline is a toy saturated store, not an actual neural language model.
- No learned weight updates occur at any stage. BrainGrow is a retrieval architecture, not a generative one.

---

## Future Directions

- **Embodied feedback loop** — replace static ingestion with agent-environment interaction; slots activate based on reward signal
- **Hierarchical pruning** — staged fine-to-coarse pruning mirroring cortical development
- **Cross-domain generalization** — measure whether concepts in overlapping slot regions produce emergent analogical reasoning
- **Rigorous baseline comparison** — train an equivalently-sized static model on the same corpus; compare retrieval accuracy and epistemic calibration
- **Threshold learning** — derive confidence threshold from the empirical similarity distribution of the ingested corpus rather than setting it manually

---

## Publication

A technical report describing this architecture and the experimental results above is in preparation for arXiv submission (cs.NE / cs.LG).

---

## License

MIT License. Open for research use and inspection.  
Vektas Solutions · vektassolutions.com
