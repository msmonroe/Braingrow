# BrainGrow

**Developmental AI Architecture — Proof of Concept**
Vektas Solutions · April 2026 · Author: Matthew Monroe

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: Research](https://img.shields.io/badge/license-research--prototype-lightgrey.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-paper--in--preparation-b31b1b.svg)]()

---

## Overview

Current large language models are trained in a single static run — compressing all knowledge into a frozen weight matrix before the model ever interacts with the world. BrainGrow inverts this paradigm.

Inspired by human neurodevelopment — where an infant is born with *more* synaptic connections than an adult, and the brain sculpts intelligence through use-dependent pruning — BrainGrow pre-allocates a large vector space and allows knowledge to grow into it organically through staged exposure and interaction. Dormant capacity is preserved for future expansion rather than discarded.

> **Core hypothesis:** Hallucination in AI systems is not primarily a data or scale problem — it is an architectural property of fully saturated knowledge representations that have no structural mechanism for honest uncertainty. A pre-allocated space with dormant capacity produces three structurally distinct epistemic states without any trained behavior.

---

## Key Experimental Findings

These results were produced by the proof-of-concept implementation in this repository. An arXiv paper documenting the full methodology and results is in preparation.

| Finding | Description |
| --- | --- |
| **Spatial domain separation** | Domains ingested at different stages occupy geometrically distinct UMAP regions with no cross-contamination. Science and history clusters do not bleed into each other. |
| **Routing isolation** | Queries activate only domain-relevant slots. A science query does not touch the history cluster. |
| **Non-destructive expansion** | Adding a new domain does not shift or corrupt previously activated regions. |
| **Pruning and reclamation** | After a pruning pass, a new domain successfully claims reclaimed dormant slots without corrupting existing knowledge. |
| **Hallucination as architecture** | A DenseModel (no dormant space) always returns the nearest confident neighbor regardless of relevance. BrainGrow returns `HONEST UNKNOWN` when the query falls into dormant space — no training required. |
| **Three-tier epistemic output** | Confident / Honest Unknown / Boundary Violation emerge structurally from the geometry of the vector space. No current LLM has this as a first-class architectural output. |
| **Positive/negative knowledge imbalance** | Domains with high positive:negative slot ratios produce elevated false-confident responses. Counterbalancing negative examples measurably reduces this. The root of hallucination is asymmetric knowledge representation, not insufficient data. |
| **Online epistemic correction** | Boundary violation rate drops measurably after reactive negative slot ingestion — **without retraining**. This is structurally unavailable in dense weight-matrix architectures. |
| **Emergent unlabeled clustering** | TinyStories ingestion (100k unlabeled story snippets) produced sub-clusters within a single domain that were not manually defined — unsupervised concept organisation emerging from developmental growth alone. |

---

## Architecture

```
[ Pre-allocate 200,000 vector slots — large, mostly empty ]
          ↓
[ Stage 1: Feed Domain A text → vectors activate in sparse regions ]
          ↓
[ Stage 2: Feed Domain B text → grows into NEW unused regions ]
          ↓
[ Query: route through active vectors only (FAISS GPU-accelerated) ]
          ↓
[ Epistemic verdict: Confident / Honest Unknown / Boundary Violation ]
          ↓
[ Boundary violation → auto-ingest negative counterexample (no retrain) ]
          ↓
[ Pruning pass: decay dormant, reinforce active ]
          ↓
[ Expansion: new domain claims previously dormant space ]
```

### Three-Tier Epistemic Verdict

Every query returns one of three structurally distinct verdicts:

| Verdict | Condition | Meaning |
| --- | --- | --- |
| ✓ **Confident** | Similarity ≥ threshold, positive domain | The query landed in a well-populated positive knowledge region. |
| ⚠️ **Boundary Violation** | Nearest slot is a registered negative domain | The query concept exists but the combination is architecturally invalid. A negative counterexample is auto-ingested. |
| ❓ **Honest Unknown** | Similarity < threshold | The query fell into dormant space — no learned representation found. BrainGrow does not confabulate. |

---

## The Six Tabs

| Tab | What It Demonstrates |
| --- | --- |
| **Grow** | Pre-allocated vector space. Knowledge grows into it progressively. Active vs. dormant vectors visualised in real time via UMAP / PCA. Includes Stage Diff and Refresh UMAP controls. |
| **Query** | Routing through active vectors only via FAISS index. Boundary violations trigger reactive negative ingestion and are logged in Tab 5. |
| **Prune** | Use-dependent pruning pass. Dormant vectors decay. Active vectors strengthen. Before/after comparison visualised. FAISS index rebuilt post-prune. |
| **Compare** | Hallucination demo. Runs identical queries against a saturated DenseModel and BrainGrow side-by-side. Demonstrates that hallucination is an architectural property — not a scale or data-quantity problem. |
| **Network** | Save / load complete network state as `.bgstate` files. Autosave after every Ingest Stage. Includes **hallucination risk audit** and **reactive correction log** (Knowledge Maintenance panel). |
| **TinyStories** | Scale test against the `roneneldan/TinyStories` corpus — 100,000 real-world story snippets, 200,000 slot space, unlabeled developmental growth. Three progressive stages (smoke test → small scale → full scale). |

---

## Project Structure

```
braingrow/
├── main.py                    # Gradio app entry point (6-tab UI)
├── session.py                 # BrainGrowSession — all business logic, no state in main.py
├── vector_space.py            # Pre-allocation, FAISS indexing, activation tracking, pruning
├── growth_engine.py           # Staged ingestion, batch encoding, slot assignment
├── query_router.py            # FAISS-backed routing through active vectors only
├── comparison_harness.py      # DenseModel vs BrainGrow hallucination comparison (Tab 4)
├── knowledge_maintenance.py   # Reactive correction + proactive hallucination risk audit
├── tinystories_loader.py      # TinyStories data pipeline (Tab 6, requires datasets)
├── visualizer.py              # UMAP projection & Plotly charts
├── instrumentation.py         # Optional timing / error tracing (BRAINGROW_TRACE=1)
├── utils.py                   # Shared unit-normalised encoding utilities
├── experiment_4_7.py          # Section 4.7 reproducibility script (online epistemic correction)
├── requirements.txt           # Core Python dependencies
├── saves/                     # .bgstate network snapshots (autosave target)
└── tests/                     # Pytest suite (one test file per module)
```

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
- faiss-gpu *(recommended — GPU-accelerated retrieval)* or faiss-cpu *(fallback)*
- datasets *(optional — required for Tab 6 TinyStories only)*

---

## Setup

```bash
# Clone the project
git clone https://github.com/msmonroe/Braingrow.git
cd Braingrow

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# PyTorch — choose one:
# GPU (CUDA 12.x — recommended for FAISS acceleration):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only (sufficient for the POC):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# FAISS — choose one:
pip install faiss-gpu       # GPU-accelerated (RTX / CUDA required)
pip install faiss-cpu       # CPU fallback — graceful degradation, no code changes needed

# Optional: TinyStories scale experiment (Tab 6)
pip install datasets

# Launch
python main.py
```

Then open the URL printed in the console (default: `http://localhost:7860`).

> **Note:** If `faiss` is not installed, BrainGrow falls back to brute-force cosine similarity search automatically. All functionality is preserved — only retrieval speed is affected at large slot counts.

### Tracing / Instrumentation

```bash
BRAINGROW_TRACE=1 python main.py

# Redirect trace output to a file:
BRAINGROW_TRACE=1 BRAINGROW_LOG=braingrow.log python main.py
```

---

## Demo Script

Follow these steps to observe the developmental dynamics in action:

| Step | Action | Expected Observation |
| --- | --- | --- |
| 1 | **Initialize** | Launch app. UMAP shows 200,000 grey dormant slots. |
| 2 | **Stage 1 — Science** | Ingest science chunks. UMAP lights up in a sparse cluster. Histogram shows a tiny active fraction. |
| 3 | **Stage 2 — History** | Ingest history chunks. A new cluster appears in a *different* region. Science cluster unchanged. |
| 4 | **Query — Science** | Ask a science question. Routing highlights the science cluster only. Status bar shows `FAISS ✓` when GPU index is active. |
| 5 | **Query — History** | Ask a history question. Routing highlights the history cluster. No cross-contamination. |
| 6 | **Cross-domain collision** | Ask "What is the thermodynamics of cooking ancient Rome?" — observe `⚠️ BOUNDARY VIOLATION` and note the auto-correction count increment. |
| 7 | **Prune** | Run pruning at threshold 0.2. Low-activation slots grey out. Core concepts survive. FAISS index rebuilds automatically. |
| 8 | **Expand** | Ingest Stage 3 (e.g. cooking). Grows into space freed by pruning. |
| 9 | **Compare** | Switch to Tab 4. Run Known / Partial / Unknown queries. BrainGrow returns honest uncertainty; DenseModel hallucinates. |
| 10 | **Audit** | Switch to Tab 5. Click *Run Hallucination Risk Audit*. Observe domain risk scores. Click *Show Correction Log* to see reactive corrections accumulated from Step 6. |
| 11 | **Save** | Save the network state to `saves/` before lengthy experiments. |
| 12 | **TinyStories** | Switch to Tab 6. Run Stage A (smoke test, ~1k chunks), then Stage B (10k), then Stage C (full scale). Enable Autosave first. |

---

## Knowledge Maintenance (Tab 5)

BrainGrow includes an active knowledge maintenance system that transforms it from a static knowledge store into a self-improving developmental system.

### Reactive Correction

When a query triggers a **Boundary Violation**, `KnowledgeMaintenance.on_boundary_violation()` automatically ingests a targeted negative counterexample slot into the vector space — without retraining. Subsequent queries of the same type encounter the negative slot first and return `BOUNDARY VIOLATION` rather than false confidence.

### Proactive Audit

`KnowledgeMaintenance.audit_hallucination_risk()` scans all registered domains and computes a hallucination risk score based on the ratio of positive to negative slots:

| Risk Level | Condition |
| --- | --- |
| ⛔ HIGH | No negative examples, or positive:negative ratio ≥ 5:1 |
| ⚠️ MEDIUM | Positive:negative ratio ≥ 2:1 |
| ✅ BALANCED | Positive:negative ratio < 2:1 |

> **Research finding:** Hallucination risk is proportional to positive/negative slot ratio in the nearest knowledge neighbourhood. Networks with high positive/negative ratios confabulate confidently; networks with balanced positive/negative knowledge maintain epistemic humility. This mirrors the asymmetric positive/negative learning dynamic observed in biological neural development.

---

## Reproducibility — Section 4.7 Experiment

To reproduce the *Online Epistemic Correction Without Retraining* finding:

```bash
python experiment_4_7.py
```

This script:
1. Ingests positive knowledge across three domains (science, history, cooking)
2. Runs a fixed set of cross-domain collision queries — records baseline boundary violation rate
3. Allows reactive corrections to accumulate across a second pass
4. Re-runs the identical query set — measures post-correction violation rate
5. Runs the hallucination risk audit before and after
6. Compares against DenseModel (which cannot self-correct without retraining)
7. Saves full results to `experiment_4_7_results.json`

---

## FAISS Retrieval Indexing

`vector_space.py` maintains a FAISS index alongside the slot tensor, reducing query time from O(n) brute-force to approximately O(log n):

| Active Slots | Brute-Force | FAISS (GPU) |
| --- | --- | --- |
| 10,000 | < 1 ms | < 1 ms |
| 200,000 | ~10 ms | < 1 ms |
| 1,000,000 | ~50 ms | < 1 ms |
| 10,000,000 | ~500 ms | ~2 ms |

Index type selection is automatic:
- `IndexFlatIP` — exact search, used below 500k active slots
- `IndexIVFFlat` — approximate search, used above 500k slots

The index is built lazily on first query after startup or load, rebuilt after pruning passes, and marked dirty on every `assign_slot()` call. It is **not** persisted in `.bgstate` files — it rebuilds automatically on load.

---

## Key Design Decisions

| Decision | Rationale |
| --- | --- |
| 200,000 pre-allocated slots | Sufficient headroom for TinyStories full-scale run without live reallocation. |
| `all-MiniLM-L6-v2` (384-dim) | Compact, fast, well-calibrated for semantic similarity at CPU speeds. |
| FAISS `IndexFlatIP` | Exact inner-product search on unit vectors = exact cosine similarity. GPU-accelerated on CUDA hardware. |
| Reinforce threshold 0.92 | Near-duplicate chunks strengthen existing slots rather than wasting dormant space. |
| Three-tier epistemic verdict | Confident / Honest Unknown / Boundary Violation emerge from geometry — no trained behavior. |
| Reactive negative ingestion | Boundary violations auto-generate counterexamples — online correction without retraining. |
| Thread-safe `RLock` | Gradio's concurrent callbacks can write without race conditions. |
| `BrainGrowSession` business-logic class | All state and logic isolated from Gradio; trivially testable and replaceable. |
| `.bgstate` persistence | Full snapshot (embeddings + activations + metadata) prevents data loss on long runs. |
| FAISS not persisted in `.bgstate` | Index rebuilds in seconds from the slot tensor — no serialisation complexity. |

---

## Success Metrics

The POC is considered successful when it demonstrates:

- **Spatial separation** — domains ingested at different stages occupy geometrically distinct regions.
- **Non-destructive expansion** — adding a new domain does not shift or corrupt previously activated regions.
- **Routing isolation** — queries correctly activate domain-relevant slots and ignore unrelated ones.
- **Pruning recovery** — after a pruning pass, a new domain successfully claims reclaimed dormant space.
- **Honest uncertainty** — BrainGrow returns low-confidence or no results for unknown-concept queries while DenseModel confidently hallucinates.
- **Boundary violation detection** — cross-domain collision queries surface `BOUNDARY VIOLATION` rather than false confidence.
- **Online epistemic correction** — boundary violation rate drops measurably after reactive negative ingestion, without retraining.
- **Visual legibility** — a non-technical observer can watch the space grow and intuitively understand what is happening.
- **Scale durability** — the TinyStories pipeline ingests 100k story chunks across 200,000 slots without slot exhaustion or UMAP collapse.

---

## Running Tests

```bash
pytest tests/
```

The test suite covers all core modules: vector space, growth engine, query router, comparison harness, knowledge maintenance, session, visualizer, instrumentation, utilities, and the TinyStories loader.

---

## Future Directions

- **Embodied feedback loop** — replace static text ingestion with agent-environment interaction; slots activate based on reward signal, not just semantic similarity.
- **Hierarchical pruning** — staged fine-to-coarse pruning mirroring cortical development.
- **Cross-domain generalization** — test whether concepts in overlapping vector regions produce emergent analogical reasoning.
- **Circuit tracing integration** — map BrainGrow's epistemic geometry onto transformer feature circuits using attribution graph techniques.
- **Foundation model bridging** — investigate whether targeted feature steering in transformer models can replicate BrainGrow's online epistemic correction at the circuit level.
- **Comparison baseline** — train an equivalently-sized static model on the same data; compare query accuracy and representational geometry.
- **Publication** — arXiv paper in preparation. POC results constitute a viable submission to NeurIPS, ICLR, or AAAI.

---

## Citation

Paper in preparation. If you use this work, please cite:

```
Monroe, M. (2026). BrainGrow: A Developmental Vector Architecture for Epistemic AI.
arXiv preprint. Vektas Solutions.
GitHub: https://github.com/msmonroe/Braingrow
```

---

## License

Research prototype. Not licensed for commercial use.
Vektas Solutions · vektassolutions.com · April 2026
