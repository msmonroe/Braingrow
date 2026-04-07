# BrainGrow

**Developmental AI Architecture — Proof of Concept**  
Vektas Solutions · April 2026 · Author: Matthew Monroe

![Python](https://img.shields.io/badge/python-3.11+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-research--poc-orange) ![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

---

## Overview

Current large language models are trained in a single static run — compressing all knowledge into a frozen weight matrix before the model ever interacts with the world. BrainGrow inverts this paradigm.

Inspired by human neurodevelopment — where an infant is born with *more* synaptic connections than an adult, and the brain sculpts intelligence through use-dependent pruning — BrainGrow pre-allocates a large vector space and allows knowledge to grow into it organically through staged exposure and interaction. Dormant capacity is preserved for future expansion rather than discarded.

> **Core hypothesis:** AGI-adjacent behavior may emerge not from more data fed into a static architecture, but from developmental dynamics — growth, reinforcement, and pruning over time.

### Key Contribution: Three-Tier Epistemic Architecture

BrainGrow produces three distinct epistemic states that emerge from architectural properties of the vector space alone — no RLHF, no fine-tuning, no trained suppression:

| State | Meaning | Trigger |
| --- | --- | --- |
| ✓ **Confident** | Query maps strongly to a known positive domain | Similarity above threshold, positive domain |
| 🤔 **Honest Unknown** | Query falls into dormant space — genuinely not learned yet | Similarity below threshold |
| ⚠️ **Out-of-Bounds** | Query maps to a registered negative domain — system flags the boundary crossing | Nearest slot is a negative domain, regardless of similarity |

No current LLM has this. They have one output state — confident — regardless of which category a query falls into.

---

## Architecture

```
[ Pre-allocate 200,000 vector slots — large, mostly empty ]
          ↓
[ Stage 1: Feed Domain A text → vectors activate in sparse regions ]
          ↓
[ Stage 2: Feed Domain B text → grows into NEW unused regions ]
          ↓
[ Query: route through active vectors only ]
          ↓
[ Pruning pass: decay dormant, reinforce active ]
          ↓
[ Expansion: new domain claims previously dormant space ]
```

### The Six Tabs

| Tab | What It Demonstrates |
| --- | --- |
| **Grow** | Pre-allocated vector space. Knowledge grows into it progressively. Active vs. dormant vectors visualised in real time via UMAP / PCA. Includes Stage Diff and Refresh UMAP controls. |
| **Query** | Routing through active vectors only. New domain knowledge grows into previously dormant space without overwriting existing knowledge. |
| **Prune** | Use-dependent pruning pass. Dormant vectors decay. Active vectors strengthen. Before/after comparison visualised. |
| **Compare** | Hallucination demo. Runs identical queries against a saturated DenseModel and BrainGrow side-by-side. BrainGrow returns one of three epistemic states (Confident / Honest Unknown / Out-of-Bounds); DenseModel always returns confident regardless of whether it should. Demonstrates that hallucination is an architectural property — not a scale or data-quantity problem. |
| **Network** | Save / load complete network state as `.bgstate` files. Autosave after every Ingest Stage (essential for long TinyStories runs). |
| **TinyStories** | Scale test against the `roneneldan/TinyStories` corpus — 100,000 real-world story snippets, 200,000 slot space, unlabeled developmental growth. Three progressive stages (smoke test → small scale → full scale). |

---

## Project Structure

```
braingrow/
├── main.py                  # Gradio app entry point (6-tab UI)
├── session.py               # BrainGrowSession — all business logic, no state in main.py
├── vector_space.py          # Pre-allocation, activation tracking, pruning (200k slots)
├── growth_engine.py         # Staged ingestion, batch encoding, slot assignment
├── query_router.py          # Routes queries through active vectors only
├── comparison_harness.py    # DenseModel vs BrainGrow hallucination comparison (Tab 4)
├── tinystories_loader.py    # TinyStories data pipeline (Tab 6, requires datasets)
├── visualizer.py            # UMAP projection & Plotly charts
├── instrumentation.py       # Optional timing / error tracing (BRAINGROW_TRACE=1)
├── utils.py                 # Shared unit-normalised encoding utilities
├── requirements.txt         # Core Python dependencies
├── saves/                   # .bgstate network snapshots (autosave target)
└── tests/                   # Pytest suite (one test file per module)
```

---

## Requirements

* Python 3.11+
* PyTorch 2.x (CPU or CUDA)
* sentence-transformers
* Gradio 4+
* Plotly
* UMAP-learn
* NumPy
* scikit-learn
* datasets *(optional — required for Tab 6 TinyStories only)*

---

## Setup

```bash
# Clone / download the project, then:
cd braingrow

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# PyTorch (CPU build — sufficient for the POC):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Optional: TinyStories scale experiment (Tab 6)
pip install datasets

# Launch
python main.py
```

Then open the URL printed in the console (default: `http://localhost:7860`).

### Tracing / Instrumentation

To enable verbose timing and error traces during development:

```bash
BRAINGROW_TRACE=1 python main.py

# Redirect trace output to a file:
BRAINGROW_TRACE=1 BRAINGROW_LOG=braingrow.log python main.py
```

---

## Demo Script

Follow these steps to see the developmental dynamics in action:

| Step | Action | Expected Observation |
| --- | --- | --- |
| 1 | **Initialize** | Launch app. UMAP shows 200,000 grey dormant slots. |
| 2 | **Stage 1 — Science** | Ingest science chunks. UMAP lights up in a sparse cluster. Histogram shows a tiny active fraction. |
| 3 | **Stage 2 — History** | Ingest history chunks. A new cluster appears in a *different* region. Science cluster unchanged. |
| 4 | **Query — Science** | Ask a science question. Routing highlights the science cluster only. |
| 5 | **Query — History** | Ask a history question. Routing highlights the history cluster. No cross-contamination. |
| 6 | **Prune** | Run pruning at threshold 0.2. Low-activation slots grey out. Core concepts survive. |
| 7 | **Expand** | Ingest Stage 3 (e.g. cooking). Grows into space freed by pruning. |
| 8 | **Compare** | Switch to Tab 4. Run Known / Partial / Unknown queries. BrainGrow returns one of three epistemic states; DenseModel returns confident for all of them. |
| 9 | **Save** | Switch to Tab 5. Save the network state to `saves/` before lengthy experiments. |
| 10 | **TinyStories** | Switch to Tab 6. Run Stage A (smoke test, ~1k chunks), then Stage B (10k), then Stage C (full scale). Enable Autosave first. |

---

## Success Metrics

The POC is considered successful when it demonstrates:

* **Spatial separation** — domains ingested at different stages occupy geometrically distinct regions.
* **Non-destructive expansion** — adding a new domain does not shift or corrupt previously activated regions.
* **Routing isolation** — queries correctly activate domain-relevant slots and ignore unrelated ones.
* **Pruning recovery** — after a pruning pass, a new domain successfully claims reclaimed dormant space.
* **Three-tier epistemic output** — BrainGrow returns Confident, Honest Unknown, or Out-of-Bounds depending on where in the vector space a query lands, while DenseModel always returns confident.
* **Visual legibility** — a non-technical observer can watch the space grow and intuitively understand what is happening.
* **Scale durability** — the TinyStories pipeline ingests 100k story chunks across 200,000 slots without slot exhaustion or UMAP collapse.

---

## Key Design Decisions

| Decision | Rationale |
| --- | --- |
| 200,000 pre-allocated slots | Sufficient headroom for TinyStories full-scale run without live reallocation. |
| `all-MiniLM-L6-v2` (384-dim) | Compact, fast, well-calibrated for semantic similarity at CPU speeds. |
| Reinforce threshold 0.92 | Near-duplicate chunks strengthen existing slots rather than wasting dormant space. |
| Thread-safe `RLock` | Gradio's concurrent callbacks can write without race conditions. |
| `BrainGrowSession` business-logic class | All state and logic isolated from Gradio; trivially testable and replaceable. |
| `.bgstate` persistence | Full snapshot (embeddings + activations + metadata) prevents data loss on long runs. |

---

## Running Tests

```bash
pytest tests/
```

The test suite covers all core modules: vector space, growth engine, query router, comparison harness, session, visualizer, instrumentation, utilities, and the TinyStories loader.

Tests use a deterministic mock encoder (sha256-seeded 8-dimensional unit vectors) so the full suite runs in seconds without GPU or network access. The mock encoder's output dimension (8) is intentionally smaller than the production `all-MiniLM-L6-v2` encoder (384-dim) — this is by design for test speed and does not affect correctness of the behavioral assertions.

---

## Future Directions

* **Embodied feedback loop** — replace static text ingestion with agent-environment interaction; slots activate based on reward signal, not just semantic similarity.
* **Hierarchical pruning** — staged fine-to-coarse pruning mirroring cortical development.
* **Cross-domain generalization** — test whether concepts in overlapping vector regions produce emergent analogical reasoning.
* **Comparison baseline** — train an equivalently-sized static model on the same data; compare query accuracy and representational geometry.
* **Publication** — POC results constitute a viable workshop paper submission to NeurIPS, ICLR, or AAAI.

---

## License

MIT License — Copyright (c) 2026 Matthew Monroe / Vektas Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
