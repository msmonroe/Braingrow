# BrainGrow

**Developmental AI Architecture — Proof of Concept**  
Vektas Solutions · March 2026 · Author: Matthew Monroe

---

## Overview

Current large language models are trained in a single static run — compressing all knowledge into a frozen weight matrix before the model ever interacts with the world. BrainGrow inverts this paradigm.

Inspired by human neurodevelopment — where an infant is born with *more* synaptic connections than an adult, and the brain sculpts intelligence through use-dependent pruning — BrainGrow pre-allocates a large vector space and allows knowledge to grow into it organically through staged exposure and interaction. Dormant capacity is preserved for future expansion rather than discarded.

> **Core hypothesis:** AGI-adjacent behavior may emerge not from more data fed into a static architecture, but from developmental dynamics — growth, reinforcement, and pruning over time.

---

## Architecture

```
[ Pre-allocate 10,000 vector slots — large, mostly empty ]
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

### The Three Tabs

| Tab | What It Demonstrates |
|-----|----------------------|
| **Grow** | Pre-allocated vector space. Knowledge grows into it progressively. Active vs. dormant vectors visualised in real time. |
| **Query** | Routing through active vectors only. New domain knowledge grows into previously dormant space without overwriting existing knowledge. |
| **Prune** | Use-dependent pruning pass. Dormant vectors decay. Active vectors strengthen. Before/after comparison visualised. |

---

## Project Structure

```
braingrow/
├── main.py              # Gradio app entry point (3-tab UI)
├── vector_space.py      # Pre-allocation, activation tracking, pruning
├── growth_engine.py     # Staged ingestion, embedding, slot assignment
├── query_router.py      # Routes queries through active vectors only
├── visualizer.py        # UMAP projection & Plotly charts
├── requirements.txt     # Python dependencies
└── .gitignore
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

---

## Setup

```bash
# Clone / download the project, then:
cd braingrow

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# PyTorch (CPU build — sufficient for the POC):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Launch
python main.py
```

Then open the URL printed in the console (default: `http://localhost:7860`).

---

## Demo Script

Follow these steps to see the developmental dynamics in action:

| Step | Action | Expected Observation |
|------|--------|----------------------|
| 1 | **Initialize** | Launch app. UMAP shows 10,000 grey dormant slots. |
| 2 | **Stage 1 — Science** | Ingest 10 science chunks. UMAP lights up in a sparse cluster. Histogram shows ~0.1% active. |
| 3 | **Stage 2 — History** | Ingest 10 history chunks. A new cluster appears in a *different* region. Science cluster unchanged. |
| 4 | **Query — Science** | Ask a science question. Routing highlights the science cluster only. |
| 5 | **Query — History** | Ask a history question. Routing highlights the history cluster. No cross-contamination. |
| 6 | **Prune** | Run pruning at threshold 0.2. Low-activation slots grey out. Core concepts survive. |
| 7 | **Expand** | Ingest Stage 3 (e.g. cooking). Grows into space freed by pruning. |

---

## Success Metrics

The POC is considered successful when it demonstrates:

- **Spatial separation** — domains ingested at different stages occupy geometrically distinct regions.
- **Non-destructive expansion** — adding a new domain does not shift or corrupt previously activated regions.
- **Routing isolation** — queries correctly activate domain-relevant slots and ignore unrelated ones.
- **Pruning recovery** — after a pruning pass, a new domain successfully claims reclaimed dormant space.
- **Visual legibility** — a non-technical observer can watch the space grow and intuitively understand what is happening.

---

## Future Directions

- **Embodied feedback loop** — replace static text ingestion with agent-environment interaction; slots activate based on reward signal, not just semantic similarity.
- **Hierarchical pruning** — staged fine-to-coarse pruning mirroring cortical development.
- **Cross-domain generalization** — test whether concepts in overlapping vector regions produce emergent analogical reasoning.
- **Comparison baseline** — train an equivalently-sized static model on the same data; compare query accuracy and representational geometry.
- **Publication** — POC results constitute a viable workshop paper submission to NeurIPS, ICLR, or AAAI.

---

## License

Internal research prototype. Vektas Solutions · vektassolutions.com
