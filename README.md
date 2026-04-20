# BrainGrow

**A Sparse Vector Store with Lifecycle-Based Capacity Management and Explicit Abstention**
Vektas Solutions · Author: Matthew Monroe

---

## Status

This project began as an attempt to test whether epistemic abstention (i.e., a
system refusing to answer queries that fall outside its training distribution)
could be made *structural* — a property of the architecture rather than of a
learned or post-hoc threshold. An ablation experiment (see **RAG Comparison**
below) disproved that hypothesis at the corpus scales tested: a flat vector
store with the same encoder and the same similarity threshold produces
identical verdicts on 100/100 queries across three systems.

What remains is still a useful engineering artifact — a managed vector store
with a lifecycle-based capacity model, persistent state, and a three-state
query output (`Confident` / `Honest Unknown` / `Out-of-Bounds`) as a
first-class API. The "developmental architecture" framing has been removed
from this README; the code itself is unchanged and reproduces the reported
behavior.

If you're evaluating this project as a contribution to AI safety or
hallucination research, read the **RAG Comparison** section first so you
understand what the experiment did and did not show.

---

## What This Repo Is

A Python implementation of a sparse vector store that pre-allocates a fixed
slot capacity, ingests text into it via a frozen sentence encoder, manages
slot activation over time, and routes queries through active slots with a
similarity threshold for abstention. The system runs headlessly or through
a Gradio UI with six tabs for interactive exploration.

The engineering ideas that are still interesting independent of the original
epistemics framing:

- **Pre-allocated capacity with O(1) dormant-slot allocation** via a deque,
  avoiding live reallocation during ingestion.
- **Use-dependent lifecycle**: slots are reinforced on query hits, decayed
  periodically, and pruned below a configurable activation threshold. This
  is a memory-management pattern for long-running retrieval systems, not an
  epistemics mechanism.
- **Near-duplicate reinforcement**: incoming embeddings above a cosine
  threshold against an existing active slot reinforce that slot rather than
  claiming a new one.
- **Three-state query output**: `Confident`, `Honest Unknown` (sub-threshold
  similarity), and `Out-of-Bounds` (nearest match falls in a registered
  negative domain). This is a cleaner output API than most RAG pipelines
  surface by default, even if it does not constitute a novel abstention
  mechanism.
- **`.bgstate` persistence**: full snapshot of embeddings + activations +
  metadata for interrupted-experiment resumption.

---

## Architecture

```
[ Pre-allocate N vector slots — large, mostly dormant ]
          ↓
[ Ingest: encode text chunks, assign to dormant slots, mark active ]
          ↓
[ Query: encode, cosine-similarity against active slots only ]
          ↓
[ Threshold: similarity ≥ 0.60 → Confident; else Honest Unknown ]
          ↓
[ Negative-domain check: nearest match in negative domain → Out-of-Bounds ]
          ↓
[ Lifecycle: reinforce on hit, periodic decay, prune below threshold ]
          ↓
[ Pruned slots return to dormant pool, available for future ingestion ]
```

The ablation showed that the abstention verdict is produced entirely by the
threshold step. The lifecycle steps are useful for long-run capacity
management but do not contribute to the abstention decision on a per-query
basis.

### The Six Tabs (Gradio UI)

| Tab | Purpose |
| --- | --- |
| **Grow** | Ingest text into the pre-allocated space. Real-time UMAP shows active vs. dormant slots. |
| **Query** | Route queries through active slots. Returns verdict, nearest match, similarity. |
| **Prune** | Run a pruning pass. Low-activation slots return to the dormant pool. |
| **Compare** | Side-by-side hallucination demo: BrainGrow vs. a no-threshold retrieval baseline (note: this is NOT a comparison against a true dense language model — see RAG Comparison section). |
| **Network** | Save and load `.bgstate` snapshots. Autosave after each ingestion stage. |
| **TinyStories** | Scale test against the `roneneldan/TinyStories` corpus. Staged runs from smoke test (~1k chunks) through full scale (100k chunks). |

---

## Requirements

- Python 3.11+
- PyTorch 2.x (CPU or CUDA)
- sentence-transformers
- Gradio 4+
- Plotly, UMAP-learn, NumPy, scikit-learn
- `datasets` *(optional — required for the TinyStories tab only)*
- `faiss-cpu` *(optional — required for the FAISS ablation baseline)*

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

# PyTorch (CPU build is sufficient for the POC):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Optional: TinyStories scale experiment (Tab 6)
pip install datasets

# Optional: FAISS baseline for the RAG comparison experiment
pip install faiss-cpu

# Launch the Gradio UI
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

## RAG Comparison (the ablation experiment)

This is the experiment that tested whether BrainGrow's abstention behavior
comes from its lifecycle machinery or from the similarity threshold alone.

### What it does

Runs 100 fabricated queries against three systems — **BrainGrow**, a
**torch-tensor flat store + threshold**, and a **FAISS `IndexFlatIP` +
threshold** — using the same encoder (`all-MiniLM-L6-v2`), the same 0.60
threshold, and the same 30-chunk 3-domain corpus. Then compares
verdict-by-verdict.

The key metric: **BrainGrow vs TorchFlatThreshold agreement rate**.

### Observed result

100% verdict agreement across all three pairwise comparisons. All three
systems return identical verdicts on all 100 queries. The lifecycle
machinery does not contribute to the abstention decision — the similarity
threshold produces it.

### Running it

From the repo root:

```bash
python -m experiments.rag_comparison
```

Expect 1–3 minutes on CPU (bottlenecked by sentence-transformers encoding
of 30 chunks + 100 queries).

### Output

Console:
- Per-system scorecards (accuracy per bucket, abstention precision/recall/F1)
- Pairwise verdict-agreement rates between systems
- Per-query disagreement table (for manual inspection)
- A bottom-line interpretation printed in plain English

File:
- `rag_comparison_results.json` — full per-query per-system output for
  deeper analysis or plotting

### Query set

100 queries, four buckets of 25, generated with `random.Random(42)`:

| Bucket            | Expected verdict | What it tests |
|-------------------|------------------|---------------|
| `PURE_NONSENSE`   | `HONEST_UNKNOWN` | Can the system abstain with no semantic anchor? |
| `LEXICAL_OVERLAP` | `HONEST_UNKNOWN` | Can it resist "quantum fermentation"–style token traps? |
| `IN_DOMAIN`       | `CONFIDENT`      | Does it still answer canonical in-domain questions? |
| `NEAR_DOMAIN`     | `HONEST_UNKNOWN` | Where does the threshold fall on the margin? |

To inspect the generated query list without running the full experiment:

```bash
python evaluation/fabricated_queries.py
```

### Known limits of this experiment

- **Sample corpus is 30 chunks.** Matches the original prototype scale but
  is far from production. Larger-scale re-runs (TinyStories 1k+) would
  strengthen or weaken the null result.
- **Threshold is fixed at 0.60 for all systems.** A threshold sweep is a
  natural follow-up — would show whether one system has a better
  precision/recall curve than another across the threshold range.
- **One encoder only.** Swapping encoders may shift absolute numbers.
  Relative behavior between the three systems should be stable because
  they share the encoder.
- **No "Out-of-Bounds" testing.** The negative-domain mechanism is not
  exercised here. It has no flat-threshold equivalent, so it sits outside
  the core comparison.
- **Reinforcement side-effects.** `QueryRouter.route_query()` increments
  activation scores on every match. Over 100 queries this drifts the
  BrainGrow session. Run on a fresh session each time.

### Note on the Compare tab's "DenseModel"

The baseline in Tab 4 of the Gradio UI labeled "DenseModel" is a
fully-saturated retrieval store configured to never abstain. It is **not**
a dense neural language model (e.g., GPT-2, LLaMA). A proper comparison
against an actual generative LLM would require an evaluator layer to
score generated responses as hallucinated vs. abstained vs. correct, and
is not implemented here. Treat Tab 4 as a pedagogical illustration of
thresholded-vs-unthresholded retrieval, not a benchmark against LLMs.

---

## Project Structure

```
braingrow/
├── main.py                  # Gradio app entry point (6-tab UI)
├── session.py               # BrainGrowSession — all business logic
├── vector_space.py          # Pre-allocation, activation tracking, pruning
├── growth_engine.py         # Staged ingestion, batch encoding, slot assignment
├── query_router.py          # Routes queries through active vectors only
├── comparison_harness.py    # Tab 4 harness (no-threshold retrieval baseline)
├── tinystories_loader.py    # TinyStories data pipeline (Tab 6)
├── visualizer.py            # UMAP projection & Plotly charts
├── instrumentation.py       # Optional timing / error tracing
├── utils.py                 # Shared unit-normalized encoding utilities
├── requirements.txt         # Core Python dependencies
├── saves/                   # .bgstate network snapshots (autosave target)
├── tests/                   # Pytest suite (one test file per module)
│
├── baselines/               # RAG comparison — flat-threshold baselines
│   ├── __init__.py
│   └── flat_threshold.py    # TorchFlatThreshold, FAISSFlatThreshold
│
├── evaluation/              # RAG comparison — query set + scoring
│   ├── __init__.py
│   ├── fabricated_queries.py  # Deterministic 100-query generator (seed=42)
│   ├── metrics.py             # Scoring + inter-system agreement
│   └── runner.py              # Headless drivers for baselines and BrainGrow
│
└── experiments/             # RAG comparison — entry point
    ├── __init__.py
    └── rag_comparison.py    # Main entry; writes rag_comparison_results.json
```

---

## Demo Script

To walk through the Gradio UI end-to-end:

| Step | Action | Expected Observation |
| --- | --- | --- |
| 1 | **Initialize** | Launch app. UMAP shows 200,000 grey dormant slots. |
| 2 | **Stage 1 — Science** | Ingest science chunks. UMAP lights up a sparse cluster. |
| 3 | **Stage 2 — History** | Ingest history chunks. A new cluster appears in a different region. |
| 4 | **Query — Science** | Ask a science question. Routing highlights the science cluster. |
| 5 | **Query — History** | Ask a history question. Routing highlights the history cluster. |
| 6 | **Prune** | Run pruning at threshold 0.2. Low-activation slots grey out. |
| 7 | **Expand** | Ingest Stage 3 (e.g., cooking). Grows into space freed by pruning. |
| 8 | **Compare (Tab 4)** | Run Known / Partial / Unknown queries. BrainGrow abstains on sub-threshold; the no-threshold baseline returns nearest-match regardless. *(See caveat above — this is thresholded-vs-unthresholded retrieval, not LLM hallucination.)* |
| 9 | **Save** | Switch to Tab 5. Save the network state to `saves/` before lengthy experiments. |
| 10 | **TinyStories** | Switch to Tab 6. Run Stage A (smoke test, ~1k chunks), then Stage B (10k), then Stage C (full scale). Enable Autosave first. |

---

## Key Design Decisions

| Decision | Rationale |
| --- | --- |
| 200,000 pre-allocated slots | Sufficient headroom for TinyStories full-scale run without live reallocation. |
| `all-MiniLM-L6-v2` (384-dim) | Compact, fast, well-calibrated for semantic similarity at CPU speeds. |
| Reinforce threshold 0.92 | Near-duplicate chunks strengthen existing slots rather than claiming new ones. |
| Confidence threshold 0.60 | Chosen empirically from the encoder's similarity distribution. Not learned; users of other encoders should treat threshold selection as a first-order concern. |
| Thread-safe `RLock` | Gradio's concurrent callbacks can write without race conditions. |
| `BrainGrowSession` business-logic class | State and logic isolated from Gradio; trivially testable and replaceable. |
| `.bgstate` persistence | Full snapshot prevents data loss on long-running experiments. |

---

## Running Tests

```bash
pytest tests/
```

The test suite covers the core modules: vector space, growth engine, query
router, comparison harness, session, visualizer, instrumentation, utilities,
and the TinyStories loader.

---

## Future Directions

Directions that are defensible given what the ablation showed:

- **Threshold sweep.** Evaluate precision/recall across the 0.0–1.0
  threshold range on a larger query set to characterize the operating
  curve, rather than reporting a single operating point.
- **Encoder sensitivity.** Re-run the comparison across multiple
  sentence encoders to understand how threshold calibration generalizes.
- **Scale comparison.** Re-run the RAG comparison at TinyStories scale
  (1k+, 10k+, 100k+ chunks) to confirm or refute the null result across
  corpus sizes.
- **Capacity-management benchmark.** Characterize the lifecycle's real
  contribution — ingest over time with interleaved pruning, measure
  whether the managed store outperforms an unmanaged flat store on
  memory footprint and retrieval latency for long-running workloads.
  This is the direction where the lifecycle is likely to earn its keep.
- **Out-of-Bounds evaluation.** The negative-domain mechanism has no
  empirical evaluation in the current codebase. A labeled corpus with
  designated negative domains would let us measure its precision/recall.

Directions that would require a different architecture, not follow-up
experiments on this one:

- **True structural abstention.** If the research question "can abstention
  emerge from architecture rather than a hyperparameter?" is interesting,
  candidate mechanisms to investigate include learned per-domain
  thresholds, density-based abstention over the active vector space, or
  a separate boundary classifier trained on the active/dormant
  distinction. None of these are in the current code; each would be a
  separate project.

---

## License

Internal research prototype. Vektas Solutions · vektassolutions.com
