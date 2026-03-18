"""
main.py — BrainGrow Gradio application entry point.

Three-tab UI backed by a single shared VectorSpace instance:
  Tab 1 — Grow   : staged text ingestion, live UMAP + histogram
  Tab 2 — Query  : semantic routing through active slots only
  Tab 3 — Prune  : threshold-based pruning with before/after visualisation

Run:
    python main.py
Then open the URL printed to the console.
"""

from __future__ import annotations

import re
from typing import Tuple

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

from growth_engine import GrowthEngine
from query_router import QueryRouter
from vector_space import VectorSpace
from visualizer import Visualizer

# ---------------------------------------------------------------------------
# Shared state — one model load for the whole session
# ---------------------------------------------------------------------------
print("Loading sentence-transformers model (all-MiniLM-L6-v2)…")
_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

vs = VectorSpace()
engine = GrowthEngine(vs, _model)
router = QueryRouter(vs, _model)
viz = Visualizer()

# Store pre-prune activation snapshot for comparison chart
_prune_before: np.ndarray | None = None

# ---------------------------------------------------------------------------
# Tab 1 — Grow helpers
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str) -> list[str]:
    """Split multiline / multi-sentence text into individual concept chunks."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 2:
        return lines
    # Single-line input: split on sentence boundaries
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences if sentences else [text.strip()]


def ingest(text_input: str, domain_label: str) -> Tuple:
    if not text_input.strip():
        return (
            "⚠️  Please enter some text.",
            viz.plot_umap(vs),
            viz.plot_histogram(vs),
        )
    domain = domain_label.strip() or "default"
    chunks = [(c, domain) for c in _split_into_chunks(text_input)]

    result = engine.ingest_stage(chunks)

    status = (
        f"Stage {result['stage_number']} complete — "
        f"{len(result['slots_activated'])} new slots activated, "
        f"{len(result['slots_reinforced'])} reinforced — "
        f"{result['dormant_remaining']:,} dormant remaining."
    )
    return status, viz.plot_umap(vs), viz.plot_histogram(vs)


def view_diff() -> gr.Plot:
    diff = engine.get_stage_diff()
    if not diff["new_slots"]:
        return viz.plot_umap(vs)
    return viz.plot_stage_diff(vs, diff["new_slots"])


def reset_all() -> Tuple:
    vs.reset()
    engine.reset()
    return (
        "Vector space reset — all slots dormant.",
        viz.plot_umap(vs),
        viz.plot_histogram(vs),
    )


# ---------------------------------------------------------------------------
# Tab 2 — Query helpers
# ---------------------------------------------------------------------------

def query(text_input: str, top_k: int) -> Tuple[str, str]:
    if not text_input.strip():
        return "⚠️  Please enter a query.", ""

    result = router.route_query(text_input.strip(), top_k=int(top_k))
    ratio = f"Active: {result['active_count']:,}  |  Dormant: {result['dormant_count']:,}"

    if not result["matches"]:
        return "No active slots found — ingest some text first.", ratio

    lines = []
    for m in result["matches"]:
        lines.append(
            f"**[{m['domain']}]** {m['label']}  \n"
            f"  similarity: `{m['similarity']:.4f}`  |  "
            f"activation: `{m['activation']:.4f}`"
        )
    return "\n\n---\n\n".join(lines), ratio


# ---------------------------------------------------------------------------
# Tab 3 — Prune helpers
# ---------------------------------------------------------------------------

def run_prune(threshold: float) -> Tuple[str, gr.Plot]:
    global _prune_before
    _prune_before = vs.activation.detach().numpy().copy()
    result = vs.prune(threshold=float(threshold))
    after = vs.activation.detach().numpy().copy()
    fig = viz.plot_prune_comparison(_prune_before, after)
    status = (
        f"Pruning complete — threshold: {threshold:.2f}  |  "
        f"Pruned: {result['pruned_count']:,} slots  |  "
        f"Active before → after: {result['before_active']:,} → {result['after_active']:,}"
    )
    return status, fig


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_HEADER_MD = """
# 🧠 BrainGrow
**Developmental AI Architecture — POC**  ·  Vektas Solutions  ·  March 2026

> Pre-allocates 10,000 vector slots.  Knowledge grows into dormant regions
> through staged exposure — no static training run.
"""

_GROW_INTRO = """
### Tab 1 — Grow
Feed text into the vector space one stage at a time.  Each sentence / line
becomes a concept that *grows into* the nearest dormant region.  Watch the
UMAP light up as domains form geometrically distinct clusters.
"""

_QUERY_INTRO = """
### Tab 2 — Query
Routes your question through **active slots only** — dormant space is ignored.
Matched slots are reinforced, raising their activation score.
"""

_PRUNE_INTRO = """
### Tab 3 — Prune
Slots below the activation threshold are zeroed out and their space is
reclaimed.  The before / after histogram shows how the activation landscape
shifts — and how room opens for the next growth stage.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="BrainGrow — Developmental AI POC",
        theme=gr.themes.Soft(),
        css=".label-wrap { font-weight: 600; }",
    ) as demo:
        gr.Markdown(_HEADER_MD)

        with gr.Tabs():

            # ----------------------------------------------------------------
            # TAB 1 — GROW
            # ----------------------------------------------------------------
            with gr.Tab("Grow"):
                gr.Markdown(_GROW_INTRO)
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        grow_text = gr.Textbox(
                            label="Text Input",
                            placeholder=(
                                "Paste text here.\n"
                                "Each line / sentence becomes one concept chunk."
                            ),
                            lines=10,
                        )
                        grow_domain = gr.Textbox(
                            label="Domain Label",
                            placeholder="e.g. science, history, cooking",
                        )
                        with gr.Row():
                            grow_btn = gr.Button("Ingest Stage", variant="primary")
                            diff_btn = gr.Button("Stage Diff")
                            reset_btn = gr.Button("Reset", variant="stop")
                        grow_status = gr.Textbox(
                            label="Status", interactive=False, lines=2
                        )

                    with gr.Column(scale=2):
                        grow_umap = gr.Plot(label="Vector Space (UMAP / PCA)")
                        grow_hist = gr.Plot(label="Activation Histogram")

                grow_btn.click(
                    fn=ingest,
                    inputs=[grow_text, grow_domain],
                    outputs=[grow_status, grow_umap, grow_hist],
                )
                diff_btn.click(
                    fn=view_diff,
                    inputs=[],
                    outputs=[grow_umap],
                )
                reset_btn.click(
                    fn=reset_all,
                    inputs=[],
                    outputs=[grow_status, grow_umap, grow_hist],
                )

            # ----------------------------------------------------------------
            # TAB 2 — QUERY
            # ----------------------------------------------------------------
            with gr.Tab("Query"):
                gr.Markdown(_QUERY_INTRO)
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        query_text = gr.Textbox(
                            label="Query",
                            placeholder="Ask a question or enter a concept…",
                            lines=4,
                        )
                        query_k = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Top-K results",
                        )
                        query_btn = gr.Button("Route Query", variant="primary")
                        query_ratio = gr.Textbox(
                            label="Active / Dormant Ratio", interactive=False
                        )

                    with gr.Column(scale=2):
                        query_results = gr.Markdown(label="Matched Concepts")

                query_btn.click(
                    fn=query,
                    inputs=[query_text, query_k],
                    outputs=[query_results, query_ratio],
                )

            # ----------------------------------------------------------------
            # TAB 3 — PRUNE
            # ----------------------------------------------------------------
            with gr.Tab("Prune"):
                gr.Markdown(_PRUNE_INTRO)
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        prune_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                            label="Prune Threshold",
                        )
                        prune_btn = gr.Button("Run Prune Pass", variant="primary")
                        prune_status = gr.Textbox(
                            label="Status", interactive=False, lines=2
                        )

                    with gr.Column(scale=2):
                        prune_fig = gr.Plot(label="Before / After Comparison")

                prune_btn.click(
                    fn=run_prune,
                    inputs=[prune_slider],
                    outputs=[prune_status, prune_fig],
                )

    return demo


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, server_name="0.0.0.0")
