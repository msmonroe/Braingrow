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

from comparison_harness import (
    BrainGrowModel,
    DenseModel,
    known_queries,
    partial_queries,
    unknown_queries,
)
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

# Tab 4 comparison models — dense_model is rebuilt whenever new data is ingested
dense_model = DenseModel([], _model)
braingrow_model = BrainGrowModel(vs, _model)

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
    global dense_model
    if not text_input.strip():
        return (
            "⚠️  Please enter some text.",
            viz.plot_umap(vs),
            viz.plot_histogram(vs),
        )
    domain = domain_label.strip() or "default"
    chunks = [(c, domain) for c in _split_into_chunks(text_input)]

    result = engine.ingest_stage(chunks)
    dense_model = DenseModel(engine.all_chunks, _model)

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
    global dense_model
    vs.reset()
    engine.reset()
    dense_model = DenseModel([], _model)
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
# Tab 4 — Compare helpers
# ---------------------------------------------------------------------------

_QUERY_MAP = {
    "Known": known_queries,
    "Partial": partial_queries,
    "Unknown": unknown_queries,
}


def get_query_choices(query_type: str) -> gr.Dropdown:
    """Return a refreshed Dropdown matching the selected query type."""
    choices = _QUERY_MAP.get(query_type, [])
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def run_comparison_tab(query_type: str, selected_query: str) -> Tuple:
    if not selected_query:
        return "<p>⚠️ Select a query first.</p>", None, None, "No query selected."

    if dense_model.embeddings.shape[0] == 0:
        msg = (
            "<p>⚠️ No data ingested yet. "
            "Go to <b>Tab 1 — Grow</b> and ingest some text first.</p>"
        )
        return msg, None, None, "No data in vector space."

    d_result = dense_model.query(selected_query)
    b_result = braingrow_model.query(selected_query)

    # Encode query and build unit vector for UMAP overlay
    q_emb = _model.encode(selected_query).astype(np.float32)
    q_norm = float(np.linalg.norm(q_emb)) + 1e-8
    q_np = q_emb / q_norm

    # Verdict logic per spec:
    #   Dense on Unknown queries → HALLUCINATED (red)
    #   BrainGrow when not confident → HONEST (green)
    is_unknown = query_type == "Unknown"
    d_hallucinated = is_unknown and d_result["confident"]
    b_honest = not b_result["confident"]

    d_row_bg = "rgba(255,60,60,0.22)" if d_hallucinated else "rgba(200,200,200,0.06)"
    b_row_bg = "rgba(60,200,100,0.22)" if b_honest else "rgba(200,200,200,0.06)"
    d_verdict_html = (
        '<span style="color:#ff6b6b;font-weight:bold;">⚠ HALLUCINATED</span>'
        if d_hallucinated
        else '<span style="color:#aaa;">✓ Confident</span>'
    )
    b_verdict_html = (
        '<span style="color:#69db7c;font-weight:bold;">✓ HONEST (uncertain)</span>'
        if b_honest
        else '<span style="color:#aaa;">✓ Confident</span>'
    )

    html = f"""
<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px;">
  <thead>
    <tr style="border-bottom:2px solid #555;">
      <th style="padding:8px 12px;text-align:left;">Model</th>
      <th style="padding:8px 12px;text-align:left;">Nearest Concept</th>
      <th style="padding:8px 12px;text-align:left;">Domain</th>
      <th style="padding:8px 12px;text-align:left;">Similarity</th>
      <th style="padding:8px 12px;text-align:left;">Verdict</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background:{d_row_bg};">
      <td style="padding:8px 12px;font-weight:bold;">Dense</td>
      <td style="padding:8px 12px;">{d_result['label'] or '—'}</td>
      <td style="padding:8px 12px;">{d_result['domain'] or '—'}</td>
      <td style="padding:8px 12px;">{d_result['similarity']:.4f}</td>
      <td style="padding:8px 12px;">{d_verdict_html}</td>
    </tr>
    <tr style="background:{b_row_bg};">
      <td style="padding:8px 12px;font-weight:bold;">BrainGrow</td>
      <td style="padding:8px 12px;">{b_result['label'] or '—'}</td>
      <td style="padding:8px 12px;">{b_result['domain'] or '—'}</td>
      <td style="padding:8px 12px;">{b_result['similarity']:.4f}</td>
      <td style="padding:8px 12px;">{b_verdict_html}</td>
    </tr>
  </tbody>
</table>
"""

    dense_fig = viz.plot_dense_umap(
        dense_model.embeddings, dense_model.labels, dense_model.domains, q_np
    )
    bg_fig = viz.plot_umap(vs, q_np)

    status = (
        f"Query: '{selected_query[:60]}'"
        f"  |  Dense sim: {d_result['similarity']:.4f}"
        f"  |  BrainGrow sim: {b_result['similarity']:.4f}"
    )
    return html, dense_fig, bg_fig, status


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

_COMPARE_INTRO = """
### Tab 4 — Compare (Hallucination Demo)

Select a **Query Type** and run the comparison.

- **Known** — queries whose concepts were ingested via Tab 1.
- **Partial** — loosely related to ingested domains.
- **Unknown** — entirely fabricated concepts neither model has seen.

The **Dense** model always returns a confident answer — *hallucination*.
**BrainGrow** returns honest uncertainty when the query lands near dormant space.

> *Hallucination is not a scale problem. It is an architectural property of a saturated vector space.*
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="BrainGrow — Developmental AI POC") as demo:
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

            # ----------------------------------------------------------------
            # TAB 4 — COMPARE
            # ----------------------------------------------------------------
            with gr.Tab("Compare"):
                gr.Markdown(_COMPARE_INTRO)
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        compare_type = gr.Dropdown(
                            choices=["Known", "Partial", "Unknown"],
                            value="Known",
                            label="Query Type",
                        )
                        compare_query = gr.Dropdown(
                            choices=known_queries,
                            value=known_queries[0],
                            label="Query",
                        )
                        compare_btn = gr.Button("Run Comparison", variant="primary")
                        compare_status = gr.Textbox(
                            label="Status", interactive=False, lines=2
                        )

                    with gr.Column(scale=2):
                        compare_table = gr.HTML(label="Comparison Results")
                        with gr.Row():
                            compare_dense_umap = gr.Plot(
                                label="Dense Model — Occupied Space"
                            )
                            compare_bg_umap = gr.Plot(
                                label="BrainGrow — Dormant Space"
                            )

                compare_type.change(
                    fn=get_query_choices,
                    inputs=[compare_type],
                    outputs=[compare_query],
                )
                compare_btn.click(
                    fn=run_comparison_tab,
                    inputs=[compare_type, compare_query],
                    outputs=[
                        compare_table,
                        compare_dense_umap,
                        compare_bg_umap,
                        compare_status,
                    ],
                )

    return demo


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        theme=gr.themes.Soft(),
        css=".label-wrap { font-weight: 600; }",
    )
