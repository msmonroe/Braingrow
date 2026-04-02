"""
main.py — BrainGrow Gradio application entry point.

Thin UI layer: builds the Gradio interface and wires callbacks to the
BrainGrowSession business-logic class.  No application state lives here.

Run:
    python main.py
Then open the URL printed to the console.
"""

from __future__ import annotations

from typing import Tuple

import gradio as gr

from comparison_harness import known_queries
from session import BrainGrowSession, STAGE_PRESETS

# ---------------------------------------------------------------------------
# Single shared session — owns all state and business logic
# ---------------------------------------------------------------------------
session = BrainGrowSession()


# ---------------------------------------------------------------------------
# UI-layer helpers (Gradio component construction only)
# ---------------------------------------------------------------------------

def _refresh_saves_dropdown() -> gr.Dropdown:
    files = session.list_saves()
    return gr.Dropdown(choices=files, value=files[0] if files else None)


def save_network(description: str) -> Tuple[str, gr.Dropdown]:
    return session.save_network(description), _refresh_saves_dropdown()


def delete_save(selected_path: str) -> Tuple[str, gr.Dropdown]:
    return session.delete_save(selected_path), _refresh_saves_dropdown()


def get_query_choices(query_type: str) -> gr.Dropdown:
    choices = session.get_query_choices(query_type)
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


_HEADER_MD = """
# 🧠 BrainGrow
**Developmental AI Architecture — POC**  ·  Vektas Solutions  ·  March 2026

> Pre-allocates 200,000 vector slots.  Knowledge grows into dormant regions
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

_NETWORK_INTRO = """
### Tab 5 — Network (Save / Load)

Persist and restore the complete network state — all embeddings, activations,
domains, and stage history — as a `.bgstate` file.

- **Save Network** — snapshot the current state to `saves/`.  
- **Load Network** — restore a previous snapshot into the active vector space.  
- **Autosave** — enable to checkpoint automatically after every Ingest Stage.
  Essential for long TinyStories runs (20–30 minutes — protect against data loss).
"""

_TINYSTORIES_INTRO = """
### Tab 6 — TinyStories Experiment

Scale test against the **roneneldan/TinyStories** corpus — 100,000 real-world
story snippets, 200,000 slot space, unlabeled developmental growth.

Three progressive stages (run in order, catch bugs early):

| Stage | Chunks | Purpose |
|-------|--------|---------|
| **A — Smoke test** | 1,000 | Verify pipeline, check UMAP renders at new scale |
| **B — Small scale** | 10,000 | Check clustering, run Tab 4 hallucination comparison |
| **C — Full scale** | 100k sample | All three progressive tests, screenshot results |

Requires: `pip install datasets`

> *Enable Autosave in Tab 5 before starting Stage C.  A 30-minute run without persistence is a one-time demo, not a research asset.*
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
                            umap_btn = gr.Button("Refresh UMAP")
                            reset_btn = gr.Button("Reset", variant="stop")
                        grow_status = gr.Textbox(
                            label="Status", interactive=False, lines=2
                        )

                    with gr.Column(scale=2):
                        grow_umap = gr.Plot(label="Vector Space (UMAP / PCA)")
                        grow_hist = gr.Plot(label="Activation Histogram")

                grow_btn.click(
                    fn=session.ingest,
                    inputs=[grow_text, grow_domain],
                    outputs=[grow_status, grow_umap, grow_hist],
                )
                diff_btn.click(
                    fn=session.view_diff,
                    inputs=[],
                    outputs=[grow_umap],
                )
                umap_btn.click(
                    fn=session.refresh_umap,
                    inputs=[],
                    outputs=[grow_umap],
                )
                reset_btn.click(
                    fn=session.reset_all,
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
                    fn=session.query,
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
                    fn=session.run_prune,
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
                    fn=session.run_comparison_tab,
                    inputs=[compare_type, compare_query],
                    outputs=[
                        compare_table,
                        compare_dense_umap,
                        compare_bg_umap,
                        compare_status,
                    ],
                )

            # ----------------------------------------------------------------
            # TAB 5 — NETWORK
            # ----------------------------------------------------------------
            with gr.Tab("Network"):
                gr.Markdown(_NETWORK_INTRO)
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        net_description = gr.Textbox(
                            label="Save Description (optional)",
                            placeholder="e.g. tinystories_10k, after-pruning-v2…",
                            lines=1,
                        )
                        net_save_btn = gr.Button("💾  Save Network", variant="primary")
                        net_save_status = gr.Textbox(
                            label="Save Status", interactive=False, lines=2
                        )
                        gr.Markdown("---")
                        net_autosave = gr.Checkbox(
                            label="🔄  Enable Autosave after each Ingest Stage",
                            value=False,
                        )
                        net_autosave_status = gr.Textbox(
                            label="Autosave Status", interactive=False, lines=1
                        )

                    with gr.Column(scale=1, min_width=280):
                        net_saves_dropdown = gr.Dropdown(
                            choices=session.list_saves(),
                            label="Saved Networks (.bgstate)",
                            interactive=True,
                        )
                        with gr.Row():
                            net_refresh_btn  = gr.Button("🔄  Refresh List")
                            net_load_btn     = gr.Button("Load Selected", variant="primary")
                            net_delete_btn   = gr.Button("🗑️  Delete", variant="stop")
                        net_load_status = gr.Textbox(
                            label="Load Status", interactive=False, lines=4
                        )

                with gr.Row():
                    net_info_btn = gr.Button("Network Info")
                    net_info_md  = gr.Markdown()

                with gr.Row():
                    net_umap = gr.Plot(label="Vector Space (UMAP)")
                    net_hist = gr.Plot(label="Activation Histogram")

                net_save_btn.click(
                    fn=save_network,
                    inputs=[net_description],
                    outputs=[net_save_status, net_saves_dropdown],
                )
                net_refresh_btn.click(
                    fn=_refresh_saves_dropdown,
                    inputs=[],
                    outputs=[net_saves_dropdown],
                )
                net_load_btn.click(
                    fn=session.load_network,
                    inputs=[net_saves_dropdown],
                    outputs=[net_load_status, net_umap, net_hist],
                )
                net_delete_btn.click(
                    fn=delete_save,
                    inputs=[net_saves_dropdown],
                    outputs=[net_load_status, net_saves_dropdown],
                )
                net_autosave.change(
                    fn=session.toggle_autosave,
                    inputs=[net_autosave],
                    outputs=[net_autosave_status],
                )
                net_info_btn.click(
                    fn=session.get_network_info,
                    inputs=[],
                    outputs=[net_info_md],
                )

            # ----------------------------------------------------------------
            # TAB 6 — TINYSTORIES
            # ----------------------------------------------------------------
            with gr.Tab("TinyStories"):
                gr.Markdown(_TINYSTORIES_INTRO)
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        ts_preset = gr.Dropdown(
                            choices=list(STAGE_PRESETS.keys()),
                            value=list(STAGE_PRESETS.keys())[0] if STAGE_PRESETS else None,
                            label="Experiment Stage Preset",
                        )
                        gr.Markdown("*Or set custom values:*")
                        ts_custom_sample = gr.Number(
                            label="Sample Size (stories)",
                            value=2000,
                            precision=0,
                        )
                        ts_custom_chunks = gr.Number(
                            label="Max Chunks",
                            value=1000,
                            precision=0,
                        )
                        with gr.Row():
                            ts_run_btn = gr.Button(
                                "🚀  Load & Ingest TinyStories",
                                variant="primary",
                            )
                            ts_umap_btn = gr.Button("Refresh UMAP")
                        ts_status = gr.Textbox(
                            label="Status", interactive=False, lines=5
                        )

                    with gr.Column(scale=2):
                        ts_umap = gr.Plot(label="Vector Space (UMAP)")
                        ts_hist = gr.Plot(label="Activation Histogram")

                ts_run_btn.click(
                    fn=session.run_tinystories_stage,
                    inputs=[ts_preset, ts_custom_sample, ts_custom_chunks],
                    outputs=[ts_status, ts_umap, ts_hist],
                )
                ts_umap_btn.click(
                    fn=session.refresh_umap,
                    inputs=[],
                    outputs=[ts_umap],
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
