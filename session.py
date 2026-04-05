"""
session.py — BrainGrow application session.

BrainGrowSession owns all shared state and business logic.
main.py is a thin Gradio UI layer that delegates entirely to this class.

KnowledgeMaintenance is wired in here:
  - Instantiated in __init__ alongside the other components
  - on_boundary_violation() called automatically from query() when the
    router detects a boundary violation — reactive negative slot ingestion
  - run_audit() exposes the proactive hallucination risk audit to the UI
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from instrumentation import log_event, traced
from comparison_harness import (
    BrainGrowModel,
    DenseModel,
    known_queries,
    partial_queries,
    unknown_queries,
)
from growth_engine import GrowthEngine
from knowledge_maintenance import KnowledgeMaintenance
from query_router import QueryRouter
from utils import encode_unit_numpy
from vector_space import VectorSpace
from visualizer import Visualizer

# Optional TinyStories loader (requires 'datasets' package)
try:
    from tinystories_loader import (
        prepare_experiment,
        STAGE_PRESETS,
        _check_datasets_available,
    )
    _TINYSTORIES_AVAILABLE = True
except ImportError:
    _TINYSTORIES_AVAILABLE = False
    STAGE_PRESETS: dict = {}

    def _check_datasets_available() -> bool:
        return False


_QUERY_MAP = {
    "Known":   known_queries,
    "Partial": partial_queries,
    "Unknown": unknown_queries,
}


class BrainGrowSession:
    """Owns all shared state and implements every piece of application
    business logic. Gradio callbacks in main.py are thin wrappers that
    delegate to this class."""

    SAVES_DIR: Path = Path(__file__).parent / "saves"

    def __init__(self) -> None:
        self.SAVES_DIR.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"BrainGrow starting on device: {device}")
        if device == "cpu":
            print("WARNING: CUDA not available — encoding will be slow")

        print("Loading sentence-transformers model (all-MiniLM-L6-v2)…")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._model = self._model.to(device)
        print("Model loaded.")

        self.vs             = VectorSpace()
        self.engine         = GrowthEngine(self.vs, self._model)
        self.router         = QueryRouter(self.vs, self._model)
        self.viz            = Visualizer()
        self.dense_model    = DenseModel([], self._model)
        self.braingrow_model = BrainGrowModel(self.vs, self._model)

        # Active knowledge maintenance — reactive corrections + proactive audit
        self.maintenance = KnowledgeMaintenance(
            vector_space  = self.vs,
            model         = self._model,
            growth_engine = self.engine,
        )

        self.autosave_enabled: bool          = False
        self._prune_before: Optional[np.ndarray] = None

    # --------------------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _split_into_chunks(text: str) -> List[str]:
        """Split multiline / multi-sentence text into individual concept chunks."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 2:
            return lines
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return sentences if sentences else [text.strip()]

    @staticmethod
    def _format_file_size(path: str) -> str:
        try:
            size = os.path.getsize(path)
            if size >= 1_000_000:
                return f"{size / 1_000_000:.1f} MB"
            return f"{size / 1_000:.0f} KB"
        except OSError:
            return "?"

    def _both_plots(self) -> Tuple:
        return self.viz.plot_umap(self.vs), self.viz.plot_histogram(self.vs)

    # --------------------------------------------------------------------------
    # Tab 1 — Grow
    # --------------------------------------------------------------------------

    @traced
    def ingest(self, text_input: str, domain_label: str) -> Tuple:
        if not text_input.strip():
            return (
                "⚠️ Please enter some text.",
                self.viz.plot_umap(self.vs),
                self.viz.plot_histogram(self.vs),
            )

        domain = domain_label.strip() or "default"
        chunks = [(c, domain) for c in self._split_into_chunks(text_input)]
        log_event("ingest: %d chunks domain=%r", len(chunks), domain)

        n_before = len(self.dense_model.labels)
        result = self.engine.ingest_stage(
            chunks,
            autosave   = self.autosave_enabled,
            saves_dir  = str(self.SAVES_DIR),
        )
        self.dense_model.add_chunks(self.engine.all_chunks[n_before:])

        log_event(
            "ingest done: stage=%d activated=%d reinforced=%d dormant=%d",
            result["stage_number"],
            len(result["slots_activated"]),
            len(result["slots_reinforced"]),
            result["dormant_remaining"],
        )

        autosave_note = " | autosaved ✓" if self.autosave_enabled else ""
        status = (
            f"Stage {result['stage_number']} complete — "
            f"{len(result['slots_activated'])} new slots activated, "
            f"{len(result['slots_reinforced'])} reinforced — "
            f"{result['dormant_remaining']:,} dormant remaining."
            + autosave_note
            + " | Click 'Refresh UMAP' to visualize."
        )
        return status, None, self.viz.plot_histogram(self.vs)

    @traced
    def refresh_umap(self):
        return self.viz.plot_umap(self.vs)

    @traced
    def view_diff(self):
        diff = self.engine.get_stage_diff()
        if not diff["new_slots"]:
            return self.viz.plot_umap(self.vs)
        return self.viz.plot_stage_diff(self.vs, diff["new_slots"])

    @traced
    def reset_all(self) -> Tuple:
        log_event("reset_all: clearing %d active slots", self.vs.n_active)
        self.vs.reset()
        self.engine.reset()
        self.dense_model   = DenseModel([], self._model)
        self.maintenance   = KnowledgeMaintenance(
            vector_space  = self.vs,
            model         = self._model,
            growth_engine = self.engine,
        )
        return (
            "Vector space reset — all slots dormant. | Click 'Refresh UMAP' to visualize.",
        ) + self._both_plots()

    # --------------------------------------------------------------------------
    # Tab 2 — Query
    # --------------------------------------------------------------------------

    @traced
    def query(self, text_input: str, top_k: int) -> Tuple[str, str]:
        if not text_input.strip():
            return "⚠️ Please enter a query.", ""

        log_event("query: %r top_k=%d", text_input.strip()[:80], top_k)
        result = self.router.route_query(text_input.strip(), top_k=int(top_k))

        ratio = f"Active: {result['active_count']:,} | Dormant: {result['dormant_count']:,}"
        if result.get("faiss_used"):
            ratio += " | FAISS ✓"

        if not result["matches"]:
            return "No active slots found — ingest some text first.", ratio

        if result["boundary_violation"]:
            # ── Reactive maintenance: auto-ingest a negative example ──────────
            correction = self.maintenance.on_boundary_violation(
                query_text     = text_input.strip(),
                nearest_domain = result["nearest_domain"],
            )
            log_event(
                "boundary_violation: query=%r domain=%r → negative slot %d",
                text_input.strip()[:60],
                result["nearest_domain"],
                correction["slot_result"]["slot_idx"],
            )
            correction_note = (
                f"\n\n_Maintenance: negative counterexample auto-ingested "
                f"(slot {correction['slot_result']['slot_idx']}) — "
                f"total corrections this session: {self.maintenance.correction_count()}_"
            )
            return (
                f"🚫 **BOUNDARY VIOLATION** — concept exists but combination is invalid\n"
                f"Nearest domain: `{result['nearest_domain']}`"
                + correction_note,
                ratio,
            )

        lines = []
        for m in result["matches"]:
            lines.append(
                f"**[{m['domain']}]** {m['label']} \n"
                f" similarity: `{m['similarity']:.4f}` | "
                f"activation: `{m['activation']:.4f}`"
            )
        return "\n\n---\n\n".join(lines), ratio

    # --------------------------------------------------------------------------
    # Tab 3 — Prune
    # --------------------------------------------------------------------------

    @traced
    def run_prune(self, threshold: float) -> Tuple:
        log_event("run_prune: threshold=%.2f active_before=%d", threshold, self.vs.n_active)

        with self.vs._lock:
            self._prune_before = self.vs.activation.detach().numpy().copy()
            result = self.vs.prune(threshold=float(threshold))
            after  = self.vs.activation.detach().numpy().copy()

        fig = self.viz.plot_prune_comparison(self._prune_before, after)

        log_event(
            "run_prune done: pruned=%d active_after=%d",
            result["pruned_count"], result["after_active"],
        )

        status = (
            f"Pruning complete — threshold: {threshold:.2f} | "
            f"Pruned: {result['pruned_count']:,} slots | "
            f"Active before → after: {result['before_active']:,} → {result['after_active']:,}"
        )
        return status, fig

    # --------------------------------------------------------------------------
    # Tab 4 — Compare
    # --------------------------------------------------------------------------

    def get_query_choices(self, query_type: str) -> List[str]:
        return _QUERY_MAP.get(query_type, [])

    @traced
    def run_comparison_tab(self, query_type: str, selected_query: str) -> Tuple:
        if not selected_query:
            return "<p>⚠️ Select a query first.</p>", None, None, "No query selected."

        if self.dense_model.embeddings.shape[0] == 0:
            msg = (
                "<p>⚠️ No data ingested yet. "
                "Go to <b>Tab 1 — Grow</b> and ingest some text first.</p>"
            )
            return msg, None, None, "No data in vector space."

        log_event("compare: type=%r query=%r", query_type, selected_query[:80])

        d_result = self.dense_model.query(selected_query)
        b_result = self.braingrow_model.query(selected_query)
        q_np     = encode_unit_numpy(self._model, selected_query)

        is_unknown     = query_type == "Unknown"
        d_hallucinated = is_unknown and d_result["confident"]
        b_verdict      = b_result["verdict"]
        b_is_honest    = b_verdict == "HONEST (uncertain)"
        b_is_violation = "BOUNDARY VIOLATION" in b_verdict

        d_row_bg = "rgba(255,60,60,0.22)"  if d_hallucinated  else "rgba(200,200,200,0.06)"
        b_row_bg = (
            "rgba(255,180,0,0.22)"  if b_is_violation
            else "rgba(60,200,100,0.22)" if b_is_honest
            else "rgba(200,200,200,0.06)"
        )

        d_verdict_html = (
            '<span style="color:#ff6b6b;font-weight:bold;">⚠ HALLUCINATED</span>'
            if d_hallucinated
            else '<span style="color:#aaa;">✓ Confident</span>'
        )
        b_verdict_html = (
            '<span style="color:#ffd43b;font-weight:bold;">⚠️ BOUNDARY VIOLATION</span>'
            if b_is_violation
            else '<span style="color:#69db7c;font-weight:bold;">✓ HONEST (uncertain)</span>'
            if b_is_honest
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

        dense_fig = self.viz.plot_dense_umap(
            self.dense_model.embeddings, self.dense_model.labels,
            self.dense_model.domains, q_np,
        )
        bg_fig = self.viz.plot_umap(self.vs, q_np)

        status = (
            f"Query: '{selected_query[:60]}'"
            f" | Dense sim: {d_result['similarity']:.4f}"
            f" | BrainGrow sim: {b_result['similarity']:.4f}"
        )
        return html, dense_fig, bg_fig, status

    # --------------------------------------------------------------------------
    # Tab 5 — Network (Save / Load)
    # --------------------------------------------------------------------------

    def list_saves(self) -> List[str]:
        return [str(f) for f in sorted(self.SAVES_DIR.glob("*.bgstate"), reverse=True)]

    @traced
    def save_network(self, description: str) -> str:
        if self.vs.n_active == 0:
            return "⚠️ Nothing to save — vector space is empty."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.SAVES_DIR / f"network_{timestamp}.bgstate"
        self.vs.save(str(path), description=description.strip())
        size_str = self._format_file_size(str(path))

        log_event("save_network: %s active=%d stage=%d size=%s",
                  path.name, self.vs.n_active, self.vs.stage_number, size_str)

        return (
            f"✅ Saved: {path.name} ({size_str}) | "
            f"{self.vs.n_active:,} active slots | Stage {self.vs.stage_number}"
        )

    @traced
    def load_network(self, selected_path: str) -> Tuple:
        if not selected_path:
            return ("⚠️ No save file selected.",) + self._both_plots()
        if not os.path.exists(selected_path):
            return (f"⚠️ File not found: {selected_path}",) + self._both_plots()

        new_vs, meta = VectorSpace.load(selected_path)

        # Copy loaded state into the existing vs instance so all downstream
        # object references (engine, router, maintenance) stay valid.
        self.vs.N             = new_vs.N
        self.vs.D             = new_vs.D
        self.vs.slots         = new_vs.slots
        self.vs.activation    = new_vs.activation
        self.vs.slot_labels   = new_vs.slot_labels
        self.vs.slot_domains  = new_vs.slot_domains
        self.vs.stage_number  = new_vs.stage_number
        self.vs._step         = 0
        self.vs.dormant_queue = new_vs.dormant_queue
        self.vs.negative_domains = new_vs.negative_domains
        # FAISS index will rebuild lazily on first query after load
        self.vs._faiss_index    = None
        self.vs._faiss_slot_map = []
        self.vs._faiss_dirty    = True

        log_event(
            "load_network: %s active=%d stage=%d",
            os.path.basename(selected_path), new_vs.n_active, new_vs.stage_number,
        )

        self.engine.stage_number    = new_vs.stage_number
        self.engine._stage_history  = []

        reconstructed_chunks = [
            (new_vs.slot_labels[idx], new_vs.slot_domains.get(idx, "unknown"))
            for idx in sorted(new_vs.slot_labels.keys())
        ]
        self.engine.all_chunks = reconstructed_chunks
        self.dense_model = DenseModel(reconstructed_chunks, self._model)

        # Reset maintenance log — corrections from previous session are not reloaded
        self.maintenance = KnowledgeMaintenance(
            vector_space  = self.vs,
            model         = self._model,
            growth_engine = self.engine,
        )

        domains  = sorted(set(self.vs.slot_domains.values()))
        desc     = meta.get("description") or "—"
        size_str = self._format_file_size(selected_path)

        status = (
            f"✅ Loaded: {os.path.basename(selected_path)} ({size_str})\n"
            f"Saved at: {meta.get('saved_at', '?')} | Description: {desc}\n"
            f"Active slots: {self.vs.n_active:,} | "
            f"Total slots: {meta['n_slots']:,} | "
            f"Stage: {self.vs.stage_number} | "
            f"Domains: {', '.join(domains) if domains else 'none'}"
        )
        return (status,) + self._both_plots()

    @traced
    def delete_save(self, selected_path: str) -> str:
        if not selected_path:
            return "⚠️ No file selected."
        if not os.path.exists(selected_path):
            return f"⚠️ File not found: {selected_path}"
        log_event("delete_save: %s", os.path.basename(selected_path))
        os.remove(selected_path)
        return f"🗑️ Deleted: {os.path.basename(selected_path)}"

    def get_network_info(self) -> str:
        active   = self.vs.n_active
        dormant  = self.vs.N - active
        domains  = sorted(set(self.vs.slot_domains.values()))
        utilised = f"{active / self.vs.N * 100:.1f}%"
        faiss    = "FAISS ✓" if self.vs.faiss_available else "brute-force"
        return (
            f"**Active slots:** {active:,} | **Dormant:** {dormant:,} | "
            f"**Utilisation:** {utilised} | "
            f"**Stage:** {self.vs.stage_number} | "
            f"**Retrieval:** {faiss} | "
            f"**Domains ({len(domains)}):** {', '.join(domains) if domains else 'none'} | "
            f"**Total capacity:** {self.vs.N:,}"
        )

    @traced
    def toggle_autosave(self, enabled: bool) -> str:
        log_event("toggle_autosave: %s", enabled)
        self.autosave_enabled = enabled
        return (
            "Autosave enabled ✅ — VectorSpace will be saved after each Ingest Stage."
            if enabled else "Autosave disabled."
        )

    # --------------------------------------------------------------------------
    # Knowledge Maintenance — audit (callable from UI or standalone)
    # --------------------------------------------------------------------------

    def run_audit(self) -> str:
        """
        Run a proactive hallucination risk audit across all registered domains.
        Returns a formatted text report suitable for display in any Gradio textbox.
        """
        if self.vs.n_active == 0:
            return "⚠️ No active slots — ingest some text first, then run audit."

        log_event("run_audit: active=%d domains=%d",
                  self.vs.n_active, len(set(self.vs.slot_domains.values())))

        report = self.maintenance.audit_hallucination_risk()

        corrections = self.maintenance.correction_count()
        footer = (
            f"\n\nReactive corrections this session: {corrections}"
            if corrections
            else "\n\nNo reactive corrections made this session."
        )

        return report.as_text() + footer

    # --------------------------------------------------------------------------
    # Tab 6 — TinyStories
    # --------------------------------------------------------------------------

    @traced
    def run_tinystories_stage(
        self,
        preset_name:   str,
        custom_sample: int,
        custom_chunks: int,
    ) -> Tuple:
        if not _check_datasets_available():
            msg = (
                "⚠️ The **datasets** package is required for TinyStories ingestion.\n"
                "Install it then restart the app:\n\n"
                "```\npip install datasets\n```"
            )
            return msg, None, None

        if preset_name in STAGE_PRESETS:
            p            = STAGE_PRESETS[preset_name]
            sample_size  = p["sample_size"]
            max_chunks   = p["max_chunks"]
        else:
            sample_size = int(custom_sample)
            max_chunks  = int(custom_chunks)

        log_event("tinystories: preset=%r sample=%d max_chunks=%d",
                  preset_name, sample_size, max_chunks)

        status_lines = [f"📥 Loading {sample_size:,} TinyStories snippets…"]

        try:
            chunks = prepare_experiment(
                sample_size  = sample_size,
                max_chunks   = max_chunks,
                domain_label = "stories",
            )
        except Exception as exc:
            return f"❌ Error loading TinyStories: {exc}", None, None

        status_lines.append(f"⏳ Ingesting {len(chunks):,} chunks (batched encoding)…")

        n_before = len(self.dense_model.labels)
        result = self.engine.ingest_stage_batched(
            chunks,
            batch_size = 512,
            autosave   = self.autosave_enabled,
            saves_dir  = str(self.SAVES_DIR),
        )
        self.dense_model.add_chunks(self.engine.all_chunks[n_before:])

        autosave_note = " | autosaved ✓" if self.autosave_enabled else ""
        status_lines.append(
            f"✅ Stage {result['stage_number']} complete — "
            f"{len(result['slots_activated']):,} new slots, "
            f"{len(result['slots_reinforced']):,} reinforced — "
            f"{result['dormant_remaining']:,} dormant remaining."
            + autosave_note
            + " | Click 'Refresh UMAP' to visualize."
        )

        return "\n".join(status_lines), None, self.viz.plot_histogram(self.vs)
