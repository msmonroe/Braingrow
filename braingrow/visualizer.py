"""
visualizer.py — Interactive Plotly chart generation for BrainGrow.

Four chart types:
  plot_umap()            — 2D projection of all slots (active by domain, dormant grey)
  plot_histogram()       — activation score distribution
  plot_stage_diff()      — highlights slots newly activated vs prior stages
  plot_prune_comparison()— before/after activation density overlaid
"""

from __future__ import annotations
from typing import List

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from vector_space import VectorSpace

# UMAP is optional; fall back to PCA if not installed
try:
    import umap as umap_module
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

from sklearn.decomposition import PCA

# Max dormant slots shown in the UMAP scatter to keep rendering fast
_MAX_DORMANT_SHOWN = 300
# Minimum points needed to attempt UMAP (needs n_neighbors + 1 at minimum)
_UMAP_MIN_POINTS = 20


def _reduce_2d(vectors: np.ndarray) -> np.ndarray:
    """Project *vectors* [N, D] to 2D using UMAP when possible, else PCA."""
    n = len(vectors)
    if _UMAP_AVAILABLE and n >= _UMAP_MIN_POINTS:
        try:
            reducer = umap_module.UMAP(
                n_components=2,
                n_neighbors=min(15, n - 1),
                min_dist=0.1,
                n_epochs=200,  # faster than default 500
            )
            return reducer.fit_transform(vectors)
        except Exception:
            pass
    pca = PCA(n_components=2)
    return pca.fit_transform(vectors)


class Visualizer:
    """Stateless renderer — all state lives in VectorSpace."""

    # ------------------------------------------------------------------
    def plot_umap(self, vs: VectorSpace) -> go.Figure:
        """
        2-D projection of the vector space.
        Active slots are coloured by domain; dormant slots are grey.
        """
        active_mask = vs.get_active_mask()
        active_indices = active_mask.nonzero(as_tuple=True)[0].tolist()
        dormant_indices = (~active_mask).nonzero(as_tuple=True)[0].tolist()

        shown_dormant = dormant_indices[:_MAX_DORMANT_SHOWN]
        all_indices = active_indices + shown_dormant

        fig = go.Figure()

        if not all_indices:
            fig.update_layout(
                title="Vector Space — empty (no slots active yet)",
                template="plotly_dark",
                xaxis_title="dim-1",
                yaxis_title="dim-2",
            )
            return fig

        vectors = vs.slots[all_indices].detach().numpy()
        coords = _reduce_2d(vectors)

        n_active = len(active_indices)

        # --- dormant scatter (grey, small) ---
        if shown_dormant:
            dom_xy = coords[n_active:]
            fig.add_trace(go.Scatter(
                x=dom_xy[:, 0],
                y=dom_xy[:, 1],
                mode="markers",
                marker=dict(color="rgba(140,140,140,0.18)", size=3),
                name=f"Dormant ({len(dormant_indices):,})",
                hoverinfo="name",
            ))

        # --- active scatter coloured by domain ---
        if active_indices:
            act_xy = coords[:n_active]
            domains = [vs.slot_domains.get(i, "unknown") for i in active_indices]
            labels = [vs.slot_labels.get(i, f"slot_{i}") for i in active_indices]
            activations = [float(vs.activation[i].item()) for i in active_indices]

            unique_domains = sorted(set(domains))
            palette = px.colors.qualitative.Plotly
            domain_color = {d: palette[i % len(palette)] for i, d in enumerate(unique_domains)}

            for domain in unique_domains:
                idx_in_active = [j for j, d in enumerate(domains) if d == domain]
                d_xy = act_xy[idx_in_active]
                d_labels = [labels[j] for j in idx_in_active]
                d_act = [activations[j] for j in idx_in_active]
                marker_sizes = [6 + a * 10 for a in d_act]

                fig.add_trace(go.Scatter(
                    x=d_xy[:, 0],
                    y=d_xy[:, 1],
                    mode="markers",
                    marker=dict(
                        color=domain_color[domain],
                        size=marker_sizes,
                        opacity=0.85,
                        line=dict(width=0.5, color="white"),
                    ),
                    name=f"{domain} ({len(idx_in_active)})",
                    text=[
                        f"<b>{lbl}</b><br>activation: {a:.3f}"
                        for lbl, a in zip(d_labels, d_act)
                    ],
                    hoverinfo="text",
                ))

        fig.update_layout(
            title=(
                f"Vector Space — {len(active_indices):,} active "
                f"/ {len(dormant_indices):,} dormant"
            ),
            template="plotly_dark",
            xaxis_title="dim-1",
            yaxis_title="dim-2",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=40, r=30, t=50, b=40),
        )
        return fig

    # ------------------------------------------------------------------
    def plot_histogram(self, vs: VectorSpace) -> go.Figure:
        """Activation distribution across all active slots."""
        activations = vs.activation.detach().numpy()
        active_vals = activations[activations > 0]
        n_active = len(active_vals)
        n_total = len(activations)

        fig = go.Figure()
        if n_active > 0:
            fig.add_trace(go.Histogram(
                x=active_vals,
                nbinsx=40,
                marker_color="rgba(80, 200, 140, 0.8)",
                name="Active slots",
            ))

        fig.update_layout(
            title=f"Activation Distribution — {n_active:,} / {n_total:,} slots active",
            xaxis_title="Activation Score",
            yaxis_title="Slot Count",
            template="plotly_dark",
            margin=dict(l=40, r=30, t=50, b=40),
        )
        return fig

    # ------------------------------------------------------------------
    def plot_stage_diff(
        self,
        vs: VectorSpace,
        new_slot_indices: List[int],
    ) -> go.Figure:
        """Highlight slots activated in the latest stage vs prior stages."""
        active_mask = vs.get_active_mask()
        active_indices = active_mask.nonzero(as_tuple=True)[0].tolist()

        if not active_indices:
            return go.Figure()

        new_set = set(new_slot_indices)
        vectors = vs.slots[active_indices].detach().numpy()
        coords = _reduce_2d(vectors)

        old_local = [j for j, idx in enumerate(active_indices) if idx not in new_set]
        new_local = [j for j, idx in enumerate(active_indices) if idx in new_set]

        fig = go.Figure()
        if old_local:
            old_xy = coords[old_local]
            fig.add_trace(go.Scatter(
                x=old_xy[:, 0],
                y=old_xy[:, 1],
                mode="markers",
                marker=dict(color="rgba(90, 130, 210, 0.65)", size=7),
                name=f"Previous stages ({len(old_local)})",
            ))
        if new_local:
            new_xy = coords[new_local]
            fig.add_trace(go.Scatter(
                x=new_xy[:, 0],
                y=new_xy[:, 1],
                mode="markers",
                marker=dict(
                    color="rgba(255, 185, 0, 0.95)",
                    size=12,
                    symbol="star",
                    line=dict(width=1, color="white"),
                ),
                name=f"New this stage ({len(new_local)})",
            ))

        fig.update_layout(
            title="Stage Diff — New Activations vs Prior Stages",
            template="plotly_dark",
            xaxis_title="dim-1",
            yaxis_title="dim-2",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=40, r=30, t=50, b=40),
        )
        return fig

    # ------------------------------------------------------------------
    def plot_prune_comparison(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> go.Figure:
        """Overlaid before/after activation histograms for the prune tab."""
        before_active = before[before > 0]
        after_active = after[after > 0]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=before_active,
            nbinsx=40,
            marker_color="rgba(210, 80, 80, 0.65)",
            name=f"Before pruning ({len(before_active):,} active)",
            opacity=0.75,
        ))
        fig.add_trace(go.Histogram(
            x=after_active,
            nbinsx=40,
            marker_color="rgba(80, 200, 130, 0.75)",
            name=f"After pruning  ({len(after_active):,} active)",
            opacity=0.75,
        ))
        fig.update_layout(
            barmode="overlay",
            title="Prune Comparison — Before vs After",
            xaxis_title="Activation Score",
            yaxis_title="Slot Count",
            template="plotly_dark",
            legend=dict(x=0.55, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=40, r=30, t=50, b=40),
        )
        return fig
