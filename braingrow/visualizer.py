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
# Max active slots shown — sample when network exceeds this for performance
_MAX_ACTIVE_SHOWN = 5_000
# Minimum points needed to attempt UMAP (needs n_neighbors + 1 at minimum)
_UMAP_MIN_POINTS = 20


def _add_query_star(fig: go.Figure, coords: np.ndarray, query_vector) -> None:
    """Overlay a yellow star marker at coords[-1] when query_vector is provided."""
    if query_vector is None:
        return
    q_xy = coords[-1:]
    fig.add_trace(go.Scatter(
        x=q_xy[:, 0],
        y=q_xy[:, 1],
        mode="markers",
        marker=dict(
            color="rgba(255,255,0,0.95)",
            size=18,
            symbol="star",
            line=dict(width=1.5, color="white"),
        ),
        name="Query",
        hoverinfo="name",
    ))


def _domain_colour_map(domains: list) -> dict:
    """Map each unique domain name to a Plotly qualitative colour."""
    unique = sorted(set(domains))
    palette = px.colors.qualitative.Plotly
    return {d: palette[i % len(palette)] for i, d in enumerate(unique)}


# Shared layout kwargs for all 2-D scatter plots
_SCATTER_LAYOUT = dict(
    template="plotly_dark",
    xaxis_title="dim-1",
    yaxis_title="dim-2",
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
    margin=dict(l=40, r=30, t=50, b=40),
)


def _reduce_2d(vectors: np.ndarray, labels: list | None = None) -> np.ndarray:
    """Project *vectors* [N, D] to 2D using UMAP when possible, else PCA."""
    n = len(vectors)

    # Cap at 5,000 points before passing to UMAP to keep projection fast
    if n > 5_000:
        idx = np.random.choice(n, 5_000, replace=False)
        vectors = vectors[idx]
        if labels is not None:
            labels[:] = [labels[i] for i in idx]
        n = 5_000

    if _UMAP_AVAILABLE and n >= _UMAP_MIN_POINTS:
        try:
            reducer = umap_module.UMAP(
                n_components=2,
                n_neighbors=min(15, n - 1),
                min_dist=0.1,
                metric='cosine',
                low_memory=True,
                n_jobs=-1,
            )
            return reducer.fit_transform(vectors)
        except Exception:
            pass
    pca = PCA(n_components=2)
    return pca.fit_transform(vectors)


class Visualizer:
    """Stateless renderer — all state lives in VectorSpace."""

    # ------------------------------------------------------------------
    def plot_umap(self, vs: VectorSpace, query_vector: np.ndarray | None = None) -> go.Figure:
        """
        2-D projection of the vector space.
        Active slots are coloured by domain; dormant slots are grey.
        If *query_vector* is supplied it is rendered as a yellow star so the
        caller can visually see where the query lands relative to active vs
        dormant space.
        """
        active_mask = vs.get_active_mask()
        active_indices = active_mask.nonzero(as_tuple=True)[0].tolist()
        dormant_indices = (~active_mask).nonzero(as_tuple=True)[0].tolist()

        # Sample active indices when the network is very large to keep UMAP fast
        _sampled = False
        if len(active_indices) > _MAX_ACTIVE_SHOWN:
            sample_idx = np.random.choice(
                len(active_indices), _MAX_ACTIVE_SHOWN, replace=False
            )
            active_indices = [active_indices[i] for i in sample_idx]
            _sampled = True

        shown_dormant = dormant_indices[:_MAX_DORMANT_SHOWN]
        all_indices = active_indices + shown_dormant

        fig = go.Figure()

        if not all_indices and query_vector is None:
            fig.update_layout(
                title="Vector Space — empty (no slots active yet)",
                template="plotly_dark",
                xaxis_title="dim-1",
                yaxis_title="dim-2",
            )
            return fig

        # Build the full set of vectors to project together so that the
        # query star is embedded in the same coordinate space.
        slot_vecs = (
            vs.slots[all_indices].detach().numpy()
            if all_indices
            else np.empty((0, vs.D), dtype=np.float32)
        )
        if query_vector is not None:
            q_arr = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            vectors = (
                np.concatenate([slot_vecs, q_arr], axis=0)
                if len(slot_vecs) > 0
                else q_arr
            )
        else:
            vectors = slot_vecs

        if len(vectors) < 2:
            fig.update_layout(
                title="Not enough data to project",
                template="plotly_dark",
            )
            return fig

        coords = _reduce_2d(vectors)

        n_active = len(active_indices)

        # --- dormant scatter (grey, small) ---
        if shown_dormant:
            dom_xy = coords[n_active : n_active + len(shown_dormant)]
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

            domain_color = _domain_colour_map(domains)

            for domain in domain_color:
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

        # --- query vector star marker ---
        _add_query_star(fig, coords, query_vector)

        fig.update_layout(
            title=(
                f"Vector Space — {len(active_mask.nonzero(as_tuple=True)[0]):,} active"
                + (f" (showing {_MAX_ACTIVE_SHOWN:,} sampled)" if _sampled else "")
                + f" / {len(dormant_indices):,} dormant"
            ),
            **_SCATTER_LAYOUT,
        )
        return fig

    # ------------------------------------------------------------------
    def plot_dense_umap(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        domains: List[str],
        query_vector: np.ndarray | None = None,
    ) -> go.Figure:
        """
        UMAP / PCA plot for the DenseModel.

        All slots are occupied (no dormant space). The query star will
        visually land inside the cluster of coloured points, showing that
        the dense model always forces an answer into occupied space.
        """
        fig = go.Figure()

        if len(embeddings) == 0:
            fig.update_layout(
                title="Dense Model — no data ingested yet",
                template="plotly_dark",
            )
            return fig

        # Include query vector in the joint projection so it shares the
        # same coordinate space as the training embeddings.
        if query_vector is not None:
            q_arr = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            vectors = np.concatenate([embeddings, q_arr], axis=0)
        else:
            vectors = embeddings

        if len(vectors) < 2:
            fig.update_layout(
                title="Not enough data to project",
                template="plotly_dark",
            )
            return fig

        coords = _reduce_2d(vectors)
        n_emb = len(embeddings)

        domain_color = _domain_colour_map(domains)

        for domain in domain_color:
            idx_in = [j for j, d in enumerate(domains) if d == domain]
            d_xy = coords[idx_in]
            d_labels = [labels[j] for j in idx_in]
            fig.add_trace(go.Scatter(
                x=d_xy[:, 0],
                y=d_xy[:, 1],
                mode="markers",
                marker=dict(
                    color=domain_color[domain],
                    size=7,
                    opacity=0.85,
                    line=dict(width=0.5, color="white"),
                ),
                name=f"{domain} ({len(idx_in)})",
                text=d_labels,
                hoverinfo="text",
            ))

        _add_query_star(fig, coords, query_vector)

        fig.update_layout(
            title=f"Dense Model — {n_emb:,} slots (fully occupied, no dormant space)",
            **_SCATTER_LAYOUT,
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
            **_SCATTER_LAYOUT,
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
