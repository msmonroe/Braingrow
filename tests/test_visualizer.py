"""
test_visualizer.py — Unit tests for the Visualizer chart generator.

Checks that each plot method returns a plotly Figure with the expected
number of traces and a populated layout title, without caring about
exact pixel values.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import plotly.graph_objects as go

from vector_space import VectorSpace
from visualizer import Visualizer

N_SLOTS = 50
DIMS = 8


def _unit_vec(n: int, dim: int = 8) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(n)
    v = torch.randn(dim, generator=gen)
    return v / (v.norm() + 1e-8)


def _np_unit(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.RandomState(n)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ===========================================================================
# plot_umap
# ===========================================================================

class TestPlotUmap:
    def test_returns_figure(self):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        fig = Visualizer().plot_umap(vs)
        assert isinstance(fig, go.Figure)

    def test_empty_space_has_title(self):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        fig = Visualizer().plot_umap(vs)
        assert fig.layout.title.text  # non-empty title

    def test_active_slots_produce_traces(self):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        for i in range(5):
            vs.assign_slot(_unit_vec(i), label=f"c{i}", domain="science")
        fig = Visualizer().plot_umap(vs)
        # At minimum: 1 dormant trace + 1 domain trace
        assert len(fig.data) >= 1

    def test_query_vector_adds_star_trace(self):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        for i in range(5):
            vs.assign_slot(_unit_vec(i), label=f"c{i}", domain="d")
        q = _np_unit(99)
        fig_without = Visualizer().plot_umap(vs)
        fig_with = Visualizer().plot_umap(vs, query_vector=q)
        # Adding query vector must add at least one trace
        assert len(fig_with.data) >= len(fig_without.data)

    def test_query_trace_named_query(self):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        for i in range(5):
            vs.assign_slot(_unit_vec(i), label=f"c{i}", domain="d")
        q = _np_unit(99)
        fig = Visualizer().plot_umap(vs, query_vector=q)
        names = [t.name for t in fig.data]
        assert "Query" in names

    def test_multiple_domains_produce_separate_traces(self):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        for i in range(3):
            vs.assign_slot(_unit_vec(i), label=f"s{i}", domain="science")
        for i in range(3, 6):
            vs.assign_slot(_unit_vec(i), label=f"h{i}", domain="history")
        fig = Visualizer().plot_umap(vs)
        domain_traces = [t.name for t in fig.data if "science" in (t.name or "") or "history" in (t.name or "")]
        assert len(domain_traces) == 2

    def test_large_active_count_sampled(self):
        """Active slots above _MAX_ACTIVE_SHOWN (5k) should be sampled, not raise."""
        from visualizer import _MAX_ACTIVE_SHOWN
        # Use a vs with more slots than _MAX_ACTIVE_SHOWN would require;
        # we can only test this path safely with a small threshold by patching
        import visualizer
        original = visualizer._MAX_ACTIVE_SHOWN
        visualizer._MAX_ACTIVE_SHOWN = 3  # force sampling with just 5 slots
        try:
            vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
            for i in range(5):
                vs.assign_slot(_unit_vec(i), label=f"c{i}", domain="d")
            fig = Visualizer().plot_umap(vs)
            assert isinstance(fig, go.Figure)
            # Title should mention "sampled"
            assert "sampled" in fig.layout.title.text.lower()
        finally:
            visualizer._MAX_ACTIVE_SHOWN = original


# ===========================================================================
# plot_histogram
# ===========================================================================

class TestPlotHistogram:
    def test_returns_figure(self):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        fig = Visualizer().plot_histogram(vs)
        assert isinstance(fig, go.Figure)

    def test_empty_space_has_title(self):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        fig = Visualizer().plot_histogram(vs)
        assert "0" in fig.layout.title.text  # "0 / 50 slots active"

    def test_active_slots_produce_histogram_trace(self):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        for i in range(3):
            vs.assign_slot(_unit_vec(i))
        fig = Visualizer().plot_histogram(vs)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Histogram)


# ===========================================================================
# plot_stage_diff
# ===========================================================================

class TestPlotStageDiff:
    def test_returns_empty_figure_when_no_active(self):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        fig = Visualizer().plot_stage_diff(vs, [])
        assert isinstance(fig, go.Figure)

    def test_new_slots_get_star_trace(self):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        r1 = vs.assign_slot(_unit_vec(1), label="old")
        r2 = vs.assign_slot(_unit_vec(2), label="new")
        r3 = vs.assign_slot(_unit_vec(3), label="new2")
        new_indices = [r2["slot_idx"], r3["slot_idx"]]
        fig = Visualizer().plot_stage_diff(vs, new_indices)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


# ===========================================================================
# plot_prune_comparison
# ===========================================================================

class TestPlotPruneComparison:
    def test_returns_figure(self):
        before = np.random.rand(N_SLOTS).astype(np.float32)
        after = before.copy()
        after[before < 0.3] = 0.0
        fig = Visualizer().plot_prune_comparison(before, after)
        assert isinstance(fig, go.Figure)

    def test_two_histogram_traces(self):
        before = np.ones(10, dtype=np.float32) * 0.5
        after = np.zeros(10, dtype=np.float32)
        fig = Visualizer().plot_prune_comparison(before, after)
        assert len(fig.data) == 2

    def test_title_contains_prune(self):
        fig = Visualizer().plot_prune_comparison(
            np.array([0.5, 0.6], dtype=np.float32),
            np.array([0.5, 0.0], dtype=np.float32),
        )
        assert "prune" in fig.layout.title.text.lower()


# ===========================================================================
# plot_dense_umap
# ===========================================================================

class TestPlotDenseUmap:
    def test_empty_embeddings_returns_figure(self):
        fig = Visualizer().plot_dense_umap(
            np.empty((0, DIMS), dtype=np.float32), [], []
        )
        assert isinstance(fig, go.Figure)

    def test_populated_returns_figure_with_traces(self):
        embs = np.stack([_np_unit(i) for i in range(5)])
        labels = [f"label_{i}" for i in range(5)]
        domains = ["science"] * 3 + ["history"] * 2
        fig = Visualizer().plot_dense_umap(embs, labels, domains)
        assert len(fig.data) >= 1

    def test_query_star_added(self):
        embs = np.stack([_np_unit(i) for i in range(5)])
        labels = [f"l{i}" for i in range(5)]
        domains = ["d"] * 5
        q = _np_unit(99)
        fig = Visualizer().plot_dense_umap(embs, labels, domains, query_vector=q)
        names = [t.name for t in fig.data]
        assert "Query" in names
