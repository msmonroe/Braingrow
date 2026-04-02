"""
test_query_router.py — Unit tests for QueryRouter.

Covers: routing through empty/populated space, top-K results,
        reinforcement on hit, boundary violation detection.
"""

from __future__ import annotations

import pytest
import torch

from growth_engine import GrowthEngine
from query_router import QueryRouter
from vector_space import VectorSpace

N_SLOTS = 50
DIMS = 8


def _build_router_with_data(vs, mock_model, chunks):
    """Helper: ingest *chunks* then return a QueryRouter over the same vs."""
    engine = GrowthEngine(vs, mock_model)
    engine.ingest_stage(chunks)
    return QueryRouter(vs, mock_model)


# ===========================================================================
# Empty vector space
# ===========================================================================

class TestEmptySpace:
    def test_returns_empty_matches(self, vs, mock_model):
        router = QueryRouter(vs, mock_model)
        result = router.route_query("anything")
        assert result["matches"] == []

    def test_active_count_zero(self, vs, mock_model):
        router = QueryRouter(vs, mock_model)
        result = router.route_query("anything")
        assert result["active_count"] == 0
        assert result["dormant_count"] == N_SLOTS

    def test_boundary_violation_false_when_empty(self, vs, mock_model):
        """Empty space short-circuits; result dict has consistent keys with safe defaults."""
        router = QueryRouter(vs, mock_model)
        result = router.route_query("anything")
        assert result["boundary_violation"] is False
        assert result["nearest_domain"] == ""


# ===========================================================================
# Routing with data
# ===========================================================================

class TestRouteWithData:
    def test_returns_top_k_matches(self, populated_vs, mock_model):
        router = QueryRouter(populated_vs, mock_model)
        result = router.route_query("DNA replication is a biological process", top_k=3)
        assert 1 <= len(result["matches"]) <= 3

    def test_matches_have_required_keys(self, populated_vs, mock_model):
        router = QueryRouter(populated_vs, mock_model)
        result = router.route_query("Roman Empire history", top_k=1)
        assert result["matches"]
        m = result["matches"][0]
        for key in ("slot_idx", "label", "domain", "similarity", "activation"):
            assert key in m, f"Missing key: {key}"

    def test_similarities_between_neg1_and_1(self, populated_vs, mock_model):
        router = QueryRouter(populated_vs, mock_model)
        result = router.route_query("photosynthesis light energy", top_k=5)
        for m in result["matches"]:
            assert -1.0 <= m["similarity"] <= 1.0

    def test_results_sorted_by_similarity_descending(self, populated_vs, mock_model):
        router = QueryRouter(populated_vs, mock_model)
        result = router.route_query("fermentation cooking bread", top_k=5)
        sims = [m["similarity"] for m in result["matches"]]
        assert sims == sorted(sims, reverse=True)

    def test_top_k_respected(self, populated_vs, mock_model):
        router = QueryRouter(populated_vs, mock_model)
        result = router.route_query("biology science", top_k=2)
        assert len(result["matches"]) <= 2

    def test_active_and_dormant_counts_sum_to_n(self, populated_vs, mock_model):
        router = QueryRouter(populated_vs, mock_model)
        result = router.route_query("test query", top_k=1)
        assert result["active_count"] + result["dormant_count"] == N_SLOTS

    def test_returns_query_embedding_tensor(self, populated_vs, mock_model):
        router = QueryRouter(populated_vs, mock_model)
        result = router.route_query("test", top_k=1)
        assert isinstance(result["query_embedding"], torch.Tensor)
        assert result["query_embedding"].shape == (DIMS,)

    def test_reinforcement_increases_activation(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        text = "The mitochondria is the powerhouse of the cell"
        engine.ingest_stage([(text, "science")])

        # Record activation before routing
        active_idx = list(vs.slot_domains.keys())[0]
        before = float(vs.activation[active_idx].item())

        router = QueryRouter(vs, mock_model)
        router.route_query(text, top_k=1)

        after = float(vs.activation[active_idx].item())
        assert after > before


# ===========================================================================
# Boundary violation
# ===========================================================================

class TestBoundaryViolation:
    def _setup_with_negative_domain(self, mock_model):
        """Create a VectorSpace with one slot in a 'cooking-negative' domain."""
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage([
            ("Mixing bleach and ammonia is dangerous", "cooking-negative"),
        ])
        return vs

    def test_boundary_violation_true_for_negative_domain(self, mock_model):
        vs = self._setup_with_negative_domain(mock_model)
        router = QueryRouter(vs, mock_model)
        result = router.route_query("Mixing bleach and ammonia is dangerous", top_k=1)
        assert result["boundary_violation"] is True

    def test_nearest_domain_populated(self, mock_model):
        vs = self._setup_with_negative_domain(mock_model)
        router = QueryRouter(vs, mock_model)
        result = router.route_query("Mixing bleach and ammonia is dangerous", top_k=1)
        assert result["nearest_domain"] == "cooking-negative"

    def test_no_boundary_violation_for_normal_domain(self, populated_vs, mock_model):
        router = QueryRouter(populated_vs, mock_model)
        result = router.route_query("DNA biology science", top_k=1)
        assert result["boundary_violation"] is False

    def test_nearest_domain_empty_when_no_matches(self, vs, mock_model):
        """Empty space: nearest_domain should be absent or empty."""
        router = QueryRouter(vs, mock_model)
        result = router.route_query("anything", top_k=1)
        # Early return path — no nearest_domain key
        assert result.get("nearest_domain", "") == ""
