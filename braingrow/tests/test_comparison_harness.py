"""
test_comparison_harness.py — Unit tests for DenseModel and BrainGrowModel.

Covers: DenseModel (always confident, nearest-neighbour),
        BrainGrowModel (three-way verdict: HONEST / BOUNDARY VIOLATION / Confident),
        and the predefined query list exports.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from comparison_harness import (
    BrainGrowModel,
    DenseModel,
    known_queries,
    partial_queries,
    unknown_queries,
)
from growth_engine import GrowthEngine
from vector_space import VectorSpace

N_SLOTS = 50
DIMS = 8


# ===========================================================================
# Query list exports
# ===========================================================================

class TestQueryLists:
    def test_known_queries_non_empty(self):
        assert len(known_queries) > 0
        assert all(isinstance(q, str) for q in known_queries)

    def test_partial_queries_non_empty(self):
        assert len(partial_queries) > 0

    def test_unknown_queries_non_empty(self):
        assert len(unknown_queries) > 0


# ===========================================================================
# DenseModel
# ===========================================================================

class TestDenseModel:
    def test_empty_chunks_returns_zero_embeddings(self, mock_model):
        dm = DenseModel([], mock_model)
        assert dm.embeddings.shape == (0, 384)

    def test_empty_query_returns_confident_true(self, mock_model):
        dm = DenseModel([], mock_model)
        result = dm.query("anything")
        assert result["confident"] is True
        assert result["similarity"] == 0.0

    def test_encodes_chunks_on_init(self, mock_model):
        chunks = [("text one alpha", "science"), ("text two beta", "history")]
        dm = DenseModel(chunks, mock_model)
        assert dm.embeddings.shape[0] == 2
        assert len(dm.labels) == 2
        assert len(dm.domains) == 2

    def test_skips_empty_text_on_init(self, mock_model):
        chunks = [("valid text here", "d"), ("", "d"), ("   ", "d")]
        dm = DenseModel(chunks, mock_model)
        assert dm.embeddings.shape[0] == 1

    def test_always_returns_confident_true(self, mock_model):
        chunks = [("some training text here", "d")]
        dm = DenseModel(chunks, mock_model)
        result = dm.query("completely different query xyz")
        assert result["confident"] is True

    def test_query_returns_required_keys(self, mock_model):
        chunks = [("some training data text", "science")]
        dm = DenseModel(chunks, mock_model)
        result = dm.query("any query")
        for k in ("label", "domain", "similarity", "confident"):
            assert k in result

    def test_similarity_between_neg1_and_1(self, mock_model):
        chunks = [("text A for training", "d"), ("text B for training", "d")]
        dm = DenseModel(chunks, mock_model)
        result = dm.query("query C for testing")
        assert -1.0 <= result["similarity"] <= 1.0

    def test_embeddings_are_unit_vectors(self, mock_model):
        chunks = [("unit vector test input", "d")]
        dm = DenseModel(chunks, mock_model)
        norms = np.linalg.norm(dm.embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)

    def test_domain_stored_correctly(self, mock_model):
        chunks = [("content for science domain", "science")]
        dm = DenseModel(chunks, mock_model)
        result = dm.query("content for science domain")
        assert result["domain"] == "science"

    def test_label_truncated_to_60_chars(self, mock_model):
        long_text = "Q" * 100
        dm = DenseModel([(long_text, "d")], mock_model)
        assert len(dm.labels[0]) <= 60


# ===========================================================================
# BrainGrowModel
# ===========================================================================

class TestBrainGrowModel:
    def _empty_model(self, mock_model):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        return BrainGrowModel(vs, mock_model), vs

    def _populated_model(self, mock_model, extra_chunks=None):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        engine = GrowthEngine(vs, mock_model)
        chunks = [
            ("DNA replication is a biological process", "science"),
            ("The Roman Empire fell in 476 AD", "history"),
        ]
        if extra_chunks:
            chunks.extend(extra_chunks)
        engine.ingest_stage(chunks)
        return BrainGrowModel(vs, mock_model), vs

    # --- empty space ---

    def test_empty_space_returns_honest(self, mock_model):
        bg, _ = self._empty_model(mock_model)
        result = bg.query("anything at all")
        assert result["verdict"] == "HONEST (uncertain)"
        assert result["confident"] is False

    def test_empty_space_similarity_zero(self, mock_model):
        bg, _ = self._empty_model(mock_model)
        result = bg.query("anything at all")
        assert result["similarity"] == 0.0

    # --- verdict: HONEST (low similarity) ---

    def test_honest_verdict_when_similarity_below_threshold(self, mock_model):
        """
        Ingest a single concept.  Query with a fabricated string that the mock
        model encodes to a different (random) vector, so similarity is low.
        BrainGrowModel.THRESHOLD = 0.25 — we force the issue by directly
        manipulating the activation after setting up a near-zero sim scenario.
        """
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        # Assign a known unit vector directly
        known = torch.zeros(DIMS)
        known[0] = 1.0  # points strictly along dimension 0
        r = vs.assign_slot(known, label="dim0-concept", domain="science")

        # Now set activation and use a query that points along dimension 1
        # → cosine sim = 0 which is < 0.25 threshold
        class _FixedModel:
            def encode(self, text, **kw):
                import numpy as np
                v = np.zeros(DIMS, dtype=np.float32)
                v[1] = 1.0  # orthogonal to the stored concept
                return v

        bg = BrainGrowModel(vs, _FixedModel())
        result = bg.query("orthogonal query")
        assert result["verdict"] == "HONEST (uncertain)"
        assert result["confident"] is False

    # --- verdict: BOUNDARY VIOLATION ---

    def test_boundary_violation_for_negative_domain(self, mock_model):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        engine = GrowthEngine(vs, mock_model)
        # Use a text that will be its own top match
        text = "Mixing bleach with ammonia creates toxic gas"
        engine.ingest_stage([(text, "cooking-negative")])

        bg = BrainGrowModel(vs, mock_model)
        result = bg.query(text)  # same text → high similarity → hits negative domain
        assert "BOUNDARY VIOLATION" in result["verdict"]
        assert result["confident"] is True

    # --- verdict: Confident ---

    def test_confident_verdict_for_known_concept(self, mock_model):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        # Point a slot along dim 0 and query with the same direction
        known = torch.zeros(DIMS)
        known[0] = 1.0
        vs.assign_slot(known, label="concept", domain="science")

        class _SameModel:
            def encode(self, text, **kw):
                import numpy as np
                v = np.zeros(DIMS, dtype=np.float32)
                v[0] = 1.0  # identical direction → sim = 1.0
                return v

        bg = BrainGrowModel(vs, _SameModel())
        result = bg.query("same direction concept")
        assert result["verdict"] == "✓ Confident"
        assert result["confident"] is True

    # --- result structure ---

    def test_result_has_required_keys(self, mock_model):
        bg, _ = self._populated_model(mock_model)
        result = bg.query("biological process")
        for k in ("label", "domain", "similarity", "confident", "verdict"):
            assert k in result

    def test_similarity_rounded_to_4dp(self, mock_model):
        bg, _ = self._populated_model(mock_model)
        result = bg.query("roman empire history")
        # rounded(x, 4) == round(x, 4) is True when already at 4dp
        assert result["similarity"] == round(result["similarity"], 4)
