"""
test_growth_engine.py — Unit tests for GrowthEngine.

Covers: ingest_stage, ingest_stage_batched, autosave integration,
        stage history, reset.
"""

from __future__ import annotations

import os

import pytest
import torch

from growth_engine import GrowthEngine
from vector_space import VectorSpace

N_SLOTS = 50
DIMS = 8


# ===========================================================================
# ingest_stage
# ===========================================================================

class TestIngestStage:
    def test_activates_new_slots(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        chunks = [("DNA replication is a process", "science")]
        result = engine.ingest_stage(chunks)
        assert len(result["slots_activated"]) == 1
        assert vs.n_active == 1

    def test_increments_stage_number(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage([("text one", "d")])
        engine.ingest_stage([("text two", "d")])
        assert engine.stage_number == 2

    def test_syncs_vs_stage_number(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage([("text", "d")])
        assert vs.stage_number == engine.stage_number

    def test_skips_empty_text(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        result = engine.ingest_stage([("", "d"), ("   ", "d")])
        assert len(result["slots_activated"]) == 0
        assert vs.n_active == 0

    def test_tracks_all_chunks(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        chunks = [("text A", "science"), ("text B", "history")]
        engine.ingest_stage(chunks)
        assert len(engine.all_chunks) == 2
        assert engine.all_chunks[0][0] == "text A"
        assert engine.all_chunks[1][1] == "history"

    def test_accumulates_chunks_across_stages(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage([("chunk 1", "d")])
        engine.ingest_stage([("chunk 2", "d"), ("chunk 3", "d")])
        assert len(engine.all_chunks) == 3

    def test_reinforces_near_duplicate(self, vs, mock_model):
        """The same text ingested twice should reinforce rather than open a new slot."""
        engine = GrowthEngine(vs, mock_model)
        text = "A unique reproducible sentence for reinforcement"
        engine.ingest_stage([(text, "science")])
        assert vs.n_active == 1

        result = engine.ingest_stage([(text, "science")])
        # Should reinforce the existing slot, not open a new one
        assert len(result["slots_reinforced"]) == 1
        assert vs.n_active == 1  # still only 1 active slot

    def test_returns_dormant_remaining(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        result = engine.ingest_stage([("text", "d")])
        assert result["dormant_remaining"] == N_SLOTS - 1

    def test_domain_label_stored_in_vs(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage([("unique science concept xyzq", "science")])
        domains = list(vs.slot_domains.values())
        assert "science" in domains

    def test_label_truncated_to_60_chars(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        long_text = "A" * 100
        engine.ingest_stage([(long_text, "d")])
        for label in vs.slot_labels.values():
            assert len(label) <= 60

    def test_autosave_called_when_enabled(self, vs, mock_model, tmp_path):
        engine = GrowthEngine(vs, mock_model)
        saves_dir = str(tmp_path / "saves")
        engine.ingest_stage([("autosave test content", "d")],
                            autosave=True, saves_dir=saves_dir)
        bgstate_files = list((tmp_path / "saves").glob("*.bgstate"))
        assert len(bgstate_files) == 1

    def test_no_autosave_when_disabled(self, vs, mock_model, tmp_path):
        engine = GrowthEngine(vs, mock_model)
        saves_dir = str(tmp_path / "saves")
        os.makedirs(saves_dir)
        engine.ingest_stage([("no autosave test", "d")],
                            autosave=False, saves_dir=saves_dir)
        assert list((tmp_path / "saves").glob("*.bgstate")) == []

    def test_multiple_domains_stored_separately(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage([
            ("Quantum mechanics describes subatomic particles", "science"),
            ("Napoleon was defeated at Waterloo 1815", "history"),
        ])
        domains_stored = set(vs.slot_domains.values())
        assert "science" in domains_stored
        assert "history" in domains_stored


# ===========================================================================
# ingest_stage_batched
# ===========================================================================

class TestIngestStageBatched:
    def test_activates_slots(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        chunks = [
            ("Batch text alpha prime", "d"),
            ("Batch text beta prime", "d"),
            ("Batch text gamma prime", "d"),
        ]
        result = engine.ingest_stage_batched(chunks, batch_size=2)
        assert len(result["slots_activated"]) + len(result["slots_reinforced"]) == 3

    def test_increments_stage_number(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage_batched([("text A", "d"), ("text B", "d")])
        assert engine.stage_number == 1
        assert vs.stage_number == 1

    def test_tracks_all_chunks(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        chunks = [("batched chunk X", "d"), ("batched chunk Y", "d")]
        engine.ingest_stage_batched(chunks)
        assert len(engine.all_chunks) == 2

    def test_empty_chunks_returns_safely(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        result = engine.ingest_stage_batched([])
        assert result["slots_activated"] == []
        assert result["stage_number"] == 0  # stage not incremented for empty input

    def test_autosave_creates_file(self, vs, mock_model, tmp_path):
        engine = GrowthEngine(vs, mock_model)
        saves_dir = str(tmp_path / "saves")
        engine.ingest_stage_batched(
            [("batched autosave content here", "d")],
            autosave=True, saves_dir=saves_dir
        )
        bgstate_files = list((tmp_path / "saves").glob("*.bgstate"))
        assert len(bgstate_files) == 1


# ===========================================================================
# get_stage_diff / history
# ===========================================================================

class TestStageHistory:
    def test_get_stage_diff_empty_before_any_ingest(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        diff = engine.get_stage_diff()
        assert diff["stage"] == 0
        assert diff["new_slots"] == []

    def test_get_stage_diff_returns_latest_stage_slots(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage([("stage one text zq", "d")])
        diff = engine.get_stage_diff()
        assert diff["stage"] == 1
        assert len(diff["new_slots"]) >= 1

    def test_get_all_stage_history_accumulates(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage([("stage one data xr", "d")])
        engine.ingest_stage([("stage two data yr", "d")])
        history = engine.get_all_stage_history()
        assert len(history) == 2
        assert history[0]["stage"] == 1
        assert history[1]["stage"] == 2


# ===========================================================================
# reset
# ===========================================================================

class TestReset:
    def test_reset_clears_stage_number(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage([("text", "d")])
        engine.reset()
        assert engine.stage_number == 0

    def test_reset_clears_history(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage([("text", "d")])
        engine.reset()
        assert engine.get_all_stage_history() == []

    def test_reset_clears_all_chunks(self, vs, mock_model):
        engine = GrowthEngine(vs, mock_model)
        engine.ingest_stage([("text", "d")])
        engine.reset()
        assert engine.all_chunks == []
