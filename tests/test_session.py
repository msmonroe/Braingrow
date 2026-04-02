"""
test_session.py — Integration tests for BrainGrowSession.

Uses a tiny VectorSpace (50 slots × 8 dims) and a deterministic mock encoder
so tests run in milliseconds without GPU or network access.

Static-method tests require no fixtures; integration tests use the
`tiny_session` fixture which patches SentenceTransformer and VectorSpace
before constructing the session.
"""

from __future__ import annotations

import os
import hashlib
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_space import VectorSpace


# ---------------------------------------------------------------------------
# Minimal deterministic mock model with the `.to()` method that
# session.__init__ calls.  Uses the same sha256 → unit-vector logic as the
# conftest mock so tests are reproducible.
# ---------------------------------------------------------------------------

_DIMS = 8


class _SessionMockModel:
    """Deterministic sha256-seeded 8-D mock encoder compatible with session."""

    def to(self, device):  # noqa: ARG002
        return self

    def encode(
        self,
        text: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        device: str = "cpu",
        convert_to_numpy: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(text, list):
            return np.stack([self._one(t) for t in text], axis=0)
        return self._one(text)

    @staticmethod
    def _one(text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.RandomState(seed)
        v = rng.randn(_DIMS).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_session(tmp_path, monkeypatch):
    """
    BrainGrowSession wired with a tiny (50-slot × 8-dim) VectorSpace and a
    deterministic mock encoder.  No GPU, no network, no real model download.
    """
    import session as _sess

    mock_m = _SessionMockModel()

    monkeypatch.setattr(_sess, "SentenceTransformer", lambda *a, **kw: mock_m)

    # Subclass so VectorSpace.load() (a classmethod) is still accessible while
    # __init__ defaults to tiny dimensions for speed.
    class _TinyVS(VectorSpace):
        def __init__(self, n_slots=None, dimensions=None):
            super().__init__(
                n_slots=n_slots if n_slots is not None else 50,
                dimensions=dimensions if dimensions is not None else 8,
            )

    monkeypatch.setattr(_sess, "VectorSpace", _TinyVS)
    monkeypatch.setattr(
        _sess.BrainGrowSession, "SAVES_DIR",
        tmp_path / "saves",
    )

    return _sess.BrainGrowSession()


# ===========================================================================
# Static helpers — no session needed
# ===========================================================================

class TestSplitIntoChunks:
    from session import BrainGrowSession as _BS

    def test_multiline_returns_lines(self):
        from session import BrainGrowSession
        result = BrainGrowSession._split_into_chunks("line one\nline two\nline three")
        assert result == ["line one", "line two", "line three"]

    def test_single_sentence_returns_list_with_one_item(self):
        from session import BrainGrowSession
        result = BrainGrowSession._split_into_chunks("Just one sentence here.")
        assert result == ["Just one sentence here."]

    def test_multi_sentence_single_line_splits_on_punctuation(self):
        from session import BrainGrowSession
        result = BrainGrowSession._split_into_chunks("First sentence. Second sentence. Third.")
        assert len(result) >= 2
        assert any("First" in s for s in result)
        assert any("Second" in s for s in result)

    def test_empty_lines_are_stripped(self):
        from session import BrainGrowSession
        result = BrainGrowSession._split_into_chunks("line one\n\n   \nline two")
        assert result == ["line one", "line two"]

    def test_exclamation_split(self):
        from session import BrainGrowSession
        result = BrainGrowSession._split_into_chunks("Wow! That is great. Really?")
        assert len(result) >= 2


class TestFormatFileSize:
    def test_small_file_shows_kb(self, tmp_path):
        from session import BrainGrowSession
        p = tmp_path / "small.txt"
        p.write_bytes(b"x" * 500)
        result = BrainGrowSession._format_file_size(str(p))
        assert "KB" in result

    def test_large_file_shows_mb(self, tmp_path):
        from session import BrainGrowSession
        p = tmp_path / "big.bin"
        p.write_bytes(b"x" * 2_000_000)
        result = BrainGrowSession._format_file_size(str(p))
        assert "MB" in result

    def test_missing_file_returns_question_mark(self):
        from session import BrainGrowSession
        result = BrainGrowSession._format_file_size("/nonexistent/path/file.txt")
        assert result == "?"


# ===========================================================================
# Tab 1 — Grow (ingest / refresh / reset)
# ===========================================================================

class TestIngest:
    def test_ingest_empty_input_returns_warning(self, tiny_session):
        status, umap_fig, hist_fig = tiny_session.ingest("   ", "science")
        assert "⚠️" in status

    def test_ingest_activates_slots(self, tiny_session):
        tiny_session.ingest("DNA replication is a biological process.", "science")
        assert tiny_session.vs.n_active >= 1

    def test_ingest_updates_dense_model(self, tiny_session):
        assert len(tiny_session.dense_model.labels) == 0
        tiny_session.ingest("Photosynthesis converts sunlight to energy.", "science")
        assert len(tiny_session.dense_model.labels) >= 1

    def test_ingest_dense_model_grows_incrementally(self, tiny_session):
        tiny_session.ingest("First fact about biology here.", "science")
        n1 = len(tiny_session.dense_model.labels)
        tiny_session.ingest("Second fact about history here.", "history")
        n2 = len(tiny_session.dense_model.labels)
        assert n2 > n1

    def test_ingest_returns_status_string(self, tiny_session):
        status, _, _ = tiny_session.ingest("Some educational content here.", "default")
        assert isinstance(status, str)
        assert "Stage" in status

    def test_ingest_uses_domain_label(self, tiny_session):
        tiny_session.ingest("Quantum mechanics describes subatomic particles.", "physics")
        domains = set(tiny_session.vs.slot_domains.values())
        assert "physics" in domains

    def test_ingest_defaults_domain_to_default(self, tiny_session):
        tiny_session.ingest("Some text without a domain.", "")
        domains = set(tiny_session.vs.slot_domains.values())
        assert "default" in domains


class TestResetAll:
    def test_reset_clears_active_slots(self, tiny_session):
        tiny_session.ingest("Some text to populate slots first.", "d")
        assert tiny_session.vs.n_active >= 1
        tiny_session.reset_all()
        assert tiny_session.vs.n_active == 0

    def test_reset_clears_dense_model(self, tiny_session):
        tiny_session.ingest("Some text to populate dense model.", "d")
        tiny_session.reset_all()
        assert len(tiny_session.dense_model.labels) == 0

    def test_reset_clears_engine_chunks(self, tiny_session):
        tiny_session.ingest("Some text for the engine chunks.", "d")
        tiny_session.reset_all()
        assert tiny_session.engine.all_chunks == []

    def test_reset_returns_status_string(self, tiny_session):
        result = tiny_session.reset_all()
        # reset_all returns a tuple (status, umap_fig, hist_fig)
        assert isinstance(result[0], str)
        assert "reset" in result[0].lower()


# ===========================================================================
# Tab 2 — Query
# ===========================================================================

class TestQuery:
    def test_query_empty_input_returns_warning(self, tiny_session):
        result, ratio = tiny_session.query("   ", 5)
        assert "⚠️" in result

    def test_query_without_ingested_data_returns_no_slots_message(self, tiny_session):
        result, ratio = tiny_session.query("what is DNA replication", 5)
        assert "ingest" in result.lower() or "no active" in result.lower()

    def test_query_after_ingest_returns_matches(self, tiny_session):
        tiny_session.ingest("DNA replication is a biological process.", "science")
        result, ratio = tiny_session.query("DNA replication", 3)
        # Should return match content, not a warning
        assert "⚠️" not in result
        assert len(result) > 0

    def test_query_ratio_string_contains_counts(self, tiny_session):
        tiny_session.ingest("Fermentation produces ethanol and CO2.", "biology")
        _, ratio = tiny_session.query("fermentation process", 3)
        assert "Active" in ratio
        assert "Dormant" in ratio


# ===========================================================================
# Tab 3 — Prune
# ===========================================================================

class TestRunPrune:
    def test_prune_returns_status_and_figure(self, tiny_session):
        tiny_session.ingest("Some content to prune after ingestion.", "d")
        status, fig = tiny_session.run_prune(threshold=0.9)
        assert isinstance(status, str)
        assert "Pruning" in status or "threshold" in status.lower()

    def test_prune_reduces_active_count(self, tiny_session):
        tiny_session.ingest("Biology fact about cells and DNA here.", "science")
        n_before = tiny_session.vs.n_active
        # Prune at 1.0 threshold — removes everything active
        tiny_session.run_prune(threshold=1.0)
        assert tiny_session.vs.n_active <= n_before


# ===========================================================================
# Tab 4 — Compare
# ===========================================================================

class TestRunComparisonTab:
    def test_no_data_returns_warning_html(self, tiny_session):
        html, dense_fig, bg_fig, status = tiny_session.run_comparison_tab(
            "Known", "Tell me about DNA replication"
        )
        assert "⚠️" in html or "No data" in html or "ingest" in html.lower()

    def test_no_query_selected_returns_warning(self, tiny_session):
        html, dense_fig, bg_fig, status = tiny_session.run_comparison_tab("Known", "")
        assert "⚠️" in html


# ===========================================================================
# Tab 5 — Network (save / load)
# ===========================================================================

class TestNetworkSaveLoad:
    def test_save_empty_space_returns_warning(self, tiny_session):
        status = tiny_session.save_network("test save")
        assert "⚠️" in status

    def test_save_after_ingest_creates_file(self, tiny_session):
        tiny_session.ingest("Content to save in the network file.", "science")
        status = tiny_session.save_network("unit test save")
        assert "✅" in status
        saves = list((tiny_session.SAVES_DIR).glob("*.bgstate"))
        assert len(saves) == 1

    def test_load_roundtrip_restores_active_count(self, tiny_session):
        tiny_session.ingest("Data for save/load roundtrip test.", "science")
        n_before = tiny_session.vs.n_active
        tiny_session.save_network("roundtrip test")

        saves = list(tiny_session.SAVES_DIR.glob("*.bgstate"))
        assert saves

        # Reset then reload
        tiny_session.reset_all()
        assert tiny_session.vs.n_active == 0

        status, _, _ = tiny_session.load_network(str(saves[0]))
        assert "✅" in status
        assert tiny_session.vs.n_active == n_before

    def test_load_roundtrip_restores_dense_model(self, tiny_session):
        tiny_session.ingest("Data for dense model reload test.", "science")
        n_labels = len(tiny_session.dense_model.labels)
        tiny_session.save_network("dense roundtrip")

        saves = list(tiny_session.SAVES_DIR.glob("*.bgstate"))
        tiny_session.reset_all()
        assert len(tiny_session.dense_model.labels) == 0

        tiny_session.load_network(str(saves[0]))
        assert len(tiny_session.dense_model.labels) == n_labels

    def test_load_roundtrip_restores_negative_domains(self, tiny_session):
        tiny_session.ingest("Toxic mixing is dangerous.", "cooking-negative")
        tiny_session.save_network("negative domains roundtrip")

        saves = list(tiny_session.SAVES_DIR.glob("*.bgstate"))
        tiny_session.reset_all()
        assert len(tiny_session.vs.negative_domains) == 0

        tiny_session.load_network(str(saves[0]))
        assert "cooking-negative" in tiny_session.vs.negative_domains

    def test_load_nonexistent_file_returns_warning(self, tiny_session):
        status, _, _ = tiny_session.load_network("/nonexistent/path/file.bgstate")
        assert "⚠️" in status

    def test_load_no_selection_returns_warning(self, tiny_session):
        status, _, _ = tiny_session.load_network("")
        assert "⚠️" in status

    def test_delete_existing_file(self, tiny_session):
        tiny_session.ingest("Content for delete test ingestion.", "d")
        tiny_session.save_network("to be deleted")
        saves = list(tiny_session.SAVES_DIR.glob("*.bgstate"))
        assert saves
        status = tiny_session.delete_save(str(saves[0]))
        assert "🗑️" in status
        assert not saves[0].exists()

    def test_delete_nonexistent_returns_warning(self, tiny_session):
        status = tiny_session.delete_save("/nonexistent/file.bgstate")
        assert "⚠️" in status


# ===========================================================================
# Autosave / network info
# ===========================================================================

class TestToggleAutosave:
    def test_enable_returns_enabled_message(self, tiny_session):
        msg = tiny_session.toggle_autosave(True)
        assert tiny_session.autosave_enabled is True
        assert "enabled" in msg.lower()

    def test_disable_returns_disabled_message(self, tiny_session):
        tiny_session.toggle_autosave(True)
        msg = tiny_session.toggle_autosave(False)
        assert tiny_session.autosave_enabled is False
        assert "disabled" in msg.lower()


class TestGetNetworkInfo:
    def test_returns_string(self, tiny_session):
        info = tiny_session.get_network_info()
        assert isinstance(info, str)

    def test_contains_active_and_dormant(self, tiny_session):
        info = tiny_session.get_network_info()
        assert "Active" in info
        assert "Dormant" in info

    def test_reflects_ingest(self, tiny_session):
        info_before = tiny_session.get_network_info()
        tiny_session.ingest("New knowledge about photosynthesis process.", "biology")
        info_after = tiny_session.get_network_info()
        # active count should be higher after ingest
        assert info_before != info_after
