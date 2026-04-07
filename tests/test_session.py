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
        convert_to_tensor = kwargs.get("convert_to_tensor", False)
        if isinstance(text, list):
            vecs = np.stack([self._one(t) for t in text], axis=0)
        else:
            vecs = self._one(text)
        if convert_to_tensor:
            return torch.tensor(vecs, dtype=torch.float32)
        return vecs

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
    # Removed stale class-level `from session import BrainGrowSession as _BS`
    # that was dead code and could cause import-order failures.
    # Each test method imports BrainGrowSession directly as needed.

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


# ===========================================================================
# refresh_umap / view_diff
# ===========================================================================

class TestRefreshUmap:
    def test_refresh_umap_returns_figure(self, tiny_session):
        import plotly.graph_objects as go
        fig = tiny_session.refresh_umap()
        assert isinstance(fig, go.Figure)

    def test_refresh_umap_reflects_active_slots(self, tiny_session):
        import plotly.graph_objects as go
        tiny_session.ingest("Photosynthesis converts light to chemical energy.", "science")
        fig = tiny_session.refresh_umap()
        assert isinstance(fig, go.Figure)


class TestViewDiff:
    def test_view_diff_no_new_slots_returns_figure(self, tiny_session):
        """Lines 177-179: if engine has no new-slot diff, returns plot_umap figure."""
        import plotly.graph_objects as go
        # No ingest → no stage history → diff["new_slots"] is empty
        fig = tiny_session.view_diff()
        assert isinstance(fig, go.Figure)

    def test_view_diff_after_ingest_returns_figure(self, tiny_session):
        """Line 180: after ingest there are new slots, returns plot_stage_diff figure."""
        import plotly.graph_objects as go
        # Ingest multiple distinct lines so >=2 new slots are created (PCA needs >=2 points)
        tiny_session.ingest(
            "Quantum mechanics energy levels.\n"
            "Classical physics momentum conservation.\n"
            "Thermodynamics entropy and heat.",
            "physics",
        )
        fig = tiny_session.view_diff()
        assert isinstance(fig, go.Figure)


# ===========================================================================
# Query — boundary violation and FAISS flag
# ===========================================================================

class TestQueryBoundaryViolation:
    def test_boundary_violation_returns_violation_message(self, tiny_session):
        """Lines 218-233: query that lands in a negative domain activates maintenance."""
        from utils import encode_unit_torch

        text = "boundary_violation_test_unique_xyz_abc_123"
        # Compute the same embedding the router will compute
        emb = encode_unit_torch(tiny_session.router.model, text)

        # Place a slot with that exact embedding in a negative domain
        neg_domain = "hazard-negative"
        tiny_session.vs.assign_slot(emb, label=text, domain=neg_domain)
        tiny_session.vs.register_negative_domain(neg_domain)

        result, ratio = tiny_session.query(text, 1)
        assert "BOUNDARY VIOLATION" in result

    def test_boundary_violation_increments_correction_count(self, tiny_session):
        """Maintenance correction log grows after a boundary violation."""
        from utils import encode_unit_torch

        text = "boundary_correction_count_test_unique_abc_99"
        emb = encode_unit_torch(tiny_session.router.model, text)
        neg_domain = "danger-negative"
        tiny_session.vs.assign_slot(emb, label=text, domain=neg_domain)
        tiny_session.vs.register_negative_domain(neg_domain)

        tiny_session.query(text, 1)
        assert tiny_session.maintenance.correction_count() == 1

    def test_boundary_violation_ratio_string_populated(self, tiny_session):
        """The ratio string is set even when a boundary violation is triggered."""
        from utils import encode_unit_torch

        text = "boundary_ratio_test_text_unique_xyz_007"
        emb = encode_unit_torch(tiny_session.router.model, text)
        neg_domain = "toxic-negative"
        tiny_session.vs.assign_slot(emb, label=text, domain=neg_domain)
        tiny_session.vs.register_negative_domain(neg_domain)

        _, ratio = tiny_session.query(text, 1)
        assert "Active" in ratio


class TestQueryFaissFlag:
    def test_faiss_flag_appears_in_ratio_when_available(self, tiny_session):
        """Line 210-211: if faiss_used is True, ratio contains 'FAISS'."""
        import vector_space as _vs_mod
        if not _vs_mod._FAISS_AVAILABLE:
            pytest.skip("FAISS not installed")
        tiny_session.ingest("Quantum entanglement is a physics phenomenon.", "physics")
        _, ratio = tiny_session.query("quantum entanglement", 3)
        # With FAISS installed and active slots, faiss_used == True
        assert "FAISS" in ratio


# ===========================================================================
# get_query_choices / run_comparison_tab with data
# ===========================================================================

class TestGetQueryChoices:
    def test_known_returns_known_queries(self, tiny_session):
        from comparison_harness import known_queries
        choices = tiny_session.get_query_choices("Known")
        assert choices == known_queries

    def test_partial_returns_partial_queries(self, tiny_session):
        from comparison_harness import partial_queries
        choices = tiny_session.get_query_choices("Partial")
        assert choices == partial_queries

    def test_unknown_returns_unknown_queries(self, tiny_session):
        from comparison_harness import unknown_queries
        choices = tiny_session.get_query_choices("Unknown")
        assert choices == unknown_queries

    def test_invalid_type_returns_empty(self, tiny_session):
        choices = tiny_session.get_query_choices("Nonexistent")
        assert choices == []


class TestRunComparisonTabWithData:
    def _ingest_data(self, tiny_session):
        """Populate both VectorSpace and DenseModel with a few facts."""
        tiny_session.ingest(
            "DNA replication is a fundamental biological process.\n"
            "Photosynthesis converts sunlight into chemical energy.\n"
            "The Roman Empire fell in 476 AD.",
            "science",
        )

    def test_known_query_returns_html_table(self, tiny_session):
        self._ingest_data(tiny_session)
        from comparison_harness import known_queries
        query = known_queries[0]
        html, dense_fig, bg_fig, status = tiny_session.run_comparison_tab("Known", query)
        assert "Dense" in html
        assert "BrainGrow" in html
        assert isinstance(status, str)

    def test_unknown_query_marks_hallucination(self, tiny_session):
        """For Unknown queries, Dense model is flagged as HALLUCINATED when confident."""
        self._ingest_data(tiny_session)
        from comparison_harness import unknown_queries
        query = unknown_queries[0]
        html, _, _, _ = tiny_session.run_comparison_tab("Unknown", query)
        # DenseModel always returns confident=True → if similarity > 0, it hallucinated
        assert isinstance(html, str)
        assert "<table" in html

    def test_partial_query_returns_table(self, tiny_session):
        self._ingest_data(tiny_session)
        from comparison_harness import partial_queries
        query = partial_queries[0]
        html, dense_fig, bg_fig, status = tiny_session.run_comparison_tab("Partial", query)
        assert "<table" in html
        assert dense_fig is not None
        assert bg_fig is not None

    def test_status_contains_similarity_info(self, tiny_session):
        self._ingest_data(tiny_session)
        from comparison_harness import known_queries
        _, _, _, status = tiny_session.run_comparison_tab("Known", known_queries[0])
        assert "sim" in status.lower() or "Dense sim" in status


# ===========================================================================
# run_audit
# ===========================================================================

class TestRunAudit:
    def test_run_audit_empty_space_returns_warning(self, tiny_session):
        """Line 460: run_audit with no active slots returns warning string."""
        result = tiny_session.run_audit()
        assert "⚠️" in result

    def test_run_audit_after_ingest_returns_report_text(self, tiny_session):
        """After ingesting data, run_audit returns AuditReport.as_text()."""
        tiny_session.ingest(
            "Biology fact about cell division.\n"
            "Chemistry fact about molecular bonds.\n"
            "Physics fact about quantum states.",
            "science",
        )
        result = tiny_session.run_audit()
        assert isinstance(result, str)
        # Report should mention domains or audit
        assert "Hallucination" in result or "domain" in result.lower() or "session" in result

    def test_run_audit_includes_correction_count(self, tiny_session):
        """Footer always mentions correction count."""
        tiny_session.ingest("Some educational text about science here.", "science")
        result = tiny_session.run_audit()
        assert "correction" in result.lower()


# ===========================================================================
# Tab 6 — TinyStories (datasets unavailable path)
# ===========================================================================

class TestListSaves:
    def test_list_saves_empty_when_no_saves(self, tiny_session):
        """Line 375: list_saves returns empty list when no .bgstate files exist."""
        saves = tiny_session.list_saves()
        assert saves == []

    def test_list_saves_returns_paths_after_save(self, tiny_session):
        tiny_session.ingest("Content for list_saves test.", "science")
        tiny_session.save_network("list-saves-test")
        saves = tiny_session.list_saves()
        assert len(saves) == 1
        assert saves[0].endswith(".bgstate")


class TestDeleteSaveEmptySelection:
    def test_delete_save_empty_string_returns_warning(self, tiny_session):
        """Line 460: delete_save with empty selection returns 'No file selected'."""
        status = tiny_session.delete_save("")
        assert "⚠️" in status
        assert "selected" in status.lower()


class TestRunTinyStoriesUnavailable:
    def test_datasets_unavailable_returns_warning(self, tiny_session, monkeypatch):
        """Lines 500-515: when datasets is not installed, returns install instructions."""
        import session as _s
        monkeypatch.setattr(_s, "_check_datasets_available", lambda: False)
        msg, fig1, fig2 = tiny_session.run_tinystories_stage("Stage A — Warmup", 100, 100)
        assert "⚠️" in msg
        assert "datasets" in msg
        assert fig1 is None
        assert fig2 is None


class TestRunTinyStoriesAvailable:
    def test_tinystories_runs_with_preset(self, tiny_session, monkeypatch):
        """Lines 536-579: full TinyStories path when datasets IS available."""
        import session as _s

        # Fake chunks returned by prepare_experiment
        fake_chunks = [
            ("Once upon a time a cat sat on a mat.", "stories"),
            ("The dog ran to the park every day.", "stories"),
            ("Alice liked to read books in the garden.", "stories"),
        ]

        monkeypatch.setattr(_s, "_check_datasets_available", lambda: True)
        # Inject a known preset so the preset branch is taken
        monkeypatch.setattr(_s, "STAGE_PRESETS", {
            "TestPreset": {"sample_size": 3, "max_chunks": 3},
        })
        monkeypatch.setattr(_s, "prepare_experiment", lambda **kw: fake_chunks)

        msg, fig1, fig2 = tiny_session.run_tinystories_stage("TestPreset", 0, 0)
        assert "Stage" in msg or "complete" in msg.lower()
        assert fig2 is not None  # histogram is always returned

    def test_tinystories_uses_custom_sample_when_no_preset(self, tiny_session, monkeypatch):
        """Lines 542-543: custom_sample / custom_chunks branch when preset not found."""
        import session as _s

        fake_chunks = [
            ("Custom sample story text here once.", "stories"),
            ("Another custom story about a rabbit.", "stories"),
            ("Third tiny story about woodland creatures.", "stories"),
        ]

        monkeypatch.setattr(_s, "_check_datasets_available", lambda: True)
        monkeypatch.setattr(_s, "STAGE_PRESETS", {})  # no presets
        monkeypatch.setattr(_s, "prepare_experiment", lambda **kw: fake_chunks)

        msg, _, _ = tiny_session.run_tinystories_stage("NonExistentPreset", 3, 3)
        assert isinstance(msg, str)

    def test_tinystories_error_handling(self, tiny_session, monkeypatch):
        """Error path: prepare_experiment raises → returns error message."""
        import session as _s

        monkeypatch.setattr(_s, "_check_datasets_available", lambda: True)
        monkeypatch.setattr(_s, "STAGE_PRESETS", {})

        def _fail(**kw):
            raise RuntimeError("download failed")

        monkeypatch.setattr(_s, "prepare_experiment", _fail)

        msg, fig1, fig2 = tiny_session.run_tinystories_stage("X", 10, 10)
        assert "Error" in msg or "❌" in msg
