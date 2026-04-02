"""
test_tinystories_loader.py — Unit tests for the TinyStories data pipeline.

All HuggingFace network I/O is mocked so the suite runs offline and instantly.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch
from typing import List, Tuple

import pytest


# ---------------------------------------------------------------------------
# Helpers to build a fake HF Dataset (list-of-dicts with a 'text' key)
# ---------------------------------------------------------------------------

def _fake_dataset(stories: List[str]):
    """Return a mock that looks like a HF Dataset to chunk_stories()."""

    class _FakeDataset:
        def __init__(self, records):
            self._records = records

        def __iter__(self):
            return iter(self._records)

        def __len__(self):
            return len(self._records)

        def select(self, indices):
            return _FakeDataset([self._records[i] for i in indices])

    return _FakeDataset([{"text": s} for s in stories])


_STORIES = [
    "The cat sat on the mat. The dog ran up the hill. They played all day long.",
    "Once there was a bunny. The bunny liked carrots. It hopped through the garden.",
    "A short one.",                     # this whole entry is short; will produce a tiny chunk
    "First sentence here. Second sentence here. Third sentence ends.",
    "x" * 30 + ". " + "y" * 30 + ".",  # each half is short; at exactly 50 chars or over
]


# ===========================================================================
# STAGE_PRESETS
# ===========================================================================

class TestStagePresets:
    def test_has_three_presets(self):
        from tinystories_loader import STAGE_PRESETS
        assert len(STAGE_PRESETS) == 3

    def test_preset_keys_exist(self):
        from tinystories_loader import STAGE_PRESETS
        assert any("Stage A" in k for k in STAGE_PRESETS)
        assert any("Stage B" in k for k in STAGE_PRESETS)
        assert any("Stage C" in k for k in STAGE_PRESETS)

    def test_presets_have_required_keys(self):
        from tinystories_loader import STAGE_PRESETS
        for name, cfg in STAGE_PRESETS.items():
            assert "sample_size" in cfg, f"Missing sample_size in {name!r}"
            assert "max_chunks" in cfg, f"Missing max_chunks in {name!r}"

    def test_stage_a_smaller_than_b(self):
        from tinystories_loader import STAGE_PRESETS
        a = next(v for k, v in STAGE_PRESETS.items() if "Stage A" in k)
        b = next(v for k, v in STAGE_PRESETS.items() if "Stage B" in k)
        assert a["max_chunks"] < b["max_chunks"]

    def test_stage_b_smaller_than_c(self):
        from tinystories_loader import STAGE_PRESETS
        b = next(v for k, v in STAGE_PRESETS.items() if "Stage B" in k)
        c = next(v for k, v in STAGE_PRESETS.items() if "Stage C" in k)
        assert b["max_chunks"] < c["max_chunks"]


# ===========================================================================
# _check_datasets_available
# ===========================================================================

class TestCheckDatasetsAvailable:
    def test_returns_true_when_datasets_importable(self):
        from tinystories_loader import _check_datasets_available
        fake_datasets = types.ModuleType("datasets")
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            assert _check_datasets_available() is True

    def test_returns_false_when_datasets_missing(self):
        from tinystories_loader import _check_datasets_available
        with patch.dict(sys.modules, {"datasets": None}):
            result = _check_datasets_available()
            assert result is False


# ===========================================================================
# chunk_stories
# ===========================================================================

class TestChunkStories:
    def test_returns_list_of_tuples(self):
        from tinystories_loader import chunk_stories
        ds = _fake_dataset(_STORIES[:2])
        chunks = chunk_stories(ds)
        assert isinstance(chunks, list)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in chunks)

    def test_each_chunk_is_string_pair(self):
        from tinystories_loader import chunk_stories
        ds = _fake_dataset(_STORIES[:2])
        chunks = chunk_stories(ds)
        for text, label in chunks:
            assert isinstance(text, str)
            assert isinstance(label, str)

    def test_default_domain_label_is_stories(self):
        from tinystories_loader import chunk_stories
        ds = _fake_dataset(_STORIES[:2])
        chunks = chunk_stories(ds)
        for _, label in chunks:
            assert label == "stories"

    def test_custom_domain_label_is_propagated(self):
        from tinystories_loader import chunk_stories
        ds = _fake_dataset(_STORIES[:2])
        chunks = chunk_stories(ds, domain_label="my_domain")
        for _, label in chunks:
            assert label == "my_domain"

    def test_short_chunks_are_filtered(self):
        from tinystories_loader import chunk_stories
        # A story that would produce a chunk shorter than 50 characters
        short_story = "Hi. Bye."
        ds = _fake_dataset([short_story])
        chunks = chunk_stories(ds)
        for text, _ in chunks:
            assert len(text) > 50, f"Got short chunk: {text!r}"

    def test_max_chunks_limits_output(self):
        from tinystories_loader import chunk_stories
        # Many stories, cap at 2
        many_stories = [
            "Sentence one here. Sentence two here. Sentence three here."
        ] * 20
        ds = _fake_dataset(many_stories)
        chunks = chunk_stories(ds, max_chunks=2)
        assert len(chunks) <= 2

    def test_empty_dataset_returns_empty_list(self):
        from tinystories_loader import chunk_stories
        ds = _fake_dataset([])
        chunks = chunk_stories(ds)
        assert chunks == []

    def test_produces_chunks_from_two_sentence_windows(self):
        from tinystories_loader import chunk_stories
        # Sentences long enough (> 25 chars each) so joined chunk exceeds 50 chars
        single = (
            "The quick brown fox jumps over the lazy dog. "
            "A wonderful story about animals in the forest. "
            "They lived happily ever after in the end."
        )
        ds = _fake_dataset([single])
        chunks = chunk_stories(ds)
        # Expect at least one chunk; each chunk should contain a period
        assert len(chunks) >= 1
        for text, _ in chunks:
            assert "." in text

    def test_no_chunk_exceeds_max_chunks(self):
        from tinystories_loader import chunk_stories
        many = ["A long enough sentence here. Another long enough sentence here."] * 100
        ds = _fake_dataset(many)
        limit = 7
        chunks = chunk_stories(ds, max_chunks=limit)
        assert len(chunks) == limit


# ===========================================================================
# load_tinystories (mocked)
# ===========================================================================

class TestLoadTinystories:
    def test_raises_import_error_when_datasets_missing(self):
        """If `datasets` is not available, load_tinystories raises ImportError."""
        with patch.dict(sys.modules, {"datasets": None}):
            from importlib import reload
            import tinystories_loader
            reload(tinystories_loader)          # re-evaluate module-level
            with pytest.raises(ImportError, match="datasets"):
                tinystories_loader.load_tinystories(sample_size=10)

    def test_calls_load_dataset_with_correct_args(self):
        """load_tinystories uses the roneneldan/TinyStories dataset."""
        fake_datasets = types.ModuleType("datasets")
        fake_ds = _fake_dataset(["Story one. Story two. Story three."] * 20)

        # load_dataset returns full dataset; .select() handled by _FakeDataset
        fake_datasets.load_dataset = MagicMock(return_value=fake_ds)

        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            from importlib import reload
            import tinystories_loader
            reload(tinystories_loader)

            result = tinystories_loader.load_tinystories(sample_size=5)

            fake_datasets.load_dataset.assert_called_once_with(
                "roneneldan/TinyStories", split="train"
            )
            assert len(result) <= 5


# ===========================================================================
# prepare_experiment (mocked)
# ===========================================================================

class TestPrepareExperiment:
    def test_returns_list_of_tuples(self):
        fake_datasets_mod = types.ModuleType("datasets")
        fake_ds = _fake_dataset(
            ["Sentence one here. Sentence two here. Sentence three."] * 20
        )
        fake_datasets_mod.load_dataset = MagicMock(return_value=fake_ds)

        with patch.dict(sys.modules, {"datasets": fake_datasets_mod}):
            from importlib import reload
            import tinystories_loader
            reload(tinystories_loader)

            chunks = tinystories_loader.prepare_experiment(
                sample_size=10, max_chunks=5, domain_label="test_domain"
            )

        assert isinstance(chunks, list)
        assert len(chunks) <= 5
        for text, domain in chunks:
            assert isinstance(text, str)
            assert domain == "test_domain"
