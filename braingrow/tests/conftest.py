"""
conftest.py — Shared pytest fixtures for the BrainGrow test suite.

Uses a tiny VectorSpace (50 slots × 8 dimensions) and a deterministic
mock encoder so tests run in milliseconds without GPU or network access.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import pytest
import torch

# Make braingrow package importable from the tests/ subdirectory
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_space import VectorSpace

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
N_SLOTS = 50
DIMS = 8


# ---------------------------------------------------------------------------
# Mock sentence-transformers model
# ---------------------------------------------------------------------------

class MockSentenceTransformer:
    """
    Deterministic fake encoder:  sha256(text) → seeded random unit vector.

    Supports both single-string and list-of-strings inputs, and the
    keyword arguments used by ingest_stage_batched (batch_size,
    show_progress_bar, device, convert_to_numpy).
    """

    def __init__(self, *args, **kwargs) -> None:  # accepts model name
        pass

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

    def _one(self, text: str) -> np.ndarray:
        """sha256(text) → deterministic 8-D unit vector."""
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.RandomState(seed)
        v = rng.randn(DIMS).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vs() -> VectorSpace:
    """Empty VectorSpace with 50 slots × 8 dimensions."""
    return VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)


@pytest.fixture
def mock_model() -> MockSentenceTransformer:
    """Deterministic mock sentence-transformers encoder."""
    return MockSentenceTransformer()


@pytest.fixture
def populated_vs(vs, mock_model) -> VectorSpace:
    """
    VectorSpace with ~5 active slots across two domains (science / history).
    Embeddings are determined by the mock model so tests are reproducible.
    """
    from growth_engine import GrowthEngine

    engine = GrowthEngine(vs, mock_model)
    chunks = [
        ("DNA replication is a biological process", "science"),
        ("Photosynthesis converts light to energy", "science"),
        ("The Roman Empire fell in 476 AD", "history"),
        ("The French Revolution began in 1789", "history"),
        ("Fermentation is used to make bread and beer", "cooking"),
    ]
    engine.ingest_stage(chunks)
    return vs
