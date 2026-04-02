"""
test_utils.py — Unit tests for shared encoding utilities.

Covers: encode_unit_torch and encode_unit_numpy (return type, unit length,
        float32 dtype, whitespace stripping, and zero-vector safety).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import encode_unit_numpy, encode_unit_torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ZeroModel:
    """Returns an all-zeros vector — tests the epsilon-safe normalisation path."""
    def encode(self, text, **kwargs):
        return np.zeros(8, dtype=np.float32)


# ===========================================================================
# encode_unit_torch
# ===========================================================================

class TestEncodeUnitTorch:
    def test_returns_torch_tensor(self, mock_model):
        result = encode_unit_torch(mock_model, "hello world")
        assert isinstance(result, torch.Tensor)

    def test_is_unit_length(self, mock_model):
        result = encode_unit_torch(mock_model, "test sentence for norm check")
        norm = result.norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_dtype_float32(self, mock_model):
        result = encode_unit_torch(mock_model, "some text")
        assert result.dtype == torch.float32

    def test_strips_whitespace(self, mock_model):
        """Leading/trailing whitespace should not change the result."""
        r1 = encode_unit_torch(mock_model, "hello")
        r2 = encode_unit_torch(mock_model, "  hello  ")
        torch.testing.assert_close(r1, r2)

    def test_zero_vector_no_nan(self):
        """Must not produce NaN when the model returns a zero vector."""
        result = encode_unit_torch(_ZeroModel(), "anything")
        assert not torch.isnan(result).any()

    def test_zero_vector_finite(self):
        result = encode_unit_torch(_ZeroModel(), "anything")
        assert torch.isfinite(result).all()


# ===========================================================================
# encode_unit_numpy
# ===========================================================================

class TestEncodeUnitNumpy:
    def test_returns_ndarray(self, mock_model):
        result = encode_unit_numpy(mock_model, "hello world")
        assert isinstance(result, np.ndarray)

    def test_is_unit_length(self, mock_model):
        result = encode_unit_numpy(mock_model, "test sentence for norm check")
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5

    def test_dtype_float32(self, mock_model):
        result = encode_unit_numpy(mock_model, "some text")
        assert result.dtype == np.float32

    def test_strips_whitespace(self, mock_model):
        r1 = encode_unit_numpy(mock_model, "hello")
        r2 = encode_unit_numpy(mock_model, "  hello  ")
        np.testing.assert_allclose(r1, r2, atol=1e-6)

    def test_zero_vector_no_nan(self):
        result = encode_unit_numpy(_ZeroModel(), "anything")
        assert not np.isnan(result).any()

    def test_zero_vector_finite(self):
        result = encode_unit_numpy(_ZeroModel(), "anything")
        assert np.isfinite(result).all()
