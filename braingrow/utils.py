"""utils.py — Shared encoding utilities for BrainGrow."""
from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def encode_unit_torch(model: SentenceTransformer, text: str) -> torch.Tensor:
    """Encode *text* with *model* and return a unit-length float32 tensor."""
    emb = torch.tensor(model.encode(text.strip(), device=_DEVICE), dtype=torch.float32)
    return emb / (emb.norm() + 1e-8)


def encode_unit_numpy(model: SentenceTransformer, text: str) -> np.ndarray:
    """Encode *text* with *model* and return a unit-length float32 array."""
    emb = model.encode(text.strip(), device=_DEVICE).astype(np.float32)
    return emb / (float(np.linalg.norm(emb)) + 1e-8)
