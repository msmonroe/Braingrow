"""
flat_threshold.py — Bare vector-store baselines for BrainGrow comparison.

Two implementations, same behavior:
  - TorchFlatThreshold: torch.Tensor store, matches BrainGrow's internals
  - FAISSFlatThreshold: faiss.IndexFlatIP, standard RAG primitive

Both use the same encoder as BrainGrow (all-MiniLM-L6-v2), apply the same
0.60 cosine threshold, and expose an identical query() interface. Any
observable difference between these two and BrainGrow comes from the
developmental machinery, NOT the threshold or the encoder.

Intentionally minimal: no lifecycle, no activation scores, no pruning.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_ENCODER = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.60
DEFAULT_DIM = 384


# --------------------------------------------------------------------------
# Torch tensor baseline
# --------------------------------------------------------------------------
class TorchFlatThreshold:
    """
    Flat tensor store with a cosine-similarity threshold.

    Mirrors BrainGrow's internals (unit-normalized torch tensors, matmul
    for similarity) but with zero lifecycle machinery.
    """

    def __init__(
        self,
        encoder: Union[SentenceTransformer, str] = DEFAULT_ENCODER,
        threshold: float = DEFAULT_THRESHOLD,
        dim: int = DEFAULT_DIM,
    ) -> None:
        self.encoder: SentenceTransformer = (
            SentenceTransformer(encoder) if isinstance(encoder, str) else encoder
        )
        self.threshold = threshold
        self.dim = dim

        # [N, D] unit-normalized tensor; None until first ingest
        self._embeddings: Optional[torch.Tensor] = None
        # parallel metadata
        self._labels: List[str] = []
        self._domains: List[str] = []

    # -- ingestion ----------------------------------------------------------
    def ingest(self, chunks: List[Tuple[str, str]], batch_size: int = 32) -> int:
        """
        Encode (text, domain) pairs and append them to the store.

        Returns the number of chunks added.
        """
        valid = [(t, d) for t, d in chunks if t and t.strip()]
        if not valid:
            return 0

        texts = [t for t, _ in valid]
        embs_np = self.encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embs = torch.tensor(embs_np, dtype=torch.float32)
        embs = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)

        if self._embeddings is None:
            self._embeddings = embs
        else:
            self._embeddings = torch.cat([self._embeddings, embs], dim=0)

        self._labels.extend(t[:60].strip() for t in texts)
        self._domains.extend(d for _, d in valid)
        return len(valid)

    # -- query --------------------------------------------------------------
    def query(self, text: str) -> dict:
        """
        Encode the query, compute cosine similarity against the full store,
        apply the threshold.

        Returns
        -------
        {
            verdict        : "CONFIDENT" | "HONEST_UNKNOWN",
            similarity     : float,          # max cosine similarity observed
            nearest_label  : str,            # label of best match, or ""
            nearest_domain : str,            # domain of best match, or ""
            store_size     : int,
        }
        """
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            return {
                "verdict": "HONEST_UNKNOWN",
                "similarity": 0.0,
                "nearest_label": "",
                "nearest_domain": "",
                "store_size": 0,
            }

        q_np = self.encoder.encode([text], convert_to_numpy=True)[0]
        q = torch.tensor(q_np, dtype=torch.float32)
        q = q / (q.norm() + 1e-8)

        sims = self._embeddings @ q                      # [N]
        max_sim = float(sims.max().item())
        best_idx = int(sims.argmax().item())

        verdict = "CONFIDENT" if max_sim >= self.threshold else "HONEST_UNKNOWN"
        return {
            "verdict": verdict,
            "similarity": max_sim,
            "nearest_label": self._labels[best_idx],
            "nearest_domain": self._domains[best_idx],
            "store_size": int(self._embeddings.shape[0]),
        }

    def __len__(self) -> int:
        return 0 if self._embeddings is None else int(self._embeddings.shape[0])


# --------------------------------------------------------------------------
# FAISS baseline
# --------------------------------------------------------------------------
class FAISSFlatThreshold:
    """
    IndexFlatIP-backed store with a cosine-similarity threshold.

    With L2-normalized vectors, inner product == cosine similarity, so
    IndexFlatIP is the FAISS-native equivalent of the torch baseline.
    This is the canonical RAG-shaped comparison: anyone who has ever
    deployed a retrieval system has built essentially this.
    """

    def __init__(
        self,
        encoder: Union[SentenceTransformer, str] = DEFAULT_ENCODER,
        threshold: float = DEFAULT_THRESHOLD,
        dim: int = DEFAULT_DIM,
    ) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "FAISSFlatThreshold requires faiss. Install with:\n"
                "  pip install faiss-cpu     # CPU\n"
                "  pip install faiss-gpu     # GPU (if supported)\n"
            ) from e

        import faiss
        self.encoder: SentenceTransformer = (
            SentenceTransformer(encoder) if isinstance(encoder, str) else encoder
        )
        self.threshold = threshold
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._labels: List[str] = []
        self._domains: List[str] = []

    # -- ingestion ----------------------------------------------------------
    def ingest(self, chunks: List[Tuple[str, str]], batch_size: int = 32) -> int:
        import faiss
        valid = [(t, d) for t, d in chunks if t and t.strip()]
        if not valid:
            return 0

        texts = [t for t, _ in valid]
        embs = self.encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        faiss.normalize_L2(embs)
        self._index.add(embs)

        self._labels.extend(t[:60].strip() for t in texts)
        self._domains.extend(d for _, d in valid)
        return len(valid)

    # -- query --------------------------------------------------------------
    def query(self, text: str) -> dict:
        import faiss
        if self._index.ntotal == 0:
            return {
                "verdict": "HONEST_UNKNOWN",
                "similarity": 0.0,
                "nearest_label": "",
                "nearest_domain": "",
                "store_size": 0,
            }

        q = self.encoder.encode([text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        sims, idxs = self._index.search(q, 1)
        max_sim = float(sims[0][0])
        best_idx = int(idxs[0][0])

        verdict = "CONFIDENT" if max_sim >= self.threshold else "HONEST_UNKNOWN"
        return {
            "verdict": verdict,
            "similarity": max_sim,
            "nearest_label": self._labels[best_idx],
            "nearest_domain": self._domains[best_idx],
            "store_size": int(self._index.ntotal),
        }

    def __len__(self) -> int:
        return int(self._index.ntotal)
