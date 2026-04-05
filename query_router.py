"""
query_router.py — Semantic query routing for BrainGrow.

Encodes a natural-language query with sentence-transformers and routes
it exclusively through the *active* slots of the VectorSpace, ignoring
all dormant slots. Top-K matches are returned; each matched slot is
reinforced, raising its activation score.

Retrieval is now backed by VectorSpace.search_active(), which uses a
FAISS index (GPU-accelerated on RTX 4070) when available and falls back
to brute-force cosine similarity otherwise. This reduces query time from
O(n_active) to O(log n_active) at scale.
"""

from __future__ import annotations

from typing import List, Union

import torch
from sentence_transformers import SentenceTransformer

from utils import encode_unit_torch
from vector_space import VectorSpace


class QueryRouter:

    def __init__(
        self,
        vector_space: VectorSpace,
        model: Union[SentenceTransformer, str] = "all-MiniLM-L6-v2",
    ) -> None:
        self.vs = vector_space
        self.model: SentenceTransformer = (
            SentenceTransformer(model) if isinstance(model, str)
            else model
        )

    # --------------------------------------------------------------------------

    def route_query(self, text: str, top_k: int = 5) -> dict:
        """
        Route *text* through the active vector space.

        Uses VectorSpace.search_active() which is FAISS-backed when faiss
        is installed (GPU-accelerated) and brute-force otherwise.

        Returns
        -------
        {
            matches: [
                {slot_idx, label, domain, similarity, activation}, ...
            ],
            active_count      : int,
            dormant_count     : int,
            query_embedding   : Tensor[D],
            boundary_violation: bool,   # True when nearest domain is negative
            nearest_domain    : str,
            faiss_used        : bool,   # True when FAISS index served the query
        }

        When *boundary_violation* is True the caller should surface
        "BOUNDARY VIOLATION — concept exists but combination is invalid"
        rather than the raw matches.
        """
        emb_unit = encode_unit_torch(self.model, text)

        with self.vs._lock:
            active_count  = self.vs.n_active
            dormant_count = self.vs.N - active_count

            if active_count == 0:
                return {
                    "matches":            [],
                    "active_count":       0,
                    "dormant_count":      dormant_count,
                    "query_embedding":    emb_unit,
                    "nearest_domain":     "",
                    "boundary_violation": False,
                    "faiss_used":         False,
                }

            # Delegate retrieval to VectorSpace — FAISS or brute-force
            similarities, slot_indices = self.vs.search_active(emb_unit, top_k)

            matches: List[dict] = []
            for sim, slot_idx in zip(similarities, slot_indices):
                self.vs.reinforce(slot_idx)
                matches.append({
                    "slot_idx":   slot_idx,
                    "label":      self.vs.slot_labels.get(slot_idx, f"slot_{slot_idx}"),
                    "domain":     self.vs.slot_domains.get(slot_idx, "unknown"),
                    "similarity": round(float(sim), 4),
                    "activation": round(float(self.vs.activation[slot_idx].item()), 4),
                })

            nearest_domain     = matches[0]["domain"] if matches else ""
            boundary_violation = nearest_domain in self.vs.negative_domains

            return {
                "matches":            matches,
                "active_count":       active_count,
                "dormant_count":      dormant_count,
                "query_embedding":    emb_unit,
                "nearest_domain":     nearest_domain,
                "boundary_violation": boundary_violation,
                "faiss_used":         self.vs.faiss_available,
            }
