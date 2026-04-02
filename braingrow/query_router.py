"""
query_router.py — Semantic query routing for BrainGrow.

Encodes a natural-language query with sentence-transformers and routes
it exclusively through the *active* slots of the VectorSpace, ignoring
all dormant slots.  Top-K matches are returned; each matched slot is
reinforced, raising its activation score.
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

        Returns
        -------
        {
            matches: [
                {slot_idx, label, domain, similarity, activation}, ...
            ],
            active_count     : int,
            dormant_count    : int,
            query_embedding  : Tensor[D],
            boundary_violation: bool,   # True when nearest domain contains 'negative'
            nearest_domain   : str,     # domain of the top match (empty if no matches)
        }

        When *boundary_violation* is True the caller should surface
        "BOUNDARY VIOLATION — concept exists but combination is invalid"
        rather than the raw matches.
        """
        emb_unit = encode_unit_torch(self.model, text)

        with self.vs._lock:
            active_mask = self.vs.get_active_mask()
            active_count = int(active_mask.sum().item())
            dormant_count = int((~active_mask).sum().item())

            if active_count == 0:
                return {
                    "matches": [],
                    "active_count": 0,
                    "dormant_count": dormant_count,
                    "query_embedding": emb_unit,
                    "nearest_domain": "",
                    "boundary_violation": False,
                }

            active_indices = active_mask.nonzero(as_tuple=True)[0]
            active_vecs = self.vs.slots[active_indices]  # [A, D]

            sims = active_vecs @ emb_unit  # [A]

            k = min(top_k, active_count)
            top_vals, top_local = sims.topk(k)

            matches: List[dict] = []
            for val, local_idx in zip(top_vals.tolist(), top_local.tolist()):
                slot_idx = int(active_indices[local_idx].item())
                self.vs.reinforce(slot_idx)
                matches.append({
                    "slot_idx": slot_idx,
                    "label": self.vs.slot_labels.get(slot_idx, f"slot_{slot_idx}"),
                    "domain": self.vs.slot_domains.get(slot_idx, "unknown"),
                    "similarity": round(float(val), 4),
                    "activation": round(float(self.vs.activation[slot_idx].item()), 4),
                })

            nearest_domain = matches[0]["domain"] if matches else ""
            boundary_violation = nearest_domain in self.vs.negative_domains

        return {
            "matches": matches,
            "active_count": active_count,
            "dormant_count": dormant_count,
            "query_embedding": emb_unit,
            "nearest_domain": nearest_domain,
            "boundary_violation": boundary_violation,
        }
