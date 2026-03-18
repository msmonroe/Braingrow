"""
growth_engine.py — Staged text ingestion for BrainGrow.

Accepts (text_chunk, domain_label) pairs, encodes them with
sentence-transformers, and grows them into the nearest dormant region
of the shared VectorSpace.  Does NOT batch-load all data at once;
each call to ingest_stage() represents one developmental stage.
"""

from __future__ import annotations
from typing import List, Tuple, Union

import torch
from sentence_transformers import SentenceTransformer

from vector_space import VectorSpace


class GrowthEngine:
    # Cosine-similarity threshold above which an incoming embedding is
    # considered "already represented" and reinforces an existing slot
    # rather than claiming a new dormant one.
    REINFORCE_THRESHOLD: float = 0.92

    def __init__(
        self,
        vector_space: VectorSpace,
        model: Union[SentenceTransformer, str] = "all-MiniLM-L6-v2",
    ) -> None:
        self.vs = vector_space
        self.model: SentenceTransformer = (
            model if isinstance(model, SentenceTransformer)
            else SentenceTransformer(model)
        )
        self.stage_number: int = 0
        self._stage_history: List[dict] = []

    # --------------------------------------------------------------------------
    def ingest_stage(self, chunks: List[Tuple[str, str]]) -> dict:
        """
        Process one developmental stage.

        Parameters
        ----------
        chunks : list of (text_chunk, domain_label)

        Returns
        -------
        {
            slots_activated  : List[int],
            slots_reinforced : List[int],
            dormant_remaining: int,
            stage_number     : int,
        }
        """
        self.stage_number += 1
        slots_activated: List[int] = []
        slots_reinforced: List[int] = []

        for text, domain in chunks:
            if not text.strip():
                continue
            embedding = torch.tensor(
                self.model.encode(text), dtype=torch.float32
            )
            emb_unit = embedding / (embedding.norm() + 1e-8)

            # Check for near-duplicate in the active space first
            active_mask = self.vs.get_active_mask()
            if active_mask.any():
                active_indices = active_mask.nonzero(as_tuple=True)[0]
                active_vecs = self.vs.slots[active_indices]
                sims = active_vecs @ emb_unit
                max_sim = float(sims.max().item())

                if max_sim >= self.REINFORCE_THRESHOLD:
                    best_global = int(active_indices[sims.argmax().item()].item())
                    self.vs.reinforce(best_global)
                    slots_reinforced.append(best_global)
                    continue  # no new slot needed

            # Grow into the nearest dormant region
            result = self.vs.assign_slot(
                emb_unit,
                label=text[:60].strip(),
                domain=domain,
            )
            slots_activated.append(result["slot_idx"])

        dormant_remaining = int((self.vs.activation == 0.0).sum().item())

        stage_result = {
            "slots_activated": slots_activated,
            "slots_reinforced": slots_reinforced,
            "dormant_remaining": dormant_remaining,
            "stage_number": self.stage_number,
        }
        self._stage_history.append({
            "stage": self.stage_number,
            "slots": slots_activated[:],
        })
        return stage_result

    # --------------------------------------------------------------------------
    def get_stage_diff(self) -> dict:
        """Return the slot indices activated in the most recent stage."""
        if not self._stage_history:
            return {"stage": 0, "new_slots": []}
        latest = self._stage_history[-1]
        return {"stage": latest["stage"], "new_slots": latest["slots"]}

    def get_all_stage_history(self) -> List[dict]:
        return list(self._stage_history)

    def reset(self) -> None:
        self.stage_number = 0
        self._stage_history = []
