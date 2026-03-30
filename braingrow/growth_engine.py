"""
growth_engine.py — Staged text ingestion for BrainGrow.

Accepts (text_chunk, domain_label) pairs, encodes them with
sentence-transformers, and grows them into the nearest dormant region
of the shared VectorSpace.  Does NOT batch-load all data at once;
each call to ingest_stage() represents one developmental stage.
"""

from __future__ import annotations
from typing import List, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from vector_space import VectorSpace

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
            SentenceTransformer(model) if isinstance(model, str)
            else model
        )
        self.stage_number: int = 0
        self._stage_history: List[dict] = []
        # All (text, domain) pairs ever ingested — passed to DenseModel for
        # identical training data on Tab 4 comparison.
        self.all_chunks: List[Tuple[str, str]] = []

    # --------------------------------------------------------------------------
    def ingest_stage(
        self,
        chunks: List[Tuple[str, str]],
        autosave: bool = False,
        saves_dir: str = 'saves',
    ) -> dict:
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

        # Batch-encode all texts in one GPU call instead of one-at-a-time
        valid_chunks = [(t, d) for t, d in chunks if t.strip()]
        if not valid_chunks:
            return self._finalise_stage([], [], autosave, saves_dir)

        texts_list = [t for t, _ in valid_chunks]
        domains_list = [d for _, d in valid_chunks]
        total = len(texts_list)
        embeddings_np = self.model.encode(
            texts_list,
            device=_DEVICE,
            batch_size=512,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        for i, (emb_np, text, domain) in enumerate(zip(embeddings_np, texts_list, domains_list)):
            if i % 100 == 0:
                print(f"Progress: {i}/{total} chunks ingested")
            self._process_embedding(emb_np, text, domain, slots_activated, slots_reinforced)

        print(f"Progress: {total}/{total} chunks ingested")
        return self._finalise_stage(slots_activated, slots_reinforced, autosave, saves_dir)

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
        self.all_chunks = []

    # --------------------------------------------------------------------------
    # Private helpers — eliminate duplication between ingest_stage variants
    # --------------------------------------------------------------------------
    def _process_embedding(
        self,
        emb_np: np.ndarray,
        text: str,
        domain: str,
        slots_activated: List[int],
        slots_reinforced: List[int],
    ) -> None:
        """Reinforce an existing near-duplicate slot or grow into a dormant one."""
        self.all_chunks.append((text, domain))
        embedding = torch.tensor(emb_np, dtype=torch.float32)
        emb_unit = embedding / (embedding.norm() + 1e-8)

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
                return

        result = self.vs.assign_slot(emb_unit, label=text[:60].strip(), domain=domain)
        slots_activated.append(result["slot_idx"])

    def _finalise_stage(
        self,
        slots_activated: List[int],
        slots_reinforced: List[int],
        autosave: bool,
        saves_dir: str,
    ) -> dict:
        """Build the stage result dict, update history, sync state, and autosave."""
        dormant_remaining = int((self.vs.activation == 0.0).sum().item())
        stage_result = {
            "slots_activated": slots_activated,
            "slots_reinforced": slots_reinforced,
            "dormant_remaining": dormant_remaining,
            "stage_number": self.stage_number,
        }
        self._stage_history.append({"stage": self.stage_number, "slots": slots_activated[:]})
        self.vs.stage_number = self.stage_number
        if autosave:
            path = self.vs.autosave(saves_dir)
            print(f'Autosaved to {path}')
        return stage_result

    # --------------------------------------------------------------------------
    def ingest_stage_batched(
        self,
        chunks: List[Tuple[str, str]],
        batch_size: int = 512,
        autosave: bool = False,
        saves_dir: str = 'saves',
    ) -> dict:
        """
        High-throughput ingestion for large datasets (e.g. TinyStories).

        Encodes all *chunks* in a single batched GPU call instead of one
        text at a time.  Use this when ingesting thousands of chunks.

        Parameters
        ----------
        chunks      : list of (text_chunk, domain_label)
        batch_size  : sentences per encoding batch (GPU memory trade-off)
        autosave    : write a .bgstate checkpoint after this stage
        saves_dir   : directory for autosave files
        """
        if not chunks:
            return {"slots_activated": [], "slots_reinforced": [],
                    "dormant_remaining": int((self.vs.activation == 0).sum().item()),
                    "stage_number": self.stage_number}

        self.stage_number += 1
        slots_activated: List[int] = []
        slots_reinforced: List[int] = []

        texts  = [c[0] for c in chunks if c[0].strip()]
        labels = [c[1] for c in chunks if c[0].strip()]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            device=_DEVICE,
            convert_to_numpy=True,
        )

        total = len(texts)
        for i, (emb_np, label, text) in enumerate(zip(embeddings, labels, texts)):
            if i % 100 == 0:
                print(f"Progress: {i}/{total} chunks ingested")
            self._process_embedding(emb_np, text, label, slots_activated, slots_reinforced)

        print(f"Progress: {total}/{total} chunks ingested")
        return self._finalise_stage(slots_activated, slots_reinforced, autosave, saves_dir)
