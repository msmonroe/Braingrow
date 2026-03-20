"""
comparison_harness.py — Hallucination comparison harness for BrainGrow Tab 4.

Provides DenseModel (always-confident nearest-neighbour) and BrainGrowModel
(honest uncertainty via dormant space) running on identical training data.

Demonstrates that hallucination is an architectural property of a saturated
vector space — not a scale or data-quantity problem.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from vector_space import VectorSpace

# ---------------------------------------------------------------------------
# Predefined query sets (copy verbatim from spec)
# ---------------------------------------------------------------------------

known_queries: List[str] = [
    "Tell me about DNA replication",
    "When did the Western Roman Empire fall",
    "How does fermentation work",
]

partial_queries: List[str] = [
    "How did science influence the Silk Road",
    "What is the thermodynamics of cooking",
]

unknown_queries: List[str] = [
    "What is the capital of Zorbania",
    "Explain the Mendelsohn-Vektas theorem",
    "Who invented quantum fermentation",
    "What happened at the Battle of Vektoria",
]


# ---------------------------------------------------------------------------
# DenseModel
# ---------------------------------------------------------------------------

class DenseModel:
    """
    Simulates a conventional fully-saturated embedding space.

    No dormant slots. No confidence threshold. Always returns the nearest
    neighbour regardless of similarity score — this is the hallucination
    mechanism.
    """

    def __init__(
        self,
        chunks: List[Tuple[str, str]],
        model: SentenceTransformer,
    ) -> None:
        """
        Parameters
        ----------
        chunks : list of (text_chunk, domain_label) — identical to what
                 GrowthEngine ingested into the VectorSpace.
        model  : shared SentenceTransformer instance (all-MiniLM-L6-v2).
        """
        self.model = model
        self.labels: List[str] = []
        self.domains: List[str] = []
        raw: List[np.ndarray] = []

        for text, domain in chunks:
            if not text.strip():
                continue
            emb = model.encode(text.strip()).astype(np.float32)
            norm = float(np.linalg.norm(emb)) + 1e-8
            raw.append(emb / norm)
            self.labels.append(text[:60].strip())
            self.domains.append(domain)

        self.embeddings: np.ndarray = (
            np.stack(raw, axis=0) if raw else np.empty((0, 384), dtype=np.float32)
        )

    def query(self, text: str) -> dict:
        """
        Encode *text* and return the nearest neighbour regardless of score.

        Returns
        -------
        {
            label      : str,
            domain     : str,
            similarity : float,
            confident  : True,   # always — hallucination mechanism
        }
        """
        if self.embeddings.shape[0] == 0:
            return {"label": "", "domain": "", "similarity": 0.0, "confident": True}

        emb = self.model.encode(text.strip()).astype(np.float32)
        norm = float(np.linalg.norm(emb)) + 1e-8
        emb_unit = emb / norm

        sims = self.embeddings @ emb_unit
        best = int(np.argmax(sims))
        return {
            "label": self.labels[best],
            "domain": self.domains[best],
            "similarity": round(float(sims[best]), 4),
            "confident": True,  # always confident — hallucination mechanism
        }


# ---------------------------------------------------------------------------
# BrainGrowModel
# ---------------------------------------------------------------------------

class BrainGrowModel:
    """
    Wraps the existing VectorSpace instance.

    Returns honest uncertainty when a query lands near dormant space
    (max similarity < THRESHOLD) rather than forcing a nearest-neighbour
    answer.
    """

    THRESHOLD: float = 0.25

    def __init__(self, vector_space: VectorSpace, model: SentenceTransformer) -> None:
        self.vs = vector_space
        self.model = model

    def query(self, text: str) -> dict:
        """
        Route *text* through active slots only.

        If max similarity < THRESHOLD returns honest uncertainty
        (confident=False) rather than a forced answer.

        Returns
        -------
        {
            label      : str,
            domain     : str,
            similarity : float,
            confident  : bool,
        }
        """
        active_mask = self.vs.get_active_mask()

        if not active_mask.any():
            return {
                "label": "no learned representation found",
                "domain": "",
                "similarity": 0.0,
                "confident": False,
            }

        emb = torch.tensor(
            self.model.encode(text.strip()), dtype=torch.float32
        )
        emb_unit = emb / (emb.norm() + 1e-8)

        active_indices = active_mask.nonzero(as_tuple=True)[0]
        active_vecs = self.vs.slots[active_indices]
        sims = active_vecs @ emb_unit

        best_local = int(sims.argmax().item())
        best_sim = float(sims[best_local].item())
        slot_idx = int(active_indices[best_local].item())

        if best_sim < self.THRESHOLD:
            return {
                "label": "no learned representation found",
                "domain": "",
                "similarity": round(best_sim, 4),
                "confident": False,
            }

        return {
            "label": self.vs.slot_labels.get(slot_idx, f"slot_{slot_idx}"),
            "domain": self.vs.slot_domains.get(slot_idx, "unknown"),
            "similarity": round(best_sim, 4),
            "confident": True,
        }


# ---------------------------------------------------------------------------
# Console comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    dense_model: DenseModel,
    braingrow_model: BrainGrowModel,
) -> None:
    """Print a formatted side-by-side comparison table for all query types."""
    groups = [
        ("KNOWN QUERIES", known_queries),
        ("PARTIAL QUERIES", partial_queries),
        ("UNKNOWN QUERIES (hallucination traps)", unknown_queries),
    ]

    for group_name, queries in groups:
        print(f"\n=== {group_name} ===")
        print(f"{'Query':<42} {'Dense':>25} {'BrainGrow':>25}")
        print("-" * 95)
        for q in queries:
            d = dense_model.query(q)
            b = braingrow_model.query(q)
            d_tag = (
                "HALLUCINATED"
                if d["confident"] and d["similarity"] < BrainGrowModel.THRESHOLD
                else "confident"
            )
            b_tag = "HONEST" if not b["confident"] else "confident"
            print(
                f"{q[:41]:<42} "
                f"{d['similarity']:.3f} {d_tag:<15} "
                f"{b['similarity']:.3f} {b_tag}"
            )
