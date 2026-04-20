"""
comparison_harness.py — Hallucination comparison harness for BrainGrow Tab 4.

Provides DenseModel (always-confident nearest-neighbour) and BrainGrowModel
(honest uncertainty via dormant space) running on identical training data.

Demonstrates that hallucination is an architectural property of a saturated
vector space — not a scale or data-quantity problem.

v2 changes
----------
- BrainGrowModel.THRESHOLD raised from 0.40 → 0.60 (consistent with
  epistemic.py DEFAULT_CONFIDENCE_THRESHOLD). The prior value of 0.40
  caused fabricated concepts with weak partial matches (e.g. "quantum
  fermentation" → fermentation chunk, sim=0.4035) to be incorrectly
  classified as Confident. This was the primary failure mode the
  architecture is designed to prevent.
- BrainGrowModel now delegates verdict classification to epistemic.classify()
  for consistency across all code paths.
- THRESHOLD exposed as a constructor parameter for UI configurability.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from epistemic import classify, DEFAULT_CONFIDENCE_THRESHOLD, EpistemicState
from utils import encode_unit_numpy, encode_unit_torch
from vector_space import VectorSpace

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------------
# Predefined query sets
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
    mechanism. A saturated store has nowhere to abstain to.
    """

    def __init__(
        self,
        chunks: List[Tuple[str, str]],
        model: SentenceTransformer,
    ) -> None:
        self.model = model
        self.labels: List[str] = []
        self.domains: List[str] = []

        valid = [(t, d) for t, d in chunks if t.strip()]
        if valid:
            texts = [t for t, _ in valid]
            self.labels = [t[:60].strip() for t in texts]
            self.domains = [d for _, d in valid]
            embs_raw = model.encode(
                texts, device=_DEVICE, batch_size=512, convert_to_numpy=True,
            )
            norms = np.linalg.norm(embs_raw, axis=1, keepdims=True)
            self.embeddings: np.ndarray = (embs_raw / (norms + 1e-8)).astype(np.float32)
        else:
            self.embeddings = np.empty((0, 384), dtype=np.float32)

    def query(self, text: str) -> dict:
        """
        Encode *text* and return the nearest neighbour regardless of score.
        Always confident — this is the hallucination mechanism.
        """
        if self.embeddings.shape[0] == 0:
            return {"label": "", "domain": "", "similarity": 0.0, "confident": True}

        emb_unit = encode_unit_numpy(self.model, text)
        sims = self.embeddings @ emb_unit
        best = int(np.argmax(sims))

        return {
            "label": self.labels[best],
            "domain": self.domains[best],
            "similarity": round(float(sims[best]), 4),
            "confident": True,
        }

    def add_chunks(self, new_chunks: List[Tuple[str, str]]) -> None:
        """Encode and append *new_chunks* without re-encoding historical data."""
        valid = [(t, d) for t, d in new_chunks if t.strip()]
        if not valid:
            return
        texts = [t for t, _ in valid]
        embs_raw = self.model.encode(
            texts, device=_DEVICE, batch_size=512, convert_to_numpy=True
        )
        norms = np.linalg.norm(embs_raw, axis=1, keepdims=True)
        new_embs = (embs_raw / (norms + 1e-8)).astype(np.float32)
        self.embeddings = (
            np.vstack([self.embeddings, new_embs])
            if self.embeddings.shape[0] > 0
            else new_embs
        )
        self.labels.extend([t[:60].strip() for t in texts])
        self.domains.extend([d for _, d in valid])


# ---------------------------------------------------------------------------
# BrainGrowModel
# ---------------------------------------------------------------------------

class BrainGrowModel:
    """
    Wraps the existing VectorSpace instance.

    Routes queries exclusively through active slots. Delegates epistemic
    classification to epistemic.classify(), which enforces the three-tier
    state: CONFIDENT / HONEST_UNKNOWN / OUT_OF_BOUNDS.

    The confidence threshold (default 0.60) is consistent with
    epistemic.DEFAULT_CONFIDENCE_THRESHOLD. This threshold was chosen
    empirically: all-MiniLM-L6-v2 cosine similarities above 0.60 indicate
    strong semantic overlap; values below 0.40 are effectively orthogonal.
    The 0.40–0.60 band is treated conservatively as HONEST_UNKNOWN to
    prevent partial-match false positives (e.g. "quantum fermentation"
    partially matching a fermentation chunk at sim=0.40).
    """

    def __init__(
        self,
        vector_space: VectorSpace,
        model: SentenceTransformer,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.vs = vector_space
        self.model = model
        self.confidence_threshold = confidence_threshold

    def query(self, text: str) -> dict:
        """
        Route *text* through active slots and classify epistemically.

        Returns
        -------
        {
            label      : str,
            domain     : str,
            similarity : float,
            confident  : bool,
            verdict    : str,   # human-readable: "✓ Confident" / "HONEST (uncertain)" / "⚠️ BOUNDARY VIOLATION"
            state      : str,   # EpistemicState enum value for programmatic use
        }
        """
        active_mask = self.vs.get_active_mask()
        active_count = int(active_mask.sum().item())
        dormant_count = int((~active_mask).sum().item())

        if active_count == 0:
            return {
                "label": "no learned representation found",
                "domain": "",
                "similarity": 0.0,
                "confident": False,
                "verdict": "HONEST (uncertain)",
                "state": EpistemicState.HONEST_UNKNOWN.value,
            }

        emb_unit = encode_unit_torch(self.model, text)
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        active_vecs = self.vs.slots[active_indices]
        sims = active_vecs @ emb_unit

        top_k = min(5, active_count)
        top_vals, top_local = sims.topk(top_k)

        matches = []
        for val, local_idx in zip(top_vals.tolist(), top_local.tolist()):
            slot_idx = int(active_indices[local_idx].item())
            matches.append({
                "slot_idx": slot_idx,
                "label": self.vs.slot_labels.get(slot_idx, f"slot_{slot_idx}"),
                "domain": self.vs.slot_domains.get(slot_idx, "unknown"),
                "similarity": round(float(val), 4),
                "activation": round(float(self.vs.activation[slot_idx].item()), 4),
            })

        nearest_domain = matches[0]["domain"] if matches else ""
        boundary_violation = nearest_domain in self.vs.negative_domains

        router_result = {
            "matches": matches,
            "active_count": active_count,
            "dormant_count": dormant_count,
            "boundary_violation": boundary_violation,
            "nearest_domain": nearest_domain,
        }

        epistemic = classify(router_result, confidence_threshold=self.confidence_threshold)

        if epistemic.state == EpistemicState.CONFIDENT:
            verdict = "✓ Confident"
            label = matches[0]["label"] if matches else ""
            confident = True
        elif epistemic.state == EpistemicState.OUT_OF_BOUNDS:
            verdict = "⚠️ BOUNDARY VIOLATION"
            label = matches[0]["label"] if matches else ""
            confident = True
        else:
            verdict = "HONEST (uncertain)"
            label = "no learned representation found"
            confident = False

        return {
            "label": label,
            "domain": nearest_domain,
            "similarity": matches[0]["similarity"] if matches else 0.0,
            "confident": confident,
            "verdict": verdict,
            "state": epistemic.state.value,
        }


# ---------------------------------------------------------------------------
# Console comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    dense_model: DenseModel,
    braingrow_model: BrainGrowModel,
) -> None:
    """Print a formatted side-by-side comparison table for all query types."""
    threshold = braingrow_model.confidence_threshold

    groups = [
        ("KNOWN QUERIES", known_queries),
        ("PARTIAL QUERIES", partial_queries),
        ("UNKNOWN QUERIES (hallucination traps)", unknown_queries),
    ]

    for group_name, queries in groups:
        print(f"\n=== {group_name} ===")
        print(f"{'Query':<42} {'Dense':>25} {'BrainGrow':>30}")
        print("-" * 100)
        for q in queries:
            d = dense_model.query(q)
            b = braingrow_model.query(q)

            # Dense is hallucinating if it returns confident on a low-sim result
            d_tag = (
                "HALLUCINATED"
                if d["similarity"] < threshold
                else "confident"
            )
            b_tag = b["verdict"]

            print(
                f"{q[:41]:<42} "
                f"{d['similarity']:.4f} {d_tag:<15} "
                f"{b['similarity']:.4f} {b_tag}"
            )
