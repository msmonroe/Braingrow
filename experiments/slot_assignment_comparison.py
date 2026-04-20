"""
experiments/slot_assignment_comparison.py

Validates the core architectural claim of BrainGrow v2:
that semantic-aware slot assignment produces meaningfully different
(more structured) spatial organization than sequential assignment.

This experiment is referenced in Section 5 of the paper.

Method
------
1. Ingest the same two-domain corpus (science + history) twice:
   - Run A: VectorSpace with semantic-aware assignment (v2, default)
   - Run B: VectorSpace with sequential assignment (v1 baseline)

2. For each run, compute:
   - Intra-domain centroid distance (within each domain)
   - Inter-domain centroid distance (between domains)
   - Domain separability ratio: inter / intra (higher = better separation)
   - Silhouette score over active slot embeddings

3. Print a comparison table.

If the v2 semantic assignment is doing real work, the separability ratio
and silhouette score should be measurably higher than v1 sequential.
If they are identical or v1 wins, the geometry claim must be softened
in the paper.

Usage
-----
    python experiments/slot_assignment_comparison.py

Requires: sentence-transformers, torch, scikit-learn
"""

from __future__ import annotations

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from collections import deque
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import numpy as np

from vector_space import VectorSpace
from utils import encode_unit_torch


# --------------------------------------------------------------------------
# Minimal sequential-assignment VectorSpace (v1 baseline)
# --------------------------------------------------------------------------

class VectorSpaceV1Sequential(VectorSpace):
    """
    Identical to VectorSpace except assign_slot() uses sequential deque
    pop (the v1 behavior). Used as the comparison baseline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Restore sequential deque
        self._seq_deque = deque(range(self.N))

    def assign_slot(self, embedding, label="", domain=""):
        with self._lock:
            emb_unit = embedding / (embedding.norm() + 1e-8)

            if self._seq_deque:
                slot_idx = int(self._seq_deque.popleft())
            else:
                slot_idx = int(self.activation.argmin().item())

            activation_before = float(self.activation[slot_idx].item())
            self.slots[slot_idx] = emb_unit
            self.activation[slot_idx] = 0.5
            self._mark_active(slot_idx)
            if label:
                self.slot_labels[slot_idx] = label
            if domain:
                self.slot_domains[slot_idx] = domain

            return {
                "slot_idx": slot_idx,
                "was_dormant": activation_before == 0.0,
                "activation_before": activation_before,
                "activation_after": float(self.activation[slot_idx].item()),
                "best_similarity": None,
                "candidates_checked": 1,
            }


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

SCIENCE_CHUNKS = [
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "DNA replication occurs during the S phase of the cell cycle.",
    "Newton's third law states that every action has an equal and opposite reaction.",
    "Electrons occupy discrete energy levels called orbitals around the nucleus.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Mitochondria generate ATP through oxidative phosphorylation.",
    "Plate tectonics explains the movement of Earth's lithospheric plates.",
    "Quantum entanglement links particles so that the state of one affects the other.",
    "The Krebs cycle produces NADH and FADH2 used in the electron transport chain.",
    "General relativity describes gravity as the curvature of spacetime.",
    "Enzymes lower the activation energy required for biochemical reactions.",
    "The electromagnetic spectrum includes radio waves, infrared, visible light, and X-rays.",
    "CRISPR-Cas9 enables precise editing of DNA sequences in living organisms.",
    "Thermodynamics governs energy transfer and the direction of chemical reactions.",
    "The Higgs boson gives other particles mass through the Higgs field.",
]

HISTORY_CHUNKS = [
    "The French Revolution began in 1789 with the storming of the Bastille.",
    "Julius Caesar was assassinated on the Ides of March in 44 BC.",
    "The Black Death killed approximately one third of Europe's population in the 14th century.",
    "The Magna Carta of 1215 limited the power of the English monarchy.",
    "The Battle of Hastings in 1066 established Norman rule over England.",
    "The Ottoman Empire lasted from 1299 to 1922 and spanned three continents.",
    "The Treaty of Versailles formally ended World War One in 1919.",
    "Genghis Khan united the Mongol tribes and founded the largest contiguous land empire.",
    "The Renaissance was a cultural movement that began in Italy in the 14th century.",
    "The American Civil War was fought from 1861 to 1865 over slavery and states' rights.",
    "The printing press invented by Gutenberg around 1440 transformed the spread of knowledge.",
    "The fall of Constantinople in 1453 ended the Byzantine Empire.",
    "The Industrial Revolution began in Britain in the late 18th century.",
    "Napoleon was exiled to Saint Helena after his defeat at Waterloo in 1815.",
    "The Cold War between the United States and Soviet Union lasted from 1947 to 1991.",
]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(vs: VectorSpace) -> dict:
    """Compute separability metrics over active slots."""
    domain_reg = vs.domain_registry
    domains = list(domain_reg.keys())

    if len(domains) < 2:
        return {"error": "Need at least 2 domains"}

    all_vecs = []
    all_labels = []

    for domain_idx, domain in enumerate(domains):
        indices = domain_reg[domain]
        vecs = vs.slots[indices].detach().numpy()
        all_vecs.append(vecs)
        all_labels.extend([domain_idx] * len(indices))

    all_vecs_np = np.vstack(all_vecs)
    all_labels_np = np.array(all_labels)

    # Centroids per domain
    centroids = {}
    for domain_idx, domain in enumerate(domains):
        mask = all_labels_np == domain_idx
        centroids[domain] = all_vecs_np[mask].mean(axis=0)

    # Intra-domain average distance (lower = tighter clusters)
    intra_distances = []
    for domain_idx, domain in enumerate(domains):
        mask = all_labels_np == domain_idx
        vecs = all_vecs_np[mask]
        c = centroids[domain]
        dists = np.linalg.norm(vecs - c, axis=1)
        intra_distances.append(dists.mean())
    avg_intra = np.mean(intra_distances)

    # Inter-domain centroid distance (higher = better separation)
    c_list = [centroids[d] for d in domains]
    inter_dist = np.linalg.norm(c_list[0] - c_list[1])

    separability_ratio = inter_dist / (avg_intra + 1e-8)

    # Silhouette score
    if len(set(all_labels)) > 1 and len(all_vecs_np) > 2:
        sil = silhouette_score(all_vecs_np, all_labels_np, metric='cosine')
    else:
        sil = None

    return {
        "avg_intra_distance": round(float(avg_intra), 4),
        "inter_centroid_distance": round(float(inter_dist), 4),
        "separability_ratio": round(float(separability_ratio), 4),
        "silhouette_score": round(float(sil), 4) if sil is not None else None,
        "n_active": vs.n_active,
    }


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def run_experiment(
    vs_class,
    model: SentenceTransformer,
    label: str,
    n_slots: int = 5000,  # smaller for fast experiment
) -> dict:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    vs = vs_class(n_slots=n_slots)

    for chunk in SCIENCE_CHUNKS:
        emb = encode_unit_torch(model, chunk)
        vs.assign_slot(emb, label=chunk[:40], domain="science")

    for chunk in HISTORY_CHUNKS:
        emb = encode_unit_torch(model, chunk)
        vs.assign_slot(emb, label=chunk[:40], domain="history")

    metrics = compute_metrics(vs)
    print(f"  Active slots      : {metrics['n_active']}")
    print(f"  Avg intra-distance: {metrics['avg_intra_distance']}")
    print(f"  Inter-centroid    : {metrics['inter_centroid_distance']}")
    print(f"  Separability ratio: {metrics['separability_ratio']}")
    print(f"  Silhouette score  : {metrics['silhouette_score']}")

    return metrics


def main():
    print("Loading sentence encoder...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Run both variants
    v2_metrics = run_experiment(
        VectorSpace,
        model,
        "v2 — Semantic-Aware Slot Assignment (BrainGrow)"
    )
    v1_metrics = run_experiment(
        VectorSpaceV1Sequential,
        model,
        "v1 — Sequential Slot Assignment (Baseline)"
    )

    # Summary
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'v2 Semantic':>14} {'v1 Sequential':>14}")
    print(f"  {'-'*58}")
    for key in ["avg_intra_distance", "inter_centroid_distance",
                "separability_ratio", "silhouette_score"]:
        v2_val = v2_metrics.get(key, "N/A")
        v1_val = v1_metrics.get(key, "N/A")
        label = key.replace("_", " ").title()
        print(f"  {label:<30} {str(v2_val):>14} {str(v1_val):>14}")

    print()
    if (v2_metrics.get("separability_ratio", 0) >
            v1_metrics.get("separability_ratio", 0)):
        print("  RESULT: Semantic assignment produces better domain separation.")
        print("  The geometry claim is SUPPORTED by this experiment.")
    else:
        print("  RESULT: Sequential and semantic assignment produce similar separation.")
        print("  The geometry claim should be SOFTENED in the paper.")
        print("  The encoder is doing most of the structural work.")
    print()


if __name__ == "__main__":
    main()
