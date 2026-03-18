"""
vector_space.py — Core data structure for BrainGrow.

Pre-allocates N=10,000 vector slots of D=384 dimensions (matching
all-MiniLM-L6-v2).  Each slot begins life as a random unit vector
(dormant, activation=0).  Knowledge "grows into" dormant regions via
assign_slot().  Active slots can be reinforced on query hits and decayed
or pruned over time.
"""

from __future__ import annotations
from typing import Dict

import torch


class VectorSpace:
    N: int = 10_000   # total pre-allocated slots
    D: int = 384      # embedding dimension (all-MiniLM-L6-v2)

    REINFORCE_STEP: float = 0.10   # activation gain per query hit
    DECAY_RATE: float = 0.005      # activation loss per decay() call

    # --------------------------------------------------------------------------
    def __init__(self) -> None:
        # Pre-allocate and normalise to unit sphere so cosine sim == dot product
        raw = torch.randn(self.N, self.D)
        self.slots: torch.Tensor = raw / raw.norm(dim=1, keepdim=True)

        # 0.0 = dormant, 0.0 < x <= 1.0 = active
        self.activation: torch.Tensor = torch.zeros(self.N)

        # Human-readable metadata per slot
        self.slot_labels: Dict[int, str] = {}
        self.slot_domains: Dict[int, str] = {}

        self._step: int = 0  # internal tick for future use

    # --------------------------------------------------------------------------
    # Slot assignment
    # --------------------------------------------------------------------------
    def assign_slot(
        self,
        embedding: torch.Tensor,
        label: str = "",
        domain: str = "",
    ) -> dict:
        """
        Find the nearest dormant slot to *embedding*, store the embedding
        there, activate it, and return metadata.

        Returns
        -------
        {
            slot_idx       : int,
            was_dormant    : bool,
            activation_before : float,
            activation_after  : float,
        }
        """
        dormant_mask: torch.Tensor = self.activation == 0.0

        # Fallback: if the space is nearly full, reuse the least-active slot
        if not dormant_mask.any():
            dormant_mask = self.activation <= self.activation[self.activation > 0].min()

        dormant_indices = dormant_mask.nonzero(as_tuple=True)[0]
        dormant_vecs = self.slots[dormant_indices]  # [M, D]

        emb_unit = embedding / (embedding.norm() + 1e-8)  # unit vector
        sims = dormant_vecs @ emb_unit                    # [M]

        best_local = int(sims.argmax().item())
        slot_idx = int(dormant_indices[best_local].item())

        activation_before = float(self.activation[slot_idx].item())
        self.slots[slot_idx] = emb_unit
        self.activation[slot_idx] = 0.5  # initial activation on arrival

        if label:
            self.slot_labels[slot_idx] = label
        if domain:
            self.slot_domains[slot_idx] = domain

        return {
            "slot_idx": slot_idx,
            "was_dormant": activation_before == 0.0,
            "activation_before": activation_before,
            "activation_after": float(self.activation[slot_idx].item()),
        }

    # --------------------------------------------------------------------------
    # Reinforcement, decay, pruning
    # --------------------------------------------------------------------------
    def reinforce(self, slot_idx: int) -> None:
        """Increment activation toward 1.0 on a query hit."""
        new_val = min(1.0, float(self.activation[slot_idx].item()) + self.REINFORCE_STEP)
        self.activation[slot_idx] = new_val

    def decay(self) -> None:
        """Reduce activation of all active slots by DECAY_RATE (one step)."""
        self._step += 1
        active = self.activation > 0.0
        self.activation[active] = torch.clamp(
            self.activation[active] - self.DECAY_RATE,
            min=0.0,
        )

    def prune(self, threshold: float = 0.2) -> dict:
        """
        Zero out every active slot whose activation is below *threshold*.
        Returns a summary dict.
        """
        before_active = int((self.activation > 0).sum().item())
        prune_mask = (self.activation > 0) & (self.activation < threshold)
        pruned_indices = prune_mask.nonzero(as_tuple=True)[0].tolist()

        self.activation[prune_mask] = 0.0
        # Reset slot vector to random unit vector so it can be re-claimed
        if pruned_indices:
            raw = torch.randn(len(pruned_indices), self.D)
            self.slots[pruned_indices] = raw / raw.norm(dim=1, keepdim=True)
            for idx in pruned_indices:
                self.slot_labels.pop(idx, None)
                self.slot_domains.pop(idx, None)

        after_active = int((self.activation > 0).sum().item())
        return {
            "pruned_count": len(pruned_indices),
            "before_active": before_active,
            "after_active": after_active,
        }

    # --------------------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------------------
    def get_active_mask(self) -> torch.Tensor:
        """Boolean tensor [N] — True for active slots."""
        return self.activation > 0.0

    def reset(self) -> None:
        """Restore the vector space to its initial state."""
        raw = torch.randn(self.N, self.D)
        self.slots = raw / raw.norm(dim=1, keepdim=True)
        self.activation = torch.zeros(self.N)
        self.slot_labels = {}
        self.slot_domains = {}
        self._step = 0
