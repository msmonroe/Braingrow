"""
vector_space.py — Core data structure for BrainGrow.

Pre-allocates N=200,000 vector slots of D=384 dimensions (matching
all-MiniLM-L6-v2).  Each slot begins life as a random unit vector
(dormant, activation=0).  Knowledge "grows into" dormant regions via
assign_slot().  Active slots can be reinforced on query hits and decayed
or pruned over time.

N increased to 200,000 for TinyStories scale experiment.
"""

from __future__ import annotations
import os
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List

import torch


class VectorSpace:
    N: int = 200_000  # total pre-allocated slots (scaled for TinyStories)
    D: int = 384      # embedding dimension (all-MiniLM-L6-v2)

    REINFORCE_STEP: float = 0.10   # activation gain per query hit
    DECAY_RATE: float = 0.005      # activation loss per decay() call

    # --------------------------------------------------------------------------
    def __init__(
        self,
        n_slots: int | None = None,
        dimensions: int | None = None,
    ) -> None:
        if n_slots is not None:
            self.N = n_slots
        if dimensions is not None:
            self.D = dimensions

        # Pre-allocate and normalise to unit sphere so cosine sim == dot product
        raw = torch.randn(self.N, self.D)
        self.slots: torch.Tensor = raw / raw.norm(dim=1, keepdim=True)

        # 0.0 = dormant, 0.0 < x <= 1.0 = active
        self.activation: torch.Tensor = torch.zeros(self.N)

        # Human-readable metadata per slot
        self.slot_labels: Dict[int, str] = {}
        self.slot_domains: Dict[int, str] = {}

        self._step: int = 0  # internal tick for future use
        self.stage_number: int = 0  # synced from GrowthEngine on each stage
        # Protects all mutations so Gradio concurrent callbacks don't race.
        self._lock: threading.RLock = threading.RLock()
        # O(1) dormant-slot allocation — deque is replenished by prune()
        self.dormant_queue: deque = deque(range(self.N))
        # Domains registered as negative (boundary-violation detection)
        self.negative_domains: set = set()

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
        with self._lock:
            emb_unit = embedding / (embedding.norm() + 1e-8)  # unit vector

            if self.dormant_queue:
                # O(1) — take next pre-tracked dormant slot from the front
                slot_idx = int(self.dormant_queue.popleft())
            else:
                # Fallback: reuse the least-active slot when queue is exhausted
                slot_idx = int(self.activation.argmin().item())

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
        with self._lock:
            new_val = min(1.0, float(self.activation[slot_idx].item()) + self.REINFORCE_STEP)
            self.activation[slot_idx] = new_val

    def decay(self) -> None:
        """Reduce activation of all active slots by DECAY_RATE (one step)."""
        with self._lock:
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
        with self._lock:
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
                    self.dormant_queue.append(idx)  # slot is dormant again

            after_active = int((self.activation > 0).sum().item())
            return {
                "pruned_count": len(pruned_indices),
                "before_active": before_active,
                "after_active": after_active,
            }

    # --------------------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------------------
    def register_negative_domain(self, label: str) -> None:
        """Mark *label* as a negative domain (triggers boundary violation on query)."""
        with self._lock:
            self.negative_domains.add(label)

    def get_active_mask(self) -> torch.Tensor:
        """Boolean tensor [N] — True for active slots."""
        return self.activation > 0.0

    @property
    def n_active(self) -> int:
        """Count of currently active slots."""
        return int((self.activation > 0).sum().item())

    @property
    def domain_registry(self) -> Dict[str, List[int]]:
        """Inverse of slot_domains: maps domain name → list of slot indices."""
        result: Dict[str, List[int]] = {}
        for slot_idx, domain in self.slot_domains.items():
            result.setdefault(domain, []).append(slot_idx)
        return result

    def reset(self) -> None:
        """Restore the vector space to its initial state."""
        with self._lock:
            raw = torch.randn(self.N, self.D)
            self.slots = raw / raw.norm(dim=1, keepdim=True)
            self.activation = torch.zeros(self.N)
            self.slot_labels = {}
            self.slot_domains = {}
            self._step = 0
            self.stage_number = 0
            self.dormant_queue = deque(range(self.N))
            self.negative_domains = set()

    # --------------------------------------------------------------------------
    # Persistence — save / load / autosave
    # --------------------------------------------------------------------------
    def save(self, path: str, description: str = '') -> str:
        """
        Persist the full network state to *path* using torch.save.
        Files use the .bgstate extension.
        Returns the path that was written.
        """
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            'slots':            self.slots,
            'activation':       self.activation,
            'slot_labels':      self.slot_labels,
            'n_active':         self.n_active,
            'stage_number':     self.stage_number,
            'domain_registry':  self.domain_registry,
            'negative_domains': self.negative_domains,
            'metadata': {
                'saved_at':    datetime.now().isoformat(),
                'n_slots':     self.slots.shape[0],
                'dimensions':  self.slots.shape[1],
                'version':     '1.0',
                'description': description,
            },
        }
        torch.save(state, path)
        return path

    @classmethod
    def load(cls, path: str):
        """
        Reconstruct a VectorSpace from a .bgstate file.

        Returns
        -------
        (VectorSpace, metadata_dict)
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No such .bgstate file: {path}")
        # weights_only=False required to deserialize metadata dicts
        # and non-tensor state (slot_labels, domain_registry, etc.)
        # Safe for .bgstate files created by this system on this machine.
        # Do not load .bgstate files received from untrusted external sources.
        state = torch.load(path, map_location='cpu', weights_only=False)
        meta = state['metadata']

        vs = cls(
            n_slots=meta['n_slots'],
            dimensions=meta['dimensions'],
        )
        vs.slots       = state['slots']
        vs.activation  = state['activation']
        vs.slot_labels = state['slot_labels']
        vs.stage_number = state.get('stage_number', 0)

        # Reconstruct slot_domains from the saved domain_registry
        domain_registry = state.get('domain_registry', {})
        vs.slot_domains = {}
        for domain, indices in domain_registry.items():
            for idx in indices:
                vs.slot_domains[int(idx)] = domain

        # Restore negative-domain registry
        vs.negative_domains = set(state.get('negative_domains', set()))

        # Rebuild dormant_queue from activation (slots with activation == 0)
        vs.dormant_queue = deque(
            i for i, v in enumerate(vs.activation.tolist()) if v == 0.0
        )

        return vs, meta

    def autosave(self, saves_dir: str = 'saves') -> str:
        """Write a timestamped autosave to *saves_dir* and return the path."""
        os.makedirs(saves_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(saves_dir, f'autosave_{timestamp}.bgstate')
        return self.save(path, description='autosave')
