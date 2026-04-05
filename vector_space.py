"""
vector_space.py — Core data structure for BrainGrow.

Pre-allocates N=200,000 vector slots of D=384 dimensions (matching
all-MiniLM-L6-v2). Each slot begins life as a random unit vector
(dormant, activation=0). Knowledge "grows into" dormant regions via
assign_slot(). Active slots can be reinforced on query hits and decayed
or pruned over time.

FAISS indexing added for O(log n) retrieval at scale.
Falls back to brute-force cosine similarity if faiss is not installed,
so the repo remains runnable on CPU-only machines without faiss.

N increased to 200,000 for TinyStories scale experiment.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Optional FAISS import — graceful fallback to brute-force if not available
# ---------------------------------------------------------------------------
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


class VectorSpace:

    N: int = 200_000   # total pre-allocated slots (scaled for TinyStories)
    D: int = 384        # embedding dimension (all-MiniLM-L6-v2)

    REINFORCE_STEP: float = 0.10   # activation gain per query hit
    DECAY_RATE:     float = 0.005  # activation loss per decay() call

    # FAISS index type thresholds
    _FAISS_FLAT_MAX:    int = 500_000   # use IndexFlatIP below this
    _FAISS_IVF_NLIST:   int = 256       # IVF cluster count above flat threshold

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

        # 0.0 = dormant,  0.0 < x <= 1.0 = active
        self.activation: torch.Tensor = torch.zeros(self.N)

        # Human-readable metadata per slot
        self.slot_labels:  Dict[int, str] = {}
        self.slot_domains: Dict[int, str] = {}

        self._step:        int = 0   # internal tick for future use
        self.stage_number: int = 0   # synced from GrowthEngine on each stage

        # Protects all mutations so Gradio concurrent callbacks don't race.
        self._lock: threading.RLock = threading.RLock()

        # O(1) dormant-slot allocation — deque is replenished by prune()
        self.dormant_queue: deque = deque(range(self.N))

        # Domains registered as negative (boundary-violation detection)
        self.negative_domains: set = set()

        # ------------------------------------------------------------------
        # FAISS index — built lazily on first query, rebuilt after prune()
        # ------------------------------------------------------------------
        self._faiss_index = None          # faiss index object or None
        self._faiss_slot_map: List[int] = []  # position i → global slot_idx
        self._faiss_dirty: bool = False   # True when index needs rebuild

    # --------------------------------------------------------------------------
    # FAISS index management
    # --------------------------------------------------------------------------

    @property
    def _use_faiss(self) -> bool:
        """True when faiss is available and there are active slots to index."""
        return _FAISS_AVAILABLE and self.n_active > 0

    def _build_faiss_index(self) -> None:
        """
        (Re)build the FAISS index from all currently active slots.
        Uses IndexFlatIP (exact inner-product search) for slot counts
        below _FAISS_FLAT_MAX, IndexIVFFlat above.

        Called automatically before the first query after any mutation
        that sets _faiss_dirty = True.
        """
        if not _FAISS_AVAILABLE:
            return

        active_mask = self.activation > 0.0
        active_indices = active_mask.nonzero(as_tuple=True)[0].tolist()

        if not active_indices:
            self._faiss_index = None
            self._faiss_slot_map = []
            self._faiss_dirty = False
            return

        # Build float32 numpy matrix of active slot vectors
        vecs = self.slots[active_indices].numpy().astype(np.float32)

        n = len(active_indices)

        if n < self._FAISS_FLAT_MAX:
            # Exact search — no approximation, no training required
            index_cpu = faiss.IndexFlatIP(self.D)
        else:
            # Approximate search — faster at large scale, slight accuracy tradeoff
            quantizer = faiss.IndexFlatIP(self.D)
            nlist = min(self._FAISS_IVF_NLIST, n // 10 or 1)
            index_cpu = faiss.IndexIVFFlat(quantizer, self.D, nlist, faiss.METRIC_INNER_PRODUCT)
            index_cpu.train(vecs)

        index_cpu.add(vecs)

        # Move to GPU if available
        try:
            res = faiss.StandardGpuResources()
            self._faiss_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        except (AttributeError, RuntimeError):
            # faiss-cpu build — no GPU transfer available
            self._faiss_index = index_cpu

        self._faiss_slot_map = active_indices
        self._faiss_dirty = False

    def _faiss_search(
        self,
        query_vec: torch.Tensor,
        top_k: int,
    ) -> Tuple[List[float], List[int]]:
        """
        Search the FAISS index for top_k nearest active slots.

        Returns
        -------
        (similarities, global_slot_indices)
        """
        if self._faiss_dirty or self._faiss_index is None:
            self._build_faiss_index()

        if self._faiss_index is None:
            return [], []

        q = query_vec.numpy().astype(np.float32).reshape(1, -1)
        k = min(top_k, len(self._faiss_slot_map))
        distances, local_indices = self._faiss_index.search(q, k)

        sims:    List[float] = distances[0].tolist()
        globals: List[int]   = [
            self._faiss_slot_map[i]
            for i in local_indices[0]
            if i >= 0  # FAISS returns -1 for empty results
        ]
        return sims, globals

    def search_active(
        self,
        query_vec: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[List[float], List[int]]:
        """
        Public search interface used by QueryRouter.

        Returns (similarities, global_slot_indices) for the top_k active
        slots nearest to query_vec.

        Uses FAISS when available; falls back to brute-force otherwise.
        """
        with self._lock:
            if self._use_faiss:
                return self._faiss_search(query_vec, top_k)
            else:
                return self._brute_force_search(query_vec, top_k)

    def _brute_force_search(
        self,
        query_vec: torch.Tensor,
        top_k: int,
    ) -> Tuple[List[float], List[int]]:
        """O(n) cosine similarity fallback — used when faiss is not installed."""
        active_mask = self.activation > 0.0
        active_indices = active_mask.nonzero(as_tuple=True)[0]

        if len(active_indices) == 0:
            return [], []

        active_vecs = self.slots[active_indices]
        sims = active_vecs @ query_vec
        k = min(top_k, len(active_indices))
        top_vals, top_local = sims.topk(k)

        globals = [int(active_indices[i].item()) for i in top_local.tolist()]
        return top_vals.tolist(), globals

    # --------------------------------------------------------------------------
    # Slot assignment
    # --------------------------------------------------------------------------

    def assign_slot(
        self,
        embedding: torch.Tensor,
        label:  str = "",
        domain: str = "",
    ) -> dict:
        """
        Find the nearest dormant slot to *embedding*, store the embedding
        there, activate it, and return metadata.

        Marks the FAISS index dirty so it will be rebuilt before the next query.

        Returns
        -------
        {
            slot_idx         : int,
            was_dormant      : bool,
            activation_before: float,
            activation_after : float,
        }
        """
        with self._lock:
            emb_unit = embedding / (embedding.norm() + 1e-8)

            if self.dormant_queue:
                slot_idx = int(self.dormant_queue.popleft())
            else:
                slot_idx = int(self.activation.argmin().item())

            activation_before = float(self.activation[slot_idx].item())

            self.slots[slot_idx]      = emb_unit
            self.activation[slot_idx] = 0.5

            if label:
                self.slot_labels[slot_idx]  = label
            if domain:
                self.slot_domains[slot_idx] = domain

            self._faiss_dirty = True   # index must be rebuilt before next query

            return {
                "slot_idx":          slot_idx,
                "was_dormant":       activation_before == 0.0,
                "activation_before": activation_before,
                "activation_after":  float(self.activation[slot_idx].item()),
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
        Rebuilds FAISS index after pruning.

        Returns a summary dict.
        """
        with self._lock:
            before_active = int((self.activation > 0).sum().item())

            prune_mask    = (self.activation > 0) & (self.activation < threshold)
            pruned_indices = prune_mask.nonzero(as_tuple=True)[0].tolist()

            self.activation[prune_mask] = 0.0

            if pruned_indices:
                raw = torch.randn(len(pruned_indices), self.D)
                self.slots[pruned_indices] = raw / raw.norm(dim=1, keepdim=True)
                for idx in pruned_indices:
                    self.slot_labels.pop(idx, None)
                    self.slot_domains.pop(idx, None)
                    self.dormant_queue.append(idx)

            after_active = int((self.activation > 0).sum().item())

            # Rebuild FAISS index now — prune changes the active set significantly
            self._faiss_dirty = True
            self._build_faiss_index()

            return {
                "pruned_count":  len(pruned_indices),
                "before_active": before_active,
                "after_active":  after_active,
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

    @property
    def faiss_available(self) -> bool:
        """True if faiss is installed and the GPU index is active."""
        return _FAISS_AVAILABLE and self._faiss_index is not None

    def reset(self) -> None:
        """Restore the vector space to its initial state."""
        with self._lock:
            raw = torch.randn(self.N, self.D)
            self.slots      = raw / raw.norm(dim=1, keepdim=True)
            self.activation = torch.zeros(self.N)
            self.slot_labels  = {}
            self.slot_domains = {}
            self._step        = 0
            self.stage_number = 0
            self.dormant_queue    = deque(range(self.N))
            self.negative_domains = set()
            self._faiss_index     = None
            self._faiss_slot_map  = []
            self._faiss_dirty     = False

    # --------------------------------------------------------------------------
    # Persistence — save / load / autosave
    # --------------------------------------------------------------------------

    def save(self, path: str, description: str = '') -> str:
        """
        Persist the full network state to *path* using torch.save.
        Files use the .bgstate extension.
        Returns the path that was written.
        Note: FAISS index is NOT persisted — it is rebuilt on first query after load.
        """
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            'slots':           self.slots,
            'activation':      self.activation,
            'slot_labels':     self.slot_labels,
            'n_active':        self.n_active,
            'stage_number':    self.stage_number,
            'domain_registry': self.domain_registry,
            'negative_domains': self.negative_domains,
            'metadata': {
                'saved_at':   datetime.now().isoformat(),
                'n_slots':    self.slots.shape[0],
                'dimensions': self.slots.shape[1],
                'version':    '1.1',
                'description': description,
                'faiss_available': _FAISS_AVAILABLE,
            },
        }
        torch.save(state, path)
        return path

    @classmethod
    def load(cls, path: str):
        """
        Reconstruct a VectorSpace from a .bgstate file.
        FAISS index will be rebuilt lazily on first query.

        Returns
        -------
        (VectorSpace, metadata_dict)
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No such .bgstate file: {path}")

        state = torch.load(path, map_location='cpu', weights_only=False)
        meta  = state['metadata']

        vs = cls(
            n_slots    = meta['n_slots'],
            dimensions = meta['dimensions'],
        )
        vs.slots        = state['slots']
        vs.activation   = state['activation']
        vs.slot_labels  = state['slot_labels']
        vs.stage_number = state.get('stage_number', 0)

        domain_registry = state.get('domain_registry', {})
        vs.slot_domains = {}
        for domain, indices in domain_registry.items():
            for idx in indices:
                vs.slot_domains[int(idx)] = domain

        vs.negative_domains = set(state.get('negative_domains', set()))

        vs.dormant_queue = deque(
            i for i, v in enumerate(vs.activation.tolist()) if v == 0.0
        )

        # FAISS index will be built lazily on first query after load
        vs._faiss_dirty = True

        return vs, meta

    def autosave(self, saves_dir: str = 'saves') -> str:
        """Write a timestamped autosave to *saves_dir* and return the path."""
        os.makedirs(saves_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(saves_dir, f'autosave_{timestamp}.bgstate')
        return self.save(path, description='autosave')
