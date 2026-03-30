"""
test_vector_space.py — Unit tests for VectorSpace.

Covers: init, assign_slot, reinforce, decay, prune,
        get_active_mask, n_active, domain_registry, reset,
        save / load / autosave.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from vector_space import VectorSpace

N_SLOTS = 50
DIMS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(n: int, dim: int = 8) -> torch.Tensor:
    """Return a deterministic unit vector via seeded randn."""
    gen = torch.Generator()
    gen.manual_seed(n)
    v = torch.randn(dim, generator=gen)
    return v / (v.norm() + 1e-8)


# ===========================================================================
# Initialisation
# ===========================================================================

class TestInit:
    def test_default_class_constants(self):
        assert VectorSpace.N == 200_000
        assert VectorSpace.D == 384

    def test_custom_dimensions(self):
        vs = VectorSpace(n_slots=50, dimensions=8)
        assert vs.slots.shape == (50, 8)
        assert vs.activation.shape == (50,)

    def test_slots_are_unit_vectors(self):
        vs = VectorSpace(n_slots=20, dimensions=8)
        norms = vs.slots.norm(dim=1)
        assert torch.allclose(norms, torch.ones(20), atol=1e-5)

    def test_all_slots_dormant_on_init(self):
        vs = VectorSpace(n_slots=20, dimensions=8)
        assert vs.activation.sum() == 0.0

    def test_slot_labels_empty_on_init(self):
        vs = VectorSpace(n_slots=20, dimensions=8)
        assert vs.slot_labels == {}

    def test_slot_domains_empty_on_init(self):
        vs = VectorSpace(n_slots=20, dimensions=8)
        assert vs.slot_domains == {}

    def test_stage_number_zero_on_init(self):
        vs = VectorSpace(n_slots=20, dimensions=8)
        assert vs.stage_number == 0


# ===========================================================================
# assign_slot
# ===========================================================================

class TestAssignSlot:
    def test_assign_activates_dormant_slot(self, vs):
        emb = _unit_vec(1)
        result = vs.assign_slot(emb, label="test", domain="science")
        assert result["was_dormant"] is True
        assert result["activation_after"] == 0.5

    def test_assign_stores_label(self, vs):
        emb = _unit_vec(2)
        result = vs.assign_slot(emb, label="my-label", domain="science")
        assert vs.slot_labels[result["slot_idx"]] == "my-label"

    def test_assign_stores_domain(self, vs):
        emb = _unit_vec(3)
        result = vs.assign_slot(emb, label="x", domain="history")
        assert vs.slot_domains[result["slot_idx"]] == "history"

    def test_assign_multiple_slots_increases_n_active(self, vs):
        for i in range(5):
            vs.assign_slot(_unit_vec(i + 10), label=f"c{i}", domain="d")
        assert vs.n_active == 5

    def test_assign_stores_embedding_as_unit_vector(self, vs):
        emb = _unit_vec(7) * 3.0  # not unit length
        result = vs.assign_slot(emb)
        stored = vs.slots[result["slot_idx"]]
        assert abs(stored.norm().item() - 1.0) < 1e-5

    def test_assign_when_full_reuses_least_active(self):
        """When all slots are active, falls back to reusing min-activation slot."""
        vs = VectorSpace(n_slots=5, dimensions=8)
        for i in range(5):
            vs.assign_slot(_unit_vec(i), label=f"c{i}", domain="d")
        # All 5 slots active; assigning a 6th should not raise
        result = vs.assign_slot(_unit_vec(99), label="overflow", domain="d")
        assert "slot_idx" in result


# ===========================================================================
# reinforce / decay
# ===========================================================================

class TestReinforceDecay:
    def test_reinforce_increases_activation(self, vs):
        result = vs.assign_slot(_unit_vec(1))
        idx = result["slot_idx"]
        before = float(vs.activation[idx].item())
        vs.reinforce(idx)
        after = float(vs.activation[idx].item())
        assert after > before

    def test_reinforce_caps_at_one(self, vs):
        result = vs.assign_slot(_unit_vec(1))
        idx = result["slot_idx"]
        vs.activation[idx] = 1.0
        vs.reinforce(idx)
        assert float(vs.activation[idx].item()) == 1.0

    def test_decay_reduces_active_slots(self, vs):
        result = vs.assign_slot(_unit_vec(1))
        idx = result["slot_idx"]
        before = float(vs.activation[idx].item())
        vs.decay()
        after = float(vs.activation[idx].item())
        assert after < before

    def test_decay_does_not_go_below_zero(self, vs):
        result = vs.assign_slot(_unit_vec(1))
        idx = result["slot_idx"]
        vs.activation[idx] = 0.001
        for _ in range(10):
            vs.decay()
        assert float(vs.activation[idx].item()) >= 0.0

    def test_decay_leaves_dormant_slots_unchanged(self, vs):
        vs.decay()
        assert vs.activation.sum() == 0.0


# ===========================================================================
# prune
# ===========================================================================

class TestPrune:
    def test_prune_zeros_below_threshold(self, vs):
        result = vs.assign_slot(_unit_vec(1))
        idx = result["slot_idx"]
        vs.activation[idx] = 0.1  # below default threshold 0.2
        summary = vs.prune(threshold=0.2)
        assert summary["pruned_count"] == 1
        assert float(vs.activation[idx].item()) == 0.0

    def test_prune_preserves_above_threshold(self, vs):
        result = vs.assign_slot(_unit_vec(1))
        idx = result["slot_idx"]
        vs.activation[idx] = 0.5
        vs.prune(threshold=0.2)
        assert float(vs.activation[idx].item()) == 0.5

    def test_prune_returns_correct_counts(self, vs):
        for i in range(3):
            r = vs.assign_slot(_unit_vec(i))
            vs.activation[r["slot_idx"]] = 0.1  # all below threshold

        summary = vs.prune(threshold=0.2)
        assert summary["pruned_count"] == 3
        assert summary["before_active"] == 3
        assert summary["after_active"] == 0

    def test_prune_removes_labels_and_domains(self, vs):
        result = vs.assign_slot(_unit_vec(1), label="gone", domain="x")
        idx = result["slot_idx"]
        vs.activation[idx] = 0.05
        vs.prune(threshold=0.2)
        assert idx not in vs.slot_labels
        assert idx not in vs.slot_domains

    def test_pruned_slot_returns_to_unit_vector(self, vs):
        result = vs.assign_slot(_unit_vec(1))
        idx = result["slot_idx"]
        original = vs.slots[idx].clone()
        vs.activation[idx] = 0.05
        vs.prune(threshold=0.2)
        # Slot should have a new random unit vector
        new_vec = vs.slots[idx]
        assert abs(new_vec.norm().item() - 1.0) < 1e-5


# ===========================================================================
# Properties and utilities
# ===========================================================================

class TestProperties:
    def test_n_active_counts_correctly(self, vs):
        assert vs.n_active == 0
        vs.assign_slot(_unit_vec(1))
        assert vs.n_active == 1
        vs.assign_slot(_unit_vec(2))
        assert vs.n_active == 2

    def test_domain_registry_groups_by_domain(self, vs):
        r1 = vs.assign_slot(_unit_vec(1), domain="science")
        r2 = vs.assign_slot(_unit_vec(2), domain="science")
        r3 = vs.assign_slot(_unit_vec(3), domain="history")
        reg = vs.domain_registry
        assert set(reg["science"]) == {r1["slot_idx"], r2["slot_idx"]}
        assert reg["history"] == [r3["slot_idx"]]

    def test_domain_registry_empty_when_no_active(self, vs):
        assert vs.domain_registry == {}

    def test_get_active_mask_boolean_tensor(self, vs):
        mask = vs.get_active_mask()
        assert mask.dtype == torch.bool
        assert mask.shape == (N_SLOTS,)
        assert not mask.any()

    def test_get_active_mask_reflects_activations(self, vs):
        r = vs.assign_slot(_unit_vec(1))
        mask = vs.get_active_mask()
        assert mask[r["slot_idx"]].item() is True


# ===========================================================================
# reset
# ===========================================================================

class TestReset:
    def test_reset_clears_labels(self, vs):
        vs.assign_slot(_unit_vec(1), label="keep-me", domain="x")
        vs.reset()
        assert vs.slot_labels == {}

    def test_reset_clears_domains(self, vs):
        vs.assign_slot(_unit_vec(1), domain="x")
        vs.reset()
        assert vs.slot_domains == {}

    def test_reset_zeros_activation(self, vs):
        vs.assign_slot(_unit_vec(1))
        vs.reset()
        assert vs.activation.sum() == 0.0

    def test_reset_reinitialises_slots_as_unit_vectors(self, vs):
        vs.assign_slot(_unit_vec(1))
        vs.reset()
        norms = vs.slots.norm(dim=1)
        assert torch.allclose(norms, torch.ones(N_SLOTS), atol=1e-5)

    def test_reset_clears_stage_number(self, vs):
        vs.stage_number = 7
        vs.reset()
        assert vs.stage_number == 0


# ===========================================================================
# save / load / autosave
# ===========================================================================

class TestPersistence:
    def test_save_creates_file(self, vs):
        vs.assign_slot(_unit_vec(1), label="x", domain="d")
        with tempfile.NamedTemporaryFile(suffix=".bgstate", delete=False) as f:
            path = f.name
        try:
            vs.save(path, description="test-save")
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_save_returns_path(self, vs):
        vs.assign_slot(_unit_vec(1))
        with tempfile.NamedTemporaryFile(suffix=".bgstate", delete=False) as f:
            path = f.name
        try:
            returned = vs.save(path)
            assert returned == path
        finally:
            os.unlink(path)

    def test_save_creates_parent_dirs(self, tmp_path):
        vs = VectorSpace(n_slots=N_SLOTS, dimensions=DIMS)
        vs.assign_slot(_unit_vec(1))
        nested = str(tmp_path / "a" / "b" / "snapshot.bgstate")
        vs.save(nested)
        assert os.path.exists(nested)

    def test_load_restores_activation(self, vs):
        vs.assign_slot(_unit_vec(1))
        with tempfile.NamedTemporaryFile(suffix=".bgstate", delete=False) as f:
            path = f.name
        try:
            vs.save(path)
            vs2, _ = VectorSpace.load(path)
            assert vs2.n_active == vs.n_active
        finally:
            os.unlink(path)

    def test_load_restores_labels(self, vs):
        r = vs.assign_slot(_unit_vec(1), label="hello", domain="x")
        with tempfile.NamedTemporaryFile(suffix=".bgstate", delete=False) as f:
            path = f.name
        try:
            vs.save(path)
            vs2, _ = VectorSpace.load(path)
            assert vs2.slot_labels[r["slot_idx"]] == "hello"
        finally:
            os.unlink(path)

    def test_load_restores_domain_registry(self, vs):
        vs.assign_slot(_unit_vec(1), domain="science")
        vs.assign_slot(_unit_vec(2), domain="history")
        with tempfile.NamedTemporaryFile(suffix=".bgstate", delete=False) as f:
            path = f.name
        try:
            vs.save(path)
            vs2, _ = VectorSpace.load(path)
            assert "science" in vs2.domain_registry
            assert "history" in vs2.domain_registry
        finally:
            os.unlink(path)

    def test_load_restores_stage_number(self, vs):
        vs.stage_number = 5
        vs.assign_slot(_unit_vec(1))
        with tempfile.NamedTemporaryFile(suffix=".bgstate", delete=False) as f:
            path = f.name
        try:
            vs.save(path)
            vs2, _ = VectorSpace.load(path)
            assert vs2.stage_number == 5
        finally:
            os.unlink(path)

    def test_load_metadata_description(self, vs):
        vs.assign_slot(_unit_vec(1))
        with tempfile.NamedTemporaryFile(suffix=".bgstate", delete=False) as f:
            path = f.name
        try:
            vs.save(path, description="my-snapshot")
            _, meta = VectorSpace.load(path)
            assert meta["description"] == "my-snapshot"
            assert meta["version"] == "1.0"
            assert "saved_at" in meta
        finally:
            os.unlink(path)

    def test_load_slots_tensor_values_match(self, vs):
        for i in range(4):
            vs.assign_slot(_unit_vec(i))
        with tempfile.NamedTemporaryFile(suffix=".bgstate", delete=False) as f:
            path = f.name
        try:
            vs.save(path)
            vs2, _ = VectorSpace.load(path)
            assert torch.allclose(vs.slots, vs2.slots, atol=1e-6)
            assert torch.allclose(vs.activation, vs2.activation, atol=1e-6)
        finally:
            os.unlink(path)

    def test_autosave_creates_timestamped_file(self, vs, tmp_path):
        vs.assign_slot(_unit_vec(1))
        saves_dir = str(tmp_path / "saves")
        path = vs.autosave(saves_dir=saves_dir)
        assert os.path.exists(path)
        assert path.endswith(".bgstate")
        assert "autosave_" in os.path.basename(path)
