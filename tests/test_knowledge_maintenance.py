"""
test_knowledge_maintenance.py — Unit tests for KnowledgeMaintenance.

Covers:
  - on_boundary_violation  (with and without source_domain)
  - audit_hallucination_risk  (empty, HIGH, MEDIUM, BALANCED, < min_slots skipped)
  - ingest_negative_examples
  - correction_log / correction_count
  - AuditReport.as_text()  (all risk branches)
  - _encode (returns unit-normalised tensor)
"""

from __future__ import annotations

import hashlib
from typing import List, Union

import numpy as np
import pytest
import torch

from knowledge_maintenance import (
    AuditReport,
    DomainRiskReport,
    KnowledgeMaintenance,
    _HIGH_RISK_RATIO,
    _MEDIUM_RISK_RATIO,
    _MIN_SLOTS_TO_AUDIT,
)
from vector_space import VectorSpace

# ---------------------------------------------------------------------------
# Constants — tiny VS so tests are fast
# ---------------------------------------------------------------------------

_N = 200
_D = 8


# ---------------------------------------------------------------------------
# Mock sentence-transformer that returns tensors when convert_to_tensor=True
# ---------------------------------------------------------------------------

class _MockModel:
    """Deterministic sha256-seeded 8-D encoder; supports both numpy and tensor output."""

    def to(self, device):
        return self

    def encode(
        self,
        text: Union[str, List[str]],
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        if isinstance(text, list):
            vecs = np.stack([self._one(t) for t in text], axis=0)
        else:
            vecs = self._one(text).reshape(1, -1)

        if convert_to_tensor:
            return torch.tensor(vecs, dtype=torch.float32)
        return vecs

    @staticmethod
    def _one(text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.RandomState(seed)
        v = rng.randn(_D).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vs():
    return VectorSpace(n_slots=_N, dimensions=_D)


@pytest.fixture
def model():
    return _MockModel()


@pytest.fixture
def km(vs, model):
    return KnowledgeMaintenance(vector_space=vs, model=model)


def _unit_vec(seed: int, dim: int = _D) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    v = torch.randn(dim, generator=gen)
    return v / (v.norm() + 1e-8)


def _populate_domain(vs, domain: str, count: int, seed_offset: int = 0):
    """Assign *count* slots labelled with *domain*."""
    for i in range(count):
        vs.assign_slot(
            _unit_vec(seed_offset + i),
            label=f"{domain}_slot_{i}",
            domain=domain,
        )


# ===========================================================================
# _encode
# ===========================================================================

class TestEncode:
    def test_returns_torch_tensor(self, km):
        result = km._encode("some text here")
        assert isinstance(result, torch.Tensor)

    def test_result_is_unit_length(self, km):
        result = km._encode("some text here")
        norm = float(result.norm().item())
        assert abs(norm - 1.0) < 1e-4

    def test_different_texts_differ(self, km):
        a = km._encode("text about science alpha")
        b = km._encode("text about cooking beta")
        # They should not be identical
        assert not torch.allclose(a, b)


# ===========================================================================
# on_boundary_violation
# ===========================================================================

class TestOnBoundaryViolation:
    def test_returns_required_keys(self, km):
        result = km.on_boundary_violation("what is science", "science_negative")
        for key in ("action", "negative_label", "negative_domain", "slot_result", "logged"):
            assert key in result

    def test_action_is_correct(self, km):
        result = km.on_boundary_violation("what is science", "science_negative")
        assert result["action"] == "negative_slot_ingested"

    def test_logged_is_true(self, km):
        result = km.on_boundary_violation("query text", "some_domain")
        assert result["logged"] is True

    def test_label_uses_boundary_prefix_without_source(self, km):
        result = km.on_boundary_violation("some query text", "domain_a")
        assert result["negative_label"].startswith("BOUNDARY")

    def test_label_uses_not_prefix_with_source_domain(self, km):
        result = km.on_boundary_violation("some query text", "domain_a", source_domain="science")
        assert result["negative_label"].startswith("NOT science")

    def test_negative_domain_has_auto_suffix(self, km):
        result = km.on_boundary_violation("query text", "orig_domain")
        assert result["negative_domain"] == "orig_domain_negative_auto"

    def test_negative_domain_registered(self, km, vs):
        result = km.on_boundary_violation("query text", "orig_domain")
        assert result["negative_domain"] in vs.negative_domains

    def test_slot_assigned(self, km, vs):
        before = vs.n_active
        km.on_boundary_violation("query text", "domain")
        assert vs.n_active == before + 1

    def test_slot_result_has_slot_idx(self, km):
        result = km.on_boundary_violation("query text", "domain")
        assert "slot_idx" in result["slot_result"]
        assert isinstance(result["slot_result"]["slot_idx"], int)

    def test_correction_log_grows(self, km):
        assert km.correction_count() == 0
        km.on_boundary_violation("query 1", "domain_a")
        assert km.correction_count() == 1
        km.on_boundary_violation("query 2", "domain_b")
        assert km.correction_count() == 2

    def test_correction_log_contains_expected_keys(self, km):
        km.on_boundary_violation("my query", "neg_domain")
        log = km.correction_log()
        assert len(log) == 1
        entry = log[0]
        for key in ("timestamp", "query_text", "nearest_domain", "negative_label",
                    "negative_domain", "slot_idx"):
            assert key in entry

    def test_label_truncated_to_60_chars(self, km):
        long_query = "x" * 100
        result = km.on_boundary_violation(long_query, "domain")
        assert len(result["negative_label"]) <= len("BOUNDARY — ") + 60


# ===========================================================================
# correction_log / correction_count
# ===========================================================================

class TestCorrectionLog:
    def test_empty_initially(self, km):
        assert km.correction_count() == 0
        assert km.correction_log() == []

    def test_log_returns_copy(self, km):
        km.on_boundary_violation("q", "d")
        log = km.correction_log()
        log.clear()
        # Internal log should be unaffected
        assert km.correction_count() == 1


# ===========================================================================
# ingest_negative_examples
# ===========================================================================

class TestIngestNegativeExamples:
    def test_returns_list_of_slot_results(self, km, vs):
        results = km.ingest_negative_examples(
            ["This concept is not science", "Another non-science fact"],
            domain="science_negative",
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_slots_activated(self, km, vs):
        before = vs.n_active
        km.ingest_negative_examples(["Not a history fact"], domain="history_negative")
        assert vs.n_active == before + 1

    def test_domain_registered_as_negative(self, km, vs):
        km.ingest_negative_examples(
            ["text a", "text b"],
            domain="cooking_negative",
            register_domain=True,
        )
        assert "cooking_negative" in vs.negative_domains

    def test_domain_not_registered_when_flag_false(self, km, vs):
        km.ingest_negative_examples(
            ["text a"],
            domain="unregistered_neg",
            register_domain=False,
        )
        assert "unregistered_neg" not in vs.negative_domains

    def test_skips_empty_text(self, km, vs):
        before = vs.n_active
        results = km.ingest_negative_examples(
            ["", "   ", "valid text here"],
            domain="domain_neg",
        )
        assert vs.n_active == before + 1
        assert len(results) == 1

    def test_empty_list_returns_empty(self, km):
        results = km.ingest_negative_examples([], domain="domain_neg")
        assert results == []


# ===========================================================================
# audit_hallucination_risk
# ===========================================================================

class TestAuditHallucinationRisk:
    def test_empty_space_total_domains_zero(self, km):
        report = km.audit_hallucination_risk()
        assert report.total_domains == 0

    def test_below_min_slots_not_assessed(self, km, vs):
        """A domain with fewer than _MIN_SLOTS_TO_AUDIT positive slots is skipped."""
        _populate_domain(vs, "tiny_domain", _MIN_SLOTS_TO_AUDIT - 1, seed_offset=0)
        report = km.audit_hallucination_risk()
        assert report.total_domains == 0

    def test_high_risk_when_no_negatives(self, km, vs):
        """Domain with >= _MIN_SLOTS_TO_AUDIT positive slots, zero negatives → HIGH."""
        _populate_domain(vs, "science", _MIN_SLOTS_TO_AUDIT, seed_offset=0)
        report = km.audit_hallucination_risk()
        assert report.total_domains == 1
        assert len(report.high_risk) == 1
        assert report.high_risk[0].domain == "science"
        assert report.high_risk[0].risk_level == "HIGH"

    def test_high_risk_ratio_above_threshold(self, km, vs):
        """Many positives + one negative but ratio still above HIGH threshold → HIGH."""
        pos_count = int(_HIGH_RISK_RATIO * 2) + 1   # well above threshold
        _populate_domain(vs, "history", pos_count, seed_offset=0)
        # One negative slot
        neg_domain = "history_negative_auto"
        vs.register_negative_domain(neg_domain)
        vs.assign_slot(_unit_vec(999), label="not history", domain=neg_domain)
        report = km.audit_hallucination_risk()
        # ratio = pos_count / 1 ≫ HIGH_RISK_RATIO
        assert any(r.risk_level == "HIGH" for r in report.high_risk)

    def test_medium_risk_domain(self, km, vs):
        """Ratio between MEDIUM and HIGH thresholds → MEDIUM."""
        # pos = 4, neg = 1 → ratio = 4.0 which is > _MEDIUM_RISK_RATIO(2) but < _HIGH(5)
        _populate_domain(vs, "cooking", 4, seed_offset=10)
        neg_domain = "cooking_negative_auto"
        vs.register_negative_domain(neg_domain)
        # Add 1 negative
        vs.assign_slot(_unit_vec(900), label="not cooking", domain=neg_domain)
        report = km.audit_hallucination_risk()
        all_domains = report.high_risk + report.medium_risk + report.low_risk + report.balanced
        cooking_report = next((r for r in all_domains if r.domain == "cooking"), None)
        assert cooking_report is not None
        assert cooking_report.risk_level in ("HIGH", "MEDIUM")

    def test_balanced_domain(self, km, vs):
        """Ratio below MEDIUM threshold → BALANCED."""
        # 3 positives, 3 negatives → ratio ≈ 1.0 < _MEDIUM_RISK_RATIO(2)
        _populate_domain(vs, "physics", 3, seed_offset=20)
        neg_domain = "physics_negative_auto"
        vs.register_negative_domain(neg_domain)
        for i in range(3):
            vs.assign_slot(_unit_vec(800 + i), label=f"not physics {i}", domain=neg_domain)
        report = km.audit_hallucination_risk()
        assert len(report.balanced) >= 1
        assert report.balanced[0].risk_level == "BALANCED"

    def test_report_has_summary_string(self, km, vs):
        _populate_domain(vs, "math", _MIN_SLOTS_TO_AUDIT, seed_offset=50)
        report = km.audit_hallucination_risk()
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_domain_risk_report_fields(self, km, vs):
        _populate_domain(vs, "biology", _MIN_SLOTS_TO_AUDIT, seed_offset=60)
        report = km.audit_hallucination_risk()
        r = report.high_risk[0]
        assert r.domain == "biology"
        assert r.positive_slots >= _MIN_SLOTS_TO_AUDIT
        assert r.negative_slots == 0
        assert r.ratio > 0
        assert r.risk_level == "HIGH"
        assert isinstance(r.recommendation, str)


# ===========================================================================
# AuditReport.as_text()
# ===========================================================================

class TestAuditReportAsText:
    def _make_report(self, **kwargs) -> AuditReport:
        import time
        defaults = dict(timestamp=time.time(), total_domains=0)
        defaults.update(kwargs)
        return AuditReport(**defaults)

    def _make_domain_report(self, domain="d", pos=5, neg=0, ratio=5.0,
                             risk="HIGH", rec="test rec"):
        return DomainRiskReport(
            domain=domain, positive_slots=pos, negative_slots=neg,
            ratio=ratio, risk_level=risk, recommendation=rec,
        )

    def test_as_text_contains_header(self):
        report = self._make_report()
        text = report.as_text()
        assert "Hallucination Risk Audit" in text

    def test_no_domains_message_when_empty(self):
        report = self._make_report(total_domains=0)
        text = report.as_text()
        assert "No domains" in text or "Ingest more data" in text

    def test_high_risk_section_appears(self):
        report = self._make_report(
            total_domains=1,
            high_risk=[self._make_domain_report(risk="HIGH")],
        )
        text = report.as_text()
        assert "HIGH RISK" in text

    def test_medium_risk_section_appears(self):
        report = self._make_report(
            total_domains=1,
            medium_risk=[self._make_domain_report(domain="m", risk="MEDIUM")],
        )
        text = report.as_text()
        assert "MEDIUM RISK" in text

    def test_balanced_section_appears(self):
        report = self._make_report(
            total_domains=1,
            balanced=[self._make_domain_report(domain="b", risk="BALANCED")],
        )
        text = report.as_text()
        assert "BALANCED" in text

    def test_domain_name_present_in_high_risk(self):
        report = self._make_report(
            total_domains=1,
            high_risk=[self._make_domain_report(domain="astrophysics", risk="HIGH")],
        )
        text = report.as_text()
        assert "astrophysics" in text

    def test_recommendation_present_in_high_risk(self):
        report = self._make_report(
            total_domains=1,
            high_risk=[self._make_domain_report(rec="Add more negatives please!")],
        )
        text = report.as_text()
        assert "Add more negatives please!" in text

    def test_domains_assessed_count_shown(self):
        report = self._make_report(total_domains=3)
        text = report.as_text()
        assert "3" in text
