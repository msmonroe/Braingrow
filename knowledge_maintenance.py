"""
knowledge_maintenance.py — Active Knowledge Maintenance for BrainGrow.

Two complementary mechanisms that transform BrainGrow from a static
knowledge store into a self-improving developmental system:

  1. Reactive maintenance — KnowledgeMaintenance.on_boundary_violation()
     When the query router detects a boundary violation (a query lands in
     a negative domain), this module auto-ingests a targeted negative
     example into the vector space. The system learns from its own
     mistakes without retraining.

  2. Proactive auditing — KnowledgeMaintenance.audit_hallucination_risk()
     Scans all registered domains and computes a hallucination risk score
     based on the ratio of positive to negative slots. Domains with high
     positive/negative ratios are flagged before they cause problems.
     Returns a structured risk report for display in the Gradio UI.

Connects to existing modules via VectorSpace and GrowthEngine interfaces.
No changes required to vector_space.py, growth_engine.py, or query_router.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer

from vector_space import VectorSpace


# ---------------------------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------------------------

# Ratio of positive slots to negative slots above which a domain is at risk
_HIGH_RISK_RATIO:   float = 5.0   # 5:1 positive:negative → HIGH
_MEDIUM_RISK_RATIO: float = 2.0   # 2:1 → MEDIUM
_MIN_SLOTS_TO_AUDIT: int  = 3     # ignore domains with fewer than this many slots


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DomainRiskReport:
    """Risk assessment for a single domain."""
    domain:          str
    positive_slots:  int
    negative_slots:  int
    ratio:           float          # positive / (negative + 1e-6)
    risk_level:      str            # "HIGH" | "MEDIUM" | "LOW" | "BALANCED"
    recommendation:  str


@dataclass
class AuditReport:
    """Full hallucination risk audit across all registered domains."""
    timestamp:      float
    total_domains:  int
    high_risk:      List[DomainRiskReport] = field(default_factory=list)
    medium_risk:    List[DomainRiskReport] = field(default_factory=list)
    low_risk:       List[DomainRiskReport] = field(default_factory=list)
    balanced:       List[DomainRiskReport] = field(default_factory=list)
    summary:        str = ""

    def as_text(self) -> str:
        """Return a human-readable text summary for the Gradio UI."""
        lines = [
            f"═══ Hallucination Risk Audit ═══",
            f"Domains assessed: {self.total_domains}",
            f"High risk: {len(self.high_risk)}   "
            f"Medium: {len(self.medium_risk)}   "
            f"Low: {len(self.low_risk)}   "
            f"Balanced: {len(self.balanced)}",
            "",
        ]

        if self.high_risk:
            lines.append("⛔ HIGH RISK (immediate counterbalancing recommended):")
            for r in self.high_risk:
                lines.append(
                    f"  {r.domain:<30} "
                    f"+{r.positive_slots} / -{r.negative_slots}  ratio {r.ratio:.1f}x"
                )
                lines.append(f"    → {r.recommendation}")
            lines.append("")

        if self.medium_risk:
            lines.append("⚠️  MEDIUM RISK (counterbalancing advised):")
            for r in self.medium_risk:
                lines.append(
                    f"  {r.domain:<30} "
                    f"+{r.positive_slots} / -{r.negative_slots}  ratio {r.ratio:.1f}x"
                )
            lines.append("")

        if self.balanced:
            lines.append("✅ BALANCED domains:")
            for r in self.balanced:
                lines.append(
                    f"  {r.domain:<30} "
                    f"+{r.positive_slots} / -{r.negative_slots}"
                )

        if not self.total_domains:
            lines.append("No domains with sufficient slots found. Ingest more data first.")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# KnowledgeMaintenance
# ---------------------------------------------------------------------------

class KnowledgeMaintenance:
    """
    Active knowledge maintenance for BrainGrow.

    Parameters
    ----------
    vector_space : VectorSpace
        The shared VectorSpace instance (same object used by QueryRouter
        and GrowthEngine).
    model : SentenceTransformer
        The shared encoder (all-MiniLM-L6-v2). Reused — not re-loaded.
    growth_engine : optional
        GrowthEngine instance. If provided, negative examples are ingested
        via the standard pipeline (chunk → encode → assign_slot). If None,
        embeddings are written directly to VectorSpace for lightweight use.
    """

    def __init__(
        self,
        vector_space: VectorSpace,
        model: SentenceTransformer,
        growth_engine=None,
    ) -> None:
        self.vs    = vector_space
        self.model = model
        self.ge    = growth_engine

        # Log of all reactive corrections made this session
        self._correction_log: List[dict] = []

    # --------------------------------------------------------------------------
    # 1. Reactive maintenance — called by session on boundary violation
    # --------------------------------------------------------------------------

    def on_boundary_violation(
        self,
        query_text:     str,
        nearest_domain: str,
        source_domain:  Optional[str] = None,
    ) -> dict:
        """
        React to a boundary violation by ingesting a targeted negative example.

        When a query lands in a negative domain, we generate a synthetic
        negative example — a slot that explicitly represents "this query
        concept does NOT belong in this domain" — and ingest it with a
        negative domain label. This counterbalances future queries of the
        same type.

        Parameters
        ----------
        query_text      : the original query that triggered the violation
        nearest_domain  : the negative domain the query landed in
        source_domain   : optional — the positive domain the query came from
                          (used to construct a more specific negative label)

        Returns
        -------
        {
            action          : str,
            negative_label  : str,
            negative_domain : str,
            slot_result     : dict,   # from VectorSpace.assign_slot()
            logged          : bool,
        }
        """
        # Build a synthetic negative label describing the invalid combination
        if source_domain:
            negative_label = (
                f"NOT {source_domain} — {query_text[:60].strip()}"
            )
        else:
            negative_label = f"BOUNDARY — {query_text[:60].strip()}"

        negative_domain = f"{nearest_domain}_negative_auto"

        # Register the new domain as negative so future queries trigger
        # BOUNDARY VIOLATION rather than CONFIDENT when they land here
        self.vs.register_negative_domain(negative_domain)

        # Encode the query text and assign it as a negative slot
        emb = self._encode(query_text)
        slot_result = self.vs.assign_slot(
            embedding = emb,
            label     = negative_label,
            domain    = negative_domain,
        )

        record = {
            "timestamp":       time.time(),
            "query_text":      query_text,
            "nearest_domain":  nearest_domain,
            "negative_label":  negative_label,
            "negative_domain": negative_domain,
            "slot_idx":        slot_result["slot_idx"],
        }
        self._correction_log.append(record)

        return {
            "action":          "negative_slot_ingested",
            "negative_label":  negative_label,
            "negative_domain": negative_domain,
            "slot_result":     slot_result,
            "logged":          True,
        }

    # --------------------------------------------------------------------------
    # 2. Proactive auditing — call periodically or on demand
    # --------------------------------------------------------------------------

    def audit_hallucination_risk(self) -> AuditReport:
        """
        Scan all registered domains and return a hallucination risk report.

        Risk is measured as the ratio of positive slots to negative slots
        in each domain neighbourhood. A domain with many positive examples
        and few or no negative counterbalances is at high risk of producing
        confident hallucinations — per the BrainGrow experimental finding
        that richer positive knowledge makes fabricated combinations harder
        to catch.

        Returns
        -------
        AuditReport — structured report with per-domain risk levels and
        a formatted text summary suitable for display in the Gradio UI.
        """
        report = AuditReport(
            timestamp     = time.time(),
            total_domains = 0,
        )

        # Build domain→slots map, separating positive and negative domains
        domain_registry = self.vs.domain_registry
        negative_domains = self.vs.negative_domains

        # Identify base domain names (strip _negative / _negative_auto suffixes)
        def base_domain(d: str) -> str:
            return (
                d.replace("_negative_auto", "")
                 .replace("_negative", "")
                 .strip()
            )

        all_bases: set = set()
        for d in domain_registry:
            all_bases.add(base_domain(d))

        assessed = 0
        for base in sorted(all_bases):
            # Collect positive slot count
            pos_slots = sum(
                len(slots)
                for domain, slots in domain_registry.items()
                if base_domain(domain) == base
                and domain not in negative_domains
            )

            # Collect negative slot count (auto-generated or manually registered)
            neg_slots = sum(
                len(slots)
                for domain, slots in domain_registry.items()
                if base_domain(domain) == base
                and domain in negative_domains
            )

            if pos_slots < _MIN_SLOTS_TO_AUDIT:
                continue  # not enough data to assess

            assessed += 1
            ratio = pos_slots / (neg_slots + 1e-6)

            if neg_slots == 0:
                risk_level = "HIGH"
                recommendation = (
                    f"No negative examples. Ingest counterexamples for '{base}' "
                    f"to prevent confident hallucination on out-of-domain queries."
                )
            elif ratio >= _HIGH_RISK_RATIO:
                risk_level = "HIGH"
                recommendation = (
                    f"Positive/negative ratio {ratio:.1f}x. Add at least "
                    f"{int(pos_slots / _HIGH_RISK_RATIO) - neg_slots} more "
                    f"negative examples to bring ratio below {_HIGH_RISK_RATIO}x."
                )
            elif ratio >= _MEDIUM_RISK_RATIO:
                risk_level = "MEDIUM"
                recommendation = (
                    f"Ratio {ratio:.1f}x — moderate risk. "
                    f"Consider adding negative counterexamples."
                )
            else:
                risk_level     = "BALANCED"
                recommendation = "Positive/negative balance is healthy."

            domain_report = DomainRiskReport(
                domain         = base,
                positive_slots = pos_slots,
                negative_slots = neg_slots,
                ratio          = ratio,
                risk_level     = risk_level,
                recommendation = recommendation,
            )

            if risk_level == "HIGH":
                report.high_risk.append(domain_report)
            elif risk_level == "MEDIUM":
                report.medium_risk.append(domain_report)
            elif risk_level == "BALANCED":
                report.balanced.append(domain_report)
            else:
                report.low_risk.append(domain_report)

        report.total_domains = assessed
        report.summary = (
            f"{len(report.high_risk)} high-risk, "
            f"{len(report.medium_risk)} medium-risk, "
            f"{len(report.balanced)} balanced domains."
        )
        return report

    # --------------------------------------------------------------------------
    # 3. Manual negative ingestion — for human-directed counterbalancing
    # --------------------------------------------------------------------------

    def ingest_negative_examples(
        self,
        examples:       List[str],
        domain:         str,
        register_domain: bool = True,
    ) -> List[dict]:
        """
        Manually ingest a list of negative example strings into *domain*.

        Use this when the audit flags a high-risk domain and you want to
        provide explicit counterexamples rather than relying on auto-generation.

        Parameters
        ----------
        examples        : list of text strings describing what the domain is NOT
        domain          : negative domain label (e.g. "science_negative")
        register_domain : if True, registers *domain* as a negative domain

        Returns
        -------
        List of slot_result dicts from VectorSpace.assign_slot()
        """
        if register_domain:
            self.vs.register_negative_domain(domain)

        results = []
        for text in examples:
            if not text.strip():
                continue
            emb = self._encode(text)
            result = self.vs.assign_slot(
                embedding = emb,
                label     = text[:80].strip(),
                domain    = domain,
            )
            results.append(result)

        return results

    # --------------------------------------------------------------------------
    # Correction log
    # --------------------------------------------------------------------------

    def correction_log(self) -> List[dict]:
        """Return the log of all reactive corrections made this session."""
        return list(self._correction_log)

    def correction_count(self) -> int:
        """Number of reactive corrections made this session."""
        return len(self._correction_log)

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _encode(self, text: str) -> torch.Tensor:
        """Encode *text* to a unit-normalised torch tensor."""
        emb = self.model.encode(
            [text],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return emb[0].cpu()
