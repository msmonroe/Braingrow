"""
runner.py — Headless evaluation runner.

Drives any of the three systems (BrainGrow / TorchFlatThreshold /
FAISSFlatThreshold) against the fabricated-query set and produces
a List[QueryResult] for the metrics module.

BrainGrow's native query_router.route_query() does NOT itself apply the
0.60 threshold — that logic lives downstream in the epistemic classifier.
To keep the comparison apples-to-apples, we apply the same threshold
to all three systems here, using max-similarity over top-K matches.
This matches the paper's Section 3.5 decision procedure exactly.
"""
from __future__ import annotations

from typing import List, Protocol

from evaluation.fabricated_queries import FabricatedQuery
from evaluation.metrics import QueryResult


BRAINGROW_THRESHOLD = 0.60


# --------------------------------------------------------------------------
# Protocol for flat baselines (TorchFlatThreshold, FAISSFlatThreshold)
# --------------------------------------------------------------------------
class _HasQueryMethod(Protocol):
    def query(self, text: str) -> dict: ...


def run_baseline(
    system_name: str,
    baseline: _HasQueryMethod,
    queries: List[FabricatedQuery],
) -> List[QueryResult]:
    """Run a flat baseline (torch or FAISS) over the query set."""
    results: List[QueryResult] = []
    for q in queries:
        r = baseline.query(q.text)
        results.append(QueryResult(
            query_text=q.text,
            bucket=q.bucket,
            expected=q.expected,
            verdict=r["verdict"],
            similarity=r["similarity"],
            nearest_label=r.get("nearest_label", ""),
            nearest_domain=r.get("nearest_domain", ""),
        ))
    return results


# --------------------------------------------------------------------------
# BrainGrow driver
# --------------------------------------------------------------------------
def run_braingrow(
    router,
    queries: List[FabricatedQuery],
    threshold: float = BRAINGROW_THRESHOLD,
) -> List[QueryResult]:
    """
    Run BrainGrow via its QueryRouter.

    We take the max similarity across the top-K matches returned by
    route_query() and apply the confidence threshold ourselves. This
    reproduces the paper's Section 3.5 classifier procedure exactly:

      if boundary_violation: OUT_OF_BOUNDS  (mapped to CONFIDENT here,
                                             since it's a non-abstain
                                             verdict — this experiment
                                             doesn't test negative
                                             domains)
      elif max_sim < threshold: HONEST_UNKNOWN
      else: CONFIDENT

    Reinforcement side-effect: QueryRouter.route_query() reinforces every
    matched slot. Over 100 queries this will drift activation scores.
    That's fine for this experiment (we're measuring per-query verdicts,
    not long-run behavior), but be aware if you re-run on the same
    session object.
    """
    results: List[QueryResult] = []
    for q in queries:
        routing = router.route_query(q.text, top_k=5)
        matches = routing.get("matches", [])
        if not matches:
            max_sim = 0.0
            nearest_label = ""
            nearest_domain = ""
        else:
            max_sim = max(m["similarity"] for m in matches)
            best = max(matches, key=lambda m: m["similarity"])
            nearest_label = best.get("label", "")
            nearest_domain = best.get("domain", "")

        if routing.get("boundary_violation"):
            verdict = "CONFIDENT"  # non-abstain; paper classifies as OOB
        elif max_sim < threshold:
            verdict = "HONEST_UNKNOWN"
        else:
            verdict = "CONFIDENT"

        results.append(QueryResult(
            query_text=q.text,
            bucket=q.bucket,
            expected=q.expected,
            verdict=verdict,
            similarity=max_sim,
            nearest_label=nearest_label,
            nearest_domain=nearest_domain,
        ))
    return results
