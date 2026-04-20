"""
metrics.py — Scoring for the RAG comparison experiment.

Two classes of metric:

1. Per-system correctness (precision/recall/F1 on abstention)
   Treats HONEST_UNKNOWN as the "positive" class because abstention is
   the behavior we're evaluating.

     TN = correctly abstained   (expected=HU, verdict=HU)
     FP = hallucination         (expected=HU, verdict=CONFIDENT)
     FN = over-abstention       (expected=CONFIDENT, verdict=HU)
     TP = correctly confident   (expected=CONFIDENT, verdict=CONFIDENT)

     Abstention precision = TN / (TN + FN)  — of things we abstained on,
                                               how many should have been?
     Abstention recall    = TN / (TN + FP)  — of things we should have
                                               abstained on, how many did we?

2. Inter-system verdict agreement
   For each pair of systems, how often do they produce the same verdict?
   If BrainGrow and FlatThreshold agree on 100/100 queries, the
   developmental machinery is not contributing to the abstention decision.
   This is the single most important number in the whole experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class QueryResult:
    """One system's verdict on one query."""
    query_text: str
    bucket: str
    expected: str
    verdict: str          # "CONFIDENT" | "HONEST_UNKNOWN"
    similarity: float
    nearest_label: str = ""
    nearest_domain: str = ""

    @property
    def correct(self) -> bool:
        return self.verdict == self.expected


@dataclass
class SystemScorecard:
    """Per-bucket and overall metrics for one system."""
    system_name: str
    per_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)
    overall: Dict[str, float] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"=== {self.system_name} ==="]
        for bucket, m in self.per_bucket.items():
            lines.append(
                f"  {bucket:<18s}  n={int(m['n']):3d}  "
                f"correct={int(m['correct']):3d}  "
                f"accuracy={m['accuracy']:.2%}"
            )
        o = self.overall
        lines.append(
            f"  {'OVERALL':<18s}  n={int(o['n']):3d}  "
            f"correct={int(o['correct']):3d}  "
            f"accuracy={o['accuracy']:.2%}"
        )
        lines.append(
            f"  Abstention  precision={o['abst_precision']:.2%}  "
            f"recall={o['abst_recall']:.2%}  "
            f"F1={o['abst_f1']:.2%}"
        )
        lines.append(
            f"  Confusion   TP={int(o['tp'])}  FP={int(o['fp'])}  "
            f"FN={int(o['fn'])}  TN={int(o['tn'])}"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_system(system_name: str, results: List[QueryResult]) -> SystemScorecard:
    """Compute per-bucket accuracy and overall precision/recall on abstention."""
    by_bucket: Dict[str, List[QueryResult]] = {}
    for r in results:
        by_bucket.setdefault(r.bucket, []).append(r)

    per_bucket: Dict[str, Dict[str, float]] = {}
    for bucket, rs in by_bucket.items():
        n = len(rs)
        correct = sum(1 for r in rs if r.correct)
        per_bucket[bucket] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else 0.0,
        }

    # Overall + confusion matrix (abstention as positive class)
    tp = fp = fn = tn = 0
    for r in results:
        if r.expected == "HONEST_UNKNOWN" and r.verdict == "HONEST_UNKNOWN":
            tn += 1
        elif r.expected == "HONEST_UNKNOWN" and r.verdict == "CONFIDENT":
            fp += 1
        elif r.expected == "CONFIDENT" and r.verdict == "HONEST_UNKNOWN":
            fn += 1
        elif r.expected == "CONFIDENT" and r.verdict == "CONFIDENT":
            tp += 1

    total = len(results)
    correct = tp + tn
    abst_precision = tn / (tn + fn) if (tn + fn) else 0.0
    abst_recall = tn / (tn + fp) if (tn + fp) else 0.0
    abst_f1 = (
        2 * abst_precision * abst_recall / (abst_precision + abst_recall)
        if (abst_precision + abst_recall) else 0.0
    )

    overall = {
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "abst_precision": abst_precision,
        "abst_recall": abst_recall,
        "abst_f1": abst_f1,
    }
    return SystemScorecard(system_name, per_bucket, overall)


# --------------------------------------------------------------------------
# Inter-system agreement
# --------------------------------------------------------------------------
def agreement_matrix(
    results_by_system: Dict[str, List[QueryResult]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    For every ordered system pair, compute:
        agree_rate   — fraction of queries where both systems agree
        disagree_rows — list of (query_text, bucket, verdict_a, verdict_b,
                                 sim_a, sim_b) for manual inspection

    The single most important output of the whole experiment:
    if BrainGrow and FlatThreshold have agree_rate ≈ 1.0, the paper's
    "structural" framing is not supported by evidence.
    """
    systems = list(results_by_system.keys())
    out: Dict[Tuple[str, str], Dict[str, float]] = {}

    for i, a in enumerate(systems):
        for b in systems[i + 1:]:
            res_a = results_by_system[a]
            res_b = results_by_system[b]
            # Align on query text to be safe against ordering drift.
            lookup_b = {r.query_text: r for r in res_b}

            agree = 0
            disagree_rows: List[tuple] = []
            total = 0
            for r_a in res_a:
                r_b = lookup_b.get(r_a.query_text)
                if r_b is None:
                    continue
                total += 1
                if r_a.verdict == r_b.verdict:
                    agree += 1
                else:
                    disagree_rows.append((
                        r_a.query_text, r_a.bucket,
                        r_a.verdict, r_b.verdict,
                        round(r_a.similarity, 4), round(r_b.similarity, 4),
                    ))

            out[(a, b)] = {
                "n": total,
                "agree": agree,
                "agree_rate": agree / total if total else 0.0,
                "disagree_rows": disagree_rows,
            }
    return out


def render_agreement(agreement: Dict[Tuple[str, str], Dict[str, float]]) -> str:
    lines = ["=== Inter-system verdict agreement ==="]
    for (a, b), m in agreement.items():
        lines.append(
            f"  {a} vs {b}:  "
            f"{int(m['agree'])}/{int(m['n'])} agree  "
            f"({m['agree_rate']:.2%})"
        )
    lines.append("")
    lines.append("=== Disagreement details ===")
    for (a, b), m in agreement.items():
        rows = m["disagree_rows"]
        if not rows:
            lines.append(f"  {a} vs {b}: no disagreements.")
            continue
        lines.append(f"  {a} vs {b}:  {len(rows)} disagreement(s):")
        lines.append(
            f"    {'bucket':<18s}  {a:<14s}  {b:<14s}  sim_a   sim_b   query"
        )
        for text, bucket, v_a, v_b, sim_a, sim_b in rows:
            lines.append(
                f"    {bucket:<18s}  {v_a:<14s}  {v_b:<14s}  "
                f"{sim_a:>5.3f}   {sim_b:>5.3f}   {text[:60]}"
            )
    return "\n".join(lines)
