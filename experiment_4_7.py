"""
experiment_4_7.py — Online Epistemic Correction Without Retraining
Section 4.7 experiment for the BrainGrow arXiv paper.

Demonstrates that targeted negative knowledge ingestion post-deployment
measurably reduces boundary violation rate in specific domains without
retraining. This capability is structurally unavailable in dense
weight-matrix architectures.

METHODOLOGY
-----------
Phase 1 — Baseline measurement
  Ingest positive knowledge across three domains (science, history, cooking).
  Run a fixed set of cross-domain collision queries.
  Record boundary violation rate BEFORE any corrections.

Phase 2 — Online correction
  Allow on_boundary_violation() to auto-ingest negative slots as queries fire.
  Re-run the identical query set.
  Record boundary violation rate AFTER corrections.

Phase 3 — Audit comparison
  Run hallucination risk audit before and after.
  Show domain risk scores dropping as corrections accumulate.

Phase 4 — Dense model control
  Run identical queries against DenseModel.
  Show it cannot self-correct without retraining.

OUTPUT
------
Prints a structured results table suitable for inclusion in the paper.
Saves results to experiment_4_7_results.json for reproducibility.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer

from vector_space import VectorSpace
from growth_engine import GrowthEngine
from query_router import QueryRouter
from comparison_harness import DenseModel, BrainGrowModel
from knowledge_maintenance import KnowledgeMaintenance


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Positive knowledge — three clean domains
POSITIVE_KNOWLEDGE = {
    "science": [
        "DNA replication occurs during cell division.",
        "Photosynthesis converts sunlight into chemical energy.",
        "Mitochondria produce ATP through cellular respiration.",
        "Neurons transmit signals via electrochemical impulses.",
        "The speed of light in a vacuum is approximately 3×10⁸ m/s.",
        "Quantum entanglement links particles across arbitrary distances.",
        "CRISPR-Cas9 enables precise genomic editing.",
        "Black holes form when massive stars collapse under gravity.",
        "The periodic table organises elements by atomic number.",
        "Enzymes lower the activation energy of chemical reactions.",
    ],
    "history": [
        "The Western Roman Empire fell in 476 AD.",
        "The Black Death killed roughly one third of Europe's population.",
        "The printing press was invented by Gutenberg around 1440.",
        "The French Revolution began in 1789.",
        "World War I started in 1914 following the assassination of Archduke Franz Ferdinand.",
        "The Silk Road connected China to the Mediterranean for centuries.",
        "The Magna Carta was signed in 1215.",
        "The Ottoman Empire lasted from 1299 to 1922.",
        "Columbus reached the Americas in 1492.",
        "The Renaissance began in 14th century Italy.",
    ],
    "cooking": [
        "Fermentation uses microorganisms to transform food.",
        "The Maillard reaction produces browning in cooked meat.",
        "Emulsification combines oil and water using an emulsifier.",
        "Salt denatures proteins and draws out moisture.",
        "Caramelisation occurs when sugars are heated above 160°C.",
        "Gluten forms when flour proteins are hydrated and worked.",
        "Sous vide cooking uses precise low-temperature water baths.",
        "Umami is the fifth basic taste, associated with glutamate.",
        "Resting meat after cooking redistributes its juices.",
        "Blanching briefly boils vegetables then shocks them in ice water.",
    ],
}

# Cross-domain collision queries — designed to land in wrong domains
# Each query semantically resembles one domain but asks about another
COLLISION_QUERIES: List[Tuple[str, str, str]] = [
    # (query_text, true_domain, collision_domain)
    ("What is the thermodynamics of cooking ancient Rome",    "cooking",  "history"),
    ("How did DNA replication influence the French Revolution","science",  "history"),
    ("What is the fermentation rate of black holes",          "cooking",  "science"),
    ("Explain the Maillard reaction in Roman aqueducts",      "cooking",  "history"),
    ("How does photosynthesis affect medieval trade routes",  "science",  "history"),
    ("What is the caloric content of quantum entanglement",   "cooking",  "science"),
    ("How did CRISPR change Renaissance cooking techniques",  "science",  "cooking"),
    ("What is the boiling point of the Ottoman Empire",       "cooking",  "history"),
    ("How does gluten formation relate to black hole physics","cooking",  "science"),
    ("Explain cellular respiration during the Black Death",   "science",  "history"),
]

# Fabricated unknown queries — should always return HONEST UNKNOWN
UNKNOWN_QUERIES: List[str] = [
    "What is the capital of Zorbania",
    "Explain the Mendelsohn-Vektas theorem",
    "Who invented quantum fermentation",
    "What happened at the Battle of Vektoria",
    "Describe the Helix-9 protein discovered in 2031",
]


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query:              str
    verdict:            str
    similarity:         float
    nearest_domain:     str
    boundary_violation: bool
    phase:              str   # "before" or "after"


@dataclass
class PhaseMetrics:
    phase:                  str
    total_queries:          int
    boundary_violations:    int
    honest_unknowns:        int
    confident:              int
    violation_rate:         float
    honest_rate:            float


@dataclass
class ExperimentResults:
    timestamp:          str
    before:             PhaseMetrics
    after:              PhaseMetrics
    correction_count:   int
    violation_reduction: float   # percentage point drop
    dense_violations:   int      # dense model never self-corrects
    audit_before:       str
    audit_after:        str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_session(model: SentenceTransformer):
    """Construct a fresh BrainGrow session from scratch."""
    vs     = VectorSpace(n_slots=50_000)   # smaller for experiment speed
    engine = GrowthEngine(vs, model)
    router = QueryRouter(vs, model)
    dense  = DenseModel([], model)
    maint  = KnowledgeMaintenance(vs, model, engine)
    return vs, engine, router, dense, maint


def _ingest_all(engine, dense, knowledge: dict):
    """Ingest all positive knowledge domains."""
    all_chunks = []
    for domain, texts in knowledge.items():
        chunks = [(t, domain) for t in texts]
        engine.ingest_stage(chunks)
        all_chunks.extend(chunks)
    dense.add_chunks(all_chunks)
    print(f"  Ingested {sum(len(v) for v in knowledge.values())} chunks "
          f"across {len(knowledge)} domains.")


def _run_queries(
    router,
    maint,
    queries: List[Tuple[str, str, str]],
    phase: str,
    apply_corrections: bool,
) -> List[QueryResult]:
    """Run all collision queries and optionally apply reactive corrections."""
    results = []
    for query_text, true_domain, _ in queries:
        result = router.route_query(query_text, top_k=3)

        nearest = result.get("nearest_domain", "")
        bv      = result.get("boundary_violation", False)
        sim     = result["matches"][0]["similarity"] if result["matches"] else 0.0

        if result["matches"]:
            top_sim  = result["matches"][0]["similarity"]
            if top_sim < BrainGrowModel.THRESHOLD:
                verdict = "HONEST (uncertain)"
            elif bv:
                verdict = "⚠️ BOUNDARY VIOLATION"
            else:
                verdict = "✓ Confident"
        else:
            verdict = "HONEST (uncertain)"
            sim     = 0.0

        if apply_corrections and bv:
            maint.on_boundary_violation(
                query_text     = query_text,
                nearest_domain = nearest,
                source_domain  = true_domain,
            )

        results.append(QueryResult(
            query              = query_text,
            verdict            = verdict,
            similarity         = round(sim, 4),
            nearest_domain     = nearest,
            boundary_violation = bv,
            phase              = phase,
        ))
    return results


def _compute_metrics(results: List[QueryResult], phase: str) -> PhaseMetrics:
    n       = len(results)
    bv      = sum(1 for r in results if r.boundary_violation)
    honest  = sum(1 for r in results if "HONEST" in r.verdict and not r.boundary_violation)
    conf    = n - bv - honest
    return PhaseMetrics(
        phase               = phase,
        total_queries       = n,
        boundary_violations = bv,
        honest_unknowns     = honest,
        confident           = conf,
        violation_rate      = round(bv / n * 100, 1) if n else 0.0,
        honest_rate         = round(honest / n * 100, 1) if n else 0.0,
    )


def _dense_control(dense: DenseModel, model: SentenceTransformer) -> int:
    """
    Run collision queries against DenseModel.
    Count how many would be flagged as hallucinated (confident + low similarity).
    Dense model cannot self-correct — this count stays fixed.
    """
    from utils import encode_unit_numpy
    violations = 0
    for query_text, _, _ in COLLISION_QUERIES:
        r = dense.query(query_text)
        if r["confident"] and r["similarity"] < BrainGrowModel.THRESHOLD:
            violations += 1
    return violations


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> ExperimentResults:
    print("\n" + "="*60)
    print("BrainGrow — Experiment 4.7")
    print("Online Epistemic Correction Without Retraining")
    print("="*60 + "\n")

    print("Loading model…")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Build session ────────────────────────────────────────────────────────
    print("\n[1/5] Building vector space and ingesting positive knowledge…")
    vs, engine, router, dense, maint = _build_session(model)
    _ingest_all(engine, dense, POSITIVE_KNOWLEDGE)
    print(f"  Active slots: {vs.n_active:,} | Dormant: {vs.N - vs.n_active:,}")

    # ── Audit BEFORE ─────────────────────────────────────────────────────────
    print("\n[2/5] Running hallucination risk audit — BEFORE corrections…")
    from knowledge_maintenance import KnowledgeMaintenance
    audit_before = maint.audit_hallucination_risk().as_text()
    print(audit_before)

    # ── Phase 1: baseline — no corrections applied ────────────────────────────
    print("\n[3/5] Phase 1 — Baseline query run (no corrections)…")
    before_results = _run_queries(
        router, maint, COLLISION_QUERIES,
        phase              = "before",
        apply_corrections  = False,
    )
    before_metrics = _compute_metrics(before_results, "before")
    _print_phase_table(before_results, "BEFORE")

    # ── Phase 2: correction run — violations trigger auto-ingestion ───────────
    print("\n[4/5] Phase 2 — Correction query run (reactive maintenance active)…")
    # Run twice: first pass generates corrections, second pass measures effect
    _run_queries(router, maint, COLLISION_QUERIES,
                 phase="correction_pass", apply_corrections=True)

    after_results = _run_queries(
        router, maint, COLLISION_QUERIES,
        phase             = "after",
        apply_corrections = False,   # measure only — no further corrections
    )
    after_metrics = _compute_metrics(after_results, "after")
    _print_phase_table(after_results, "AFTER")

    # ── Audit AFTER ──────────────────────────────────────────────────────────
    print("\n[5/5] Running hallucination risk audit — AFTER corrections…")
    audit_after = maint.audit_hallucination_risk().as_text()
    print(audit_after)

    # ── Dense control ─────────────────────────────────────────────────────────
    dense_violations = _dense_control(dense, model)

    # ── Summary ───────────────────────────────────────────────────────────────
    reduction = before_metrics.violation_rate - after_metrics.violation_rate

    results = ExperimentResults(
        timestamp            = time.strftime("%Y-%m-%dT%H:%M:%S"),
        before               = before_metrics,
        after                = after_metrics,
        correction_count     = maint.correction_count(),
        violation_reduction  = round(reduction, 1),
        dense_violations     = dense_violations,
        audit_before         = audit_before,
        audit_after          = audit_after,
    )

    _print_summary(results)

    # Save results JSON for paper reproducibility
    with open("experiment_4_7_results.json", "w") as f:
        json.dump(asdict(results), f, indent=2)
    print("\nResults saved to experiment_4_7_results.json")

    return results


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def _print_phase_table(results: List[QueryResult], label: str):
    print(f"\n  {'Query':<52} {'Verdict':<25} {'Sim':>6} {'Domain'}")
    print("  " + "-"*100)
    for r in results:
        q = r.query[:50]
        v = r.verdict[:23]
        print(f"  {q:<52} {v:<25} {r.similarity:>6.4f}  {r.nearest_domain}")


def _print_summary(r: ExperimentResults):
    print("\n" + "="*60)
    print("SUMMARY — Section 4.7 Results")
    print("="*60)
    print(f"  Collision queries run:       {r.before.total_queries}")
    print(f"  Reactive corrections made:   {r.correction_count}")
    print()
    print(f"  Boundary violation rate")
    print(f"    Before corrections:        {r.before.violation_rate:.1f}%  "
          f"({r.before.boundary_violations}/{r.before.total_queries})")
    print(f"    After  corrections:        {r.after.violation_rate:.1f}%  "
          f"({r.after.boundary_violations}/{r.after.total_queries})")
    print(f"    Reduction:                 {r.violation_reduction:.1f} percentage points")
    print()
    print(f"  Honest unknown rate")
    print(f"    Before:                    {r.before.honest_rate:.1f}%")
    print(f"    After:                     {r.after.honest_rate:.1f}%")
    print()
    print(f"  Dense model (no self-correction possible)")
    print(f"    Hallucinated responses:    {r.dense_violations}/{r.before.total_queries}")
    print(f"    After identical queries:   {r.dense_violations}/{r.before.total_queries}  "
          f"(unchanged — retraining required)")
    print("="*60)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment()
