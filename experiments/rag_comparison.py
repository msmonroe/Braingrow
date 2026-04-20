"""
rag_comparison.py — Main entry point for the BrainGrow vs flat-threshold
comparison experiment.

Run from the braingrow/ repo root:

    python -m experiments.rag_comparison

Or:

    python experiments/rag_comparison.py

What it does:
  1. Generates a deterministic 100-query set (seed=42, four buckets of 25)
  2. Ingests the SAMPLE_CORPUS into three systems:
       - BrainGrow           (the paper's architecture)
       - TorchFlatThreshold  (flat torch tensor + 0.60 threshold, no lifecycle)
       - FAISSFlatThreshold  (IndexFlatIP + 0.60 threshold — canonical RAG)
  3. Runs all 100 queries against each system
  4. Prints per-system scorecards, pairwise verdict agreement, and per-query
     disagreement details
  5. Writes rag_comparison_results.json with the full per-query output

The number that matters most: BrainGrow vs TorchFlatThreshold agreement rate.
If it's ~100%, the developmental machinery is not contributing to the
paper's headline claim and the framing needs to be revised.

Corpus note: SAMPLE_CORPUS below is a minimal 30-chunk, 3-domain placeholder.
Replace it with your actual repo sample corpus if you have one stored as
text (e.g., by loading from files/science.txt, files/history.txt, etc.).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure we can import from the braingrow repo root when invoked via either
# `python -m experiments.rag_comparison` or `python experiments/rag_comparison.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Third-party / BrainGrow imports (must follow sys.path mutation above)
from sentence_transformers import SentenceTransformer

from baselines.flat_threshold import (
    DEFAULT_ENCODER, DEFAULT_THRESHOLD,
    TorchFlatThreshold, FAISSFlatThreshold,
)
from evaluation import (
    generate_queries, SEED,
    run_baseline, run_braingrow,
    score_system, agreement_matrix, render_agreement,
)

# BrainGrow modules — these are flat-imported from the repo root.
from vector_space import VectorSpace
from growth_engine import GrowthEngine
from query_router import QueryRouter


# --------------------------------------------------------------------------
# Sample corpus — 3 domains × 10 chunks
# --------------------------------------------------------------------------
# Replace with your actual repo corpus if preferable. Kept inline so this
# file is self-contained and reproducible without reference to external
# files.
SAMPLE_CORPUS: List[Tuple[str, str]] = [
    # --- science (10) ---
    ("Photosynthesis is the process by which green plants convert sunlight into chemical energy stored as glucose.", "science"),
    ("Cells are the basic structural and functional units of all living organisms.", "science"),
    ("DNA carries the genetic instructions used in the growth, development, and reproduction of living organisms.", "science"),
    ("Newton's second law states that the force acting on an object equals its mass times acceleration.", "science"),
    ("The theory of evolution explains how species change over time through natural selection.", "science"),
    ("Atoms are the basic units of matter, consisting of protons, neutrons, and electrons.", "science"),
    ("Entropy is a measure of disorder in a thermodynamic system, and it tends to increase over time.", "science"),
    ("Mitosis is the process by which a eukaryotic cell divides into two genetically identical daughter cells.", "science"),
    ("The immune system defends the body against pathogens using white blood cells and antibodies.", "science"),
    ("Einstein's theory of general relativity describes gravity as the curvature of spacetime caused by mass.", "science"),

    # --- history (10) ---
    ("World War II ended in 1945 with the surrender of Japan following the atomic bombings of Hiroshima and Nagasaki.", "history"),
    ("Napoleon Bonaparte was a French military leader who rose to prominence during the French Revolution and later crowned himself Emperor.", "history"),
    ("The Roman Empire was one of the largest and most influential civilizations in the ancient world, lasting from 27 BC to 476 AD.", "history"),
    ("The French Revolution began in 1789 and led to the fall of the monarchy and the rise of democratic principles in France.", "history"),
    ("Julius Caesar was a Roman general and statesman whose assassination in 44 BC triggered the end of the Roman Republic.", "history"),
    ("The fall of Rome in 476 AD was caused by a combination of political instability, economic decline, and barbarian invasions.", "history"),
    ("The Cold War was a geopolitical tension between the United States and the Soviet Union that lasted from 1947 to 1991.", "history"),
    ("The Renaissance was a cultural movement that began in Italy in the 14th century and spread across Europe over the following centuries.", "history"),
    ("The Magna Carta, signed in 1215, established the principle that the king was subject to the law.", "history"),
    ("The Industrial Revolution transformed manufacturing, transportation, and agriculture beginning in late-18th-century Britain.", "history"),

    # --- cooking (10) ---
    ("Caramelizing onions involves slowly cooking them over low heat until their natural sugars brown and develop a sweet flavor.", "cooking"),
    ("The Maillard reaction is a chemical reaction between amino acids and reducing sugars that gives browned food its distinctive flavor.", "cooking"),
    ("Bread rises because yeast consumes sugars and produces carbon dioxide gas, which gets trapped in the gluten network.", "cooking"),
    ("An emulsion in cooking combines two liquids that normally don't mix, such as oil and vinegar in a vinaigrette.", "cooking"),
    ("Searing a steak requires a very hot pan to create a flavorful brown crust through the Maillard reaction.", "cooking"),
    ("Deglazing a pan involves adding liquid to dissolve the flavorful browned bits stuck to the bottom after searing.", "cooking"),
    ("A roux is a mixture of equal parts flour and fat, cooked together and used to thicken sauces and soups.", "cooking"),
    ("Baking typically involves moist heat in an enclosed oven, while roasting uses dry heat at higher temperatures.", "cooking"),
    ("Fermentation is the metabolic process by which microorganisms convert sugars into alcohol, acids, or gases.", "cooking"),
    ("Braising is a cooking technique that combines searing and slow cooking in liquid, ideal for tough cuts of meat.", "cooking"),
]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(
    threshold: float = DEFAULT_THRESHOLD,
    encoder_name: str = DEFAULT_ENCODER,
    n_slots: int = 200_000,
    output_path: str = "rag_comparison_results.json",
    skip_faiss: bool = False,
) -> None:
    print(f"BrainGrow RAG comparison — seed={SEED}, threshold={threshold}, encoder={encoder_name}")
    print(f"Corpus: {len(SAMPLE_CORPUS)} chunks across "
          f"{len(set(d for _, d in SAMPLE_CORPUS))} domains")
    print()

    # --- Shared encoder so we're not loading the model three times ------
    print("[1/5] Loading shared encoder...")
    encoder = SentenceTransformer(encoder_name)

    # --- Build BrainGrow ------------------------------------------------
    print("[2/5] Building BrainGrow and ingesting corpus...")
    vs = VectorSpace(n_slots=n_slots, dimensions=384)
    growth = GrowthEngine(vs, model=encoder)
    growth.ingest_stage(SAMPLE_CORPUS, batch_size=32)
    router = QueryRouter(vs, model=encoder)
    print(f"       active slots: {vs.n_active}  (expected ≈ {len(SAMPLE_CORPUS)})")

    # --- Build flat baselines ------------------------------------------
    print("[3/5] Building flat-threshold baselines and ingesting corpus...")
    torch_baseline = TorchFlatThreshold(encoder=encoder, threshold=threshold)
    torch_baseline.ingest(SAMPLE_CORPUS)

    faiss_baseline = None
    if not skip_faiss:
        try:
            faiss_baseline = FAISSFlatThreshold(encoder=encoder, threshold=threshold)
            faiss_baseline.ingest(SAMPLE_CORPUS)
        except ImportError as e:
            print(f"       FAISS unavailable ({e}); skipping FAISS baseline.")
            faiss_baseline = None

    # --- Generate queries ----------------------------------------------
    print("[4/5] Generating deterministic query set...")
    queries = generate_queries(seed=SEED)
    print(f"       {len(queries)} queries across 4 buckets")

    # --- Run ------------------------------------------------------------
    print("[5/5] Running all systems against query set...\n")
    results = {
        "BrainGrow": run_braingrow(router, queries, threshold=threshold),
        "TorchFlatThreshold": run_baseline("TorchFlatThreshold", torch_baseline, queries),
    }
    if faiss_baseline is not None:
        results["FAISSFlatThreshold"] = run_baseline(
            "FAISSFlatThreshold", faiss_baseline, queries
        )

    # --- Score ----------------------------------------------------------
    for name, res in results.items():
        print(score_system(name, res).render())
        print()

    # --- Agreement ------------------------------------------------------
    agreement = agreement_matrix(results)
    print(render_agreement(agreement))

    # --- Persist --------------------------------------------------------
    print(f"\nWriting per-query results to {output_path}")
    serializable = {
        "seed": SEED,
        "threshold": threshold,
        "encoder": encoder_name,
        "corpus_size": len(SAMPLE_CORPUS),
        "systems": {
            name: [
                {
                    "query": r.query_text,
                    "bucket": r.bucket,
                    "expected": r.expected,
                    "verdict": r.verdict,
                    "similarity": round(r.similarity, 4),
                    "nearest_label": r.nearest_label,
                    "nearest_domain": r.nearest_domain,
                    "correct": r.correct,
                }
                for r in res
            ]
            for name, res in results.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    # --- Bottom-line interpretation ------------------------------------
    bg_vs_torch = agreement.get(("BrainGrow", "TorchFlatThreshold"))
    if bg_vs_torch is not None:
        rate = bg_vs_torch["agree_rate"]
        print("\n" + "=" * 70)
        print("BOTTOM LINE — BrainGrow vs TorchFlatThreshold verdict agreement:")
        print(f"  {rate:.2%}  ({int(bg_vs_torch['agree'])}/{int(bg_vs_torch['n'])} queries)")
        print("=" * 70)
        if rate >= 0.99:
            print("Interpretation: BrainGrow produces the same verdict as a flat")
            print("vector store + threshold on essentially every query. The")
            print("developmental machinery (activation scores, decay, pruning,")
            print("pre-allocation) is not contributing to the abstention result.")
            print("The paper's 'structural epistemic boundaries' framing is not")
            print("supported by this evidence. Recommend revising the framing.")
        elif rate >= 0.90:
            print("Interpretation: Near-total agreement, but not quite identical.")
            print("Inspect the disagreements above — they're where the")
            print("developmental machinery actually changes behavior (if anywhere).")
        else:
            print("Interpretation: Meaningful verdict divergence exists. Inspect")
            print("the disagreement details to understand what BrainGrow's")
            print("machinery is doing differently, and whether it's an improvement.")


if __name__ == "__main__":
    main()
