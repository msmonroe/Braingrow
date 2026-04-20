"""
fabricated_queries.py — Deterministic 100-query generator for RAG comparison.

Seeded with SEED=42 for full reproducibility. Four buckets of 25 queries each:

  PURE_NONSENSE       — made-up proper nouns, no semantic anchor in any domain
  LEXICAL_OVERLAP     — fake concepts that share tokens with real domain vocab
                        (the "quantum fermentation" class — where threshold
                        calibration actually matters)
  IN_DOMAIN           — real, well-known questions in science/history/cooking
  NEAR_DOMAIN         — real concepts adjacent to but probably not in the
                        30-chunk sample corpus (edge-of-calibration cases)

Each query carries an `expected` label: "CONFIDENT" or "HONEST_UNKNOWN".
Near-domain is labeled "HONEST_UNKNOWN" because the 30-chunk corpus is tiny;
at larger corpus sizes these expectations should be reviewed.

IMPORTANT: Expectations for NEAR_DOMAIN are deliberately conservative. If
BrainGrow or a baseline returns CONFIDENT on a near-domain query with high
actual similarity, that's not automatically wrong — it means the threshold
is generous, and the precision/recall numbers will reflect that. Inspect
the per-query output before treating these as failures.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


SEED = 42
QUERIES_PER_BUCKET = 25


# --------------------------------------------------------------------------
# Query schema
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FabricatedQuery:
    text: str
    bucket: str          # "PURE_NONSENSE" | "LEXICAL_OVERLAP" | "IN_DOMAIN" | "NEAR_DOMAIN"
    expected: str        # "CONFIDENT" | "HONEST_UNKNOWN"
    note: str = ""       # optional provenance hint

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "bucket": self.bucket,
            "expected": self.expected,
            "note": self.note,
        }


# --------------------------------------------------------------------------
# Word banks
# --------------------------------------------------------------------------
# Fabricated proper nouns — pronounceable but unmistakably not real.
# Any overlap with real names is coincidental; if you find one, replace it.
_FAKE_PLACES = [
    "Zorbania", "Vektoria", "Mendelport", "Kessarine", "Thalmuria",
    "Quirindor", "Balthaven", "Novastral", "Trelavia", "Xintharia",
    "Palorium", "Drovenia", "Grelthwood", "Vorathia", "Morspike",
]
_FAKE_PEOPLE = [
    "Bartholomew Kesslin", "Ilse Mendelsohn-Vek", "Corvin Thrax",
    "Avalon Drosselmeyer", "Niamh Quellstrom", "Radu Petrovsky-Lune",
    "Helga Brennenkamp", "Osric Fennwick", "Marisol Chantico-Vel",
    "Tarquin Ashvale", "Lyra Orsini-Kul", "Dmitri Valenburg",
]
_FAKE_THINGS = [
    "Vektas theorem", "Kesslin constant", "Drosselmeyer hypothesis",
    "Mendelsohn-Vek principle", "Orsini transform", "Quellstrom paradox",
    "Thrax conjecture", "Fennwick inequality", "Chantico regularizer",
    "Valenburg lemma", "Ashvale projection", "Brennenkamp identity",
]
_FAKE_EVENTS = [
    "Battle of Vektoria", "Siege of Mendelport", "Thalmurian Uprising",
    "Kessarine Accords", "Novastral Schism", "Second Xintharian War",
    "Treaty of Palorium", "Drovenia Succession", "Grelthwood Massacre",
    "Morspike Insurrection",
]

# Real tokens from each domain — used to build lexical-overlap traps.
# These are deliberately domain-canonical so an encoder will score the
# trap query with non-trivial similarity to real chunks.
_SCIENCE_TOKENS = [
    "quantum", "photosynthesis", "mitochondria", "cellular", "molecular",
    "thermodynamic", "relativistic", "chromosome", "enzyme", "catalyst",
]
_HISTORY_TOKENS = [
    "dynasty", "empire", "revolution", "treaty", "monarchy",
    "conquest", "federation", "succession", "inquisition", "reformation",
]
_COOKING_TOKENS = [
    "fermentation", "emulsion", "braise", "sauté", "caramelization",
    "deglaze", "reduction", "proofing", "julienne", "tempering",
]

# Fake suffixes that make domain tokens unmistakably fabricated when joined.
_FAKE_SUFFIXES = [
    "-theta", "-prime", "-null", "-delta", "-inverse",
    " of Kesslin", " of the Second Order", " paradigm",
    " phenomenon", " procedure", " coefficient",
]


# --------------------------------------------------------------------------
# Bucket generators
# --------------------------------------------------------------------------
def _gen_pure_nonsense(rng: random.Random, n: int) -> List[FabricatedQuery]:
    """Fabricated queries with no semantic anchor in any real domain."""
    templates = [
        ("What is the capital of {place}?", _FAKE_PLACES),
        ("Who invented the {thing}?", _FAKE_THINGS),
        ("Explain the {thing}.", _FAKE_THINGS),
        ("What happened at the {event}?", _FAKE_EVENTS),
        ("Describe the legacy of {person}.", _FAKE_PEOPLE),
        ("When did the {event} take place?", _FAKE_EVENTS),
        ("What language is spoken in {place}?", _FAKE_PLACES),
        ("Who was {person}?", _FAKE_PEOPLE),
    ]
    out: List[FabricatedQuery] = []
    while len(out) < n:
        tpl, bank = rng.choice(templates)
        token = rng.choice(bank)
        text = tpl.format(
            place=token, thing=token, event=token, person=token,
        )
        if any(q.text == text for q in out):
            continue
        out.append(FabricatedQuery(
            text=text,
            bucket="PURE_NONSENSE",
            expected="HONEST_UNKNOWN",
            note="fabricated proper noun; no domain anchor",
        ))
    return out


def _gen_lexical_overlap(rng: random.Random, n: int) -> List[FabricatedQuery]:
    """Fake concepts that share tokens with real domain vocabulary."""
    templates = [
        "Who invented {domain_token}{suffix}?",
        "Explain the theory of {domain_token}{suffix}.",
        "What is the {domain_token}{suffix}?",
        "How does {domain_token}{suffix} work?",
        "Describe the history of {domain_token}{suffix}.",
    ]
    all_tokens = _SCIENCE_TOKENS + _HISTORY_TOKENS + _COOKING_TOKENS
    out: List[FabricatedQuery] = []
    while len(out) < n:
        tpl = rng.choice(templates)
        token = rng.choice(all_tokens)
        suffix = rng.choice(_FAKE_SUFFIXES)
        text = tpl.format(domain_token=token, suffix=suffix)
        if any(q.text == text for q in out):
            continue
        out.append(FabricatedQuery(
            text=text,
            bucket="LEXICAL_OVERLAP",
            expected="HONEST_UNKNOWN",
            note=f"lexical trap on token '{token}'",
        ))
    return out


def _gen_in_domain(rng: random.Random, n: int) -> List[FabricatedQuery]:
    """
    Real, well-known questions in each domain. Intentionally broad so they
    will likely hit whatever is in the 30-chunk sample corpus.

    If the sample corpus is narrow (e.g., only Renaissance history, only
    cell biology), some of these will land in NEAR_DOMAIN territory
    instead and the metrics will reflect that honestly.
    """
    # Hand-curated rather than templated — these need to actually make sense.
    science = [
        "What is photosynthesis?",
        "How does DNA replicate?",
        "What is Newton's second law?",
        "What causes gravity?",
        "How do cells divide?",
        "What is the theory of evolution?",
        "What is an atom?",
        "How does the immune system work?",
        "What is entropy?",
    ]
    history = [
        "When did World War II end?",
        "Who was Napoleon Bonaparte?",
        "What was the Roman Empire?",
        "When did the French Revolution happen?",
        "Who was Julius Caesar?",
        "What caused the fall of Rome?",
        "What was the Cold War?",
        "When did the Renaissance begin?",
    ]
    cooking = [
        "How do you caramelize onions?",
        "What is the Maillard reaction?",
        "How do you make bread rise?",
        "What is an emulsion in cooking?",
        "How do you sear a steak?",
        "What does it mean to deglaze a pan?",
        "How do you make a roux?",
        "What is the difference between baking and roasting?",
    ]
    pool = [(q, "science") for q in science] + \
           [(q, "history") for q in history] + \
           [(q, "cooking") for q in cooking]
    rng.shuffle(pool)
    out = [
        FabricatedQuery(
            text=q, bucket="IN_DOMAIN", expected="CONFIDENT",
            note=f"canonical {domain} question",
        )
        for q, domain in pool[:n]
    ]
    return out


def _gen_near_domain(rng: random.Random, n: int) -> List[FabricatedQuery]:
    """
    Real, specific concepts adjacent to the domains but probably not in a
    30-chunk sample. These are the interesting edge cases — where threshold
    calibration starts to matter.

    Expected label is HONEST_UNKNOWN because the 30-chunk corpus cannot
    reasonably cover these. At larger corpus scale this labeling should
    be revisited.
    """
    near_science = [
        "What is CRISPR-Cas9 gene editing?",
        "Explain quantum entanglement in layman's terms.",
        "What is a tardigrade?",
        "How do mRNA vaccines work?",
        "What is dark matter?",
        "How does CRISPR differ from traditional genetic engineering?",
        "What are prions and how do they differ from viruses?",
        "Explain the Higgs boson and why it matters.",
        "What is epigenetic inheritance?",
    ]
    near_history = [
        "What was the Treaty of Westphalia?",
        "Who was Cyrus the Great?",
        "What happened at the Defenestration of Prague?",
        "What was the Hundred Years' War?",
        "Who were the Picts?",
        "What was the significance of the Edict of Nantes?",
        "What was the Taiping Rebellion?",
        "Who was Eleanor of Aquitaine?",
        "What caused the collapse of the Bronze Age civilizations?",
    ]
    near_cooking = [
        "How do you make a proper beurre blanc?",
        "What is the technique for confiting duck?",
        "How do you clarify butter into ghee?",
        "What is spherification in molecular gastronomy?",
        "How do you make kimchi at home?",
        "What is the sous-vide method for short ribs?",
        "Explain the Japanese cutting technique 'usuzukuri'.",
        "What is the Neapolitan ragu technique?",
        "How do you temper chocolate using the tabling method?",
    ]
    pool = near_science + near_history + near_cooking
    rng.shuffle(pool)
    out = [
        FabricatedQuery(
            text=q, bucket="NEAR_DOMAIN", expected="HONEST_UNKNOWN",
            note="real concept, probably absent from 30-chunk sample",
        )
        for q in pool[:n]
    ]
    return out


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def generate_queries(seed: int = SEED) -> List[FabricatedQuery]:
    """
    Return a deterministic 100-query set under the given seed.

    Ordering is NOT shuffled across buckets — each bucket is contiguous in
    the returned list. This makes per-bucket inspection trivial.
    """
    rng = random.Random(seed)
    queries: List[FabricatedQuery] = []
    queries.extend(_gen_pure_nonsense(rng, QUERIES_PER_BUCKET))
    queries.extend(_gen_lexical_overlap(rng, QUERIES_PER_BUCKET))
    queries.extend(_gen_in_domain(rng, QUERIES_PER_BUCKET))
    queries.extend(_gen_near_domain(rng, QUERIES_PER_BUCKET))
    return queries


if __name__ == "__main__":
    qs = generate_queries()
    print(f"Generated {len(qs)} queries with seed={SEED}\n")
    current_bucket = None
    for i, q in enumerate(qs):
        if q.bucket != current_bucket:
            current_bucket = q.bucket
            print(f"\n--- {current_bucket} (expected={q.expected}) ---")
        print(f"{i:3d}. {q.text}")
