"""
epistemic.py — Three-tier epistemic state for BrainGrow.

Separates the epistemic classification logic from query routing so it is
independently testable and formally describable in the paper.

The three states are structurally enforced by the architecture, not trained:

    CONFIDENT       — query embedding finds a high-similarity match (>=
                      CONFIDENCE_THRESHOLD) within the active vector space.
                      The system has seen this concept.

    HONEST_UNKNOWN  — query embedding finds no active slot exceeding the
                      confidence threshold. The system has not seen this
                      concept. It does not guess.

    OUT_OF_BOUNDS   — query embedding's nearest active match belongs to a
                      registered negative domain. The concept exists in the
                      space but the combination or context is invalid.

These states emerge from two hyperparameters:
    - CONFIDENCE_THRESHOLD: minimum cosine similarity for a CONFIDENT result
    - Negative domain registry: set of domain labels marked as boundaries

Neither is trained. Both are explicit architectural decisions, which is the
point: epistemic honesty is enforced structurally, not learned post-hoc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class EpistemicState(Enum):
    CONFIDENT = "confident"
    HONEST_UNKNOWN = "honest_unknown"
    OUT_OF_BOUNDS = "out_of_bounds"


@dataclass
class EpistemicResult:
    state: EpistemicState
    top_similarity: Optional[float]      # cosine sim of best match, or None
    nearest_domain: Optional[str]        # domain of best match, or None
    matches: List[dict]                  # raw match list from QueryRouter
    active_count: int
    dormant_count: int
    confidence_threshold: float          # threshold used for this result
    explanation: str                     # human-readable rationale

    @property
    def is_confident(self) -> bool:
        return self.state == EpistemicState.CONFIDENT

    @property
    def is_honest_unknown(self) -> bool:
        return self.state == EpistemicState.HONEST_UNKNOWN

    @property
    def is_out_of_bounds(self) -> bool:
        return self.state == EpistemicState.OUT_OF_BOUNDS


# Default confidence threshold. Cosine similarity in 384-dim space
# with all-MiniLM-L6-v2: values above ~0.75 indicate strong semantic overlap.
# Values below ~0.4 are effectively orthogonal (unrelated concepts).
DEFAULT_CONFIDENCE_THRESHOLD = 0.60


def classify(
    router_result: dict,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> EpistemicResult:
    """
    Classify a QueryRouter result into one of the three epistemic states.

    Parameters
    ----------
    router_result : dict
        The dict returned by QueryRouter.route_query().
    confidence_threshold : float
        Minimum cosine similarity to classify a result as CONFIDENT.
        Default 0.60; tune based on encoder and domain density.

    Returns
    -------
    EpistemicResult
    """
    matches = router_result.get("matches", [])
    active_count = router_result.get("active_count", 0)
    dormant_count = router_result.get("dormant_count", 0)
    boundary_violation = router_result.get("boundary_violation", False)
    nearest_domain = router_result.get("nearest_domain", None) or None

    top_similarity = matches[0]["similarity"] if matches else None

    # --- OUT_OF_BOUNDS: concept exists but is in a prohibited domain ---
    if boundary_violation and matches:
        return EpistemicResult(
            state=EpistemicState.OUT_OF_BOUNDS,
            top_similarity=top_similarity,
            nearest_domain=nearest_domain,
            matches=matches,
            active_count=active_count,
            dormant_count=dormant_count,
            confidence_threshold=confidence_threshold,
            explanation=(
                f"Nearest active slot (sim={top_similarity:.3f}) belongs to "
                f"registered negative domain '{nearest_domain}'. "
                f"Concept exists in vector space but is marked out-of-bounds."
            ),
        )

    # --- HONEST_UNKNOWN: no match exceeds confidence threshold ---
    if not matches or top_similarity is None or top_similarity < confidence_threshold:
        return EpistemicResult(
            state=EpistemicState.HONEST_UNKNOWN,
            top_similarity=top_similarity,
            nearest_domain=nearest_domain,
            matches=matches,
            active_count=active_count,
            dormant_count=dormant_count,
            confidence_threshold=confidence_threshold,
            explanation=(
                f"Best match similarity {top_similarity:.3f} is below confidence "
                f"threshold {confidence_threshold:.2f}. "
                f"Query is outside the system's known space ({active_count} active slots)."
            ) if top_similarity is not None else (
                f"No active slots in vector space ({active_count} active). "
                f"System has not ingested any knowledge yet."
            ),
        )

    # --- CONFIDENT: high-similarity match in a non-negative domain ---
    return EpistemicResult(
        state=EpistemicState.CONFIDENT,
        top_similarity=top_similarity,
        nearest_domain=nearest_domain,
        matches=matches,
        active_count=active_count,
        dormant_count=dormant_count,
        confidence_threshold=confidence_threshold,
        explanation=(
            f"Top match similarity {top_similarity:.3f} >= threshold "
            f"{confidence_threshold:.2f} in domain '{nearest_domain}'. "
            f"System has seen this concept."
        ),
    )


def summarize(result: EpistemicResult) -> str:
    """Return a one-line human-readable summary suitable for UI display."""
    if result.state == EpistemicState.CONFIDENT:
        return (
            f"[CONFIDENT] {result.nearest_domain} "
            f"(sim={result.top_similarity:.3f})"
        )
    elif result.state == EpistemicState.HONEST_UNKNOWN:
        sim_str = f", best sim={result.top_similarity:.3f}" if result.top_similarity else ""
        return f"[HONEST UNKNOWN] Query outside known space{sim_str}"
    else:
        return (
            f"[OUT OF BOUNDS] Nearest domain='{result.nearest_domain}' "
            f"is a registered boundary (sim={result.top_similarity:.3f})"
        )
