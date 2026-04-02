"""
tinystories_loader.py — Data pipeline for the TinyStories scale experiment.

Loads the roneneldan/TinyStories dataset from Hugging Face, samples a
configurable number of stories, splits them into 2-sentence chunks, and
returns them in BrainGrow's (text, domain_label) format.

Requires:
    pip install datasets

Progressive scaling plan (run in this order):
    Stage A — Smoke test   : 1,000  chunks  (~2 min)
    Stage B — Small scale  : 10,000 chunks  (~5 min)
    Stage C — Full scale   : 100,000 sample (~20-30 min)
"""

from __future__ import annotations

from typing import List, Tuple


def _check_datasets_available() -> bool:
    """Return True if the `datasets` package is importable."""
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


def load_tinystories(sample_size: int = 100_000) -> object:
    """
    Download and sample *sample_size* stories from roneneldan/TinyStories.

    Returns the sampled Hugging Face Dataset object.

    Parameters
    ----------
    sample_size : number of stories to sample (default 100,000)
    """
    if not _check_datasets_available():
        raise ImportError(
            "The 'datasets' package is required for TinyStories.\n"
            "Install it with:  pip install datasets"
        )
    import random
    from datasets import load_dataset  # type: ignore

    print(f"Loading TinyStories dataset (sampling {sample_size:,} stories)…")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    total = len(dataset)
    actual_size = min(sample_size, total)
    indices = random.sample(range(total), actual_size)
    sample = dataset.select(indices)
    print(f"Sampled {actual_size:,} stories from {total:,} total.")
    return sample


def chunk_stories(
    dataset,
    max_chunks: int = 200_000,
    domain_label: str = "stories",
) -> List[Tuple[str, str]]:
    """
    Split TinyStories entries into 2-sentence chunks.

    Each story is split on '. ' boundaries and paired into overlapping
    2-sentence windows.  Fragments under 50 characters are discarded.

    Parameters
    ----------
    dataset     : Hugging Face Dataset (from load_tinystories)
    max_chunks  : cap on total chunks returned (default 200,000)
    domain_label: BrainGrow domain tag assigned to every chunk

    Returns
    -------
    List of (text_chunk, domain_label) tuples ready for ingest_stage_batched()
    """
    chunks: List[Tuple[str, str]] = []
    for item in dataset:
        text = item["text"].strip()
        sentences = text.split(". ")
        for i in range(0, len(sentences) - 1, 2):
            chunk = sentences[i] + ". " + sentences[i + 1]
            if len(chunk) > 50:
                chunks.append((chunk, domain_label))
            if len(chunks) >= max_chunks:
                return chunks[:max_chunks]
    return chunks[:max_chunks]


def prepare_experiment(
    sample_size: int = 100_000,
    max_chunks: int = 200_000,
    domain_label: str = "stories",
) -> List[Tuple[str, str]]:
    """
    Convenience wrapper: load + chunk in one call.

    Parameters
    ----------
    sample_size  : stories to sample from Hugging Face
    max_chunks   : maximum chunks to return after splitting
    domain_label : BrainGrow domain tag

    Returns
    -------
    List of (text_chunk, domain_label) ready for GrowthEngine.ingest_stage_batched()
    """
    dataset = load_tinystories(sample_size)
    chunks = chunk_stories(dataset, max_chunks=max_chunks, domain_label=domain_label)
    print(f"Prepared {len(chunks):,} chunks for ingestion.")
    return chunks


# ---------------------------------------------------------------------------
# Stage presets matching the progressive scaling plan
# ---------------------------------------------------------------------------
STAGE_PRESETS = {
    "Stage A — Smoke test (1k chunks)":    {"sample_size": 2_000,   "max_chunks": 1_000},
    "Stage B — Small scale (10k chunks)":  {"sample_size": 15_000,  "max_chunks": 10_000},
    "Stage C — Full scale (100k sample)":  {"sample_size": 100_000, "max_chunks": 200_000},
}
