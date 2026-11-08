"""
rag/generation.py — Part 1 Task 4: grounded generation.

This generator never calls an external LLM. It is a deterministic, rule-based
function (consistent with the project-wide MOCK_LLM mode -- grounded
generation is explicitly one of the "agent's language-model calls" the brief
says must default to deterministic, zero-network behaviour) that:

  1. Looks at the top-1 retrieved chunk's similarity score.
  2. If similarity is below the calibrated REFUSAL_THRESHOLD, returns the
     fixed "I don't know" fallback sentence -- no guessing.
  3. Otherwise, composes an answer using ONLY the retrieved chunk text,
     citing the source KB id, following the same labelled-excerpt +
     use-only-this-evidence pattern as the course's grounded prompts.
"""

from __future__ import annotations

REFUSAL = "I do not have enough information in the provided policy excerpts to answer that."

# Calibrated empirically in scripts/calibrate_threshold.py -- see that script's
# printed output and README.md for the measured in-scope/out-of-scope
# similarity clusters that produced this value.
REFUSAL_THRESHOLD = 0.5781


def similarity_from_distance(distance: float) -> float:
    """Convert Chroma's cosine distance (1 - cosine_similarity) back to similarity."""
    return 1.0 - distance


def generate_grounded_answer(
    query: str,
    retrieved_chunks: list[dict],
    threshold: float = REFUSAL_THRESHOLD,
) -> str:
    """Compose a deterministic, evidence-only answer from retrieved chunks."""
    if not retrieved_chunks:
        return REFUSAL

    top_similarity = similarity_from_distance(retrieved_chunks[0]["distance"])
    if top_similarity < threshold:
        return REFUSAL

    # Use the top chunk as the primary answer; if a second chunk is from a
    # different parent doc and also above threshold, append it as extra
    # context (mirrors the course's multi-excerpt grounded prompt pattern).
    seen_parents = set()
    parts = []
    for chunk in retrieved_chunks:
        sim = similarity_from_distance(chunk["distance"])
        if sim < threshold:
            continue
        parent = chunk["metadata"].get("parent_id")
        if parent in seen_parents:
            continue
        seen_parents.add(parent)
        title = chunk["metadata"].get("title", parent)
        parts.append(f"According to Ola policy [{parent}: {title}]: {chunk['text']}")
        if len(seen_parents) >= 2:
            break

    return " ".join(parts) if parts else REFUSAL
