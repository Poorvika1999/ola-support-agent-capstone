"""
scripts/calibrate_threshold.py — Part 1 Task 4: empirically calibrate the
"I don't know" similarity threshold.

Measures top-1 cosine similarity (1 - distance) for >=3 in-scope queries and
>=2 deliberately out-of-scope queries against the fixed-overlap collection,
prints both clusters, and reports the midpoint cutoff actually observed --
not an untested preset.
"""

from __future__ import annotations

from rag.embedding_backend import create_embedding_model
from rag.generation import similarity_from_distance
from rag.indexer import build_both_collections
from rag.retriever import retrieve_chunks

IN_SCOPE_QUERIES = [
    "How long do I have to wait before my billing dispute refund comes back?",
    "If I complain about the same issue three times does it get escalated faster?",
    "Can I call customer support at midnight for a non-emergency issue?",
    "What happens to my ticket data after it is closed?",
]

OUT_OF_SCOPE_QUERIES = [
    "What's the best pizza topping combination?",
    "Can you recommend a good movie to watch this weekend?",
]


def measure_top1_similarity(collection, model, queries: list[str]) -> list[float]:
    sims = []
    for q in queries:
        retrieved = retrieve_chunks(collection, model, q, top_k=1)
        if not retrieved:
            sims.append(0.0)
            continue
        sim = similarity_from_distance(retrieved[0]["distance"])
        sims.append(sim)
        print(f"  [{sim:.4f}] {q!r} -> top-1 parent={retrieved[0]['metadata'].get('parent_id')}")
    return sims


def main() -> None:
    model, backend = create_embedding_model()
    print(f"Embedding backend in use: {backend}\n")
    fixed_collection, _ = build_both_collections(model=model)

    print("\nIN-SCOPE queries (expect higher similarity):")
    in_scope_sims = measure_top1_similarity(fixed_collection, model, IN_SCOPE_QUERIES)

    print("\nOUT-OF-SCOPE queries (expect lower similarity):")
    out_scope_sims = measure_top1_similarity(fixed_collection, model, OUT_OF_SCOPE_QUERIES)

    min_in_scope = min(in_scope_sims)
    max_out_scope = max(out_scope_sims)

    print(f"\nMin in-scope top-1 similarity:     {min_in_scope:.4f}")
    print(f"Max out-of-scope top-1 similarity: {max_out_scope:.4f}")

    if min_in_scope > max_out_scope:
        threshold = (min_in_scope + max_out_scope) / 2
        print(f"\nClusters are separated. Recommended threshold (midpoint): {threshold:.4f}")
    else:
        # Clusters overlap -- report honestly rather than hiding it.
        threshold = (min_in_scope + max_out_scope) / 2
        print(
            f"\nWARNING: clusters overlap under the current embedding backend "
            f"(min in-scope {min_in_scope:.4f} <= max out-of-scope {max_out_scope:.4f}). "
            f"Midpoint threshold {threshold:.4f} is a best-effort fallback -- "
            f"re-calibrate once the real BGE model is available, since true "
            f"semantic embeddings should separate these clusters much better "
            f"than the lexical-overlap fallback embedder."
        )


if __name__ == "__main__":
    main()
