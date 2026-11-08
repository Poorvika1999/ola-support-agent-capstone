"""
rag/evaluation.py — Part 1 Task 5: evaluate and compare both chunking
strategies with document-level precision/recall.

For each of 5 test queries, we know (by design -- these queries were each
written to target one specific KB policy document) which single parent
document is "relevant". We retrieve top_k chunks, map each chunk back to its
parent_id, dedup, then score precision/recall against that one relevant doc,
separately for each collection (fixed-overlap vs sentence-based).
"""

from __future__ import annotations

from rag.embedding_backend import create_embedding_model
from rag.indexer import build_both_collections
from rag.retriever import retrieve_chunks

# Each query here is one of the >=5 in-scope queries required by Task 4/5.
# relevant_doc is the single KB id this query was written to target.
EVAL_QUERIES = [
    {
        "query": "How long do I have to wait before my billing dispute refund comes back?",
        "relevant_doc": "KB04",
    },
    {
        "query": "If I complain about the same billing issue three times does it get escalated faster?",
        "relevant_doc": "KB07",
    },
    {
        "query": "Can I call customer support at midnight for a non-emergency account access issue?",
        "relevant_doc": "KB06",
    },
    {
        "query": "What happens to my ticket data after it is closed?",
        "relevant_doc": "KB12",
    },
    {
        "query": "Do VIP customers get faster support for their tickets?",
        "relevant_doc": "KB10",
    },
]

TOP_K = 3


def evaluate_collection(collection, model, eval_queries: list[dict], top_k: int = TOP_K) -> dict:
    """Return per-query precision/recall plus averages for one collection."""
    per_query_results = []

    for item in eval_queries:
        query = item["query"]
        relevant_doc = item["relevant_doc"]

        retrieved = retrieve_chunks(collection, model, query, top_k=top_k)
        retrieved_parent_ids = []
        for chunk in retrieved:
            parent = chunk["metadata"].get("parent_id")
            if parent not in retrieved_parent_ids:
                retrieved_parent_ids.append(parent)  # dedup, preserve rank order

        relevant_retrieved = 1 if relevant_doc in retrieved_parent_ids else 0
        num_retrieved_unique = len(retrieved_parent_ids)
        num_relevant_total = 1  # single relevant doc per query by design

        precision = relevant_retrieved / num_retrieved_unique if num_retrieved_unique else 0.0
        recall = relevant_retrieved / num_relevant_total

        per_query_results.append(
            {
                "query": query,
                "relevant_doc": relevant_doc,
                "retrieved_parent_ids": retrieved_parent_ids,
                "relevant_retrieved": relevant_retrieved,
                "num_retrieved_unique": num_retrieved_unique,
                "precision": precision,
                "recall": recall,
            }
        )

    avg_precision = sum(r["precision"] for r in per_query_results) / len(per_query_results)
    avg_recall = sum(r["recall"] for r in per_query_results) / len(per_query_results)

    return {
        "per_query": per_query_results,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
    }


def print_evaluation(name: str, results: dict) -> None:
    print(f"\n{'=' * 72}\nCollection: {name}\n{'=' * 72}")
    for r in results["per_query"]:
        print(f"\nQuery: {r['query']}")
        print(f"  Relevant doc (ground truth): {r['relevant_doc']}")
        print(f"  Retrieved parent docs (deduped, ranked): {r['retrieved_parent_ids']}")
        print(
            f"  Precision = relevant_retrieved / retrieved_unique "
            f"= {r['relevant_retrieved']} / {r['num_retrieved_unique']} = {r['precision']:.3f}"
        )
        print(f"  Recall    = relevant_retrieved / relevant_total = {r['relevant_retrieved']} / 1 = {r['recall']:.3f}")
    print(f"\nAverage precision across {len(results['per_query'])} queries: {results['avg_precision']:.3f}")
    print(f"Average recall across {len(results['per_query'])} queries:    {results['avg_recall']:.3f}")


def main() -> None:
    model, backend = create_embedding_model()
    print(f"Embedding backend in use: {backend}")
    fixed_collection, sentence_collection = build_both_collections(model=model)

    fixed_results = evaluate_collection(fixed_collection, model, EVAL_QUERIES)
    print_evaluation("ola_policy_fixed_overlap", fixed_results)

    sentence_results = evaluate_collection(sentence_collection, model, EVAL_QUERIES)
    print_evaluation("ola_policy_sentence", sentence_results)

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    print(
        f"fixed-overlap : avg precision={fixed_results['avg_precision']:.3f}, "
        f"avg recall={fixed_results['avg_recall']:.3f}"
    )
    print(
        f"sentence      : avg precision={sentence_results['avg_precision']:.3f}, "
        f"avg recall={sentence_results['avg_recall']:.3f}"
    )


if __name__ == "__main__":
    main()
