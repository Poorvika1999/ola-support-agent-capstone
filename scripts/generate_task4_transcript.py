"""
scripts/generate_task4_transcript.py — Part 1 Task 4 demonstration.

Runs the 5 in-scope eval queries (from rag/evaluation.py, reused here per the
brief's instruction that Task 5 reuses "the same >=5 queries from Task 4")
plus 1 deliberately out-of-scope query through the full retrieve -> generate
loop, and writes a transcript to output/task4_grounded_generation.txt showing
retrieval ranks and the final grounded (or refused) answer for each.
"""

from __future__ import annotations

from pathlib import Path

from rag.embedding_backend import create_embedding_model
from rag.evaluation import EVAL_QUERIES
from rag.generation import REFUSAL_THRESHOLD, generate_grounded_answer, similarity_from_distance
from rag.indexer import build_both_collections
from rag.retriever import retrieve_chunks

OUT_OF_SCOPE_QUERY = "Can you recommend a good movie to watch this weekend?"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "output" / "task4_grounded_generation.txt"


def run_query(collection, model, query: str, top_k: int = 3) -> str:
    retrieved = retrieve_chunks(collection, model, query, top_k=top_k)
    answer = generate_grounded_answer(query, retrieved, threshold=REFUSAL_THRESHOLD)

    lines = [f"QUERY: {query}", "-" * 72]
    for rank, chunk in enumerate(retrieved, start=1):
        sim = similarity_from_distance(chunk["distance"])
        lines.append(
            f"  Rank {rank}: parent={chunk['metadata'].get('parent_id')} "
            f"similarity={sim:.4f} text={chunk['text'][:90]}..."
        )
    lines.append(f"\nANSWER: {answer}\n")
    return "\n".join(lines)


def main() -> None:
    model, backend = create_embedding_model()
    fixed_collection, _ = build_both_collections(model=model)

    blocks = [
        "=" * 72,
        "TASK 4 — GROUNDED GENERATION TRANSCRIPT (Ola Support Agent)",
        f"Embedding backend: {backend}",
        f"REFUSAL_THRESHOLD (similarity): {REFUSAL_THRESHOLD}",
        "Collection: ola_policy_fixed_overlap",
        "=" * 72,
        "",
    ]

    for item in EVAL_QUERIES:
        blocks.append(run_query(fixed_collection, model, item["query"]))

    blocks.append("--- DELIBERATELY OUT-OF-SCOPE QUERY (must trigger refusal fallback) ---\n")
    blocks.append(run_query(fixed_collection, model, OUT_OF_SCOPE_QUERY))

    transcript = "\n".join(blocks)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(transcript, encoding="utf-8")

    print(transcript)
    print(f"\nTranscript written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
