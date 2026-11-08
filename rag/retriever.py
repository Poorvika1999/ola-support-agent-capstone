"""
rag/retriever.py — top-k retrieval against either Chroma collection.

Same pattern as the course's retrieve_policy_chunks(): embed the query with
the same model used for indexing, run collection.query(), return ranked
chunks with metadata and distance for inspection.
"""

from __future__ import annotations


def retrieve_chunks(collection, model, query: str, top_k: int = 3) -> list[dict]:
    """Embed `query` and return the top_k nearest chunks from `collection`."""
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []

    for doc, meta, dist in zip(docs, metas, dists):
        retrieved.append({"text": doc, "metadata": meta, "distance": dist})

    return retrieved


def print_retrieved_chunks(query: str, retrieved: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"Query: {query}")
    print("=" * 72)
    for rank, chunk in enumerate(retrieved, start=1):
        print(f"\nRank {rank}")
        print(f"  Parent doc : {chunk['metadata'].get('parent_id')}")
        print(f"  Topic      : {chunk['metadata'].get('topic')}")
        print(f"  Distance   : {chunk['distance']:.4f}")
        print(f"  Text       : {chunk['text']}")
