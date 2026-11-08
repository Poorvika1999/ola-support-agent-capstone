"""
rag/indexer.py — Part 1 Task 3: embed and index both chunking strategies into
two separate ChromaDB collections.

Follows the course's conventions: BAAI/bge-small-en-v1.5 via SentenceTransformer
for embeddings (manual .encode(), embedding_function=None on the collection),
chromadb.PersistentClient for on-disk storage, collection.upsert(...).

Two collections:
  - ola_policy_fixed_overlap   (fixed-size-with-overlap chunks)
  - ola_policy_sentence        (sentence-based chunks)
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

from rag.chunking import chunk_by_sentence, chunk_fixed_overlap
from rag.embedding_backend import create_embedding_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_PATH = PROJECT_ROOT / "knowledge_base" / "policies.json"
CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma_store")

COLLECTION_FIXED = "ola_policy_fixed_overlap"
COLLECTION_SENTENCE = "ola_policy_sentence"

FIXED_CHUNK_SIZE = 100
FIXED_CHUNK_OVERLAP = 20
SENTENCE_MAX_WORDS = 60


def load_knowledge_base(path: Path = KB_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _build_chunk_records(docs: list[dict], chunk_fn, **chunk_kwargs) -> list[dict]:
    """Turn KB docs into flat chunk records ready for Chroma upsert."""
    records: list[dict] = []
    for doc in docs:
        chunks = chunk_fn(doc["text"], **chunk_kwargs)
        for chunk_index, chunk_text in enumerate(chunks):
            records.append(
                {
                    "id": f"{doc['id']}_{chunk_index}",
                    "text": chunk_text,
                    "metadata": {
                        "parent_id": doc["id"],
                        "topic": doc["topic"],
                        "title": doc["title"],
                        "chunk_index": chunk_index,
                    },
                }
            )
    return records


def _index_records(collection, model, records: list[dict]) -> None:
    if not records:
        print("No records to index.")
        return
    ids = [r["id"] for r in records]
    documents = [r["text"] for r in records]
    metadatas = [r["metadata"] for r in records]
    embeddings = model.encode(documents, convert_to_numpy=True).tolist()
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def build_both_collections(
    model=None,
    client: chromadb.ClientAPI | None = None,
) -> tuple:
    """Build (or refresh) both Chroma collections from the KB. Returns (fixed_collection, sentence_collection)."""
    if model is None:
        model, backend_name = create_embedding_model()
        print(f"Embedding backend in use: {backend_name}")
    client = client or get_chroma_client()
    docs = load_knowledge_base()

    # hnsw:space="cosine" makes Chroma's returned `distance` literally
    # (1 - cosine_similarity), which is what the threshold calibration in
    # rag/evaluation.py assumes. Without this, Chroma defaults to squared L2.
    fixed_collection = client.get_or_create_collection(
        name=COLLECTION_FIXED, embedding_function=None, metadata={"hnsw:space": "cosine"}
    )
    sentence_collection = client.get_or_create_collection(
        name=COLLECTION_SENTENCE, embedding_function=None, metadata={"hnsw:space": "cosine"}
    )

    fixed_records = _build_chunk_records(
        docs, chunk_fixed_overlap, chunk_size=FIXED_CHUNK_SIZE, overlap=FIXED_CHUNK_OVERLAP
    )
    sentence_records = _build_chunk_records(docs, chunk_by_sentence, max_words_per_chunk=SENTENCE_MAX_WORDS)

    _index_records(fixed_collection, model, fixed_records)
    _index_records(sentence_collection, model, sentence_records)

    print(f"Indexed {fixed_collection.count()} chunks into {COLLECTION_FIXED} "
          f"(from {len(docs)} docs, fixed-overlap strategy).")
    print(f"Indexed {sentence_collection.count()} chunks into {COLLECTION_SENTENCE} "
          f"(from {len(docs)} docs, sentence strategy).")

    return fixed_collection, sentence_collection


if __name__ == "__main__":
    build_both_collections()
