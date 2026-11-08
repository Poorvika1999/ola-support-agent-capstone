"""
rag/embedding_backend.py — embedding model loader with a local offline fallback.

Primary path: BAAI/bge-small-en-v1.5 via SentenceTransformer (matches the
course convention exactly). This requires a one-time download from
HuggingFace Hub on first use, then runs fully offline/local afterward.

Fallback path: if that download is unavailable (no network, blocked host,
etc.), fall back to a deterministic local hashing-based bag-of-words embedder
so the rest of the pipeline (indexing, retrieval, generation) can still be
exercised end-to-end without network access. This is a dev/CI resilience
feature, not a replacement for the real model — the fallback captures lexical
overlap, not true semantic similarity, so retrieval-quality numbers captured
under the fallback should not be treated as final. Run with the real model
before capturing submission transcripts.

Both paths expose the same .encode(texts, convert_to_numpy=True) -> ndarray
interface, so nothing else in the pipeline needs to know which one is active.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
FALLBACK_DIM = 384  # matches BGE-small's output dimension

_WORD_RE = re.compile(r"[a-z0-9]+")


class LocalFallbackEmbedder:
    """Deterministic, offline, hashing-based bag-of-words embedder.

    Not a semantic embedding model -- it captures lexical/keyword overlap
    only. Used solely so the pipeline can be exercised without network
    access. Interface-compatible with SentenceTransformer.encode().
    """

    name = "local-fallback-hashing-bow"

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(FALLBACK_DIM, dtype=np.float32)
        words = _WORD_RE.findall(text.lower())
        if not words:
            return vec
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % FALLBACK_DIM
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, texts, convert_to_numpy: bool = True, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        matrix = np.stack([self._embed_one(t) for t in texts])
        return matrix if convert_to_numpy else matrix.tolist()


def create_embedding_model():
    """Try to load the real BGE model; fall back to the local embedder on failure.

    Returns a tuple (model, backend_name) so callers can log/report which
    backend is actually active.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return model, EMBEDDING_MODEL_NAME
    except Exception as exc:  # network error, blocked host, missing cache, etc.
        print(
            f"[embedding_backend] Could not load '{EMBEDDING_MODEL_NAME}' "
            f"({type(exc).__name__}: {exc}). Falling back to local offline "
            f"embedder for this run. Retrieval quality under the fallback is "
            f"lexical-overlap only -- re-run with network access before "
            f"capturing final submission transcripts."
        )
        return LocalFallbackEmbedder(), LocalFallbackEmbedder.name
