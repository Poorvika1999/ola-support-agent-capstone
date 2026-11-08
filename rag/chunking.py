"""
rag/chunking.py — the two required chunking strategies for Part 1 Task 3:

1. chunk_fixed_overlap  — fixed-size word-count chunks with a sliding overlap.
   Convention follows the course's RAG pipeline session: 100 words per chunk,
   20-word overlap (same defaults used in rag_pipeline.py's chunk_text()).
2. chunk_by_sentence     — one chunk per sentence (or small sentence group),
   the brief's second required strategy. Not covered in the course material,
   so this is a straightforward regex sentence-splitter written for this
   project: it groups consecutive sentences up to a soft word cap so a chunk
   is never a single fragment-sized sentence nor an entire paragraph.
"""

from __future__ import annotations

import re


def chunk_fixed_overlap(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """Split text into fixed-size word chunks with a sliding overlap.

    Same algorithm as the course's rag_pipeline.py chunk_text(): slide a
    window of `chunk_size` words forward by (chunk_size - overlap) words
    each step, so consecutive chunks share `overlap` words of context.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_by_sentence(text: str, max_words_per_chunk: int = 60) -> list[str]:
    """Split text into sentence-based chunks.

    Sentences are grouped consecutively into a chunk until adding the next
    sentence would exceed `max_words_per_chunk`, then a new chunk starts.
    This keeps each chunk to roughly one coherent idea (one or two sentences)
    rather than chunking on a fixed word count that can cut a sentence in half.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current and current_word_count + sentence_word_count > max_words_per_chunk:
            chunks.append(" ".join(current))
            current = []
            current_word_count = 0
        current.append(sentence)
        current_word_count += sentence_word_count

    if current:
        chunks.append(" ".join(current))

    return chunks
