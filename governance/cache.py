"""
governance/cache.py — Part 4: in-memory response cache.

Keyed by the exact (query, record_id) pair. A real deployment would also
want a TTL/invalidation story (ticket status changes over time!), but the
brief asks specifically for an in-memory cache with a before/after
cache-hit demo, so that's what this is -- deliberately simple, and its
simplicity (no TTL) is noted rather than hidden.
"""

from __future__ import annotations

import time

from agents.crew import build_crew

_CACHE: dict[tuple[str, str], dict] = {}


def _cache_key(query: str, record_id: str) -> tuple[str, str]:
    return (query.strip().lower(), record_id.strip().upper())


def clear_cache() -> None:
    _CACHE.clear()


def cached_crew_run(query: str, record_id: str) -> tuple[dict, bool, float]:
    """Returns (response_dict, was_cache_hit, elapsed_seconds)."""
    key = _cache_key(query, record_id)
    start = time.perf_counter()

    if key in _CACHE:
        response = _CACHE[key]
        elapsed = time.perf_counter() - start
        return response, True, elapsed

    crew = build_crew(user_query=query, record_id=record_id)
    result = crew.kickoff()
    response = result.pydantic.model_dump() if result.pydantic is not None else {"raw": result.raw}
    _CACHE[key] = response
    elapsed = time.perf_counter() - start
    return response, False, elapsed


if __name__ == "__main__":
    import os

    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

    clear_cache()
    query = "Do VIP customers get faster support for their tickets?"
    record_id = "TCK-0030"

    print("First call (expect a cache MISS -- runs the full crew):")
    response_1, was_hit_1, elapsed_1 = cached_crew_run(query, record_id)
    print(f"  cache_hit={was_hit_1}  elapsed={elapsed_1:.4f}s")

    print("\nSecond call, same (query, record_id) (expect a cache HIT -- no crew run):")
    response_2, was_hit_2, elapsed_2 = cached_crew_run(query, record_id)
    print(f"  cache_hit={was_hit_2}  elapsed={elapsed_2:.4f}s")

    assert response_1 == response_2
    assert was_hit_1 is False and was_hit_2 is True
    assert elapsed_2 < elapsed_1
    print(f"\nSpeedup: {elapsed_1 / max(elapsed_2, 1e-9):.1f}x faster on the cache hit.")
    print("Responses are identical, and the second call never touched the crew. PASSED.")
