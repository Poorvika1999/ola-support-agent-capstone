"""
api/logging_utils.py — Part 3: structured JSON-Lines request/response
logging. Every string field passed through `log_event` is masked with
agents/guardrails.py's `mask_pii` BEFORE it is written to disk -- masking
happens at the logging boundary itself, not as an afterthought, so a phone
number typed into a query can never land in the log file unredacted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agents.guardrails import mask_pii

LOG_PATH = Path(__file__).resolve().parent.parent / "output" / "api_requests.jsonl"


def _mask_value(value: Any) -> Any:
    if isinstance(value, str):
        masked, _ = mask_pii(value)
        return masked
    return value


def log_event(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Append one JSON-Lines record to LOG_PATH. Returns the record actually
    written (masked), mainly so tests/demos can assert on it directly."""
    record = {
        "timestamp": round(time.time(), 3),
        "event": event_type,
        **{key: _mask_value(value) for key, value in payload.items()},
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_log_lines() -> list[dict[str, Any]]:
    """Convenience reader used by tests/demos, not by the API itself."""
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
