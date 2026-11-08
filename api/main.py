"""
api/main.py — Part 3: FastAPI + WebSocket deployment of the Ola support crew.

Endpoints:
  POST /support/query          -- run the full 3-agent crew on a policy
                                   question + ticket id, return the
                                   validated SupportResponse.
  GET  /support/ticket/{id}    -- direct ticket lookup, bypassing the crew
                                   entirely (cheap, no RAG/LLM overhead).
  WS   /ws/support             -- same as the POST endpoint, but streamed:
                                   client sends {"query": ..., "record_id": ...}
                                   JSON messages, server replies with one
                                   SupportResponse JSON per message, handling
                                   client disconnects gracefully.

Every request (HTTP and WebSocket) is guarded the same way before it ever
reaches the crew: prompt-injection detection (agents/guardrails.py) can
block a request outright, and PII masking is applied before anything is
written to the structured JSON-Lines log (api/logging_utils.py) -- never to
the response itself, since customers legitimately need to see their own
phone number if they typed it.
"""

from __future__ import annotations

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from agents.crew import build_crew
from agents.guardrails import detect_prompt_injection, mask_pii
from agents.tools import check_support_ticket_status
from api.logging_utils import log_event

app = FastAPI(title="Ola Support Agent API", version="1.0.0")


class SupportQueryRequest(BaseModel):
    query: str
    record_id: str


def _run_crew_and_extract(query: str, record_id: str) -> dict:
    """Sync path, used by the HTTP endpoint -- FastAPI runs a `def` (not
    `async def`) route handler in a worker thread, off the event loop, so
    the blocking `crew.kickoff()` is safe here."""
    crew = build_crew(user_query=query, record_id=record_id)
    result = crew.kickoff()
    if result.pydantic is not None:
        return result.pydantic.model_dump()
    return {"raw": result.raw}


async def _run_crew_and_extract_async(query: str, record_id: str) -> dict:
    """Async path, used by the WebSocket handler. The WS handler is
    `async def` and runs directly on the event loop, so the blocking
    `crew.kickoff()` is NOT safe there -- CrewAI itself detects this and
    raises (confirmed empirically: 'Agent execution was invoked
    synchronously from within a running event loop'). `crew.kickoff_async()`
    is CrewAI's own answer to exactly this situation."""
    crew = build_crew(user_query=query, record_id=record_id)
    result = await crew.kickoff_async()
    if result.pydantic is not None:
        return result.pydantic.model_dump()
    return {"raw": result.raw}


def _check_injection_and_log(query: str, record_id: str, channel: str) -> tuple[bool, str | None]:
    masked_query, _ = mask_pii(query)
    is_suspicious, pattern = detect_prompt_injection(query)
    log_event(
        f"{channel}_query_received",
        {"query": masked_query, "record_id": record_id, "prompt_injection_suspected": is_suspicious},
    )
    if is_suspicious:
        log_event(f"{channel}_query_blocked", {"record_id": record_id, "matched_pattern": pattern})
    return is_suspicious, pattern


@app.post("/support/query")
def support_query(request: SupportQueryRequest) -> dict:
    """Run the full crew (RAG retrieval + ticket lookup + composed reply)."""
    is_suspicious, pattern = _check_injection_and_log(request.query, request.record_id, "http")
    if is_suspicious:
        return {"error": "Request blocked: potential prompt injection detected.", "matched_pattern": pattern}

    response = _run_crew_and_extract(request.query, request.record_id)
    log_event(
        "http_query_answered",
        {"record_id": request.record_id, "escalation_recommended": response.get("escalation_recommended")},
    )
    return response


@app.get("/support/ticket/{record_id}")
def get_ticket_status(record_id: str) -> dict:
    """Direct ticket lookup -- no RAG, no crew, just the Part 2 Task 6 tool."""
    log_event("http_ticket_lookup", {"record_id": record_id})
    return check_support_ticket_status(record_id)


@app.websocket("/ws/support")
async def ws_support(websocket: WebSocket) -> None:
    await websocket.accept()
    log_event("ws_connected", {})
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            record_id = data.get("record_id", "")

            is_suspicious, pattern = _check_injection_and_log(query, record_id, "ws")
            if is_suspicious:
                await websocket.send_json(
                    {"error": "Request blocked: potential prompt injection detected.", "matched_pattern": pattern}
                )
                continue

            response = await _run_crew_and_extract_async(query, record_id)
            log_event(
                "ws_query_answered",
                {"record_id": record_id, "escalation_recommended": response.get("escalation_recommended")},
            )
            await websocket.send_json(response)
    except WebSocketDisconnect:
        # Expected when the client closes the connection -- not an error.
        log_event("ws_disconnected", {})
