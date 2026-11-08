"""
agents/schemas.py — Part 2 (Pydantic structured-output requirement): the
schema the Composer Agent's final crew output is validated against.

Design note on HOW this gets validated with zero LLM-side JSON generation:
CrewAI threads `task.response_model` straight into `BaseLLM.call()`'s
`response_model` parameter for any ZERO-TOOL agent (confirmed by reading
crewai/agents/agent_executor.py: `effective_response_model = None if
self.original_tools else self.response_model`). Our Composer Agent has no
tools, so when `composer_task.response_model = SupportResponse` is set
(see agents/crew.py), MockLLM.call() receives `response_model=SupportResponse`
for that call and can return a validated `SupportResponse` instance
*directly* -- CrewAI accepts a returned BaseModel as the final answer as-is
(`if isinstance(answer, BaseModel): ...` in agent_executor.py), with no
ReAct text parsing and no second "reformat as JSON" LLM call involved. This
is the cleanest way to get real Pydantic-validated structured output out of
a fully offline, deterministic MockLLM.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SupportResponse(BaseModel):
    """The validated shape of every final crew response."""

    policy_answer: str = Field(
        ..., description="The Retrieval Agent's grounded policy answer, or its refusal text if ungrounded."
    )
    ticket_status: str = Field(
        ..., description="Raw status line reported by the Lookup Agent's ticket-lookup tool."
    )
    escalation_recommended: bool = Field(
        ..., description="Whether the ticket's escalation_score crossed ESCALATION_THRESHOLD."
    )
    escalation_score: float | None = Field(
        default=None,
        description="The computed escalation score for the looked-up ticket, if one was found.",
    )
    final_reply: str = Field(
        ..., description="The single combined, customer-facing reply."
    )
