"""
review/autogen_review.py — Part 4 Task 14: AutoGen review stage sitting
AFTER CrewAI's output (agents/crew.py). Two frameworks, two separate
pipeline stages: CrewAI orchestrates the main 3-agent crew; AutoGen's
`RoundRobinGroupChat` reviews that crew's *finished* response for
compliance/safety concerns before it would be sent to the customer.

WHY ReplayChatCompletionClient: same MOCK_LLM philosophy as the rest of
this project -- zero API keys, zero network calls, fully deterministic.
`autogen_ext.models.replay.ReplayChatCompletionClient` is AutoGen's own
official deterministic-playback client, built for exactly this. The two
canned responses it replays are NOT hardcoded text -- they are computed by
`assess_support_response()` below, a plain rule-based function that
inspects the actual crew output (reusing agents/guardrails.py's
groundedness check). So the *judgment* is deterministic Python, and
ReplayChatCompletionClient just lets that judgment flow through AutoGen's
real multi-agent plumbing (RoundRobinGroupChat, message passing, structured
output) rather than bypassing AutoGen entirely.

TRAP 5 AVOIDANCE (confirmed empirically, not assumed -- see
scripts/diagnose_autogen.py): an `AssistantAgent` with
`output_content_type=VerdictModel` emits a `StructuredMessage[VerdictModel]`
into the group chat. If the `RoundRobinGroupChat` doesn't know about that
message type up front, it raises
`ValueError: Message type <class '...StructuredMessage[VerdictModel]'> is
not registered.` the moment that message is logged -- confirmed by
deliberately omitting `custom_message_types` once and watching it fail.
`custom_message_types=[StructuredMessage[VerdictModel]]` on the team avoids
this.
"""

from __future__ import annotations

import asyncio
import json

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ModelInfo
from autogen_ext.models.replay import ReplayChatCompletionClient

from agents.guardrails import check_groundedness
from review.verdict import VerdictModel


def assess_support_response(response: dict) -> tuple[str, VerdictModel]:
    """The actual review logic: plain rules over the crew's SupportResponse
    dict. Returns (reviewer_prose, verdict) -- both get fed into
    ReplayChatCompletionClient so AutoGen's real plumbing carries them."""
    policy_answer = response.get("policy_answer", "")
    ticket_status = response.get("ticket_status", "")
    escalation_recommended = response.get("escalation_recommended", False)

    concerns: list[str] = []
    grounded = check_groundedness(policy_answer)
    if not grounded:
        concerns.append("Policy answer is not traceable to any cited knowledge-base document.")

    unknown_ticket = "UNKNOWN_TICKET" in ticket_status
    if unknown_ticket:
        concerns.append("Ticket id was not found in the register; verify with the customer before proceeding.")

    if not grounded:
        risk_level, approved = "high", False
    elif escalation_recommended or unknown_ticket:
        risk_level, approved = "medium", True
    else:
        risk_level, approved = "low", True

    if concerns:
        prose = "Concerns found during review: " + "; ".join(concerns)
    else:
        prose = "No concerns found. The policy answer is grounded and the ticket status is consistent with records."

    verdict = VerdictModel(approved=approved, risk_level=risk_level, concerns=concerns, rationale=prose)
    return prose, verdict


async def run_review_async(response: dict) -> VerdictModel:
    reviewer_prose, verdict = assess_support_response(response)

    reviewer_agent = AssistantAgent(
        name="policy_reviewer",
        model_client=ReplayChatCompletionClient([reviewer_prose]),
        system_message="You review Ola support responses for compliance and safety concerns.",
    )

    verdict_client = ReplayChatCompletionClient(
        [verdict.model_dump_json()],
        model_info=ModelInfo(
            vision=False, function_calling=False, json_output=True,
            structured_output=True, family="unknown",
        ),
    )
    verdict_agent = AssistantAgent(
        name="verdict_agent",
        model_client=verdict_client,
        system_message="You output the final structured verdict as JSON matching VerdictModel.",
        output_content_type=VerdictModel,
    )

    team = RoundRobinGroupChat(
        participants=[reviewer_agent, verdict_agent],
        max_turns=2,
        custom_message_types=[StructuredMessage[VerdictModel]],
    )

    result = await team.run(task="Review this Ola support response for compliance and safety.")

    for message in reversed(result.messages):
        if isinstance(message, StructuredMessage) and isinstance(message.content, VerdictModel):
            return message.content

    raise RuntimeError("AutoGen review team did not produce a VerdictModel message.")


def run_review(response: dict) -> VerdictModel:
    """Sync wrapper for callers (e.g. api/main.py) that aren't already async."""
    return asyncio.run(run_review_async(response))


if __name__ == "__main__":
    print("=" * 72)
    print("Scenario A: a grounded, well-formed response -> expect approved/low-risk")
    print("=" * 72)
    good_response = {
        "policy_answer": "According to Ola policy [KB10: VIP Customer Handling Policy]: VIP tickets are routed to senior staff.",
        "ticket_status": "Record ID: TCK-0001\nStatus: Open\nEscalation Score: 0.00 (Threshold: 0.39 -> Normal Priority)",
        "escalation_recommended": False,
    }
    verdict_a = run_review(good_response)
    print(json.dumps(verdict_a.model_dump(), indent=2))

    print("\n" + "=" * 72)
    print("Scenario B: an ungrounded (fabricated, uncited) policy answer -> expect NOT approved / high risk")
    print("=" * 72)
    bad_response = {
        "policy_answer": "Ola always issues refunds instantly, no questions asked.",  # no [KBxx:] citation, not the refusal
        "ticket_status": "Record ID: TCK-0030\nStatus: Open\nEscalation Score: 0.77 (Threshold: 0.39 -> ESCALATION RECOMMENDED)",
        "escalation_recommended": True,
    }
    verdict_b = run_review(bad_response)
    print(json.dumps(verdict_b.model_dump(), indent=2))

    assert verdict_a.approved is True and verdict_a.risk_level == "low"
    assert verdict_b.approved is False and verdict_b.risk_level == "high"
    print("\nBoth review scenarios behaved as expected.")
