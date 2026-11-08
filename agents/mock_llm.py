"""
agents/mock_llm.py — the deterministic, zero-network, zero-API-key LLM that
drives the entire CrewAI crew (Part 2 "MOCK_LLM" requirement).

Instead of calling any real model, `MockLLM.call()` inspects the growing
ReAct-style prompt/scratchpad CrewAI builds for each agent step and decides,
with plain if/else logic and regex, whether the agent should invoke its one
tool or give its Final Answer. No tool call result is ever invented: every
piece of ticket/policy data in a final answer is something that actually
came back from agents/crew_tools.py in an earlier step.

HOW THIS WAS BUILT (verified empirically in this sandbox against the
installed CrewAI version, not assumed from documentation -- see
scripts/diagnose_llm_call.py and scripts/diagnose_context.py):

  * `BaseLLM.call()`'s `tools` and `available_functions` parameters come
    through as None for this CrewAI version. The only reliable signal about
    which agent is calling is `from_agent.role`. So tool dispatch is done by
    role, not by tool name or schema inspection.

  * CrewAI's own prompt template embeds the literal text
    "Observation: the result of the action" inside the *static* ReAct
    instructions (shown to the agent as the example format), BEFORE the
    literal marker "Begin!" ever appears. The *real* "Observation: <result>"
    line (appended after a tool actually runs) only ever appears AFTER
    "Begin!". If the parser naively searched the whole prompt text for
    "Observation:", it would match the placeholder on the agent's very
    first turn -- before any tool has run -- and treat
    "the result of the action" as if it were a genuine tool result.
    TRAP 1 AVOIDANCE: we only ever look for a real Observation inside the
    text that comes after the LAST "Begin!" marker, and additionally refuse
    to treat anything starting with "the result of the action" as real,
    even there.

  * Tool dispatch is never done via "does this tool's *name* contain
    keyword X" (TRAP 2) -- that breaks the moment a tool name itself
    contains a misleading substring (the brief's own example: a tool named
    `rag_lookup` would wrongly match a "lookup" keyword check intended for
    ticket lookups). Instead, each agent in this project is given exactly
    one tool, and MockLLM decides what to do purely from `from_agent.role` --
    a value CrewAI passes to every call() reliably, independent of the
    prompt text or the tool's own name.
"""

from __future__ import annotations

import json
import re

from crewai.llms.base_llm import BaseLLM

from agents.schemas import SupportResponse
from governance.authorization import set_current_agent_role

# --- Agent roles this MockLLM knows how to drive (must match agents/crew.py) ---
RETRIEVAL_AGENT_ROLE = "Policy Retrieval Specialist"
LOOKUP_AGENT_ROLE = "Ticket Status Specialist"
COMPOSER_AGENT_ROLE = "Response Composer"

# --- The one tool each of the above agents is given (agents/crew_tools.py) ---
RETRIEVAL_TOOL_NAME = "retrieve_policy_context"
LOOKUP_TOOL_NAME = "check_ticket_status"

# --- Literal markers CrewAI's own prompt templates are built from ---
CURRENT_TASK_MARKER = "Current Task:"
EXPECTED_CRITERIA_MARKER = "This is the expected criteria"
CONTEXT_MARKER = "This is the context you're working with:"
BEGIN_MARKER = "Begin!"
NO_TOOL_CUE = "Provide your complete response:"
CUSTOMER_QUESTION_MARKER = "Customer Question:"
CONTEXT_ITEM_SEPARATOR = "----------"
PLACEHOLDER_OBSERVATION_PREFIX = "the result of the action"

# Used only internally to flatten the messages list back into one string
# without merging adjacent messages' text together (CrewAI appends a fixed
# "Analyze the tool result..." nudge as its own message right after the
# assistant message that contains the real Observation line; without an
# unambiguous boundary here, a plain "\n" join would let that nudge text
# bleed into the extracted Observation value).
_MESSAGE_BOUNDARY = "\n<<<MOCKLLM_MSG_BOUNDARY>>>\n"

TICKET_ID_RE = re.compile(r"TCK-\d{4}", re.IGNORECASE)


class MockLLM(BaseLLM):
    """Deterministic rule-based stand-in for a real LLM. No network call,
    no API key, fully reproducible given the same prompt history."""

    model: str = "mock-llm-ola-support-v1"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model", "mock-llm-ola-support-v1")
        super().__init__(**kwargs)

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        full_text = self._flatten_messages(messages)
        role = getattr(from_agent, "role", "") or ""
        # Record who is about to act -- governance/authorization.py's
        # require_role() checks this the moment the tool it guards actually
        # executes, moments after this call() returns an Action for it.
        set_current_agent_role(role)

        task_description = self._extract_task_description(full_text)
        context_block = self._extract_context_block(full_text)
        observation = self._extract_latest_real_observation(full_text)

        if role == RETRIEVAL_AGENT_ROLE:
            return self._handle_retrieval_agent(task_description, observation)
        if role == LOOKUP_AGENT_ROLE:
            return self._handle_lookup_agent(task_description, observation)
        if role == COMPOSER_AGENT_ROLE:
            return self._handle_composer_agent(context_block, response_model)

        # Defensive fallback: never silently guess for a role we don't know.
        return (
            "Thought: I was not configured to handle this agent role.\n"
            f"Final Answer: MockLLM has no behavior defined for role '{role}'."
        )

    # ------------------------------------------------------------------
    # Prompt parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_messages(messages) -> str:
        if isinstance(messages, str):
            return messages
        parts = []
        for message in messages:
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            parts.append(content)
        return _MESSAGE_BOUNDARY.join(parts)

    @staticmethod
    def _extract_task_description(full_text: str) -> str:
        idx = full_text.find(CURRENT_TASK_MARKER)
        if idx == -1:
            return ""
        start = idx + len(CURRENT_TASK_MARKER)
        end = full_text.find(EXPECTED_CRITERIA_MARKER, start)
        if end == -1:
            end = len(full_text)
        return full_text[start:end].strip()

    @staticmethod
    def _extract_customer_question(task_description: str) -> str:
        """Isolate the raw customer question from the task description, so
        none of our own instructional wrapper phrasing ("Answer this Ola
        policy question using the retrieval tool...") is sent to the RAG
        tool as part of the query -- it would otherwise dilute the
        similarity score, which matters most under the lexical-overlap
        fallback embedder but is good hygiene regardless of embedder."""
        idx = task_description.find(CUSTOMER_QUESTION_MARKER)
        if idx == -1:
            return task_description.strip()
        return task_description[idx + len(CUSTOMER_QUESTION_MARKER):].strip()

    @staticmethod
    def _extract_context_block(full_text: str) -> str:
        """Text between 'This is the context you're working with:' and
        whichever of the Begin!/no-tool cues appears next (or end of text)."""
        idx = full_text.find(CONTEXT_MARKER)
        if idx == -1:
            return ""
        start = idx + len(CONTEXT_MARKER)
        candidates = [
            pos
            for pos in (full_text.find(BEGIN_MARKER, start), full_text.find(NO_TOOL_CUE, start))
            if pos != -1
        ]
        end = min(candidates) if candidates else len(full_text)
        return full_text[start:end].strip()

    @staticmethod
    def _extract_latest_real_observation(full_text: str) -> str | None:
        """TRAP 1 AVOIDANCE (see module docstring): only look for a real tool
        Observation in the text AFTER the last literal 'Begin!' marker --
        never in the static instructions that precede it, where CrewAI's own
        ReAct template shows 'Observation: the result of the action' as a
        format example rather than a real result."""
        begin_idx = full_text.rfind(BEGIN_MARKER)
        if begin_idx == -1:
            # No-tool agent (never emits a ReAct loop) or template not yet
            # rendered -- either way, there is no real observation to find.
            return None

        scratchpad = full_text[begin_idx:]
        obs_idx = scratchpad.rfind("\nObservation:")
        if obs_idx == -1:
            return None

        observation = scratchpad[obs_idx + len("\nObservation:"):]
        # Stop at the next message boundary -- the real Observation line is
        # the tail of its own assistant message; CrewAI's next message (a
        # fixed "Analyze the tool result..." nudge) must not be absorbed
        # into the extracted value.
        boundary_idx = observation.find(_MESSAGE_BOUNDARY)
        if boundary_idx != -1:
            observation = observation[:boundary_idx]
        observation = observation.strip()

        if observation.startswith(PLACEHOLDER_OBSERVATION_PREFIX):
            # Should be unreachable post-"Begin!", but kept as an explicit
            # safety net against exactly the trap the brief calls out.
            return None
        return observation or None

    # ------------------------------------------------------------------
    # Per-role behavior
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_retrieval_agent(task_description: str, observation: str | None) -> str:
        if observation is not None:
            # The tool already returned a fully-formed grounded answer (or a
            # refusal) -- pass it through verbatim rather than re-wording it,
            # so nothing is added that the RAG pipeline didn't itself produce.
            return f"Thought: I have retrieved the relevant policy context.\nFinal Answer: {observation}"

        query = MockLLM._extract_customer_question(task_description or "")
        action_input = json.dumps({"query": query})
        return (
            "Thought: I need to retrieve the relevant Ola policy excerpt before I can answer.\n"
            f"Action: {RETRIEVAL_TOOL_NAME}\n"
            f"Action Input: {action_input}"
        )

    @staticmethod
    def _handle_lookup_agent(task_description: str, observation: str | None) -> str:
        if observation is not None:
            return f"Thought: I have the ticket's status and escalation score.\nFinal Answer: {observation}"

        match = TICKET_ID_RE.search(task_description or "")
        record_id = match.group(0).upper() if match else ""
        action_input = json.dumps({"record_id": record_id})
        return (
            "Thought: I need to look up this ticket's current status and escalation score.\n"
            f"Action: {LOOKUP_TOOL_NAME}\n"
            f"Action Input: {action_input}"
        )

    @staticmethod
    def _handle_composer_agent(context_block: str, response_model) -> "str | SupportResponse":
        if not context_block:
            fallback = (
                "I do not have enough information from the retrieval and lookup "
                "steps to compose a response."
            )
            if response_model is not None:
                return response_model(
                    policy_answer="", ticket_status="", escalation_recommended=False,
                    escalation_score=None, final_reply=fallback,
                )
            return (
                "Thought: No upstream retrieval or lookup results were provided to compose from.\n"
                f"Final Answer: {fallback}"
            )

        upstream_parts = [p.strip() for p in context_block.split(CONTEXT_ITEM_SEPARATOR) if p.strip()]
        policy_answer = upstream_parts[0] if len(upstream_parts) > 0 else ""
        ticket_status = upstream_parts[1] if len(upstream_parts) > 1 else ""

        composed_lines = ["Here is what I found for your request:"]
        composed_lines.extend(f"- {part}" for part in upstream_parts)
        composed = "\n".join(composed_lines)

        if response_model is None:
            return f"Thought: I now have both the policy context and the ticket status.\nFinal Answer: {composed}"

        # Structured path (Task.response_model set -- see agents/crew.py): parse
        # the deterministic "Escalation Score: X (Threshold: Y -> VERDICT)" line
        # that agents/tools.py always emits when a ticket IS found, so the
        # structured fields come from the real tool output, never invented.
        escalation_match = re.search(
            r"Escalation Score:\s*([0-9.]+).*?->\s*(ESCALATION RECOMMENDED|Normal Priority)",
            ticket_status,
            re.DOTALL,
        )
        escalation_score = float(escalation_match.group(1)) if escalation_match else None
        escalation_recommended = (
            escalation_match.group(2) == "ESCALATION RECOMMENDED" if escalation_match else False
        )

        return response_model(
            policy_answer=policy_answer,
            ticket_status=ticket_status,
            escalation_recommended=escalation_recommended,
            escalation_score=escalation_score,
            final_reply=composed,
        )
