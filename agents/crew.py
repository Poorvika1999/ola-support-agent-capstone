"""
agents/crew.py — Part 2 Task 7: assemble the 3-agent CrewAI crew.

  1. Retrieval Agent   (role="Policy Retrieval Specialist") -- RAG tool
  2. Lookup Agent       (role="Ticket Status Specialist")     -- ticket-status tool
  3. Composer Agent     (role="Response Composer")            -- zero tools,
     synthesizes the final customer-facing reply from both upstream outputs.

Process.sequential: each task's output automatically becomes available
context for the tasks that follow it; the Composer task additionally
declares context=[...] explicitly for clarity/documentation even though
sequential process would supply it anyway (verified empirically --
see scripts/diagnose_context.py).

Every agent's llm=MockLLM(): no network call, no API key, fully
deterministic given the same ticket data and KB.
"""

from __future__ import annotations

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from crewai import Agent, Crew, Process, Task

from agents.crew_tools import check_ticket_status, retrieve_policy_context
from agents.mock_llm import COMPOSER_AGENT_ROLE, LOOKUP_AGENT_ROLE, RETRIEVAL_AGENT_ROLE, MockLLM
from agents.schemas import SupportResponse


def build_crew(user_query: str, record_id: str) -> Crew:
    """Build a fresh 3-agent / 3-task crew for one support request.

    `user_query` drives the Retrieval Agent (a policy question in natural
    language). `record_id` drives the Lookup Agent (a ticket id like
    'TCK-0030'). The Composer Agent combines both into one reply.
    """
    llm = MockLLM()

    retrieval_agent = Agent(
        role=RETRIEVAL_AGENT_ROLE,
        goal="Answer Ola support-policy questions using only the retrieved policy excerpts.",
        backstory=(
            "You are a policy specialist for Ola customer support. You never answer from "
            "memory -- you always retrieve the relevant policy excerpt first and base your "
            "answer only on what was retrieved."
        ),
        tools=[retrieve_policy_context],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    lookup_agent = Agent(
        role=LOOKUP_AGENT_ROLE,
        goal="Look up a specific Ola support ticket's status and escalation score.",
        backstory=(
            "You are a ticket-operations specialist. You only report what the ticket "
            "lookup tool returns -- you never guess a ticket's status."
        ),
        tools=[check_ticket_status],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    composer_agent = Agent(
        role=COMPOSER_AGENT_ROLE,
        goal="Compose one clear, customer-facing reply from the retrieval and lookup results.",
        backstory=(
            "You are the final voice to the customer. You never invent facts -- you only "
            "restate what the retrieval and lookup specialists already found."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # The literal marker "Customer Question:" matters: MockLLM extracts the
    # text after it verbatim as the retrieval tool's query, so none of this
    # instruction's own wrapper words ("Answer this Ola policy question
    # using...") get sent to the tool and dilute the similarity score. See
    # agents/mock_llm.py's _extract_customer_question.
    retrieval_task = Task(
        description=(
            "Answer this Ola policy question using the retrieval tool.\n\n"
            f"Customer Question: {user_query}"
        ),
        expected_output="A grounded answer to the policy question, or a refusal if ungrounded.",
        agent=retrieval_agent,
    )

    lookup_task = Task(
        description=(
            f"Look up the status and escalation score for Ola support ticket {record_id} "
            "using the ticket lookup tool."
        ),
        expected_output="The ticket's status, resolution time, and escalation recommendation.",
        agent=lookup_agent,
    )

    composer_task = Task(
        description=(
            "Compose one combined, customer-facing reply using the policy answer and the "
            "ticket status gathered above."
        ),
        expected_output="A single clear reply combining both pieces of information.",
        agent=composer_agent,
        context=[retrieval_task, lookup_task],
        # Threaded into MockLLM.call()'s `response_model` param for this
        # zero-tool agent (see agents/schemas.py's docstring for exactly how
        # CrewAI wires this through); output_pydantic is also set so the
        # TaskOutput's `.pydantic` attribute is populated the standard way.
        response_model=SupportResponse,
        output_pydantic=SupportResponse,
    )

    return Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[retrieval_task, lookup_task, composer_task],
        process=Process.sequential,
        verbose=False,
    )


if __name__ == "__main__":
    # Demonstration scenarios, matching the brief's "demonstrate both tools
    # invoked on different sample queries" requirement plus edge cases:
    #   1. Happy path: an in-scope policy query (known to retrieve correctly
    #      even under this sandbox's lexical-overlap fallback embedder --
    #      see rag/evaluation.py) + a valid, escalation-worthy ticket.
    #   2. Out-of-scope policy query -> the RAG refusal guardrail should fire.
    #   3. Unknown ticket id -> the lookup tool's honest "not found" path.
    scenarios = [
        ("Happy path (grounded policy answer + escalation-worthy ticket)",
         "Do VIP customers get faster support for their tickets?", "TCK-0030"),
        ("Out-of-scope policy query (refusal guardrail should fire)",
         "Can you recommend a good movie to watch this weekend?", "TCK-0001"),
        ("Unknown ticket id (honest not-found, never invented)",
         "What happens to my ticket data after it is closed?", "TCK-9999"),
    ]

    for title, query, record_id in scenarios:
        print("\n" + "#" * 72)
        print(f"# SCENARIO: {title}")
        print(f"# query={query!r} record_id={record_id!r}")
        print("#" * 72)
        crew = build_crew(user_query=query, record_id=record_id)
        result = crew.kickoff()
        print("\n--- FINAL CREW OUTPUT (raw) ---")
        print(result.raw)
        print("\n--- VALIDATED SupportResponse (Pydantic) ---")
        if result.pydantic is not None:
            print(result.pydantic.model_dump_json(indent=2))
        else:
            print("(no structured output attached to this TaskOutput)")
