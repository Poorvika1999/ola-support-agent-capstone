"""
agents/memory.py — Part 2 (LangChain memory requirement): multi-turn
conversational memory using LangChain's own history primitives, kept
deterministic and offline like the rest of this project.

We use `InMemoryChatMessageHistory` (per-session message store) and
`RunnableWithMessageHistory` (the wrapper that injects/updates that history
around a Runnable) exactly as LangChain intends. The Runnable itself is a
small deterministic function -- NOT a call into MockLLM/CrewAI -- because
this module's job is specifically to demonstrate LangChain's own memory
mechanics (state correctly persisting within a session_id, and correctly
absent for a new one), independent of the CrewAI crew built in
agents/crew.py. The response is a rule-based acknowledgement that proves it
can see prior turns (by quoting how many came before, and the previous
user message) without ever inventing information.
"""

from __future__ import annotations

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

# One InMemoryChatMessageHistory per session_id, created lazily. This dict
# IS the "memory store" -- RunnableWithMessageHistory only orchestrates
# reads/writes against whatever get_session_history() hands it back.
_SESSION_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def _respond(payload: dict) -> str:
    """Deterministic responder: acknowledges the current input and proves
    it can see prior turns in `history` by counting and quoting them.
    Never fabricates ticket/policy content -- this module only demonstrates
    memory plumbing, not the support-agent logic itself."""
    user_input = payload["input"]
    history = payload.get("history", [])
    turn_number = (len(history) // 2) + 1  # each prior turn = 1 human + 1 ai message

    if not history:
        return (
            f"(turn {turn_number}, no prior turns in this session) "
            f"Noted: '{user_input}'."
        )

    previous_human_messages = [m.content for m in history if m.type == "human"]
    last_user_message = previous_human_messages[-1] if previous_human_messages else None
    return (
        f"(turn {turn_number}, {len(previous_human_messages)} prior turn(s) remembered) "
        f"Noted: '{user_input}'. Your previous message in this session was: '{last_user_message}'."
    )


_runnable = RunnableLambda(_respond)

chat_with_memory = RunnableWithMessageHistory(
    _runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def run_demo() -> None:
    print("=" * 72)
    print("Session A -- multi-turn conversation (memory should accumulate)")
    print("=" * 72)
    config_a = {"configurable": {"session_id": "session-A"}}
    turns_a = [
        "Hi, I have a question about ticket TCK-0030.",
        "Is it escalated?",
        "What's Ola's refund policy in general?",
    ]
    for turn in turns_a:
        reply = chat_with_memory.invoke({"input": turn}, config=config_a)
        print(f"\nUser: {turn}\nAssistant: {reply}")

    print("\n" + "=" * 72)
    print("Session B -- a brand-new session_id (memory should be EMPTY, not")
    print("leaking anything from Session A)")
    print("=" * 72)
    config_b = {"configurable": {"session_id": "session-B"}}
    reply_b = chat_with_memory.invoke({"input": "Hello, this is a new conversation."}, config=config_b)
    print(f"\nUser: Hello, this is a new conversation.\nAssistant: {reply_b}")

    print("\n" + "=" * 72)
    print("Verification")
    print("=" * 72)
    history_a = get_session_history("session-A")
    history_b = get_session_history("session-B")
    print(f"session-A stored messages: {len(history_a.messages)} (expect {2 * len(turns_a)})")
    print(f"session-B stored messages: {len(history_b.messages)} (expect 2)")
    assert len(history_a.messages) == 2 * len(turns_a)
    assert len(history_b.messages) == 2
    print("PASSED: session-B's history is independent of session-A's.")


if __name__ == "__main__":
    run_demo()
