"""
governance/authorization.py — Part 4: least-autonomy enforcement.

The requirement: only the Lookup Agent may call the ticket-status tool
(and, symmetrically here, only the Retrieval Agent may call the RAG tool).
In agents/crew.py this is already true "by construction" -- each agent is
only ever given one tool. But construction alone isn't *enforcement*: this
module adds a runtime guard that actually refuses an unauthorized call,
so the restriction holds even if a future change to the crew (e.g. giving
an agent an extra tool, or a bug in dispatch) tried to violate it.

Mechanism: agents/mock_llm.py knows exactly which agent role is about to
act on every call() -- that's `from_agent.role`, the same reliable signal
Trap 2's avoidance already depends on (see agents/mock_llm.py's module
docstring). MockLLM records that role here via `set_current_agent_role()`
at the top of every call(), immediately before it decides whether to emit
a tool Action. The moment CrewAI's executor actually invokes the tool
function moments later, `require_role()` checks that recorded role and
raises if it doesn't match the tool's one authorized role.
"""

from __future__ import annotations

from functools import wraps
from typing import Callable, TypeVar

_current_agent_role: str | None = None


class UnauthorizedToolCallError(Exception):
    """Raised when a tool is invoked while the recorded calling agent's role
    does not match the tool's one authorized role."""


def set_current_agent_role(role: str | None) -> None:
    global _current_agent_role
    _current_agent_role = role


def get_current_agent_role() -> str | None:
    return _current_agent_role


F = TypeVar("F", bound=Callable)


def require_role(allowed_role: str) -> Callable[[F], F]:
    """Decorator: wrap a tool's underlying function so it only executes when
    `get_current_agent_role()` matches `allowed_role` at call time."""

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            actual_role = get_current_agent_role()
            if actual_role != allowed_role:
                raise UnauthorizedToolCallError(
                    f"Tool '{fn.__name__}' may only be called by the '{allowed_role}' agent; "
                    f"the current calling agent is '{actual_role}'."
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator
