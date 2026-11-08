"""
governance/budget.py — Part 4: per-request token/cost budget cap.

MockLLM makes no real network call, so there is no real per-token dollar
cost to meter. The honest thing to do under that constraint is simulate a
budget check against an *estimated* token count, using the same rough
heuristic CrewAI itself falls back to for a model it has no native
tokenizer for (confirmed in this project's own terminal output: "Using an
approximate input token count ... assuming one token per four characters")
-- so the estimate here is consistent with what CrewAI already assumes
about MockLLM, not an arbitrary invented number.
"""

from __future__ import annotations

# A generous default: the system prompt + task template the crew actually
# sends per agent call runs a few hundred characters even for a short
# query (see scripts/diagnose_llm_call.py's captured prompt), so a cap in
# the low thousands is "generous but not infinite" for a real request.
DEFAULT_BUDGET_TOKENS = 2000

CHARS_PER_TOKEN = 4  # same heuristic CrewAI's own token-count fallback uses


class BudgetExceededError(Exception):
    """Raised when a request's estimated token cost exceeds its budget cap."""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def enforce_budget(query: str, record_id: str, budget_tokens: int = DEFAULT_BUDGET_TOKENS) -> int:
    """Rough upfront estimate of this request's token cost (the raw query +
    record_id text -- a cheap, honest proxy given MockLLM has no real
    per-token cost). Returns the estimate if it fits in `budget_tokens`,
    else raises BudgetExceededError before the crew is ever built."""
    estimated = estimate_tokens(query) + estimate_tokens(record_id)
    if estimated > budget_tokens:
        raise BudgetExceededError(
            f"Estimated {estimated} tokens for this request exceeds the "
            f"per-request budget of {budget_tokens} tokens."
        )
    return estimated


if __name__ == "__main__":
    print("Case 1: normal-length query under the default budget -> should PASS")
    normal_query = "Do VIP customers get faster support for their tickets?"
    estimated = enforce_budget(normal_query, "TCK-0030")
    print(f"  Estimated tokens: {estimated} (budget {DEFAULT_BUDGET_TOKENS}) -> accepted\n")

    print("Case 2: same normal query, but an artificially tiny per-request budget -> should be REJECTED")
    try:
        enforce_budget(normal_query, "TCK-0030", budget_tokens=5)
        print("  BUG: request was not rejected")
    except BudgetExceededError as e:
        print(f"  Correctly rejected: {e}\n")

    print("Case 3: a deliberately huge query exceeding the DEFAULT budget -> should be REJECTED")
    huge_query = "Please explain in extreme detail " + ("the Ola refund policy " * 400)
    try:
        enforce_budget(huge_query, "TCK-0030")
        print("  BUG: request was not rejected")
    except BudgetExceededError as e:
        print(f"  Correctly rejected: {e}")
