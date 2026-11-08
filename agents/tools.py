"""
agents/tools.py — Part 2 Task 6 (lookup tool with designed escalation score)
and the RAG lookup tool used by the Retrieval Agent.

ESCALATION SCORE DESIGN
------------------------
escalation_score = 0.6 * escalated_flag + 0.4 * normalized_recency

  escalated_flag      = 1.0 if the ticket's `escalated` field is True else 0.0
  normalized_recency  = min(days_since_created / 30.0, 1.0)

This is a weighted combination, not a bare boolean OR: a ticket that is NOT
flagged `escalated` but has been open for the full 30-day window still
accumulates up to 0.4 of urgency signal from recency alone, while a flagged
ticket starts at 0.6 regardless of age. Score is clipped to [0, 1].

THRESHOLD
---------
ESCALATION_THRESHOLD = 0.39

Justification: computed directly from dataset.SUPPORT_TICKETS (45 records,
seed=42). Sorting every record's escalation_score, the 80th percentile value
is 0.3867 (see scripts/analyze_escalation_distribution.py for the
reproducible calculation). We round to 0.39. This means the top ~20% most
urgent tickets by this combined flag+recency signal are recommended for
escalation -- consistent with the brief's own example framing ("this
threshold corresponds to the Nth percentile ... in my generated data").
"""

from __future__ import annotations

from dataset import SUPPORT_TICKETS

ESCALATION_THRESHOLD = 0.39

_TICKETS_BY_ID = {t["record_id"]: t for t in SUPPORT_TICKETS}


def compute_escalation_score(escalated: bool, days_since_created: int) -> float:
    escalated_flag = 1.0 if escalated else 0.0
    normalized_recency = min(days_since_created / 30.0, 1.0)
    score = 0.6 * escalated_flag + 0.4 * normalized_recency
    return round(min(max(score, 0.0), 1.0), 4)


def check_support_ticket_status(record_id: str) -> dict:
    """Look up one ticket by record_id. Honest miss if not found -- never invent a row.

    Returns a dict with status, resolution_time_hours, escalation_score, and
    a human-readable summary line (used by the MockLLM's observation parser
    to synthesize a Final Answer without the LLM layer ever touching the
    dataset directly).
    """
    ticket = _TICKETS_BY_ID.get(record_id.strip().upper())
    if ticket is None:
        return {
            "record_id": record_id,
            "found": False,
            "summary": f"UNKNOWN_TICKET: {record_id} was not found in the Ola support-ticket register.",
        }

    score = compute_escalation_score(ticket["escalated"], ticket["days_since_created"])
    recommend_escalation = score >= ESCALATION_THRESHOLD
    verdict = "ESCALATION RECOMMENDED" if recommend_escalation else "Normal Priority"

    summary = (
        f"Ticket {ticket['record_id']} ({ticket['category']}): status is '{ticket['status']}' "
        f"(created {ticket['days_since_created']} days ago, escalated_flag={ticket['escalated']}). "
        f"Escalation score: {score:.2f} (threshold {ESCALATION_THRESHOLD} -> {verdict}). "
        f"Resolution time on record: {ticket['resolution_time_hours']} hours."
    )

    return {
        "record_id": ticket["record_id"],
        "found": True,
        "category": ticket["category"],
        "status": ticket["status"],
        "resolution_time_hours": ticket["resolution_time_hours"],
        "days_since_created": ticket["days_since_created"],
        "escalated_flag": ticket["escalated"],
        "escalation_score": score,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "recommend_escalation": recommend_escalation,
        "summary": summary,
    }


def check_support_ticket_status_tool_text(record_id: str) -> str:
    """Text-formatted version of check_support_ticket_status, in the same
    structured-fields-plus-Summary-line shape the MockLLM's observation
    parser expects (see agents/mock_llm.py)."""
    result = check_support_ticket_status(record_id)
    if not result["found"]:
        return result["summary"]

    return (
        f"Record ID: {result['record_id']}\n"
        f"Category: {result['category']}\n"
        f"Status: {result['status']}\n"
        f"Days Active: {result['days_since_created']}\n"
        f"Escalated Flag: {result['escalated_flag']}\n"
        f"Escalation Score: {result['escalation_score']:.2f} "
        f"(Threshold: {result['escalation_threshold']} -> "
        f"{'ESCALATION RECOMMENDED' if result['recommend_escalation'] else 'Normal Priority'})\n"
        f"Resolution Time Hours: {result['resolution_time_hours']}\n"
        f"Summary: {result['summary']}"
    )


if __name__ == "__main__":
    # Quick sanity check against a few real record ids from the dataset.
    for rid in ["TCK-0001", "TCK-0030", "TCK-0045", "TCK-9999"]:
        print(check_support_ticket_status_tool_text(rid))
        print("-" * 60)
