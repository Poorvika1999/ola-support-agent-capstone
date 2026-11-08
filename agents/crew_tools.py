"""
agents/crew_tools.py — CrewAI @tool wrappers around the Part 1 RAG pipeline
and the Part 2 Task 6 ticket-lookup logic.

Each tool is intentionally given to exactly ONE agent in agents/crew.py. This
is a deliberate design choice, not an accident: because `tools` and
`available_functions` are not reliably passed into our custom MockLLM.call()
(confirmed empirically against this CrewAI version -- both come through as
None), the MockLLM cannot dispatch by inspecting a tool's schema at runtime.
Giving each agent exactly one tool, and dispatching by `from_agent.role`
instead (which IS reliably passed), sidesteps the problem entirely rather
than falling into the brief's named Trap 2 (matching by a substring of the
tool's name). See agents/mock_llm.py for the dispatch logic itself.
"""

from __future__ import annotations

from crewai.tools import tool

from agents.mock_llm import LOOKUP_AGENT_ROLE, RETRIEVAL_AGENT_ROLE
from agents.tools import check_support_ticket_status_tool_text
from governance.authorization import require_role
from rag.embedding_backend import create_embedding_model
from rag.generation import generate_grounded_answer
from rag.indexer import COLLECTION_FIXED, get_chroma_client
from rag.retriever import retrieve_chunks

# Lazy module-level singletons: the embedding model and Chroma collection are
# expensive to (re)load, so every call to the tool below reuses the same one
# instead of re-opening the client/model per call.
_embedding_model = None
_collection = None


def _get_collection():
    global _embedding_model, _collection
    if _collection is None:
        _embedding_model, backend_name = create_embedding_model()
        print(f"[crew_tools] Embedding backend in use: {backend_name}")
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_FIXED, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
    return _collection, _embedding_model


@tool("retrieve_policy_context")
@require_role(RETRIEVAL_AGENT_ROLE)
def retrieve_policy_context(query: str) -> str:
    """Retrieve the relevant Ola support-policy excerpt for a natural-language
    question and generate a grounded answer from it, or refuse if nothing
    retrieved is similar enough to the question. Uses the fixed-overlap
    Chroma collection built in Part 1."""
    collection, model = _get_collection()
    retrieved = retrieve_chunks(collection, model, query, top_k=3)
    return generate_grounded_answer(query, retrieved)


@tool("check_ticket_status")
@require_role(LOOKUP_AGENT_ROLE)
def check_ticket_status(record_id: str) -> str:
    """Look up a single Ola support ticket by its record_id (e.g. 'TCK-0030')
    and return its status, resolution time, and computed escalation score
    against the dataset-derived threshold."""
    return check_support_ticket_status_tool_text(record_id)
