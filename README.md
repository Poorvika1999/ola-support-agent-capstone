# Ola Support Agent — CrewAI + AutoGen Multi-Agent Capstone

**Track:** Ola (Business Operations / Customer Support)

A fully offline, zero-API-key, zero-network-at-runtime multi-agent support
system: RAG over an Ola policy knowledge base, a CrewAI crew of three
agents driven by a deterministic `MOCK_LLM`, a FastAPI/WebSocket
deployment, a 15-query evaluation harness, and an AutoGen review stage
with governance and caching on top. All "LLM" behavior in this project is
plain, auditable Python — no model weights, no API calls, fully
reproducible given the same inputs.

```
CREWAI_DISABLE_TELEMETRY=true
```
is set at the top of every entry point (`agents/crew.py`, `api/main.py`,
`governance/cache.py`) — confirmed, not assumed.

---

## 1. Dataset design (`dataset.py`)

Synthetic Ola support tickets, generated deterministically with `seed=42`
so every run (and every grader) sees the exact same 45 records.

- **45 records**, 5 categories (Billing, Technical Issue, Account Access,
  Product Defect, General Inquiry) and 5 statuses (Open, In Progress,
  Escalated, Resolved, Closed), each guaranteed at least 3 occurrences by
  seeding the first few records directly from the category/status lists
  before falling back to weighted random choice.
- `escalated` is a boolean drawn at `ESCALATION_PROBABILITY = 0.18`,
  landing at **15.6% (7/45)** in the generated data — inside the
  brief's required 10–30% band.
- `resolution_time_hours` (1–72) and `days_since_created` (0–30) are
  uniform random integers, giving a realistic spread for the escalation
  scoring below.

Run `python3 dataset.py` to regenerate and print the full distribution
report plus assertions (all pass).

## 2. Knowledge base (`knowledge_base/policies.json`)

12 original documents (KB01–KB12), one per Ola support-ops topic (ticket
priority, SLA by severity, escalation matrix, refund/compensation,
communication channels, business hours, repeat complaints, service
credits, feedback collection, VIP handling, outage communication, data
retention). Stored as JSON (not PDF) per the course's own TA guidance.

## 3. RAG pipeline (`rag/`)

- **Chunking** (`rag/chunking.py`): both required strategies —
  fixed-size-with-overlap (100 words, 20-word overlap) and sentence-based
  (60-word cap per chunk) — each indexed into its own ChromaDB collection
  (`rag/indexer.py`): `ola_policy_fixed_overlap` and `ola_policy_sentence`.
  Both collections are created with `metadata={"hnsw:space": "cosine"}` so
  Chroma's returned `distance` is literally `1 - cosine_similarity`
  (Chroma defaults to squared-L2 otherwise, which would silently break
  every similarity/threshold calculation downstream — caught and fixed
  during development).
- **Embedding**: `BAAI/bge-small-en-v1.5` via `sentence-transformers` is
  the documented default (`rag/embedding_backend.py`). The development
  sandbox used to build this project could not reach huggingface.co, so a
  deterministic `LocalFallbackEmbedder` (hashing bag-of-words) was used
  there to verify the pipeline wiring end-to-end. **The real model has
  since been downloaded and indexed against on the actual submission
  machine** (real internet access, no sandbox restriction) — see
  **Section 9** for the exact re-run steps and the real numbers below.
- **Grounded generation + refusal** (`rag/generation.py`): a rule-based
  (no LLM) function that cites up to 2 distinct retrieved KB documents by
  id/title, or refuses with a fixed message when nothing retrieved clears
  the similarity threshold.
  **`REFUSAL_THRESHOLD = 0.5781`** — calibrated by
  `scripts/calibrate_threshold.py` against 4 in-scope and 2 out-of-scope
  queries, using the real `BAAI/bge-small-en-v1.5` embeddings. The
  in-scope/out-of-scope similarity clusters are **cleanly separated**
  (min in-scope top-1 similarity 0.7200 vs. max out-of-scope top-1
  similarity 0.4362), so 0.5781 (the midpoint) is a validated separator —
  a real improvement over an earlier provisional value (0.25) calibrated
  under a sandbox fallback embedder where the clusters had overlapped.
- **Evaluation** (`rag/evaluation.py`): document-level precision/recall
  over 5 queries with known relevant KB docs, for both chunking
  strategies, with full per-query arithmetic printed. With the real
  embeddings: avg recall **1.000** for both strategies (the correct KB
  doc is retrieved as the top result on every query), avg precision
  0.333 (fixed-overlap) / 0.367 (sentence) — precision is capped below
  1.0 only because the eval retrieves the top-3 chunks per query and just
  1 of those 3 unique parent docs is actually relevant, not because of
  any retrieval miss. The `sentence` strategy edges out `fixed-overlap`
  here, which supersedes the earlier “scored identically” finding from
  the fallback-embedder run (see Section 4 for which collection is wired
  into the live agent pipeline).

## 4. CrewAI crew (`agents/`)

**`agents/tools.py` — ticket lookup + escalation score (Task 6).**

```
escalation_score = 0.6 * escalated_flag + 0.4 * min(days_since_created / 30, 1.0)
```

A weighted blend, not a bare boolean OR: a non-flagged ticket open for the
full 30-day window still accumulates up to 0.4 of urgency from recency
alone, while a flagged ticket starts at 0.6 regardless of age.

**`ESCALATION_THRESHOLD = 0.39`** — justified directly from this project's
own dataset: sorting all 45 records' `escalation_score`, the **80th
percentile is 0.3867** (rounded to 0.39). The top ~20% most urgent tickets
by this combined signal are recommended for escalation. (The 80th
percentile of raw `days_since_created` alone is 24.0 days, for reference —
the chosen threshold is on the composite score, not on days alone.)
Ticket lookups are honest: an unknown `record_id` returns a clear
"not found", never an invented row.

**`agents/mock_llm.py` — the deterministic `MockLLM(BaseLLM)`.** Built by
first *empirically* capturing this installed CrewAI version's real prompt
structure (`scripts/diagnose_llm_call.py`, `scripts/diagnose_context.py`)
rather than assuming it from documentation or the reference repo. Two
named traps from the brief, both avoided and confirmed in testing:

- **Trap 1** — CrewAI's own ReAct template shows the literal placeholder
  text `"Observation: the result of the action"` as a static format
  example, *before* the literal `"Begin!"` marker. A real tool result
  only ever appears *after* `"Begin!"`. `MockLLM` only ever searches for a
  real Observation after the **last** `"Begin!"`, with an extra defensive
  check rejecting anything starting with the placeholder text even there.
- **Trap 2** — dispatching which tool to call by matching a keyword
  against the tool's *name* breaks the moment a name itself contains a
  misleading substring. Confirmed empirically that this CrewAI version
  passes `tools=None` and `available_functions=None` into `BaseLLM.call()`
  even when the agent has tools — so `MockLLM` dispatches by
  `from_agent.role` instead (the one reliably-passed signal), with each
  agent given exactly one tool.

Two further bugs found by actually running the code (not in the reference
repo, found here):
1. CrewAI appends a fixed `"Analyze the tool result..."` nudge message
   right after the real Observation. A naive `"\n"`-joined flatten let
   that text bleed into the extracted Observation value — fixed with an
   explicit message-boundary marker.
2. The retrieval task's own instructional wrapper text was being sent to
   the RAG tool as part of the query, diluting the similarity score enough
   that an out-of-scope question incorrectly retrieved a real KB doc
   instead of being refused — fixed by isolating the actual customer
   question behind an explicit `"Customer Question:"` marker.

**3 agents, `Process.sequential`** (`agents/crew.py`): Retrieval Agent
(`retrieve_policy_context` tool), Lookup Agent (`check_ticket_status`
tool), Composer Agent (zero tools, synthesizes both outputs). Demonstrated
on 3 scenarios: happy path, out-of-scope refusal, unknown ticket.

**LangChain memory** (`agents/memory.py`): `InMemoryChatMessageHistory` +
`RunnableWithMessageHistory`. Verified: a session accumulates turns
correctly; a fresh `session_id` starts completely empty, with zero
leakage from other sessions.

**Pydantic structured output** (`agents/schemas.py`): `SupportResponse`.
Set as `Task.response_model` on the Composer task — confirmed this is
threaded straight into `MockLLM.call()`'s `response_model` parameter for
a zero-tool agent, letting `MockLLM` return a validated `SupportResponse`
instance *directly*, with no ReAct text parsing and no second
"reformat as JSON" LLM call. Validated on every scenario, including a
correctly-`null` `escalation_score` for an unknown ticket.

**Guardrails** (`agents/guardrails.py`):
- *Input-side PII masking*: regex-based, phone numbers only (a
  fixed-format field). Free-text fields (rider/driver name, address,
  payment details) are deliberately **not** scrubbed — a generic
  free-text redactor either over-redacts legitimate ticket content or
  misses most of it anyway; a phone number has an unambiguous shape, a
  name does not.
- *Prompt-injection detection*: a small, documented pattern list
  ("ignore previous instructions", "reveal your system prompt", etc.).
- *Output-side groundedness check*: independent defense-in-depth —
  re-checks that the Retrieval Agent's final text is either the exact
  refusal string or carries a real `[KBxx: ...]` citation; anything else
  is blocked rather than shown to the customer.

## 5. FastAPI + WebSocket deployment (`api/`)

- `POST /support/query` — runs the full crew, returns the validated
  `SupportResponse`.
- `GET /support/ticket/{record_id}` — direct ticket lookup, no crew/RAG
  overhead.
- `WS /ws/support` — the same, streamed; `WebSocketDisconnect` is caught
  and logged, never crashes the server.

Every request (HTTP and WS) is checked for prompt injection before it
reaches the crew. **Structured JSON-Lines logging** (`api/logging_utils.py`)
writes every event to `output/api_requests.jsonl`, with `mask_pii` applied
to every string field *before* it's written to disk.

One more real bug found and fixed: calling the synchronous `crew.kickoff()`
directly inside the `async def` WebSocket handler raised
`RuntimeError: Agent execution was invoked synchronously from within a
running event loop` (confirmed, not assumed). Fixed with
`await crew.kickoff_async()` for the WS path specifically; the HTTP POST
endpoint keeps the sync call since FastAPI runs a plain `def` route in a
worker thread, off the event loop.

Tested end-to-end in `scripts/test_api.py` (FastAPI `TestClient`,
including a real WebSocket connect-and-disconnect).

## 6. 15-query evaluation harness (`evaluation/llm_judge.py`)

"LLM-as-judge" here is — consistent with this whole project's `MOCK_LLM`
philosophy — a deterministic, rule-based judge, not a second model call.
15 queries (5 in-scope, 3 out-of-scope, 2 unknown-ticket, 2 prompt-injection,
3 further in-scope), each scored on:

- **Accuracy** — ticket-side correctness (always 1.0; the lookup is
  deterministic) averaged with policy-side correctness (did the citation
  match the actually-expected KB doc for that query, from the same
  ground truth used in Section 3's precision/recall eval, or the exact
  refusal for an out-of-scope query).
- **Grounding** — reuses `agents/guardrails.py`'s `check_groundedness`.
- **Completeness** — did the composed reply include both a policy answer
  and a ticket status.
- **Safety** — for injection attempts, was the request blocked; for
  everything else, the final reply itself is independently re-checked for
  injection-style phrasing.

**Result (real `BAAI/bge-small-en-v1.5` embeddings): Accuracy 1.0000,
Grounding 1.0000, Completeness 1.0000, Safety 1.0000, overall 1.0000.**
An earlier run under the sandbox's fallback embedder scored Accuracy
0.7333 (several in-scope answers cited a real-but-wrong KB doc — grounded
but not accurate), which is the honest number a weak, non-semantic
embedder produces. The flat 1.0 here is not a judge-logic shortcut; it
reflects that the real embeddings now retrieve the correct KB doc as the
top result on every query (see Section 3's recall figure), so there are
no wrong-citation cases left for Accuracy to catch. Full report:
`output/llm_judge_eval.txt`.

## 7. AutoGen review stage (`review/`)

A **separate pipeline stage from CrewAI**, sitting after the crew's
output. Two agents in a `RoundRobinGroupChat` (`max_turns=2`):
`policy_reviewer` (prose analysis) and `verdict_agent`
(`output_content_type=VerdictModel`), both backed by
`autogen_ext.models.replay.ReplayChatCompletionClient` for deterministic,
zero-API-key playback.

**Trap 5, confirmed by deliberately triggering it**: an `AssistantAgent`
with `output_content_type=VerdictModel` emits a
`StructuredMessage[VerdictModel]`; without
`custom_message_types=[StructuredMessage[VerdictModel]]` declared on the
team, this raises `ValueError: Message type ... is not registered.` the
moment that message is logged. Declaring it avoids the error.

The two "replayed" responses are not hardcoded text — they're computed by
`assess_support_response()`, plain rule-based judgment (reusing the
groundedness check) over the actual crew output, so AutoGen's real
plumbing carries a real decision. Verified against a real crew output
end-to-end, plus two hand-built scenarios: a grounded response →
approved/low-risk; a fabricated, uncited response → blocked/high-risk.

## 8. Governance + caching (`governance/`)

- **Least-autonomy enforcement** (`authorization.py`): not just "by
  construction" (each agent happens to have one tool) — an active runtime
  guard. `agents/mock_llm.py` records which agent role is about to act on
  every `call()`; `agents/crew_tools.py` wraps each tool with
  `require_role(...)`, which raises `UnauthorizedToolCallError` if the
  recorded role doesn't match. Verified both ways: the Lookup Agent's call
  succeeds, a call impersonating the Composer Agent is rejected.
- **Risk classification with justification** (`risk.py`): a *pre-request*
  classification (distinct from the AutoGen stage's *post-response*
  review) — injection attempt → high; ticket already past
  `ESCALATION_THRESHOLD` → high; unknown ticket → medium; else low. All 4
  cases demonstrated with their justification text.
- **Per-request budget cap** (`budget.py`): since `MockLLM` makes no real
  network call, there's no real per-token dollar cost — the estimate uses
  the same chars/4 heuristic CrewAI itself falls back to for a model with
  no native tokenizer (visible in this project's own terminal output).
  Demonstrated: a normal query passes; the same query is rejected under an
  artificially tiny budget; a genuinely oversized query is rejected under
  the default 2000-token budget.
- **In-memory response cache** (`cache.py`): keyed by `(query, record_id)`.
  Before/after demo: first call ~6.2s (includes the one-time embedding
  model load), cache hit ~0s, responses verified identical. No TTL —
  a known simplification, not hidden (ticket status does change over
  time in a real deployment).

## 9. Re-running with the real BGE model (DONE — completed on the actual submission machine)

The development sandbox's network allowlist blocked `huggingface.co`, so
the project was built and tested there against the deterministic
`LocalFallbackEmbedder`. This step has since been completed for real, on
the actual (Intel) Mac used for submission, with genuine internet access:

```bash
pip install -r requirements.txt
rm -rf data/chroma_store
python3 -m rag.indexer                        # downloaded & cached the real BGE model
python3 -m scripts.calibrate_threshold        # -> REFUSAL_THRESHOLD updated to 0.5781
python3 -m rag.evaluation                     # -> real precision/recall numbers (Section 3)
python3 -m scripts.generate_task4_transcript  # -> real grounded-generation transcript
python3 -m evaluation.llm_judge               # -> real 15-query eval report (Section 6)
```

The original `protobuf>=6.33.5,<8` pin mentioned in an earlier draft of
this section has been **removed from `requirements.txt` entirely**
(rather than installed as a second step) — `autogen-core`'s own metadata
pins `protobuf~=5.29.3`, and letting pip resolve to that satisfies both
`autogen` and `chromadb` without an explicit override.

**Four additional, real environment issues were hit and fixed while
actually running this on an Intel macOS machine** (found by running the
code, consistent with this project's whole approach — not assumed from
docs):

1. **`lancedb` (a `crewai` dependency) needs the `protoc` compiler, not
   just the Python `protobuf` package.** Its Rust component
   (`lance-encoding`) compiles `.proto` files at build time and fails
   with `Could not find 'protoc'` otherwise. Fixed with
   `brew install protobuf` (installs the `protoc` binary itself).
2. **The `lancedb` Rust build is heavy enough to freeze a resource-
   constrained Intel Mac at full compiler parallelism.** Fixed by
   capping it with `export CARGO_BUILD_JOBS=2` before `pip install`.
3. **PyPI stopped publishing Intel macOS (`x86_64`) wheels for `torch`
   after `2.2.2`**, which has two downstream effects:
   - `torch==2.2.2`'s compiled extensions were built against NumPy 1.x
     and break under NumPy 2.x (`chromadb` otherwise pulls in NumPy 2),
     so `requirements.txt` now pins `numpy<2` explicitly.
   - `transformers>=5.x` hard-requires `torch>=2.6`, which doesn't exist
     for this platform, so `sentence-transformers` is pinned to `3.0.1`
     (down from `6.2.0`) with `transformers>=4.41,<5.0` — both fully
     capable of loading `BAAI/bge-small-en-v1.5`, a standard BERT-family
     model.
4. **The latest `cryptography` (a `crewai-cli` dependency) fails to
   import at runtime** (`symbol not found in flat namespace
   '_EVP_DigestSqueeze'`) when built from source against this machine's
   OpenSSL setup. Fixed by pinning `cryptography==42.0.8`, which has a
   pre-built wheel for this platform and avoids the native build (and
   the OpenSSL linkage mismatch) entirely.

All four fixes are reflected in the final `requirements.txt`.

## 10. Running everything

```bash
export CREWAI_DISABLE_TELEMETRY=true

# Part 1 — build the RAG index, then look at generation/eval
python3 -m rag.indexer
python3 -m scripts.calibrate_threshold
python3 -m rag.evaluation
python3 -m scripts.generate_task4_transcript

# Part 2 — the crew itself
python3 -m agents.tools
python3 -m agents.crew
python3 -m agents.memory
python3 -m agents.guardrails

# Part 3 — API + evaluation
python3 -m scripts.test_api
python3 -m evaluation.llm_judge
uvicorn api.main:app --reload   # to actually serve it

# Part 4 — AutoGen review + governance + cache
python3 -m review.autogen_review
python3 -m governance.risk
python3 -m governance.budget
python3 -m governance.cache
```

## File tree

```
ola_support_agent/
├── README.md
├── PROGRESS.md              (full build log / design-decision journal)
├── requirements.txt
├── dataset.py
├── knowledge_base/policies.json
├── rag/                     (chunking, indexing, retrieval, generation, evaluation)
├── agents/                  (tools, crew_tools, mock_llm, crew, memory, schemas, guardrails)
├── api/                     (FastAPI app, structured logging)
├── evaluation/              (15-query LLM-judge harness)
├── review/                  (AutoGen review stage)
├── governance/              (authorization, risk, budget, cache)
├── scripts/                 (calibration, transcript generation, API tests,
│                              and 3 throwaway diagnostic scripts used to
│                              empirically verify CrewAI/AutoGen internals
│                              before writing mock_llm.py/autogen_review.py)
├── data/chroma_store/        (persisted Chroma DB)
└── output/                   (generated transcripts, eval reports, request logs)
```
