# Ola Support Agent — Build Notes

Notes on how I built this and why I made the calls I made. Keeping this
mainly so I can walk through it confidently later (thresholds, the
groundedness logic, the trap-avoidance in the mock LLM) without having to
re-derive any of it from scratch.

## What this is

A CrewAI + AutoGen multi-agent system for the Ola (business ops /
customer support) track. Everything runs offline — no API keys, no
network calls at runtime. All "LLM" behavior is a deterministic,
rule-based `MOCK_LLM`, which the brief requires for reproducible grading.
Four parts, 100 marks total.

---

## Part 1 — RAG pipeline

**Dataset (`dataset.py`).** 45 synthetic support tickets, generated with
`seed=42` so every run produces the same records. I seeded the first few
records directly from the category/status lists before falling back to
weighted random choice, so every category and status is guaranteed to show
up at least a few times. `escalated` lands at 15.6% (7/45), inside the
required 10–30% band.

**Knowledge base (`knowledge_base/policies.json`).** 12 original docs
(KB01–KB12), one per Ola support topic — ticket priority, SLA by
severity, escalation matrix, refunds, communication channels, business
hours, repeat complaints, service credits, feedback, VIP handling,
outage comms, data retention.

**Chunking + indexing (`rag/chunking.py`, `rag/indexer.py`).** Both
required strategies — fixed-overlap (100 words, 20-word overlap) and
sentence-based (60-word cap) — each indexed into its own Chroma
collection. I hit an early bug here: Chroma defaults to squared-L2
distance unless you explicitly set `hnsw:space="cosine"`, which would
have silently broken the `1 - distance = similarity` math everywhere
downstream. Caught it early by checking the actual numbers rather than
assuming the defaults.

**Grounded generation + refusal threshold (`rag/generation.py`,
`scripts/calibrate_threshold.py`).** `REFUSAL_THRESHOLD = 0.5781`,
calibrated against the real `BAAI/bge-small-en-v1.5` embeddings on my own
machine: in-scope queries score a minimum of 0.72 similarity, while
out-of-scope queries top out at 0.44 — a clean, validated separation. (I
originally had this running against a provisional 0.25 while developing
in a sandboxed environment without real internet access, where a weaker
fallback embedder produced overlapping, unreliable clusters. Once I had
real internet on my own Mac, I re-indexed with the real model and
recalibrated properly — more on that below.)

**Precision/recall eval (`rag/evaluation.py`).** 5 queries with known
relevant docs, precision/recall computed per query with the arithmetic
printed. With the real embeddings: recall is a perfect 1.000 for both
chunking strategies — the correct doc is retrieved as the top result
every time. Precision sits at 0.333–0.367, which isn't a retrieval
problem; the eval pulls the top 3 chunks per query and only 1 of those 3
unique docs is actually relevant, so 1/3 is close to the ceiling given
how the eval is written.

### Getting the real embedding model working

I built and tested most of this in an environment that couldn't reach
huggingface.co, so I developed against a deterministic hashing
fallback embedder just to confirm the pipeline wiring was correct
end-to-end. Once I had the project on my own machine with real
internet, I went to re-index with the actual `BAAI/bge-small-en-v1.5`
model and hit four separate environment issues, in order:

1. `lancedb` (pulled in by `crewai`) needs the `protoc` compiler
   installed to build its Rust component, not just the Python
   `protobuf` package. Fixed with `brew install protobuf`.
2. That same Rust build is heavy enough that it froze my Mac at full
   compiler parallelism. Capped it with `export CARGO_BUILD_JOBS=2`
   before reinstalling, which fixed it.
3. PyPI doesn't ship a `torch` wheel newer than `2.2.2` for Intel macOS.
   That has two knock-on effects: `torch==2.2.2`'s compiled extensions
   break under NumPy 2.x (which `chromadb` otherwise pulls in), so I
   pinned `numpy<2`; and `transformers>=5.0` hard-requires `torch>=2.6`,
   which doesn't exist for my platform, so I downgraded to
   `sentence-transformers==3.0.1` with `transformers>=4.41,<5.0` — both
   still load `BAAI/bge-small-en-v1.5` fine, since it's a standard
   BERT-family model.
4. The newest `cryptography` (a dependency of `crewai-cli`) failed to
   import at runtime with a missing OpenSSL symbol
   (`_EVP_DigestSqueeze`) when built from source against my system's
   OpenSSL. Pinned `cryptography==42.0.8` to get a prebuilt wheel
   instead of building from source.

All four fixes are reflected in `requirements.txt`. After that, I reran
the full chain — `rag.indexer`, `scripts.calibrate_threshold`,
`rag.evaluation`, `scripts.generate_task4_transcript`,
`evaluation.llm_judge` — with the real embeddings in place.

---

## Part 2 — CrewAI crew

**Ticket lookup + escalation score (`agents/tools.py`).**
`escalation_score = 0.6*escalated_flag + 0.4*min(days_since_created/30, 1.0)`
— a weighted blend rather than a flat boolean OR, so a long-open but
not-flagged ticket still accumulates urgency from age alone.
`ESCALATION_THRESHOLD = 0.39` comes from the real 80th percentile of the
score distribution across all 45 dataset records (measured: 0.3867,
rounded to 0.39) — computed directly from the dataset, not picked
arbitrarily. Unknown ticket IDs get an honest "not found," never an
invented row.

**MockLLM + 3-agent crew (`agents/mock_llm.py`, `agents/crew_tools.py`,
`agents/crew.py`).** Three agents, `Process.sequential`: a Retrieval
Agent, a Lookup Agent, and a Composer Agent with no tools of its own.

**LangChain memory (`agents/memory.py`).** `InMemoryChatMessageHistory`
+ `RunnableWithMessageHistory`. Verified a session accumulates turns
correctly and a fresh `session_id` starts completely empty. Both classes
are technically deprecated in this LangChain version in favor of
LangGraph persistence, but the brief names these two specifically, so I
used them as asked.

**Structured output (`agents/schemas.py`).** A `SupportResponse`
Pydantic model, wired through `Task.response_model` into `MockLLM.call()`
for the zero-tool Composer agent — this bypasses CrewAI's ReAct text
parsing entirely and returns a validated model directly. Confirmed
correct on all three demo scenarios, including a correctly-null
`escalation_score` for an unknown ticket.

**Guardrails (`agents/guardrails.py`).** PII masking is regex-based and
only targets phone numbers, since that's the one fixed-format field;
free-text fields (names, addresses, payment info) are deliberately left
alone, because a generic redactor either over-redacts real ticket content
or misses most of it anyway. Prompt-injection detection runs off a small
documented pattern list. A separate output-side groundedness check
independently re-verifies that the final answer is either the exact
refusal string or carries a real `[KBxx: ...]` citation — anything else
gets blocked before it reaches the customer.

### A couple of things worth remembering for later

**Why I dispatch tools by `from_agent.role` instead of matching on the
tool's name.** I confirmed empirically, with two throwaway scripts
(`scripts/diagnose_llm_call.py`, `scripts/diagnose_context.py`), that this
version of CrewAI passes `tools=None` and `available_functions=None`
into `BaseLLM.call()` even when the agent genuinely has tools. So
`from_agent.role` is the only signal I can actually rely on — since each
agent only ever has one tool, dispatching by role and hardcoding which
tool that role uses sidesteps the substring-matching trap entirely.

**Why I slice the prompt after the literal "Begin!" marker to find the
real tool observation.** CrewAI's own ReAct template shows a placeholder
line, `"Observation: the result of the action"`, in the static
instructions shown to the agent — and that text appears *before* the
`"Begin!"` marker in the rendered prompt. The real observation only ever
appears after it. So `MockLLM` only searches text after the last
`"Begin!"`, with an extra check that rejects anything that still starts
with the placeholder text.

**Two real bugs I found by actually running this, not from any
reference material:**
- CrewAI appends a fixed nudge message right after the real observation.
  Joining all messages with a plain newline let that nudge text bleed
  into what I extracted as the "observation" — fixed by joining on an
  unambiguous boundary marker instead.
- The retrieval task's own instructional wrapper text was getting sent
  to the RAG tool as part of the query, diluting the similarity score
  enough that an out-of-scope question (asking for a movie
  recommendation) incorrectly matched a real KB doc instead of being
  refused. Fixed by isolating the actual customer question behind an
  explicit marker that `MockLLM` parses for.

**On the chunking-strategy choice:** I wired the fixed-overlap collection
into the live agent pipeline as the "production" one. At the time, under
the fallback embedder, both strategies scored identically on the Part 1
eval, so it was a reasonable-but-arbitrary pick rather than a measured
win. With the real embeddings now in, the sentence-based strategy
actually edges it out slightly (0.367 vs 0.333 precision) — worth
mentioning if asked, though I haven't rewired the pipeline over it since
it's a small difference either way.

---

## Part 3 — FastAPI + logging + eval

**Endpoints (`api/main.py`).** `POST /support/query` runs the full crew;
`GET /support/ticket/{id}` does a direct lookup with no crew overhead;
`WS /ws/support` does the same over a WebSocket, with
`WebSocketDisconnect` handled gracefully.

**Logging (`api/logging_utils.py`).** Every event is written as JSON
Lines to `output/api_requests.jsonl`, with every string field run through
`mask_pii` first.

**15-query LLM-judge eval (`evaluation/llm_judge.py`).** A deterministic,
rule-based judge — in keeping with the MOCK_LLM approach, not a second
model call — scoring Accuracy, Grounding, Completeness, and Safety.
With the real embeddings, this now scores a perfect 1.0 across all four
metrics. Under the earlier fallback embedder it scored Accuracy 0.73
(several in-scope answers cited a real-but-wrong KB doc — grounded, but
not accurate), which was the honest result of a weak embedder, not a
logic bug. The current 1.0 isn't suspicious for the same reason — the
real embeddings genuinely retrieve the right doc every time, so there
are no wrong-citation cases left for Accuracy to catch.

Tested via `scripts/test_api.py`, including a real WebSocket
connect-and-disconnect.

**One more real bug, found here:** calling the synchronous
`crew.kickoff()` directly inside the WebSocket's `async def` handler
raised a `RuntimeError` about being invoked from within a running event
loop. Fixed by using `await crew.kickoff_async()` specifically on the WS
path; the HTTP POST endpoint keeps the sync call since FastAPI runs plain
`def` routes in a worker thread, off the event loop, where blocking is
fine.

---

## Part 4 — AutoGen review + governance + caching

**AutoGen review stage (`review/verdict.py`, `review/autogen_review.py`).**
A 2-agent `RoundRobinGroupChat` (`policy_reviewer`, `verdict_agent`),
backed by `ReplayChatCompletionClient` for deterministic playback. The
two "replayed" responses aren't hardcoded text — they come from
`assess_support_response()`, a real rule-based judgment reusing the
groundedness check, so AutoGen's actual plumbing is carrying a real
decision rather than a script. Verified against a real crew output, plus
two hand-built scenarios (grounded → approved; fabricated/uncited →
blocked).

**Least-autonomy enforcement (`governance/authorization.py`).** An active
runtime guard, not just "true by construction": `mock_llm.py` records
which role is acting on each call, and `crew_tools.py` wraps each tool
with a role check that raises if the acting role doesn't match. Verified
both the success and the rejection path.

**Risk classification (`governance/risk.py`).** A pre-request
classification, separate from the AutoGen stage's post-response review:
injection attempt → high, ticket already past the escalation threshold →
high, unknown ticket → medium, else low.

**Budget cap (`governance/budget.py`).** Since `MockLLM` makes no real
API call, there's no real dollar cost to track, so the token estimate
uses the same chars/4 heuristic CrewAI itself falls back to for a model
with no native tokenizer. Demonstrated a normal query passing, the same
query getting rejected under an artificially tiny budget, and a
genuinely oversized query getting rejected under the default budget.

**Response cache (`governance/cache.py`).** Keyed by `(query,
record_id)`. First call takes ~6.2s (includes the one-time embedding
model load), the cache hit is ~0s, and I verified the responses are
identical. No TTL — a known simplification I'm not hiding, since ticket
status does change over time in a real deployment.

**Dependency bug here:** installing `autogen-agentchat`/`autogen-ext`
downgraded `protobuf` to a version `chromadb` couldn't use (via its
`opentelemetry` dependency chain). I originally fixed this with an
explicit `protobuf>=6.33.5,<8` pin, then later simplified it by removing
the pin from `requirements.txt` altogether — `autogen-core`'s own
`protobuf~=5.29.3` constraint resolves fine on its own and satisfies
both libraries without an override.

---

## File tree

```
ola_support_agent/
├── dataset.py
├── knowledge_base/
│   ├── __init__.py
│   └── policies.json
├── rag/
│   ├── __init__.py
│   ├── chunking.py
│   ├── embedding_backend.py
│   ├── indexer.py
│   ├── retriever.py
│   ├── generation.py
│   └── evaluation.py
├── agents/
│   ├── __init__.py
│   ├── tools.py
│   ├── crew_tools.py
│   ├── mock_llm.py
│   ├── crew.py
│   ├── memory.py
│   ├── schemas.py
│   └── guardrails.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── logging_utils.py
├── evaluation/
│   ├── __init__.py
│   └── llm_judge.py
├── review/
│   ├── __init__.py
│   ├── verdict.py
│   └── autogen_review.py
├── governance/
│   ├── __init__.py
│   ├── authorization.py
│   ├── risk.py
│   ├── budget.py
│   └── cache.py
├── scripts/
│   ├── __init__.py
│   ├── calibrate_threshold.py
│   ├── generate_task4_transcript.py
│   ├── test_api.py
│   ├── diagnose_llm_call.py      (throwaway — used to verify CrewAI's real prompt structure)
│   └── diagnose_context.py       (throwaway, same purpose)
├── data/chroma_store/             (persisted Chroma DB, built by rag/indexer.py)
└── output/                        (generated transcripts, eval reports, request logs)
```
