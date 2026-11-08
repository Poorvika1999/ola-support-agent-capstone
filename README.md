# Ola Support Agent — CrewAI + AutoGen Multi-Agent Capstone

**Track:** Ola (Business Operations / Customer Support)

This is an offline, multi-agent customer support system for Ola: RAG over
a policy knowledge base, a CrewAI crew of three agents, a FastAPI/
WebSocket deployment, a 15-query evaluation harness, and an AutoGen
review stage with governance and caching on top. There's no API key and
no network call at runtime — every "LLM" in this project is a
deterministic, rule-based `MOCK_LLM`, which is what the brief requires
for reproducible grading. `CREWAI_DISABLE_TELEMETRY=true` is set at the
top of every entry point (`agents/crew.py`, `api/main.py`,
`governance/cache.py`).

---

## 1. Dataset (`dataset.py`)

Synthetic Ola support tickets, generated with `seed=42` so every run
produces the same 45 records. I seeded the first few records directly
from the category/status lists so every category (Billing, Technical
Issue, Account Access, Product Defect, General Inquiry) and status
(Open, In Progress, Escalated, Resolved, Closed) shows up at least 3
times, before falling back to weighted random choice for the rest.
`escalated` is drawn at an 18% probability and lands at 15.6% (7/45) in
the generated data, inside the required 10–30% band. Run
`python3 dataset.py` to regenerate it and see the full distribution
report with its assertions.

## 2. Knowledge base (`knowledge_base/policies.json`)

12 original documents (KB01–KB12), one per Ola support-ops topic:
ticket priority, SLA by severity, the escalation matrix, refunds,
communication channels, business hours, repeat complaints, service
credits, feedback, VIP handling, outage communication, and data
retention. Stored as JSON rather than PDF, per the course's own
guidance.

## 3. RAG pipeline (`rag/`)

Both required chunking strategies are implemented — fixed-size with
overlap (100 words, 20-word overlap) and sentence-based (60-word cap) —
each indexed into its own Chroma collection. I explicitly set
`metadata={"hnsw:space": "cosine"}` on both collections; Chroma defaults
to squared-L2 distance otherwise, which would have silently broken the
`1 - distance = similarity` math everywhere downstream. I only caught
that by checking the actual numbers rather than trusting the defaults.

For embeddings, I use `BAAI/bge-small-en-v1.5` via `sentence-transformers`.
I did most of the development in an environment that couldn't reach
huggingface.co, so I built and tested the pipeline against a
deterministic hashing fallback embedder just to confirm the wiring was
correct end-to-end. Once I had the project on my own machine with real
internet, I re-indexed with the actual model — see the setup notes near
the bottom for what that took.

`REFUSAL_THRESHOLD = 0.5781` in `rag/generation.py` — calibrated with
`scripts/calibrate_threshold.py` against 4 in-scope and 2 out-of-scope
queries, using the real embeddings. In-scope queries score a minimum of
0.72 similarity; out-of-scope queries top out at 0.44. That's a clean
separation, so 0.5781 (the midpoint) is a validated threshold rather
than a guess. Earlier, while developing against the fallback embedder, I
had this provisionally set to 0.25 because the clusters genuinely
overlapped under that weaker embedder — I flagged that explicitly at the
time rather than hiding it, and fixed it properly once I had the real
model.

For evaluation (`rag/evaluation.py`), I run 5 queries with known
relevant docs and compute precision/recall with the arithmetic printed.
With the real embeddings, recall is a perfect 1.000 for both chunking
strategies — the correct doc comes back as the top result every time.
Precision sits around 0.33–0.37, but that's not a retrieval miss: the
eval pulls the top 3 chunks per query, and only 1 of those 3 unique docs
is actually relevant, so 1/3 is close to the ceiling given how the eval
itself is written. The sentence-based strategy edges out fixed-overlap
slightly here (0.367 vs 0.333), which is worth knowing even though I
kept fixed-overlap wired into the live pipeline (see Section 4).

## 4. CrewAI crew (`agents/`)

**Ticket lookup + escalation score (`agents/tools.py`).**

```
escalation_score = 0.6 * escalated_flag + 0.4 * min(days_since_created / 30, 1.0)
```

I used a weighted blend instead of a plain boolean OR so that a
non-flagged ticket open for the full 30 days still accumulates urgency
from age, while a flagged ticket starts at 0.6 regardless of age.
`ESCALATION_THRESHOLD = 0.39` comes from the real 80th percentile of the
score distribution across all 45 records (0.3867, rounded) — I computed
that directly from the dataset rather than picking a round number.
Unknown ticket IDs get an honest "not found," never an invented row.

**The mock LLM (`agents/mock_llm.py`).** Before writing this, I captured
CrewAI's actual prompt structure with two throwaway scripts
(`scripts/diagnose_llm_call.py`, `scripts/diagnose_context.py`) rather
than assuming it from docs. Two things I had to work around:

- CrewAI's own ReAct template shows a placeholder line —
  `"Observation: the result of the action"` — in the static instructions
  shown to the agent, and that line appears *before* the literal
  `"Begin!"` marker in the rendered prompt. The real tool result only
  ever shows up after it. So `MockLLM` only searches for an observation
  after the last `"Begin!"`, with an extra check that rejects anything
  still matching the placeholder text.
- Dispatching which tool to call by matching a keyword against the
  tool's name breaks the moment a name contains a misleading substring —
  and I confirmed this version of CrewAI passes `tools=None` and
  `available_functions=None` into `BaseLLM.call()` even when the agent
  genuinely has tools. So I dispatch by `from_agent.role` instead, since
  each agent only ever has one tool.

Two more bugs I only found by actually running this (not from any
reference material): CrewAI appends a fixed nudge message right after
the real observation, and a naive newline-join let that nudge text bleed
into what I was extracting — fixed by joining on an unambiguous boundary
marker instead. And the retrieval task's own instructional wrapper text
was getting sent to the RAG tool as part of the query, diluting the
similarity score enough that an out-of-scope question incorrectly
matched a real KB doc instead of getting refused — fixed by isolating
the actual customer question behind an explicit marker.

Three agents run sequentially: a Retrieval Agent, a Lookup Agent, and a
Composer Agent with no tools of its own that synthesizes both outputs.
I've demonstrated this on three scenarios: a happy path, an out-of-scope
refusal, and an unknown ticket.

**Memory (`agents/memory.py`).** `InMemoryChatMessageHistory` +
`RunnableWithMessageHistory`, as the brief specifically names these
classes (they're technically deprecated in this LangChain version in
favor of LangGraph persistence, but I used what was asked for). A
session accumulates turns correctly, and a fresh `session_id` starts
completely empty.

**Structured output (`agents/schemas.py`).** A `SupportResponse` Pydantic
model, set as the Composer task's `response_model`. That threads straight
into `MockLLM.call()`'s own `response_model` parameter for the zero-tool
agent, so I get a validated model back directly — no ReAct text parsing,
no second "reformat as JSON" step. This holds on every scenario,
including a correctly-null `escalation_score` for an unknown ticket.

**Guardrails (`agents/guardrails.py`).** PII masking only targets phone
numbers, since that's the one field with a fixed, recognizable format —
I deliberately left free-text fields (names, addresses, payment details)
alone, since a generic redactor either over-redacts real content or
misses most of it anyway. There's a small documented pattern list for
prompt-injection detection, and a separate output-side groundedness
check that independently verifies the final answer is either the exact
refusal string or carries a real `[KBxx: ...]` citation — anything else
gets blocked before it reaches the customer.

## 5. FastAPI + WebSocket deployment (`api/`)

`POST /support/query` runs the full crew; `GET /support/ticket/{id}`
does a direct lookup with no crew overhead; `WS /ws/support` does the
same over a WebSocket, with disconnects handled gracefully. Every
request, HTTP or WS, is checked for prompt injection before it reaches
the crew. Every logged event goes through `mask_pii` before it's written
to `output/api_requests.jsonl`.

One real bug here: calling the synchronous `crew.kickoff()` directly
inside the WebSocket's `async def` handler raised a `RuntimeError` about
being invoked from within a running event loop. I fixed it by using
`await crew.kickoff_async()` specifically on the WS path; the HTTP POST
endpoint keeps the sync call since FastAPI runs plain `def` routes in a
worker thread, off the event loop, where that's safe.

Tested end-to-end in `scripts/test_api.py`, including a real WebSocket
connect and disconnect.

## 6. 15-query evaluation harness (`evaluation/llm_judge.py`)

The "judge" here is a deterministic, rule-based scorer, consistent with
the rest of the project's `MOCK_LLM` approach — not a second model call.
15 queries (5 in-scope, 3 out-of-scope, 2 unknown-ticket, 2
prompt-injection, 3 more in-scope), scored on Accuracy, Grounding,
Completeness, and Safety.

With the real embeddings, this scores a perfect 1.0 across all four
metrics. Earlier, under the fallback embedder, Accuracy came in at 0.73
— several in-scope answers cited a real-but-wrong KB doc, so they were
grounded but not accurate, which is the honest result of a weak embedder
rather than a logic bug. The 1.0 now isn't a shortcut in the judge
logic; it's because the real embeddings genuinely retrieve the correct
doc on every query, so there are no wrong-citation cases left to catch.
Full report in `output/llm_judge_eval.txt`.

## 7. AutoGen review stage (`review/`)

A separate stage from CrewAI, sitting after the crew's output: two
agents in a `RoundRobinGroupChat` (`policy_reviewer` and
`verdict_agent`), backed by `ReplayChatCompletionClient` for
deterministic playback. I hit one real error here — an `AssistantAgent`
with a structured output type emits a `StructuredMessage`, and without
declaring `custom_message_types` on the team, AutoGen raises a
"Message type is not registered" error the moment that message gets
logged. Declaring it fixes it.

The two "replayed" responses aren't hardcoded text — they come from
`assess_support_response()`, a real rule-based judgment that reuses the
groundedness check, so AutoGen's actual plumbing is carrying a genuine
decision rather than a script. I verified this against a real crew
output, plus two hand-built scenarios: a grounded response gets
approved/low-risk, and a fabricated, uncited one gets blocked/high-risk.

## 8. Governance + caching (`governance/`)

Least-autonomy enforcement (`authorization.py`) is an active runtime
guard, not just true by construction: `mock_llm.py` records which role
is acting on every call, and `crew_tools.py` wraps each tool with a role
check that raises if the acting role doesn't match. I verified both the
success path and the rejection path.

Risk classification (`risk.py`) runs before the request even reaches the
crew — separate from the AutoGen stage's review after the fact — with
four rules in priority order: injection attempt → high, ticket already
past the escalation threshold → high, unknown ticket → medium, else low.

The budget cap (`budget.py`) estimates tokens with the same chars/4
heuristic CrewAI itself falls back to for a model with no native
tokenizer, since `MockLLM` makes no real call and so has no real dollar
cost to track. I demonstrated a normal query passing, the same query
getting rejected under an artificially tiny budget, and an oversized
query getting rejected under the default budget.

The response cache (`cache.py`) is keyed by `(query, record_id)`. First
call takes about 6.2 seconds (the one-time embedding model load), the
cache hit is near-instant, and I verified the responses come back
identical. There's no TTL — a known simplification, not something I'm
hiding, since ticket status does change over time in a real deployment.

## 9. Setting up the real embedding model

I did most of the build in an environment without real internet access,
so everything above was developed and tested against a deterministic
fallback embedder. Once I had the project on my own (Intel) Mac with
real internet, I re-ran the real pipeline:

```bash
pip install -r requirements.txt
rm -rf data/chroma_store
python3 -m rag.indexer                        # downloads & caches the real BGE model
python3 -m scripts.calibrate_threshold        # -> REFUSAL_THRESHOLD = 0.5781
python3 -m rag.evaluation                     # -> real precision/recall numbers (Section 3)
python3 -m scripts.generate_task4_transcript  # -> real grounded-generation transcript
python3 -m evaluation.llm_judge               # -> real 15-query eval report (Section 6)
```

That wasn't totally smooth — I hit four separate environment issues
along the way, in order:

1. `lancedb` (pulled in by `crewai`) needs the `protoc` compiler to
   build its Rust component, not just the Python `protobuf` package. I
   fixed this with `brew install protobuf`.
2. That Rust build turned out to be heavy enough to freeze my Mac at
   full compiler parallelism. I capped it with
   `export CARGO_BUILD_JOBS=2` before reinstalling.
3. PyPI doesn't ship a `torch` wheel newer than `2.2.2` for Intel macOS.
   That meant two things: `torch==2.2.2`'s compiled extensions break
   under NumPy 2.x (which `chromadb` otherwise pulls in), so I pinned
   `numpy<2`; and `transformers>=5.0` hard-requires `torch>=2.6`, which
   doesn't exist for this platform, so I downgraded to
   `sentence-transformers==3.0.1` with `transformers>=4.41,<5.0` — both
   still load `BAAI/bge-small-en-v1.5` fine.
4. The newest `cryptography` (pulled in by `crewai-cli`) failed to
   import at runtime with a missing OpenSSL symbol when built from
   source against my system's OpenSSL setup. I pinned
   `cryptography==42.0.8` to get a prebuilt wheel instead of building
   from source.

All four are reflected in `requirements.txt`. I also removed an earlier
`protobuf>=6.33.5,<8` pin I'd added to work around a separate conflict
between `autogen-core` and `chromadb` — it turned out `autogen-core`'s
own looser constraint (`protobuf~=5.29.3`) resolves fine on its own
without an override.

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
├── PROGRESS.md              (build notes / design-decision log)
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
│                              and 2 throwaway diagnostic scripts I used to
│                              verify CrewAI's internals before writing
│                              mock_llm.py)
├── data/chroma_store/        (persisted Chroma DB)
└── output/                   (generated transcripts, eval reports, request logs)
```
