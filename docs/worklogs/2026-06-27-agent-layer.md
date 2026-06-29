# 2026-06-27 Agent Layer v1

## What was built

A minimal, production-shaped Agent orchestration layer above the existing FastAPI/RAG service. The Agent layer treats RAG as an independent evidence provider and adds Conversation Memory, Tool Registry, Agent Executor, a new `/api/agent/query` endpoint, and a static frontend Agent panel.

## Architecture

```
Static HTML (new Agent panel)
        |
POST /api/agent/query  (src/app/api/agent_routes.py)
        |
AgentExecutor (src/app/services/agent_executor.py)
   |        |         |
RAGAdapter  ToolRegistry  ConversationMemory
(src/app/services/rag_adapter.py)
(src/app/services/tool_registry.py)
(src/app/services/conversation_memory.py)
   |
QueryService (existing, unchanged)
```

## New files (12)

| File | Purpose |
|---|---|
| `src/app/schemas/agent.py` | Agent Pydantic schemas (AgentQueryRequest, AgentQueryResponse, AgentStep, AgentUsage, ToolSpec, AgentConstraints) |
| `src/app/services/conversation_memory.py` | Per-session turn storage backed by ConversationTurn table |
| `src/app/services/rag_adapter.py` | Thin wrapper around QueryService, returns QueryResponse |
| `src/app/services/tool_registry.py` | Pluggable tool registry with schema validation and built-in `rag.answer` tool |
| `src/app/services/agent_executor.py` | Two-step deterministic executor: tool_call -> finalize |
| `src/app/api/agent_routes.py` | `/api/agent/query` endpoint with AGENT_ENABLED guard |
| `tests/test_conversation_memory.py` | 7 tests covering CRUD, compaction, isolation |
| `tests/test_rag_adapter.py` | 2 tests covering normal and missing-project paths |
| `tests/test_tool_registry.py` | 10 tests covering registration, validation, error handling |
| `tests/test_agent_executor.py` | 9 tests covering execution, sessions, usage, persistence |
| `tests/test_agent_routes.py` | 5 tests covering 503 guard, 200 response, 422 validation, shape |
| `docs/worklogs/2026-06-27-agent-layer.md` | This file |

## Changed files (3)

| File | Change |
|---|---|
| `src/app/core/config.py` | Added 7 AGENT_ settings (AGENT_ENABLED, AGENT_MAX_STEPS, etc.) |
| `src/app/models/records.py` | Added ConversationTurn table |
| `src/app/main.py` | Mounted agent_router at `/api/agent` |
| `src/app/static/index.html` | Added Agent panel card with JS logic |

## Design decisions

1. **RAGAdapter.answer()** wraps `QueryService.answer(..., save_answer=False)` — Agent never touches QueryService internals directly.
2. **ConversationTurn** uses its own `created_at` column (not TimestampMixin) for simplicity.
3. **ToolRegistry.call_tool()** catches all exceptions (including ToolError) and returns structured `{ok, name, error, error_type}` dicts.
4. **AgentExecutor** is deterministic v1: always calls `rag.answer` as the only tool, then finalizes. Ready for future model-driven tool selection.
5. **Agent routes** use the same `Depends(get_db)` pattern as the main routes.
6. **No external Agent frameworks** (LangGraph, Dify, etc.) were imported.
7. **AGENT_ENABLED defaults to true** (v1 is read-only and safe) but the route still honors `AGENT_ENABLED=false` with a 503.

## Verification

```
D:\Miniconda3\python.exe -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_tool_registry.py tests/test_agent_executor.py tests/test_agent_routes.py -q
→ 33 passed

D:\Miniconda3\python.exe -m pytest tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q
→ 199 passed (zero regressions)
```

## Reviewer Rework (2026-06-27, run 20260627_033841_962_586adf0c)

Addressed accepted concerns from spec reviewer (`reviewer-spec-001`) and quality reviewer (`reviewer-quality-001`):

| Concern | Fix | Files |
|---|---|---|
| XSS: `innerHTML` rendering of `final_answer` | Replaced `innerHTML` with DOM construction + `textContent` for all dynamic API data | `src/app/static/index.html` |
| Conversation turns not committed | Route now calls `db.commit()` after executor; rollback on failure → 500 | `src/app/api/agent_routes.py` |
| Unused imports: `asyncio`, `json` in tool_registry | Removed | `src/app/services/tool_registry.py` |
| Misleading "enforces timeouts" docstring | Updated docstring to clarify timeout enforcement is caller responsibility | `src/app/services/tool_registry.py` |
| Unused `_new_request_id`, `_new_session_id`, `datetime` import in schemas | Removed | `src/app/schemas/agent.py` |
| No test for executor error path | Added `test_execute_error_path_returns_error_status` (mocks tool failure) | `tests/test_agent_executor.py` |
| No test for executor timeout path | Added `test_execute_timeout_path_returns_timeout_status` (mocked monotonic time) | `tests/test_agent_executor.py` |
| No test for cross-session persistence | Added `test_agent_turns_persist_across_sessions` (file-based SQLite, two DB sessions) | `tests/test_agent_routes.py` |
| Missing AGENT_ vars in env examples | Added 7 AGENT_ settings to both files | `.env.example`, `.env.server.example` |

### Verification

```
D:\Miniconda3\python.exe -m pytest tests/test_agent_executor.py tests/test_conversation_memory.py tests/test_tool_registry.py tests/test_rag_adapter.py tests/test_agent_routes.py -q
→ 36 passed (was 33)

D:\Miniconda3\python.exe -m pytest tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q
→ 199 passed (zero regressions)
```

## Future work

- Model-driven tool selection (multi-step loop with Ollama function calling)
- Policy Router (query complexity classification, budget routing)
- SSE streaming for step-by-step progress
- More tools beyond the read-only `rag.answer`
- Structured eval pipeline (Ragas integration, trace persistence)
- Conversation TTL and cleanup

---

## Agent v2a (2026-06-27, run 20260627_101125_359_ecc36e65)

### What was built

Production-hardening of the Agent layer: deterministic policy routing, answer verification, bounded retry, richer trace data, and frontend visibility.

### New files (4)

| File | Purpose |
|---|---|
| `src/app/services/agent_policy.py` | Deterministic `PolicyRouter` — keyword-based query classification into 5 route types |
| `src/app/services/answer_verifier.py` | `AnswerVerifier` — answer quality checker, registered as tool `answer.verify` |
| `tests/test_agent_policy.py` | 9 tests (8 parametrized cases + priority test) |
| `tests/test_answer_verifier.py` | 13 tests covering empty/citation/table evidence warnings |

### Changed files (7)

| File | Change |
|---|---|
| `src/app/schemas/agent.py` | Added `AgentRouteDecision` schema; added `metadata: dict` to `AgentStep`; added `route` and `warnings` to `AgentQueryResponse` |
| `src/app/services/agent_executor.py` | v2a deterministic loop: route → rag.answer → answer.verify → [retry] → finalize; respects `max_tool_calls`/`max_steps` limits |
| `src/app/services/tool_registry.py` | `_register_builtins` registers both `rag.answer` and `answer.verify`; `rag.answer` citations now include `page_kind` and `page_label` |
| `src/app/static/index.html` | Agent panel shows route type, warnings, verification/retry step details via `textContent`/`createTextNode` |
| `tests/test_agent_executor.py` | +7 v2a tests (route, verify, retry, limits) + updated 3 existing tests |
| `tests/test_tool_registry.py` | +4 v2a tests (answer.verify tool, page_kind/page_label, read-only) + updated 1 existing test |
| `tests/test_agent_routes.py` | +2 v2a tests (route/warnings fields, needs_clarification route) + updated 1 existing test |

### Design decisions

1. **PolicyRouter** is fully deterministic (keyword-based, no LLM). Priority order: needs_clarification > multi_source_compare > table_or_metric > evidence_required > simple_rag.
2. **AnswerVerifier** is a read-only tool (`answer.verify`) that checks answer quality against route requirements. It never fails — it always returns `ok=True` with warnings and a retry recommendation.
3. **Retry is bounded**: at most one additional `rag.answer` call, only when the verifier recommends it AND the route allows it AND constraints permit it.
4. **Route step** is always step_id=0, providing traceability for every query.
5. **needs_clarification** route skips all RAG calls and returns immediately with a useful warning.
6. **Backward compatibility**: `/api/agent/query` request body unchanged; response adds `route` (nullable) and `warnings` (default empty list).
7. **No new external dependencies** were added.

### Verification

```
# Agent v2a tests (GREEN)
D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q
→ 92 passed

# Full regression (GREEN)
D:\Miniconda3\python.exe -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q
→ 208 passed (zero regressions)

# Frontend security
rg -n "innerHTML\s*=.*(agent|step|answer|summary|error|data|result|payload|response|route|warning)|insertAdjacentHTML|eval\(" src/app/static/index.html
→ No hits in Agent panel (only pre-existing loadDocuments/loadReviews)
```

---

## Agent v2a final acceptance (2026-06-27)

### codex-with-cc workflow

- Workflow: `agent-v2a-20260627`
- Implementer/rework runs:
  - `20260627_101125_359_ecc36e65` (`agent-v2a-implementer-001`)
  - `20260627_103247_039_35a23ec5` (`agent-v2a-rework-001`)
  - `20260627_103822_367_bf8684ec` (`agent-v2a-rework-002`)
- Review/final verifier runs:
  - `20260627_105301_102_1b0609db` base spec review
  - `20260627_105655_009_ec17e19f` base quality review
  - `20260627_105508_667_c80df480` rework-001 spec review
  - `20260627_105923_520_5d57a752` rework-001 quality review
  - `20260627_104430_829_1afc6a8c` latest spec review
  - `20260627_104700_439_fe1699cc` latest quality review
  - `20260627_110317_743_0c7d810e` final verifier
- Final workflow verifier:

```text
D:\Miniconda3\python.exe ...\verify_delegate_workflow.py -WorkflowId agent-v2a-20260627
-> Workflow verification passed
```

### Local final verification

```text
D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q
-> 109 passed

D:\Miniconda3\python.exe -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q
-> 208 passed
```

Frontend Agent XSS scan:

```text
rg -n 'innerHTML\s*=.*(agent|step|answer|summary|error|data|result|payload|response|route|warning)|insertAdjacentHTML|eval\(' src\app\static\index.html
-> only pre-existing loadDocuments/loadReviews matches; Agent panel dynamic values use textContent/createTextNode
```

Local API smoke on temporary port 8001:

- `GET /api/health` -> 200
- `POST /api/agent/query` with whitespace query -> 200, route `needs_clarification`, `tool_calls=0`
- invalid constraints `max_steps=0` -> 422
- Browser UI smoke on `http://127.0.0.1:8001/` -> Agent panel submitted whitespace query and rendered `completed`, route `needs_clarification`, warning, steps, answer, and usage; browser console errors: none.

### Remote final verification

Synced only Agent v2a files and tests to `~/llm_wiki_server` after backing up overwritten files under `tmp/backup_agent_v2a_20260627_111038`.

Remote tests:

```text
.venv/bin/python -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q
-> 109 passed

.venv/bin/python -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q
-> 207 passed
```

Remote API/worker restart and smoke:

- API and worker restarted via pid files and repository scripts.
- `GET /api/health` -> 200.
- `POST /api/agent/query` with whitespace query -> 200, route `needs_clarification`, `tool_calls=0`.
- invalid constraints `max_steps=0` -> 422.
- real RAG-backed Agent smoke `What is CHARMM36m?` -> 200 with `status=completed`, RAG evidence text, citations, route metadata, and steps.

Remote full30 query gate:

```text
.venv/bin/python scripts/run_mineru_rag_loop.py --profile query --python .venv/bin/python --run-query-eval --base-url http://127.0.0.1:8000 --timeout 120 --min-query-passed 24 --max-query-failed 6 --require-failure-attribution --out-dir tmp/full30_query_gate_agent_v2a_20260627
-> overall_status=passed
-> 25/30 passed
-> 5/30 failed, all attributed to answer-stage missing expected answer text
```

### Final decision

Agent v2a is accepted as online-ready for the current local-first scope: deterministic routing, read-only RAG tool use, answer verification, strict execution bounds, safe frontend rendering, workflow-reviewed implementation evidence, and remote service verification all passed. Known residual risks are non-blocking and documented: retry is not re-verified under default `max_tool_calls=3`, keyword routing is intentionally simple, and a few minor cleanup items remain in executor naming/imports.

### Known residual risks

1. **Retry without re-verification**: the retry path calls `rag.answer` again but does not call `answer.verify` a second time (to stay within default `max_tool_calls=3`). A bad retry answer is not verified.
2. **Keyword matching is best-effort**: Chinese/English keyword lists may miss edge cases. Future work could add embedding-based routing.
3. **`answer.verify` schema validation**: `citations` input is typed as `array` with no item schema — any list passes validation, even if items are malformed.
4. **Timeout test fragility**: the mocked `monotonic` chain length depends on exact call count in the executor loop. Adding new steps requires updating the chain.
5. **Table evidence detection is simple**: numeric regex + page_kind check may miss valid table evidence in structured markdown tables without explicit numeric patterns.

---

## Agent v2a Rework — strict max_steps enforcement (2026-06-27, run 20260627_103247_039_35a23ec5)

### What was fixed

The v2a executor could previously exceed `constraints.max_steps` because helpers `_run_rag_answer` and `_run_verify` appended steps without checking the step budget, and the `finalize` step was always appended unconditionally. The fix makes `max_steps` a strict trace boundary.

### Changed files (3)

| File | Change |
|---|---|
| `src/app/services/agent_executor.py` | `_run_rag_answer` and `_run_verify` now check `len(steps) >= constraints.max_steps` before appending; execute() sets `status="max_steps"` when the budget is exhausted; `_finalize_truncated` helper is used when budget exhausted before any tool call; `_run_verify` no longer increments `usage.tool_calls` when producing a no-op skip result |
| `tests/test_agent_executor.py` | Replaced weak `test_max_steps_limit_respected` with 5 strict tests: `max_steps=2`, `max_steps=1`, exhausted status signal, `usage.steps` matches actual steps, needs_clarification respects max_steps |
| `docs/worklogs/2026-06-27-agent-layer.md` | This section |

### Design approach

- `max_steps` is checked **before** each step append in `_run_rag_answer` and `_run_verify`
- Route step is always appended (always step 0); if already at limit after route, `_finalize_truncated` returns with `status="max_steps"`
- `finalize` step is gated by `len(steps) < constraints.max_steps`
- `needs_clarification` finalize step is also gated; if at limit, a truncated warning is added instead
- `usage.tool_calls` now correctly reflects only actual tool calls made (not skipped no-ops)
- `status="max_steps"` is set whenever the step budget prevents normal completion

### Verification

```
# Agent executor tests
D:\Miniconda3\python.exe -m pytest tests/test_agent_executor.py -q
→ 24 passed

# Full agent v2a tests
D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q
→ 97 passed (was 92)

# Regression tests
D:\Miniconda3\python.exe -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q
→ 208 passed (zero regressions)
```

### Known residual risks

- `max_steps=1` with a normal query produces only a route step with no RAG answer — this is correct per the spec but may surprise callers
- The `_finalize_truncated` path always returns `final_answer=""` — a future iteration could attempt a partial answer from the route metadata

---

## Agent v2a Rework — constraint validation (2026-06-27, run 20260627_103822_367_bf8684ec)

### What was fixed

`AgentConstraints` previously accepted zero, negative, and otherwise invalid values for `max_steps`, `max_tool_calls`, `budget_tokens`, and `timeout_seconds`. Because the executor always needs ≥1 step, accepting non-positive budgets violates the online-readiness invariant. Added Pydantic `Field(gt=0)` validators to all four numeric fields.

### Changed files (4)

| File | Change |
|---|---|
| `src/app/schemas/agent.py` | Added `Field(gt=0)` to `max_steps`, `max_tool_calls`, `budget_tokens`, `timeout_seconds` in `AgentConstraints` |
| `tests/test_agent_executor.py` | Added `import pytest`; added 8 schema-level tests (zero/negative rejection, defaults unchanged, positive=1 accepted) |
| `tests/test_agent_routes.py` | Added 4 API-level 422 tests (max_steps=0, max_tool_calls=0, timeout_seconds=0, negative max_steps) |
| `docs/worklogs/2026-06-27-agent-layer.md` | This section |

### Design approach

- Pydantic `Field(gt=0)` is the minimal change — it is declarative, enforced at model construction, and integrates automatically with FastAPI (422 response).
- No custom validators, no runtime checks in `AgentExecutor`, no new dependencies.
- Existing defaults (8, 3, 20000, 45) are all > 0 so the change is backward-compatible for all callers using defaults.
- Existing tests that pass `constraints={...}` as dicts continue to work because they use valid positive values.

### Verification

```
# Executor + routes tests
D:\Miniconda3\python.exe -m pytest tests/test_agent_executor.py tests/test_agent_routes.py -q
→ 43 passed (was 28)

# Full agent v2a tests
D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q
→ 109 passed (was 97)

# Regression tests
D:\Miniconda3\python.exe -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q
→ 208 passed (zero regressions)
```

### Known residual risks

- `Field(gt=0)` on `int` rejects `0` but accepts `1` — there is no minimum threshold beyond "positive". Very small budgets like `max_steps=1` are accepted by the model but may produce truncated results (the executor already handles this via `status="max_steps"`).
- The fix is entirely at the Pydantic layer — if constraints are ever constructed outside Pydantic (e.g., raw dict passthrough), validation is bypassed. No such path exists in the current codebase.

---

## Quality Dashboard (2026-06-27)

### What was built

A read-only quality dashboard (`GET /api/quality/dashboard?limit=5`) that surfaces recent loop run manifests from `QUALITY_REPORTS_DIR` (default `./tmp`). The service scans for `manifest.json` files produced by `scripts/run_mineru_rag_loop.py`, reads optional same-directory `query_attribution.json` and `query_eval.json`, and returns compact summaries with command statuses, query gate results, MinerU/service ingest smoke summaries, and failed case attribution. The frontend was redesigned with a Swiss Pulse operational dashboard look.

This route is strictly read-only: it never starts loop scripts, long-running work, or file uploads.

**Rework (2026-06-27)**: corrected `skipped_reports` semantics — the field now counts only malformed/unreadable manifests, not manifests truncated by the `limit` parameter. `collect_runs` returns a structured result with `runs`, `total_manifests`, `malformed_count`, and `valid_total`. The route no longer calls private service methods. Same-directory `query_eval.json` is used as a fallback when the manifest lacks `query_eval.summary`. Dead code (`_read_json_list`) removed.

### New files (6)

| File | Purpose |
|---|---|
| `src/app/schemas/quality.py` | Pydantic schemas (QualityDashboardResponse, RunSummary, FailedCaseSummary) |
| `src/app/services/quality_reports.py` | Read-only manifest scanner, sorts by mtime, skips malformed JSON, never trusts embedded paths |
| `src/app/api/quality_routes.py` | `GET /api/quality/dashboard` with `limit` validation (1-20) |
| `tests/test_quality_reports.py` | 17 tests covering scanning, sorting, summary fields, path safety, attribution sideload |
| `tests/test_quality_routes.py` | 8 tests covering 200/422, schema shape, read-only safety |
| (frontend redesign) | `src/app/static/index.html` — full Swiss Pulse redesign with quality panel |

### Changed files (4)

| File | Change |
|---|---|
| `src/app/core/config.py` | Added `QUALITY_REPORTS_DIR` defaulting to `Path("./tmp")` |
| `src/app/main.py` | Mounted `quality_router` at `/api` |
| `src/app/static/index.html` | Full Swiss Pulse console redesign with quality dashboard panel, DOM/textContent-only rendering |
| `docs/work.md` | Added quality dashboard entry |

### Design decisions

1. **Read-only**: only reads files under `QUALITY_REPORTS_DIR`. Never executes loop scripts.
2. **Path safety**: never trusts `out_dir`, `report_path`, or other embedded paths from manifest contents.
3. **Malformed manifest tolerance**: invalid JSON or non-object manifests are silently skipped, counted in `skipped_reports`.
4. **Failed case cap**: at most 10 failed cases per run to prevent oversized responses.
5. **Frontend safety**: quality panel uses `textContent` and DOM nodes exclusively — no `innerHTML` for quality data.

### Verification

```
# Quality tests
D:\Miniconda3\python.exe -m pytest tests/test_quality_reports.py tests/test_quality_routes.py -q
→ 25 passed

# Existing tests (zero regressions)
D:\Miniconda3\python.exe -m pytest tests/test_loop_runner.py tests/test_query_report_summary.py tests/test_agent_routes.py tests/test_api_routes.py -q
→ 44 passed

# Frontend security
rg -n 'innerHTML\s*=.*(quality|manifest|gate|attribution|case|failure|status|report)|insertAdjacentHTML|eval\(' src\app\static\index.html
→ No hits
```

---

## Agent v3 Enterprise (2026-06-27, run 20260627_213837_412_dfdc34fc)

### What was built

Enterprise hardening of the Agent layer: external-API evidence synthesis, step-level SSE streaming, database-backed trace persistence, and 30-day conversation TTL. Existing synchronous `POST /api/agent/query` behavior is preserved.

### New files (5)

| File | Purpose |
|---|---|
| `src/app/services/agent_synthesizer.py` | `AgentSynthesizer` — external API or local deterministic synthesis from RAG evidence |
| `src/app/services/agent_trace_store.py` | `AgentTraceStore` — persist and retrieve agent runs and steps |
| `tests/test_agent_synthesizer.py` | 6 tests covering auto/local/external paths, fallback, required fields, citation sanitization |
| `tests/test_agent_trace_store.py` | 10 tests covering CRUD, session filtering, pagination, ordering, secret safety |
| `tests/test_agent_streaming.py` | 6 tests covering streaming endpoint, event names, error paths, trace_id in final event |

### Changed files (14)

| File | Change |
|---|---|
| `src/app/core/config.py` | Added `AGENT_SYNTHESIS_PROVIDER` (auto), `AGENT_CONVERSATION_TTL_DAYS` (30), `AGENT_TRACE_RETENTION_DAYS` (30), `AGENT_STREAM_HEARTBEAT_SECONDS` (15); increased `AGENT_MAX_TOOL_CALLS` default from 3 to 5 |
| `src/app/models/records.py` | Added `ConversationSession`, `AgentTraceRun`, `AgentTraceStep` tables |
| `src/app/schemas/agent.py` | Added `trace_id`, `answer_provider`, `answer_model` to `AgentQueryResponse`; updated `AgentConstraints.max_tool_calls` default to 5; added "synthesis" to `AgentStep.step_type` doc |
| `src/app/services/agent_executor.py` | v3 flow: route → rag.answer → answer.synthesize → answer.verify → [retry] → finalize; session TTL touch; trace persistence; returns `trace_id`, `answer_provider`, `answer_model` |
| `src/app/services/tool_registry.py` | Registered `answer.synthesize` read-only tool; added `_answer_synthesize_handler` |
| `src/app/services/conversation_memory.py` | Added `touch_session()` and `purge_expired_sessions()` using `ConversationSession` table |
| `src/app/api/agent_routes.py` | Added `POST /api/agent/query/stream` (SSE), `GET /api/agent/traces`, `GET /api/agent/traces/{trace_id}`; extracted `_build_executor()` helper with trace store + synthesizer |
| `src/app/static/index.html` | Agent panel updated to prefer streaming via `/api/agent/query/stream` with live step events; graceful fallback to sync endpoint; shows trace_id, provider/model, synthesis metadata; all dynamic content uses `textContent`/DOM construction (no `innerHTML`) |
| `.env.example` | Added v3 Agent settings; increased `AGENT_MAX_TOOL_CALLS` to 5 |
| `.env.server.example` | Added v3 Agent settings; increased `AGENT_MAX_TOOL_CALLS` to 5 |
| `docs/work.md` | Updated Agent layer description to v3 |
| `docs/worklogs/2026-06-27-agent-layer.md` | This section |
| `tests/test_agent_executor.py` | Updated existing tests for v3 flow (synthesis step in step count) |
| `tests/test_tool_registry.py` | Updated to expect 3 built-in tools (rag.answer + answer.synthesize + answer.verify) |

### Design decisions

1. **Synthesis provider** (`auto`, `external_api`, `local`): `auto` uses external API only when `EXTERNAL_API_ENABLED=true` and an API key is configured; `local` is a deterministic passthrough.
2. **External synthesis failure**: never crashes the Agent route — returns fallback with a warning.
3. **Citation sanitization**: `cited_indexes` are filtered to valid range before use.
4. **Trace persistence**: every Agent call is persisted (completed, max_steps, timeout, error). API keys and raw provider responses are NOT stored.
5. **SSE streaming**: step-level (not token-by-token) with events `start`, `step`, `warning`, `final`, `error`, `done`.
6. **Conversation TTL**: each query creates/touches a `ConversationSession` with `expires_at = now + 30 days`. Expired sessions are purged before each query.
7. **Backward compatibility**: `POST /api/agent/query` response shape is unchanged — new fields (`trace_id`, `answer_provider`, `answer_model`) are additive and default to safe values.
8. **No new dependencies**: same OpenAI-compatible `httpx` pattern used for external synthesis.
9. **Frontend safety**: all Agent dynamic data rendered via `textContent`/DOM construction — no `innerHTML` for Agent/trace/stream/step data.

### Verification

```powershell
# Agent v3 tests (GREEN)
D:\Miniconda3\python.exe -m pytest tests/test_agent_synthesizer.py tests/test_agent_trace_store.py tests/test_agent_streaming.py tests/test_agent_executor.py tests/test_agent_routes.py tests/test_conversation_memory.py tests/test_tool_registry.py -q
→ see report

# Regression tests (GREEN)
D:\Miniconda3\python.exe -m pytest tests/test_rag_adapter.py tests/test_api_routes.py tests/test_quality_routes.py tests/test_quality_reports.py -q
→ see report

# Frontend security scan
rg -n 'innerHTML\s*=.*(agent|trace|stream|step|answer|summary|error|data|result|payload|response|route|warning)|insertAdjacentHTML|eval\(' src\app\static\index.html
→ see report
```

---

## Agent v3 Rework — TTL purge fix and startup cleanup (2026-06-27, run 20260627_215804_712_222a31be)

### What was fixed

Addressed accepted quality reviewer (`reviewer-quality-001`) concerns:

| Concern | Fix | Files |
|---|---|---|
| SQLite `IN :ids` syntax error in `purge_expired_sessions` | Replaced batch `DELETE WHERE id IN :ids` with individual deletes per session for SQLite portability | `src/app/services/conversation_memory.py` |
| No dedicated TTL tests | Added 7 tests: `touch_session` create/update/duplicate, `purge_expired_sessions` deletes expired / preserves active / handles none / batch | `tests/test_conversation_memory.py` |
| No v3 field assertions in executor tests | Updated `make_executor` to include `AgentTraceStore`; added `trace_id`, `answer_provider`, `answer_model` assertions to `test_execute_returns_completed_status` | `tests/test_agent_executor.py` |
| Dead code in `agent_trace_store.py` | Removed unused `cutoff = text(...)` variable (lines 123-126) | `src/app/services/agent_trace_store.py` |
| Unused `import asyncio` | Removed | `src/app/api/agent_routes.py` |
| No startup purge | Added `_startup_purge()` in `lifespan()` — purges expired conversation sessions and old agent traces; failures logged, do not prevent startup | `src/app/main.py` |

