# Worklog: Controlled Complex Agent (Phase 4)

Date: 2026-06-28
Role: implementer
Run: 20260628_143535_828_a376bba2

## Summary

Implemented Phase 4 of the RAG Agent project plan: a minimal Controlled Complex Agent path that adds a deterministic `plan` step only for `complex_multi_hop` queries, without replacing the current executor architecture.

## Changes

### 1. Schema: `src/app/schemas/agent.py`

- Added `ComplexPlan` Pydantic model with structured fields:
  - `plan_type` (default: `"controlled_complex"`)
  - `allowed_tools` — whitelist: `rag.retrieve_evidence`, `rag.answer`, `answer.synthesize`, `answer.verify`
  - `forbidden_tools` — explicit blocklist: `shell`, `sql`, `write`, `network`, `file_system_write`, `arbitrary_code_execution`
  - `max_steps` and `max_tool_calls` — inherited from request constraints
  - `subtasks` — planned substep list
- Updated `AgentRouteDecision` docstring to include `complex_multi_hop`
- Updated `AgentStep.step_type` docstring to include `"plan"`

### 2. Policy Router: `src/app/services/agent_policy.py`

- Added `_COMPLEX_MULTI_HOP_TERMS` for detecting multi-hop/multi-part structural patterns:
  - English: `"first find"`, `"then determine"`, `"step by step"`, `"multi-step"`, `"multi-hop"`, `"multi-part"`, `"multiple questions"`, `"several parts"`, etc.
  - Chinese: `"首先找到"`, `"然后计算"`, `"分步"`, `"多步"`, `"先确定"`, `"再计算"`, etc.
- Added `_NUMBERED_SUBQUESTION_PATTERN` regex to detect queries with 2+ numbered sub-questions (e.g. `"1. ...\n2. ..."`)
- Updated priority order: `needs_clarification` → `complex_multi_hop` → `multi_source_compare` → `table_or_metric` → `evidence_required` → `simple_rag`
- Complex queries get `max_retries=0` (plan adds structure; retry would add noise)

### 3. Agent Executor: `src/app/services/agent_executor.py`

- Imported `ComplexPlan` schema
- Added plan step logic after route step and before `needs_clarification` check:
  - Only triggers when `route.route == "complex_multi_hop"` AND `max_steps` not yet hit
  - Builds a `ComplexPlan` from request constraints
  - Emits an `AgentStep` with `step_type="plan"` and full structured metadata
  - Re-checks `max_steps_hit` after appending plan step
  - The existing execution path (retrieve → rag.answer → synthesize → verify → finalize) continues unchanged
- Updated class docstring to reflect the new plan step in the execution flow
- Step limit enforcement unchanged: plan step counts toward max_steps; if budget exhausted, flow truncates gracefully

### 4. Tests: `tests/test_agent_policy.py`

Added 6 test cases:
- `test_complex_multi_hop_terms_route_correctly` — parametrized (10 queries) covering English and Chinese multi-hop terms
- `test_numbered_subquestions_route_to_complex_multi_hop` — parametrized (2 queries) for numbered sub-questions
- `test_simple_questions_do_not_trigger_complex_multi_hop` — parametrized (6 queries) ensuring simple queries stay on simple routes
- `test_single_numbered_item_does_not_trigger_complex` — single "1." is not enough
- `test_priority_complex_over_comparison` — complex multi-hop wins over comparison
- `test_complex_multi_hop_before_table_or_metric` — complex multi-hop wins over table/metric

Updated:
- `test_route_decision_valid_routes_only` — now includes `complex_multi_hop` in valid routes

### 5. Tests: `tests/test_agent_executor.py`

Added 6 test cases:
- `test_complex_query_includes_plan_step` — verifies plan step appears, has structured metadata (plan_type, allowed_tools, forbidden_tools, subtasks), and is positioned after route but before retrieve/rag.answer
- `test_simple_query_does_not_include_plan_step` — verifies simple queries complete normally without a plan step
- `test_plan_step_metadata_has_whitelist_and_forbidden_tools` — verifies only safe read-only tools in allowed_tools; dangerous tools (shell/sql/write/network) in forbidden_tools
- `test_complex_plan_step_with_tight_max_steps` — max_steps=2 respects limit (route + plan, truncated)
- `test_complex_plan_step_with_max_steps_3` — max_steps=3 fits route + plan + 1 more step
- `test_complex_plan_with_tight_max_tool_calls` — max_tool_calls=1 still includes plan step (plan is not a tool call)

Added helper: `make_complex_query_executor()` — builds an executor with RAGAdapter that supports both `answer()` and `retrieve_evidence()` for complex query tests.

## Verification

All tests pass:

```
D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_agent_executor.py tests/test_agent_streaming.py -q
111 passed in 25.68s

D:\Miniconda3\python.exe -m pytest tests/test_agent_routes.py tests/test_agent_trace_store.py tests/test_conversation_memory.py -q
33 passed in 33.61s
```

## Acceptance Criteria Check

1. ✅ `PolicyRouter` can route clearly complex/multi-part questions to `complex_multi_hop` — 12 parametrized test cases prove detection
2. ✅ Simple questions do not trigger `complex_multi_hop` and do not receive a plan step — 6 parametrized non-trigger tests + `test_simple_query_does_not_include_plan_step`
3. ✅ Complex queries include a `plan` step before retrieval/synthesis, with bounded structured metadata — `test_complex_query_includes_plan_step`
4. ✅ The plan step exposes only safe read-only whitelist tools and explicitly excludes shell/SQL/write/network — `test_plan_step_metadata_has_whitelist_and_forbidden_tools`
5. ✅ Existing RAG/evidence/synthesis/verify path remains compatible and still completes for complex queries — all 111 tests pass including evidence pack and synthesis tests
6. ✅ `max_steps` and `max_tool_calls` constraints remain strictly respected — `test_complex_plan_step_with_tight_max_steps`, `test_complex_plan_step_with_max_steps_3`, `test_complex_plan_with_tight_max_tool_calls`
7. ✅ Tests cover all required scenarios
8. ✅ Worklog written

## No Changes To

- RAG retrieval, QueryService, ingestion, benchmark definitions, loop runner
- Environment files, quality dashboard code
- ToolRegistry (whitelist tools remain unchanged)
- Existing tests were never loosened or removed
- No new dependencies added
- No LangGraph, LlamaIndex, RAGFlow, or other orchestration framework
- No model-controlled shell, SQL, writes, or network calls

## Residual Notes

- The `complex_multi_hop` route currently uses deterministic keyword matching. Future enhancement: optionally use a lightweight classifier for more nuanced detection of borderline multi-hop questions.
- The plan step is entirely deterministic (no LLM involved in planning). If the project later adds LLM-driven planning, it should be behind a separate feature flag and approval gate.
- Trace persistence and SSE streaming naturally expose the new `plan` step because they already stream/persist all steps — no changes needed in trace store or streaming code.
