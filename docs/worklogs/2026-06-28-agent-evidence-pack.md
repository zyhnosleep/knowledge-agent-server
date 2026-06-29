# Agent Evidence Pack — 2026-06-28

## What Changed

### Evidence Schemas (`src/app/schemas/agent.py`)
- Added `EvidenceItem` model with fields: `index`, `document_id`, `chunk_id`, `page_slug`, `page_title`, `page_kind`, `page_label`, `score`, `excerpt`, `evidence_kind`, `source_stage`, `support_hint`
- Added `EvidencePack` model with `status` and `items` list
- Both models are JSON-safe and exclude secrets / raw provider responses

### Retrieve-Only RAG (`src/app/services/search.py`)
- Added `QueryService.retrieve_evidence(project_slug, question, limit)` that reuses existing RAG routing/context selection but does NOT draft an answer, verify, or persist `QuestionAnswer`
- Added `_determine_source_stage()` and `_determine_support_hint()` static/class methods for deterministic evidence classification
- Source stages: `document_table`, `document_figure`, `profile_term`, `claim`, `source_chunk`, `wiki_page`, `unknown`

### RAGAdapter (`src/app/services/rag_adapter.py`)
- Added `RAGAdapter.retrieve_evidence(db, project_slug, question, limit)` method
- For missing projects, returns a valid empty `EvidencePack` with `status="project_not_found"` instead of crashing

### Tool Registry (`src/app/services/tool_registry.py`)
- Registered read-only `rag.retrieve_evidence` built-in tool (input: `project_slug`, `question`; optional `limit`; output: `status`, `items`)
- Existing `rag.answer`, `answer.synthesize`, `answer.verify` remain backward-compatible
- Registration is guarded by `hasattr(rag_adapter, "retrieve_evidence")` for backward compatibility

### Agent Executor (`src/app/services/agent_executor.py`)
- Normal flow is now: `route` → `rag.retrieve_evidence` → `rag.answer` → `answer.synthesize` → `answer.verify` → optional retry → finalize
- Retrieve step uses `step_type="retrieve"` and `tool_name="rag.retrieve_evidence"`
- Retrieve step metadata includes: `evidence_count`, `table_evidence_count`, `evidence_kinds`, `source_stages`, `support_hints`
- `needs_clarification` still performs no RAG or retrieve tool call
- Retrieve step is skipped when tool is not registered (backward compatibility)
- Step/tool limits are respected

### Agent Synthesizer (`src/app/services/agent_synthesizer.py`)
- `synthesize()` now accepts optional `evidence_pack` parameter
- Local fallback is identical with or without evidence_pack
- External API path includes evidence pack excerpts in the prompt when available

## Tests Passed

All targeted test suites pass:

```
tests/test_rag_adapter.py — 9 passed (2 original + 4 schema + 3 retrieve_evidence)
tests/test_tool_registry.py — 21 passed (15 original + 6 retrieve_evidence)
tests/test_agent_executor.py — 37 passed (32 original + 5 retrieve step)
tests/test_agent_streaming.py — 5 passed
tests/test_agent_synthesizer.py — 11 passed (7 original + 4 evidence_pack)
tests/test_query_service.py (--table/evidence/rag_context filter) — passed
tests/test_agent_routes.py — passed
tests/test_agent_trace_store.py — passed
tests/test_conversation_memory.py — passed
```

## What Remains

- Complex Agent planning / eval dashboard
- Evidence pack consumption in synthesis (currently trace-visible but local fallback ignores it)
- LangGraph or other agent framework integration (explicitly out of scope)
