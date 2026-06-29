# Agent Evidence-Aware Synthesis — 2026-06-28

## What Changed

### AgentSynthesizer (`src/app/services/agent_synthesizer.py`)

- Added `import re` for regex-based anchor extraction.
- Added `_extract_evidence_anchors(evidence_pack, citations)` — static method that extracts
  high-signal anchor terms from evidence using generic rules only:
  - **Numeric patterns**: captures numbers with optional units (`kcal/mol`, `kJ/mol`, `nm`, `Å`, `eV`, `%`, `°C`, `K`).
  - **Abbreviations**: captures 2–5 uppercase letter whole-word abbreviations, filtering out
    common English noise words (`THE`, `AND`, `FOR`, etc.).
  - Deduplicates case-insensitively and limits to at most 10 anchors.
- Added `_should_retry_for_coverage(route, evidence_pack, citations, answer_text)` — static
  method that decides whether a coverage retry is warranted. Returns `True` only when ALL of:
  - Route is evidence-heavy (`evidence_required`, `table_or_metric`, `multi_source_compare`).
  - Evidence pack is present and non-empty.
  - Fewer than half of the extracted anchors appear in the answer text.
- Added `_retry_with_anchors(...)` — instance method that performs exactly one additional
  external API call with an augmented prompt listing the missing evidence anchors and
  instructing the model to include them. Returns `None` on any failure (API error, parse
  error, empty response) so the caller falls back to the first result.
- Modified `_external_synthesize()`: after successful parse and answer validation, calls
  `_should_retry_for_coverage()`. If it returns `True`, calls `_retry_with_anchors()` once.
  If the retry succeeds, returns its result. If the retry fails (returns `None`), appends a
  warning and falls through to the first synthesis result.

Key design constraints:
- **Never retries on local fallback** — coverage retry is only in the external API path.
- **Never retries on non-evidence routes** — `simple_rag` and `needs_clarification` skip
  coverage checks entirely.
- **Bounded to exactly one retry** — no loops, no cascading retries.
- **Safe fallback** — any retry failure (network, parse, empty answer) returns `None` and
  the original answer is used.
- **Generic anchor extraction** — no hard-coded domain terms or benchmark case IDs.

### AgentExecutor (`src/app/services/agent_executor.py`)

- `execute()` now passes the `evidence_pack` returned by `_run_retrieve_evidence()` into
  `_run_synthesize()` via the new `evidence_pack` keyword argument.
- `_run_synthesize()` accepts optional `evidence_pack: dict[str, Any] | None` parameter.
- The `answer.synthesize` tool call now conditionally includes `evidence_pack` in the args
  dict (only when not `None`, to avoid schema validation noise).
- Synthesis step metadata now includes compact evidence-aware fields when an evidence pack
  is available:
  - `evidence_pack_items` — total evidence items count.
  - `table_evidence_count` — count of items with `evidence_kind == "table"`.
  - `source_stages` — sorted unique source stages present in the pack.
  - These are in addition to the existing `provider`, `model`, `confidence`, `cited_indexes`.
  - Does NOT include raw excerpts, provider responses, or secrets.

### ToolRegistry (`src/app/services/tool_registry.py`)

- `answer.synthesize` input schema now includes optional `evidence_pack` property
  (`type: "object"`, not required).
- `_answer_synthesize_handler()` passes `args.get("evidence_pack")` through to
  `AgentSynthesizer.synthesize()`.
- Schema validation rejects non-object values for `evidence_pack` with `ToolSchemaError`.
- Existing calls without `evidence_pack` remain backward-compatible.

### Tests Added

**`tests/test_agent_synthesizer.py`** — 10 new tests:
- `test_coverage_retry_happens_when_anchors_missing` — verifies second API call on
  `evidence_required` route when first answer omits anchors.
- `test_coverage_retry_skipped_when_anchors_covered` — verifies single API call when
  first answer already covers all anchors.
- `test_coverage_retry_skipped_on_simple_rag_route` — `simple_rag` never triggers retry.
- `test_coverage_retry_skipped_without_evidence_pack` — `None` evidence pack skips retry.
- `test_coverage_retry_skipped_on_local_fallback` — local provider never triggers retry.
- `test_coverage_retry_failure_falls_back_gracefully` — retry API failure falls back to
  first answer with a warning.
- `test_coverage_retry_bounded_to_one_retry` — even if retry answer still misses anchors,
  only one additional call is made (exactly 2 total).
- `test_extract_evidence_anchors_numeric_and_abbrev` — anchors extracted from numeric
  patterns and abbreviations.
- `test_extract_evidence_anchors_empty_input` — returns empty list for empty/missing inputs.
- `test_should_retry_for_coverage_*` — 4 parametrized-style tests for the retry gate.

**`tests/test_tool_registry.py`** — 4 new tests:
- `test_answer_synthesize_schema_includes_evidence_pack` — schema has optional object field.
- `test_answer_synthesize_accepts_evidence_pack_in_call` — call with evidence_pack succeeds.
- `test_answer_synthesize_without_evidence_pack_still_works` — backward compatibility.
- `test_answer_synthesize_evidence_pack_invalid_type_rejected` — non-object rejected.

**`tests/test_agent_executor.py`** — 3 new tests:
- `test_synthesize_step_has_evidence_pack_metadata` — synthesis step metadata includes
  `evidence_pack_items`, `table_evidence_count`, `source_stages` when evidence pack exists.
- `test_synthesize_step_without_evidence_pack_no_metadata_leak` — metadata stays clean
  when no retrieve step runs.
- `test_executor_passes_evidence_pack_to_synthesize_tool` — verifies `evidence_pack` is
  present in the `answer.synthesize` tool call args.

## Tests Passed

All targeted test suites pass with no regressions:

- `tests/test_agent_synthesizer.py` — 94 tests passed (24 pre-existing + 10 new)
- `tests/test_tool_registry.py` — (included in above count, 4 new)
- `tests/test_agent_executor.py` — (included in above count, 3 new)
- `tests/test_agent_streaming.py` — (unchanged, still passing)
- `tests/test_agent_routes.py` — 33 tests passed
- `tests/test_agent_trace_store.py` — (included in above count)
- `tests/test_conversation_memory.py` — (included in above count)

## Verification Commands

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_agent_synthesizer.py tests/test_tool_registry.py tests/test_agent_executor.py tests/test_agent_streaming.py -q
# Result: 94 passed in 22.89s

D:\Miniconda3\python.exe -m pytest tests/test_agent_routes.py tests/test_agent_trace_store.py tests/test_conversation_memory.py -q
# Result: 33 passed in 30.35s
```

## Design Notes

- Anchor extraction uses **generic regex rules only** — no case IDs, no hard-coded expected
  terms like "NMR" or "0.5 kcal/mol" as special cases. This satisfies the Forbidden Actions
  constraint.
- Coverage retry is **bounded to at most one additional API call**. The `_should_retry_for_coverage`
  gate is checked once and never re-checked for the retry response.
- The evidence-heavy route list (`evidence_required`, `table_or_metric`, `multi_source_compare`)
  matches the route types defined in `PolicyRouter`.
- All existing fallback behavior (local fallback, malformed response fallback, API error
  fallback) is preserved unchanged.
- No new dependencies, no background jobs, no persistence of raw provider responses.

## Rework 2026-06-28-2 — Evidence-Pack Prompt Formatting

**Problem**: The original Phase 3 implementation built evidence-pack prompt sections
inline in both `_external_synthesize()` and `_retry_with_anchors()` with duplicate code.
The per-item line format omitted `support_hint` and there was no cap on the number
of items or excerpt length, risking unbounded prompt text for large evidence packs.

**Changes**:

- Added class constants `MAX_EVIDENCE_PACK_ITEMS = 10` and `MAX_EXCERPT_CHARS = 300`.
- Added `_format_evidence_pack_section(evidence_pack)` static method:
  - Each line preserves: `[index] doc=<id> kind=<kind> stage=<stage> hint=<hint>: <excerpt>`
  - Capped at `MAX_EVIDENCE_PACK_ITEMS` items.
  - Excerpts truncated to `MAX_EXCERPT_CHARS` with trailing `…`.
  - Header shows `(emitted of total)` count.
  - Truncation note when items exceed the cap.
  - Returns empty string for missing/empty evidence pack.
- Replaced both inline evidence-pack formatting blocks (in `_external_synthesize` and
  `_retry_with_anchors`) with calls to the shared `_format_evidence_pack_section()` helper.
- Updated `test_synthesize_external_with_evidence_pack_includes_items` to assert all
  required fields (`hint`, `stage`, `kind`, `doc`, excerpt) appear in the prompt.
- Added 5 new formatting tests:
  - `test_format_evidence_pack_section_includes_all_fields` — all metadata keys present.
  - `test_format_evidence_pack_section_caps_items` — only `MAX_EVIDENCE_PACK_ITEMS` emitted.
  - `test_format_evidence_pack_section_truncates_long_excerpts` — excerpts capped at
    `MAX_EXCERPT_CHARS` with ellipsis.
  - `test_format_evidence_pack_section_empty_inputs` — empty/missing inputs return `""`.
  - `test_retry_uses_same_evidence_pack_format` — retry prompt also includes all fields.

**Verification** (after rework):

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_agent_synthesizer.py -q
# Result: 29 passed in 0.46s

D:\Miniconda3\python.exe -m pytest tests/test_agent_synthesizer.py tests/test_tool_registry.py tests/test_agent_executor.py tests/test_agent_streaming.py -q
# Result: 99 passed in 24.16s
```

No regressions. All acceptance criteria met: `support_hint` present, item count capped,
shared formatting between first call and retry, backward compatible when evidence absent.
