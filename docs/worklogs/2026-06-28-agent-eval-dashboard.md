# Agent Eval Dashboard — Phase 2 Implementation

Date: 2026-06-28
Workflow: rag-agent-eval-dashboard-20260628

## Summary

Implemented Phase 2 of `docs/2026-06-28-rag-agent-project-plan.md`: added read-only
Agent/Evidence evaluation metrics (`agent_metrics`) to the existing quality dashboard.

## Changes

### Schema (`src/app/schemas/quality.py`)
- Added `AgentMetrics` Pydantic model with fields: `query_total`, `query_selected`,
  `query_completed`, `query_passed`, `query_failed`, `failure_reason_counts`,
  `likely_stage_counts`, `failed_likely_stage_counts`, `unattributed_case_ids`,
  `retrieval_coverage`, `table_evidence_cases`, `agent_tool_counts`,
  `agent_provider_counts`.
- Added `agent_metrics: AgentMetrics | None = None` to `RunSummary`.
- Existing fields unchanged; backward-compatible.

### Service (`src/app/services/quality_reports.py`)
- Added `_build_agent_metrics(manifest, run_dir)` method:
  - Reads query totals from same-directory `query_eval.json` with manifest summary fallback.
  - Reads stage/failure counts from manifest `query_eval.attribution_summary`, falling
    back to same-directory `query_attribution.json` when manifest summary is absent.
  - Computes `retrieval_coverage` from citation source-hint fields in attribution cases.
  - Computes `table_evidence_cases` from citation excerpt groups and table failure reasons.
  - Reads optional `agent_trace_summary.json` or `agent_traces.json` for tool/provider counts.
  - Returns `None` when no query/agent artifacts are available.
  - Never trusts manifest-embedded paths (`report_path`, `attribution_path`).
- Added `_read_agent_trace_sidecar(run_dir)` helper.
- Added module-level helpers: `_safe_int()`, `_coerce_string_keys()`.
- Added `collections.Counter` import.

### Route (`src/app/api/quality_routes.py`)
- No changes needed — Pydantic response model automatically serializes `agent_metrics`.

### Tests (`tests/test_quality_reports.py`)
Added `TestAgentMetrics` class (12 tests):
- `test_null_when_no_artifacts`
- `test_from_manifest_attribution_summary`
- `test_from_query_attribution_fallback`
- `test_retrieval_coverage`
- `test_table_evidence_cases`
- `test_agent_trace_sidecar`
- `test_agent_trace_sidecar_fallback_to_traces`
- `test_malformed_sidecar_treated_as_missing`
- `test_manifest_embedded_paths_ignored`
- `test_unattributed_case_ids_from_manifest_gate`
- `test_query_totals_from_query_eval_json`
- `test_existing_fields_still_work`

### Tests (`tests/test_quality_routes.py`)
Added `TestAgentMetricsViaRoute` class (2 tests):
- `test_route_response_includes_agent_metrics`
- `test_route_agent_metrics_null_when_no_artifacts`

## Verification

- `pytest tests/test_quality_reports.py tests/test_quality_routes.py -q`: 50 passed
- `pytest tests/test_agent_routes.py tests/test_agent_trace_store.py tests/test_conversation_memory.py -q`: 33 passed
- No regressions in existing quality dashboard functionality.
- Malformed optional sidecars are treated as missing, not as malformed manifests.
- Manifest-embedded paths are never used for sidecar reads.

## Design Decisions

- `agent_metrics` is nullable: `None` for runs without query/agent artifacts.
- Returns `None` when no meaningful query data is available (no totals, no stage counts).
  This prevents noise from purely structural manifest runs.
- `retrieval_coverage` is itself nullable: `None` when `query_attribution.json` is absent.
  Distinct from an empty dict to make absence explicit vs zero-evidence.
- `agent_tool_counts` and `agent_provider_counts` default to empty dicts when no trace
  sidecar exists — distinct from null to match the "defaulting to empty dicts" spec.
- Two-level sidecar resolution for traces: `agent_trace_summary.json` first (preferred
  pre-aggregated format), then `agent_traces.json` (individual traces aggregated inline).
- Gate `unattributed_case_ids` sourced from both manifest gate details and attribution
  file fallback.

## Trace-Only Rework (2026-06-28)

### Issue

The initial implementation returned `None` for `agent_metrics` when no query totals
or stage counts existed, even when `agent_trace_summary.json` or `agent_traces.json`
was present. This meant trace-only runs were hidden from the dashboard.

### Fix

Changed the early-return guard in `_build_agent_metrics` (line ~453) from:

```python
if (query_total is None and query_passed is None and query_failed is None
    and not failure_reason_counts and not likely_stage_counts):
    return None
```

to include trace data checks:

```python
if (query_total is None and query_passed is None and query_failed is None
    and not failure_reason_counts and not likely_stage_counts
    and not agent_tool_counts and not agent_provider_counts):
    return None
```

### New Tests (`tests/test_quality_reports.py`)

Added `TestTraceOnlyMetrics` class (4 tests):
- `test_trace_summary_only_no_query_artifacts` — `agent_trace_summary.json` present, no query artifacts → non-null with tool/provider counts
- `test_traces_json_only_no_query_artifacts` — `agent_traces.json` present, no query artifacts → non-null
- `test_malformed_trace_no_query_still_null` — malformed trace + no query → `None`
- `test_null_when_no_artifacts_at_all` — no query + no trace → `None`

### Verification

- `pytest tests/test_quality_reports.py tests/test_quality_routes.py -q`: 54 passed (4 new + 50 existing)
