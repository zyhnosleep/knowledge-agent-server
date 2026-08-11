# Task 16 Parse-Version Routing Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make acceptance evaluation and read-only shadow testing actually query the requested staged parse versions while preserving the production active-version pointer.

**Architecture:** `RAGAdapter` and the acceptance script already carry `parse_version_map`; the missing propagation is inside `QueryService`. QueryService will use a per-document shadow version in every child-chunk SQL condition, pass the map to the vector store, and validate/expand shadow hits against the same expected version. Documents without a map entry continue to use their active version, and no database pointer is changed.

**Tech Stack:** Python 3.12, SQLAlchemy, SQLite-vec/pgvector, pytest, PostgreSQL development environment.

---

## Root-cause evidence

- The v2 report retrieved chunk IDs whose parse versions ended in `-2cf11c0eec9b` (the staged rebuild), while v3 retrieved IDs from the active version without that suffix. The two reports therefore did not compare the intended same read path.
- `RAGAdapter` and `scripts/evaluate_canonical_retrieval.py` pass `parse_version_map` into `QueryService`, and vector-store classes already accept a shadow map, but `QueryService._search_source_chunks()` omitted it when calling `get_vector_store(...).search()`.
- All SQL retrieval paths still use `_active_child_chunk_conditions()`, which only matches `Document.active_parse_version`; shadow children are therefore invisible to lexical fallback and post-vector filtering.
- `_finalize_contexts()` rejects any hit whose version differs from `Document.active_parse_version`, which would discard a valid shadow hit even after retrieval.
- Existing tests already reproduce these three failures: staged child selection, staged evidence routing, and forwarding the map to the vector store.
- The corrected staged retrieval report now contains only the requested `-2cf11c0eec9b` versions for all mapped documents; its integrity gates and Recall@5/10 are both complete (`1.0`). This rules out the former v3 active/staged mix-up as an explanation for retrieval degradation.
- The staged full-answer report currently fails only ten table/metric answer cases. Every one of those rows has `citation_passed=true` but `required_terms_passed=false`; the missing values are absent from the Top-8 answer contexts. This points to table-evidence coverage/answer assembly, not parse-version leakage.
- The preserved old test read-only report is an `/api/query` run with 29/30 passed cases. It is a different runtime snapshot and evaluator schema from the direct `RAGAdapter.answer` full30 report, so it is a comparison reference, not yet an apples-to-apples regression result.

## Task 1: Add the failing regression coverage

**Files:**
- Test: `tests/test_canonical_retrieval.py`
- Test: `tests/test_vector_retrieval.py`

- [x] Confirm the existing three shadow-version tests fail before production changes.
- [x] Add one focused test proving an unmapped document in a mixed query still uses its active version while a mapped document uses its staged version.
- [x] Run the focused tests and record the expected failures before implementation.

## Task 2: Propagate shadow versions through QueryService

**Files:**
- Modify: `src/app/services/search.py`

- [x] Keep `_active_child_chunk_condition()` as the strict canonical/legacy SQL helper used by existing unit tests.
- [x] Add an instance helper that builds the selected child condition as: mapped `document_id + parse_version` branches for shadow documents, plus the existing active/legacy branch for all unmapped documents.
- [x] Replace every QueryService retrieval query that currently calls `_active_child_chunk_conditions()` with the selected-version helper, including overview, table, figure, scientific-anchor, claim, and source-chunk paths.
- [x] Pass `parse_version_map=self.parse_version_map` to `VectorStore.search()` so both SQLite-vec and pgvector use the same shadow selection.
- [x] When a shadow map is present, augment paper routing text with the mapped document's staged child text so routing cannot discard a document whose active `raw_text`/profile is intentionally old.
- [x] In `_finalize_contexts()`, compare a hit against `self.parse_version_map.get(document_id, document.active_parse_version)` rather than always against the active pointer.
- [x] Preserve the invariant that shadow reads never mutate `Document.active_parse_version`; parent and neighbor expansion must remain within the hit's parse version.

## Task 3: Verify the routing fix locally

- [x] Run the three existing shadow tests plus the new mixed-version test; all must pass.
- [x] Run the parse-version/vector-focused tests; all 26 selected tests pass. The broader canonical file still contains five pre-existing Task 15 prompt-budget failures unrelated to this fix.
- [x] Run `git diff --check` and inspect the final diff for unrelated changes.

## Task 4: Re-run the acceptance comparison with correct routing

- [x] Sync only the verified `search.py` change and its tests to `/home/zhangyh/knowledge-agent-dev`.
- [x] Restart only the development API if required; do not modify the test environment and keep GPU1 unused.
- [x] Run the staged full30 evaluation with the parse-version map and save a new report: `runtime/task15/dev-new-tokenizer-full30-v4-shadow-fixed.json`.
- [ ] Run the active-pointer baseline through the same code path and save `runtime/task15/dev-new-tokenizer-full30-v4-active-baseline.json`.
- [ ] Compare document-level Recall@5/10, evidence-term coverage, answer/citation pass rate, and latency. If active and staged are equivalent, do not change chunking; target the table-evidence path instead.
- [ ] Run the unchanged `/api/query` evaluator against the current development active pointer to compare fairly with `test-old-full30-readonly`; do not write to test.
- [ ] Update `docs/work.md` with the root cause, the fact that v3 was an invalid comparison, the corrected reports, and the activation decision.

## Task 5: Table-evidence follow-up (only after active comparison)

- [ ] Do not change parser or chunk limits based on v3. First identify whether the active baseline contains the same missing table values as staged.
- [ ] If both versions miss the same values, add a failing retrieval test for multi-table numeric questions and fix table-first candidate grouping/neighbor expansion so all requested table groups reach the answer context without changing source citations.
- [ ] If only staged misses values, compare table Child inventory, embeddings, and row-group boundaries for the affected documents; adjust structured table splitting only with a focused regression test and a new staged version.
- [ ] Re-run retrieval-only and full-answer acceptance after the focused fix; keep all versions inactive until the comparison gate is met.

## Acceptance criteria

- Shadow evaluation returns staged chunk IDs for mapped documents and leaves active pointers unchanged.
- Active evaluation returns active chunk IDs when no map is supplied.
- Mixed maps never leak an inactive version from an unmapped document.
- Focused and full local tests pass; development services remain stable.
- No staged version is activated as part of the routing fix itself.
