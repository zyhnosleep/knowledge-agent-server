# Canonical Table Evidence Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make canonical table and metric answers receive complete, version-scoped evidence for every requested table/row group while keeping citations bound to the original Child chunks.

**Architecture:** Keep MinerU parsing, the pinned Qwen tokenizer, semantic Parent/Child boundaries, embeddings, active pointers, and test data unchanged. Add a query-time table evidence assembly stage after Child ranking: group hits by `document_id + parse_version + table_id`, load only same-version sibling table Children, select requested table/dataset/metric row groups deterministically, and pass each selected Child as a separate citation-bearing context. Canonical answer rendering will use the strict retrieval tokenizer and complete Child units rather than a character window; the existing context budget remains the hard upper bound.

**Tech Stack:** Python 3.12, SQLAlchemy, pytest, pinned Qwen tokenizer, SQLite/pgvector shadow routing.

---

## Evidence and non-goals

- Old test `29/30` is a legacy `/api/query` run that reads complete tables from `document.metadata_json["document_intelligence"]["tables"]`; it is not an apples-to-apples canonical answer baseline.
- Active canonical is not an acceptable new-quality baseline: its report has `source_fidelity_completeness=0`, `structured_limit_completeness=0`, `config_identity_completeness=0`, `source_version_identity_completeness=0`, and `child_token_limit_completeness≈0.00066`.
- Staged shadow has all strict integrity fields equal to `1.0`, Recall@5/10 equal to `1.0`, citation/location equal to `30/30`, and ten answer failures; nine are table/metric cases where requested values are absent from the final top contexts although citation validation passes.
- Current code still limits retrieval to `MAX_CONTEXTS=8`, table citations to five, draft evidence to 6000 retrieval tokens, and `_prompt_context_text()` to a 2400-character table window or 1600-character ordinary window.
- Do not modify `parser.py`, MinerU settings, `semantic_chunking.py`, table chunk boundaries, tokenizer identity, embedding model, active pointers, old data, or the test environment in this task. Do not activate staged as part of this task.

## Task 1: Add focused failing retrieval and prompt tests

**Files:**
- Modify: `D:/LLM_wiki/.worktrees/internal-pilot/tests/test_canonical_retrieval.py`

- [ ] **Step 1: Add a same-table sibling coverage test.**

Create one canonical document with a table Parent and three table Children sharing `table_id`, each containing a different requested row/value. Make the vector/lexical hit be only the middle Child, call the query-time table assembly helper (or the public source retrieval path), and assert that all requested row Children are returned in deterministic row order. Assert every context has its own `citation.chunk_id`, `citation.excerpt`, `citation.source_spans`, `citation.table_id`, and `citation.parse_version`.

- [ ] **Step 2: Add a multi-table coverage test.**

Create two tables in one document and a question naming both table labels/datasets. Assert the assembled evidence includes at least one relevant Child from each requested table, even when one table has a lower raw vector score.

- [ ] **Step 3: Add a shadow-version isolation test.**

Store active and staged Children with the same `table_id` but different values. Build `QueryService(parse_version_map={document_id: staged_version})`, assemble table evidence, and assert only staged excerpts are present and `Document.active_parse_version` is unchanged.

- [ ] **Step 4: Add a complete-table prompt test.**

Use a table context whose evidence is longer than 2400 characters and assert canonical table prompt rendering preserves the complete evidence when it fits the configured token budget. The test must fail against the current character-window implementation.

- [ ] **Step 5: Run the tests and record the expected RED state.**

Run:

```powershell
$env:TEMP='D:\\temp'; $env:TMP='D:\\temp'
pytest -q tests/test_canonical_retrieval.py -k "table or shadow or prompt"
```

Expected: the new coverage tests fail because canonical retrieval returns only the hit Child, does not group same-table siblings, and `_prompt_context_text()` applies the character window.

## Task 2: Implement version-scoped table evidence assembly

**Files:**
- Modify: `D:/LLM_wiki/.worktrees/internal-pilot/src/app/services/search.py`
- Test: `D:/LLM_wiki/.worktrees/internal-pilot/tests/test_canonical_retrieval.py`

- [ ] **Step 1: Add typed table identity helpers.**

Extract `table_id` from the same canonical source-span/metadata path used by `_canonical_chunk_identifiers()`. Treat a context as table evidence only when `block_type == "table"` and it has a valid table identity; never infer a table ID from an arbitrary block ID or caption string.

- [ ] **Step 2: Load only same-version sibling Children.**

For each ranked table hit, query `DocumentChunk` with the hit's `document_id`, the expected active/shadow `parse_version`, `chunk_role == "child"`, `block_type == "table"`, and the same `table_id`. Exclude references and all other parse versions. Order by ordinal and deduplicate by Child ID. Keep the SQL branch compatible with `parse_version_map` so shadow reads cannot leak active values.

- [ ] **Step 3: Select requested row groups deterministically.**

Use existing table/dataset/metric selectors (`_query_priority_anchors()`, `_extract_generic_table_terms()`, `_question_row_selectors()`, and `_table_block_matches_query()`) to score sibling Children. Always include the hit Child, then include siblings containing requested table labels/datasets/metrics or the row group needed to complete the same table. Preserve caption/header/footnote Children when they carry identity needed to interpret a row. Do not include unrelated tables merely because they share a page.

- [ ] **Step 4: Convert each sibling into an independent RetrievedContext.**

Call `_expand_child_hit()` for every selected sibling, carrying its own score, exact excerpt, source spans, table ID, parse version, and parent ID. Do not merge sibling text into one synthetic citation. Keep the original ranked hit first, then stable ordinal order for additions.

- [ ] **Step 5: Integrate assembly without changing raw ranking.**

Call the new assembly stage from `_search_source_chunks()` only after table filtering/ranking and before `_finalize_contexts()`. Keep non-table retrieval unchanged. For multi-table questions, reserve a context slot for each explicitly requested table before filling remaining slots by score. The final context list may exceed the raw hit count but must still be bounded by the existing context-token budget.

## Task 3: Remove canonical table character truncation while preserving the token budget

**Files:**
- Modify: `D:/LLM_wiki/.worktrees/internal-pilot/src/app/services/search.py`
- Test: `D:/LLM_wiki/.worktrees/internal-pilot/tests/test_canonical_retrieval.py`

- [ ] **Step 1: Preserve complete table Child units.**

Update `_prompt_context_text()` so canonical table contexts are returned in full. Do not slice `citation.excerpt` or `prompt_text` by characters. Ordinary narrative behavior must remain unchanged in this task unless required by an existing regression test.

- [ ] **Step 2: Enforce the exact retrieval-token budget.**

Keep `_fit_contexts_to_token_budget()` as the sole budget gate. It must use the pinned local tokenizer already exposed by `StructuredEvidenceBuilder`; missing strict tokenizer assets must raise the existing diagnostic error instead of silently switching to byte counts. If a complete sibling Child does not fit, omit that Child as a whole and retain its citation only if the caller has not promised it as supporting evidence; never truncate a Child internally.

- [ ] **Step 3: Verify citation and prompt invariants.**

Assert that prompt assembly can contain several Children from one table, that repeated caption/header text does not duplicate unnecessarily, and that each citation excerpt remains the exact original Child text. Do not use a Parent citation to cover sibling values.

## Task 4: Run local red-green verification and review

**Files:**
- Test: `D:/LLM_wiki/.worktrees/internal-pilot/tests/test_canonical_retrieval.py`
- Test: `D:/LLM_wiki/.worktrees/internal-pilot/tests/test_vector_retrieval.py`
- Test: related canonical retrieval/config tests selected from the existing suite

- [ ] **Step 1: Run focused tests.**

```powershell
$env:TEMP='D:\\temp'; $env:TMP='D:\\temp'
pytest -q tests/test_canonical_retrieval.py -k "table or shadow or prompt"
pytest -q tests/test_vector_retrieval.py
```

Expected: all focused tests pass, including the new same-table, multi-table, shadow isolation, and complete-prompt tests.

- [ ] **Step 2: Run canonical retrieval regression.**

```powershell
$env:TEMP='D:\\temp'; $env:TMP='D:\\temp'
pytest -q tests/test_canonical_retrieval.py
```

Resolve regressions without changing tests to fit the implementation. Existing unrelated failures must be investigated rather than marked as expected.

- [ ] **Step 3: Review the diff and test boundaries.**

Run `git diff --check` and inspect that only `search.py`, the focused test file, this plan, and the progress record changed. Confirm no parser/chunker/config/active-pointer changes slipped in.

## Task 5: Development-only shadow acceptance

**Files/artifacts:**
- Deploy only verified `search.py` and its tests to `/home/<user>/knowledge-agent-dev`.
- Generate new staged shadow retrieval/full-answer reports under `/home/<user>/knowledge-agent-dev/runtime/task15/`.
- Modify: `D:/LLM_wiki/.worktrees/internal-pilot/docs/work.md`

- [ ] **Step 1: Preflight the development service and GPU boundary.**

Confirm the test environment is untouched, the staged parse-version map still points to the strict candidate versions, and all rebuild/evaluation commands use `CUDA_VISIBLE_DEVICES=0`. Do not stop or start GPU1 workloads.

- [ ] **Step 2: Run staged retrieval-only evaluation.**

Require Recall@5/10 `1.0`, source-location validity `1.0`, citation validity `1.0`, and no parse-version leakage. Stop before full-answer evaluation if retrieval regresses.

- [ ] **Step 3: Run staged full-answer evaluation.**

Compare the same 30 cases and record answer pass rate, required-term coverage, citation validity, source location, and P50/P95 latency. The table fix is accepted only if the ten known table/metric misses are resolved or a concrete evidence-backed blocker is documented; do not activate staged solely because retrieval passes.

- [ ] **Step 4: Keep staged inactive and update `docs/work.md`.**

Record the root cause, exact files/lines changed, test output, report paths, active/staged semantics, GPU proof, and the decision not to activate or delete old data. Do not modify test data or run cleanup.

## Acceptance gates

- Same-table and multi-table evidence coverage tests pass.
- Every surfaced citation points to an original Child with exact text, source spans, table ID, and parse version.
- Shadow reads never expose active-version table text and never mutate the active pointer.
- Canonical table prompt evidence is not character-truncated; the exact tokenizer budget remains enforced.
- Local focused and canonical retrieval tests pass.
- Development staged shadow evaluation shows no retrieval/citation/location regression and improves the known table answer failures before any activation decision.
