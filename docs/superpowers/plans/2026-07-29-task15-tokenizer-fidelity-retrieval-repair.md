# Task 15 Tokenizer, Fidelity, and Retrieval Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild development with real pinned Qwen token counts, lossless bounded Child chunks, auditable PDF coverage, and deterministic bounded Parent expansion, then require the new development version to match the unchanged old test baseline.

**Architecture:** Canonical parsing remains MinerU-first and all retrievable Children remain direct embeddings with `embedding_text == text`. A strict local tokenizer identity becomes part of the ParseVersion key and manifest, semantic splitting proves that every canonical source unit is represented, and answer drafting expands retrieved Children without character truncation under a real-token budget. Development is the only mutable server environment and may use GPU0 only.

**Tech Stack:** Python 3.12, transformers/Qwen tokenizer, MinerU 3.4, SQLAlchemy, PostgreSQL/pgvector, Ollama Qwen embeddings, pytest.

---

## Hard Boundaries

- Work only in `D:/LLM_wiki/.worktrees/internal-pilot` locally and `/home/zhangyh/knowledge-agent-dev` remotely.
- Do not modify, restart, migrate, or write into `/home/zhangyh/knowledge-agent-test`.
- Do not delete legacy/V1/V2/V3/V4 chunks, vectors, parse versions, artifacts, MinerU output, or source files.
- Development inference and rebuild commands must export `CUDA_VISIBLE_DEVICES=0`; GPU1 is out of scope.
- Preserve Parent `500/1200/1800`, Child `180/400/600`, overlap `50`, and semantic break percentile `20` for the first comparison.
- All Child types use direct embedding. No ingestion-time LLM contextual prefix is permitted.

### Task 1: Pin and Require the Real Qwen Tokenizer

**Files:**
- Modify: `src/app/core/config.py`
- Modify: `.env.development.example`
- Modify: `.env.test.example`
- Modify: `src/app/services/structured_evidence.py`
- Modify: `src/app/services/semantic_chunking.py`
- Modify: `src/app/services/pipeline.py`
- Modify: `scripts/rebuild_canonical_index.py`
- Test: `tests/test_contextual_ingestion_config.py`
- Test: `tests/test_structured_evidence.py`
- Test: `tests/test_ingestion_stages.py`
- Test: `tests/test_rebuild_canonical_index.py`

- [ ] Add failing tests that require the production ingestion tokenizer loader to raise a diagnostic error when the pinned local snapshot is unavailable; injected test counters must continue to work without a tokenizer.
- [ ] Run the focused tests and verify they fail because the current loader changes to `utf8_bytes_fallback`.
- [ ] Pin `Qwen/Qwen3-Embedding-4B` to revision `5cf2132abc99cad020ac570b19d031efec650f2b`, load it with `local_files_only=True`, hash the tokenizer asset files, and expose a stable identity dictionary.
- [ ] Build a canonical ingestion configuration snapshot containing tokenizer name/revision/content hash, embedding model/dimension, splitting model, break percentile, Parent/Child limits, overlap, and explicit parser/recovery/structured-splitting/fidelity schema revisions so changed algorithms cannot resume stale checkpoints.
- [ ] Hash canonical JSON of that snapshot, append a short configuration hash to the `canonical-v4-<source-sha>` ParseVersion key, and share the key helper with the rebuild script.
- [ ] Preserve the snapshot/hash when parse and repair update `manifest_json`; reject a queued stage if the live configuration fingerprint differs from its ParseVersion fingerprint.
- [ ] Add report fields for tokenizer availability and configuration identity and make readiness require both.
- [ ] Run focused tests and require all to pass.

### Task 2: Enforce Lossless Source and Structured-Chunk Fidelity

**Files:**
- Modify: `src/app/services/canonical_adapters.py`
- Modify: `src/app/services/semantic_chunking.py`
- Modify: `src/app/services/structured_evidence.py`
- Modify: `src/app/services/pipeline.py`
- Modify: `scripts/rebuild_canonical_index.py`
- Test: `tests/test_canonical_quality.py`
- Test: `tests/test_semantic_chunking.py`
- Test: `tests/test_structured_evidence.py`
- Test: `tests/test_rebuild_canonical_index.py`

- [ ] Add failing PDF tests where MinerU covers a page but omits a continuous source passage of at least 80 normalized characters, and require only the missing passage to become a `pdf_text_recovery` block.
- [ ] Add failing tests that ignore whitespace, line-break, dehyphenation, repeated header/footer, and formula-encoding noise. Every nonblank page must be sequence-matched for contiguous omissions; normalized page coverage below `0.90` is an additional warning/diagnostic threshold, never a prerequisite for detecting an omitted span.
- [ ] Implement normalized per-page sequence matching on every nonblank page. Keep MinerU blocks primary, independently recover every unmatched meaningful contiguous passage of at least 80 normalized characters even when aggregate coverage is at least `0.90`, and record page, coverage ratio, source offsets, largest unmatched span, and recovery reason.
- [ ] Add failing chunk-fidelity tests proving every retrievable canonical narrative unit, table header/row/cell/footnote, formula token, and figure source description is represented before semantic split succeeds.
- [ ] Add a structured failure containing document ID, page, block type, source ID, and a bounded diagnostic excerpt when reconstruction fails.
- [ ] Add failing tests for a formula over 600 real tokens, a table row/cell over 600 real tokens, and a figure source description over 600 real tokens; assert globally that no retrievable Child of any block type exceeds 600 real tokens.
- [ ] Split formulas first at LaTeX row/case boundaries and finally with lossless tokenizer windows. Preserve the full formula Parent; label direct-embedding Child parts deterministically.
- [ ] Split overlong table rows by column groups with repeated caption/header/row identity; split a single overlong cell with lossless tokenizer windows. Preserve complete cells/rows in the table Parent and never emit a Child over 600 tokens.
- [ ] Split overlong figure source descriptions into lossless tokenizer windows while preserving the complete figure Parent, asset identity, caption, source spans, and deterministic Child-part metadata.
- [ ] Emit source-fidelity and structured-limit completeness metrics and require both to equal `1.0` before activation/readiness.
- [ ] Run focused tests and require all to pass.

### Task 3: Rebuild Paper Profiles When Active Source Text Changes

**Files:**
- Modify: `src/app/services/paper_profile.py`
- Modify: `src/app/services/pipeline.py`
- Test: `tests/test_canonical_indexing.py`
- Test: `tests/test_pipeline_sac_kg.py`

- [ ] Add a failing activation test with a current-version but stale `paper_profile` and changed active Child text.
- [ ] Store a SHA-256 fingerprint of the profile source text in each generated profile and reuse a cached profile only when its version and source fingerprint both match.
- [ ] During canonical activation, build `Document.raw_text` from active Child text, preserve stable source identity, and force deterministic profile refresh when the active source fingerprint changed.
- [ ] Keep profile generation deterministic and local; do not add an LLM call.
- [ ] Run the focused indexing/profile tests and require all to pass.

### Task 4: Bound Parent and Table Expansion Without Character Truncation

**Files:**
- Modify: `src/app/services/search.py`
- Modify: `src/app/services/vector_store.py`
- Test: `tests/test_canonical_retrieval.py`

- [ ] Add failing tests showing current `_prompt_context_text()` cuts Parent/table evidence at 1600/2400 characters and current context selection can expand more than six unique Parents.
- [ ] Add failing tests requiring Top-10 retrieval results to remain unchanged while answer drafting selects at most six unique Parents under 10,000 real tokens.
- [ ] Make answer-context budgeting use the same pinned local Qwen tokenizer in strict mode; missing or mismatched tokenizer assets must fail diagnostically rather than use UTF-8 bytes.
- [ ] Remove character-window truncation from canonical answer contexts. Process contexts by descending Child score, deduplicate Parent IDs, and include a complete Parent only when it fits the remaining token budget; otherwise retain the matched Child without truncating it.
- [ ] For table hits, include the matched row Child, table caption/header/footnotes, and previous/next table Child when each complete unit fits. Every surfaced adjacent table Child must remain separate citation-bearing evidence with its own exact excerpt/source spans; never let neighbor-only values inherit the matched row's citation. Do not expand an oversized full table Parent.
- [ ] Deduplicate the 50-token narrative overlap deterministically in prompt assembly without changing citation excerpts or retrieved Child ranking.
- [ ] Run `pytest tests/test_canonical_retrieval.py -q` and require all tests to pass.

### Task 5: Make the Full Rebuild Batch-Atomic

**Files:**
- Modify: `scripts/rebuild_canonical_index.py`
- Modify: `scripts/evaluate_canonical_retrieval.py`
- Modify: `src/app/services/parse_versions.py`
- Modify: `src/app/services/search.py`
- Modify: `src/app/services/vector_store.py`
- Test: `tests/test_rebuild_canonical_index.py`
- Test: `tests/test_parse_versions.py`
- Test: `tests/test_canonical_retrieval.py`

- [ ] Add a failing two-document test proving that one failed staged document leaves both existing active ParseVersions unchanged.
- [ ] Run every selected document only through `index`, leaving each new ParseVersion in `ready_to_activate` while the old active version remains queryable.
- [ ] Validate the staged ParseVersions directly: source/version identity, Child inventory, direct embeddings, vector rows, table validity, source spans, artifacts, tokenizer/config identity, and zero over-limit Child chunks must all be complete.
- [ ] Add a read-only shadow ParseVersion map through query routing, SQL retrieval, vector retrieval, and the acceptance evaluator. Prove retrieval and answer tests can target staged versions while every `Document.active_parse_version` remains unchanged.
- [ ] Refactor activation validation and side effects into a no-commit operation. Add a batch activation operation that locks all selected documents and versions in stable order, verifies every row again, refreshes `raw_text`, stable source identity and `paper_profile`, completes activation checkpoints/run progress/document status, supersedes prior active versions, and changes all active pointers in one database transaction.
- [ ] If validation or flush fails, roll back the transaction and leave every previous active pointer/status, `raw_text`, profile, checkpoint, run progress, and document status unchanged.
- [ ] Preserve all inactive data and artifacts; do not invoke cleanup.
- [ ] Run focused rebuild/parse-version tests and require all to pass.

### Task 6: Local Regression and Independent Review

**Files:**
- Modify: `docs/work.md`

- [ ] Run all focused tokenizer, parsing, chunking, retrieval, indexing, and rebuild tests.
- [ ] Run the complete local suite and require at least the clean baseline `1673 passed, 3 skipped` with no failures.
- [ ] Run `git diff --check`.
- [ ] Review the implementation against every confirmed decision, then perform a separate code-quality review and resolve all important findings.
- [ ] Record local test totals and the exact remaining server work in `docs/work.md`.

### Task 7: Development-Only Side-by-Side Rebuild and Acceptance

**Files:**
- Deploy only changed source/config files to `/home/zhangyh/knowledge-agent-dev`
- Generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/tokenizer-preflight.json`
- Generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/test-old-baseline-readonly.json`
- Generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/canonical-v4-tokenizer-rebuild.json`
- Generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/dev-new-tokenizer-retrieval.json`
- Generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/dev-new-tokenizer-full30.json`

- [ ] Download the pinned tokenizer snapshot during deployment, then prove runtime loading is local-only and its revision/content hash match the ParseVersion snapshot.
- [ ] Freeze development and test inventories; capture the identical 30 byte-matched cases from test into a read-only old-baseline artifact, and do not mutate test.
- [ ] Build all 17 new versions side by side on GPU0. Do not activate any new version until all documents succeed and all strict metrics equal `1.0`.
- [ ] Run the 30 retrieval cases without answer generation against the staged shadow ParseVersions. Require Recall@5 `1.0`, Recall@10 `1.0`, source identity validity `1.0`, and no regression versus the old test baseline on any identical case; stop before answer tests on failure.
- [ ] Run the 30 full answer cases against the staged shadow ParseVersions only after retrieval passes. Require at least `29/30`, no regression versus the old test baseline total, and record latency separately.
- [ ] Atomically activate the new versions only after the full batch meets every hard gate. Preserve every old version and artifact.
- [ ] Reconfirm test HEAD, services, document/chunk counts, and representative source hashes are unchanged.
- [ ] Update `docs/work.md` with deployed hashes, tokenizer identity, chunk counts/distribution, fidelity metrics, Recall@5/10, answer score, latency, GPU proof, report paths, and old-data retention.

## Stop Conditions

Stop immediately if tokenizer loading falls back, a Child exceeds its model-token limit, source reconstruction is incomplete, a development process uses GPU1, any test state changes, a command requests old-data deletion, or a hard acceptance metric is below its threshold. Preserve the current active development version and diagnose with a failing test before any further rebuild work.
