# Task 15 Development Rebuild and Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild all development documents with the corrected MinerU-first `canonical-v4` pipeline on GPU0, use raw embeddings for every Child without ingestion-time LLM calls, preserve all old RAG data, and compare the result with the unchanged old test environment.

**Architecture:** Development is the only mutable environment. Test is a read-only old-version baseline. V1/V2/V3 artifacts remain preserved and inactive; `canonical-v4` isolates the all-direct embedding policy. Two real-paper canaries precede the full 17-document rebuild; query testing starts only after every rebuild integrity metric passes.

**Tech Stack:** Python 3.12, SQLAlchemy, PostgreSQL/pgvector, MinerU 3.4.0, Ollama, systemd user services, pytest.

---

## Hard Boundaries

- Development: `/home/zhangyh/knowledge-agent-dev`, API `8002`, Ollama `11435`, database `knowledge_agent_dev`.
- Test: `/home/zhangyh/knowledge-agent-test`, API `8001`, Ollama `11436`, database `knowledge_agent_test`.
- Development may use GPU0 only. GPU1 is outside Task 15.
- Do not update, migrate, restart, or write into the test environment.
- Do not use `--delete-old-after-acceptance` or `--confirm-delete-old-data`.
- Preserve old chunks, vectors, parse versions, artifacts, MinerU outputs, source files, and staging copies.
- Never resume, overwrite, or activate a V1/V2/V3 checkpoint with V4 code.
- Keep development maintenance mode enabled until all 17 documents pass strict rebuild integrity.
- Run no retrieval/full30/citation test before the 17-document report has `ready_for_acceptance=true`.

## Verified Start

- No rebuild is currently running.
- Local and development copies of the missing-page fix have matching SHA-256 hashes.
- The fix keeps MinerU primary and supplements only MinerU-uncovered pages from pypdf.
- Real `ff14sb` validation supplements pages 7, 8, 9, 15, 16, 45, 46, 48, and 50; coverage becomes 50/50.
- Focused local and development regression: `126 passed`.
- Development has 17 documents; test retains its old 15 documents and 1628 chunks.
- The first `ff14sb` all-Child contextualization attempt was stopped after more than 120 LLM calls and about 50 minutes. Its inactive `canonical-v1` checkpoint is preserved.
- The approved final policy embeds all six retrievable Child types directly and never constructs the contextualization model during ingestion.

---

### Task 1: Freeze the Pre-Resume State

**Files:**
- Server generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/pre-resume-state-20260728.json`

- [ ] Confirm `pgrep -af '[r]ebuild_canonical_index.py'` returns no process.
- [ ] Record both checkout HEADs, Alembic heads, service states, maintenance mode, MinerU settings, models, dimensions, document/chunk/vector/parse-version counts, and source SHA-256 values.
- [ ] Confirm development has 17 documents and test still has 15 documents / 1628 chunks.
- [ ] Confirm local/development hashes match for `canonical_adapters.py`, `pipeline.py`, and their two new regression-test files.
- [ ] Reconfirm test checkout, database, source files, and services are unchanged before proceeding.

### Task 2: Enforce GPU0 on Every Development Entry Point

**Files:**
- Modify: `deploy/systemd/knowledge-agent-dev-worker.service`
- Test: `tests/test_deployment_config.py`
- Deploy: `~/.config/systemd/user/knowledge-agent-dev-worker.service`

- [ ] Add a failing test requiring the development worker unit to contain `Environment=CUDA_VISIBLE_DEVICES=0` while retaining all eight ingestion queues.
- [ ] Run `pytest tests/test_deployment_config.py -q`; expect the new assertion to fail.
- [ ] Add only `Environment=CUDA_VISIBLE_DEVICES=0` under the development worker `[Service]` section. Do not edit test units.
- [ ] Run `pytest tests/test_deployment_config.py tests/test_contextual_ingestion_config.py -q`; expect all tests to pass.
- [ ] Deploy only the development worker unit, run `systemctl --user daemon-reload`, and restart development API/worker.
- [ ] Inspect each development service PID through `/proc/<pid>/environ`; require `CUDA_VISIBLE_DEVICES=0`, `MINERU_ENABLED=true`, and the development MinerU binary.
- [ ] Use `nvidia-smi` to prove development processes use GPU0 only. Do not stop or change GPU1 processes.

### Task 3: Run the Clean `ff14sb` V4 Canary

**Files:**
- Server generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/canary-ff14sb-v4.json`

- [x] Run this development-only command:

```bash
cd /home/zhangyh/knowledge-agent-dev
set -a
. runtime/app.env
set +a
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=src .venv/bin/python scripts/rebuild_canonical_index.py --resume \
  --document-id 098a4ce8-d772-46a5-ae67-e0594a355460 \
  --report runtime/task15/canary-ff14sb-v4.json
```

- [x] Require the selected parse version to start with `canonical-v4-`; no V1/V2/V3 checkpoint or artifact may be modified.
- [x] Require `failed_documents=0`, `contextual_prefix_completeness=1.0`, `plain_embedding_completeness=1.0`, all other completeness/validity fields `1.0`, and `ready_for_acceptance=true`.
- [x] Require every Child to have `embedding_text == text` and no contextualization fields; contextualization eligible and contextualized counts must both be zero.
- [x] Load the active artifact and require `primary_parser=mineru`, 50/50 pages, and zero fatal quality issues.
- [x] Require the nine fallback blocks to have `parser_source=pypdf_text_layer` and `fallback_reason=mineru_page_missing`.
- [x] Stop on failure; attribute the exact failed stage before changing code.

### Task 4: Run the `opls5` Table Canary

**Files:**
- Server generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/canary-opls5-v4.json`

- [x] Repeat Task 3 with document ID `1180eb94-80d3-4fca-be00-ca4b53268747` and report `runtime/task15/canary-opls5-v4.json`.
- [x] Require the same strict report fields and `ready_for_acceptance=true`.
- [x] Require every table status to be `accepted_mineru`, `repaired_by_vision`, or `cross_page_merged`.
- [x] Require no `validation_failed` table, non-empty normalized Markdown, and valid source spans for every table block.

### Task 5: Rebuild All 17 Development Documents

**Files:**
- Server generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/canonical-rebuild-dry-run.json`
- Server generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/canonical-rebuild-report.json`

- [x] Run a fresh dry-run with explicit GPU0 binding; require 17/17 sources present and zero failures.
- [x] Run the full command:

```bash
PYTHONPATH=src .venv/bin/python scripts/rebuild_canonical_index.py --resume \
  --report runtime/task15/canonical-rebuild-report.json
```

- [x] Monitor stage checkpoints and GPU ownership without starting a second rebuild process.
- [x] Require: 17 succeeded, 0 failed, parse/context-eligible prefix/plain embedding/vector/pgvector/table/span/artifact completeness all `1.0`, and `ready_for_acceptance=true`.
- [x] If any document fails, keep maintenance enabled, write a failing regression test, deploy a focused development-only fix, and resume from its checkpoint.
- [x] Verify old chunks, vectors, inactive parse versions, artifacts, MinerU directories, and source hashes remain present.

### Task 6: Freeze a Fair Overlapping Comparison Set

**Files:**
- Read: `benchmarks/query/internal_research_v1.json`
- Read: `docs/query_acceptance/canonical_ingestion_v1.json`
- Server generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/comparison-case-audit.json`

- [x] Resolve every expected stable source identity in both environments.
- [x] Classify each case as `overlap`, `development-only`, or `missing`.
- [x] Match sources primarily by SHA-256 and secondarily by stable `sources/<slug>` identity.
- [x] Use only byte-identical `overlap` cases for old-vs-new scores.
- [x] Freeze case order, expected SHA-256, required terms, expected block type, and locator requirements in the runtime audit file.

### Task 7: Compare Old Test with New Development

**Files:**
- Server generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/test-old-full30/`
- Server generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/dev-new-full30/`
- Server generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/canonical-acceptance.json`
- Server generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/old-vs-new-comparison.json`

- [ ] Capture the old baseline through test API `8001`, but write every report under the development runtime directory.
- [ ] Do not change test code, database, files, environment, or services during baseline capture.
- [ ] After rebuild success only, set development maintenance mode false and restart development API/worker.
- [ ] Run the identical ordered cases against development API `8002` with the same timeout, warm-up policy, and top-k values.
- [ ] Run canonical acceptance on development using `scripts/evaluate_canonical_retrieval.py` and a runtime-filtered case file when committed cases are not present in both datasets.
- [ ] Require development Recall@10 to be at least the old baseline.
- [ ] Require development Recall@5 to be at least `max(0.80, old Recall@5 - 0.05)`.
- [ ] Require citation validity, source identity validity, source location validity, and contextual-prefix citation exclusion to be `1.0`.
- [ ] Investigate and rerun if development P95 latency regresses by more than 25%.
- [ ] Record per-case ranks, Recall@5/10, citation/location results, latency, and failure attribution.

### Task 8: Verify Workflow and Record Progress

**Files:**
- Modify: `docs/work.md`
- Modify: `deploy/internal-pilot.md`

- [ ] Verify desktop development workflows: login, upload queueing, stage progress, Markdown view/download, RAG query, Agent query, citation navigation/highlight, and health/queue status.
- [ ] Require no pending queue message, running checkpoint, or failed active parse version.
- [ ] Reconfirm test HEAD, counts, service timestamps, and representative hashes match Task 1.
- [ ] Record deployed hashes, model/MinerU versions, rebuild totals, repair counts, strict metrics, Recall@5/10, citation/location validity, P50/P95, GPU0 proof, report paths, test unchanged status, and old-data retention.
- [ ] Run the full local test suite and whitespace validation; require all tests to pass and no diff errors.

---

## Stop Conditions

Stop immediately if a development process uses GPU1, test state changes, a canary fails, any rebuild metric is below `1.0`, a source hash differs, contextual prefix appears in a citation, a citation points to the wrong source, or a command requests old-data deletion. Do not lower thresholds; diagnose, add a regression test, apply a focused development-only fix, and resume.

## Completion Definition

Task 15 is complete only when all 17 development documents are active and strictly complete, comparison reports are reproducible, development workflows pass, test remains unchanged, GPU0 isolation is proven, and all old RAG data remains retained.
