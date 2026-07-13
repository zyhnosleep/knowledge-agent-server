# Remove Wiki Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the Wiki subsystem completely and make Document/DocumentChunk/PDF the only source path.

**Architecture:** Delete WikiPage storage, filesystem mounts, renderer, lint API, Wiki fallback/query code, and pipeline generation. Replace source reconstruction with raw document text and chunks. Add a migration/cleanup command for the `wiki_pages` table and `data/wiki` files.

**Tech Stack:** FastAPI, SQLAlchemy, SQLite, pytest, vanilla frontend.

---

### Task 1: Establish removal guards

**Files:** `tests/test_wiki_removed.py`

- [ ] Add tests asserting application startup has no `/wiki` static mount, API route, or `WikiPage` mapper, and that RAG retrieval does not call Wiki fallback.
- [ ] Run the focused tests and confirm they fail against the current implementation.

### Task 2: Remove storage and configuration

**Files:** `src/app/models/records.py`, `src/app/db/session.py`, `src/app/core/config.py`, `src/app/main.py`, `src/app/services/filesystem.py`

- [ ] Remove `PageKind`, `WikiPage`, Project wiki relationship, `wiki_page_id`, Wiki settings, directory creation, static mount, and Wiki indexes.
- [ ] Keep raw upload and cache directories intact.
- [ ] Run model/config tests.

### Task 3: Remove API and pipeline integration

**Files:** `src/app/api/routes.py`, `src/app/services/pipeline.py`, `src/app/services/wiki.py`, `src/app/services/wiki_quality.py`

- [ ] Remove Wiki imports, routes, source-page reconstruction, rendering, quality reports, and Wiki-specific pipeline prompts/outputs.
- [ ] Delete obsolete service modules.
- [ ] Update document deletion to remove only Document/Chunk/vector/trace/session resources.
- [ ] Run API and pipeline tests.

### Task 4: Make RAG document-only

**Files:** `src/app/services/search.py`, `src/app/schemas/agent.py`, `src/app/services/paper_profile.py`, `src/app/services/rag_adapter.py`

- [ ] Remove Wiki-first methods, Wiki fallback, Wiki citation promotion, Wiki-link normalization, and Wiki page context types.
- [ ] Use document chunks/raw text for source display and citations.
- [ ] Preserve document scope and evidence-anchor behavior.
- [ ] Run query, Agent, and adapter tests.

### Task 5: Remove obsolete tests/docs and add migration

**Files:** `tests/test_wiki_renderer.py`, `tests/test_wiki_quality.py`, `tests/test_model_constraints.py`, affected API/query tests, `scripts/remove_wiki_data.py`, `docs/work.md`

- [ ] Delete Wiki-only tests and rewrite mixed tests around Document/Chunk behavior.
- [ ] Add an idempotent cleanup script that drops `wiki_pages` and removes `data/wiki` after an explicit backup confirmation.
- [ ] Update project documentation to remove Wiki terminology.

### Task 6: Full verification

- [ ] Run focused tests, then the complete test suite.
- [ ] Run `rg -n -i "wiki|WikiPage|wiki_dir|wiki_pages" src tests scripts docs` and remove every production/documentation reference except the migration's historical cleanup note.
- [ ] Run `git diff --check` and compile checks.

