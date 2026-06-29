# 2026-06-26 Agent Query Loop

## Current Result

- Local pytest: `D:\Miniconda3\python.exe -m pytest tests/test_query_service.py tests/test_paper_profile.py -q` -> `182 passed`.
- Remote pytest: `.venv/bin/python -m pytest tests/test_query_service.py tests/test_paper_profile.py -q` -> `182 passed`.
- Remote representative set: `representative4_after_citation_marker.json` -> `4/4 passed`.
- Remote smoke set: `smoke5_after_rg_fix.json` -> `5/5 passed`.
- Remote full30: `full30_after_table_first_round5.json` -> `24/30 passed`.
- Previous full30 baseline after MinerU/table loop was `12/30 passed`, so this loop improved query acceptance by 12 cases and removed the long HTTP timeout pattern.

## Remote Access

Use the public endpoint when outside the LAN. Keep concrete host, port, username, and local key path in a private operator note, not in the repo:

```powershell
ssh -i <private-key-path> -p <public-port> <user>@<public-host>
scp -i <private-key-path> -P <public-port> <files> <user>@<public-host>:<server-project>/tmp/
```

The old `llm-wiki-server` alias points at the LAN address and can time out on public network.

## OCR And Parser State

- Requested package is `mineru[all]`.
- Remote environment currently uses MinerU `3.4.0` with full extras, including the pipeline backend and local model support.
- Local public-network validation installed `mineru[all]` into `D:\Miniconda3`; `mineru --version` reports `3.4.0` and `pip check` reports no broken requirements.
- Local MinerU smoke parsed page 0 of `Knowledge graph.pdf` with `MINERU_MODEL_SOURCE=modelscope`, `-b pipeline`, `--start 0 --end 0`, producing `content_list_v2.json` and Markdown under `tmp/mineru-smoke-kg-page0/Knowledge graph/auto/`.
- The application `_parse_pdf_with_mineru()` path was revalidated against the real CLI and now captures MinerU UTF-8 logs with replacement decoding to avoid Windows GBK `UnicodeDecodeError`.
- The loop runner now has an explicit real MinerU parser smoke gate. It is off by default and only runs when `--run-mineru-smoke` is passed. Latest real run: `tmp/loop_runs/mineru_rag_20260626_162941_579993/manifest.json`, `overall_status=passed`, `parser_mode=pdf_mineru`, `chunks=11`.
- The loop runner now has an explicit real service ingest gate. It is off by default and only runs when `--run-service-ingest-smoke` is passed. It uploads a PDF to `/api/ingest/upload`, polls `/api/runs`, then records `/api/documents/{id}/quality` and `/api/wiki/lint` in the manifest.
- Latest public-remote service ingest gate: `tmp/service_ingest_real_20260627_0052/manifest.json`, `overall_status=passed`, `duplicate_skipped=false`, `parser_mode=pdf_mineru`, 16 pages, 16 page outputs, 10 tables, 6 figures, and `content_list_v2.json` present. `/api/wiki/lint` returned one `missing_index` issue because this smoke ran in RAG-only mode with `SAC_KG_ENABLED=false`.
- Query failure attribution gate is now part of the runner. When `--run-query-eval` is used, the loop writes `query_attribution.json` and includes an attribution summary in `manifest.json`. Explicit query gate thresholds are available through `--min-query-passed`, `--max-query-failed`, and `--require-failure-attribution`.
- Latest public-remote target6 attribution run: `tmp/query_attribution_target6_20260627_0102/manifest.json`, `6/6 passed`.
- Latest public-remote full30 query gate run: `tmp/full30_query_gate_20260627_0123/manifest.json`, `25/30 passed`; gate passed with `min_query_passed=24`, `max_query_failed=6`, and `require_failure_attribution=true`. Remaining 5 failures all have `likely_stage=answer`, `missing_citation=[]`, and matching source hints.
- A local isolated RAG-only ingest smoke passed in `tmp/e2e_mineru_ingest_20260627_001505/`: document `ready`, run `completed`, `parser_mode=pdf_mineru`, `chunks=11`, `page_outputs=1`. Ollama was unavailable, so embeddings/model answers fell back locally; query smoke still returned 2 citations via retrieval fallback.
- `.env` remote settings use `MINERU_ENABLED=true`, `MINERU_BACKEND=pipeline`, `MINERU_MODEL_SOURCE=modelscope`, and a long MinerU timeout.
- Current OCR/parser path is MinerU pipeline OCR and layout/table/formula extraction, not direct local Qwen OCR.
- If we later use local Qwen, use it as an upstream vision/OCR model only after measuring against MinerU output. Do not replace MinerU blindly; MinerU is now the document parser baseline.

## Multi-Agent Coordination

- Singer/planner: confirmed the next target should be real service ingest + targeted query eval + failure attribution before any agent/front-end build.
- Russell/backend worker: extended `scripts/run_mineru_rag_loop.py` with explicit `--run-mineru-smoke`, added `scripts/mineru_parser_smoke.py`, and covered it with tests.
- Goodall/evidence analyst: reviewed the remaining 6 full30 failures and classified them mainly as retrieval/windowing/scientific-anchor issues, with `oplsaa_mechanism` and some `helix-coil`/`molten globule` wording needing original-paper benchmark checks.

## Current Loop Gates

1. Local default gate: `D:\Miniconda3\python.exe scripts\run_mineru_rag_loop.py --profile quick --python D:\Miniconda3\python.exe`.
2. Local real MinerU gate: add `--run-mineru-smoke --mineru-smoke-pdf "Knowledge graph.pdf" --mineru-smoke-start 0 --mineru-smoke-end 0 --mineru-bin D:\Miniconda3\Scripts\mineru.exe --mineru-model-source modelscope --timeout 900`.
3. Local isolated ingest gate: run RAG-only ingest in a temporary data directory when Ollama is unavailable; treat it as parser/chunk/citation plumbing evidence only.
4. Server ingest gate: run `scripts/run_mineru_rag_loop.py --run-service-ingest-smoke` against a separate `mineru-rag-smoke` project; it must prove `document_ready`, no duplicate skip, `parser_mode=pdf_mineru`, page outputs, `content_list_v2.json`, and lint reachability.
5. Full30 gate: only proceed toward agent/front-end work when full30 remains at least `24/30`, has no new timeout/source-contamination regressions, and every remaining failure has a concrete attribution. Current gate result: `25/30`, remaining failures are answer-stage only.

## Borrowing From Open Source Projects

Borrowing happens after the ingestion/query loop has a stable failure taxonomy, before building agent/front-end workflow features.

- RAGFlow/MinerU: document parsing, table layout, formula/table evidence preservation.
- Open WebUI/AnythingLLM: local-first model/provider settings, knowledge upload UX, admin controls.
- Dify/FastGPT/MaxKB: workflow/agent orchestration patterns and app publishing concepts.
- Onyx/R2R: API-first retrieval, hybrid search, connectors, background sync, evaluation discipline.
- Kotaemon: citation-first PDF QA and source preview UX.

Do not import a whole platform until we know which subsystem is actually needed. Prefer borrowing proven ideas and specific implementation patterns.

## Loop Procedure

1. Run focused local tests.
2. Sync changed files to the public remote after backing up the remote copies.
3. Run remote pytest.
4. Restart API and worker.
5. Run a representative set for the current fix.
6. Run smoke5 to catch regressions.
7. Run full30 only after representative and smoke pass.
8. Summarize failures by reason, not by isolated case.
9. Ask reviewer to check risks before declaring the loop stable.

Useful commands:

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_query_service.py tests/test_paper_profile.py -q
python scripts/query_report_summary.py tmp/query_benchmark/full30_after_public_loop_round4.json --benchmark benchmarks/query/internal_research_v1.json
```

Remote benchmark command:

```bash
.venv/bin/python scripts/query_eval.py benchmarks/query/internal_research_v1.json \
  --base-url http://127.0.0.1:8000 \
  --timeout 120 \
  --output tmp/query_benchmark/full30_after_table_first_round5.json \
  --markdown tmp/query_benchmark/full30_after_table_first_round5.md
```

Remote full30 query gate command:

```bash
.venv/bin/python scripts/run_mineru_rag_loop.py \
  --profile query \
  --python .venv/bin/python \
  --run-query-eval \
  --base-url http://127.0.0.1:8000 \
  --timeout 120 \
  --min-query-passed 24 \
  --max-query-failed 6 \
  --require-failure-attribution \
  --out-dir tmp/full30_query_gate_<stamp>
```

Remote targeted attribution command:

```bash
.venv/bin/python scripts/run_mineru_rag_loop.py \
  --profile query \
  --python .venv/bin/python \
  --run-query-eval \
  --base-url http://127.0.0.1:8000 \
  --timeout 120 \
  --case-id charmm36_mechanism \
  --case-id charmm36idpsff_overview \
  --case-id charmm36m_mechanism \
  --case-id ff14sb_overview \
  --case-id ff99sb_disp_overview \
  --case-id oplsaa_mechanism \
  --out-dir tmp/query_attribution_target6_<stamp>
```

Remote service ingest command:

```bash
.venv/bin/python scripts/run_mineru_rag_loop.py \
  --skip-pytest \
  --run-service-ingest-smoke \
  --service-ingest-pdf tmp/Knowledge_graph_service_smoke.pdf \
  --service-ingest-project-slug mineru-rag-smoke \
  --service-ingest-project-name "MinerU RAG Smoke" \
  --service-ingest-timeout 1800 \
  --service-ingest-poll-interval 5 \
  --service-ingest-http-timeout 60 \
  --base-url http://127.0.0.1:8000 \
  --out-dir tmp/service_ingest_real_<stamp>
```

## What Changed In This Loop

- Added canonical source identity aliases so exact paper routing is more stable.
- Expanded scientific profile terms and anchors for force-field papers.
- Added same-document scientific anchor context selection with diversity by evidence term.
- Preserved late anchors in long chunks, such as `GLH`.
- Allowed profile-term contexts from one source page to compete before final `MAX_CONTEXTS` truncation.
- Added supported term completion for evidence-backed variants such as `R_g -> radius of gyration/Rg`, `C36m -> CHARMM36m`, `IDPs -> IDP`, `2 kT -> 2kT`, and `34 + organic liquids -> 34 organic liquids`.
- Added deterministic scientific answers for Chinese profile-term evidence queries to avoid long model-generation timeouts.
- Added a RAG citation-marker fallback only when citations are returned but all valid markers were removed during renumbering.
- Added a short Chinese support note to generic Chinese table answers to avoid non-Chinese regression when table values contain many English headers.
- Changed RAG table/metric routing so structured table contexts are returned before profile-term/source chunk supplements, preventing scientific profile evidence from crowding out late matched tables such as OPLS5 Table 7.
- Added a service ingest loop gate that turns the previously manual upload/run/quality/lint check into a manifest-backed smoke test.
- Hardened the service ingest gate so duplicate-skip runs fail by default and run polling searches paginated `/api/runs` results instead of only the newest 50.
- Added structured query attribution JSON for each query eval run, including expected-term presence, citation sources, source-hint matching, and a coarse likely failure stage.
- Hardened query attribution to use case-insensitive term matching, per-citation citation group checks, failed-only stage counts, and explicit query acceptance thresholds in the runner.

## Remaining Failures

Current full30 remaining failures after attribution run:

- `charmm36m_overview`: answer missing `NMR`; citation/source evidence is present.
- `ff99sb_ildn_mechanism`: answer missing `0.5 kcal/mol`; citation/source evidence is present.
- `opls4_overview`: answer missing `van der Waals`, `GLH`; citation/source evidence is present.
- `opls4_mechanism`: answer missing `GLH`; citation/source evidence is present.
- `opls5_mechanism`: answer missing `FXA`, `-2.4`; citation/source evidence is present.

Goodall attribution notes:

- `charmm36idpsff_overview`, `charmm36m_mechanism`, `ff14sb_overview`, and probably `ff99sb_disp_overview` should first be treated as retrieval/windowing issues.
- `oplsaa_mechanism` needs original-paper verification for `hydration free energy` before adding rules.
- `charmm36_mechanism` may need both retrieval for `SPARTA` and benchmark wording review for `helix-coil` versus helical/extended equilibrium.
- Next targeted eval should record expected-term presence in answer, citations, raw contexts, paper match scope, before/after context lists, route terms, window positions, citation selection, and parser metadata.

Next repair priority:

1. Answer-stage completeness: when citation evidence already contains required scientific anchors, make the final answer reliably carry those anchors without benchmark hardcoding.
2. Keep source/citation gates strict: do not use answer completion to hide weak citation coverage.
3. Preserve table semantic filtering and full30 regression coverage before moving into agent/frontend implementation.

## OPLS5 Table 7 Follow-Up

`opls5_table_metrics` now passes after returning structured table contexts before profile/source supplements. A reviewer/explorer still recommended a deeper table-specific hardening pass:

- Split table matching terms into model terms, broad metric terms, and specific semantic anchors.
- Treat `binding RMSE`, `root mean square errors`, and `relative binding free energy` as aliases.
- Preserve summary rows such as `TotalWeightedAverage`, `Overall`, `Average`, and `RMS error` in table excerpts.
- Add noisy OPLS/RMSE table tests so generic RMSE tables cannot pass solely because they mention `OPLS4`/`OPLS5`.

## Reviewer Risk Notes

- Citation marker fallback can hide weak citation coverage if overused; keep it limited to RAG-first responses with returned citations.
- Profile-term same-source expansion improves recall but can crowd context budget; preserve cross-paper compare tests.
- Long anchor windowing can cut sentence boundaries; add tests for question-anchor priority if this area changes again.
- Avoid benchmark hardcoding. Every new term should be backed by evidence text, normalized aliases, or a general scientific notation rule.
