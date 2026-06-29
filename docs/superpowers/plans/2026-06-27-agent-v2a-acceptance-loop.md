# Agent v2a Acceptance Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the current Agent v2a implementation from "implemented by worker" to "reviewed, verified, synced to server, and accepted for online use."

**Architecture:** Codex main thread remains the decision owner. Claude Code workers are used only through the codex-with-cc task-file workflow for spec review, quality review, any rework, and final verification. The final acceptance loop is: main-thread self-review -> spec reviewer -> quality reviewer -> optional rework loop -> final-verifier -> local tests -> browser/UI check -> remote sync and server tests -> Codex acceptance decision.

**Tech Stack:** FastAPI, Pydantic, pytest, static HTML/JavaScript, local RAG/Ollama stack, codex-with-cc Claude Code delegation.

---

## Current Baseline

- Date baseline: 2026-06-27.
- Existing workflow id: `agent-v2a-20260627`.
- Implementer task id: `agent-v2a-implementer-001`.
- Implementer run id: `20260627_101125_359_ecc36e65`.
- Implementer artifact verification already passed with `verify_delegate_run.py`.
- Implementer report claims:
  - Agent v2a route policy is implemented.
  - Answer verifier is implemented and registered as `answer.verify`.
  - Agent executor route -> RAG -> verify -> optional one retry loop is implemented.
  - `/api/agent/query` response now includes `route` and `warnings`.
  - Agent panel shows route, warnings, verify and retry steps.
  - Focused Agent v2a tests passed: `92 passed`.
  - Regression subset passed: `208 passed`.
- Codex has not accepted the implementation yet. Remaining gates are self-review, two Claude Code review gates, final-verifier, local regression, browser check, remote sync, and remote smoke/regression.

## Acceptance Definition

The Agent v2a work is accepted only when all of these are true:

- Codex main thread review finds no blocker in route policy, verifier behavior, executor limits, response schema compatibility, frontend safety, or tests.
- A codex-with-cc `spec` reviewer accepts the implementer task.
- A codex-with-cc `quality` reviewer accepts the implementer task.
- Any reviewer blocker triggers a rework task and the review loop repeats for the new implementer task.
- A codex-with-cc `final-verifier` accepts the aggregate workflow.
- `verify_delegate_workflow.py -WorkflowId agent-v2a-20260627` passes.
- Local focused and regression pytest commands pass.
- Frontend Agent panel has no new dynamic HTML injection for Agent data.
- Server copy is backed up, synced, restarted, and passes health plus `/api/agent/query` smoke.
- Existing `/api/query` remains healthy enough to preserve the prior full30 gate baseline.

## API Decision

The production Agent answering path should keep using the existing local RAG/Ollama path for now. Do not add an external LLM API as a hard runtime dependency in this acceptance loop.

External API usage is allowed only for orchestration/report-only work that already belongs to the codex-with-cc workflow. If answer quality later requires a stronger model, add that as a separate model-routing task behind configuration, with tests proving local Qwen remains the default fallback.

---

### Task 1: Main-Thread Self-Review

**Files:**
- Inspect: `D:/LLM_wiki/src/app/services/agent_policy.py`
- Inspect: `D:/LLM_wiki/src/app/services/answer_verifier.py`
- Inspect: `D:/LLM_wiki/src/app/services/agent_executor.py`
- Inspect: `D:/LLM_wiki/src/app/services/tool_registry.py`
- Inspect: `D:/LLM_wiki/src/app/schemas/agent.py`
- Inspect: `D:/LLM_wiki/src/app/static/index.html`
- Inspect: `D:/LLM_wiki/tests/test_agent_policy.py`
- Inspect: `D:/LLM_wiki/tests/test_answer_verifier.py`
- Inspect: `D:/LLM_wiki/tests/test_agent_executor.py`
- Inspect: `D:/LLM_wiki/tests/test_tool_registry.py`
- Inspect: `D:/LLM_wiki/tests/test_agent_routes.py`

- [ ] **Step 1: Confirm current git state**

Run:

```powershell
git status --short
```

Expected: Agent v2a files are modified or untracked, with unrelated earlier dirty files left untouched.

- [ ] **Step 2: Review route policy**

Check:

- Empty or ambiguous question returns `needs_clarification`.
- Compare keywords route to `multi_source_compare`.
- Table, metric, unit, percentage, and numeric intent route to `table_or_metric`.
- Citation, evidence, source, page, or reference intent route to `evidence_required`.
- General questions route to `simple_rag`.
- Single-letter unit matching does not make ordinary words route to `table_or_metric`.

- [ ] **Step 3: Review answer verifier**

Check:

- Empty answer recommends retry and still returns a structured verification result.
- Missing citations recommend retry for routes that require evidence.
- Missing table evidence recommends retry for `table_or_metric`.
- The verifier is deterministic and local.
- It does not call external APIs.

- [ ] **Step 4: Review executor limits and retry behavior**

Check:

- Normal routes create route, RAG, verify, and finalize steps.
- `needs_clarification` skips RAG and returns a warning.
- `max_steps` and `max_tool_calls` are never exceeded.
- Timeout path preserves a useful error step.
- Retry is capped at one extra `rag.answer`.
- If retry answer is not re-verified because `max_tool_calls=3`, record this as either an accepted risk or a blocker before reviewer dispatch.

- [ ] **Step 5: Review tool registry**

Check:

- Built-in tools include `rag.answer`.
- Built-in tools include `answer.verify`.
- RAG citation dicts include `page_kind` and `page_label`.
- Tool errors are returned as structured values rather than uncaught exceptions.

- [ ] **Step 6: Review response schema and API compatibility**

Check:

- Request body remains backward-compatible for `/api/agent/query`.
- Response adds `route` and `warnings` without breaking existing clients.
- Step metadata is typed as a safe dictionary and can carry route, verification, retry, and error fields.

- [ ] **Step 7: Review frontend safety**

Run:

```powershell
rg -n "innerHTML\s*=.*(agent|step|answer|summary|error|data|result|payload|response|route|warning)|insertAdjacentHTML|eval\(" src\app\static\index.html
```

Expected: no Agent panel dynamic data is rendered through `innerHTML`, `insertAdjacentHTML`, or `eval`.

### Task 2: Local Test Gate Before Delegated Reviews

**Files:**
- Test: `D:/LLM_wiki/tests/test_agent_policy.py`
- Test: `D:/LLM_wiki/tests/test_answer_verifier.py`
- Test: `D:/LLM_wiki/tests/test_agent_executor.py`
- Test: `D:/LLM_wiki/tests/test_tool_registry.py`
- Test: `D:/LLM_wiki/tests/test_agent_routes.py`
- Test: `D:/LLM_wiki/tests/test_conversation_memory.py`
- Test: `D:/LLM_wiki/tests/test_rag_adapter.py`
- Test: `D:/LLM_wiki/tests/test_api_routes.py`
- Test: `D:/LLM_wiki/tests/test_query_service.py`
- Test: `D:/LLM_wiki/tests/test_paper_profile.py`

- [ ] **Step 1: Run focused Agent v2a tests**

Run:

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run local regression subset**

Run:

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q
```

Expected: all tests pass.

- [ ] **Step 3: If either command fails, stop review dispatch and create a rework implementer task**

First rework task id:

```text
agent-v2a-rework-001
```

If another rework loop is needed after `agent-v2a-rework-001`, increment the final three digits by one and keep all dependencies explicit.

Rework must depend on:

```text
-DependsOn agent-v2a-implementer-001
```

### Task 3: Create Spec Review TaskFile

**Files:**
- Create: `D:/LLM_wiki/.codex/codex_with_cc/tasks/20260627/019100-agent-v2a-spec-review.md`

- [ ] **Step 1: Write the task file**

Required task metadata:

```text
WorkflowId: agent-v2a-20260627
TaskId: agent-v2a-spec-review-001
Role: reviewer
ReviewForTaskId: agent-v2a-implementer-001
ReviewKind: spec
SessionKey: agent-v2a-main
DependsOn: agent-v2a-implementer-001
Scope: .
```

Required review focus:

- Check every Agent v2a acceptance item from the implementer TaskFile.
- Verify route types are exactly `simple_rag`, `evidence_required`, `table_or_metric`, `multi_source_compare`, and `needs_clarification`.
- Verify executor flow: route, RAG, verify, optional one retry, finalize.
- Verify clarification route skips RAG.
- Verify `route`, `warnings`, and step `metadata` are exposed.
- Verify `answer.verify` is registered.
- Verify `page_kind` and `page_label` propagate in citations.
- Verify tests named in the implementer report exist and exercise the behavior.
- Return `DONE_WITH_CONCERNS` or `FAIL` if any acceptance criterion is missing.

- [ ] **Step 2: Validate the task file**

Run:

```powershell
pwsh -NoProfile -File C:\Users\响睡觉\.codex\plugins\cache\aiskyhub\codex-with-cc\1.0.9\skills\codex-with-cc\windows_scripts\validate_delegate_task.ps1 -TaskFile .\.codex\codex_with_cc\tasks\20260627\019100-agent-v2a-spec-review.md -Role reviewer -ReviewForTaskId agent-v2a-implementer-001 -ReviewKind spec
```

Expected: validation passes.

### Task 4: Dispatch Spec Review Through codex-with-cc

**Files:**
- Read-only scope: `D:/LLM_wiki`

- [ ] **Step 1: Spawn a fresh Codex child thread**

Child thread requirements:

- Inherit parent model.
- Use reasoning effort `medium`.
- Use `fork_context=false`.
- The child thread must set `CODEX_CLAUDE_CHILD_THREAD=1`.

- [ ] **Step 2: Child thread invokes Claude Code delegate**

Run inside the child thread:

```powershell
$workflowRoot = 'C:\Users\响睡觉\.codex\plugins\cache\aiskyhub\codex-with-cc\1.0.9\skills\codex-with-cc'
$env:CODEX_CLAUDE_CHILD_THREAD = '1'
pwsh -NoProfile -File (Join-Path $workflowRoot 'windows_scripts\delegate_to_claude.ps1') `
  -TaskFile .\.codex\codex_with_cc\tasks\20260627\019100-agent-v2a-spec-review.md `
  -WorkflowId agent-v2a-20260627 `
  -TaskId agent-v2a-spec-review-001 `
  -Role reviewer `
  -ReviewForTaskId agent-v2a-implementer-001 `
  -ReviewKind spec `
  -SessionKey agent-v2a-main `
  -Scope . `
  -SessionMode PrimaryReuse `
  -BypassPermissions `
  -Tests "D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q" `
  -Tests "D:\Miniconda3\python.exe -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q"
```

Expected: reviewer report status is `DONE` or `DONE_WITH_CONCERNS` with concrete findings and verification evidence.

- [ ] **Step 3: Verify spec reviewer run artifact**

Run:

```powershell
$taskId = 'agent-v2a-spec-review-001'
$runId = Get-ChildItem -Path .\.codex\codex_with_cc\claude-delegate -Filter 'config_*.json' | ForEach-Object { $j = Get-Content -Path $_.FullName | ConvertFrom-Json; if ($j.workflowId -eq 'agent-v2a-20260627' -and $j.taskId -eq $taskId) { $j.runId } } | Select-Object -Last 1
if (-not $runId) { throw "No run id found for $taskId" }
D:\Miniconda3\python.exe C:\Users\响睡觉\.codex\plugins\cache\aiskyhub\codex-with-cc\1.0.9\skills\codex-with-cc\scripts\verify_delegate_run.py -RunId $runId
```

Expected: run verification passes.

### Task 5: Create Quality Review TaskFile

**Files:**
- Create: `D:/LLM_wiki/.codex/codex_with_cc/tasks/20260627/019200-agent-v2a-quality-review.md`

- [ ] **Step 1: Write the task file**

Required task metadata:

```text
WorkflowId: agent-v2a-20260627
TaskId: agent-v2a-quality-review-001
Role: reviewer
ReviewForTaskId: agent-v2a-implementer-001
ReviewKind: quality
SessionKey: agent-v2a-main
DependsOn: agent-v2a-spec-review-001
Scope: .
```

Required review focus:

- Maintainability of route policy and verifier code.
- Executor correctness under timeout, max steps, max tool calls, empty answer, tool error, and retry.
- Whether retry without a second verification is acceptable for online readiness.
- Whether any route keyword is too broad, especially single-letter or lowercase unit matching.
- Frontend XSS safety for Agent answer, citations, route, warnings, and step metadata.
- Backward compatibility of `/api/agent/query`.
- Test sufficiency and missing edge cases.

- [ ] **Step 2: Validate the task file**

Run:

```powershell
pwsh -NoProfile -File C:\Users\响睡觉\.codex\plugins\cache\aiskyhub\codex-with-cc\1.0.9\skills\codex-with-cc\windows_scripts\validate_delegate_task.ps1 -TaskFile .\.codex\codex_with_cc\tasks\20260627\019200-agent-v2a-quality-review.md -Role reviewer -ReviewForTaskId agent-v2a-implementer-001 -ReviewKind quality
```

Expected: validation passes.

### Task 6: Dispatch Quality Review Through codex-with-cc

**Files:**
- Read-only scope: `D:/LLM_wiki`

- [ ] **Step 1: Dispatch only after spec review has been read by Codex**

Proceed only if the spec review has no blocker. If spec review has a blocker, skip this task and go to Task 7.

- [ ] **Step 2: Child thread invokes Claude Code delegate**

Run inside the child thread:

```powershell
$workflowRoot = 'C:\Users\响睡觉\.codex\plugins\cache\aiskyhub\codex-with-cc\1.0.9\skills\codex-with-cc'
$env:CODEX_CLAUDE_CHILD_THREAD = '1'
pwsh -NoProfile -File (Join-Path $workflowRoot 'windows_scripts\delegate_to_claude.ps1') `
  -TaskFile .\.codex\codex_with_cc\tasks\20260627\019200-agent-v2a-quality-review.md `
  -WorkflowId agent-v2a-20260627 `
  -TaskId agent-v2a-quality-review-001 `
  -Role reviewer `
  -ReviewForTaskId agent-v2a-implementer-001 `
  -ReviewKind quality `
  -SessionKey agent-v2a-main `
  -Scope . `
  -SessionMode PrimaryReuse `
  -BypassPermissions `
  -Tests "D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q" `
  -Tests "D:\Miniconda3\python.exe -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q"
```

Expected: reviewer report status is `DONE` or `DONE_WITH_CONCERNS`; any blocker is clearly marked in `Findings`.

- [ ] **Step 3: Verify quality reviewer run artifact**

Run:

```powershell
$taskId = 'agent-v2a-quality-review-001'
$runId = Get-ChildItem -Path .\.codex\codex_with_cc\claude-delegate -Filter 'config_*.json' | ForEach-Object { $j = Get-Content -Path $_.FullName | ConvertFrom-Json; if ($j.workflowId -eq 'agent-v2a-20260627' -and $j.taskId -eq $taskId) { $j.runId } } | Select-Object -Last 1
if (-not $runId) { throw "No run id found for $taskId" }
D:\Miniconda3\python.exe C:\Users\响睡觉\.codex\plugins\cache\aiskyhub\codex-with-cc\1.0.9\skills\codex-with-cc\scripts\verify_delegate_run.py -RunId $runId
```

Expected: run verification passes.

### Task 7: Rework Loop If Any Blocker Exists

**Files:**
- Possible modify: `D:/LLM_wiki/src/app/services/agent_policy.py`
- Possible modify: `D:/LLM_wiki/src/app/services/answer_verifier.py`
- Possible modify: `D:/LLM_wiki/src/app/services/agent_executor.py`
- Possible modify: `D:/LLM_wiki/src/app/services/tool_registry.py`
- Possible modify: `D:/LLM_wiki/src/app/schemas/agent.py`
- Possible modify: `D:/LLM_wiki/src/app/static/index.html`
- Possible modify: `D:/LLM_wiki/tests/test_agent_policy.py`
- Possible modify: `D:/LLM_wiki/tests/test_answer_verifier.py`
- Possible modify: `D:/LLM_wiki/tests/test_agent_executor.py`
- Possible modify: `D:/LLM_wiki/tests/test_tool_registry.py`
- Possible modify: `D:/LLM_wiki/tests/test_agent_routes.py`

- [ ] **Step 1: Decide blocker status**

Treat these as blockers:

- Missing required route type.
- Incorrect route ordering that causes compare/table/evidence intent to be misclassified.
- Ordinary prose incorrectly routed to table mode because of broad unit matching.
- `needs_clarification` calls RAG.
- `max_tool_calls` or `max_steps` can be exceeded.
- `/api/agent/query` schema breaks existing request compatibility.
- Agent frontend renders server-controlled strings through dynamic HTML.
- `answer.verify` is absent from `ToolRegistry`.
- Local focused tests fail.

- [ ] **Step 2: Create a rework implementer TaskFile**

Task id sequence:

```text
agent-v2a-rework-001
agent-v2a-rework-002
agent-v2a-rework-003
```

Each rework task must include:

```text
WorkflowId: agent-v2a-20260627
Role: implementer
SessionKey: agent-v2a-main
DependsOn: agent-v2a-implementer-001
DependsOn: agent-v2a-spec-review-001
DependsOn: agent-v2a-quality-review-001
Scope: exact files needed for the blocker
```

Rework verification commands:

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q
D:\Miniconda3\python.exe -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q
```

- [ ] **Step 3: Repeat review gates for the rework task**

For each rework implementer task, create and dispatch:

```text
agent-v2a-rework-001-spec-review
agent-v2a-rework-001-quality-review
```

Do not proceed to final-verifier until the latest implementer task has accepted spec and quality review gates.

### Task 8: Final-Verifier Task

**Files:**
- Create: `D:/LLM_wiki/.codex/codex_with_cc/tasks/20260627/019900-agent-v2a-final-verifier.md`

- [ ] **Step 1: Write final-verifier TaskFile**

Required task metadata:

```text
WorkflowId: agent-v2a-20260627
TaskId: agent-v2a-final-verifier-001
Role: final-verifier
SessionKey: agent-v2a-main
DependsOn: agent-v2a-implementer-001
DependsOn: agent-v2a-spec-review-001
DependsOn: agent-v2a-quality-review-001
Scope: .
```

Final verifier must check:

- Implementer report has required headings.
- Spec and quality reviews exist and are accepted.
- Declared verification commands appear in reports with outcomes.
- No parallel writable scopes overlap.
- Residual risks are explicitly listed.
- Local focused and regression tests are sufficient for acceptance.
- Online readiness is either accepted or blocked with exact reasons.

- [ ] **Step 2: Validate final-verifier TaskFile**

Run:

```powershell
pwsh -NoProfile -File C:\Users\响睡觉\.codex\plugins\cache\aiskyhub\codex-with-cc\1.0.9\skills\codex-with-cc\windows_scripts\validate_delegate_task.ps1 -TaskFile .\.codex\codex_with_cc\tasks\20260627\019900-agent-v2a-final-verifier.md -Role final-verifier
```

Expected: validation passes.

- [ ] **Step 3: Dispatch final-verifier through child thread**

Run inside the child thread:

```powershell
$workflowRoot = 'C:\Users\响睡觉\.codex\plugins\cache\aiskyhub\codex-with-cc\1.0.9\skills\codex-with-cc'
$env:CODEX_CLAUDE_CHILD_THREAD = '1'
pwsh -NoProfile -File (Join-Path $workflowRoot 'windows_scripts\delegate_to_claude.ps1') `
  -TaskFile .\.codex\codex_with_cc\tasks\20260627\019900-agent-v2a-final-verifier.md `
  -WorkflowId agent-v2a-20260627 `
  -TaskId agent-v2a-final-verifier-001 `
  -Role final-verifier `
  -SessionKey agent-v2a-main `
  -Scope . `
  -SessionMode PrimaryReuse `
  -BypassPermissions `
  -Tests "D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q" `
  -Tests "D:\Miniconda3\python.exe -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q"
```

Expected: final-verifier returns `DONE` or `DONE_WITH_CONCERNS` without blockers.

### Task 9: Workflow Artifact Verification

**Files:**
- Inspect: `D:/LLM_wiki/.codex/codex_with_cc/claude-delegate/workflow_agent-v2a-20260627.json`

- [ ] **Step 1: Verify each new run**

Run once per review/final-verifier run id:

```powershell
$taskIds = @('agent-v2a-spec-review-001', 'agent-v2a-quality-review-001', 'agent-v2a-final-verifier-001')
foreach ($taskId in $taskIds) {
  $runId = Get-ChildItem -Path .\.codex\codex_with_cc\claude-delegate -Filter 'config_*.json' | ForEach-Object { $j = Get-Content -Path $_.FullName | ConvertFrom-Json; if ($j.workflowId -eq 'agent-v2a-20260627' -and $j.taskId -eq $taskId) { $j.runId } } | Select-Object -Last 1
  if (-not $runId) { throw "No run id found for $taskId" }
  D:\Miniconda3\python.exe C:\Users\响睡觉\.codex\plugins\cache\aiskyhub\codex-with-cc\1.0.9\skills\codex-with-cc\scripts\verify_delegate_run.py -RunId $runId
}
```

Expected: every run passes.

- [ ] **Step 2: Verify aggregate workflow**

Run:

```powershell
D:\Miniconda3\python.exe C:\Users\响睡觉\.codex\plugins\cache\aiskyhub\codex-with-cc\1.0.9\skills\codex-with-cc\scripts\verify_delegate_workflow.py -WorkflowId agent-v2a-20260627
```

Expected: workflow verification passes, including review gates and final-verifier gate.

### Task 10: Final Local Regression

**Files:**
- Test: `D:/LLM_wiki/tests`
- Inspect: `D:/LLM_wiki/src/app/static/index.html`

- [ ] **Step 1: Run focused Agent tests**

Run:

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run regression subset**

Run:

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run frontend dynamic HTML scan**

Run:

```powershell
rg -n "innerHTML\s*=.*(agent|step|answer|summary|error|data|result|payload|response|route|warning)|insertAdjacentHTML|eval\(" src\app\static\index.html
```

Expected: no Agent panel dynamic data issue.

### Task 11: Browser/UI Smoke

**Files:**
- Inspect visually: `D:/LLM_wiki/src/app/static/index.html`

- [ ] **Step 1: Start local API only if no existing usable server is active**

Run:

```powershell
try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health | Select-Object -ExpandProperty StatusCode } catch { "DOWN" }
```

Expected: `200` means reuse server; `DOWN` means start local server using the repository's normal command.

- [ ] **Step 2: Smoke `/api/agent/query`**

Run:

```powershell
$body = @{ project_slug = "internal-research"; query = "请总结当前知识库的核心主题"; session_id = "local-agent-v2a-smoke" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/agent/query -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 8
```

Expected response contains:

- `final_answer`
- `steps`
- `route`
- `warnings`
- `session_id`

- [ ] **Step 3: Use browser inspection for Agent panel**

Check:

- Route is visible.
- Warnings are visible when present.
- Verify/retry steps render without layout overlap.
- Citations render with page labels when present.
- No console error appears when querying.

### Task 12: Remote Sync, Backup, Restart, and Server Tests

**Files to sync:**
- `src/app/schemas/agent.py`
- `src/app/services/agent_policy.py`
- `src/app/services/answer_verifier.py`
- `src/app/services/agent_executor.py`
- `src/app/services/tool_registry.py`
- `src/app/static/index.html`
- `tests/test_agent_policy.py`
- `tests/test_answer_verifier.py`
- `tests/test_agent_executor.py`
- `tests/test_tool_registry.py`
- `tests/test_agent_routes.py`
- `docs/worklogs/2026-06-27-agent-layer.md`

- [ ] **Step 1: Create local sync archive**

Run:

```powershell
New-Item -ItemType Directory -Force -Path tmp | Out-Null
tar -czf tmp\agent-v2a-sync-20260627.tgz src/app/schemas/agent.py src/app/services/agent_policy.py src/app/services/answer_verifier.py src/app/services/agent_executor.py src/app/services/tool_registry.py src/app/static/index.html tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py docs/worklogs/2026-06-27-agent-layer.md
```

Expected: `tmp/agent-v2a-sync-20260627.tgz` exists.

- [ ] **Step 2: Upload archive**

Run:

```powershell
ssh -i D:/codex_ssh/llm_wiki_server_ed25519 -o IdentitiesOnly=yes -p 28294 zhangyh@ffsampling.a1.luyouxia.net "mkdir -p ~/llm_wiki_server/tmp"
scp -i D:/codex_ssh/llm_wiki_server_ed25519 -o IdentitiesOnly=yes -P 28294 tmp/agent-v2a-sync-20260627.tgz zhangyh@ffsampling.a1.luyouxia.net:~/llm_wiki_server/tmp/agent-v2a-sync-20260627.tgz
```

Expected: upload succeeds.

- [ ] **Step 3: Backup remote files before extraction**

Run:

```powershell
ssh -i D:/codex_ssh/llm_wiki_server_ed25519 -o IdentitiesOnly=yes -p 28294 zhangyh@ffsampling.a1.luyouxia.net "cd ~/llm_wiki_server && mkdir -p tmp/backup_agent_v2a_20260627 && tar -tzf tmp/agent-v2a-sync-20260627.tgz | while read f; do if [ -e \"$f\" ]; then mkdir -p \"tmp/backup_agent_v2a_20260627/$(dirname \"$f\")\"; cp -a \"$f\" \"tmp/backup_agent_v2a_20260627/$f\"; fi; done"
```

Expected: remote backup directory exists. Existing files are copied before replacement.

- [ ] **Step 4: Extract remote archive**

Run:

```powershell
ssh -i D:/codex_ssh/llm_wiki_server_ed25519 -o IdentitiesOnly=yes -p 28294 zhangyh@ffsampling.a1.luyouxia.net "cd ~/llm_wiki_server && tar -xzf tmp/agent-v2a-sync-20260627.tgz"
```

Expected: changed files are present on remote.

- [ ] **Step 5: Run remote focused tests**

Run:

```powershell
ssh -i D:/codex_ssh/llm_wiki_server_ed25519 -o IdentitiesOnly=yes -p 28294 zhangyh@ffsampling.a1.luyouxia.net "cd ~/llm_wiki_server && .venv/bin/python -m pytest tests/test_agent_policy.py tests/test_answer_verifier.py tests/test_agent_executor.py tests/test_tool_registry.py tests/test_agent_routes.py -q"
```

Expected: all tests pass.

- [ ] **Step 6: Run remote regression subset**

Run:

```powershell
ssh -i D:/codex_ssh/llm_wiki_server_ed25519 -o IdentitiesOnly=yes -p 28294 zhangyh@ffsampling.a1.luyouxia.net "cd ~/llm_wiki_server && .venv/bin/python -m pytest tests/test_conversation_memory.py tests/test_rag_adapter.py tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q"
```

Expected: all tests pass.

- [ ] **Step 7: Restart API and worker with existing scripts**

Run:

```powershell
ssh -i D:/codex_ssh/llm_wiki_server_ed25519 -o IdentitiesOnly=yes -p 28294 zhangyh@ffsampling.a1.luyouxia.net "cd ~/llm_wiki_server && pkill -f 'uvicorn app.main:app' || true; pkill -f 'python -m app.workers.runner' || true; ./scripts/start_api.sh; ./scripts/start_worker.sh; ./scripts/status.sh"
```

Expected: API and worker are running.

- [ ] **Step 8: Run remote health and Agent smoke**

Run:

```powershell
ssh -i D:/codex_ssh/llm_wiki_server_ed25519 -o IdentitiesOnly=yes -p 28294 zhangyh@ffsampling.a1.luyouxia.net "cd ~/llm_wiki_server && curl -fsS http://127.0.0.1:8000/api/health && curl -fsS -X POST http://127.0.0.1:8000/api/agent/query -H 'Content-Type: application/json' -d '{\"project_slug\":\"internal-research\",\"query\":\"请总结当前知识库的核心主题\",\"session_id\":\"remote-agent-v2a-smoke\"}'"
```

Expected: health succeeds and Agent response contains `final_answer`, `steps`, `route`, and `warnings`.

- [ ] **Step 9: Preserve `/api/query` baseline**

Run remote full30 gate only if Agent v2a touched shared RAG/query behavior or if smoke reveals quality drift. The command is:

```powershell
ssh -i D:/codex_ssh/llm_wiki_server_ed25519 -o IdentitiesOnly=yes -p 28294 zhangyh@ffsampling.a1.luyouxia.net "cd ~/llm_wiki_server && .venv/bin/python scripts/run_mineru_rag_loop.py --profile query --python .venv/bin/python --run-query-eval --base-url http://127.0.0.1:8000 --timeout 120 --min-query-passed 24 --max-query-failed 6 --require-failure-attribution --out-dir tmp/full30_query_gate_agent_v2a_20260627"
```

Expected: at least `24/30` pass, no timeout wave, and every failure has attribution.

### Task 13: Final Codex Acceptance Decision

**Files:**
- Update if accepted: `D:/LLM_wiki/docs/work.md`
- Update if accepted: `D:/LLM_wiki/docs/worklogs/2026-06-27-agent-layer.md`

- [ ] **Step 1: Decide status**

Accept only if:

- Review gates pass.
- Final-verifier passes.
- Workflow verifier passes.
- Local tests pass.
- Remote tests pass.
- Remote `/api/agent/query` smoke passes.

- [ ] **Step 2: If accepted, record final evidence**

Record:

- local focused test result
- local regression result
- spec reviewer run id
- quality reviewer run id
- final-verifier run id
- workflow verifier result
- remote focused test result
- remote regression result
- remote health result
- remote Agent smoke result
- full30 result if run

- [ ] **Step 3: If not accepted, start another loop**

Create the next implementer task with:

```text
TaskId: agent-v2a-rework-001
WorkflowId: agent-v2a-20260627
SessionKey: agent-v2a-main
DependsOn: latest failing review or verifier task
```

The loop stops only when Task 13 Step 1 passes or a blocker requires user product decision.
