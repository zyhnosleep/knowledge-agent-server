# Agent Attachment Concurrency Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Agent 临时附件查询的 SQLite 并发锁、跨项目引用污染、超时和 `attachment_id` 丢失问题，并在真实服务器上完成双会话并发验收。

**Architecture:** 在 Agent 开始长耗时检索或模型调用前，先提交会话和用户消息的短事务，避免 SQLite 写锁贯穿整个请求。对明确要求“只使用当前附件”的请求建立受限执行路径：只检索当前会话附件、只生成附件引用、跳过项目级 RAG，并继续使用现有 trace/finalize 结构。附件身份从检索证据一直传递到最终 Citation，保证前端可以准确定位文件。

**Tech Stack:** FastAPI、SQLAlchemy 2.x、SQLite、Pydantic、pytest、现有 AgentExecutor/RAGAdapter。

---

## Current Baseline

- `src/app/services/search.py` 和 `tests/test_query_service.py` 有尚未提交的第 4 阶段改动，服务器 full30 已达到 `28/30`，门槛为 `passed >= 27`、`failed <= 3` 且失败均有归因。
- 第 5 阶段真实双会话测试发现：一个请求约 95 秒后被标记为 timeout，另一个请求因 `sqlite3.OperationalError: database is locked` 返回 HTTP 500。
- 附件答案混入 7 条项目文档引用，附件引用中的 `attachment_id` 为 `null`。
- 本计划不恢复任何 Wiki 路径或 Wiki 数据结构。

## File Map

- Modify: `src/app/services/agent_executor.py`
  - 缩短初始写事务；识别附件限定请求；建立附件专用执行路径；传递附件引用身份。
- Modify: `src/app/services/session_attachments.py`
  - 在 `EvidenceItem` 中写入 `attachment_id`。
- Modify: `tests/test_agent_executor.py`
  - 覆盖初始事务提前提交，以及两个独立数据库会话并发执行时不锁库。
- Modify: `tests/test_session_attachments.py`
  - 覆盖附件 ID、附件限定查询、无项目引用和会话隔离。
- Verify only: `src/app/schemas/agent.py`, `src/app/schemas/common.py`
  - 两个 schema 已经包含 `attachment_id`，无需迁移或新增字段。
- Verify only: `src/app/api/agent_routes.py`
  - 保留请求结束时的最终 commit；Executor 内只增加早期事务边界。
- Verify only: `src/app/services/search.py`, `tests/test_query_service.py`
  - 保留并验证第 4 阶段已经完成的检索路由改动。

---

### Task 1: 固化第 4 阶段 RAG 基线

**Files:**
- Verify: `src/app/services/search.py`
- Verify: `tests/test_query_service.py`

- [ ] **Step 1: 检查工作区范围**

Run:

```powershell
git status --short
git diff -- src/app/services/search.py tests/test_query_service.py
```

Expected: 只看到已确认的科学术语归一化、论文归属路由和对应测试；不包含 Wiki 回退逻辑。

- [ ] **Step 2: 运行第 4 阶段聚焦测试**

Run:

```powershell
python -m pytest tests/test_query_service.py -q
```

Expected: 全部通过。

- [ ] **Step 3: 单独提交 RAG 基线**

```powershell
git add src/app/services/search.py tests/test_query_service.py
git commit -m "Improve scientific paper routing"
```

Expected: 第 4 阶段改动形成独立提交，后续并发修复不与其混合。

---

### Task 2: 保留附件身份到最终引用

**Files:**
- Modify: `src/app/services/session_attachments.py:213`
- Modify: `src/app/services/agent_executor.py:1178`
- Test: `tests/test_session_attachments.py:157`

- [ ] **Step 1: 写失败测试，要求 EvidenceItem 带附件 ID**

在 `test_retrieve_attachment_evidence_is_session_scoped` 的现有断言中增加：

```python
assert pack.items[0].attachment_id == "a1"
```

- [ ] **Step 2: 写失败测试，要求最终 Citation 带附件 ID**

在附件回答测试中增加：

```python
attachment_citations = [
    citation for citation in response.citations
    if citation.page_kind == "session_attachment"
]
assert attachment_citations
assert attachment_citations[0].attachment_id == "a1"
```

- [ ] **Step 3: 确认测试先失败**

Run:

```powershell
python -m pytest tests/test_session_attachments.py -q
```

Expected: 新增断言因 `attachment_id is None` 失败。

- [ ] **Step 4: 在检索证据中设置附件 ID**

在 `retrieve_session_attachment_evidence()` 构造 `EvidenceItem` 时加入：

```python
EvidenceItem(
    index=index,
    document_id=None,
    chunk_id=chunk.id,
    attachment_id=attachment.id,
    page_slug=None,
    page_title=attachment.title or attachment.file_name,
    page_kind="session_attachment",
    page_label=chunk.page_label,
    score=float(score),
    excerpt=chunk.text[:1000],
    evidence_kind="session_attachment",
    source_stage="session_attachment",
    support_hint="direct" if score > 0 else "contextual",
)
```

- [ ] **Step 5: 在最终 Citation 中传递附件 ID**

在 `_session_attachment_citations()` 构造 `Citation` 时加入：

```python
Citation(
    document_id=item.get("document_id"),
    chunk_id=item.get("chunk_id"),
    attachment_id=item.get("attachment_id"),
    page_slug=item.get("page_slug"),
    page_title=item.get("page_title"),
    page_kind=item.get("page_kind") or "session_attachment",
    score=float(item.get("score") or 0.0),
    page_label=item.get("page_label"),
    excerpt=excerpt,
)
```

- [ ] **Step 6: 运行测试并提交**

Run:

```powershell
python -m pytest tests/test_session_attachments.py -q
git add src/app/services/session_attachments.py src/app/services/agent_executor.py tests/test_session_attachments.py
git commit -m "Preserve session attachment citation identity"
```

Expected: 附件测试全部通过，提交只包含附件身份链路。

---

### Task 3: 缩短 Agent 初始 SQLite 写事务

**Files:**
- Modify: `src/app/services/agent_executor.py:89`
- Test: `tests/test_agent_executor.py`

- [ ] **Step 1: 写早期提交行为测试**

创建一个记录 `commit()` 调用的 SQLAlchemy Session 包装或 monkeypatch，在阻塞的 RAG stub 开始执行时断言已有一次 commit：

```python
def test_execute_commits_session_turn_before_rag(monkeypatch) -> None:
    db = make_db()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    commit_count = 0
    real_commit = db.commit

    def tracked_commit() -> None:
        nonlocal commit_count
        commit_count += 1
        real_commit()

    monkeypatch.setattr(db, "commit", tracked_commit)

    class InspectingRAG:
        def answer(self, db, project_slug, question, document_id=None):
            assert commit_count >= 1
            return QueryResponse(
                answer_markdown="ok",
                citations=[],
                verification_status="local-only",
            )

    executor = build_executor_with_rag(db, InspectingRAG())
    executor.execute(AgentQueryRequest(project_slug="demo", query="hello"))
```

测试辅助函数应复用现有 `ToolRegistry`、`ConversationMemory` 和 `AgentTraceStore` 的构造方式，不引入生产代码依赖。

- [ ] **Step 2: 写双连接并发回归测试**

使用 `tmp_path / "agent-concurrency.db"` 建立文件型 SQLite，引擎参数包含 `check_same_thread=False` 和较短 `timeout`。线程 A 在进入 RAG 后等待 Event；线程 B 使用独立 Session 执行另一个 session 的请求。断言线程 B 在 Event 释放前完成且没有 `OperationalError`：

```python
assert second_finished.wait(timeout=2.0)
assert errors == []
release_first.set()
first_thread.join(timeout=2.0)
second_thread.join(timeout=2.0)
```

- [ ] **Step 3: 确认测试先暴露当前锁持有问题**

Run:

```powershell
python -m pytest tests/test_agent_executor.py -q -k "commits_session_turn_before_rag or concurrent_sessions"
```

Expected: 当前实现没有在 RAG 调用前提交，至少一个新增测试失败。

- [ ] **Step 4: 在长操作前提交短事务**

在 `execute()` 完成 `touch_session()`、过期会话清理、用户 turn 写入和 compact 后，立即提交：

```python
self._memory.touch_session(
    session_id,
    project_slug=request.project_slug,
    ttl_days=settings.agent_conversation_ttl_days,
    document_id=request.document_id,
)
self._memory.purge_expired_sessions()
self._memory.add_turn(
    session_id, role="user", content=request.query, step_type="user_query"
)
self._compact_if_needed(session_id, constraints)
self._db.commit()
```

如果 commit 失败，执行 `rollback()` 后重新抛出异常；不要进入检索或模型阶段。请求末尾仍由 route 层提交工具 turn、最终 answer 和 trace。

- [ ] **Step 5: 运行聚焦测试并提交**

Run:

```powershell
python -m pytest tests/test_agent_executor.py -q -k "commit or concurrent or persists_turns or resumes_existing_session"
git add src/app/services/agent_executor.py tests/test_agent_executor.py
git commit -m "Release Agent session writes before long queries"
```

Expected: 并发测试稳定通过，历史记录和会话恢复测试不退化。

---

### Task 4: 建立附件限定查询路径

**Files:**
- Modify: `src/app/services/agent_executor.py:256`
- Test: `tests/test_session_attachments.py:220`

- [ ] **Step 1: 写意图识别参数化测试**

为私有 helper 增加覆盖，至少包含以下输入：

```python
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("只根据当前附件回答 token 和 color", True),
        ("请读取这个临时文件，不要引用项目文档", True),
        ("Answer using only the attachment", True),
        ("比较附件与项目文档", False),
        ("介绍项目中的 CHARMM36 文献", False),
    ],
)
def test_attachment_only_query_detection(query: str, expected: bool) -> None:
    assert AgentExecutor._is_attachment_only_query(query) is expected
```

识别规则必须同时满足“附件词”和“限定词”，避免普通提及附件时错误跳过项目 RAG。

- [ ] **Step 2: 写附件专用路径失败测试**

构造包含 `helios-314159`、`crimson` 的当前会话附件，并使用一个调用即抛错的 RAG stub：

```python
class ForbiddenProjectRAG:
    def answer(self, db, project_slug, question, document_id=None):
        raise AssertionError("project RAG must not run for attachment-only queries")
```

执行 `只根据当前附件回答 token 和 color` 后断言：

```python
assert response.status == "completed"
assert "helios-314159" in response.final_answer
assert "crimson" in response.final_answer
assert response.citations
assert all(c.page_kind == "session_attachment" for c in response.citations)
assert all(c.attachment_id == "alpha-attachment" for c in response.citations)
assert not any(step.tool_name == "rag.answer" for step in response.steps)
```

- [ ] **Step 3: 写混合查询保持原路径测试**

对“比较附件与项目文档”使用可计数的 RAG stub，断言项目 `rag.answer` 被调用，附件引用与项目引用均可出现。这样可以防止附件专用路径吞掉明确的跨来源比较需求。

- [ ] **Step 4: 确认测试先失败**

Run:

```powershell
python -m pytest tests/test_session_attachments.py -q -k "attachment_only or mixed_attachment"
```

Expected: 当前实现仍调用项目 RAG，附件专用测试失败。

- [ ] **Step 5: 实现严格的附件限定识别**

在 `AgentExecutor` 增加静态 helper，使用大小写不敏感的确定性规则：

```python
@staticmethod
def _is_attachment_only_query(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    has_attachment = any(
        term in normalized
        for term in ("附件", "临时文件", "attachment", "attached file")
    )
    has_exclusive_scope = any(
        term in normalized
        for term in (
            "只根据",
            "仅根据",
            "只使用",
            "仅使用",
            "不要引用项目",
            "only the attachment",
            "only attachment",
            "using only",
        )
    )
    is_comparison = any(term in normalized for term in ("比较", "对比", "compare"))
    return has_attachment and has_exclusive_scope and not is_comparison
```

- [ ] **Step 6: 调整检索顺序并短路项目 RAG**

在 Agent route step 后先检索当前会话附件。若 `_is_attachment_only_query(request.query)` 为真且附件 pack 非空：

```python
evidence_pack = session_attachment_pack.model_dump()
citations = self._session_attachment_citations(evidence_pack)
answer_text = self._draft_session_attachment_answer(request.query, citations)
tool_calls = usage.tool_calls
warnings.append("Answered only from temporary attachments scoped to this session.")
```

该分支不得调用 `_run_retrieve_evidence()`、`_run_rag_answer()` 或合并项目 citations。普通查询和明确比较查询继续走现有项目 RAG + 附件合并路径。附件限定但当前会话没有附件 evidence 时，保留现有普通路径，避免无依据地产生答案。

- [ ] **Step 7: 限制附件专用分支的后续合成**

附件专用分支直接使用确定性 extractive answer，并跳过可能再次引入项目上下文或远程超时的 synthesizer；仍执行现有 finalize、trace 和会话记录逻辑。增加 trace metadata：

```python
{
    "source_scope": "session_attachments_only",
    "project_rag_skipped": True,
}
```

- [ ] **Step 8: 运行测试并提交**

Run:

```powershell
python -m pytest tests/test_session_attachments.py tests/test_agent_executor.py -q
git add src/app/services/agent_executor.py tests/test_session_attachments.py tests/test_agent_executor.py
git commit -m "Isolate attachment-only Agent queries"
```

Expected: 附件限定、混合查询、会话历史、trace 和通用 Agent 测试全部通过。

---

### Task 5: 本地完整回归

**Files:**
- Verify: all source and tests

- [ ] **Step 1: 运行核心服务测试**

Run:

```powershell
python -m pytest tests/test_session_attachments.py tests/test_agent_executor.py tests/test_agent_routes.py tests/test_conversation_memory.py tests/test_query_service.py -q
```

Expected: 全部通过，无数据库锁、无超时测试失败。

- [ ] **Step 2: 运行完整测试套件**

Run:

```powershell
python -m pytest -q
```

Expected: 全部通过；如存在与本次无关的既有失败，必须记录完整 case 和原因，不能静默忽略。

- [ ] **Step 3: 检查差异和提交历史**

Run:

```powershell
git status --short
git diff --check
git log -6 --oneline
```

Expected: `git diff --check` 无输出；计划内代码均已提交；无 Wiki 文件或 Wiki 路由重新出现。

---

### Task 6: 同步服务器并完成真实双会话验收

**Files:**
- Deploy: committed source and tests
- Verify artifact: `~/llm_wiki_server/logs/api.log`

- [ ] **Step 1: 推送当前 main**

Run:

```powershell
git push origin main
```

Expected: GitHub `main` 指向本轮最后一个提交。

- [ ] **Step 2: 同步服务器代码并重启 API**

按照 `docs/deploy-no-docker.md` 的现有非 Docker 流程，在 `~/llm_wiki_server` 拉取对应提交、安装必要依赖并重启 `llm-wiki-server`。重启后检查：

```bash
curl -fsS http://127.0.0.1:8000/api/health
```

Expected: HTTP 200，服务状态正常。

- [ ] **Step 3: 重新上传两个隔离附件**

Session `concurrency-alpha` 上传：

```text
token: helios-314159
color: crimson
```

Session `concurrency-beta` 上传：

```text
token: selene-271828
color: cobalt
```

两个附件必须位于不同 session；project 可以保持之前的 `internal-research` 与 `agent`。

- [ ] **Step 4: 同时发送两个附件限定请求**

并发调用 `/api/agent/query`，问题统一采用明确限定语义，例如：

```text
只根据当前会话的临时附件回答：token 和 color 分别是什么？不要引用项目文档。
```

记录每个请求的 HTTP 状态、耗时、response status、答案、citations 和 steps。

- [ ] **Step 5: 执行硬性验收**

两个响应必须同时满足：

```text
HTTP status = 200
response.status = completed
elapsed < server timeout
alpha answer contains helios-314159 and crimson
beta answer contains selene-271828 and cobalt
no answer contains the other session's token or color
all citations.page_kind = session_attachment
all citations.attachment_id are non-null and belong to the current session
no step.tool_name = rag.answer
```

- [ ] **Step 6: 检查服务器日志**

Run on server:

```bash
grep -nE "database is locked|OperationalError|timeout|Traceback" ~/llm_wiki_server/logs/api.log | tail -100
```

Expected: 本轮请求时间段没有 `database is locked`、HTTP 500 traceback 或 Agent timeout。

---

### Task 7: 第 4 阶段质量收尾与最终记录

**Files:**
- Verify: `src/app/services/search.py`
- Verify: `tests/test_query_service.py`
- Update: `docs/work.md`

- [ ] **Step 1: 重新运行远端 full30 gate**

Run on server:

```bash
.venv/bin/python scripts/run_mineru_rag_loop.py \
  --profile query \
  --python .venv/bin/python \
  --run-query-eval \
  --base-url http://127.0.0.1:8000 \
  --timeout 120 \
  --min-query-passed 27 \
  --max-query-failed 3 \
  --require-failure-attribution \
  --out-dir tmp/full30_post_concurrency_20260713
```

Expected: 至少 `27/30`，失败不超过 3 个，所有失败有 attribution，且无 citation/source contamination。

- [ ] **Step 2: 判断是否继续优化剩余两条答案术语**

若仍为 `charmm36_overview` 缺 `CHARMM22`、`ff99sb_disp_overview` 缺 `London dispersion`，将其记录为 answer-stage completeness backlog。只有能通过通用、证据支持的科学锚点补全机制修复时才继续；不得按 benchmark case ID 或固定答案硬编码。

- [ ] **Step 3: 更新项目工作记录**

在 `docs/work.md` 记录：

```text
- full30 最终通过/失败数量和 artifact 路径
- 双会话并发请求的耗时与状态
- 两个 session 的引用隔离结果
- attachment_id 完整性结果
- 服务器 commit SHA 与重启时间
```

- [ ] **Step 4: 提交记录并推送**

```powershell
git add docs/work.md
git commit -m "Record Agent concurrency acceptance"
git push origin main
```

Expected: GitHub、服务器代码和工作记录指向同一已验收版本。

---

## Final Acceptance Gate

只有以下条件全部满足，才把第 5 阶段标记为完成：

- 两个真实并发请求均 HTTP 200 且 `status=completed`。
- 无 `database is locked`、超时或 HTTP 500。
- 每个答案只包含本 session 的 token 和 color。
- 附件限定查询没有项目文档引用，没有跨 session 引用。
- 所有附件 Citation 的 `attachment_id` 非空且正确。
- 本地完整测试套件通过。
- 远端 full30 保持 `>=27/30` 且所有失败有归因。
- GitHub、服务器和工作记录的 commit SHA 一致。

## Self-Review

- Spec coverage: 数据库锁、超时、引用污染、附件 ID、会话隔离、full30 回归、部署与服务器验收均有对应任务。
- Scope: 不重构 ORM/数据库模型，不恢复 Wiki，不改变普通项目 RAG 的行为。
- Type consistency: `EvidenceItem.attachment_id` 和 `Citation.attachment_id` 均已存在，计划只补齐值传递。
- Risk control: 附件专用分支要求“附件词 + 限定词”，比较类问题明确保留混合检索路径。
