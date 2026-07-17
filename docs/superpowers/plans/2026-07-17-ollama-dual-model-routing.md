# Ollama Dual-Model Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将知识库问答改造成双 GPU、自动/快速/深度模型路由、会话级模式、真正 token 流式输出和可见排队的 Ollama 推理链路。

**Architecture:** 保留现有 Agent、RAG、数据库和文档存储，在 Ollama 客户端之上增加明确的 fast/deep/embedding 端点与确定性 `AgentModelRouter`。同步和 SSE 两条 API 共享同一执行器；SSE 通过线程安全事件桥实时传递 route、queue、step、token、citation 和 final。会话数据库保存用户选择，应用层模型租约管理器负责有界并发和取消。

**Tech Stack:** FastAPI、Pydantic 2、SQLAlchemy 2、Alembic、httpx、Ollama API、SSE、原生 JavaScript、pytest。

---

## File responsibility map

- `src/app/core/config.py`: fast/deep/embedding Ollama 配置、上下文和并发容量。
- `src/app/services/ai.py`: 可注入端点的 Ollama 客户端、普通与流式文本生成、Embedding 独立端点。
- `src/app/services/agent_model_router.py`: 回答模式、路由到模型的纯确定性决策。
- `src/app/services/model_runtime.py`: 模型租约、队列位置、取消和运行快照。
- `src/app/services/agent_synthesizer.py`: 证据提示词、Markdown 流式综合和引用索引清洗。
- `src/app/services/agent_executor.py`: 会话模式解析、路由选择、实时事件发射与 trace 元数据。
- `src/app/services/conversation_memory.py`: 会话回答模式读取和更新。
- `src/app/schemas/agent.py`: 请求、会话和流式元数据契约。
- `src/app/models/records.py`: `ConversationSession.answer_mode` 持久化字段。
- `src/app/api/agent_routes.py`: SSE 事件桥、断开取消和模式恢复。
- `src/app/api/routes.py`: 推理就绪状态。
- `src/app/static/index.html`: 模式选择、流式 token、队列选择与停止按钮。
- `src/app/db/alembic/versions/*_add_conversation_answer_mode.py`: PostgreSQL 迁移。
- `src/app/db/session.py`: SQLite 兼容迁移。
- `deploy/systemd/llm-wiki-ollama-*.service`: 双 Ollama 用户服务模板。
- `scripts/benchmark_agent_models.py`: 可重复的 fast/deep 延迟和路由基准。

### Task 1: Add explicit Ollama profiles and deterministic model routing

**Files:**
- Modify: `src/app/core/config.py`
- Create: `src/app/services/agent_model_router.py`
- Modify: `.env.example`
- Modify: `.env.server.example`
- Test: `tests/test_agent_model_router.py`
- Test: `tests/test_model_constraints.py`

- [ ] **Step 1: Write failing profile and routing tests**

```python
from app.services.agent_model_router import AgentModelRouter


def test_auto_routes_simple_to_fast():
    target = AgentModelRouter().select("auto", "simple_rag")
    assert target.profile == "fast"
    assert target.context_length == 16384


def test_auto_routes_complex_to_deep():
    for route in ("table_or_metric", "multi_source_compare", "complex_multi_hop"):
        target = AgentModelRouter().select("auto", route)
        assert target.profile == "deep"
        assert target.context_length == 32768


def test_manual_modes_override_route():
    router = AgentModelRouter()
    assert router.select("fast", "complex_multi_hop").profile == "fast"
    assert router.select("deep", "simple_rag").profile == "deep"
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_agent_model_router.py tests/test_model_constraints.py`

Expected: collection fails because `agent_model_router` and the new Settings fields do not exist.

- [ ] **Step 3: Add configuration fields and the pure router**

Add Settings aliases with backward-compatible defaults:

```python
ollama_fast_base_url: str = Field(default="http://localhost:11435", alias="OLLAMA_FAST_BASE_URL")
ollama_deep_base_url: str = Field(default="http://localhost:11436", alias="OLLAMA_DEEP_BASE_URL")
ollama_embedding_base_url: str = Field(default="http://localhost:11435", alias="OLLAMA_EMBEDDING_BASE_URL")
ollama_fast_model: str = Field(default="qwen3:14b", alias="OLLAMA_FAST_MODEL")
ollama_deep_model: str = Field(default="qwen3.6:27b", alias="OLLAMA_DEEP_MODEL")
ollama_fast_context_length: int = Field(default=16384, alias="OLLAMA_FAST_CONTEXT_LENGTH")
ollama_deep_context_length: int = Field(default=32768, alias="OLLAMA_DEEP_CONTEXT_LENGTH")
ollama_fast_parallelism: int = Field(default=1, alias="OLLAMA_FAST_PARALLELISM")
ollama_deep_parallelism: int = Field(default=1, alias="OLLAMA_DEEP_PARALLELISM")
```

Create immutable `InferenceTarget(profile, base_url, model, context_length)` and `AgentModelRouter.select(answer_mode, route)`; map `simple_rag` and `evidence_required` to fast, and the three complex routes to deep. Manual modes always override automatic routing.

- [ ] **Step 4: Document all aliases in both env examples**

Use `-1` for `OLLAMA_KEEP_ALIVE`, set fast/deep contexts explicitly, and keep `OLLAMA_EMBEDDING_MODEL=qwen3-embedding:8b` independent of both generation models.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `pytest -q tests/test_agent_model_router.py tests/test_model_constraints.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/app/core/config.py src/app/services/agent_model_router.py .env.example .env.server.example tests/test_agent_model_router.py tests/test_model_constraints.py
git commit -m "Add dual Ollama inference profiles"
```

### Task 2: Separate generation and embedding clients

**Files:**
- Modify: `src/app/services/ai.py`
- Modify: `src/app/services/search.py`
- Modify: `src/app/services/pipeline.py`
- Test: `tests/test_ai_client.py`
- Test: `tests/test_query_service.py`

- [ ] **Step 1: Write failing client tests**

```python
def test_client_uses_injected_base_url(monkeypatch):
    client = OllamaClient(base_url="http://127.0.0.1:11436")
    assert client.base_url == "http://127.0.0.1:11436"


def test_stream_chat_sends_context_and_disables_thinking(monkeypatch):
    client = OllamaClient(base_url="http://deep")
    chunks = list(client.stream_chat(
        messages=[{"role": "user", "content": "answer"}],
        model="deep-model",
        context_length=32768,
    ))
    assert captured_payload["stream"] is True
    assert captured_payload["think"] is False
    assert captured_payload["options"]["num_ctx"] == 32768
```

Also assert `embed()` posts only to `OLLAMA_EMBEDDING_BASE_URL` with `OLLAMA_EMBEDDING_MODEL`, regardless of selected answer mode.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_ai_client.py tests/test_query_service.py`

Expected: constructor and `stream_chat` signature failures.

- [ ] **Step 3: Implement injectable clients and streaming NDJSON parsing**

`OllamaClient.__init__` accepts optional `base_url`; `generate_chat` returns final content plus Ollama timing metadata; `stream_chat` uses `httpx.Client.stream`, parses each NDJSON line, yields content deltas, and returns/records the terminal metadata. Both send `think=False`, `keep_alive=-1`, and explicit `num_ctx`.

Create one Embedding client configured with the embedding URL. Existing parser and pipeline batch calls retain their configured batch model behavior; only query-time generation is routed in later tasks.

- [ ] **Step 4: Preserve cancellation**

Accept `cancel_event: threading.Event | None`; before each read, terminate iteration when set and close the HTTP stream.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `pytest -q tests/test_ai_client.py tests/test_query_service.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/app/services/ai.py src/app/services/search.py src/app/services/pipeline.py tests/test_ai_client.py tests/test_query_service.py
git commit -m "Separate Ollama generation and embedding clients"
```

### Task 3: Persist session answer mode and expose it in APIs

**Files:**
- Modify: `src/app/models/records.py`
- Modify: `src/app/schemas/agent.py`
- Modify: `src/app/services/conversation_memory.py`
- Modify: `src/app/api/agent_routes.py`
- Modify: `src/app/db/session.py`
- Create: `src/app/db/alembic/versions/f5c2d8e41a7b_add_conversation_answer_mode.py`
- Test: `tests/test_conversation_memory.py`
- Test: `tests/test_agent_routes.py`
- Test: `tests/test_migrations.py`

- [ ] **Step 1: Write failing persistence and API tests**

```python
def test_new_session_defaults_to_auto(db):
    memory = ConversationMemory(db)
    memory.touch_session("s1", project_slug="demo", ttl_days=7)
    assert db.get(ConversationSession, "s1").answer_mode == "auto"


def test_requested_mode_updates_existing_session(db):
    memory = ConversationMemory(db)
    memory.touch_session("s1", project_slug="demo", ttl_days=7, answer_mode="deep")
    assert memory.get_answer_mode("s1") == "deep"
```

Assert `AgentQueryRequest.answer_mode` accepts only `auto|fast|deep|None`, and `AgentSessionRead.answer_mode` is returned from `/api/agent/sessions`.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_conversation_memory.py tests/test_agent_routes.py tests/test_migrations.py`

Expected: missing field and method failures.

- [ ] **Step 3: Add schema/model/memory behavior**

Use `Literal["auto", "fast", "deep"]`. A request with `answer_mode=None` reuses the stored session value; an explicitly supplied value updates that session. A newly created session defaults to `auto`.

- [ ] **Step 4: Add PostgreSQL and SQLite migrations**

Create revision `f5c2d8e41a7b` with `down_revision = "9bd042a0f5ba"`. Alembic upgrade adds non-null `VARCHAR(16)` with server default `auto`; downgrade drops it. SQLite compatibility migration adds the column through the existing additive migration table in `src/app/db/session.py`.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `pytest -q tests/test_conversation_memory.py tests/test_agent_routes.py tests/test_migrations.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/app/models/records.py src/app/schemas/agent.py src/app/services/conversation_memory.py src/app/api/agent_routes.py src/app/db/session.py src/app/db/alembic/versions tests/test_conversation_memory.py tests/test_agent_routes.py tests/test_migrations.py
git commit -m "Persist Agent answer mode per session"
```

### Task 4: Route synthesis and remove JSON double generation

**Files:**
- Modify: `src/app/services/agent_synthesizer.py`
- Modify: `src/app/services/agent_executor.py`
- Modify: `src/app/services/tool_registry.py`
- Test: `tests/test_agent_synthesizer.py`
- Test: `tests/test_agent_executor.py`
- Test: `tests/test_tool_registry.py`

- [ ] **Step 1: Write failing synthesis routing tests**

```python
def test_fast_target_calls_14b_plain_markdown():
    result = synthesizer.synthesize(..., target=fast_target)
    assert fake_client.calls[0]["model"] == "qwen3:14b"
    assert fake_client.calls[0]["context_length"] == 16384
    assert result["model"] == "qwen3:14b"


def test_markdown_citation_indexes_are_sanitized():
    fake_client.text = "结论 [0][7][0]"
    result = synthesizer.synthesize(..., citations=[citation])
    assert result["cited_indexes"] == [0]
```

Assert no structured-schema request or retry occurs for final synthesis.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_agent_synthesizer.py tests/test_agent_executor.py tests/test_tool_registry.py`

Expected: missing `target` and plain-chat behavior failures.

- [ ] **Step 3: Implement evidence-only Markdown synthesis**

Build one prompt containing bounded evidence, ask for inline `[n]` citations, call `generate_chat`, and extract valid indexes with `re.findall(r"\[(\d+)\]")`. Preserve the evidence fallback only for explicit model failure or empty content, and include a warning.

- [ ] **Step 4: Resolve target after policy routing**

The executor resolves the stored/requested answer mode, calls `AgentModelRouter.select`, records requested mode, actual profile, model, context and reason in the route/synthesis trace, and passes the same target to the synthesizer. Inject the configured synthesizer instead of constructing a second hidden instance in `ToolRegistry`.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `pytest -q tests/test_agent_synthesizer.py tests/test_agent_executor.py tests/test_tool_registry.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/app/services/agent_synthesizer.py src/app/services/agent_executor.py src/app/services/tool_registry.py tests/test_agent_synthesizer.py tests/test_agent_executor.py tests/test_tool_registry.py
git commit -m "Route Agent synthesis across fast and deep models"
```

### Task 5: Add model leases, queue events, and cancellation

**Files:**
- Create: `src/app/services/model_runtime.py`
- Modify: `src/app/services/agent_executor.py`
- Test: `tests/test_model_runtime.py`
- Test: `tests/test_agent_executor.py`

- [ ] **Step 1: Write failing queue tests**

Test FIFO order, `queue` callback positions, cancellation before acquisition, lease release on exception, and independent fast/deep capacities.

```python
def test_cancelled_waiter_never_acquires():
    runtime = ModelRuntime({"deep": 1})
    with runtime.acquire("deep"):
        cancel.set()
        with pytest.raises(ModelRequestCancelled):
            runtime.acquire("deep", cancel_event=cancel)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_model_runtime.py tests/test_agent_executor.py`

Expected: missing runtime module.

- [ ] **Step 3: Implement a process-wide FIFO runtime**

Use `threading.Condition` with per-profile active counts and ticket queues. `acquire()` accepts `on_queue(position)` and `cancel_event`; its context manager always releases and notifies. Expose a read-only snapshot for health reporting.

- [ ] **Step 4: Wrap only model generation in a lease**

Retrieval and evidence preparation remain concurrent. Acquire the selected profile immediately before final generation. Emit a queue event when waiting. Never change model automatically.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `pytest -q tests/test_model_runtime.py tests/test_agent_executor.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/app/services/model_runtime.py src/app/services/agent_executor.py tests/test_model_runtime.py tests/test_agent_executor.py
git commit -m "Add cancellable Agent model queues"
```

### Task 6: Stream live executor and model events over SSE

**Files:**
- Modify: `src/app/services/agent_synthesizer.py`
- Modify: `src/app/services/agent_executor.py`
- Modify: `src/app/api/agent_routes.py`
- Test: `tests/test_agent_streaming.py`
- Test: `tests/test_agent_routes.py`

- [ ] **Step 1: Write failing live-event tests**

Assert event order includes `start`, `route`, `step`, multiple `token`, `citation`, `final`, `done`; assert a token is available before the executor thread finishes; assert disconnect sets cancellation and no final event is emitted after cancellation.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_agent_streaming.py tests/test_agent_routes.py`

Expected: current endpoint emits steps only after full execution and has no token event.

- [ ] **Step 3: Add executor event sink and streaming synthesis**

`AgentExecutor` accepts `event_sink(event_name, data)` and `cancel_event`. Emit route and each completed step immediately. `AgentSynthesizer.synthesize_stream` forwards model deltas through `token`, assembles final Markdown, sanitizes citations, emits `citation`, and returns the same result shape as synchronous synthesis.

- [ ] **Step 4: Replace `asyncio.to_thread` blocking wait with an event bridge**

Create an `asyncio.Queue`; the worker thread calls `loop.call_soon_threadsafe(queue.put_nowait, event)`. The async generator waits with the configured heartbeat timeout, yields SSE events as they arrive, and watches `request.is_disconnected()`. On disconnect, set cancellation and drain/finish the worker safely.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `pytest -q tests/test_agent_streaming.py tests/test_agent_routes.py`

Expected: all tests pass with live token ordering.

- [ ] **Step 6: Commit**

```bash
git add src/app/services/agent_synthesizer.py src/app/services/agent_executor.py src/app/api/agent_routes.py tests/test_agent_streaming.py tests/test_agent_routes.py
git commit -m "Stream live Agent tokens and progress events"
```

### Task 7: Add frontend mode, streaming, queue switch, and stop controls

**Files:**
- Modify: `src/app/static/index.html`
- Test: `tests/test_static_frontend.py`

- [ ] **Step 1: Write failing frontend contract tests**

Assert the page contains accessible `auto/fast/deep` controls, sends `answer_mode`, handles `route/queue/token/citation`, uses `AbortController`, displays the actual model, offers “继续等待” and “切换快速”, and restores `answer_mode` from session data.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_static_frontend.py`

Expected: missing mode and token handling assertions fail.

- [ ] **Step 3: Add the session mode control**

Use a keyboard-accessible segmented control with `aria-pressed`. Keep an in-memory mode tied to `activeSessionId`; session list responses are authoritative. New sessions initialize `auto`.

- [ ] **Step 4: Render incremental events**

Append `token` deltas to one current assistant message, update phase labels for route/step/queue, and replace provisional citations after `citation/final`. Show actual profile and model beside the answer.

- [ ] **Step 5: Add queue choice and cancellation**

On `queue`, render “继续等待 27B” and “立即切换快速回答”. The switch action aborts the queued request and resubmits the same question with `answer_mode=fast`. “停止生成” aborts without automatic retry and labels preserved text incomplete.

- [ ] **Step 6: Run tests and verify GREEN**

Run: `pytest -q tests/test_static_frontend.py`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/app/static/index.html tests/test_static_frontend.py
git commit -m "Add Agent answer modes and live streaming UI"
```

### Task 8: Add readiness, performance metadata, and deployment templates

**Files:**
- Modify: `src/app/schemas/common.py`
- Modify: `src/app/api/routes.py`
- Modify: `src/app/services/agent_trace_store.py`
- Create: `src/app/services/model_readiness.py`
- Create: `deploy/systemd/llm-wiki-ollama-fast.service`
- Create: `deploy/systemd/llm-wiki-ollama-deep.service`
- Modify: `deploy/internal-pilot.md`
- Create: `scripts/benchmark_agent_models.py`
- Test: `tests/test_api_routes.py`
- Test: `tests/test_agent_trace_store.py`
- Test: `tests/test_deployment_config.py`

- [ ] **Step 1: Write failing readiness and trace tests**

Assert health distinguishes API health from model readiness; reports fast/deep/embedding model status, context and queue snapshot; trace preserves queue time, first-token time, prompt/eval token counts and throughput without secrets.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_api_routes.py tests/test_agent_trace_store.py tests/test_deployment_config.py`

Expected: missing readiness fields and service templates.

- [ ] **Step 3: Implement bounded readiness checks**

Query `/api/tags` and `/api/ps` on both Ollama instances with a short timeout. Cache the result briefly so `/api/health` does not load models or block normal traffic. Return `degraded` when a required model is missing or not prewarmed.

- [ ] **Step 4: Persist performance metadata**

Store timing in trace step metadata and expose it through existing trace APIs. Do not log prompts, credentials, cookies or service environment.

- [ ] **Step 5: Add deployment services and benchmark script**

Fast service uses GPU 0, port 11435, Flash Attention, q8 KV and permanent keep-alive. Deep service uses GPU 1 and port 11436 with the same runtime flags. Benchmark script runs fixed fast/deep requests and reports TTFT, total latency, tokens/s, selected model and citation count as JSON.

- [ ] **Step 6: Run tests and verify GREEN**

Run: `pytest -q tests/test_api_routes.py tests/test_agent_trace_store.py tests/test_deployment_config.py`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/app/schemas/common.py src/app/api/routes.py src/app/services/agent_trace_store.py src/app/services/model_readiness.py deploy scripts/benchmark_agent_models.py tests/test_api_routes.py tests/test_agent_trace_store.py tests/test_deployment_config.py
git commit -m "Add Ollama readiness and performance observability"
```

### Task 9: Full verification, server rollout, and acceptance

**Files:**
- Modify only if verification exposes a tested defect.

- [ ] **Step 1: Run focused backend and frontend suites**

```powershell
$env:PYTHONPATH='.;src'
pytest -q tests/test_agent_model_router.py tests/test_ai_client.py tests/test_conversation_memory.py tests/test_agent_synthesizer.py tests/test_model_runtime.py tests/test_agent_executor.py tests/test_agent_streaming.py tests/test_agent_routes.py tests/test_static_frontend.py tests/test_api_routes.py
```

Expected: zero failures.

- [ ] **Step 2: Run the complete suite**

Run: `$env:PYTHONPATH='.;src'; pytest -q`

Expected: zero failures.

- [ ] **Step 3: Verify repository quality**

Run: `git diff --check` and confirm only intended files changed.

- [ ] **Step 4: Push commits and migrate the pilot database**

Push `codex/internal-pilot`, back up the pilot database, run Alembic upgrade, and verify the new `conversation_sessions.answer_mode` column without modifying document paths or raw files.

- [ ] **Step 5: Install and start both Ollama user services**

Stop the temporary 11435/11436 processes, install the reviewed user units, start both, preload the three models, and verify `ollama ps` reports full GPU placement with 16K/32K contexts.

- [ ] **Step 6: Restart API and run smoke checks**

Verify health, auto/fast/deep routes, session restoration, live SSE token events, cancellation, deep queue switching, PDF citations, and that new uploads remain under `runtime/data/raw/<project>/`.

- [ ] **Step 7: Run performance and five-user acceptance**

Use `scripts/benchmark_agent_models.py` plus five concurrent public-gateway requests. Record TTFT/P50/P95, GPU memory, OOM count, queue behavior, citation validity and actual model. Compare with the 290-second baseline.

- [ ] **Step 8: Commit any evidence-only documentation and report measured gaps**

Do not claim the 30–60/60–90 targets unless the measured results satisfy them. If a target is missed, leave the verified functional rollout in place only when stable and report the exact bottleneck and next parameter experiment.
