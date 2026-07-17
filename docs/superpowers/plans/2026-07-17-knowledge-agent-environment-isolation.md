# Knowledge Agent Environment Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前双模型单测试栈改造成同一服务器上的开发/测试双栈，两个环境分别固定使用一张 GPU、`qwen3.5:9b` 和 `qwen3-embedding:4b`，保留测试数据与公网入口，并把活动产品命名统一为 Knowledge Agent。

**Architecture:** 应用层只保留一个 `generation` 推理档位和一个独立 Embedding 客户端；开发、测试分别运行独立 API/Ollama/Systemd 服务、PostgreSQL 数据库和数据目录。先完成代码与迁移工具，在 GPU 0 部署空开发环境验证，再对测试环境执行快照、2560 维重建、固定问题集验收和人工切换。

**Tech Stack:** Python 3.12、FastAPI、SQLAlchemy、Alembic、PostgreSQL、pgvector、Ollama、Qwen3.5 9B、Qwen3 Embedding 4B、Systemd user services、Caddy/Cloudflare Tunnel、pytest。

---

## 文件结构与职责

- `src/app/core/config.py`：统一生成模型、Embedding、上下文和单队列配置；默认产品名改为 Knowledge Agent。
- `src/app/services/agent_model_router.py`：收敛为确定性的单生成目标，不再根据问题或用户模式选择模型。
- `src/app/services/model_runtime.py`：只管理 `generation` FIFO 队列。
- `src/app/services/model_readiness.py`：只探测当前环境的生成模型和 Embedding 模型。
- `src/app/schemas/agent.py`、`src/app/services/conversation_memory.py`、`src/app/models/records.py`、`src/app/api/agent_routes.py`：删除回答模式 API 与持久化行为。
- `src/app/static/index.html`：删除模式选择及切换逻辑，统一显示 Knowledge Agent。
- `src/app/db/alembic/versions/*_single_model_and_2560_vectors.py`：删除会话模式列并把 pgvector 索引表重建为 2560 维。
- `scripts/reindex_embeddings.py`：在维护模式下对现有 chunk 重新生成 JSON embedding 与 pgvector 行，支持校验和失败退出。
- `scripts/build_retrieval_acceptance.py`：保存固定问题集并比较迁移前后 Top-K 召回、引用与延迟。
- `deploy/systemd/knowledge-agent-*.service`：开发/测试 API、Ollama 和测试 Tunnel 服务模板。
- `.env.development.example`、`.env.test.example`：两套互斥的运行配置模板。
- `tests/`：覆盖单模型路由、API 兼容边界、2560 维迁移、重建脚本、品牌和双栈部署配置。

### Task 1: 收敛为单生成模型配置

**Files:**
- Modify: `src/app/core/config.py`
- Modify: `src/app/services/agent_model_router.py`
- Modify: `src/app/services/model_runtime.py`
- Modify: `src/app/services/agent_executor.py`
- Modify: `tests/test_model_constraints.py`
- Modify: `tests/test_agent_model_router.py`
- Modify: `tests/test_model_runtime.py`
- Modify: `tests/test_agent_executor.py`

- [ ] **Step 1: 写单模型配置和路由失败测试**

在 `tests/test_model_constraints.py` 断言：

```python
def test_default_generation_stack_is_single_qwen_profile() -> None:
    settings = Settings()
    assert settings.ollama_generation_model == "qwen3.5:9b"
    assert settings.ollama_generation_context_length == 32768
    assert settings.ollama_generation_parallelism == 1
    assert settings.ollama_embedding_model == "qwen3-embedding:4b"
    assert settings.ollama_embedding_dimensions == 2560
    assert not hasattr(settings, "ollama_fast_model")
    assert not hasattr(settings, "ollama_deep_model")
```

在 `tests/test_agent_model_router.py` 断言所有非澄清路由返回同一个 `generation` 目标，`needs_clarification` 返回 `none`。

- [ ] **Step 2: 运行测试确认因旧 fast/deep 配置失败**

Run: `pytest -q tests/test_model_constraints.py tests/test_agent_model_router.py -x`

Expected: FAIL，显示默认模型仍为 14B 或路由仍返回 `fast/deep`。

- [ ] **Step 3: 实现统一配置和目标选择**

将配置收敛为：

```python
ollama_generation_base_url: str = Field(
    default="http://localhost:11435", alias="OLLAMA_GENERATION_BASE_URL"
)
ollama_generation_model: str = Field(
    default="qwen3.5:9b", alias="OLLAMA_GENERATION_MODEL"
)
ollama_generation_context_length: int = Field(
    default=32768, gt=0, alias="OLLAMA_GENERATION_CONTEXT_LENGTH"
)
ollama_generation_parallelism: int = Field(
    default=1, gt=0, alias="OLLAMA_GENERATION_PARALLELISM"
)
ollama_embedding_model: str = Field(
    default="qwen3-embedding:4b", alias="OLLAMA_EMBEDDING_MODEL"
)
ollama_embedding_dimensions: int = Field(default=2560, alias="OLLAMA_EMBEDDING_DIMENSIONS")
```

`InferenceTarget.profile` 只允许 `none|generation`。`AgentModelRouter.select(route)` 除澄清路由外始终返回配置中的 9B 目标。`ModelRuntime` 只创建 `generation` 容量，执行器的 route trace 继续记录问题路由，但不再用它选择模型。

- [ ] **Step 4: 运行单元测试并修复调用签名**

Run: `pytest -q tests/test_model_constraints.py tests/test_agent_model_router.py tests/test_model_runtime.py tests/test_agent_executor.py`

Expected: PASS；队列快照只有 `generation`。

- [ ] **Step 5: 提交单模型后端**

```bash
git add src/app/core/config.py src/app/services/agent_model_router.py src/app/services/model_runtime.py src/app/services/agent_executor.py tests/test_model_constraints.py tests/test_agent_model_router.py tests/test_model_runtime.py tests/test_agent_executor.py
git commit -m "Unify Agent generation on Qwen 3.5 9B"
```

### Task 2: 删除回答模式 API、会话字段和前端控件

**Files:**
- Modify: `src/app/schemas/agent.py`
- Modify: `src/app/services/conversation_memory.py`
- Modify: `src/app/models/records.py`
- Modify: `src/app/api/agent_routes.py`
- Modify: `src/app/static/index.html`
- Create: `src/app/db/alembic/versions/6c1a2d9e4b70_remove_answer_mode.py`
- Modify: `tests/test_agent_routes.py`
- Modify: `tests/test_agent_streaming.py`
- Modify: `tests/test_conversation_memory.py`
- Modify: `tests/test_migrations.py`
- Modify: `tests/test_static_frontend.py`

- [ ] **Step 1: 写失败测试定义无模式 API**

断言 `AgentQueryRequest` 不再声明 `answer_mode`；会话响应不返回 `answer_mode`；前端不存在 `answerModeControl`、`auto/fast/deep` 按钮和“切换快速”文案；SSE 仍支持 `route/queue/token/citation/final`。

```python
def test_agent_query_has_no_answer_mode() -> None:
    assert "answer_mode" not in AgentQueryRequest.model_fields

def test_frontend_has_no_model_mode_controls() -> None:
    page = Path("src/app/static/index.html").read_text(encoding="utf-8")
    assert 'id="answerModeControl"' not in page
    assert "切换快速" not in page
```

- [ ] **Step 2: 运行测试确认旧模式仍存在**

Run: `pytest -q tests/test_agent_routes.py tests/test_conversation_memory.py tests/test_static_frontend.py -x`

Expected: FAIL，指出 schema、会话或页面仍包含 `answer_mode`。

- [ ] **Step 3: 删除回答模式行为并新增 Alembic 迁移**

从 schema、会话模型、memory 方法、路由请求处理和前端请求体中删除 `answer_mode`。迁移的 PostgreSQL/SQLite upgrade 使用 batch alter 删除 `conversation_sessions.answer_mode`；downgrade 恢复非空字符串列并以 `auto` 作为服务回滚兼容值。

- [ ] **Step 4: 保持流式和取消能力通过测试**

Run: `pytest -q tests/test_agent_routes.py tests/test_agent_streaming.py tests/test_conversation_memory.py tests/test_migrations.py tests/test_static_frontend.py`

Expected: PASS，SSE 事件和停止生成仍可用。

- [ ] **Step 5: 提交模式删除**

```bash
git add src/app/schemas/agent.py src/app/services/conversation_memory.py src/app/models/records.py src/app/api/agent_routes.py src/app/static/index.html src/app/db/alembic/versions/6c1a2d9e4b70_remove_answer_mode.py tests
git commit -m "Remove Agent answer mode selection"
```

### Task 3: 单端点健康检查和证据引用契约

**Files:**
- Modify: `src/app/services/model_readiness.py`
- Modify: `src/app/api/routes.py`
- Modify: `src/app/services/agent_synthesizer.py`
- Modify: `src/app/services/answer_verifier.py`
- Modify: `tests/test_model_readiness.py`
- Modify: `tests/test_api_routes.py`
- Modify: `tests/test_agent_synthesizer.py`
- Modify: `tests/test_answer_verifier.py`

- [ ] **Step 1: 写失败测试约束健康响应与引用边界**

健康响应只包含 `generation`、`embedding` 和一个 `generation` 队列。引用测试提供三条证据并让模型返回 `[0] [4] [2]`，期望最终仅保留有效编号并映射到原文页码。

```python
assert set(payload["models"]) == {"generation", "embedding"}
assert set(payload["queues"]) == {"generation"}
assert payload["models"]["generation"]["model"] == "qwen3.5:9b"
assert payload["models"]["embedding"]["dimensions"] == 2560
```

- [ ] **Step 2: 运行测试确认仍有 fast/deep 状态**

Run: `pytest -q tests/test_model_readiness.py tests/test_api_routes.py tests/test_agent_synthesizer.py tests/test_answer_verifier.py -x`

Expected: FAIL，健康响应仍包含 fast/deep。

- [ ] **Step 3: 实现单端点探测并保留后端引用验证**

同一 Ollama base URL 的 `/api/tags` 与 `/api/ps` 只请求一次，再分别判断 9B 与 4B 的 `ready|idle|missing|unreachable`。不要把引用编号、页码或链接生成职责移给模型；模型只返回被引用索引，后端继续执行范围检查和 citation 映射。

- [ ] **Step 4: 运行健康与引用测试**

Run: `pytest -q tests/test_model_readiness.py tests/test_api_routes.py tests/test_agent_synthesizer.py tests/test_answer_verifier.py`

Expected: PASS。

- [ ] **Step 5: 提交健康和引用契约**

```bash
git add src/app/services/model_readiness.py src/app/api/routes.py src/app/services/agent_synthesizer.py src/app/services/answer_verifier.py tests/test_model_readiness.py tests/test_api_routes.py tests/test_agent_synthesizer.py tests/test_answer_verifier.py
git commit -m "Report one Knowledge Agent model runtime"
```

### Task 4: 迁移 pgvector 到 2560 维并提供可恢复重建工具

**Files:**
- Create: `src/app/db/alembic/versions/81f6b74cc203_resize_embeddings_to_2560.py`
- Create: `scripts/reindex_embeddings.py`
- Modify: `tests/test_migrations.py`
- Create: `tests/test_reindex_embeddings.py`
- Modify: `tests/test_vector_retrieval.py`

- [ ] **Step 1: 写失败测试定义数据库迁移和重建命令**

迁移文本必须创建 `vector(2560)`，upgrade 先删除旧 pgvector 索引表再重建空表，downgrade 对称恢复 `vector(4096)`。重建命令必须拒绝非维护模式、逐文档提交、校验每个 embedding 长度并在任一文档失败时非零退出。

```python
def test_reindex_refuses_without_maintenance_flag() -> None:
    result = runner.invoke(app, ["--project", "internal-research"])
    assert result.exit_code != 0
    assert "--maintenance-confirmed" in result.output
```

- [ ] **Step 2: 运行测试确认迁移与脚本不存在**

Run: `pytest -q tests/test_migrations.py tests/test_reindex_embeddings.py tests/test_vector_retrieval.py -x`

Expected: FAIL，缺少迁移或 `scripts.reindex_embeddings`。

- [ ] **Step 3: 实现安全迁移和幂等重建**

重建工具读取现有 `DocumentChunk.content`，批量调用现有 `OllamaClient.embed_texts()`，把 JSON embedding 与 `PGVectorStore.replace_document_chunks()` 在同一文档事务内更新。提供：

```text
--maintenance-confirmed
--project <slug>|--all-projects
--batch-size 16
--resume-after-document <uuid>
--verify-only
--report <json path>
```

报告包含文档数、chunk 数、成功数、失败文档、维度、模型、开始/结束时间；`--verify-only` 断言所有可检索 chunk 的 JSON 向量为 2560 维且 pgvector 行数一致。

- [ ] **Step 4: 运行迁移和重建测试**

Run: `pytest -q tests/test_migrations.py tests/test_reindex_embeddings.py tests/test_vector_retrieval.py`

Expected: PASS。

- [ ] **Step 5: 提交向量迁移工具**

```bash
git add src/app/db/alembic/versions/81f6b74cc203_resize_embeddings_to_2560.py scripts/reindex_embeddings.py tests/test_migrations.py tests/test_reindex_embeddings.py tests/test_vector_retrieval.py
git commit -m "Migrate document embeddings to 2560 dimensions"
```

### Task 5: 建立固定 20 题检索验收集和对比报告

**Files:**
- Create: `scripts/build_retrieval_acceptance.py`
- Create: `benchmarks/query/knowledge_agent_retrieval_v1.json`
- Create: `tests/test_retrieval_acceptance.py`
- Modify: `docs/query_acceptance/internal_research_v1.md`

- [ ] **Step 1: 写失败测试定义可重复报告格式**

报告必须为每题记录 `question`、`expected_document_ids`、`retrieved_document_ids`、`top_k`、`citation_pages`、`latency_ms` 和 `passed`，汇总 `recall_at_k`、有效引用率和 P95 延迟。

- [ ] **Step 2: 运行测试确认工具不存在**

Run: `pytest -q tests/test_retrieval_acceptance.py -x`

Expected: FAIL，缺少验收模块。

- [ ] **Step 3: 实现固定 15–20 题的采集和运行模式**

工具提供 `capture` 和 `run` 子命令。`capture` 从当前测试文章与已验证问答中生成候选，但必须把最终 20 题写成静态 JSON；`run` 不再调用模型生成问题，只运行固定题集并输出版本化 JSON 报告，保证旧 8B 和新 4B 可公平比较。通过标准固定为：新模型 `recall@5 >= max(0.80, 旧模型 recall@5 - 0.05)`，引用有效率为 100%，热状态端到端 P95 不超过 45 秒。

- [ ] **Step 4: 运行工具测试**

Run: `pytest -q tests/test_retrieval_acceptance.py tests/test_query_eval.py`

Expected: PASS。

- [ ] **Step 5: 提交验收工具**

```bash
git add scripts/build_retrieval_acceptance.py benchmarks/query/knowledge_agent_retrieval_v1.json tests/test_retrieval_acceptance.py docs/query_acceptance/internal_research_v1.md
git commit -m "Add fixed Knowledge Agent retrieval acceptance"
```

### Task 6: 统一 Knowledge Agent 品牌并创建双环境部署模板

**Files:**
- Modify: `src/app/__init__.py`
- Modify: `src/app/core/config.py`
- Modify: `src/app/static/index.html`
- Modify: `README.md`
- Delete: `deploy/systemd/llm-wiki-pilot.service`
- Delete: `deploy/systemd/llm-wiki-ollama-fast.service`
- Delete: `deploy/systemd/llm-wiki-ollama-deep.service`
- Delete: `deploy/systemd/llm-wiki-tunnel.service`
- Create: `deploy/systemd/knowledge-agent-dev-api.service`
- Create: `deploy/systemd/knowledge-agent-dev-ollama.service`
- Create: `deploy/systemd/knowledge-agent-test-api.service`
- Create: `deploy/systemd/knowledge-agent-test-ollama.service`
- Create: `deploy/systemd/knowledge-agent-test-tunnel.service`
- Create: `.env.development.example`
- Create: `.env.test.example`
- Modify: `deploy/cloudflared/config.yml.example`
- Modify: `deploy/internal-pilot.md`
- Modify: `tests/test_deployment_config.py`
- Rename: `tests/test_wiki_removed.py` to `tests/test_legacy_content_routes_removed.py`

- [ ] **Step 1: 写失败测试定义品牌和双栈模板**

部署测试断言开发 API 为 8002/GPU 0/11435/无认证，测试 API 为 8001/GPU 1/11436/启用认证；两个 Ollama 模板均为 32K、Flash Attention、q8_0、5m、并发 1。品牌扫描只覆盖活动源代码、当前部署模板和 README，允许历史迁移规范保留旧名说明。

```python
assert "APP_NAME=Knowledge Agent" in dev_env
assert "AUTH_ENABLED=false" in dev_env
assert "DATABASE_URL=postgresql+psycopg://knowledge_agent_dev" in dev_env
assert "CUDA_VISIBLE_DEVICES=1" in test_ollama
assert "OLLAMA_HOST=127.0.0.1:11436" in test_ollama
```

- [ ] **Step 2: 运行测试确认旧模板和名称存在**

Run: `pytest -q tests/test_deployment_config.py tests/test_static_frontend.py tests/test_legacy_content_routes_removed.py -x`

Expected: FAIL，缺少新模板或仍显示旧产品名。

- [ ] **Step 3: 创建五个 Systemd 服务和两套环境模板**

开发服务目录使用 `/home/zhangyh/knowledge-agent-dev`，测试使用 `/home/zhangyh/knowledge-agent-test`。两套 API 的 `EnvironmentFile` 分别指向自身 `runtime/app.env`。Tunnel 只依赖并转发测试 API。Ollama 模型目录可共享 `/home/zhangyh/knowledge-agent-models`，其他运行目录不得共享。

- [ ] **Step 4: 更新活动品牌和部署文档**

将应用默认名、页面标题、README 和部署文档改为 Knowledge Agent。历史清理测试改为中性文件名，但可以保留对已删除旧路由字面量的回归断言。

- [ ] **Step 5: 运行品牌与部署测试**

Run: `pytest -q tests/test_deployment_config.py tests/test_static_frontend.py tests/test_legacy_content_routes_removed.py tests/test_app_startup.py`

Expected: PASS。

- [ ] **Step 6: 提交命名和部署模板**

```bash
git add -A
git commit -m "Separate Knowledge Agent development and test stacks"
```

### Task 7: 全量本地回归与远程改名前置检查

**Files:**
- Modify if required by failures: affected `src/` and `tests/` files only
- Create runtime artifact: `tmp/knowledge-agent-preflight.json` (do not commit)

- [ ] **Step 1: 运行静态检查和全量测试**

Run: `git diff --check && pytest -q`

Expected: 全部测试通过，无 diff 格式错误。

- [ ] **Step 2: 扫描活动命名和旧模型配置**

Run:

```bash
rg -n -i "llm.wiki|llm_wiki|llm-wiki|qwen3:14b|qwen3.6:27b|qwen3-embedding:8b|answerModeControl" src deploy README.md .env*.example tests
```

Expected: 除明确的旧路由移除测试外无结果；任何结果必须逐项说明或删除。

- [ ] **Step 3: 只读采集服务器前置状态**

记录当前 Git 提交、服务、数据库大小、文档/chunk/向量行数、数据目录大小、GPU、磁盘余量、Ollama 模型列表、Tunnel 地址和公网 401/200 状态到 `tmp/knowledge-agent-preflight.json`。不得输出密码或 secret。

- [ ] **Step 4: 拉取新模型但不切换服务**

在共享模型目录执行：

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen3.5:9b
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen3-embedding:4b
```

Expected: `/api/tags` 同时列出两个新模型；当前测试 API 和公网仍正常。

- [ ] **Step 5: 推送代码提交**

Run: `git push origin codex/internal-pilot`

Expected: 远程分支指向本地已验证提交。

### Task 8: 部署并验收 GPU 0 开发环境

**Files:**
- Server create: `/home/zhangyh/knowledge-agent-dev/runtime/app.env`
- Server create: `/home/zhangyh/.config/systemd/user/knowledge-agent-dev-api.service`
- Server create: `/home/zhangyh/.config/systemd/user/knowledge-agent-dev-ollama.service`

- [ ] **Step 1: 克隆独立开发目录并创建空数据库**

克隆指定提交到 `/home/zhangyh/knowledge-agent-dev`，创建 `knowledge_agent_dev` 数据库，安装现有锁定依赖。`runtime/app.env` 使用 8002、11435、GPU 0、开发数据库、独立 data/raw/cache/tmp，并设置 `AUTH_ENABLED=false`。

- [ ] **Step 2: 启动开发 Ollama 并验证 GPU 绑定**

Run:

```bash
systemctl --user enable --now knowledge-agent-dev-ollama.service
curl -fsS http://127.0.0.1:11435/api/tags
nvidia-smi
```

Expected: Ollama 进程只出现在 GPU 0；9B 与 4B 均可用。

- [ ] **Step 3: 迁移空数据库并启动开发 API**

Run:

```bash
cd /home/zhangyh/knowledge-agent-dev
.venv/bin/alembic upgrade head
systemctl --user enable --now knowledge-agent-dev-api.service
curl -fsS http://127.0.0.1:8002/api/health
```

Expected: API `ok`，模型为 `ready|idle`，数据库向量表为 `vector(2560)`。

- [ ] **Step 4: 通过 SSH 转发验收页面、上传与问答**

本机转发 8002，确认无需登录、可以创建项目和上传一份非生产样本文档；问答返回 9B 模型名、有效引用和原文件链接。测试环境公网在整个过程中保持 200（授权）和 401（未授权）。

- [ ] **Step 5: 验证开发模型 5 分钟卸载**

触发 9B 与 4B 后轮询 `/api/ps`，不发送生成或 embedding 请求。Expected: 300 秒左右 GPU 0 模型列表为空，开发健康状态为 `idle/ok`；GPU 1 当前测试不受影响。

### Task 9: 测试环境维护迁移与切换

**Files:**
- Server create: `/home/zhangyh/knowledge-agent-test/runtime/app.env`
- Server create: `/home/zhangyh/.config/systemd/user/knowledge-agent-test-*.service`
- Backup create: `/home/zhangyh/knowledge-agent-backups/<timestamp>/`

- [ ] **Step 1: 采集旧 8B 固定题集基线**

在切换前运行 `scripts/build_retrieval_acceptance.py run`，保存旧模型 Top-K、引用和延迟报告到时间戳备份目录。报告不得包含密码、session cookie 或 API secret。

- [ ] **Step 2: 启用维护页并确认写入被阻止**

先让公网网关返回维护页，再验证上传、删除和 Agent 请求不能到达旧 API。保留只读备份检查通道，不对公网开放。

- [ ] **Step 3: 创建数据库与文件快照**

使用 `pg_dump --format=custom` 备份当前数据库；用同一 Linux 文件系统内的受控复制创建原始文件与解析结果快照；保存 `sha256sum` 清单。确认数据库备份可列出、文件数与哈希清单一致后才能继续。

- [ ] **Step 4: 建立新测试目录和数据库**

将已批准提交克隆到 `/home/zhangyh/knowledge-agent-test`，恢复数据库到 `knowledge_agent_test`，复制正式数据到新目录并保持相对路径不变。运行 Alembic 到 head，确认业务表记录数与迁移前一致，新的 pgvector 表为空且为 2560 维。

- [ ] **Step 5: 启动 GPU 1 测试 Ollama**

停用旧 deep Ollama 服务后启动 `knowledge-agent-test-ollama.service`。验证进程只在 GPU 1，9B 和 4B 标签存在，GPU 0 开发服务仍 active。

- [ ] **Step 6: 重建并校验全部测试向量**

Run:

```bash
cd /home/zhangyh/knowledge-agent-test
.venv/bin/python scripts/reindex_embeddings.py --all-projects --batch-size 16 --maintenance-confirmed --report runtime/reindex-report.json
.venv/bin/python scripts/reindex_embeddings.py --all-projects --maintenance-confirmed --verify-only --report runtime/reindex-verify.json
```

Expected: 所有 ready 文档成功，失败文档为空，JSON embedding 与 pgvector 均为 2560 维，chunk 与索引计数匹配。

- [ ] **Step 7: 启动新测试 API 并运行固定题集**

启动 `knowledge-agent-test-api.service`，先仅本机访问。运行新 4B 固定题集报告，要求失败重建文档为 0、无效引用为 0、引用有效率 100%、`recall@5 >= max(0.80, 旧模型 recall@5 - 0.05)`、热状态端到端 P95 不超过 45 秒；任一条件不满足就保持维护模式并回滚，不用主观判断强行上线。

- [ ] **Step 8: 切换 Tunnel 并恢复公网**

更新 Tunnel/Caddy 只指向 8001 新测试 API，启动 `knowledge-agent-test-tunnel.service`。验证未授权 401、授权页面 200、健康 `ok`、上传、解析、问答、PDF Range 206、页码引用和原文件链接。

- [ ] **Step 9: 停用旧服务但不删除回滚资源**

停用并 disable 旧 API/Ollama/Tunnel 服务；保留旧目录、数据库备份和 14B/27B/8B 模型文件 7 天。记录恢复旧服务所需的完整命令。

### Task 10: 仓库重命名、最终验收和七天清理门槛

**Files:**
- Modify: local Git remote URL
- Rename after server stability: `D:\LLM_wiki` to `D:\knowledge-agent`
- Update: operator deployment record in `docs/work.md`

- [ ] **Step 1: 重命名 GitHub 仓库并更新远程**

在确认 CLI 已认证且当前用户有仓库管理权限后执行仓库重命名为 `knowledge-agent-server`，更新本地、开发服务器和测试服务器的 `origin`，并执行 `git fetch --all --prune`。若没有管理权限，停止本步骤并报告，不影响已部署服务。

- [ ] **Step 2: 更新本地工作区路径**

结束所有占用旧目录的进程，验证 worktree 与主仓库均 clean，再使用单一 PowerShell 流程把根目录迁移到 `D:\knowledge-agent`。迁移后运行 `git status`、`git worktree list` 和一次聚焦测试，禁止跨 shell 拼接移动命令。

- [ ] **Step 3: 执行最终全量验证**

Run locally: `pytest -q`

Run on server for both environments:

```bash
systemctl --user is-active knowledge-agent-dev-api knowledge-agent-dev-ollama knowledge-agent-test-api knowledge-agent-test-ollama knowledge-agent-test-tunnel
curl -fsS http://127.0.0.1:8002/api/health
curl -fsS http://127.0.0.1:8001/api/health
```

同时验证 GPU 绑定、32K、5m、队列容量 1、模型名称、向量维度、公网 401/200 和固定检索题集结果。

- [ ] **Step 4: 提交部署记录**

在 `docs/work.md` 记录最终 Git 提交、两个环境目录/端口、数据库、服务名、模型、备份位置、基线/新报告和回滚命令；不得写入密码或 secret。

```bash
git add docs/work.md
git commit -m "Record Knowledge Agent isolated deployment"
git push origin codex/internal-pilot
```

- [ ] **Step 5: 设置七天后人工清理门槛**

七天内不自动删除任何旧数据库、模型或目录。七天后仅在用户确认新环境稳定、备份可恢复且活动配置无旧路径时，单独执行清理；清理不属于本次即时发布的自动步骤。
