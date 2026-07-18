# Knowledge Agent Redis Asynchronous Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为开发和测试环境部署本机 Redis 与隔离 RQ worker，使上传异步执行且任务不跨环境。

**Architecture:** 一个仅绑定 `127.0.0.1:6379` 的原生 Redis 7.4 进程提供共享队列，开发和测试分别使用 DB 1、DB 2。两个 systemd user worker 分别读取自身环境文件、数据库、数据目录和 Ollama 地址，API 只负责入队。

**Tech Stack:** Redis 7.4、RQ、Python 3.12、systemd user services、pytest。

---

## 文件结构与职责

- `.env.development.example`：开发队列连接 `redis://127.0.0.1:6379/1`。
- `.env.test.example`：测试队列连接 `redis://127.0.0.1:6379/2`。
- `deploy/systemd/knowledge-agent-redis.service`：启动只监听本机并启用 AOF 的现有 Redis 7.4 可执行文件。
- `deploy/systemd/knowledge-agent-dev-worker.service`：运行开发环境 `app.workers.runner`。
- `deploy/systemd/knowledge-agent-test-worker.service`：运行测试环境 `app.workers.runner`。
- `deploy/systemd/knowledge-agent-dev-api.service`：声明 Redis 启动顺序和依赖。
- `deploy/systemd/knowledge-agent-test-api.service`：声明 Redis 启动顺序和依赖。
- `tests/test_deployment_config.py`：静态验证 Redis 安全边界、DB 隔离及 worker 配置完整性。
- `docs/work.md`：在最终环境切换后记录 Redis/worker 状态和回滚命令，不记录 secret。

### Task 1: 用测试定义完整的 Redis 与 worker 部署契约

**Files:**
- Modify: `tests/test_deployment_config.py`

- [ ] **Step 1: 写失败测试**

将临时的“不得出现 Redis”断言替换为完整契约：

```python
def test_redis_is_local_and_each_environment_has_an_isolated_worker() -> None:
    redis = _read("deploy/systemd/knowledge-agent-redis.service")
    dev_env = _read(".env.development.example")
    test_env = _read(".env.test.example")
    dev_worker = _read("deploy/systemd/knowledge-agent-dev-worker.service")
    test_worker = _read("deploy/systemd/knowledge-agent-test-worker.service")

    assert "%h/local/redis/bin/redis-server" in redis
    assert "--bind 127.0.0.1 --port 6379" in redis
    assert "--protected-mode yes" in redis
    assert "--appendonly yes" in redis
    assert "REDIS_URL=redis://127.0.0.1:6379/1" in dev_env
    assert "REDIS_URL=redis://127.0.0.1:6379/2" in test_env
    assert "WorkingDirectory=%h/knowledge-agent-dev" in dev_worker
    assert "WorkingDirectory=%h/knowledge-agent-test" in test_worker
    assert "app.workers.runner" in dev_worker
    assert "app.workers.runner" in test_worker
    assert "EnvironmentFile=%h/knowledge-agent-dev/runtime/app.env" in dev_worker
    assert "EnvironmentFile=%h/knowledge-agent-test/runtime/app.env" in test_worker
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Run: `D:\Miniconda3\python.exe -m pytest -q tests/test_deployment_config.py -x`

Expected: FAIL，因为 Redis 与两个 worker service 尚不存在。

- [ ] **Step 3: 不修改生产文件，确认失败不是导入或路径错误**

Run: `git status --short`

Expected: 仅测试和计划文件有变更，失败路径与计划中的文件名一致。

### Task 2: 创建 Redis、双 worker 和 API 依赖配置

**Files:**
- Modify: `.env.development.example`
- Modify: `.env.test.example`
- Create: `deploy/systemd/knowledge-agent-redis.service`
- Create: `deploy/systemd/knowledge-agent-dev-worker.service`
- Create: `deploy/systemd/knowledge-agent-test-worker.service`
- Modify: `deploy/systemd/knowledge-agent-dev-api.service`
- Modify: `deploy/systemd/knowledge-agent-test-api.service`

- [ ] **Step 1: 恢复隔离的 Redis URL**

开发模板加入：

```dotenv
REDIS_URL=redis://127.0.0.1:6379/1
```

测试模板加入：

```dotenv
REDIS_URL=redis://127.0.0.1:6379/2
```

- [ ] **Step 2: 创建本机 Redis systemd 服务**

服务必须使用服务器已有的 Redis 7.4，并使用以下核心配置：

```ini
[Service]
ExecStartPre=/usr/bin/mkdir -p %h/knowledge-agent-runtime/redis
ExecStart=%h/local/redis/bin/redis-server --bind 127.0.0.1 --port 6379 --protected-mode yes --appendonly yes --dir %h/knowledge-agent-runtime/redis
ExecStop=%h/local/redis/bin/redis-cli -h 127.0.0.1 -p 6379 shutdown
Restart=on-failure
```

- [ ] **Step 3: 创建两个 RQ worker 服务**

开发 worker 使用：

```ini
WorkingDirectory=%h/knowledge-agent-dev
EnvironmentFile=%h/knowledge-agent-dev/runtime/app.env
Environment=PYTHONPATH=%h/knowledge-agent-dev/src
ExecStart=%h/knowledge-agent-dev/.venv/bin/python -m app.workers.runner
```

测试 worker 使用对应的 `knowledge-agent-test` 路径。两者均 `Requires=knowledge-agent-redis.service`，并依赖各自 Ollama 服务。

- [ ] **Step 4: 为两个 API 声明 Redis 依赖**

在 API `[Unit]` 中加入 `After=... knowledge-agent-redis.service` 与 `Wants=... knowledge-agent-redis.service`，保证已配置 Redis 的 API 不会先于队列服务启动。

- [ ] **Step 5: 运行部署测试**

Run: `D:\Miniconda3\python.exe -m pytest -q tests/test_deployment_config.py tests/test_api_routes.py`

Expected: PASS，上传 API 既有行为无回归。

- [ ] **Step 6: 提交并推送**

```bash
git add .env.development.example .env.test.example deploy/systemd/knowledge-agent-redis.service deploy/systemd/knowledge-agent-dev-worker.service deploy/systemd/knowledge-agent-test-worker.service deploy/systemd/knowledge-agent-dev-api.service deploy/systemd/knowledge-agent-test-api.service tests/test_deployment_config.py docs/superpowers/plans/2026-07-18-redis-asynchronous-ingestion.md
git commit -m "Deploy isolated Redis ingestion workers"
git push origin codex/internal-pilot
```

### Task 3: 在服务器部署 Redis 和开发 worker

**Files:**
- Server modify: `/home/zhangyh/knowledge-agent-dev/runtime/app.env`
- Server install: `/home/zhangyh/.config/systemd/user/knowledge-agent-redis.service`
- Server install: `/home/zhangyh/.config/systemd/user/knowledge-agent-dev-worker.service`
- Server update: `/home/zhangyh/.config/systemd/user/knowledge-agent-dev-api.service`

- [ ] **Step 1: 更新开发代码并安装服务文件**

执行 `git pull --ff-only`，复制已提交的 service 模板到 user systemd 目录，并把开发 `REDIS_URL` 设置为 DB 1。不得输出数据库密码或认证 secret。

- [ ] **Step 2: 启动 Redis 与开发 worker**

Run:

```bash
systemctl --user daemon-reload
systemctl --user enable --now knowledge-agent-redis.service
systemctl --user enable --now knowledge-agent-dev-worker.service
systemctl --user restart knowledge-agent-dev-api.service
%h/local/redis/bin/redis-cli -h 127.0.0.1 -p 6379 ping
```

Expected: `PONG`，Redis 只映射 `127.0.0.1:6379`，开发 worker 为 active。

- [ ] **Step 3: 真实验证异步开发上传**

上传一份新的非生产样本文档，断言 API 快速返回排队状态；轮询 pipeline run 直到 `completed`，随后确认 document `ready`、JSON embedding 和 pgvector 都是 2560 维。

- [ ] **Step 4: 验证开发问答**

向 `/api/agent/query` 提问样本文档中可核查的问题，要求 `answer_model=qwen3.5:9b`、引用 document id 正确且原文链接存在。

### Task 4: 在维护窗口迁移测试环境并启用测试 worker

**Files:**
- Server create: `/home/zhangyh/knowledge-agent-test/runtime/app.env`
- Server install: `/home/zhangyh/.config/systemd/user/knowledge-agent-test-worker.service`
- Server update: `/home/zhangyh/.config/systemd/user/knowledge-agent-test-api.service`
- Backup create: `/home/zhangyh/knowledge-agent-backups/<timestamp>/`

- [ ] **Step 1: 使用 PostgreSQL 16 容器工具创建可恢复快照**

在数据库容器内运行与服务端同版本的 `pg_dump --format=custom`，把 dump 复制到备份目录；同时 reflink 复制旧 `runtime/data` 并生成 SHA-256 清单。用 `pg_restore --list` 验证 dump 可读。

- [ ] **Step 2: 创建测试数据库和新测试目录**

从已推送提交克隆 `/home/zhangyh/knowledge-agent-test`，创建独立 `knowledge_agent_test` 角色/数据库，恢复 dump、复制数据并运行 Alembic 到 head。迁移后业务记录数必须与快照一致，pgvector 表应为 `vector(2560)` 空表。

- [ ] **Step 3: 启动 GPU1 Ollama、Redis DB 2 worker**

启用 `knowledge-agent-test-ollama.service`、`knowledge-agent-test-worker.service`，验证 worker 只连接 Redis DB 2，GPU1 服务为 active，GPU0 开发服务不受影响。

- [ ] **Step 4: 全量重建测试 embedding**

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/reindex_embeddings.py --all-projects --batch-size 16 --maintenance-confirmed --report runtime/reindex-report.json
PYTHONPATH=src .venv/bin/python scripts/reindex_embeddings.py --all-projects --maintenance-confirmed --verify-only --report runtime/reindex-verify.json
```

Expected: 失败文档为 0，所有可检索 chunk 的 JSON 与 pgvector 均为 2560 维且数量一致。

- [ ] **Step 5: 运行固定 20 题检索验收**

要求 citation validity 100%，`recall@5 >= max(0.80, baseline - 0.05)`，P95 不超过 45 秒；不满足则保持维护状态并回滚，不恢复公网。

- [ ] **Step 6: 恢复测试 API 与临时公网入口**

验证 Redis/worker/API/Ollama 全部 active，公网未授权 401、授权页面 200，并完成上传、后台解析、问答、PDF Range 206、页码引用和原文件链接验收。

### Task 5: 最终回归与部署记录

**Files:**
- Modify: `docs/work.md`

- [ ] **Step 1: 本地全量测试**

Run: `git diff --check && D:\Miniconda3\python.exe -m pytest -q`

Expected: 全部测试通过，且不暂存用户已有的四个规格删除变更。

- [ ] **Step 2: 验证双环境与 5 分钟释放**

确认开发 API/worker/Ollama 使用 8002/Redis DB1/GPU0，测试使用 8001/Redis DB2/GPU1；停止请求约 300 秒后两个 Ollama `/api/ps` 不再列出 9B/4B 模型，健康状态仍为 `idle/ok`。

- [ ] **Step 3: 记录无 secret 的部署结果与回滚命令**

在 `docs/work.md` 写入服务名、端口、数据库、模型、备份目录、重建报告、检索报告及停新服务/恢复旧服务的命令，不写密码、cookie 或 App Secret。

- [ ] **Step 4: 提交并推送部署记录**

```bash
git add docs/work.md
git commit -m "Record Redis-backed isolated deployment"
git push origin codex/internal-pilot
```
