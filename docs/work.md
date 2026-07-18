# Knowledge Agent 工作记录

## 当前目标

- 在同一台双 GPU 服务器上隔离开发与测试环境。
- 开发环境固定 GPU 0，测试环境固定 GPU 1。
- 两套环境统一使用 `qwen3.5:9b` 和 `qwen3-embedding:4b`。
- 测试环境保留现有文章、账号、会话和公网入口。
- 开发环境使用空数据库且只通过 SSH 转发访问。

## 运行约束

- 生成上下文固定 32K，并发容量为 1。
- Flash Attention 和 q8_0 KV Cache 开启。
- 模型闲置 5 分钟后释放 GPU。
- DeepSeek 等外部 API 默认关闭。
- 引用编号、页码和来源链接由后端验证。
- 开发提交只有人工验收后才能发布到测试环境。

详细设计和执行步骤见：

- `docs/superpowers/specs/2026-07-17-knowledge-agent-environment-isolation-design.md`
- `docs/superpowers/plans/2026-07-17-knowledge-agent-environment-isolation.md`

## 2026-07-18 隔离部署记录

部署代码基线：`c009875`（`codex/internal-pilot`）。

GitHub 仓库已重命名为 `zyhnosleep/knowledge-agent-server`；本地和两套服务器代码目录的 `origin` 均已更新为新地址。

### 开发环境

- 目录：`/home/zhangyh/knowledge-agent-dev`
- API：`127.0.0.1:8002`，仅通过 SSH 转发访问
- 数据库：`knowledge_agent_dev`，空开发数据集
- Redis：`redis://127.0.0.1:6379/1`
- GPU/Ollama：GPU 0，`127.0.0.1:11435`
- 服务：`knowledge-agent-dev-api`、`knowledge-agent-dev-worker`、`knowledge-agent-dev-ollama`
- 真实验收：上传在 0.116 秒返回排队状态，worker 完成解析；JSON embedding 与 pgvector 均为 2560 维；问答模型为 `qwen3.5:9b` 且引用有效。
- 队列验收：DB 1 排队、运行中、失败任务均为 0，注册 worker 为 1。

### 测试环境

- 目录：`/home/zhangyh/knowledge-agent-test`
- API：`127.0.0.1:8001`
- 数据库：`knowledge_agent_test`
- Redis：`redis://127.0.0.1:6379/2`
- GPU/Ollama：GPU 1，`127.0.0.1:11436`
- 服务：`knowledge-agent-test-api`、`knowledge-agent-test-worker`、`knowledge-agent-test-ollama`
- 临时公网：`https://planets-intranet-dog-therapy.trycloudflare.com`，由原有 Caddy Basic Auth 网关转发到新测试 API
- 迁移结果：15 个文档、1778 个 chunk；4B 重建成功 15/15，失败 0；JSON embedding 与 pgvector 1778/1778 有效，维度 2560。
- 检索基线：旧 8B `recall@5=1.0`、引用有效率 100%、P95 846 ms。
- 新模型验收：4B `recall@5=1.0`、引用有效率 100%、P95 1098 ms，验收通过。
- 公网验收：未授权 401、授权页面 200、异步上传完成、Agent 回答与引用有效、PDF Range 返回 206。
- 新上传文件写入 `/home/zhangyh/knowledge-agent-test/runtime/data/raw/...`，不写入迁移备份目录。
- 队列验收：DB 2 排队、运行中、失败任务均为 0，注册 worker 为 1。
- 空闲释放：停止模型请求约 5 分钟后，开发与测试 `/api/ps` 均为空；GPU 1 回落到 22 MiB，GPU 0 仅保留既有 ComfyUI 的 256 MiB。

### 公共运行组件与备份

- Redis 服务：`knowledge-agent-redis.service`
- Redis 版本：服务器现有 Redis 7.4，仅监听 `127.0.0.1:6379`，AOF 位于 `/home/zhangyh/knowledge-agent-runtime/redis`
- 共享依赖运行时：`/home/zhangyh/knowledge-agent-runtime/.venv`
- 共享 Ollama 模型目录：`/home/zhangyh/knowledge-agent-models`
- 迁移前备份：`/home/zhangyh/knowledge-agent-backups/20260718T015755Z`
- 重建报告：`/home/zhangyh/knowledge-agent-test/runtime/reindex-report.json`
- 重建校验：`/home/zhangyh/knowledge-agent-test/runtime/reindex-verify.json`
- 固定题集与报告：备份目录下 `acceptance/knowledge_agent_retrieval_v1.json`、`baseline-8b.json`、`new-4b.json`

### 回滚入口

新环境只需停止，不删除任何目录或数据库：

```bash
systemctl --user disable --now knowledge-agent-test-api.service
systemctl --user disable --now knowledge-agent-test-worker.service
systemctl --user disable --now knowledge-agent-test-ollama.service
systemctl --user enable --now llm-wiki-ollama-fast.service
systemctl --user enable --now llm-wiki-ollama-deep.service
systemctl --user enable --now llm-wiki-pilot.service
```

旧目录、旧数据库、14B/27B/8B 模型和迁移前备份至少保留七天；清理必须在人工确认新环境稳定后另行执行。
