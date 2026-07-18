# Knowledge Agent Redis 异步摄入设计

## 目标

为同一服务器上的开发、测试环境补齐可运行的 Redis 与 RQ worker，使上传请求快速返回，文档解析、embedding 和索引写入由后台任务执行，同时保持两个环境的数据与任务隔离。

## 架构

- 部署一个仅监听 `127.0.0.1:6379` 的 Redis 7 实例，不开放公网。
- 开发环境使用 Redis DB 1，测试环境使用 Redis DB 2。
- 开发、测试各自运行一个 RQ worker，并连接各自的 Redis DB；两边可以使用相同的 `ingest` 队列名，但任务不会跨 DB 混用。
- API 只负责保存上传文件、创建数据库记录并入队，随后立即返回任务标识与排队状态。
- worker 从对应队列取出任务，沿用现有 `run_document_ingestion` 流程完成解析、embedding、pgvector 写入及状态更新。
- worker 使用环境自己的 `runtime/app.env`、数据库、数据目录和 Ollama 地址；开发固定 GPU0/11435，测试固定 GPU1/11436。

## 运行组件

- `knowledge-agent-redis.service`：以本机 Docker 启动 Redis 7，绑定回环地址并使用独立持久卷。
- `knowledge-agent-dev-worker.service`：在开发代码目录运行 RQ worker，读取开发环境配置。
- `knowledge-agent-test-worker.service`：在测试代码目录运行 RQ worker，读取测试环境配置。
- 两套 `.env` 模板恢复各自的 `REDIS_URL`：开发 `/1`，测试 `/2`。
- API 与 worker 服务显式依赖 Redis 服务；worker 异常退出后由 systemd 自动重启。

## 数据与安全

- Redis 只保存任务元数据与序列化参数，不保存知识库正文作为权威副本。
- Redis 端口只绑定 `127.0.0.1`，Cloudflare/Caddy 不代理 Redis。
- 开发与测试使用不同 Redis DB、数据库和数据目录，避免任务或文档串环境。
- 迁移和回滚时保留原数据库、原文件快照及失败报告；停止新 worker 即可阻止后台写入。

## 错误处理

- Redis 不可用时，配置了 `REDIS_URL` 的环境应明确返回入队失败，不静默改为同步执行。
- worker 任务失败继续使用现有 pipeline run/document 状态记录，便于页面展示失败原因和人工重试。
- Redis 和 worker 的 systemd 服务使用 `Restart=on-failure`；API 启动不等价于异步摄入就绪，部署验收必须单独检查 Redis ping 和 worker 注册。

## 验收

1. 部署配置测试证明 Redis 仅监听本机，并存在开发、测试两个 worker 服务。
2. 两套环境模板分别使用 Redis DB 1 和 DB 2。
3. 上传接口在任务入队后快速返回，worker 随后把文档推进到 `completed/ready`。
4. 开发任务不会出现在测试 Redis DB，测试任务也不会进入开发 DB。
5. Redis、worker 重启后可继续处理新任务，现有数据库和文件不受影响。
6. 完成真实文档上传、2560 维 embedding、问答与引用验收后再恢复测试公网入口。

## 非目标

- 本次不建设 Redis 高可用集群、哨兵或跨服务器队列。
- 本次不开放 Redis 公网端口，也不把会话或知识正文迁移到 Redis。
- 本次不增加 worker 横向扩容；每个环境先使用一个摄入 worker，后续根据并发测试再调整。
