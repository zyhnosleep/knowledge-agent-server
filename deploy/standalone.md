# 单实例 pgvector 部署

适用于 Linux/无 Systemd 容器、单 GPU。唯一当前配置为 standalone.env.example；历史双环境模板不再适用。本页不承诺未完成的验收。

## 1. 保护资料与环境

核查实际源码根、PGDATA、tablespace、磁盘与进程。停止集群的冷备包含全部数据及 PG 配置；在线库用 pg_dump，并用 pg_restore --list 验证可读。保留源码/config/hash、版本和回退清单。禁止 initdb、pg_resetwal、删 postmaster.pid、覆盖数据库/runtime 或盲跑升降维 SQL。

API 使用 Python 3.11+，安装本项目；测试加 [dev]。GPU 模型使用独立环境，按实测版本单独锁定，不能混装 API transformers<5。没有实测 lock 时不要声称已可重建。

2048 维 vector 先验收精确余弦排序，不承诺普通 vector HNSW 支持此维度。schema 以库内 Alembic revision 为准。

## 2. 私有配置

将模板复制为项目根 .env（0600），按实际安装修改：

- PROJECT_API_PYTHON / LOCAL_MODEL_PYTHON：两套解释器绝对路径。
- LOCAL_MODEL_IMAGE_EMBED / LOCAL_MODEL_CHAT / 可选 CHAT_ADAPTER：存在的绝对资产目录；留空 adapter 即 base。
- OLLAMA 模型名与服务 alias 一致；生成、视觉、综合、批处理/上下文化使用当前端点。文本/图嵌入使用同一 VL2B 2048 空间。
- DATA_DIR / RAW_DIR / CANONICAL_ARTIFACTS_DIR / CACHE_DIR 位于本项目 runtime/data；IMAGE_ROOTS 只能配置 parsed/cache 实际目录，宽根/越界 symlink 拒绝。
- SEMANTIC_TOKENIZER_LOCAL_PATH 是已验证的固定 tokenizer 快照，仅用于 token 计数，不是第二个检索模型，不能用 remote latest/字节估计代替。
- EMBEDDING_REVISION / PROCESSOR_HASH 取自已加载模型的 /api/embedding_identity（含 loader/软件版本），绝不补写历史向量身份。
- DATABASE_URL 使用 postgresql+psycopg；VECTOR_STORE_BACKEND=pgvector、ENABLED=true、STRICT=true。

AUTH_ENABLED=false 仅限 loopback + SSH 转发。本控制器即便开启 auth 也不开放公网，公网发布需另行审核。不要把 .env/密码写到日志或 Git。AGENT_ADAPTIVE_ENABLED 默认 false。

## 3. 首次 bootstrap

preflight 阻断缺身份、混空间、active 状态/指针不一致或不完整索引，不可绕过该门直接起 API。运维可受控先启动同一模型入口获取真实 identity；新建 shadow 版本，重建文字/真实像素向量、核对 canonical 质量/覆盖/排序后才原子激活。旧版本/索引/备份保留；无法修复的质量门明确报告，不修改 status/manifest 强行通过。

scripts/reindex_embeddings.py 是历史纯文本工具，不用于此多模态重建；SQLite→PG 工具只用于明确的一次性迁移。重建不是训练。

## 4. 生命周期

从项目根用 API 解释器：

```bash
.venv/bin/python scripts/project_ctl.py preflight
.venv/bin/python scripts/project_ctl.py start
.venv/bin/python scripts/project_ctl.py status
.venv/bin/python scripts/project_ctl.py stop
```

可指定 --root /absolute/project；输出脱敏 JSON，失败 exit 1。

start：检查两个端口归属 → pg_ctlcluster 16 main（不重建）→ strict DB contract → 模型 → API。跨进程锁、等待 exec 稳定、独立 subprocess session；重复 start 不重复启动，陌生端口停止。健康 HTTP 不是进程归属。

status 只读，不创建目录/起进程/修库。stop 只停止记录且当前 PID/start_time/cwd/exe/argv 匹配的 API/模型；Linux 支持时用 pidfd 避免 signal 时 PID 复用，并等待确认退出。不停 PG/陌生模型/Redis/worker。修改运行中 argv/解释器配置后可能拒绝 stop；先恢复匹配配置，不能模糊 pkill。

runtime/control 为私有状态/log（0700/0600），model.json/api.json、model.log/api.log。PROJECT_START_TIMEOUT 只控制启动，不扩大查询预算。controller 不提供崩溃自愈/容器启动自启；重启后执行 start 并复查 status。

## 5. 入库与可选业务

REDIS_URL 空则同步入库。配置本地受保护 Redis 时，另行加载同一 .env 运行：

```bash
.venv/bin/python -m app.workers.runner
```

worker 并发 1，保持 parse/repair/canonicalize/semantic_split/contextualize/embed/index/activate 阶段队列；worker/Redis 不归 controller 管理，不以 API 健康代替入库完成。

飞书可选，本次不启动/复制公司凭据。旧 multimodal 旁路仅保留历史复现，正式验收使用 /api/query、/api/agent/query 和 stream；旁路成功不能代替普通入口成功。

## 6. 验收与回退

跑完整 suite，在 Linux 补跑 Windows 跳过的 symlink/真实 GPU 用例。PG 距离和过滤排序与手算 cosine 一致，trace 报告实际 backend，不从配置推导。

普通链路测文本、跨轮、换题、表格、像素、拒答及独立小文档全入库。动态链路使用同 base/索引/版本/预算的配对题，逐题判事实正确/引用/适当拒答和成本；结构 passed 不是正确率，未判留空。脱 SSH 新连接复查存活、重复 start、精确 stop/start。

仅收益不足可以默认关闭动态开关并报告；安全/身份/普通回归失败阻断发布。回退先关 AGENT_ADAPTIVE_ENABLED，必要时按清单恢复代码/config；不覆盖数据/模型/数据库、不自动混用旧 embedding。
