# 多模态论文 RAG：第一版部署与实验

本轮把已有 Knowledge Agent 的入库、语义分块、向量索引和检索接到 Qwen3-VL，新增页面图片生成接口。模型没有微调。原来的单页 oracle 实验仍在独立目录。

**当前状态（2026-09-22）：已恢复 PostgreSQL 16 + pgvector + 余弦距离。** 下文首轮 SQLite 记录属于历史实验，不能代表原项目的 pgvector 检索效果。恢复操作和校验见本文末尾。

## 目录与来源

- 本地代码：`D:\LLM_wiki\.worktrees\internal-pilot`，包括部署前已有的未提交修改。
- 来源提交：`aeb2a993a4c1da32192555a812ddd220f57f9ea8`；不能用这个提交号代替实际工作树快照。
- 初始代码包 SHA-256：`169e33b676b52e01ccc9055e31d06a48d6ae1c47e5c57443c5fce57ff9931cd1`。新增多模态代码随后单独同步。
- 服务器：`/root/autodl-tmp/llm_wiki_multimodal_20260922`。
- 原始论文与目录：`runtime/corpus/papers/`、`runtime/corpus/papers.json`。
- 当前数据库：PostgreSQL `llm_wiki_multimodal`；旧 `runtime/multimodal.db` 保留作历史数据和回退来源。
- canonical 产物：`runtime/data/parsed/`；实际召回页图片：`runtime/data/cache/multimodal_pages/`。
- 开发集结果：`runtime/multimodal_dev_v1/`。

本地原目录保留，服务器使用复制部署。`.env`、密钥、原数据库和历史缓存没有从本地迁移。新配置来自 `deploy/multimodal-pilot.env.example`。

## 实际执行的链路

1. 校验 7 篇公开论文的 PDF SHA-256。
2. 调用原项目 `IngestionPipeline` 注册文档；通过 `IngestionStageRunner` 同步运行 parse → repair → canonicalize → semantic_split → contextualize → embed → index → activate，保留质量门和版本检查点。
3. 第一版选用现有 PDF 文本层解析器；MinerU 和视觉解析暂时关闭。原项目语义 Parent/Child 分块继续运行，文本向量使用 Qwen3-Embedding-4B、2560 维，存入 SQLite-vec。
4. 调用原项目 `QueryService.retrieve_evidence`，不传金标准文档/页面作用域。按照原检索顺序从 source spans 映射物理页码，去重后最多取 3 页。
5. 为这些页面提取全文和渲染图片。文本与图文条件共用同一次检索、同一页集合和同一系统提示词，唯一的输入差别是是否附加页面图片。
6. Qwen3-VL-8B-Instruct 使用 BF16、SDPA、贪心解码、单页最大 802816 像素、最多 512 个输出 token。超出 16384 token 上下文直接报错，不静默截断。

这是原项目检索链路上的新增生成分支。实验入口是 `/api/multimodal/query`；原 `/api/query` 和 Agent/Ollama 的生成配置未切换到这个分支。当前尚未加入图像向量检索、独立重排模型、图表裁剪或 Agent 多轮综合。

## 服务与运行

在服务器工作目录执行：

```bash
.venv/bin/python scripts/launch_multimodal_pilot.py model
.venv/bin/python scripts/launch_multimodal_pilot.py api
.venv/bin/python scripts/launch_multimodal_pilot.py eval
```

启动器会检查已有 PID，后台运行并写 `runtime/model.log`、`runtime/api.log`、`runtime/eval.log`。评测会等待本地 Embedding 服务可用，再入库和推理；不会覆盖已存在的预测文件。

模型服务使用已有 CUDA 环境 `/root/autodl-tmp/qwen3vl_baseline_20260922/venv`；项目使用独立 Python 3.11 `.venv`。模型服务只监听 `127.0.0.1:18080`，项目 API 只监听 `127.0.0.1:18002`。

在服务器访问 `http://127.0.0.1:18002/docs`，找到 `POST /api/multimodal/query`，请求示例：

```json
{"project_slug":"multimodal-pilot","question":"论文 Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks 的 Figure 1 包含哪两个主要组成部分？","mode":"text_image","max_pages":3}
```

在本机查看文档，可通过 SSH 转发 `18002` 后打开 `http://127.0.0.1:18002/docs`。服务未暴露公网；认证关闭只适用于这个隔离的本地监听实验。

## 评测口径

- 只运行开发集 18 题，每题文本/图文两种输入，共 36 次。测试题不参与本轮问答或调参；7 篇论文均可作为检索语料。
- 检索和生成只接收问题，不接收参考答案、题型标签或金标准页码。
- 原问题中的“给定证据页面/该页”替换为“检索到的页面”；完整问题与实际证据均留档。
- 金标准页仅在输出后用于召回诊断。原先单页证据不足题的标签不自动迁移为多页拒答标签。
- 输出检查分别记录 JSON schema 和引用 ID 合法性，不以引用 ID 合法推断事实正确。
- 这轮和单页 oracle 在证据数、提示词、解析方式等方面不同，不能直接把两者分数差异归因为图片效果。
- GPU 峰值包含驻留 Embedding 模型，不应直接与此前单模型显存相减比较。

运行结束后：

```bash
.venv/bin/python scripts/summarize_multimodal_pilot.py runtime/multimodal_dev_v1
```

结果包括 `RUN_REPORT.md`、`run_summary.json`。答案级评分应阅读实际 `retrievals.jsonl` 证据后另行记录。

## 验证与后续

新增测试与原 RAG adapter 测试在本地和服务器各通过 16 项，覆盖物理页码、证据去重排序、项目隔离、成对输入一致性、无证据拒答、布尔类型和非法引用。

先分析本轮漏召回和读图错误，再决定下一轮改检索、图表裁剪或数据集。只有证据已召回而模型稳定答错时，才准备独立训练集做 LoRA/QLoRA。当前评测题不用于训练。

## 2026-09-22 首轮实测

- 7 篇论文、146 页、363 个 Child 块全部 ready；开发集 18 题 × 2 种输入，共 36/36 次完成。
- JSON 协议合格 36/36；引用 ID 合法 35/36。未给出答案准确率，因为需要按实际多页证据重新审核。
- 12 道原可回答题的指定参考页在完整证据包命中 1/12，最终最多 3 页命中 0/12；这不是答案正确率为零。
- 数据库中 363 个块均为 narrative，没有一个通过原检索器要求的 Markdown 表格检查。6 道表格题的目标页虽已入库，却全部不能通过表格候选过滤，最终证据包仅剩 profile_term 补充结果。
- 另有页面截断和论文路由问题。下一轮应先修解析/检索兼容性、验证页级重排和术语补充分数，再考虑微调。
- `POST /api/multimodal/query` 已完成端到端冒烟检查：HTTP 200，实际输入 3 张页面图片。接口检查不等于答案质量验收。

详细诊断见服务器 `runtime/multimodal_dev_v1/FINDINGS.md`，接口记录见同目录 `API_SMOKE.json`。本地同步结果在 `D:\项目复现\multimodal_eval\llm_wiki_rag_dev_v1`。首轮预测和检索轨迹保留不变。

模型/API 服务仍在运行，批量评测已结束；服务器不会自动关机或停止计费。

## 恢复原项目的 PostgreSQL / pgvector 后端

已安装 PostgreSQL 16.15 和 pgvector 0.8.6，执行项目 Alembic 到 `a4e2c7f90120`。数据库仅监听 localhost；本实例应用由操作系统 root 用户运行，经 Unix socket peer 认证连接同名非超级用户角色 root，没有启用公网数据库或 trust 认证。

当前配置（仅本租赁实例）：

```dotenv
DATABASE_URL=postgresql+psycopg://root@/llm_wiki_multimodal
VECTOR_STORE_ENABLED=true
VECTOR_STORE_BACKEND=pgvector
```

通过 `scripts/migrate_multimodal_to_pgvector.py` 迁移：17 张业务表内容哈希一致，包含 7 篇论文、483 个总分块（120 Parent、363 Child）、7 个解析版本和 7 个 pipeline runs。未重算 Embedding；363 个现有 Child 向量重新写入 pgvector 索引。迁移脚本仅接受空目标库，对自引用关联分两阶段恢复，并保留原始更新时间戳；数据与向量索引在同一事务中提交。

`search.py` 和 `vector_store.py` 均保持部署前哈希，未修改词法路由、profile_term 权重、表格过滤或最多 3 页规则。实际 `PGVectorStore.search` 距离与手工计算 `1-cosine` 的最大误差约 `2.17e-7`，确认没有继续使用 SQLite，也没有用 JSON 向量回退冒充 pgvector。

恢复后对冻结的 18 道开发题只重跑检索：原可回答题最终参考页命中 **2/12**（首轮为 0/12）。另做 1 次图文 API 冒烟检查：HTTP 200、3 张页面图片、JSON 与引用 ID 检查通过。没有重跑完整 36 次答案生成，不报告新的答案准确率。38 项向量检索、多模态和 RAG adapter 测试通过。

备份和校验文件：

- `.env.sqlite-20260922.bak`：切换前配置。
- `runtime/backups/sqlite_before_pgvector_20260922_final/multimodal.sqlite`：完整 SQLite 快照。
- 同目录 `migration_report.json`：17 张表的数量与哈希对照。
- `runtime/backups/pgvector_restored_20260922.dump`：恢复后的 PostgreSQL 自定义格式备份，已下载到本地。
- `runtime/pgvector_restored_20260922/`：检索轨迹、余弦距离验证和 API 检查。对应本地 `D:\项目复现\multimodal_eval\pgvector_restored_20260922`。

PostgreSQL 数据目录为 `/var/lib/postgresql/16/main`。如实例重启后数据库未启动，可执行 `pg_ctlcluster 16 main start`，再启动 API；不要重复执行迁移脚本覆盖数据库。更换实例前应保留 PostgreSQL dump、原始 PDF 和 canonical/cache 文件。旧 SQLite 与首轮预测均未删除；恢复旧配置只适合尚无新增 PostgreSQL 写入的回退，后续有新增数据时须先做数据对齐。

首轮诊断遗漏了部署后端不一致这一点：SQLite-vec 默认返回 L2 距离，但原排序公式按 pgvector 余弦距离计算，造成向量候选和 JSON 回退候选分数不一致。本次通过恢复原后端解决，不修改原排序公式。表格解析兼容性与页级排序是剩余问题，应另外验证。
