# LLM Wiki Server

`LLM Wiki Server` 是一个面向服务器部署的内部知识库骨架，延续 `llm_wiki` 的 `raw -> wiki -> schema` 思路，并补上共享服务所需的数据库、任务队列、审核流和外部核查接口。

当前版本已经具备这些基础能力：

- 文档上传、去重、解析与分块
- 使用 Ollama 做本地抽取、摘要、嵌入与问答
- 使用 OpenAI 兼容 API 做可选的结构化复核
- 将数据库内容投影为 `raw/`、`wiki/`、`index.md`、`log.md` 结构
- 提供 FastAPI 接口和轻量 Web 控制台
- 提供 Redis worker 骨架，支持异步摄入

当前输入格式支持说明：

- `md` / `txt`：当前最推荐，也最稳定
- `html`：支持正文抽取
- `docx`：支持段落文本提取
- `pdf`：支持 MinerU 可选解析、页面渲染 + 多模态文档理解 + 文本层融合，但**仍需服务器实机验收**

需要特别说明：

- `pdf` 解析已经升级为“可选 MinerU 结构化解析 + 文本层质量检测 + 页面渲染 + 多模态逐页理解 + 结果融合”
- `MINERU_ENABLED=true` 时，系统会先调用本地 `mineru` CLI；失败时自动回退到现有多模态/Ollama 链路
- 当 PDF 文本层质量高时，系统仍会优先保留原始文本精度
- 当遇到扫描版 PDF、图片型 PDF、复杂双栏/表格/公式页时，会优先使用多模态页面理解增强结构保留
- OCR 不是主链，只作为 fail-safe fallback 预留

## 推荐模型

- 在线主模型：`qwen3.6:27b`
- 批处理模型：`qwen3.6:27b`
- 主嵌入：`qwen3-embedding:8b`

## 推荐部署路径

当前推荐优先使用“无 Docker 双进程版”：

- `FastAPI API` 直接运行在宿主机 Python 虚拟环境内
- `RQ worker` 单独运行
- `Redis` 在用户目录自建并监听 `127.0.0.1:6379`
- `SQLite` 作为数据库
- `MINIO_ENABLED=false`，文件直接落本地 `data/`
- `Ollama` 运行在宿主机 `127.0.0.1:11435`
- 用 `nohup` 托管 API 和 worker

这条路径最适合“无 sudo、暂时不用 Docker、先把链路跑通”的服务器环境。

## 无 Docker 快速开始

1. 复制服务器环境变量模板。
2. 创建虚拟环境并安装依赖。
3. 在宿主机准备 Redis 和新版 Ollama。
4. 分别启动 API 和 worker。

Linux 服务器示例：

```sh
cd ~/llm_wiki_server
cp .env.server.example .env
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .
# 如果需要启用 MinerU PDF 解析：
# pip install -e ".[mineru]"
chmod +x scripts/*.sh
mkdir -p logs run
./scripts/start_redis.sh
./scripts/start_ollama.sh
./scripts/start_api.sh
./scripts/start_worker.sh
./scripts/status.sh
python scripts/watch_ingest_progress.py
```

完整步骤见 [docs/deploy-no-docker.md](D:\LLM_wiki\docs\deploy-no-docker.md)。
上面的命令和 `scripts/*.sh` 都按 POSIX `sh` 兼容方式整理，不依赖 `source`。
如需配合本地 Obsidian 使用，见 [docs/obsidian-workflow.md](D:\LLM_wiki\docs\obsidian-workflow.md)。

## 环境变量建议

无 Docker 服务器版建议使用以下核心配置：

```dotenv
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=8000
DATABASE_URL=sqlite:///./data/app.db
REDIS_URL=redis://127.0.0.1:6379/0
QUEUE_JOB_TIMEOUT=9000
DATA_DIR=./data
RAW_DIR=./data/raw
WIKI_DIR=./data/wiki
CACHE_DIR=./data/cache
OLLAMA_BASE_URL=http://127.0.0.1:11435
OLLAMA_GENERATION_MODEL=qwen3.6:27b
OLLAMA_BATCH_MODEL=qwen3.6:27b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:8b
OLLAMA_VISION_MODEL=qwen3.6:27b
OLLAMA_REQUEST_TIMEOUT=600
OLLAMA_KEEP_ALIVE=5m
DOCUMENT_INTELLIGENCE_ENABLED=true
PDF_RENDER_DPI=160
OCR_FALLBACK_ENABLED=false
MINERU_ENABLED=false
MINERU_BIN=mineru
MINERU_BACKEND=pipeline
MINERU_MODEL_SOURCE=modelscope
MINERU_OUTPUT_DIR=./data/cache/mineru
MINERU_TIMEOUT=7200
MINIO_ENABLED=false
EXTERNAL_API_ENABLED=false
```

可以直接从 [.env.server.example](D:\LLM_wiki\.env.server.example) 复制。

## 运行入口

- API：`uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Worker：`python -m app.workers.runner`

常用访问入口：

- API 文档：[http://localhost:8000/docs](http://localhost:8000/docs)
- 健康检查：[http://localhost:8000/api/health](http://localhost:8000/api/health)
- 简易控制台：[http://localhost:8000/](http://localhost:8000/)

## 运行说明

- `worker` 依赖 `REDIS_URL`；无 Docker 双进程版要求 Redis 可用。
- `MINIO_ENABLED=false` 时，原始文件和 Wiki 页面都直接保存在本地 `data/`。
- `SQLite` 适合当前轻量协作和验证阶段，不适合高并发写入。
- `EXTERNAL_API_ENABLED=false` 时，问答和摄入只走本地 Ollama。
- `DOCUMENT_INTELLIGENCE_ENABLED=true` 时，PDF 会优先走页面渲染 + 多模态理解链路。
- `MINERU_ENABLED=true` 时，PDF 会先走本地 MinerU CLI；MinerU 失败或未安装时会自动回退到现有 PDF 链路。
- `OLLAMA_KEEP_ALIVE=5m` 用于避免 Ollama 长时间占用 3090 显存；大批量 MinerU 解析前可临时调成 `0`。
- 更新到当前版本后，需要重新执行 `pip install -e .`，因为新增了 PDF 渲染依赖 `PyMuPDF`。
- 如果启用 MinerU，需要执行 `pip install -e ".[mineru]"`；当前项目 extra 会安装 `mineru[all]>=3.4,<3.5`，并确保服务器可运行 `mineru` 命令。
- `QUEUE_JOB_TIMEOUT` 应明显大于 `OLLAMA_REQUEST_TIMEOUT`；PDF 和多模态摄入通常比 txt/md 慢很多。
- 摄入等待期间可以用 `tail -f logs/worker.log` 看阶段日志，或用 `python scripts/watch_ingest_progress.py` 在终端显示动态进度条。

## Obsidian 协作

- 服务器负责 `ingest / query / verify / writeback`，把原始资料编译为 `wiki/` 下的 Markdown 页面。
- `wiki/` 目录可以同步到本地，直接作为 Obsidian Vault 打开。
- 上传 `.md` / `.txt` 很适合这套流程，但服务器内部仍会做 snippet/chunk 处理，用于证据定位、去重和 fallback 检索。
- 查询主路径是 `wiki-first`，不是传统 `raw chunk first` 的 RAG。

## Docker 说明

仓库仍保留 `Dockerfile` 和 `docker-compose.yml`，便于后续回切容器化部署；但当前推荐的服务器落地路径以无 Docker 版本为主。

## 目录

```text
data/
  raw/          原始文档
  wiki/         Wiki 页面投影
  cache/        中间缓存
src/app/
  api/          FastAPI 路由
  core/         配置与日志
  db/           数据库初始化
  models/       ORM 模型
  schemas/      请求与响应结构
  services/     解析、检索、模型、Wiki 渲染
  workers/      队列 worker
scripts/
  start_redis.sh
  start_ollama.sh
  start_api.sh
  start_worker.sh
  status.sh
  watch_ingest_progress.py
```

## 外部核查

如需开启外部复核，打开 `EXTERNAL_API_ENABLED=true`，并配置：

- `EXTERNAL_API_BASE_URL`
- `EXTERNAL_API_KEY`
- `EXTERNAL_API_MODEL`

系统默认只在高风险问答和知识冲突场景触发外部复核，不做全文双跑。
