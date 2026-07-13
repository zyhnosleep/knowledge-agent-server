# 无 Docker 服务器部署

这份文档面向 Linux 用户态服务器环境，目标是在 **不使用 Docker、不依赖 sudo** 的前提下，把 `LLM Wiki Server` 以“双进程 + Redis + SQLite + 本地文件存储”的方式跑起来。

## 部署形态

- API：`uvicorn app.main:app`
- Worker：`python -m app.workers.runner`
- 队列：用户目录自建 `Redis`
- 数据库：`SQLite`
- 文件存储：本地 `data/`
- 模型服务：宿主机 `Ollama`，监听 `127.0.0.1:11435`
- 进程托管：`nohup`

## 目录约定

下面命令假设项目上传到：

```sh
~/llm_wiki_server
```

## 1. 上传项目并创建虚拟环境

```sh
cd ~/llm_wiki_server
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .
# 如果要启用 MinerU PDF 解析，再安装；当前项目 extra 使用 mineru[all]>=3.4,<3.5：
# pip install -e ".[mineru]"
chmod +x scripts/*.sh
mkdir -p logs run data
```

验证基础代码：

```sh
python -m compileall src
pytest
```

## 2. 生成服务器环境变量

直接复制模板：

```sh
cp .env.server.example .env
```

推荐值如下：

```dotenv
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=8000
DATABASE_URL=sqlite:///./data/app.db
REDIS_URL=redis://127.0.0.1:6379/0
QUEUE_JOB_TIMEOUT=9000
DATA_DIR=./data
RAW_DIR=./data/raw
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

说明：

- `REDIS_URL` 必填；这版 worker 依赖 Redis。
- `MINIO_ENABLED=false` 时，文件只走本地目录。
- `SQLite` 适合当前轻量协作，不适合高并发写入。
- 所有命令示例和 `scripts/*.sh` 都按 POSIX `sh` 兼容方式编写，不使用 `source`。
- 当前版本新增了 PDF 页面渲染依赖；更新代码后请重新执行一次 `pip install -e .`。
- 如果启用 MinerU，请执行 `pip install -e ".[mineru]"`，当前项目 extra 会安装 `mineru[all]>=3.4,<3.5`，并把 `MINERU_ENABLED=true` 写入 `.env`。
- `QUEUE_JOB_TIMEOUT` 应明显大于 `OLLAMA_REQUEST_TIMEOUT` 和 `MINERU_TIMEOUT`；如果 PDF 或多模态摄入很慢，可以继续调高。

### 可选：启用 MinerU PDF 解析

服务器硬件如果有独立 GPU，推荐把 MinerU 作为 PDF 解析第一优先级：

```sh
cd ~/llm_wiki_server
. .venv/bin/activate
pip install -e ".[mineru]"
```

然后修改 `.env`：

```dotenv
MINERU_ENABLED=true
MINERU_BIN=mineru
MINERU_BACKEND=pipeline
MINERU_MODEL_SOURCE=modelscope
MINERU_OUTPUT_DIR=./data/cache/mineru
MINERU_TIMEOUT=7200
MINERU_EXTRA_ARGS=
```

说明：

- MinerU 通过 CLI 子进程运行，解析成功后读取 `content_list_v2.json` / `content_list.json`。
- MinerU 失败、超时或输出缺失时，会自动回退到现有 Ollama Vision PDF 链路。
- 对 3090 24GB 这类单卡服务器，建议 MinerU 和 Ollama 顺序使用 GPU，不要同时跑大任务。
- 大批量解析 PDF 前后可以用 `nvidia-smi` 查看显存占用。

## 3. 在用户目录准备 Redis

如果服务器已经有你可用的 Redis，可以直接把 `.env` 里的 `REDIS_URL` 指向现成实例。

如果没有，就在用户目录自建一个：

```sh
mkdir -p ~/local/src ~/local/redis
cd ~/local/src
curl -L -o redis-7.4.0.tar.gz https://download.redis.io/releases/redis-7.4.0.tar.gz
tar -xzf redis-7.4.0.tar.gz
cd redis-7.4.0
make -j"$(nproc)"
make PREFIX="$HOME/local/redis" install
```

启动 Redis：

```sh
cd ~/llm_wiki_server
./scripts/start_redis.sh
```

默认行为：

- 绑定 `127.0.0.1`
- 端口 `6379`
- 日志写入 `logs/redis.log`
- PID 写入 `run/redis.pid`

如果你的 Redis 可执行文件不在默认位置，先设置：

```sh
export REDIS_SERVER_BIN="$HOME/local/redis/bin/redis-server"
```

## 4. 启动新版 Ollama 到 11435

这版部署假设不碰系统原有 `11434`，而是在用户态启动新版 Ollama 到 `127.0.0.1:11435`。

如果你已经把新版 Ollama 放到用户目录，例如：

```sh
$HOME/local/ollama/bin/ollama
```

直接启动：

```sh
cd ~/llm_wiki_server
./scripts/start_ollama.sh
```

如果二进制路径不同，先设置：

```sh
export OLLAMA_BIN="$HOME/local/ollama/bin/ollama"
```

拉取模型：

```sh
export OLLAMA_HOST=127.0.0.1:11435
"$OLLAMA_BIN" pull qwen3.6:27b
"$OLLAMA_BIN" pull qwen3-embedding:8b
```

## 5. 启动 API 和 worker

```sh
cd ~/llm_wiki_server
./scripts/start_api.sh
./scripts/start_worker.sh
```

默认行为：

- 自动读取项目 `.env`
- 使用 `.venv`
- API 日志写入 `logs/api.log`
- worker 日志写入 `logs/worker.log`
- PID 分别写入 `run/api.pid` 和 `run/worker.pid`

## 6. 验证服务状态

```sh
cd ~/llm_wiki_server
./scripts/status.sh
```

也可以手工检查：

```sh
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/docs
```

摄入任务等待期间，可以在另一个终端查看进度：

```sh
cd ~/llm_wiki_server
. .venv/bin/activate
python scripts/watch_ingest_progress.py
```

也可以直接看 worker 阶段日志：

```sh
tail -f logs/worker.log
```

访问入口：

- 控制台：`http://<server-ip>:8000/`
- API 文档：`http://<server-ip>:8000/docs`
- 健康检查：`http://<server-ip>:8000/api/health`

## 7. 首轮联调建议

按下面顺序做：

1. 上传 1 个 `txt` 或 `md` 文档
2. 确认 worker 消费队列
3. 检查 `data/raw/` 和文档分块记录
4. 检查文档解析状态和向量索引
5. 发起 1 次问答，确认返回 `citations`
6. 重复上传同一文档，确认去重生效

## 8. 文档与检索

- 服务器负责上传、摄入、抽取和问答。
- 原始资料保存在 `data/raw/`，检索证据来自 `DocumentChunk` 和向量索引。
- 查询主路径采用文档证据优先的 RAG，并返回可追溯 citation。

## 9. 常见问题

### worker 启动失败

通常是 Redis 没起来，先检查：

```sh
cat logs/redis.log
cat logs/worker.log
```

确认 `.env` 里的：

```dotenv
REDIS_URL=redis://127.0.0.1:6379/0
```

### 文档摄入失败

先看：

```sh
cat logs/api.log
cat logs/worker.log
```

常见原因：

- Ollama 没起来
- `OLLAMA_BASE_URL` 端口不对
- 模型还没拉取

### SQLite 锁冲突

这版使用 SQLite，适合当前轻量协作环境；如果后续并发明显增加，再升级到 PostgreSQL。

### 外部复核暂时不用

保持：

```dotenv
EXTERNAL_API_ENABLED=false
```

## 10. 可回退路径

如果后续服务器条件恢复，这版可以平滑升级回：

- PostgreSQL
- MinIO
- Docker Compose
- 正式进程托管
