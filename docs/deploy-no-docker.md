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
QUEUE_JOB_TIMEOUT=3600
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
DOCUMENT_INTELLIGENCE_ENABLED=true
PDF_RENDER_DPI=160
OCR_FALLBACK_ENABLED=false
MINIO_ENABLED=false
EXTERNAL_API_ENABLED=false
```

说明：

- `REDIS_URL` 必填；这版 worker 依赖 Redis。
- `MINIO_ENABLED=false` 时，文件只走本地目录。
- `SQLite` 适合当前轻量协作，不适合高并发写入。
- 所有命令示例和 `scripts/*.sh` 都按 POSIX `sh` 兼容方式编写，不使用 `source`。
- 当前版本新增了 PDF 页面渲染依赖；更新代码后请重新执行一次 `pip install -e .`。
- `QUEUE_JOB_TIMEOUT` 应明显大于 `OLLAMA_REQUEST_TIMEOUT`；如果 PDF 或多模态摄入很慢，可以继续把 `QUEUE_JOB_TIMEOUT` 调到 `7200`。

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
3. 检查 `data/raw/` 和 `data/wiki/`
4. 检查项目目录下的 `index.md` 和 `log.md`
5. 发起 1 次问答，确认返回 `citations`
6. 重复上传同一文档，确认去重生效

## 8. Obsidian 协作方式

这版服务器可以和 Obsidian 配合，但两者职责不同：

- 服务器负责上传、摄入、抽取、问答、`wiki/queries/` 写回
- `data/wiki/<project>/` 是主知识工件目录
- Obsidian 适合在本地打开同步下来的 `wiki/` 目录，做人类阅读、策展、双链浏览

需要注意：

- 直接上传 `.md` 文档是推荐输入方式之一
- 服务器内部仍会做 snippet/chunk 处理，用于证据定位和 fallback 检索
- 查询主路径已经收敛到 `wiki-first`

更详细的协作说明见 [docs/obsidian-workflow.md](D:\LLM_wiki\docs\obsidian-workflow.md)。

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
