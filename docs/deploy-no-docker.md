# Knowledge Agent 用户态部署

Knowledge Agent 在无 sudo 的 Linux 服务器上通过 Systemd user services 运行。当前标准拓扑是双环境：开发环境使用 GPU 0、端口 8002/11435；测试环境使用 GPU 1、端口 8001/11436。

## 准备

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
mkdir -p runtime/data/raw runtime/data/cache runtime/tmp
```

开发环境从 `.env.development.example` 创建 `runtime/app.env`，测试环境从 `.env.test.example` 创建。数据库密码、认证 secret 和 Tunnel 凭据只能写入服务器文件，并设置权限 `600`。

## 模型

```bash
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen3.5:9b
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen3-embedding:4b
```

模型缓存可由两个 Ollama 进程共享；业务数据库、文件和日志不得共享。服务模板及完整验收步骤见 `deploy/internal-pilot.md`。

## 数据库

```bash
.venv/bin/alembic upgrade head
```

从旧 Embedding 切换后，必须在维护模式运行 `scripts/reindex_embeddings.py` 并验证全部 chunk 为 2560 维。不得只修改模型名而继续使用旧向量。
