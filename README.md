# Knowledge Agent

Knowledge Agent 是一个面向团队内部使用的文献知识与证据问答系统。它保存原始文档、结构化分块、向量索引、页码和来源链接，并由后端验证 Agent 回答中的引用。

## 模型

- 回答生成：`qwen3.5:9b`
- 向量检索：`qwen3-embedding:4b`（2560 维）
- 上下文：32K
- 本地推理：Ollama、Flash Attention、q8_0 KV Cache
- 闲置策略：最后一次调用 5 分钟后释放 GPU
- 外部 API：默认关闭

## 环境

同一台双 GPU 服务器运行两套完全隔离的环境：

| 环境 | API | Ollama | GPU | 访问 |
|---|---:|---:|---:|---|
| 开发 | `127.0.0.1:8002` | `127.0.0.1:11435` | GPU 0 | SSH 转发 |
| 测试 | `127.0.0.1:8001` | `127.0.0.1:11436` | GPU 1 | 受保护的公网入口 |

开发环境使用 `.env.development.example`，测试环境使用 `.env.test.example`。正式部署时复制到各自的 `runtime/app.env` 并设置数据库密码和认证 secret，禁止提交实际 secret。

## 本地开发

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
Copy-Item .env.development.example .env
.venv\Scripts\pytest -q
.venv\Scripts\python -m uvicorn app.main:app --app-dir src --host 127.0.0.1 --port 8002
```

## 数据迁移

更换 Embedding 模型后必须在维护模式下重建向量：

```bash
.venv/bin/alembic upgrade head
.venv/bin/python scripts/reindex_embeddings.py \
  --all-projects \
  --maintenance-confirmed \
  --report runtime/reindex-report.json
```

原始文件、解析文本、页码和项目关系不会被删除。详细部署及回滚流程见 `deploy/internal-pilot.md`。
