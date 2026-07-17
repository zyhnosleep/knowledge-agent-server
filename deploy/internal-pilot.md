# Knowledge Agent 双环境部署

服务器使用两套 Systemd user service 栈。开发环境固定 GPU 0，测试环境固定 GPU 1；两套数据库、文件、日志、端口和运行配置互不共享。

## 目录

```text
/home/zhangyh/knowledge-agent-dev
/home/zhangyh/knowledge-agent-test
/home/zhangyh/knowledge-agent-models
```

从 `.env.development.example` 和 `.env.test.example` 创建各自的 `runtime/app.env`，设置权限为 `600`。测试配置包含认证 secret，禁止输出到日志或提交到 Git。

## 服务

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/knowledge-agent-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now \
  knowledge-agent-dev-ollama.service \
  knowledge-agent-dev-api.service \
  knowledge-agent-test-ollama.service \
  knowledge-agent-test-api.service \
  knowledge-agent-test-tunnel.service
```

两个 Ollama 服务共享只读模型缓存，但各自绑定 GPU 和端口。两边均使用 `qwen3.5:9b`、`qwen3-embedding:4b`、32K、Flash Attention、q8_0 KV Cache、并发 1 和 5 分钟闲置释放。

## 验收

```bash
curl -fsS http://127.0.0.1:8002/api/health
curl -fsS http://127.0.0.1:8001/api/health
systemctl --user is-active \
  knowledge-agent-dev-api.service knowledge-agent-dev-ollama.service \
  knowledge-agent-test-api.service knowledge-agent-test-ollama.service \
  knowledge-agent-test-tunnel.service
```

开发 API 只允许 SSH 转发。Cloudflare Tunnel 只能指向测试 API，不得暴露开发 API 或 Ollama 端口。测试迁移必须先启用维护页、备份数据库与文件、重建 2560 维向量并通过固定 20 题验收。
