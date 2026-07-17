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
