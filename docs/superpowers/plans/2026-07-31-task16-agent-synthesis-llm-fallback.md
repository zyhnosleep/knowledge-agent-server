# Task 16：Agent 合成、LLM Fallback 修复与系统稳定性

> 基于 full30 测试诊断结果，修复三个核心问题并重新验证。

## 决策汇总

- Agent 合成第二轮：润色 + 事实核查 + 跨文档综合，使用本地 Ollama qwen3.5:9b
- LLM fallback：失败重试 2 次（间隔 5s）+ 降级答案加 `[系统提示]` 标记
- Worker/API 冲突：排查服务器根因并修复
- full30：修完所有问题后重跑，两轮（RAG 裸答 + Agent 综合）分别评估
- 硬边界：不改 test 环境，GPU1 不用，旧数据不删

## Step 1：修复 LLM Fallback（search.py）

- [ ] 在 `_draft_answer()` 中给 `safe_model_call` 加重试逻辑：失败后重试最多 2 次，每次间隔 5 秒
- [ ] 降级答案（fallback_text）开头加 `[系统提示：LLM 生成暂时失败，以下为原始检索证据，仅供参考]`
- [ ] 写 TDD 测试：模拟 Ollama 超时 → 验证重试 2 次 → 验证降级标记
- [ ] 写 TDD 测试：模拟 Ollama 第 3 次成功 → 验证返回 LLM 答案无标记
- [ ] 运行聚焦测试，全部通过

## Step 2：启用 Agent Ollama 合成（agent_synthesizer.py + config）

- [ ] 在 `AgentSynthesizer` 中新增 `ollama` provider：
  - 构建合成 prompt：包含原始问题、RAG 答案、全部 citation 证据、conversation summary
  - System prompt 要求：润色语言、核实事实与 citation 一致性、跨文档整合对比、补遗漏数据点
  - 调用 `ollama.generate_structured()` 生成最终答案
- [ ] 在 `_resolve_provider()` 中新增 `ollama` 选项
- [ ] 把服务器 `AGENT_SYNTHESIS_PROVIDER` 从 `local` 改为 `ollama`
- [ ] 保留 `safe_model_call` 包裹：合成失败时回退到 `local`（直接透传 RAG 答案），并打 warning 日志
- [ ] 写 TDD 测试：验证 ollama provider 成功生成含 citation 的合成答案
- [ ] 写 TDD 测试：验证 ollama 失败时 fallback 到 local
- [ ] 运行聚焦测试，全部通过

## Step 3：排查并修复 Worker/API 冲突（服务器端）

- [ ] SSH 到服务器，精确诊断：
  - 单独启动 API（worker 停）观察 5 分钟 → 确认 API 稳定
  - 单独启动 Worker（API 停）观察 5 分钟 → 确认 Worker 稳定
  - 同时启动两者 → 观察崩溃模式，抓 systemd journal 和 dmesg
  - 检查：OOM killer？GPU 争抢？Redis 依赖链？systemd restart storm？
- [ ] 根据诊断结果修复（可能是资源限制、systemd unit 依赖、或进程启动时序问题）
- [ ] 写 TDD 测试验证修复（如 systemd unit 配置检查）
- [ ] 验证：API + Worker 同时运行 5 分钟不重启

## Step 4：更新评估脚本支持 Agent 全流程（evaluate_canonical_retrieval.py）

- [ ] 新增 `run_agent_acceptance()` 函数：走完整 Agent 流程（`AgentExecutor.execute()`）
- [ ] 新增 `--mode` 参数：`rag-only`（默认）/ `agent-full` / `both`
- [ ] `both` 模式下分别报告 RAG 裸答和 Agent 综合的分数
- [ ] Agent 评估指标：answer_passed、citation_passed、synthesis_applied（是否经过了第二轮）
- [ ] 保留 `required_terms` 为参考指标不入门禁（延续 v6 改动）
- [ ] 写 TDD 测试
- [ ] 运行聚焦测试，全部通过

## Step 5：同步代码到服务器并重跑 full30

- [ ] 本地全量测试通过
- [ ] `git diff --check` 无空白错误
- [ ] 同步所有变更到服务器 `/home/zhangyh/knowledge-agent-dev`
- [ ] 更新 `runtime/app.env`（`AGENT_SYNTHESIS_PROVIDER=ollama`）
- [ ] 重启服务
- [ ] 运行 full30：
  ```bash
  python scripts/evaluate_canonical_retrieval.py \
    --cases runtime/task15/internal-research-overlap30-full-answer-cases.json \
    --parse-version-map runtime/task15/canonical-v6-tokenizer-rebuild-v4-staged.json \
    --mode both \
    --report runtime/task15/dev-new-agent-full30-v1.json
  ```
- [ ] 期望：RAG 召回 1.0，Agent 综合后 answer_passed >= 27/30

## Step 6：本地回归与记录

- [ ] 本地全量测试 ≥ 基准线
- [ ] 更新 `docs/work.md`
- [ ] 如果 full30 通过 → 激活 staged 版本

## 停止条件

- LLM 生成量超过 500 次/小时（避免 Ollama 过载）
- Worker 崩溃 3 次以上未解决
- 任何 hard gate 低于阈值
