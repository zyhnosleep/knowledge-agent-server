# RAG Agent 下一阶段项目计划

日期：2026-06-28

## 1. 当前基线

本项目已经不是从零搭 RAG，而是进入“把 RAG Agent 做成可验证、可回放、可上线演进系统”的阶段。

当前已完成的关键能力：

- 本地 RAG 基座：MinerU 解析、SQLite 存储、Redis worker、本地检索与 query eval loop。
- 外部合成能力：服务器已接入 DeepSeek，Agent 回答可走 `external_api/deepseek-v4-pro`，本地 RAG 仍负责证据来源。
- Agent v3 主链路：`route -> retrieve -> rag.answer -> synthesize -> verify -> finalize`。
- Evidence Pack：新增 `rag.retrieve_evidence`，可以只取证据、不生成答案、不写 QA 记录。
- SSE streaming：`POST /api/agent/query/stream` 已支持 `start/step/warning/final/done/error` 事件。
- Trace persistence：`AgentTraceRun` / `AgentTraceStep` 已能记录 Agent 执行轨迹，并可查询。
- Conversation TTL：`ConversationSession.expires_at` 和清理逻辑已存在，默认 `AGENT_CONVERSATION_TTL_DAYS=30`。
- Quality Dashboard：已能只读扫描 loop artifacts，展示 query summary、gate、失败 case 和归因。
- 远端 full30 gate：最新 evidence pack 远端验收为 `30 selected / 25 passed / 5 failed`，gate 通过，失败主要集中在 answer stage。

GPT/deep research 报告的建议是：不要把所有请求都做成重型自治 Agent，而是把 Agent 作为“复杂查询的受控增强层”，重点做复杂度路由、证据打包、工具白名单、状态回放、评测闭环和安全边界。这个判断和我们当前项目非常契合。

## 2. 项目化总目标

下一阶段目标不是重写架构，而是在现有实现上形成企业级闭环：

1. 让 Agent 每一步都有可审计证据和指标。
2. 让复杂 Agent 能力按需触发，而不是所有问题默认进入重流程。
3. 让 streaming、trace、TTL、外部 API、质量仪表盘全部进入同一条验收链。
4. 让 benchmark 从“最终答案 pass/fail”升级到“检索、证据、合成、引用、工具、延迟、失败归因”分层评估。
5. 保持本地轻量路线：FastAPI + SQLite + Redis + MinerU + 本地/外部模型可切换；暂不做重型平台迁移。

## 3. 阶段计划

### Phase 1：Evidence-first Agent Hardening

状态：已完成本地与远端验收。

已完成内容：

- `EvidenceItem` / `EvidencePack` schema。
- `QueryService.retrieve_evidence()`。
- `RAGAdapter.retrieve_evidence()`。
- `rag.retrieve_evidence` 内置工具。
- Agent 正常流加入 retrieve step。
- trace step metadata 加入 `evidence_count`、`table_evidence_count`、`evidence_kinds`、`source_stages`、`support_hints`。
- 本地测试、远端测试、远端 API smoke、SSE smoke、full30 gate 均通过。

Phase 1 的剩余改进：

- 让 `answer.synthesize` 在执行器中真正消费 `EvidencePack`，而不是只在 trace 中可见。
- 对空证据、弱证据、表格证据不足给出更清晰的用户可见说明。

### Phase 2：Agent/Evidence Evaluation Metrics

优先级：最高，下一步最值得做。

目标：把 full30、query attribution、Agent trace 里的信息汇总进质量仪表盘，让我们能看见 Agent 的失败到底发生在哪一层。

建议实现：

- 扩展 `QualityReportsService`，在现有 `query_eval.json` / `query_attribution.json` 旁路读取基础上，增加 `agent_metrics` 汇总。
- 指标至少包括：
  - `query_total`、`query_passed`、`query_failed`。
  - `failed_likely_stage_counts`，例如当前远端结果是 `{"answer": 5}`。
  - `failure_reason_counts`。
  - `retrieval_coverage`：有 evidence 的 case 数、无 evidence 的 case 数。
  - `table_evidence_cases`：表格证据相关 case 的命中情况。
  - `agent_tool_counts`：`rag.retrieve_evidence`、`rag.answer`、`answer.synthesize`、`answer.verify` 调用分布。
  - `agent_provider_counts`：`local` / `external_api` / 具体模型分布。
- 扩展 `src/app/schemas/quality.py`，给 dashboard response 增加 typed schema，而不是返回散乱 dict。
- 增加测试覆盖：
  - attribution 文件缺失时不失败。
  - malformed json 不影响 dashboard。
  - failed stage counts 正确。
  - embedded path 不被信任，继续只读同目录 artifacts。

验收标准：

- `tests/test_quality_reports.py` 通过。
- `tests/test_quality_routes.py` 通过。
- Dashboard API 返回 `agent_metrics`。
- 远端 full30 artifact 能被 dashboard 读出 `answer stage = 5` 的失败归因。

### Phase 3：Evidence-aware Synthesis

目标：让外部 API 和本地 fallback 都围绕 evidence pack 生成答案，降低“证据里有但答案没写出来”的失败。

背景：当前 full30 失败的 5 个 case 都归因到 answer stage，不是 citation/source 阶段。这说明下一步收益最高的是合成层补强。

建议实现：

- `AgentExecutor` 调用 `answer.synthesize` 时传入 evidence pack。
- `AgentSynthesizer` prompt 明确区分：
  - 用户问题。
  - RAG answer draft。
  - Evidence pack。
  - Citation constraints。
  - Missing/weak evidence policy。
- 增加“关键术语覆盖检查”：
  - 如果 evidence 中出现 expected terms，但最终答案遗漏，则触发一次 bounded retry。
  - retry 最多一次，避免无限循环。
- 对表格证据做更强压缩：
  - 保留表头、模型名、指标名、数值、source stage。
  - 避免只把 “Table 5” 这种 mention 当作真实表格证据。

验收标准：

- Agent focused tests 通过。
- 远端 smoke 中 `rag.retrieve_evidence` 仍然在 trace 里可见。
- full30 gate 不低于当前 `25/30 passed`。
- 重点观察 5 个 answer-stage 失败 case 是否减少：
  - `charmm36m_overview`
  - `ff99sb_ildn_mechanism`
  - `opls4_overview`
  - `opls4_mechanism`
  - `opls5_mechanism`

### Phase 4：Controlled Complex Agent

目标：补充复杂 Agent 能力，但只对复杂问题触发。

不建议马上整体迁移 LangGraph。当前项目已有自研 `AgentExecutor`、trace persistence、SSE 和 TTL，直接换框架风险大。更合适的路线是先把复杂能力抽成可替换接口，后续如果需要，再把执行器适配到 LangGraph。

建议能力：

- 复杂度路由：
  - simple：单步 RAG。
  - evidence_question：retrieve + answer。
  - complex_multi_hop：plan + multi-retrieve + synthesize + verify。
  - needs_clarification：不检索，先问澄清。
- 受控 planning：
  - planner 只产出 JSON plan。
  - executor 只执行白名单工具。
  - verifier 判断 plan 是否完成。
  - 最多 N 步，默认 4 到 6 步。
- 工具权限：
  - 默认只读工具。
  - 禁止 shell、写库、任意 SQL、任意网络请求进入模型可调工具。
  - 高风险工具未来再加审批模式。
- 安全策略：
  - 检索内容永远是 data，不是 instruction。
  - evidence 中的 prompt injection 文本不得改变系统策略。

验收标准：

- 普通问题不触发复杂 plan。
- 复杂问题 trace 中能看到 plan/retrieve/synthesize/verify。
- 每个工具调用有 schema 校验、超时、错误返回。
- 超步数时 graceful finalize，而不是死循环。

### Phase 5：Streaming / Trace / TTL Enterprise Hardening

目标：把已经存在的企业级能力做扎实。

SSE streaming：

- 保持当前 step-level streaming。
- 增加 heartbeats，避免长查询时前端误判断连。
- 统一 error event shape。
- 在 final event 中稳定返回 `trace_id`、`provider`、`model`、`tool_names`、`step_summary`。

Trace persistence：

- 增加 trace 查询列表接口，支持按 session、project、status、provider 过滤。
- 对 metadata 做脱敏，禁止写入 API key、原始 provider response、环境变量。
- trace 保存 evidence 摘要，不保存过长全文。

Conversation TTL：

- 保持默认 30 天。
- 启动时或定时清理 expired sessions。
- 增加测试确认 expired session 的 turns 一并删除。

验收标准：

- `tests/test_agent_streaming.py` 通过。
- `tests/test_agent_trace_store.py` 通过。
- `tests/test_conversation_memory.py` 通过。
- 前端流式问答能显示 step progress 和 trace id。

### Phase 6：Private Benchmark & Release Gate

目标：把 benchmark 标准固定成发布门禁。

当前标准定义位置：

- benchmark case 定义：`benchmarks/query/internal_research_v1.json`。
- query eval 执行：`scripts/query_eval.py`。
- loop/gate 参数：`scripts/run_mineru_rag_loop.py`。
- 当前 gate 关键参数：
  - `--min-query-passed`
  - `--max-query-failed`
  - `--require-failure-attribution`
- 质量报告读取：`src/app/services/quality_reports.py`。

建议固定三层 gate：

- PR/local focused gate：
  - Agent、RAG adapter、tool registry、quality reports 相关测试必须通过。
- Remote smoke gate：
  - `/api/health` 正常。
  - `/api/agent/query` 正常。
  - `/api/agent/query/stream` 返回 `text/event-stream`。
  - trace detail 可查。
- Remote full30 gate：
  - selected = 30。
  - completed = 30。
  - passed >= 25，短期最低不低于现状。
  - failed <= 5，若短期迭代允许，可保持 `max_query_failed=6` 作为过渡线。
  - `require_failure_attribution=true`。
  - unattributed failures 必须为 0。

下一轮优化目标：

- 从 `25/30` 提升到 `27/30`。
- answer-stage 失败从 `5` 降到 `3` 或更少。
- 不牺牲 source/citation gate。

## 4. codex-with-cc 执行流程

后续每个阶段都按这个 loop 推进：

1. Codex 主线程读取代码、制定任务边界和验收标准。
2. 写入 codex-with-cc task file。
3. Claude Code implementer 执行实现。
4. Claude Code spec reviewer 检查是否符合任务边界。
5. Claude Code quality reviewer 检查代码质量、测试、回归风险。
6. Claude Code final verifier 只做验收，不再改代码。
7. Codex 主线程评判报告是否合格。
8. Codex 主线程本地跑测试。
9. Codex 主线程同步服务器。
10. 服务器重启 API/worker。
11. 跑远端 smoke。
12. 跑远端 full30 gate。
13. 如果 gate 不通过，回到第 1 步，只针对失败归因收窄任务。

停止条件：

- 任务文件的 acceptance criteria 全部满足。
- worker reports 均为 DONE。
- 本地测试通过。
- 远端 smoke 通过。
- full30 gate 达到当前阶段阈值。
- docs/worklog 更新完成。

## 5. 不做什么

下一阶段暂不做这些事：

- 不整体迁移到 RAGFlow、Onyx、Dify、LlamaIndex 等重平台。
- 不立刻把 AgentExecutor 全量替换成 LangGraph。
- 不让模型直接执行 shell、自由 SQL、写数据库、任意网络请求。
- 不为了 benchmark 硬编码答案。
- 不把 DeepSeek 当成检索来源；DeepSeek 只负责基于本地 RAG evidence 做合成。
- 不把所有问题都走复杂多代理流程。

## 6. 推荐立即执行的下一步

最值得立刻推进的是 Phase 2：Agent/Evidence Evaluation Metrics。

原因：

- Phase 1 evidence pack 已经跑通。
- 远端 full30 仍有 5 个失败，而且都集中在 answer stage。
- 如果 dashboard 不能把这些失败分层展示，后面复杂 Agent 会越来越难调。
- Phase 2 的改动范围小，主要集中在 quality reports/schema/tests，不会扰动 RAG 主链路。
- 做完后，我们可以明确知道 Phase 3 evidence-aware synthesis 是否真的提升了质量。

建议本轮任务名称：

`rag-agent-eval-dashboard-20260628`

最小交付：

- `agent_metrics` schema。
- quality report parser。
- dashboard route 透出。
- tests。
- docs/worklog。
- 本地 + 远端 focused tests。
- 远端 dashboard 读取 full30 artifacts 的 smoke。
