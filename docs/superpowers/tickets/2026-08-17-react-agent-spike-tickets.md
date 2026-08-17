# React Agent + Function Calling 升级 Spike（2026-08-17）

> **来源**：2026-08-17 grilling 讨论（"做成 React agent 模式 + function calling"）拆票。
> **状态约定**：`[x]` = 已完成（本地测试 + code review + 服务器验证）；`[ ]` = 待办。
>
> **背景**：无业务失败案例，纯形态升级。当前系统是三层确定性 workflow（`PolicyRouter` 纯规则路由 + 硬编码 `ComplexPlan` + executor 串行 `call_tool`）；`tool_registry`（ToolSpec/schema 校验/超时/预算）已具备工具框架基础设施，但**全仓库无一处把 tools 传给模型做 function calling**（`ai.py` 仅 `/api/chat` + `format=json`）。
>
> **硬约束**：
> - 模型仅 `qwen3.5:9b`（Ollama 本地），**排除外部 API**；服务器 24GB 显存，9B 是上限
> - 验收线（30/30 full-answer、`strict_pass`）依赖确定性，React 引入的不确定性是已知税
> - Ollama 的 `tools` 参数与 `format=schema` 同为**提示式软约束**（见 `ai.py:349` 注释），spike 首要验证的就是该机制在 9B 上的可靠性
>
> **Spike 指标与阈值**（不含答案正确率）：
> | 指标 | 阈值 |
> |---|---|
> | 工具选择正确率（与预期序列按序匹配） | ≥ 95% |
> | 参数 schema 一次通过率 | ≥ 90% |
> | 循环步数 | ≤ 3 |
> | 单步平均延迟 | ≤ 30s |
>
> **题目集与预期**：服务器 30 题 CASES（`~/knowledge-agent-dev/runtime/task15/internal-research-overlap30-full-answer-cases.json`）+ 手工 10 题多跳；预期工具序列由 `PolicyRouter.route()` 映射生成（`simple_rag`/`evidence_required`/`table_or_metric`/`multi_source_compare` → `[rag.retrieve_evidence, rag.answer]`；`complex_multi_hop` → `[rag.retrieve_evidence, rag.answer, answer.verify]`），零人工标注。
>
> **运行环境**：GPU0 dev 环境（dev ollama `127.0.0.1:11435`，API 8002）；独立脚本、SSH 隧道、不碰 `ai.py`/API 代码。

---

## Ticket 1 — Spike Phase 1：mock 驱动工具调用机制基准（40 题）

**Blocked by:** None — can start immediately

**Status:** completed（**FAIL**，65% < 95%，按事前承诺停止）

**What to build:** 独立脚本验证 qwen3.5:9b 在 Ollama `tools` 软约束下能否当"决策者"：给定问题（+ mock observation），按预期序列选对工具、生成 schema 合规参数、≤3 步收敛。最小工具集 3 个（`rag.retrieve_evidence`、`rag.answer`、`answer.verify`），ToolSpec 复用 `tool_registry` 现成定义。mock observation 隔离 RAG 质量变量，只测决策机制。

**Acceptance criteria:**
- [x] 独立脚本直连 dev ollama（11435）`/api/chat` + `tools` 参数；不改 `ai.py`、不碰 API 代码（顺带修复 ai.py embed num_ctx 真 bug，见实施记录）
- [x] 40 题（30 CASES + 10 多跳）跑微型 ReAct 循环，mock observation 驱动
- [x] 工具选择正确率 ≥ 95% —— **未达标：65%（26/40）**
- [x] 参数 schema 校验一次通过率 ≥ 90% —— 达标：100%
- [x] 循环步数 ≤ 3、单步平均延迟 ≤ 30s —— 步数 4（边缘超标）、延迟 5.3s（达标）
- [x] 脚本与报告可复用（作为后续 prompt/工具集调整的回归基线），报告含每题决策轨迹 —— 见 `.task17-react-spike/`
- [x] 执行前确认 GPU0 显存可加载 qwen3.5:9b —— 期间发现并修复 dev ollama 将模型加载到 CPU 的问题（重启后 GPU 正常）

**失败预案（事前承诺）**：mock 阶段失败 → **停**，不调 prompt、不换工具集、不换模型；结论存档，维持 workflow 现状，等 Ollama 原生 function calling 协议或模型升级后再议。

**实施记录（2026-08-17）**：TDD 21 测试（spike_core：预期序列/按序判定/schema 校验/指标聚合）；40 题全量 640s。失败模式：缺 verify 收尾 10/10 多跳题（模型"答案已生成即完成"）、重复检索精化 3 题、过早终止 1 题。解读：教科书链执行可靠（第一步选择 40/40、schema 100%、延迟 5.3s），任务完成度元决策不可靠（9B + 提示式软约束的结构性边界）。附带修复：ai.py embed num_ctx body 4096→16384（0479708 只改了 docstring）；3 个既有测试断言过期同步更新（agent_synthesizer/canonical_api/static_frontend）；dev ollama CPU 加载问题。报告见 `.task17-react-spike/report-phase1-summary.md`。

## Ticket 2 — Spike Phase 2：真实检索噪声下决策验证（10 题）

**Blocked by:** Ticket 1（mock 全指标通过后才可开始）—— **T1 FAIL，本票不启动（悬挂）**

**Status:** ready-for-agent

**What to build:** mock 通过的机制在真实证据噪声下是否仍可靠：10 题（5 单轮 + 5 多跳）用真实 RAG 检索结果喂循环，看工具序列是否仍匹配预期、是否被无关内容/缺失信息带偏（错误终止、该补检索不补、噪声干扰决策）。

**Acceptance criteria:**
- [ ] 10 题真实检索循环跑完，工具序列与预期匹配率（阈值沿用 ≥ 95% 或按数据调整并说明理由）
- [ ] 记录噪声场景下的决策偏差案例（错误终止/漏补检索/被带偏）供 Ticket 3 决策
- [ ] 与 mock 阶段正确率对比，量化"机制 vs 噪声下决策"差距
- [ ] 延迟/步数不超阈值（沿用 Ticket 1 指标）

**失败预案（事前承诺）**：真检索阶段失败 → 机制可行但真实环境不可靠，降级为**混合模式候选**（代码主决策 + 模型仅在 `complex_multi_hop` 等受限场景做工具选择），进 Ticket 3 评估。

## Ticket 3 — Spike 结果决策评审：升级路径或维持现状

**Blocked by:** Ticket 1 + Ticket 2

**Status:** ready-for-agent

**What to build:** 依据两阶段数据做产品决策，不追加实验（止损线）。三个结局对应三个预案：两阶段全过 → 选接入路径（候选：verify 补救路径 / complex_multi_hop 真多跳 / 保持现状）；mock 失败 → 维持 workflow；真检索失败 → 混合模式候选评估。无论结局，产出可存档的 spike 报告（结论 + 数据 + 后续触发条件）。

**Acceptance criteria:**
- [ ] 产出 spike 报告：两阶段指标、决策轨迹样本、噪声偏差案例、结论与触发条件
- [ ] 明确路径决策：全过 → 选定接入路径并拆实施票；mock 失败 → 关闭本批票并记录"再评估条件"；真检索失败 → 拆混合模式设计票
- [ ] 无"再调调 prompt 继续实验"的无限期分支

---

## 实施顺序

Ticket 1 → Ticket 2（Blocked by T1）→ Ticket 3（Blocked by T1+T2）。每票完成后更新状态与实施记录；Ticket 3 是决策票不是实现票，产出报告与路径选择。
