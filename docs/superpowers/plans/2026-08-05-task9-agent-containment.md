# Task 9 实施计划：Agent 收窄为"会话网关 + 受约束编排"

> 基于 tickets：`docs/superpowers/tickets/2026-08-04-task15-tickets.md` 的 Ticket 9（实施级）
> 前置依赖：RAG 30/30 已达标（`_table_citation_indexes` 分组保底修复）
> 日期：2026-08-05

---

## 1. 目标

消灭 Agent 层的答案破坏（上次评测 5/30 → ≥28/30，且**单轮路由不退化**）。核心：synthesize 从"无差别重写器"改为"受约束组织器"——单轮/对比问题直接透传 rag.answer，只有跨轮引用场景才调用 synthesize，且注入结构化 facts + 输出校验。

## 2. 前置核实（Step 0，先于一切实施）

以下依赖点决定 9.3.2 能否落地，必须先核实：

- [ ] **核实 A：结构化 facts 数据源**。`RAGAdapter.answer()` 返回的 `QueryResponse` 是否携带结构化 facts（table_facts / TableFact）？若无，Agent 要拿"RAG 已校验的结构化事实"需复用 `retrieve_evidence` 的 `EvidencePack.table_facts`，或让 answer 路径附加。**这是 9.3.2 的落地前提。**
- [ ] **核实 B：`_contextualize_retrieval_query` 的现有逻辑**。其指代扩展能否复用为 9.2.1"上一轮能解析该指代"的判定基础。
- [ ] **核实 C：`_unsupported_answer_numbers` 的数字判定规则**。RAG 层已有的"哪些数字算证据数字"逻辑可否复用为 9.3.3 的数值规范化基准。
- [ ] **核实 D：`verification_status` / evidence-insufficient 标记**。RAG 答案的 `QueryResponse.verification_status` 与 "## insufficient evidence" 前缀能否作为 9.1.4 直通前置校验的依据。

## 3. 实施步骤（TDD，按依赖排序）

### Step 1：9.2 跨轮引用识别（纯函数，依赖最少，先做）

**位置**：`agent_executor.py` 新增 `_is_cross_turn_query(query, session_id) -> bool`。

1. [ ] 1.1 红测试（`tests/test_agent_executor.py`）：
   - "那 C36 呢"（有历史轮次）→ True
   - "charmm36m 的 accuracy 是多少" → False
   - "对比之前那篇"（有历史）→ True
   - 单轮"对比 OPLS4 和 OPLS5"（无历史）→ False（防"对比"词误触发）
   - 无 session_id / 无历史轮次 → 一律 False
2. [ ] 1.2 实现：关键词/长度作为候选信号；最终判定 = 存在历史轮次 + 指代 + `_contextualize_retrieval_query` 能解析。
3. [ ] 1.3 绿。

### Step 2：9.1 路由决策矩阵（核心改动）

**位置**：`agent_executor.py:421-426` synthesize 调用块。

1. [ ] 2.1 红测试：
   - `simple_rag` 路由 + fake synthesizer → **不被调用**、答案 == rag.answer 结果
   - `complex_multi_hop` 路由 + fake synthesizer → **被调用**
   - 直通前置：evidence-insufficient 答案 → 不透传，保留降级路径
2. [ ] 2.2 实现：`route.route in {simple_rag, evidence_required, table_or_metric, multi_source_compare}` 且非跨轮引用 → 跳过 synthesize，透传 answer_text；`synth_provider="local"` / `synth_model="rag-direct"`；trace 记 `synthesis_skipped=True`。跳过后 citations 不过滤、verify 照常。
3. [ ] 2.3 绿。

### Step 3：9.3 synthesize 改造三要素

**位置**：`agent_synthesizer.py` `_build_local_messages`(:464) / `_external_synthesize`(:1056) / 输出校验(:526)。

1. [ ] 3.1 红测试（`tests/test_agent_synthesizer.py`）：
   - fake Ollama 把 `21.0` 重写成 `999.0` → 答案不含 `999.0` 且含 `21.0`
   - fake Ollama 漏掉 `1.18` → 校验失败 → 回退 rag.answer 原答案
   - fake Ollama 删 `Table 7` 或单位 → 校验失败 → 回退
2. [ ] 3.2 实现（按核实 A 的结果决定 facts 来源）：
   - 输入收窄：跨轮 prompt 只含「conversation_summary 事实 + 当前 rag_answer + citations」
   - 结构化 facts 清单注入：fact_id/document_id/parse_version/table_id/row_index/row_label/column/value/unit/term；citations 只做证据绑定
   - 输出校验：facts 遗漏 + 表号/术语/单位保留 + 新数字（复用核实 C 的规范化规则）
3. [ ] 3.3 绿。

### Step 4：9.4/9.5 记忆通道

1. [ ] 4.1 单轮跳过时 conv_summary 不传（现状：无 synthesize 调用自然不传）。
2. [ ] 4.2 跨轮保留传参（现状即可）。
3. [ ] 4.3 `_contextualize_retrieval_query` 保持现状（追问指代已处理）。

### Step 5：9.6 回归与验收

1. [ ] 5.1 本地：`tests/test_agent_executor.py tests/test_agent_synthesizer.py` 全绿；既有 1965 项回归不退化。
2. [ ] 5.2 服务器分层评测：`evaluate_agent_full.py` 跑 30 题；单轮/直接路由相对 RAG 30/30 **不退化**；跨轮单独统计；整体 ≥28/30 为辅助目标。
3. [ ] 5.3 trace 记录：synthesize 跳过/调用原因、校验结果、回退原因；统计 synthesis 调用率、跳过率、P50/P95 latency。
4. [ ] 5.4 行为级验收：直通未调 synthesizer、跨轮只收结构化 facts、非法数字回退、citations 未被无故过滤。

## 4. 验证命令

```powershell
PYTHONPATH=src pytest tests/test_agent_executor.py tests/test_agent_synthesizer.py -q
# 服务器：
python scripts/evaluate_agent_full.py --cases runtime/internal-research-overlap30-full-answer-cases.json --report runtime/task15/dev-new-agent-full30-v2.json --api-url http://127.0.0.1:8002
```

## 5. 验收标准

- 单轮/直接路由：相对 RAG 30/30 零退化
- 跨轮引用：synthesize 被调用且注入结构化 facts，输出通过校验
- 非法数字/遗漏 facts：一律回退 rag.answer 原答案
- trace 完整记录原因；既有回归不退化

## 6. 风险与约束

- **零框架依赖**（LangChain / LlamaIndex / LangGraph 不引入）
- 测试环境（GPU1）只读；active pointer 不切换
- 开发链路：本地 → push GitHub → pull 服务器 dev → restart（commit 由用户把控）
- 9.3.2 若核实 A 显示 QueryResponse 无结构化 facts：先扩展 RAG 返回，再实施 facts 注入（不跳过核实直接猜）
