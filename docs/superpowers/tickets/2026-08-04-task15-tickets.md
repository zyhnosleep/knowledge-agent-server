# Task 15 执行清单（更新至 2026-08-05）

> 基于 spec：`docs/superpowers/specs/2026-08-04-task15-rag-regression-fix.md`
> Answer 链路深化：`docs/superpowers/specs/2026-08-05-answer-chain-facts-first-gate.md`
> 状态约定：`[x]` = 已完成（代码核验确认）；`[ ]` = 待办。

---

## 当前基线（2026-08-05）

- **RAG 层（candidate full30）**：candidate 为 GPU0 / API `8002`，检索 Recall@5/10 **1.0**，citation/source **1.0**，**30 题答案通过 30/30，strict_pass=true**。根因（`_table_citation_indexes` 全局 top-24 截断 Table 7）已修复：按 table_id 分组保底 + 每表保底最高分 chunk（携带完整 table_facts）。
- **候选/正式边界**：本轮只验证 candidate；active 是 GPU1 / API `8001` 的旧版本。未切换 active pointer，未修改 active 数据库、索引或 GPU1 测试环境。
- **Agent 层**：**启动**（RAG 已达标）。上次评测 5/30，根因是 synthesize 无差别重写 RAG 草稿。

> **事实-first 口径**：facts-first 不等于移除 RAG 中的 LLM。canonical facts 是事实来源，LLM 仍可负责自然语言组织与格式化；确定性 checklist、unsupported-number 校验和 repair 负责防止漏事实、改数字或编造数字。LLM 不能替代 facts，也不能因为 contexts 被截断而反推需求清单。

---

## 已完成（代码核验确认）

### ✅ Ticket 2：typed inventory 与激活门禁

- [x] 2.1 manifest 写入 typed_inventory（`canonical_artifacts.py` 已实现，含 table_ids/child_count/source_block_ids/row_count）
- [x] 2.2 semantic_split/index 输出同构 Child inventory
- [x] 2.3 `_run_activation_gate_stage()` 逐表校验（`derive_child_inventory_from_payload` / `derive_db_typed_inventory` 已实现）
- [x] 2.5 重建脚本输出 active/candidate version + activation_block_reason

### ✅ Ticket 5：table header/row classifier 重写

- [x] 5.1 统一 classifier（`_classify_semantic_header_row` 已实现于 `table_extraction.py`）
- [x] 5.2 列级向前填充合并表头（`_compose_header_lines` / `_classify_semantic_header_row` 已实现）
- [x] 5.3 多级行标签合并

### ✅ Ticket 3 部分：LaTeX 锚点归一化（残留 1）

- [x] 3.4 共享 LaTeX 归一化（`\mathbb{\chi}_1` 变体已处理，search.py:1760）
- [x] 3.5 检索侧与验收侧共用

### ✅ Ticket 6 部分：repair 校验链强化

- [x] 6.2 LLM repair 注入 canonical facts 清单
- [x] 6.3 repair 后校验链（`_answer_lacks_requested_metrics` + `_unsupported_answer_numbers` 多个调用点已强化）

---

## 待办

### ⏸ Ticket 3 收尾：科学 evidence 窗口统一（已知风险，暂缓）

**状态**：30/30 已达标，固定窗口未造成当前失败（FRET/GAlib/fast-folding 均通过）。保留为已知风险，Agent（Ticket 9）优先；遇到"术语被窗口切掉"的回归再修。

- [ ] 3.1 移除 `_deterministic_scientific_evidence_answer_if_supported()` 及专用检索路径中残留的固定窗口（当前仍有 900/1000/1400，search.py:1221/1605/2090 等），改为 required-anchor + token budget。
- [ ] 3.3 `_window_text()` 统一 token budget；anchor 不足时返回 evidence-insufficient 而非静默丢弃。

### ⏸ Ticket 4 收尾：表格 evidence 路径（已知风险，暂缓）

**状态**：30/30 已达标，excerpt 截断未造成当前失败。保留为已知风险，Agent（Ticket 9）优先；遇到"表格 excerpt 半行截断导致引用/答案缺失"的回归再修。

- [ ] 4.2 `_table_citation_excerpt()` / `_table_block_excerpt()` 改为完整行边界（当前 `_table_citation_excerpt` 仍有 max_chars=2400，search.py:2866；`_table_block_excerpt` 默认 1200，search.py:6777）——长度上限只用于展示，不在半行结束。
- [ ] 4.4 覆盖 4000/2400 边界：目标行在 2400 字符后不被半行截断。

### 🎯 Ticket 6 深化：answer 链路 facts-first 最终门禁（针对当前 answer-path 缺口）

**来源**：`docs/superpowers/specs/2026-08-05-answer-chain-facts-first-gate.md`

**核心**：期望清单需求侧化 + 按 `table_id` 保证覆盖 + facts-first 事实载荷 + 受约束的 LLM 语言组织 + 最终校验/repair。

- [ ] 6E0 **责任点定位（先复现，再修）**：用 debug 脚本跑 `opls5_table_metrics` 单题，逐段打印调用链，确认 Table 7 在哪一环丢失：
  - `_build_rag_contexts` 返回的 contexts 中，Table 7 的 citation 是否在（retrieval 应已在）；
  - `_finalize_contexts` 之后 Table 7 是否仍在；
  - `_fit_contexts_to_token_budget` 裁剪前后，Table 7 的行数变化（是否被截断）；
  - `_table_citation_indexes` 返回了哪些表（是否含 7）；
  - repair 时 facts inventory 上限、Table 7 的 1.18/1.12 是否在 inventory 内。
  **目的**：锁定真正的责任点（可能是 `_fit_contexts_to_token_budget`、`_finalize_contexts`、`_table_citation_indexes` 或索引上限中的一或多个），据此决定 6A-6D 的精确改动面，避免按假设修错路径。
- [ ] 6A **期望清单需求侧化**：新增 `_expected_table_facts_from_question(question)` —— 从 question 解析请求的 `table_id + row + value` 集合（显式 "Table N" + 指标映射），独立于 contexts。表格映射应优先使用 canonical manifest 的 typed inventory，而不是只从已返回的 Child 猜测。**当前 `_extract_requested_metric_values` 从 contexts 反推，供给被截断时校验会失真。**
- [ ] 6B **多表 contexts 分组保底**：`_fit_contexts_to_token_budget` 对表格 context 按 `table_id` 分组保留，每个请求表至少一组完整行，且优先使用 typed inventory 枚举目标表；token budget 应裁剪非关键内容，而不能静默截断目标表尾。若预算确实无法覆盖，应显式返回 evidence-insufficient 并触发缺失门禁。
- [ ] 6C **facts-first 组装优先**：期望清单中每个 `table_id + row + value` 在 canonical facts 可找到时，必须先构成完整的 facts-first 事实载荷/答案骨架。LLM 仍可负责自然语言组织和格式化，但不能决定事实、漏掉已覆盖事实或补写新数字；只有清单事实确实缺失才允许受约束地降级 LLM，且 LLM 须声明缺失。
- [ ] 6D **LLM 角色限定**：`_answer_lacks_requested_metrics` 改为对比"期望清单 vs 答案"（不再从被裁剪的 contexts 反推 metrics）；`_unsupported_answer_numbers` 对照 canonical facts 捕获 LLM 新编数字 → 打回 → 以完整 facts-first 载荷重组装或 repair。
- [ ] 6E **集成测试覆盖真实路径**：现有单测只模拟全量 contexts，未覆盖 `_build_rag_contexts → _finalize_contexts → _fit_contexts_to_token_budget → answer` 截断路径。新增：
  - 多表 + token budget 截断表尾 → 答案仍含 Table 7 期望 value
  - question 请求 Table 7 但 contexts 无 → 门禁捕获缺失（而非内部通过）
  - fake LLM 只返回说明文字缺 1.18 → 确定性重组装；fake 编 999.0 → 打回
- [ ] 6F `failed_gates` 命名修正：区分 `retrieval_gate` 与 `answer_gate`，避免 retrieval 通过但 full-answer 失败时命名误导。

### 验收（Ticket 7 收尾）

- [ ] 7.2 30 题 full-answer 通过 **30/30**（当前 29/30，唯一 `opls5_table_metrics`）
- [ ] 7.3 facts 覆盖时，完整 facts-first 载荷必须在 LLM 组织答案前确定；LLM（如仍被调用）只负责语言组织，不负责事实补全，且不得因 contexts 截断触发无约束降级或重复 repair。记录 `opls5` latency 作为性能指标；验收条件不是简单地禁止所有 LLM 调用。
- [ ] 新增集成测试全绿；既有回归不退化

---

## Ticket 9：Agent 收窄为"会话网关 + 受约束编排"（实施级）

**目标**：消灭 Agent 层的答案破坏（上次评测 5/30 → ≥28/30）。纯 RAG 30/30 证明答案架构是对的，破坏来自 [agent_synthesizer.py](src/app/services/agent_synthesizer.py) 无差别重写 RAG 草稿。

**核心原则**：synthesize 是"受约束组织器"不是"生成器"。单轮/对比问题直接透传 rag.answer；只有跨轮引用场景才调用 synthesize，且必须注入 facts 清单 + 输出校验。

**状态（2026-08-06 核实）**：9.1-9.5 主体已在工作树实现（未提交）：直通门禁 + `direct_block_reason`（agent_executor.py:437-550）、`_is_cross_turn_query`（:1405-1447）、synthesize 四守卫 + narrow_context + anchors retry。全量实测 1982 passed / 1 failed（唯一失败是 9.1.5 的退化 stub，非实现 bug）。**真正待实现**：9.7 coverage 通道、9.8 需求侧期望、9.1.5 stub 修复、9.6 验收。实施计划见 `docs/superpowers/plans/2026-08-06-task9-agent-direct-routing-follow-up.md`（已按现状修正）。

### 9.1 路由决策矩阵（落点：agent_executor.py）

**位置**：[agent_executor.py:421-426](src/app/services/agent_executor.py#L421) 的 `# ---- answer.synthesize ----` 块。

- [ ] 9.1.1 在 `_run_synthesize` 调用前加路由判断：`route.route in {"simple_rag", "evidence_required", "table_or_metric", "multi_source_compare"}` 且**非跨轮引用** → 跳过 synthesize，直接透传 `answer_text`（rag.answer 结果），`synth_provider="local"` / `synth_model="rag-direct"`。
- [ ] 9.1.2 跳过后仍保留：citations 不过滤（不调 `cited_indexes` 过滤）、verify 照常跑、trace 记录 `synthesis_skipped=True`。
- [ ] 9.1.3 `complex_multi_hop` 与跨轮引用场景保留 synthesize 调用。
- [ ] 9.1.4 增加直通前置条件：只有当前 RAG 结果已经通过答案/引用完整性校验、不是 `evidence-insufficient` 或 `needs_clarification`，且存在可用 citations 时，才允许标记为 `rag-direct`。直通路径不得绕过既有 verify；校验失败时保留现有降级路径，不把未经验证的草稿直接返回。

- [ ] 9.1.5 **修复退化 stub 测试**：`test_route_matrix_skips_synthesis_for_single_turn_queries`（tests/test_agent_executor.py:1775-1782）的 table_or_metric 用例用无数值 stub 答案触发 verifier retry → 直通被正确阻止（符合 9.1.4）。修复：该用例 stub 答案带数值（如 `make_executor(db, rag_answer_text="test answer 1.18")`）或 citation 加 `page_kind="table"`。（✅ 已实现 2026-08-06：stub 改为 "test answer 1.18" + `_assert_direct_pass_through(rag_answer=...)`；测试转绿）
- [ ] 9.1.6 **coverage 门禁生效范围**（与 9.7 联动）：coverage 只作为 `table_or_metric` 且 RAG 已声明表格覆盖（table_facts 非空或 coverage_status != "unknown"）时的直通前置；其余路由 coverage 中立，unknown 不得阻塞直通（防 9.6.2 的直接路由退化）。（✅ 已实现：agent_executor 门禁 `coverage_partial` + `direct_block_reason="coverage_partial"` + finalize trace；新增 2 个门禁测试）

**TDD 测试**（`tests/test_agent_executor.py`，已实现，验证即可）：
- `simple_rag` 路由 + fake synthesizer 断言**不被调用**、答案 == rag.answer 结果。
- `complex_multi_hop` 路由 + fake synthesizer 断言**被调用**。
- 直通断言 helper：`_assert_direct_pass_through`（tests/test_agent_executor.py:1741-1748，`rag-direct` + `synthesis_skipped=True` + 无 synthesis step）。

### 9.2 跨轮引用识别（新增 `_is_cross_turn_query`）

**位置**：agent_executor.py 新增方法，放在 `_contextualize_retrieval_query` 附近。

- [ ] 9.2.1 实现 `_is_cross_turn_query(query: str, session_id: str | None) -> bool`：关键词和长度只能作为候选信号；最终判定必须同时满足存在可用历史轮次、当前 query 含指代/省略且上一轮上下文能够解析该指代。没有历史上下文时一律返回 `False`；单轮问题中的“对比”不能单独触发跨轮路径。
- [ ] 9.2.2 与 `_contextualize_retrieval_query` 联动：跨轮 query 已含扩展上下文时，synthesize 的 `conversation_summary` 才有意义；若指代解析失败，保留 `needs_clarification`，不得让 synthesize 猜测缺失实体。

**TDD 测试**：红——"那 C36 呢" → True；"charmm36m 的 accuracy 是多少" → False；"对比之前那篇" → True。

### 9.3 synthesize 改造三要素（落点：agent_synthesizer.py）

**位置**：
- 输入收窄：[agent_synthesizer.py:464](src/app/services/agent_synthesizer.py#L464) `_build_local_messages`（本地）+ [1056](src/app/services/agent_synthesizer.py#L1056) `_external_synthesize`（外部）
- 输出校验：synthesize 返回后（[526](src/app/services/agent_synthesizer.py#L526) `_extract_cited_indexes` / [536](src/app/services/agent_synthesizer.py#L536) `_sanitize_inline_citations` 附近）

- [ ] 9.3.1 **输入收窄**：跨轮场景的 prompt 只含「conversation_summary 中的事实 + 当前 rag_answer + citations」，不含证据全集。
- [ ] 9.3.2 **结构化 facts 清单注入**：facts 清单优先来自 RAG 已校验的结构化事实，而不是从可能被截断的 citation 文本重新猜测。每条 fact 至少保留 `fact_id`、`document_id`、`parse_version`、`table_id`、`row_index/row_label`、`column`、`value`、`unit`、`term`；citations 只负责事实的证据绑定。若 RAG 没有结构化 facts，Agent 不得从不完整 citation 反推完整事实清单，应回退 `rag.answer` 或声明 evidence-insufficient。
- [ ] 9.3.3 **输出校验**：synthesize 返回后同时校验期望 facts 是否遗漏、表号/术语/单位是否保留，以及 `_unsupported_answer_numbers` 式的新增数字。数值比较必须先做统一规范化，并排除 citation 编号、页码、年份等非答案事实数字；证据没有的数字、清单中已有但答案遗漏的事实、错误表号或单位一律判失败，失败回退到 rag.answer 原答案。

**TDD 测试**（`tests/test_agent_synthesizer.py`）：
- 红：fake Ollama 把 `21.0` 重写成 `999.0` → 最终答案不含 `999.0` 且含 `21.0`。
- 红：fake Ollama 漏掉 `1.18` → 校验失败 → 回退 rag.answer 原答案。
- 红：fake Ollama 删除 `Table 7`、`1.18` 的单位或关键术语 → 校验失败 → 回退 rag.answer 原答案。

### 9.4 conversation_summary 通道

**位置**：[agent_executor.py:1076](src/app/services/agent_executor.py#L1076) `_run_synthesize` 的 `conv_summary = self._build_conversation_summary(session_id)`。

- [ ] 9.4.1 单轮/对比路由跳过后，`conv_summary` 不传给 synthesize（无 synthesize 调用）。
- [ ] 9.4.2 跨轮引用场景保留传参（现状即可）。

### 9.5 记忆

- [ ] 9.5.1 追问指代：`_contextualize_retrieval_query` 已处理（拼上一轮 query 进当前 query），保持现状。
- [ ] 9.5.2 跨轮事实引用：走 9.4 的 conversation_summary + 9.3.2 的 facts 清单注入，不新增组件。

### 9.6 回归与验收

- [x] 9.6.1 本地测试：`tests/test_agent_executor.py tests/test_agent_synthesizer.py` 全绿；既有回归不退化（当前收集 1983 项，含 9.1.5 stub 修复后应全绿；统计以新鲜输出为准，不沿用旧 handoff 数字）。（✅ 已验收 2026-08-07：test_query_service 266 + agent_executor/streaming 82 + agent_synthesizer 83 全绿）
- [x] 9.6.2 服务器评测分层验收：`scripts/evaluate_agent_full.py`（`call_agent_api(api_url, case)` 调 `/api/agent/query`）跑完整 30 题；单轮/直接 RAG 路由必须相对已通过的 RAG 30/30 **不退化**，不得因 Agent 处理丢失事实或引用；跨轮/复杂题单独统计。整体 **≥28/30** 只能作为辅助目标，不能覆盖直接路由的回归。（✅ 已验收 2026-08-07：full30 **30/30**，答案正确率 100%、引用 100%、每题 ≤120s（max 61.7s）；报告 `runtime/task15/task16-agent-candidate-full30-20260807-optiona.json`）
- [x] 9.6.3 检查 trace/安全边界不破坏（`AgentTraceStore` 仍记录 synthesize 跳过标记、调用原因、校验结果和回退原因），并记录 synthesis 调用率、跳过率及 P50/P95 latency。（✅ 已验收 2026-08-07：synthesize 绕行率 0.0%（上轮 3.3%）、p50 23.2s / p95 57.5s；runner 报告含 `synthesis_applied_rate` + 每题 answer_provider；折中语义新增 `latency_exceeded_120s` / `timeout_with_complete_answer` harness 警告）
- [x] 9.6.4 验收必须同时确认：直通路径未调用 synthesizer、跨轮路径只接收结构化 facts 与必要上下文、非法数字/遗漏 facts 会回退、citations 未被无故过滤；不能只用最终总分判定 Agent 修复成功。（✅ 已验收 2026-08-07：synthesis_applied_rate 0.0% 实证直通全量生效；9.8 三元状态 + 保真测试全绿；citation_pass_rate 1.0；agent_executor 门禁测试（coverage_partial 等）全绿）

### 9.7 轻量 typed inventory / coverage 通道（落点：schemas/agent.py + search.py + tool_registry.py）

**目标**：让"RAG 回答覆盖了哪些目标表"在 RAG 与 Agent 之间可审计传递，替代"从被截断 facts 反推期望"。

- [ ] 9.7.1 **EvidencePack 新增可选 inventory/coverage 字段**：每表保留 `document_id / parse_version / table_id / row_count / source_block_ids / child_ids / child_count / parent_ids / row_indices`（来自 canonical manifest，不注入完整 manifest 或 child 文本）；另加 `coverage_status`（complete / partial / unknown）与 `coverage_missing_tables`。旧调用方不提供时按 unknown 兼容，禁止把 active 与 candidate 的 inventory 混用。（✅ 已实现：`TableCoverage` schema + `_build_table_coverage` 按 (document_id, parse_version, table_id) 分组对照 `load_typed_inventory`，version 来自 facts 携带的 parse_version，防 active/candidate 混用）
- [ ] 9.7.2 **per-fact schema 扩展（9.3.2 补齐）**：`TableFactEvidence` 增加 `document_id / parse_version / table_id / row_index`（或 row_label）/ `column`（现有 fact_id/unit/term 保留）；search.py 生成 facts 时补齐（生成点已可得这些信息）。（✅ 已存在：search.py:446-468 生成时已带全部字段）
- [ ] 9.7.3 **透传不做门禁**：`rag_adapter.py` / `tool_registry.py` 只透传 inventory/coverage 元数据；Agent 侧仅按 9.1.6 生效范围做直通门禁，不让工具层从 excerpt 重新猜 facts。（✅ 已实现：tool_registry handler 原样透传；rag_adapter 薄封装无需改动；executor `_merge_evidence_pack` 合并附件 facts 时降级 unknown 防伪 complete）
- [ ] 9.7.4 **测试**：candidate parse_version_map → inventory 与 facts 同版本；active 与 candidate 混合 → coverage 失败；目标表被预算裁剪 → partial；完整 → complete；旧 EvidencePack 无 inventory → unknown 兼容；unknown 不阻塞非表格路由直通。（✅ 已实现：test_query_service 5 个覆盖状态测试 + test_tool_registry 2 个透传测试 + test_agent_executor 2 个门禁测试）

### 9.8 需求侧 expected facts + unknown 语义（落点：agent_synthesizer.py）

**目标**：期望清单由"问题 + inventory"定义，不再由"已召回 facts"反推；无法唯一解析时显式 unknown，不静默回退全部。

- [ ] 9.8.1 **`_expected_facts_from_question` 需求侧化**：输入加 typed inventory，输出 `table_ref/table_id + row alias/row index + column/term alias + document/parse_version` 目标；显式 Table N、OPLS/CHARMM 别名优先用现有 QueryService 规则。（✅ 已实现：术语 term/column 匹配（含朴素单数化 values→value）+ 显式 Table N 按 table_id 数字段匹配，返回三元状态）
- [ ] 9.8.2 **移除无命中回退全部**：现有 `return matched or dict_facts`（agent_synthesizer.py:1196）改为三元状态（complete / missing / unknown）；无法唯一解析 → `expected_facts_status="unknown"`。（✅ 已实现：`return matched or dict_facts` 已移除；无术语且无表引用 → unknown，解析到但零命中 → missing）
- [ ] 9.8.3 **unknown 默认行为**：不触发 facts 遗漏回退（`_synthesis_missing_fact_fields` 短路），**保留** draft-fidelity / 表号 / unsupported-number 锚点守卫，finalize metadata 记录 `expected_facts_status` 供审计——缩小覆盖范围但不可静默禁用整条 fidelity 链。（✅ 已实现：guard_meta 透出 + synthesize 结果携带 + executor finalize trace 记录；锚点守卫不受影响）
- [ ] 9.8.4 **join 测试**：问题要求 Table 7 但 facts 被截断 → expected target 仍存在、coverage 非 complete、普通直通被阻止；现有 1.18 value / % unit / term / Table 7 / 999.0 测试保留。（✅ 已实现：4 个新三元状态测试 + 既有保真测试保留全绿；直通阻止由 9.1.6/9.7 门禁测试覆盖）

**验证命令**：
```powershell
PYTHONPATH=src pytest tests/test_agent_executor.py tests/test_agent_synthesizer.py tests/test_tool_registry.py tests/test_query_service.py -q
# 服务器：
python scripts/evaluate_agent_full.py --cases runtime/internal-research-overlap30-full-answer-cases.json --report runtime/task15/dev-new-agent-full30-v2.json --api-url http://127.0.0.1:8002
```

**方案依据**：主 spec（`docs/superpowers/specs/2026-08-04-task15-rag-regression-fix.md`）的 "Agent 部分（Ticket 9）" 小节；实施级计划 `docs/superpowers/plans/2026-08-06-task9-agent-direct-routing-follow-up.md`。**前置依赖**：RAG 30/30（已完成）。**已知风险（继承）**：contradicted 依赖 verifier 自由文本 verdict（无词表）；单位规则三处漂移（`search._FACT_UNIT_RE` vs answer_verifier vs anchors）；`synth_model="rag-direct"` 编码路由进模型名（可接受）。
---

## 硬性约束

- 零框架依赖（LangChain / LlamaIndex / LangGraph 全部不引入）
- 测试环境（GPU1）只读；不删旧数据；不单独激活单篇
- 开发链路：本地 → push GitHub → pull 服务器 dev → restart
