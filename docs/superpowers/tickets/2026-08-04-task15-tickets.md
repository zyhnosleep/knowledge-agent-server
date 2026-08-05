# Task 15 执行清单（更新至 2026-08-05）

> 基于 spec：`docs/superpowers/specs/2026-08-04-task15-rag-regression-fix.md`
> Answer 链路深化：`docs/superpowers/specs/2026-08-05-answer-chain-facts-first-gate.md`
> 状态约定：`[x]` = 已完成（代码核验确认）；`[ ]` = 待办。

---

## 当前基线（2026-08-05）

- **RAG 层（candidate full30）**：candidate 为 GPU0 / API `8002`，检索 Recall@5/10 **1.0**，citation/source **1.0**，30 题答案通过 **29/30**。
- **唯一失败**：`opls5_table_metrics` —— 最终答案缺 Table 7、1.18、1.12；检索报告显示完整的 Table 7 facts 可以被检索到。
- **当前根因判断**：问题已经收敛到 answer-path 的 facts 完整性与最终答案校验缺口，而不是基础检索失败。当前重点嫌疑是 `_fit_contexts_to_token_budget` 的多表裁剪，但现有 full30 证据尚不能证明它是唯一责任点；还必须核对 `_finalize_contexts`、`_table_citation_indexes`、表格索引上限、repair facts inventory 上限等路径。
- **候选/正式边界**：本轮只验证 candidate；active 是 GPU1 / API `8001` 的旧版本。未切换 active pointer，未修改 active 数据库、索引或 GPU1 测试环境。
- **调用链边界**：本轮评测走 `evaluate_canonical_retrieval.py → RAGAdapter.answer → QueryService.answer → Ollama`，未进入 `/api/agent/query`、`AgentExecutor` 或 `agent_synthesizer.py`。
- **Agent 层**：暂缓（用户确认先不管）。

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

### Ticket 3 收尾：科学 evidence 窗口统一

- [ ] 3.1 移除 `_deterministic_scientific_evidence_answer_if_supported()` 及专用检索路径中残留的固定窗口（当前仍有 900/1000/1400，search.py:1221/1605/2090 等），改为 required-anchor + token budget。
- [ ] 3.3 `_window_text()` 统一 token budget；anchor 不足时返回 evidence-insufficient 而非静默丢弃。

### Ticket 4 收尾：表格 evidence 路径

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

## 暂缓（方案已定，RAG 达标后再动）

### ⏸ Ticket 9：Agent 收窄为"会话网关 + 受约束编排"，synthesize 改为受约束组织器

**目标**：消灭 Agent 层的答案破坏（上次评测 5/30 → ≥28/30）。纯 RAG 29/30 证明答案架构是对的，破坏来自 [agent_synthesizer.py](src/app/services/agent_synthesizer.py) 无差别重写 RAG 草稿。

**前置依赖**：RAG 层先到 30/30（Ticket 3/4/6 完成）。synthesize 是组织器不是事实来源，救不了 RAG 层没有的事实。

- [ ] 9.1 **路由决策矩阵**：单轮（simple_rag / evidence_required / table_or_metric）与对比（multi_source_compare）跳过 synthesize，直接透传 rag.answer。破坏路径关闭。
- [ ] 9.2 **跨轮引用识别**：检测 query 中跨轮指代（"之前/刚才/对比/那个/它"），仅该场景启用 synthesize。
- [ ] 9.3 **synthesize 改造三要素**：
  - 输入收窄：记忆事实 + 当前答案，不是证据全集；
  - facts 清单注入：从 citations 提取 实体→数值→表号，要求精确复制；
  - 输出校验：`_unsupported_answer_numbers` + `_answer_lacks_requested_metrics`，证据没有的数字一律打回。
- [ ] 9.4 conversation_summary 保留为跨轮引用 synthesize 的输入通道；单轮不用。
- [ ] 9.5 记忆：追问指代走 `_contextualize_retrieval_query` 扩展；跨轮事实走 conversation_summary + citations 注入。
- [ ] 9.6 回归：Agent 评测（30 题）从 5/30 提升到 ≥28/30；单轮题不退化；trace/安全边界不破坏。

**验证**：`tests/test_agent_executor.py tests/test_agent_synthesizer.py` + Agent 30 题评测。

**方案依据**：主 spec（`docs/superpowers/specs/2026-08-04-task15-rag-regression-fix.md`）的 "Agent 部分（Ticket 9）" 小节。

---

## 硬性约束

- 零框架依赖（LangChain / LlamaIndex / LangGraph 全部不引入）
- 测试环境（GPU1）只读；不删旧数据；不单独激活单篇
- 开发链路：本地 → push GitHub → pull 服务器 dev → restart
