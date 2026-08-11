# Task 9 Agent 首轮直通与跨轮受约束综合后续实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保证已通过 RAG 验收的普通首轮问题直接返回 rag.answer，只在跨轮指代或复杂多跳场景使用受约束的 synthesize，并让表格覆盖信息在 RAG 与 Agent 之间可审计地传递。

**Architecture:** rag.answer 是 RAG 层已经完成检索、事实组装、答案组织和引用绑定的最终草稿；普通首轮不再重复调用 LLM 综合。Agent 只对候选直通结果执行确定性的草稿门禁，复杂多跳和跨轮问题才调用 synthesize，并在最终答案上执行最终校验。typed inventory 由 RAG 侧按 document_id + parse_version 读取和使用，Agent 只接收轻量的表格覆盖元数据和已结构化的 facts，不把整份 manifest 注入 LLM prompt。

**Tech Stack:** Python 3、FastAPI、Pydantic、SQLAlchemy、pytest、Ollama；候选服务为 GPU0/API 8002，GPU1/API 8001 作为旧 active 只读基线。

---

## 0. 约束与当前基线

本计划建立在以下事实之上：

- RAG candidate full30 已达到 30/30；本计划不重新设计 RAG 检索排序，也不把已通过的 RAG 答案再次送入普通首轮综合。
- **当前验证基线（2026-08-06 核实）**：Agent 专项测试 **410 passed**；全量测试 **1992 passed / 0 failed / 3 skipped**，共收集 1995 项；`py_compile` 已通过。这里的 1995 是 collected 总数，不是 passed 数。GPU0 candidate 的真实服务器 30 题评测尚未完成，不能把本地全绿写成服务器验收通过。
- **工作树现状**：Task 1/2/5 的直通主体以及 Task 3/4 的第一版 coverage、expected-facts 代码已在工作树实现（未提交）。本次 review 确认仍需补齐三类边界：retry 后最终答案没有再次 verify、显式 `coverage_status="partial"` 且 `table_facts=[]` 时可能错误直通、`_expected_facts_from_question` 仍从已召回 facts 反推需求集合。因此本计划把这些边界列为必须完成的增量，不能仅以当前测试全绿作为完成依据。
- 当前工作树包含大量用户未提交改动（含非 task9 的 auth/streaming 等）；执行时只能修改本计划列出的文件，不得使用会连带全部改动的 git add .。
- 不切换 active pointer，不修改 GPU1/API 8001，不删除旧 parse、chunk、vector 或 artifact 数据。
- 用户负责最终 commit；执行者只在每个验收点提供测试和 trace 证据，不自动提交或同步生产 active。

验收主线：

~~~text
首轮普通问题
  route → retrieve → rag.answer → draft_verify → rag-direct

首轮复杂多跳 / 跨轮指代
  route → retrieve → rag.answer → constrained synthesize → final_verify
~~~

“verify”是确定性校验，不是第二次 LLM 综合。普通首轮的目标是让校验决定是否可以直通，而不是借校验之名重新生成答案。

## 1. 文件职责与变更边界

后续实现只允许落在以下职责内：

- src/app/services/agent_executor.py：直通候选判定、draft verify / final verify 的顺序、retry 后重新校验、trace 元数据。
- src/app/services/answer_verifier.py：仅在现有结果字段不足以区分“校验工具成功”和“答案允许直通”时补充明确字段；不改变其只读、确定性性质。
- src/app/schemas/agent.py：为 EvidencePack 增加版本绑定的轻量 typed inventory / coverage 字段；保留现有 table_facts 兼容性。
- src/app/services/search.py：按目标表分组保留完整行组，在 token budget 前生成覆盖状态；使用已有 canonical table 事实，不重新引入 BM25 或固定窗口。
- src/app/services/rag_adapter.py：把 RAG 侧生成的 coverage 元数据带入 EvidencePack，保持 rag.answer 与 rag.retrieve_evidence 的版本作用域一致。
- src/app/services/tool_registry.py：透传新增 EvidencePack 字段和 narrow_context，更新工具 schema，不在工具层重新推断 facts。
- tests/test_agent_executor.py：覆盖首轮直通、复杂/跨轮综合、verify 门禁和 retry 后重验。
- tests/test_agent_synthesizer.py：覆盖跨轮收窄 prompt、value/term/unit/table label 保真及 coverage 缺失行为。
- tests/test_query_service.py：覆盖 typed inventory 版本绑定、目标表分组保留和 token budget 后的 coverage 状态。
- tests/test_tool_registry.py：覆盖 EvidencePack 新字段的 schema 与透传。
- docs/superpowers/tickets/2026-08-04-task15-tickets.md：验收完成后再更新复选框和实际统计，不提前标记完成。

## 2. Task 1：验证既有直通行为并修复退化 stub 测试

**Files:**

- Modify: tests/test_agent_executor.py
- Inspect: src/app/services/agent_executor.py:437-560

> 本任务已从"写红测"改为"验证既有实现 + 修测试"：直通行为已在工作树实现（见第 0 节现状）。

- [ ] **Step 1: 修复 `test_route_matrix_skips_synthesis_for_single_turn_queries` 的退化 stub**

当前失败点（tests/test_agent_executor.py:1775-1782）：table_or_metric 用例用默认 stub 答案 `"test answer"`（无数字、citation 无 `page_kind="table"`），`AnswerVerifier._has_table_evidence` 判缺失 → `retry_recommended=True` → 直通被正确阻止（Task 2 规格）。修复方式（二选一）：

~~~python
# 方案 A：stub 答案带数值（推荐，贴近真实 RAG 答案）
executor2 = make_executor(db, rag_answer_text="test answer 1.18")
# 方案 B：citation 标注 page_kind="table"
# Citation(document_id="d1", chunk_id="c1", score=0.92, excerpt="sample excerpt", page_kind="table")
~~~

修复后断言与既有一致（`_assert_direct_pass_through`）。

- [ ] **Step 2: 核对既有直通与对照测试覆盖**

验证既有测试已覆盖（无需新增，缺项才补）：
- 直通断言：`final_answer == rag_answer`、`answer_model == "rag-direct"`、无 synthesis step、`synthesis_skipped=True`（`_assert_direct_pass_through`，tests/test_agent_executor.py:1741-1748）；
- 对照：`complex_multi_hop` 仍调用 synthesize；跨轮 query（`_cross_turn_request` helper）调用 synthesize 且收到 `narrow_context=True`；无指代完整问题可直通。

- [ ] **Step 3: 运行验证**

Run:

~~~powershell
$env:PYTHONPATH='src'
& 'D:\Miniconda3\python.exe' -m pytest tests/test_agent_executor.py -q
~~~

Expected: 4 种直通路由全部 `rag-direct`；complex/跨轮仍综合；`test_route_matrix_skips_synthesis_for_single_turn_queries` 转绿。

## 3. Task 2：draft verify 前置门禁——验证既有实现并明确 coverage 门禁生效范围

**Files:**

- Modify: src/app/services/agent_executor.py:437-560,1036-1124（仅在验证中发现缺口时改）
- Modify: src/app/services/answer_verifier.py only if a separate direct-eligibility field is required
- Test: tests/test_agent_executor.py

> 门禁顺序（direct_candidate → draft verify → RAG retry 后重验 → skip_synthesis + direct_block_reason）已在工作树实现（agent_executor.py:437-550），本任务为逐条验证 + 修正 Task 3 落地后的 coverage 门禁范围，不重写逻辑。

- [ ] **Step 1: 定义直通候选与校验结果的边界**

保留当前路由集合，但先计算 direct_candidate：

~~~python
direct_routes = {
    "simple_rag",
    "evidence_required",
    "table_or_metric",
    "multi_source_compare",
}
direct_candidate = (
    not attachment_only
    and not max_steps_hit
    and route.route in direct_routes
    and not cross_turn
)
~~~

complex_multi_hop 和跨轮场景不需要为了决定直通而提前调用 verify；它们直接保留 synthesize，最后使用 final_verify。

- [ ] **Step 2: 对 direct candidate 先执行 draft verify**

在调用 answer.synthesize 之前调用现有确定性校验工具。直通必须同时满足：

~~~text
draft_verify.retry_recommended == False
rag_verification_status != "contradicted"
answer_text 非空
citations 非空
coverage 门禁（仅当满足下列生效条件时；见下）
~~~

**coverage 门禁生效范围（防直通回归）**：coverage 只作为 `table_or_metric` 路由且 RAG 已明确计算表格覆盖时的直通前置条件；明确 `coverage_status="partial"` 必须阻止直通，**即使 `table_facts=[]` 也不能例外**。旧 EvidencePack 没有 coverage 字段时为 `unknown`，以及非表格路由（`simple_rag` / `evidence_required` / `multi_source_compare`）保持中立，避免兼容旧调用方时把已通过的直通全部退化。不能用 `bool(table_facts)` 代替 coverage 是否已计算。

不要把 answer.verify 的 ok=True 当成答案合格标志；当前 verifier 的 ok 表示工具本身执行成功，直通资格应看 retry_recommended 与 RAG coverage。

- [ ] **Step 3: 处理 draft verify 失败和 retry**

当 draft verify 建议 retry 时，禁止先标记 rag-direct。如果路由允许 RAG retry，则 retry 后重新执行 draft verify；retry 前的结果不能沿用。若仍未通过，进入现有受约束 synthesize / fallback 路径，并让最终答案经过 final_verify。

当 draft verify 通过且没有后续 RAG retry 时，将该结果作为直通路径的最终校验结果，避免为同一答案重复消耗一次 tool call 和 step。

若 synthesize 路径的 final verify 建议重试，`_run_rag_answer` 返回的新答案和 citations 必须成为新的待校验对象：

```python
retry_answer, retry_citations, retry_calls, retry_verification_status = self._run_rag_answer(...)
if retry_answer.strip():
    tool_calls += retry_calls
    answer_text = retry_answer
    citations = retry_citations
    rag_verification_status = retry_verification_status or rag_verification_status
    verify_result = self._run_verify(
        request.query, answer_text, citations, route.route,
        constraints, steps, usage, tool_calls,
    )
```

重试后的 `verify_result` 必须覆盖旧结果，`retry_recommended`、`rag_verification_status` 和 finalize trace 必须描述最终答案；不能只解包后丢弃 `_retry_verification_status`，也不能用第一次 verify 的通过状态接受第二个答案。

- [ ] **Step 4: 为 retry 后重验写回归测试**

使用按顺序返回“第一次 `retry_recommended=True`、第二次 `retry_recommended=False`”的 fake verifier，以及第一次/第二次不同答案的 fake RAG。断言最终答案是第二次 RAG 结果、`answer.verify` 至少调用两次、finalize metadata 的 `rag_verification_status` 与第二次结果一致，并且 trace 中没有把第一次的 retry 建议当成最终状态。

- [ ] **Step 5: 保留 synthesize 路径的 final verify**

复杂/跨轮路径仍按现有顺序在 synthesize 后执行 final_verify。trace 至少记录：

~~~text
draft_verification（若执行）
final_verification（若执行）
synthesis_skipped
direct_block_reason
rag_verification_status
~~~

- [ ] **Step 6: 运行绿测**

Run:

~~~powershell
$env:PYTHONPATH='src'
& 'D:\Miniconda3\python.exe' -m pytest tests/test_agent_executor.py tests/test_tool_registry.py -q
~~~

Expected: 普通首轮无 synthesis step；复杂/跨轮仍综合；verify retry、retry 后重验、空答案、无引用和 contradicted 场景均不能错误标记为 rag-direct，且最终 trace 与最终答案一致。

## 4. Task 3：在 RAG 侧建立轻量 typed inventory / coverage 通道

**Files:**

- Modify: src/app/schemas/agent.py:182-267
- Modify: src/app/services/search.py:2742-2800,3418-3500,4647-4750
- Modify: src/app/services/rag_adapter.py:49-125
- Modify: src/app/services/tool_registry.py:298-362,497-528
- Reuse read-only loader: src/app/services/canonical_artifacts.py:1850-1875
- Test: tests/test_query_service.py, tests/test_tool_registry.py

- [ ] **Step 1: 定义版本绑定的轻量 inventory 结构**

不要把完整 manifest 或全部 child 文本放入 prompt。EvidencePack 增加一个可选的 inventory 字段，每个表记录至少保留 canonical manifest 已有的：

~~~text
document_id
parse_version
table_id
row_count
source_block_ids
child_ids
child_count
parent_ids
row_indices
~~~

另增加明确的 `coverage_status`（`complete` / `partial` / `unknown`）和 `coverage_missing_tables`，避免用空列表同时表示”没有表”和”尚未计算”。状态语义必须固定：

- `unknown`：旧调用方没有 coverage、版本/inventory 无法加载，或当前问题没有可确定的目标表；不能据此声称完整覆盖。
- `partial`：RAG 已经识别出目标表并完成 coverage 计算，但至少一个目标表/行没有进入返回 facts；目标表请求存在而 `table_facts=[]` 时仍必须是 `partial`，不能提前返回 `unknown`。
- `complete`：所有已解析的目标表和目标行都在同一版本的 inventory/facts 覆盖范围内。

旧调用方不提供这些字段时按 `unknown` 兼容，但 RAG 明确返回的 `partial` 优先于 facts 列表是否为空。

**本任务还包含 per-fact schema 扩展（9.3.2 补齐）**：当前 `TableFactEvidence` 只有 `fact_id`/`unit`/`term`（schemas/agent.py:245-247），Task 4 的 inventory↔facts join 需要每 fact 至少带 `document_id`、`parse_version`、`table_id`、`row_index`/`row_label`、`column`（value/unit/term 已有）。search.py 生成 facts 时补齐这些字段（现有行索引与表上下文在生成点已可得），旧调用方不提供时默认空串兼容。

- [ ] **Step 2: 从 canonical manifest 加载同版本 inventory**

复用 CanonicalArtifactStore.load_typed_inventory(document_id, version)，版本必须来自 QueryService.parse_version_map 或当前明确解析版本；禁止把 active 与 candidate 的 inventory 混合。manifest 缺失或版本不匹配时返回 coverage_status="unknown"，并记录可审计 warning，不伪造 complete。

- [ ] **Step 3: 在 token budget 前计算目标表覆盖**

在 `_fit_contexts_to_token_budget` 之前使用现有表格路由结果锁定目标表。为支持“目标表请求但 facts 为空”的情况，coverage 构建函数必须接收同一版本作用域的 `requested_table_scopes`（`document_id + parse_version + table_id`），不能只从 `table_facts` 推导 scope。每个目标表至少保留一个完整的 canonical row group；剩余预算再分配给非关键 evidence。若目标表在 inventory 中存在但目标 row group 未进入 facts，或目标表根本没有返回 facts，均标记 `partial` 并加入 `coverage_missing_tables`，不能让下游把“未返回”解释成“用户未请求”。

- [ ] **Step 4: 让 RAGAdapter 和工具注册表只透传元数据**

rag.retrieve_evidence 返回 EvidencePack 时带上 inventory/coverage；rag.answer 仍返回 RAG 的最终答案和 citations。Agent 只使用 coverage 做门禁，不让工具层从 excerpt 重新猜完整 facts。

- [ ] **Step 5: 为版本和截断写测试**

至少覆盖：

~~~text
candidate parse_version_map → inventory 与 facts 同版本
active inventory 与 candidate facts 混合 → coverage 失败
目标表在 inventory 中但 row group 被预算裁剪 → partial
目标表与 facts 完整 → complete
明确请求 Table 7 但 `table_facts=[]` → partial + `coverage_missing_tables` 包含 Table 7
明确请求 Table 7 但只返回 Table 2 facts → partial，不能被 table route 直通
旧 EvidencePack 无 inventory → unknown，保持兼容
~~~

Run:

~~~powershell
$env:PYTHONPATH='src'
& 'D:\Miniconda3\python.exe' -m pytest tests/test_query_service.py tests/test_tool_registry.py -q
~~~

## 5. Task 4：把 expected facts 改成需求侧目标，不再用供给反推完整性

**Files:**

- Modify: src/app/services/agent_synthesizer.py:1133-1202
- Modify: src/app/services/agent_executor.py direct coverage gate
- Test: tests/test_agent_synthesizer.py, tests/test_agent_executor.py

- [ ] **Step 1: 定义需求侧 target**

`_expected_facts_from_question` 必须显式接收 typed inventory，不能只接收 EvidencePack 后从 `table_facts` 反推。函数接口固定为：

```python
def _expected_facts_from_question(
    self,
    question: str,
    inventory: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Return (status, normalized demand-side fact targets)."""
```

返回列表是由问题解析出来的需求目标，不是已召回 facts；目标至少表达：

~~~text
table_ref / table_id
row aliases or row indices
column/term aliases
source document and parse_version
~~~

显式 Table N、OPLS/CHARMM 等已存在的表格路由别名优先使用现有 QueryService 规则；无法从 inventory 唯一解析目标表/版本时返回 `unknown`，不能静默回退成”所有已召回 facts 都是期望 facts”。问题中的列/指标需求必须独立解析，不能因为某个 Table N fact 命中就把该表的其他列判为已满足。

**unknown 的默认行为（防校验静默禁用）**：`expected_facts_status=”unknown”` 时（1）**不触发** facts 遗漏回退（无法判断期望集合，`_synthesis_missing_fact_fields` 直接短路返回空）；（2）**保留** draft-fidelity / 表号 / unsupported-number 锚点守卫（它们不依赖期望集合）；（3）trace/finalize metadata 记录 `expected_facts_status=”unknown”` 供审计。即：unknown 缩小的是”facts 字段遗漏”这一条校验的覆盖范围，不是整条 fidelity 链，且状态可见。现有实现 `return matched or dict_facts`（agent_synthesizer.py:1196）的无命中回退全部行为在本任务中移除。

- [ ] **Step 2: 用 inventory 与 canonical facts 做 exact join**

匹配顺序固定为：`document_id + parse_version + table_id + row_index/row_label + column/term`。先用 inventory 和 question 生成所有 demand targets，再用已返回 `table_facts` 建立 exact identity index；`table_facts` 只提供返回事实的 `value/unit/fact_id`，不能定义期望集合。join 结果区分 `complete`、`missing` 和 `unknown`：

```python
targets = parse_question_targets(question, inventory)
returned = index_facts_by_identity(table_facts)
missing = [target for target in targets if target["identity"] not in returned]
status = "complete" if targets and not missing else "missing"
```

`Table N` 只负责约束 `table_id`，不能以“任意一个 Table N fact 命中”作为 complete。最小回归例必须覆盖：问题要求 `Asp、C6、exptl`，只返回 `Asp=21.0` 时状态为 `missing`，而不是 `complete`。

- [ ] **Step 3: 让 synthesize fidelity guard 使用 join 结果**

保留现有 value/term/unit/table label/unsupported number 守卫，但只有“RAG draft 覆盖且 synthesize 遗漏”的已返回 target 才触发遗漏回退。若需求目标在 inventory 中存在但 facts 未返回，状态为 `missing`，由 coverage/证据不足路径处理；不能把供给缺失误报为 synthesize 遗漏，也不能让 LLM 猜测缺失值。

- [ ] **Step 4: 给直通路径接 coverage，而不是接完整 prompt facts**

普通首轮通过 rag.answer 后，Agent 只读取 coverage_status 和 RAG verification_status 决定是否直通；直通路径不重新组装答案。完整 facts 清单仅在跨轮/复杂 synthesize 需要时传入受限 prompt。

- [ ] **Step 5: 增加缺失供给的回归测试**

构造“问题明确要求 Table 7，但 table_facts 被截断为空/仅剩 Table 2”的输入，断言：

~~~text
expected target 仍存在
coverage 不为 complete
普通直通被阻止
不会因为任意一个 Table 7 fact 或无关 Table 2 fact 命中而错误通过
~~~

同时新增以下最小回归：

- 多列部分召回：问题要求 `Asp、C6、exptl`，只返回 `Asp=21.0`，状态必须为 `missing`。
- 空 facts：问题要求 Table 7，inventory 可解析但 `table_facts=[]`，expected target 仍生成，状态为 `missing`，coverage 为 `partial`，table route 不能直通。
- 错表 facts：问题要求 Table 7，只返回 Table 2，不能把 Table 2 fact 当作 Table 7 的满足项。

保留现有 1.18 value、% unit、term、Table 7 和 999.0 新数字测试，并新增 term 删除测试。

## 6. Task 5：收紧跨轮 synthesize 的输入与输出边界

**Files:**

- Modify only if needed: src/app/services/agent_synthesizer.py:82-309,460-735,760-1050,1036-1202
- Modify: src/app/services/tool_registry.py:547-588
- Test: tests/test_agent_synthesizer.py, tests/test_tool_registry.py

- [ ] **Step 1: 保持 narrow_context 语义**

跨轮 prompt 只保留：conversation summary 中可解析事实、当前 rag_answer、citations 和结构化 table_facts；不加入完整 evidence pack item 摘录，也不把 typed inventory 全量送给模型。local、ollama、external 和 coverage retry 四条路径都必须一致。

- [ ] **Step 2: 保持 LLM 角色为组织器**

Prompt 明确要求保留已覆盖事实、表号、术语、单位和引用；禁止补写 canonical facts 中不存在的数字。RAG 草稿已完整时，任何无理由的 synthesize 遗漏都回退 RAG 草稿。

- [ ] **Step 3: 运行 Agent 相关回归**

在现有共享 helper 之外，使用 fake provider 做四路径矩阵测试：local synthesis、显式 ollama synthesis、external provider synthesis、stream/retry synthesis。每条路径都断言 `narrow_context=True` 时 evidence item 摘录和完整 inventory 不出现在 prompt；coverage retry 的第二次调用也必须保持同样收窄，不能只在第一次 local 调用生效。stream 路径的断言放在 `tests/test_agent_streaming.py`，其余 provider 断言放在 `tests/test_agent_synthesizer.py`，工具透传仍由 `tests/test_tool_registry.py` 覆盖。

- [ ] **Step 4: 运行四路径回归**

Run:

~~~powershell
$env:PYTHONPATH='src'
& 'D:\Miniconda3\python.exe' -m pytest tests/test_agent_executor.py tests/test_agent_synthesizer.py tests/test_agent_streaming.py tests/test_tool_registry.py -q
~~~

Expected: 直通测试不调用 synthesizer；跨轮/复杂测试调用受约束 synthesize；local/ollama/external/stream-retry 均不泄漏完整 evidence；facts 遗漏和新数字均安全回退。

## 7. Task 6：本地全量验证与文档更新

**Files:**

- Modify after evidence is complete: docs/superpowers/tickets/2026-08-04-task15-tickets.md
- Update: C:\Users\响睡觉\AppData\Local\Temp\handoff-task9-agent-containment.md only when handing off a new verified revision

- [ ] **Step 1: 运行专项和全量测试**

前置：Task 1 Step 1 的 stub 修复必须已合入；否则专项和全量结果不能作为本计划的有效基线。

~~~powershell
$env:PYTHONPATH='src'
& 'D:\Miniconda3\python.exe' -m pytest tests/test_agent_executor.py tests/test_agent_synthesizer.py tests/test_agent_streaming.py tests/test_tool_registry.py -q
& 'D:\Miniconda3\python.exe' -m pytest tests/ -q
& 'D:\Miniconda3\python.exe' -m py_compile src/app/services/agent_executor.py src/app/services/agent_synthesizer.py src/app/services/search.py src/app/schemas/agent.py src/app/services/rag_adapter.py src/app/services/tool_registry.py
~~~

记录实际 `passed / failed / skipped` 和 collected 总数，不使用旧 handoff 的统计替代新输出。当前基线是 **410 passed**（专项）以及 **1992 passed / 0 failed / 3 skipped，1995 collected**（全量）；修正后必须重新运行并以新输出为准，不能把 collected 数写成 passed 数。

- [ ] **Step 2: 检查 direct/synthesis trace 统计**

从测试或本地 API 结果中确认：普通首轮没有 synthesis step；复杂/跨轮有 synthesize；answer_model、synthesis_skipped、coverage、verify 和 fallback reason 一致；尤其确认 retry 后最终答案对应最终 verify 结果，finalize trace 没有残留旧答案的 verification status。

- [ ] **Step 3: 更新 Task 9 文档状态**

只有本地测试和代码 review 都通过后，才把计划/ticket 中对应 [ ] 改为 [x]，并注明未完成的服务器验收项；不得把“本地测试通过”写成“服务器 30/30 已通过”。

## 8. Task 7：GPU0 candidate 服务器分层验收

**Files/Environment:**

- Server: 192.168.31.20 / SSH alias llm-wiki-server
- Working directory: ~/knowledge-agent-dev
- Candidate API（服务器内执行评测时）：http://127.0.0.1:8002；从内网主机预检时对应 http://192.168.31.20:8002
- Active/old baseline: http://127.0.0.1:8001，只读，不修改
- Script: scripts/evaluate_agent_full.py

- [ ] **Step 1: 用户同步后执行服务器预检**

确认服务加载的是 candidate 代码和 candidate parse_version_map，并记录 active/candidate 身份；评测请求只能指向 `192.168.31.20:8002`，不得误指向 `:8001`；不执行 active pointer 切换，不重建 GPU1 数据。

- [ ] **Step 2: 运行完整 Agent 评测**

~~~bash
cd ~/knowledge-agent-dev
python scripts/evaluate_agent_full.py \
  --cases runtime/internal-research-overlap30-full-answer-cases.json \
  --report runtime/task9-agent-direct-routing-full30.json \
  --api-url http://127.0.0.1:8002
~~~

- [ ] **Step 3: 按路由分层验收**

必须分别报告：

~~~text
ordinary single-turn / direct route：相对 RAG 30/30 零退化
complex_multi_hop：单独统计
cross-turn：单独统计
overall：至少 28/30 作为辅助目标
synthesis call / skip rate：按 trace 统计
P50/P95 latency：分别统计 direct 与 synthesize
~~~

opls5_table_metrics 必须保留 Table 7、1.18、1.12 及对应 citation；任何普通首轮被 synthesize 或事实丢失都视为 Agent 回归，即使总分达到 28/30 也不能接受。

- [ ] **Step 4: 服务器验收边界检查**

确认只验证 GPU0 candidate；GPU1 active/旧版本仍在、未被删除、未被写入；验收报告包含 parse_version_map 和 active/candidate 版本身份。

## 9. 最终完成条件

本计划只有同时满足以下条件才算完成：

- 普通首轮已验证 RAG 答案直接返回，answer_model="rag-direct"，无 synthesize step；`test_route_matrix_skips_synthesis_for_single_turn_queries` 的退化 stub 已修复且全绿。
- 复杂多跳和跨轮引用仍调用受约束 synthesize，并通过最终校验。
- RAG coverage 使用同一 document_id + parse_version 的 typed inventory；Agent 不从不完整 facts 反推完整期望集合。
- **coverage 门禁仅按 Task 2 Step 2 定义的生效范围执行**：`table_or_metric` 收到显式 `partial` 时，即使 `table_facts=[]` 也必须阻止直通；非表格路由 / 旧 EvidencePack 的 coverage `unknown` 不得被误用为阻塞条件。
- **retry 后必须重新 verify**：最终答案、最终 `rag_verification_status`、`retry_recommended` 和 finalize trace 必须来自同一轮最终校验。
- **expected_facts unknown 语义按 Task 4 Step 1 定义**：只缩小 facts 遗漏校验覆盖、保留锚点守卫、trace 可见，不静默禁用。
- **expected facts 必须是需求侧精确 join**：显式请求多列时只返回一列不能 complete；Table N 命中任意一个 fact 不能 complete；空 facts/错表 facts 都必须被测试覆盖。
- local、ollama、external、stream/retry 四条 narrow_context 路径均已验证无 evidence/inventory 泄漏。
- 缺失 facts、非法数字、丢失单位/术语/表号都会阻止错误直通或触发安全回退。
- 本地专项、全量测试和编译验证都有新鲜输出。
- GPU0 candidate 服务器分层验收通过，普通直通相对 RAG 30/30 零退化。
- active pointer、GPU1 旧版本和旧数据均未被修改或删除。

## 10. 继承的已知风险（本计划不解决，验收时留意）

- `rag_verification_status != "contradicted"` 依赖 verifier 自由文本 verdict 解析，无受控词表——误判最坏情况是错误直通或错误回退，均有下游守卫兜底，低风险。
- 单位规则三处漂移：`search._FACT_UNIT_RE` vs `answer_verifier` vs anchors numeric_pattern（`us`/`K` 匹配过宽）——当前 30/30 与保真测试均未触发，低风险。
- `synth_model = "rag-direct"` 把路由决策编码进模型名——可接受，usage 可追踪。
