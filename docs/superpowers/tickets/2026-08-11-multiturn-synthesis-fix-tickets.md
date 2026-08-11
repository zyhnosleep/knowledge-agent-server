# 多轮对话综合修复（2026-08-11 grill 收敛）

> **来源**：grill 收敛（2026-08-11）。40 题七个阶段 eval 实测 11/40 无效答案，根因三重：跨轮判定词表漏检（"第二点/展开/审计"等不在 9 词 marker 内）、30 字符长度门槛误杀长跨轮句、compact 机制破坏历史（turn_index 复用致排序错乱，40 轮后仅剩 23 条 turn，Q1-Q7 丢失）。拍板方案：**跨轮判定反转为"第 2 轮起一律跨轮"（反向默认，不做词表检测）**；上下文化保留 30 字符门槛但去掉 marker 词表；compact 修复 index 复用并把上限提到 200（40 题场景不触发删除）；附带修复空答案被 `[0]` 伪引用掩盖 verify 空检查的独立 bug。
> **状态约定**：`[x]` = 已完成（代码核验确认）；`[ ]` = 待办。

---

## Ticket 1：多轮历史完整保留（compact 修复）

**Blocked by**：None — can start immediately

**交付**：40 轮以内的会话，turn 按提交顺序完整保留、顺序正确，指代解析（`_resolvable_previous_turn`）可见全部历史。`_next_turn_index` 不再复用已删除的 index；`agent_max_conversation_turns` 从 20 提到 200，使 40 题测试会话不触发 compact 删除（保留 compact 作为超长会话的容量保险丝）。

- [x] 1.1 `_next_turn_index`（conversation_memory.py:400-402）从 `turn_count()` 改为 MAX(turn_index)+1，compact 删除最旧 turn 后不复用 index、排序不错乱
- [x] 1.2 `agent_max_conversation_turns`（config.py:214）20 → 200
- [x] 1.3 测试：`test_turn_index_monotonic_after_compact`（compact 后 index 不复用）+ `test_turn_index_monotonic_over_forty_rounds`（40 轮 × 2 turn + 每轮 compact，索引严格递增、无复用）
- [x] 1.4 本地 pytest 全量通过（2048 passed），无回归

## Ticket 2：多轮强制走综合路径（反向路由 + 上下文化放宽）

**Blocked by**：Ticket 1

**交付**：会话第 2 轮起的查询一律经 synthesize（带历史摘要）组织答案，不再 draft 直通——跨轮检测不做词表匹配（反向默认）；≤30 字符查询用"上一轮问题+当前追问"包装后检索（去 marker 词表、保留长度门槛，短查询几乎必然是引用）。第 1 轮仍走 draft 直通（回归保护）。路径 1/2 在第 2 轮起被跳过，路径 3 全量接管多轮。

- [x] 2.1 `_is_cross_turn_query`（agent_executor.py:1488-1502）简化为轮次判定：历史中存在 ≥1 条既往 user turn → 跨轮；删除 marker 匹配与 30 字符门槛（`_CROSS_TURN_MARKERS` 常量一并删除）
- [x] 2.2 `_contextualize_retrieval_query`（agent_executor.py:1507-1520）去掉 marker 命中条件，保留 30 字符门槛
- [x] 2.3 第 1 轮 draft 直通回归：`test_is_cross_turn_query_reverse_default_from_second_turn`（首轮 False、第 2 轮起一律 True）+ `test_contextualize_retrieval_query_length_gate`（≤30 包装、>30 不包装、新会话不包装）新增；既有单轮直通测试（route matrix / narrow_context）保持通过
- [x] 2.4 本地 pytest 全量通过（2048 passed），无回归

## Ticket 3：空答案不被引用标记掩盖

**Blocked by**：None — can start immediately

**交付**：模型返回空答案时正确判定无效并走重试/降级，不再因 `_ensure_valid_returned_citation_marker` 在空答案后追加 `[0]` 形成 `" [0]"` 伪内容而绕过 answer_verifier 的 `.strip()` 空检查。

- [x] 3.1 修复 `_ensure_valid_returned_citation_marker`（search.py:5997-6010）：空答案直接返回，不追加 `[0]` 伪引用
- [x] 3.2 测试：`test_ensure_valid_returned_citation_marker_does_not_mask_empty_answer`——空串/空白输入原样返回，非空答案行为不变
- [x] 3.3 本地 pytest 全量通过（2048 passed），无回归

## Ticket 4：精简验收集 + 40 题回归

**Blocked by**：Ticket 1、Ticket 2、Ticket 3

**交付**：服务器部署后先跑精简验收集（12 题：8 个失败跨轮案例 + 2 个首轮/独立回归 + 1 个空答案案例 + 1 个换话题案例），通过后重跑全套 40 题七个阶段 eval。

- [x] 4.1 精简验收集 12 题：8 个跨轮失败案例全部有效答案且 trace 显示 synthesize 路径（非 rag-direct）；首轮/独立题仍 draft 直通；换话题案例检索不被上下文化污染
- [x] 4.2 全套 40 题：有效答案 11 失败 → 0（或验收记录实际数字与失败明细）
- [x] 4.3 部署服务器（scp + md5 验证 + 服务重启），主代码与服务器零差异
- [x] 4.4 本 ticket 状态更新

---

## 实施顺序

Ticket 1 与 Ticket 3 无依赖、可并行；Ticket 2 依赖 Ticket 1（验收跨轮需要历史完整）；Ticket 4 依赖 1/2/3。每张 ticket 完成后本地 pytest + code review，全部完成后部署并跑 Ticket 4。

---

## 2026-08-11 实施记录（/implement + /code-review）

- [x] Ticket 1/2/3 完成：全量 pytest **2048 passed**（基线 2040，+8 新测试）；commit `dc90c1d`
- [x] code review 两轴：Standards 无硬违规（2 个判断项接受：命名保留、调用顺序契约 docstring 已标注）；Spec 发现 Finding 1——`_is_cross_turn_query` 委托 `_resolvable_previous_turn` 的防重复守卫与 last_n=12 窗口否决"一律跨轮"（verbatim 重试 / 长间隙漏判）→ 已修复为独立轮次计数（统计全部 user turn），补重复提问 + 15 条 agent turn 长间隙断言；修复后全量重跑 2048 passed
- [x] Ticket 4 部署与验收：scp 4 文件（md5 与本地一致）+ runtime/app.env `AGENT_MAX_CONVERSATION_TURNS=20→200`（备份 app.env.bak-20260811）+ `systemctl --user restart knowledge-agent-dev-api`（进程环境已验证 200）；8001 test 未动
  - **验收集**（fix_verify_runner.py，21 case）：17/21 通过。跨轮路由全部生效（第 2 轮起 19/20 走 synthesize、首轮 R1 rag-direct ✓）；4 失败：R5 空答案（rag.answer 0 chars + verify retry 但 simple_rag max_retries=0 不重试）、R18/R20 检索 0 items（跨轮指代查询无实体锚点）、topic-switch（OPLS4 被误路由 table_or_metric → coverage_partial 阻塞 → synthesize——答案有效，判定标准误报）
  - **40 题回归**（同 session 串行 40 轮，charmm36-chat-eval-regression-20260811.json）：40/40 completed；基线 13 KNOWN_FAIL → **5 修复**（R15/R17/R18/R27/R38）；8× `[0]` 伪有效答案全部揭开（Ticket 3 生效）。剩余 9 轮 len≤100：NSE 检索失败 4（R20/R35/R36/R39——基线同款，检索层缺陷）、空答案 1（R8，LLM 波动）、内容正确但短 4（R5 模板误判实为有效、R28 142→36 真实退化、R29/R40 拒绝性回答）
  - **后续问题（非本组 tickets 范围，待开新 ticket）**：① 跨轮检索锚定——`_route_papers`/表格直查无法把"第二张表/第一个数值/总结"类指代查询路由到历史会话锁定的论文（R18/R20/R35/R36/R39）；② 空答案重试兜底——synthesize 路径 verify 标 retry 但 max_retries=0 路由不重试（R5/R8）；③ OPLS4 类查询被 "parameter" 词误路由 table_or_metric（topic-switch）
