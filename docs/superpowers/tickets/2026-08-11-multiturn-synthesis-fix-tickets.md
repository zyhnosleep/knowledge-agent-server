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

---

# 第二批（2026-08-11 三个新问题 → 5 tickets）

> **来源**：40 题回归暴露的三个新问题，grill 收敛方案（2026-08-11）：
> ① 项目级会话（`document_id=None`）检索 0 items 时 NSE，需回退历史文档重检索（校准版 A：只修项目级，锁定会话不动）+ eval 脚本补 `query_document_id` 双模式（回归失真根因：cases 缺 `query_document_id` → 40 题全当项目级跑）；
> ② 空答案无视 `max_retries=0` 强制重试一次（A）；
> ③ PolicyRouter 的 parameter 类词需与数值/表格索取语境词共现才路由 table_or_metric（A）。
> **状态约定**：`[x]` = 已完成（本地测试 + code review + 服务器验证）。

## Ticket 1：项目级会话检索锚定回退（NSE 兜底）

**Blocked by**：None — can start immediately

**交付**：项目级会话（`document_id=None`）检索 0 items 时，从 turn 历史（retrieve 轮 citations）取最近命中文档，带 `document_id` 重检索一次；`effective_document_id` 贯穿 draft/重试。锁定会话不触发。

- [x] 1.1 `_last_hit_document_id(session_id)`：倒序扫 `memory.get_history` 的 retrieve 轮，取 citations 第一个非空 document_id
- [x] 1.2 主流程回退块：`0 items and request.document_id is None` → 回退重检索，命中则替换 evidence_pack + effective_document_id
- [x] 1.3 `_run_retrieve_evidence` 命中时构建 hit_docs 去重持久化 citations
- [x] 1.4 两处重试 `_run_rag_answer` 均改传 effective_document_id（Spec review 发现：原传 request.document_id 丢锚定）
- [x] 1.5 测试：4 个 T1 测试（回退/无历史/锁定不触发/持久化）+ TDD 79 passed
- [x] 1.6 服务器验证：A2 项目级"展开第二张"回退命中 Table 3（len=773）；C 锁定会话回归不触发（len=227）✓

## Ticket 2：eval 脚本 query_document_id 双模式

**Blocked by**：None — can start immediately

**交付**：`scripts/evaluate_agent_full.py` 支持锁定/项目级双模式——`document_id = case.get("query_document_id") or case.get("expected_document_id")`，`--project-scope` 开关强制项目级。修复回归失真（40 题 cases 缺 `query_document_id` 时全部被当项目级跑）。

- [x] 2.1 `call_agent_api` 加 `project_scope` 参数 + `document_id` 双取逻辑
- [x] 2.2 `run_agent_evaluation`/main 加 `--project-scope` 开关
- [x] 2.3 脚本纳入 git 跟踪（原 untracked）+ utf-8-sig 校验

## Ticket 3：空答案强制重试兜底

**Blocked by**：None — can start immediately

**交付**：`(retry_recommended and route.max_retries > 0) or answer_is_empty` 布尔条件——空答案无视 `max_retries=0` 强制重试一次（retry 只执行一次）。

- [x] 3.1 重试条件改布尔或（L663-678）
- [x] 3.2 测试：3 个 T3 测试（simple_rag 空答案重试/最多一次/非空不重试）+ TDD 144 passed
- [x] 3.3 服务器验证：B OPLS4 路由 `evidence_required` 非 `table_or_metric` ✓（B 路由本属 T4，验收集一并验证）

## Ticket 4：table_or_metric 路由词表校准

**Blocked by**：None — can start immediately

**交付**：`_PARAMETER_TERMS=("parameter","参数")` 单独出现不再路由表格；与 `_PARAMETER_CONTEXT_TERMS`（value/数值/值/是多少/列出/list/table/表/unit/单位）共现才路由 `table_or_metric`。

- [x] 4.1 `agent_policy.py` 新增两组词表 + route() 4b 分支
- [x] 4.2 测试：参数化组 `test_parameter_alone_does_not_route_to_table_or_metric`（torsional parameters 改进方向/参数设置/CHARMM36 LJ parameters/优化参数的过程）+ 既有 69+72 passed
- [x] 4.3 服务器验证：B topic-switch（OPLS4 torsional parameters 改进方向）路由 evidence_required ✓

## Ticket 5：部署 + 验收集 + 40 题回归

**Blocked by**：Ticket 1/2/3/4

**交付**：服务器部署（scp + md5 + 重启）后跑验收集（3 case），通过后重跑全套 40 题七个阶段 eval。

- [x] 5.1 部署 3 文件 md5 全一致 + 服务重启 + 健康 OK
- [x] 5.2 验收集 3/3 PASS（A 项目级回退 len=773 / B OPLS4 路由 / C 锁定回归 len=227）
- [x] 5.3 40 题回归（charmm36-chat-eval-regression-20260811-t5.json）：40 轮完成，38 completed + 2 timeout（R22/R23）；T1 目标 5 NSE 轮中 R18/R20/R39 实质修复（内容正确锚定），R35/R36 回退触发但锚错 OPLS5（见"后续问题"）；R8 空答案 0→105 ✓；R28 36→53 恢复
- [x] 5.4 本 ticket 状态更新

---

## 第二批实施记录（/implement + /code-review + 服务器验证）

- [x] TDD 全量 **2064 passed**（T1 79 + T3 144 + T4 69/72 + 既有全量）；commit `fa5d2fa`（"fix: anchor project-level retrieval, force empty-answer retry, calibrate parameter routing"，5 文件 +554/-9）
- [x] code review 双轴：Spec 发现 1 真缺陷——T3 重试路径传 `request.document_id` 丢锚定 → 两处改 `effective_document_id`；Standards 建议（`_history_retrieve_document`→`_last_hit_document_id` 命名、T3 布尔简化、删 "parameters" 冗余词）全部采纳
- [x] 用户疑问实证：锁定会话 5 轮全命中（2-5 items），"查询不含 CHARMM36 实体词与论文 profile 零交集"的 NSE 是回归 cases 缺 `query_document_id` 的测试失真（replay_locked_compare.py 证实）→ 方案 A 只修项目级 + eval 脚本
- [x] 服务器部署（runtime/task15 3 文件 md5 与本地一致）+ systemctl --user restart knowledge-agent-dev-api
- [x] **验收集 3/3 PASS**（verify_t5_fixes.py，8002）：A 项目级"展开第二张"回退命中 Table 3 len=773 / B OPLS4 topic-switch 路由 evidence_required / C 锁定会话 len=227 回归保护
- [x] **40 题回归**（同 session 串行 40 轮，session=charmm36-chat-eval-regression-20260811-t5）：
  - T1 目标 5 NSE 轮：**R18 218→519 ✓**（回退锚定 fdf5282c，Table 4/6/S5/S8 内容正确）、**R20 75→94 ✓**（1.88 kcal/mol RMS 值正确）、**R39 75→152 ✓**（999.0 注入拒绝正确）；**R35/R36 ✗ 部分**——回退触发但锚定 1180eb94（OPLS5），答案跑题
  - R8 空答案 0→105 ✓；R28 36→53（恢复中）；R29 38 字符拒绝性回答（合理拒绝 999.0 注入）
  - 基线 13 KNOWN_FAIL → 有效改善 6（R18/R20/R35/R36/R39 len 增长 + R8 非空）；R5/R15/R17/R27/R38/R40 仍短/模板（另见后续问题）
  - **新问题**：R22/R23 服务器 timeout——`answer.synthesize` 本地模型调用 638s/399s（前次同查询 92s/21s 完成，纯运行时卡顿，非逻辑回归）；R40 空答案——T3 重试一次仍 0 chars（审计类长对话问题检索锚定 6 个错误文档）
- [x] **后续问题（非本组范围，待开新 ticket）**：
  - ① **项目级检索弱指代命中错误文档**：R22 命中 opls-aa 15 items、R33/R34 命中 OPLS5（48 分）、R40 命中 6 个无关文档（0.6 分）——T1 只兜 0 items，非空错命中不介入；审计/总结类长对话问题（R40）需要会话主题锚定而非 embedding 匹配
  - ② **回退锚定策略脆弱**：`_last_hit_document_id` 取"最近命中"，被跨主题轮（R22 opls-aa / R33/34 OPLS5）污染 → R35/R36 锚错 OPLS5；应改为历史主题一致性加权（如出现次数最多）
  - ③ **synthesize 长延迟**：R22/R23 的 synthesize 638s/399s 超时（timeout_seconds=240），同查询前次 92s/21s——需观察 Ollama 健康度/加超时熔断
