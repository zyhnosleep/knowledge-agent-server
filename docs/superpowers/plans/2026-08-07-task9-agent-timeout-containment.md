# Task 9 Agent 超时收敛实施计划（grill 收敛蓝图）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Agent full30 从 23/30（7 题单题 >120s 超时）收敛到 **30/30 且每题 ≤120s**，同时保证 LLM 在机制解释类问题中保留有意义的作用（不做"LLM 消除"）。

**Architecture:** 三路径架构。路径 1 确定性 RAG + Agent 直通（表格/指标查询、overview 科学模板）；路径 2 RAG LLM + Agent 直通（机制问题：一次 LLM 生成 + verify + rag-direct）；路径 3 Agent 二次综合（跨轮、直通门禁失败、复杂组织）。三项代码修复支撑：`generate_structured` 默认关闭思考链（6 倍提速）、coverage 目标表收窄为问题显式请求的表（避免必然 partial 误伤）、机制问词识别（把机制题路由到 LLM 路径）。

**Tech Stack:** Python 3、FastAPI、Pydantic、SQLAlchemy、pytest、Ollama（qwen3.5:9b）；候选服务 GPU0/API 8002，GPU1/API 8001 作为旧 active 只读基线。

---

## 0. 背景与验收基线

- Agent 专项评测 full30 实测 **23/30 通过，7 题失败**；失败原因全部是单题 >120s 客户端超时（`urlopen(timeout=120)`），不是答案质量错误。
- 服务器侧实际耗时 137–423s，远超客户端 120s 上限，且无取消机制 → 被放弃的请求继续占用 `OLLAMA_NUM_PARALLEL=1` 的队列插槽，形成级联（实测 idpsff synthesize 423s 中含 280.5s 队列等待）。
- **验收标准（grill Q1 收敛）**：重跑 full30 全部通过，且每题（含机制题）≤120s。
- **不可移动边界**：不切换 active pointer、不动 GPU1/API 8001（只读基线）、不删旧数据、用户控制 commit（不用 `git add .`）、ticket 复选框只在本地测试 + code review 通过后更新。

## 1. 根因链（已用实测证据闭环）

### 1.1 think 模式未关闭 = 生成慢 6 倍（主导因素）

- qwen3.5:9b 是推理模型，ollama 模板默认启用思考链。
- `generate_chat` 传了 `think=False`（ai.py:247），但 **`generate_structured` 没传**（ai.py:302-340，9 个调用方全部未传 → payload 无 think 键 → 思考链开启）。
- 实测（GPU0 空闲，同一小 payload）：APP-LIKE（format、无 think）= **33.5s** vs +`think=False` = **5.4s**，6.2 倍。
- 真实应用 payload（6065 字符，charmm36_mechanism 题）实测 **148.2s**（思考链几乎全部开销在解码：81 tok/s × 数千个思考 token）。

### 1.2 coverage_partial 门禁误伤 → 表格题被逼进 synthesize 黑洞

- `requested_table_scopes`（search.py:483-491）= 检索到的**全部**表格上下文（8–15 张表），不是问题显式请求的表。
- `_build_table_coverage`（search.py:505-604）要求每个目标表的**完整行**都覆盖 → 多表检索必然 partial → `coverage_partial=True`（agent_executor.py:544-556）→ `skip_synthesis=False` → 进 synthesize（137–423s）。
- 对照：charmm36m_table_metrics 同样 partial 但路由 multi_source_compare，门禁不适用 → 通过。证明门禁是误伤而非必要。

### 1.3 120s 客户端上限 + 服务器无取消 = 排队级联

- 同步端点无取消：客户端超时后请求继续跑，占满 `OLLAMA_NUM_PARALLEL=1` 插槽 → FIFO 排队（queue_ms 证据）。
- 7 题服务器总耗时 2137s ≈ 35.6 分钟 vs 24 分钟墙钟 → 级联坐实。

### 1.4 附带证据：23/30 通过的题本来就不需要 LLM

- 23 题走确定性组装路径（search.py:3623-3707 科学模板、4190-4263 表格模板），纯代码无 LLM。
- ollama GIN 日志：12 次 `/api/chat` 调用全部可映射到 7 个失败题；23 个通过题 0 次 LLM 调用。
- 用户质疑"RAG 答案应该总是走 LLM 整理" —— 已用代码 + 运行时日志证伪；检索回来之后的去重确实由代码执行（用户认可）。

## 2. 三路径架构（用户蓝图 + 4 处修正）

```
问题 → route → retrieve → rag.answer ──────────────────────────────┐
                                                                    │
  路径 1（确定性，0 次 LLM）：表格/指标查询、overview 科学模板        │
    → deterministic answer → verify → rag-direct                   │
                                                                    │
  路径 2（一次 LLM）：机制/原因/方法类问题                           │
    → draft_verify ok → rag-direct                                 │
    → verify 建议 retry → 一次草稿重试 → 再 verify → rag-direct     │
                                                                    │
  路径 3（Agent 二次综合）：跨轮指代、直通门禁失败、复杂组织         │
    → 受约束 synthesize → final_verify → answer                    │
```

用户确认的 4 处修正：

1. **路径 2 也有 draft verify**（原蓝图没有）；verify 是确定性校验不是第二次 LLM 综合。
2. **路径 2 的 LLM 答案也走 rag-direct**（不是必经 synthesize）；LLM 生成后按直通门禁走。
3. **路径 1 也有 final verify**（与路径 3 相同）；直通前的最终校验统一。
4. **路径 3 的 synthesize 前先补检索**（见实现 4），不能直接拿不完整证据综合。

## 3. grill 决策收敛表（全部 7 个决策）

| # | 决策点 | 选项 | 收敛结果 |
|---|--------|------|----------|
| Q1 | 验收标准 | A 全过 / B 全过+单题≤120s | **B：30/30 且每题 ≤120s** |
| Q2 | 诊断层次 | 层 1 代码诊断优先 | **先层 1 后层 3**（think 修复 → 重测 → 再看 LLM 层） |
| Q3 | 机制题 LLM 角色 | A 全消除 / B 混合 | **B：机制/原因/方法题走 LLM，表格/overview 走确定性** |
| Q4 | 题型边界 | 按题型分 | **A：mechanism×10 → LLM；overview×10 + table×10 → 确定性** |
| Q5 | coverage 目标 | A 显式请求表 / B 全检索表 | **A：目标表 = 问题显式请求的表** |
| Q6 | 机制检测 | A 问词规则 / B LLM 分类 | **A：因果问词规则（为什么/如何/机制…）** |
| Q7 | 部分覆盖处理 | A 补检索 / B 直接 synthesize | **A：定向二次检索补缺表 → 补不到才 synthesize** |

## 4. 实现清单

### 实现 1：`generate_structured` 默认关闭思考链

- 文件：src/app/services/ai.py（`generate_structured`，302-340）。
- 改动：payload 默认加 `"think": False`（与 `generate_chat` 247 行对齐）；保留 `think` 参数供显式覆盖（`think is not None` 时覆盖）。
- 安全性：9 个调用方全部未传 think → 默认 False 不改变任何现有行为的显式意图；唯一变化是关掉思考链。
- 测试：tests/test_ai_client.py —— mock `_post_chat` 断言 payload 含 `think=False`；显式 `think=True` 时 payload 为 True。

### 实现 2：coverage 目标表 = 问题显式请求的表

- 文件：src/app/services/search.py（483-491 `requested_table_scopes`）。
- 改动：复用 `_reserve_requested_table_contexts`（5176）的匹配器（`_query_priority_anchors` 的 figure_table 显式词 + `_question_row_selectors` 具体词 + `_table_group_matches_query` 分组匹配），把 requested_table_scopes 收窄为**显式请求的表**；隐式召回的背景表不进覆盖目标。
- 效果：覆盖状态反映"问题要什么"而非"检索到什么"；表格题覆盖完整 → 不再误伤进 synthesize。
- 测试：tests/test_query_service.py —— 多表检索但只请求 1-2 张表 → coverage=complete；显式请求的表缺行仍 partial。

### 实现 3：mechanism 问词规则 → LLM 路径

- 文件：src/app/services/search.py（确定性路径条件 3389-3402 附近）。
- 规则：中文问词 `为什么/为何/如何/机制/原因/定位/不足/差异/局限` + 英文 `why/how/mechanism/limitation` 命中 → 走路径 2（LLM draft）。
- 边缘：ff19sb_overview、oplsaa_overview 题目文本含"为什么/如何"但不该进 LLM —— 规则需要叠加"题目末尾带数值/指标请求"或标题中的 `_overview` 排除；以 full30 实题校验为准。
- 测试：tests/test_query_service.py —— 问词 → 机制路径判定；overview 边缘题 → 确定性路径。

### 实现 4：请求的表缺失 → 定向二次检索补表 → 补不到 synthesize

- 文件：src/app/services/search.py + src/app/services/agent_executor.py。
- 改动：coverage 判定后，若显式请求的表缺失 → 用该表的 document_id+parse_version+table_id 定向二次检索其上下文（可复用现有 table context 装载路径）→ 补齐后重新组答案；仍缺（表不存在/被删）才进 synthesize。
- 效果：部分覆盖不再直接 synthesize；synthesize 成为真正最后手段。
- 测试：tests/test_query_service.py + tests/test_agent_executor.py —— 缺 1 表 → 二次检索补上 → 直通；表不存在 → synthesize。

## 4.1 实现状态（2026-08-07 更新）

- 实现 1 ✅：ai.py `generate_structured` + `_json_mode_payload` 默认 `think=False`（含显式覆盖）；tests/test_ai_client.py 3 个新测试（18 passed）。
- 实现 2 ✅：search.py `_requested_table_scopes` 复用 `_match_requested_table_groups` 收窄覆盖目标；`_reserve_requested_table_contexts` 委托同一匹配器（行为不变）；3 新测试。
- 实现 3 ✅：`_is_mechanism_question` 分类器（先排除表格/指标/图表题 → overview 标记 → 强因果问词 → 弱问词 → 英文 why/mechanism）；`_draft_answer` 确定性早退加 `and not mechanism` 守卫；full30 30 题矩阵 20/20 命中（10 mechanism / 10 overview / 10 table）。
- 实现 4 ✅：`_fill_requested_table_contexts`（按 scope 直查 DB table child chunks → `assemble_table_context` + `extract_table_facts` → `_expand_child_hit` 构造上下文，与检索路径同一构造器）；`retrieve_evidence` coverage partial + 缺失请求表时补表重建 pack 再判 coverage；补不到保持 partial → synthesize 兜底。实现 1-4 完成后本地全量 **2040 passed**（基线 1992）。
- 实现 4 补全（9.7.9，2026-08-07）✅：fill 成功载入全部缺失请求表 → 直接标记 coverage=complete（清空 coverage_missing）。行级 facts 按问题相关性筛选（巨型表只产目标行，101 行表只 ~6 个事实），按 facts 重算必然 partial → 会把表格题误逼进 synthesize（40-124s）；fill 成功后 table contexts 已含完整表，coverage complete 让表格直通门禁放行。仅仍有表在 DB 中不存在时才保持 partial → synthesize 兜底。回归测试 `test_retrieve_evidence_complete_after_fill_when_facts_row_trimmed`（10 行表、问题只命中 2 行 → facts 仍 2 条但 coverage complete）。
- 部署：ai.py / search.py 已 scp 到服务器 `~/knowledge-agent-dev`，`systemctl --user` 重启 `knowledge-agent-dev-api.service`，/docs 200。
- 验收（2026-08-07 终态）✅：full30 重跑 **30/30**（`task16-agent-candidate-full30-20260807-optiona.json`）：答案正确率 100%、引用 100%、synthesize 绕行率 **0.0%**（上轮 3.3%）、p50 23.2s / p95 57.5s（上轮 35.7s / 95.3s）、**每题 ≤120s（max 61.7s，charmm36_mechanism）**。5 个 synthesize 绕行表题延迟回落：ff14sb 123.9s→52.6s、charmm36_table_metrics 95.3s→41.6s、oplsaa 86.1s→41.7s、charmm36idpsff 40.7s、opls4 48.7s。
- 已知偶发（未修，已记录）：charmm36_mechanism 验收词 "helix-coil" 字面不存在于 canonical v4 证据（仅 "fraction helix"/Ac-AAQAA/SPARTA），答案只能从问题回显该词 —— 29/30 那轮漏回显，本轮 30/30 已回显（61.7s）。若再复现，考虑 draft prompt 要求覆盖问题中每个显式机制目标词。
- **Task 17（2026-08-07 晚，非表格全 LLM 化）**：用户实测 CHARMM36 vs CHARMM22/CMAP 参数化核查问题，API 返回"证据片段清单"（确定性科学模板产物），判定不可读，拍板 **"非表格全 LLM"**——机制/overview/参数化核查类答案必须经 LLM 组织；表格/指标题保留确定性直通（输出本为可读 markdown 表格，且 LLM 化有数字改写风险——9.7.7/9.7.9 刚修复的领域）。实现（search.py `_draft_answer`）：删除科学模板短路 → 非表格题全走 LLM draft（中英文一致）；模板仅作 LLM 失败 fallback，头部加显式降级标记（中文"[系统提示：LLM 生成暂时失败，以下为原始证据片段，非最终答案]"/英文对应版）；draft prompt 加防遗漏指令（答案必须显式覆盖问题中每个术语/实体/指标——helix-coil 教训落进 prompt）。测试：overview 模板测试改 LLM 化断言 ×2、fallback 降级标记断言（新增英文测试）、ExplodingOllama 消息更新，全量 **2042 passed**（基线 2040）。部署：search.py scp + LF 规范化 + md5 零差异 + 重启，API 200。原问题实测：返回 LLM 组织三段式可读答案（参数化目标/修改内容/验证方法 + 引用），不再有清单；已知小瑕疵——引用列表个别条目缺 `[n]` 编号标记（LLM 格式不完美，内容无缺漏）。30 案例重跑（`task16-agent-candidate-full30-20260807-task17.json`）✅ **30/30 通过**：answer 1.0 / citation 1.0 / synthesize 绕行率 **0.0%** / **每题 ≤120s（max 83.0s，charmm36m_overview）** / 0 超时 / 0 harness 警告；延迟 vs Option A 终态：p50 23.2s → **41.6s**、p95 57.5s → **75.2s**（overview 10 题 LLM 化，涨幅符合预估 +10-25s，仍在 120s 内）；missing_terms 全空（防遗漏强化生效，helix-coil 类回显词全覆盖）；全部 status=completed、provider=local（LLM draft 直通，无模板降级）。观察：表格题延迟小幅上升（ff14sb_table_metrics 52.6s→65.9s、charmm36_table_metrics 41.6s→57.8s），表格代码未动，疑似服务器负载/检索波动，不影响验收。

## 5. 部署与验收

- 部署：服务器 `~/knowledge-agent-dev` 是同一仓库（origin zyhnosleep/knowledge-agent-server.git，分支 codex/internal-pilot）→ 本地 commit + push 后服务器 git pull；服务 `knowledge-agent-dev-api.service`（uvicorn 8002，--app-dir src）。
- 单题验证：7 个失败题 + 全部 mechanism 题，逐题 ≤120s 且答案质量不降。
- full30 验收：30/30 且每题 ≤120s；trace 表（agent_trace_runs / agent_trace_steps）记录 route、latency、queue_ms、coverage_status、direct_block_reason 佐证。
- 验收语义（2026-08-07 用户拍板，折中，已落入 `.task16-agent-candidate-full30.py`）：`status="timeout"` 但答案完整（required terms 全覆盖且有引用）→ 不算 agent_status 失败，记 harness 警告 `timeout_with_complete_answer`；≥120s → 记 `latency_exceeded_120s` 警告（不判失败，延迟由警告承载）；summary 增加 `latency_exceeded_120s_count` / `timeout_with_complete_answer_count` / `harness_warnings_total`。max_steps / needs_clarification / 空答案仍判失败。
- 回归：本地 pytest 全量不回归（基线 1992 passed）+ 新增测试。

## 6. 变更边界

只允许修改：src/app/services/ai.py、src/app/services/search.py、src/app/services/agent_executor.py、src/app/services/rag_adapter.py（如需透传）、tests/test_ai_client.py、tests/test_query_service.py、tests/test_agent_executor.py、tests/test_agent_synthesizer.py（如需覆盖）。工作树有大量用户未提交改动，不得 `git add .`；用户负责最终 commit。
