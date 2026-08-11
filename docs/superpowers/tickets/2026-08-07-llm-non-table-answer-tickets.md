# Task 17：非表格答案强制经 LLM 组织（2026-08-07 grill 收敛）

> **来源**：grill 收敛（2026-08-07）。用户实测 CHARMM36 vs CHARMM22/CMAP 参数化核查问题，API 返回"证据片段清单"（确定性科学模板产物），判定不可读；拍板 **"非表格全 LLM"**——机制/overview/参数化核查类问题必须经 LLM 组织答案，表格/指标题保留确定性直通（输出本为可读 markdown 表格，且 LLM 化有数字改写风险——9.7.7/9.7.9 刚修复的领域）。
> **状态约定**：`[x]` = 已完成（代码核验确认）；`[ ]` = 待办。

---

## Ticket 1：非表格答案强制经 LLM 组织（含降级标记与防遗漏强化）

**Blocked by**：None — can start immediately

**交付**：搜索答案层对非表格/指标查询不再短路到确定性科学模板——机制/overview/参数化核查类问题（中英文一致）全部由 LLM 依据检索证据组织为可读叙述。确定性模板仅作 LLM 生成失败时的降级输出，且带显式降级标记。表格/指标题保持确定性表格直通。draft 提示词强化：答案必须显式覆盖问题中每个术语。

- [x] 1.1 删除非表格问题的科学模板短路：所有非表格/指标查询走 LLM draft（机制、overview、参数化核查等）；确定性模板仅在 LLM 生成失败时作为 fallback
- [x] 1.2 降级输出标记：LLM 失败 fallback 头部加显式标记（中文"[系统提示：LLM 生成暂时失败，以下为原始证据片段，非最终答案]"；英文对应版），与既有 raw-evidence fallback 风格一致
- [x] 1.3 防遗漏强化：draft 提示词要求答案显式覆盖问题中每个实体/术语/指标/组件名；兜底术语追加逻辑保留（机制题回显词如 helix-coil 类不再偶发缺失）
- [x] 1.4 表格/指标题行为不变：仍走确定性表格直通，LLM 不被调用
- [x] 1.5 测试更新：既有"科学证据题返回证据片段清单"断言改为 LLM 化行为；新增——非表格科学证据题 LLM 被调用、LLM 失败时降级标记存在（中英文）。"表格题 LLM 不被调用"未新增重复测试——该行为由既有表格直通测试（tests/test_query_service.py:5265 `generate_calls == 0` 断言，同测试 5259 为 answer 调用）显式覆盖，code review 核实后决定不重复
- [x] 1.6 本地 pytest 全量通过（基线 2040 passed），无回归

## Ticket 2：部署与 30 案例验收回归

**Blocked by**：Ticket 1

**交付**：服务器部署后重跑 30 案例，确认非表格 LLM 化后验收全守——每题 ≤120s、答案术语全覆盖、引用齐全、synthesize 绕行率保持 0%；用户报告的 CHARMM36 参数化核查问题实测返回 LLM 组织的可读答案；蓝图文档与 ticket 状态更新。

- [x] 2.1 部署服务器（scp + md5 验证 + 服务重启），主代码与服务器零差异
- [x] 2.2 30 案例重跑：30/30 通过，每题 ≤120s（max 83.0s，charmm36m_overview），p50 41.6s / p95 75.2s 记录在案
- [x] 2.3 answer 术语全覆盖 / citations 100% / synthesize 绕行率保持 0%
- [x] 2.4 用户原问题（CHARMM36 蛋白力场主要想修正 CHARMM22/CMAP 的哪些问题，列出参数化目标、修改内容和验证方法并给出引用）API 实测返回 LLM 组织的可读答案，非证据片段清单
- [x] 2.5 延迟涨幅记录 + 蓝图文档（2026-08-07-task9-agent-timeout-containment.md）与本 ticket 状态更新

---

## 2026-08-07 code review 修复（/implement）

- [x] 英文降级标记语法：is→are（"The following are raw evidence snippets"）
- [x] `_degradation_notice(question, kind)` helper：raw / template 两个降级路径共用双语样板（消除 standards 轴 Duplicated Code 发现）
- [x] `_is_mechanism_question` 死代码：Task 17 后 src/ 零调用点；决策保留 + docstring 注明（分类器测试维持词表契约，未来机制/overview 差异化路由可复用）
- [x] 修复后全量 pytest 2042 passed；重新部署 search.py（md5 零差异）；API 实测——原问题 LLM 组织答案（1605 字符）+ 表格题数值 51.9/95.3 完整保留
- [x] 修复未重跑 30 案例——8-11 用户指示重跑（agent 端点 8002 /api/agent/query 端到端，含 code review 修复 + Task 18 修复）：30/30 通过；模板降级 case 从 3 → 0（ff99sb_disp_overview / ff99sb_ildn_mechanism / oplsaa_overview 全部恢复 LLM 组织答案）；30 案例中 schema 解析失败 27 次（内部 JSON mode 重试恢复）+ 外层重试 4 次（Task 18 重试闭环实际触发）；p50 37.2s / p95 80.7s，synthesize 绕行率保持 0%
