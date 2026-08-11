# 检索锚定与合成超时修复（2026-08-11 第二批 40 题回归后续）

> **来源**：第二批（T1-T5）40 题回归（charmm36-chat-eval-regression-20260811-t5）暴露的 3 个后续问题，2026-08-11 to-tickets 拆票。
> **状态约定**：`[x]` = 已完成（本地测试 + code review + 服务器验证）；`[ ]` = 待办。
>
> **回归证据摘要**（trace 逐轮核查，session=charmm36-chat-eval-regression-20260811-t5）：
> - R22 项目级检索命中 opls-aa 15 items（35 citations 全为 opls-aa）；R33/R34 命中 OPLS5（score 48）；R40 命中 6 个无关文档（score 0.6）——弱指代查询（"回到第一张表/总结所有修改/审计对话"）embedding 匹配无主题锚定
> - R35/R36 回退触发但 `_last_hit_document_id` 取"最近命中"= R33/R34 的 OPLS5 → 锚错论文，答案跑题
> - R22/R23 服务器 timeout：`answer.synthesize` 本地模型调用 638s/399s（timeout_seconds=240），同查询前次回归 92s/21s 完成——运行时卡顿/排队，非逻辑回归

---

## Ticket 1 — 项目级检索会话主题锚定

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

**What to build:** 项目级会话（未锁定论文）中，弱指代 / 审计 / 总结类查询（"回到第一张表"、"总结到目前为止确认的所有修改"、"最后审计整个对话"）不再命中无关论文（opls-aa / OPLS5 / 114110 等），而是锚定到会话实际讨论的主题论文。用户无需显式锁定论文，多轮对话的主题连贯性即能约束检索范围；审计/总结类长对话查询优先以会话历史为主题来源。

**Acceptance criteria:**
- [ ] 40 题回归项目级会话：R33/R34 答案内容回到 CHARMM36 主题（当前为 OPLS5 内容），R40 不再空答案（当前检索 6 个无关文档 + rag.answer 两轮 0 字符）
- [ ] R22 类弱指代查询不再命中 opls-aa 论文
- [ ] 锁定会话（document_id 非空）行为完全不变（回归保护）
- [ ] 首轮（无历史）项目级查询行为不变
- [ ] 本地 pytest 全量通过，无回归

## Ticket 2 — 回退锚定历史一致性加权

**Blocked by:** None — can start immediately

**Status:** completed

**What to build:** 项目级会话检索 0 items 触发历史回退时，锚定的文档改为历史中出现次数最多的主题文档（多数投票，可带轮次衰减），而不是"最近一次成功命中"——后者会被跨主题轮（用户临时换话题，如 R33/R34 命中 OPLS5）污染，导致回退锚错论文。回退后答案内容应回到会话主题论文。

**Acceptance criteria:**
- [x] R35/R36 回退后锚定 CHARMM36 主题论文（当前锚错 OPLS5），答案内容为 CHARMM36 而非 OPLS5——真实回放：6/6 回退点投票锚定 fdf5282c（R35/R36 时 OPLS5 仅 6-7 票 vs 主论文 14 票）
- [x] 单一主题会话回退行为与现有一致（R18/R20/R39 不回退错）——回放：R18 锚 fdf5282c（8 票）、R20 锚 fdf5282c（9 票）、R39 锚 fdf5282c（14 票，原 ca3b89b3 亦为 CHARMM36 系，内容仍正确）
- [x] 多主题历史（含换话题轮）时投票正确选主主题——`recent_loses_to_majority` 测试
- [x] 锁定会话不触发回退（既有行为保持）——`test_locked_session_does_not_fallback` 保持通过
- [x] 本地 pytest 全量通过，无回归——2069 passed（+5 新测试）

**实施记录（2026-08-11）**：`_last_hit_document_id` → `_majority_hit_document_id`（轮次投票：每轮每文档一票、轮内去重防单轮 citations 碾压、平票按最近轮出现的文档）；5 新测试（多数/最近被否决/平票/空历史/集成回退）；code review 双轴：Standards 无硬违规（采纳命名建议改名 + 调用点注释 T1/T2 更新），Spec 无缺失/越界（指出真实回放风险 → 用回归会话 DB citations 回放 6/6 锚定主论文验证）；部署 + 验收集 3/3 PASS（A2 项目级回退答案内容为 C36 表格比较）；用户指示跳过全量 40 题回归。

## Ticket 3 — synthesize 长延迟调查与超时保护

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

**What to build:** 单轮查询不再因合成步骤（synthesize）卡死 10 分钟而整体 timeout 丢答案（R22 638s / R23 399s，timeout_seconds=240）。先调查根因（本地模型排队 / Ollama 健康度 / 长引用列表输入），再做最小防护：超时熔断或降级路径，使长延迟可观测、可恢复，最终答案不因合成步骤挂起而丢失。

**Acceptance criteria:**
- [ ] 根因定位：确认 synthesize 638s/399s 是模型排队、生成慢还是输入过大，记录证据
- [ ] 合成步骤超过预算时有降级/熔断路径，最终响应不超时丢失（R22/R23 类查询可重跑完成）
- [ ] 正常轮次（30-170s）行为不变
- [ ] 延迟与熔断事件可观测（trace / 日志有记录）
- [ ] 本地 pytest 全量通过，无回归

---

## 实施顺序

三票无依赖、可并行。Ticket 1 与 Ticket 2 改动均在检索/回退路径，落地顺序由实施者决定；每票完成后本地 pytest + code review，最后服务器部署 + 40 题回归验收（复用 charmm36-chat-eval-regression 会话模式）。
