# Answer 链路 facts-first 最终门禁：opls5_table_metrics 根因与修复

> 关联 spec：`docs/superpowers/specs/2026-08-04-task15-rag-regression-fix.md`
> 关联 tickets：`docs/superpowers/tickets/2026-08-04-task15-tickets.md`（Ticket 6 的深化）
> 日期：2026-08-05

---

## 1. 问题陈述

candidate full30 评测结果：

| 指标 | 值 |
|---|---|
| 检索 Recall@5/10 | 1.0 |
| citation / source match | 1.0 |
| 30 题答案通过 | **29/30** |
| 唯一失败 | `opls5_table_metrics` |
| 缺失 | Table 7、1.18、1.12 |
| answer_latency_ms | 21850（LLM 被调用） |

检索阶段已经拿到 Table 7 和全部 facts，但最终答案缺少 Table 7 尾部数值（1.18、1.12）。**问题不在检索，在"答案组织链路"。**

## 2. 已排除的根因

- ❌ χ1/χ2 LaTeX 归一化（已修，Recall 已 1.0）
- ❌ BM25 或排序（检索已拿到全部 facts）
- ❌ Table 7 没召回（retrieval-only 通过）
- ❌ candidate/active 路由错误（版本可见性已修）
- ❌ 数据或 facts 不存在（facts 在 canonical 里）

## 3. 根因：answer 与 retrieval 解耦，供给侧截断事实

### 3.1 根因链路

```
retrieval 阶段：EvidencePack 拿到 Table 7 + 全部 facts（Recall 1.0）
    ↓
answer() 重新 _build_rag_contexts → _finalize_contexts → _fit_contexts_to_token_budget
    ↓ 多表时 token budget 截掉表尾的 Table 7
    ↓
_deterministic_table_answer_if_supported() 拿不到完整 table → 返回 None
    ↓
降级 _draft_answer（LLM，latency 21.8s）→ LLM 漏表尾数值
    ↓
_repair_missing_table_answer() 校验依赖 _extract_requested_metric_values()
    ↓ 该函数也从截断的 contexts 取 metrics → 认为"够了"
    ↓
内部通过，验收发现 Table 7 尾部缺失
```

### 3.2 深层问题：请求清单来自供给侧，而非需求侧

关键代码 `_deterministic_table_answer_if_supported()`（[search.py:3402](src/app/services/search.py#L3402)）：

```python
metrics = self._extract_requested_metric_values(question, contexts, table_indexes)
if metrics and self._answer_lacks_requested_metrics(question, answer.answer_markdown, metrics):
    return None
```

`_extract_requested_metric_values()` 从 **contexts**（供给侧）提取 metrics。当 Table 7 在 contexts 里被 token budget 截掉后：

- `metrics` 清单里**没有 1.18 / 1.12**（因为供给被截断）
- `_answer_lacks_requested_metrics()` 校验自然认为"够了"
- 最终答案缺 Table 7 尾部，但内部校验通过

**"内部校验过、验收失败"的本质：校验标准（从截断的 contexts 反推）和验收标准（从 question 期望的 facts）用的是两个不同的 facts 来源。**

### 3.3 调用链上的三个截断/降级点

| 环节 | 代码 | 行为 |
|---|---|---|
| token budget 截断 | `_fit_contexts_to_token_budget` | 多表时按全局预算截断，靠后的表被牺牲 |
| 确定性路径降级 | `_deterministic_table_answer_if_supported` 返回 `None`（search.py:3388） | 表格不完整 → 放弃确定性 → 降级 LLM |
| 校验盲区 | `_extract_requested_metric_values`（search.py:4068）从 contexts 反推 | 供给被截断 → 校验跟着瞎 |

## 4. 修复设计（三层）

### 4.1 第 1 层：请求清单需求侧化

**从 question 构建 expected facts 清单，不从 contexts 反推。**

- 从 question 解析请求的 `table_id + row + value` 集合：
  - 显式表号："Table 7" → `{table_id: "table-7"}`
  - 指标 → 表："OPLS5 的 1.18/1.12" → 映射到对应表和行
- 清单独立于 contexts：即使 Table 7 被截断，清单里仍有它
- 用这个清单做**最终门禁**：答案缺任何一个请求事实 → 触发确定性重组装，不是降级 LLM

**实现要点**：
- 新增 `_expected_table_facts_from_question(question)`：返回 `{table_id: [row_value...]}` 期望集
- 它只依赖 question + canonical typed inventory，不依赖当前 contexts

### 4.2 第 2 层：多表 contexts 分组保底，不截断请求表

- `_fit_contexts_to_token_budget` 对**表格 contexts** 按 `table_id` 分组保底：
  - 每个请求的 table 至少保留一组完整行
  - 剩余预算再分配给普通文本
- 或：answer 的表格 evidence 直接从 canonical facts 组装，不经过字符截断路径

**呼应**：这与表格组覆盖修复（Ticket 4.3）共用同一个"按 table_id 分组"逻辑，只是作用点在 answer 侧的 contexts 构造。

### 4.3 第 3 层：facts-first 最终门禁

当 canonical facts 覆盖请求的表格组时，**最终答案必须由确定性 facts 组装**：

- `_deterministic_table_answer_if_supported()` 的判定条件改为基于**期望清单**，而非"contexts 里有什么"：
  - 期望清单中的每个 `table_id + row + value` 在 canonical facts 中可找到 → **必须走确定性组装**
  - 只有清单中的事实确实不在 canonical facts 里，才允许降级 LLM（且 LLM 要声明缺失）
- **LLM 只负责说明文字**，不能决定是否输出事实：
  - LLM 输出后，用期望清单做最终校验
  - `_answer_lacks_requested_metrics()` 改为对比"期望清单 vs 答案"，而非"contexts 反推的 metrics vs 答案"
  - 缺失 → 确定性重组装；`_unsupported_answer_numbers` 捕获新数字 → 打回

### 4.4 门禁的判定顺序（修复后）

```
_expected_table_facts_from_question(question)
    ↓
期望清单 = {table_id: [values]}
    ↓
canonical facts 覆盖期望清单？
    是 → 确定性组装答案 → 期望清单校验 → 通过则返回
        ↓ 校验失败（缺 value）→ 补组装/报缺失，不降级 LLM
    否 → 允许 LLM 生成，但必须声明哪些期望事实在证据中缺失
```

## 5. 测试设计

### 5.1 现有测试盲区

现有单元测试模拟**全量 contexts** 传入，未覆盖真实路径：

```
_build_rag_contexts → _finalize_contexts → _fit_contexts_to_token_budget → answer
```

即：测试验证了确定性函数，但**没有验证 token budget 截断多表后的真实 answer 行为**。

### 5.2 新增集成测试

- **多表 + token budget 截断表尾**：构造 contexts 含 Table 2 + Table 7，token budget 只够 Table 2 → 断言最终答案仍包含 Table 7 的期望 value（走确定性重组装）。
- **期望清单需求侧**：question 请求 Table 7 的 1.18/1.12，但 contexts 不含 Table 7 → 断言门禁捕获缺失（而不是内部校验通过）。
- **LLM 角色限定**：fake Ollama 只返回说明文字、不含 1.18 → 断言期望清单校验失败并触发确定性重组装；fake Ollama 编造 999.0 → 断言被 `_unsupported_answer_numbers` 打回。
- **`failed_gates` 命名**：改为区分 `retrieval_gate` 与 `answer_gate`，避免 retrieval 通过但 full-answer 失败时 gate 命名误导。

## 6. 验收标准

- `opls5_table_metrics`：答案包含 Table 7、1.18、1.12，且 citation 指向 Table 7
- 30 题 full-answer 通过 **30/30**
- 新增集成测试全绿；既有 248 项回归不退化
- 确定性路径在 facts 覆盖时**不再降级 LLM**（验证 answer_latency 显著下降，不再是 21.8s 级）

## 7. 范围与约束

- **不动**：检索逻辑（Recall 已 1.0）、LaTeX 归一化、版本路由
- **只改**：answer 侧 contexts 构造 + 期望清单门禁 + 校验标准
- **不引入**：外部框架
- **前置**：表格组覆盖（Ticket 4.3）与 typed inventory（Ticket 2）提供按表分组的枚举基础
