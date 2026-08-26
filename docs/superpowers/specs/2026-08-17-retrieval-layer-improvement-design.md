# 检索层能力提升：全库向量补充路由（向量兜底路由）

> 日期：2026-08-17 | 状态：approved（用户已确认方案方向与设计两节）
> 来源：SciFact (BEIR) 检索评测 nDCG@10 = 0.4422（低于 BM25 0.665 / bge 0.72），用户要求讨论并提升检索层能力

## 1. 背景与问题定位

### 1.1 评测结果（2026-08-17，584 篇子集语料，300 test claims）

| 指标 | 值 |
|---|---|
| nDCG@10 | 0.4422 |
| Recall@10 | 0.4706 |
| 未召回预期文档 | 170/300（56.7%） |
| hit@1 / hit@3 / hit@10 | 40.3% / 43.0% / 43.3%（双峰：要么排第 1，要么完全不在） |
| avg 证据数 | 3.8 / claim（min 1, max 8） |

### 1.2 链路分析（代码确认）

```
claim → _route_papers（纯词法路由）→ _build_rag_contexts → _search_source_chunks（向量+词法混合）→ 证据包
```

- `_route_papers`（search.py:899）是**纯词法路由**：query 词元与文档 profile/title/alias/key_terms 交集打分，阈值 `PAPER_ROUTE_MIN_SCORE = 2.0`（search.py:81）以下直接淘汰，lock 分支通常只留 1-5 篇。
- `_search_source_chunks`（search.py:2462）的向量检索**被 `document_ids` 过滤锁死在词法路由结果内**——路由未选中的文档，嵌入模型看不到。
- 全库向量兜底（search.py:1255-1256）**仅在 contexts 为空时触发**——路由几乎总返回非空，兜底很少生效。

**结论**：候选生成（文档选择）是纯词法的，向量检索只能"在已选中文档内排序"。SciFact claim 是密集科学声明，与 abstract 词法共享极少 → 词法阈值淘汰真相关文档 → 56.7% 未召回。架构性差距，非 embedding 模型之过。该词法路由是为内部 13 篇语料的精确锁定场景设计的（"XX 论文的表"），**不能砍掉，只能补充**。

## 2. 目标与验收标准

| 项 | 现状 | 目标 |
|---|---|---|
| SciFact nDCG@10 | 0.4422 | **≥ 0.60**；超过系统内 BM25 基线（诊断实验①定值）为佳 |
| 未召回率 | 56.7% | 显著下降 |
| avg 证据数 | 3.8 | ~8（候选池补齐到上限） |
| 内部 13 篇语料回归 | 30 题 full-answer strict_pass + retrieval cases | **不回归**（硬约束） |
| test 环境 | — | 不动（dev 开发，走 GitHub 部署） |

## 3. 诊断实验（实现前先跑，数据驱动）

同 584 子集、300 test claims，四路对照（脚本 `runtime/sci-fact-bench/eval_scifact.py` 扩展，服务器 dev 环境）：

| 配置 | 说明 |
|---|---|
| ① BM25（rank_bm25 标准实现，584 全库） | 系统内词法基线（榜单 0.665 是全量 5,183 语料，同子集对照口径才公平） |
| ② 纯向量全库 top-10（无路由、无词法 bonus） | 嵌入模型召回上限 |
| ③ 现状（路由 + 文档内混合） | 已知 0.4422 |
| ④ 改进后（路由 ∪ 全库向量补充） | 预期效果 |

判据：若 ② 显著 > ③，则确认"词法路由漏召回为主因"，向量补充方案成立；若 ② ≈ ③，则瓶颈在嵌入模型本身，补充收益有限，按实测数据如实报告。

## 4. 实现设计

**核心原则**：最小侵入——不动 `_route_papers` 返回结构与 lock 逻辑；融合零成本（`_finalize_contexts` 对非表查询纯按 `context.score` 统一排序截断，search.py:6958）；现有调用方默认行为不变。

### 4.1 改动点

1. **`_search_source_chunks` 参数化嵌入**（search.py:2438）：新增可选参数 `question_vector: list[float] | None = None`。传入则复用（不再 embed），不传则保持现状自行 embed（默认行为不变）。
2. **`_build_rag_contexts` 加"全库向量补充"**（search.py:1244 后）：当 `locked_document_ids` 为空且 `document_ids is None` 时，**总是**执行一次
   `_search_source_chunks(question, project_id, [], limit=MAX_CONTEXTS, question_vector=<复用>)`，
   并入 `contexts` → 现有 `_finalize_contexts` 统一排序截断（天然融合；两路候选的 chunk 重叠由 `_finalize_contexts` 现有 seen_keys 去重处理，search.py:7019-7025）。
   原 1255-1256 行"仅 contexts 为空时兜底"被此补充取代（同模式，逻辑统一为"总是"）。
3. **补充路表格提权关闭**（2026-08-18 回归驱动修订，`_search_source_chunks` 新增参数 `table_promotion: bool = True`）：补充路由调用传 `table_promotion=False`——全库补充只做语义兜底，不给无关文档的表格 +40 `TABLE_CONTEXT_SCORE_BOOST` 与词法路表格平手竞争（内部回归 opls5_table_metrics 实测：平手时靠原始向量分把正确文档挤出 top-10）。flag 同时关闭 `_rank_blocks` 锚点 bonus（+6/+8 权重过大，会以更小尺度复刻同样挤占）。词法路表格检索默认行为不变。

### 4.2 边界条件（行为决策）

| 场景 | 行为 |
|---|---|
| lock（用户明确问某篇论文的表/数据） | **跳过补充**，精确语义原样保留 |
| 显式 document_id 作用域（单文档问答） | **跳过补充** |
| SciFact claim / 开放问题 / cross-paper | **触发补充**（已确认 SciFact claim 不命中任何 lock 分支） |

### 4.3 非目标（本次不做）

- 不换嵌入模型（qwen3-embedding:4b 保持生产配置）
- 不加 cross-encoder rerank
- 不做查询改写（HyDE）
- 不做全量循环打分优化（pgvector 直接 top-k 产出）——当前语料规模（~1200 chunks）全循环可接受，语料长大后再优化
- 不改 `_route_papers` 的分数公式与 lock 逻辑

## 5. 测试与回归

1. **单测**（先写）：`_build_rag_contexts` 在 lock 场景不触发补充、非 lock 触发、补充结果并入排序、`_search_source_chunks` 传入 question_vector 时不重复 embed。
2. **诊断实验**（服务器 dev）：四路对照，验证 ② > ③ 假设。
3. **评测复测**：`eval_scifact.py` 300 claims 复跑，记录改进后 nDCG@10 / Recall@10 / 未召回率 / avg 证据数。
4. **内部回归**（硬验收）：30 题 full-answer strict_pass + retrieval cases 在 dev 重跑，与扩库前基线对比不回归（有回归须有分析结论）。
5. 报告更新：`.task18-corpus/T0-scifact-eval-report.md` 补改进前后对比；部署走 GitHub。

## 6. 性能与风险

- **性能**：每次问答 +1 次 pgvector 全库检索（584 篇 ≈ 1200 chunks，索引毫秒级）+ 0 次额外 embedding 调用（复用同一向量）。全量循环打分约几百 ms，可接受。
- **风险**：无关文档混入证据包 → 分数排序 + MAX_CONTEXTS 截断兜底，内部回归验证；单点改动（两函数），git 级回滚；`_search_source_chunks` 参数化默认行为不变，现有调用方零影响。

## 复现命令（服务器 dev）

```bash
cd ~/knowledge-agent-dev && set -a && . runtime/app.env && set +a && \
PYTHONPATH=src .venv/bin/python runtime/sci-fact-bench/eval_scifact.py \
    --report runtime/sci-fact-bench/eval-report.json
```
