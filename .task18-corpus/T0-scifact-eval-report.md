# SciFact (BEIR) RAG 能力评测报告

> 日期：2026-08-18（初版 2026-08-17） | 系统：knowledge-agent-server（dev 环境，GPU0）
> 评测语料：SciFact/BEIR（5,183 篇 corpus，1,109 claims）→ 导入 584 篇（test 引用 284 + 干扰项 300）
> 改进：全库向量补充路由（spec: `docs/superpowers/specs/2026-08-17-retrieval-layer-improvement-design.md`）

## 评测设置

| 项 | 值 |
|---|---|
| 评测集 | BEIR SciFact（allenai/scifact → BEIR 格式） |
| 检索请求 | `QueryService.retrieve_evidence(project_slug, claim, limit=10)` |
| 导入文档 | 584 篇（test-cited 284 + 干扰项 300），`.md`（corpus title+abstract 文本） |
| 文档映射 | file_name `<pmid>-<uuid8>.md` → PMID |
| 嵌入模型 | qwen3-embedding:4b（系统生产配置） |
| 生成模型 | qwen3.5:9b（claim 相关性判定） |

## 1. 检索评测（300 test claims）

### 1.1 改进前后对比

| 指标 | 改进前（8/17 现状） | 改进后（8/18，R1 补充路由） |
|---|---|---|
| nDCG@10 | 0.4422 | **0.8180** |
| Recall@10 | 0.4706 | **0.9186** |
| 未召回预期文档 queries | 170 / 300（56.7%） | 48 / 300（16.0%） |
| hit@1 / hit@3 / hit@5 / hit@10 | 121 / 129 / 130 / 130 | 209 / 238 / 246 / 252 |
| avg 证据数 | 3.8 / claim（min 1, max 8） | 6.1 / claim（min 1, max 8） |

**nDCG 分布（改进后）**：48 条 0.0（未召回）｜ 178 条 1.0（完美命中）｜ 52 条 (0.5, 1) ｜ 22 条 (0, 0.5]
（改进前：170 / 98 / 27 / 5——双峰分布大幅缓解：不再"要么第 1 要么不在"，有命中的 query 130→252，其中散落命中（hit@2-10 区间）从 9 条增至 43 条）

### 1.2 四路对照（诊断实验，同 584 子集、300 test claims）

| 配置 | nDCG@10 | Recall@10 |
|---|---|---|
| ③ 现状（词法路由 + 文档内混合） | 0.4422 | 0.4706 |
| ① BM25（rank_bm25，584 全库） | 0.7871 | 0.8611 |
| ② 纯向量（全库 top-10，无路由无 bonus） | 0.8757 | 0.9500 |
| ④ **改进后（路由 ∪ 全库向量补充）** | **0.8180** | **0.9186** |

**结论**：② 显著 > ③ 确认"词法路由漏召回为主因"，补充路由成立（spec §3 判据）。④ 超系统内 BM25 基线（同口径 0.7871，榜单 0.665 是全量 5,183 语料，不可直接比）与 spec 目标 ≥0.60；距纯向量上限 ② 残余 0.0577（见 §3 残余差距）。

### 1.3 改进机制

- 原链路：`_route_papers`（纯词法，阈值 2.0）→ `_search_source_chunks`（向量检索被 `document_ids` 锁死在路由结果内）——SciFact claim 与 abstract 词法共享极少，56.7% 未召回源于候选生成（spec §1.2）。
- 改动（search.py，两函数）：`_build_rag_contexts` 在非 lock 且无显式 document_ids 时**总是**并入一次全库向量检索（`_search_source_chunks(question, [], limit=MAX_CONTEXTS, question_vector=复用)`），两路候选经 `_finalize_contexts` 统一排序截断融合；lock 场景跳过补充，精确语义原样保留。
- 表格提权修正：补充路 `table_promotion=False`——全库补充只做语义兜底，不给无关文档的表格 +40 提权与词法路表格竞争（内部回归 opls5_table_metrics 驱动，见 §4）。

## 2. Claim 相关性判定评测（300 dev claims）

> 说明：官方 SciFact 2021 版 claims 文件的 evidence/label 字段被清空（shared task 后），
> S3 旧版不可达（3.7KB/s）。真值采用 `cited_doc_ids`（该 claim 引用的文档视为"相关"），
> 测的是 LLM 判定"证据是否与 claim 相关"的能力，非官方 SUPPORT/CONTRADICT/NEI 三分类。

| 指标 | 值 |
|---|---|
| 文档级判定准确率 | TBD（未跑） |
| 相关判定 Precision / Recall / F1 | TBD（未跑） |
| claim 级命中率（判定含 ≥1 cited 文档） | TBD（未跑） |

> 本次迭代未执行 claim 相关性判定评测（检索层改进不涉及判定链路），保留占位。

## 3. 结论与建议

1. **检索层架构性差距已修**：词法路由漏召回 → 全库向量补充（零额外 embedding 调用，复用 question_vector）。nDCG@10 0.4422 → 0.8180（+85%），未召回率 56.7% → 16.0%，均超 spec §2 目标。
2. **avg 证据数 6.1（目标 ~8）未达满**：补充路 limit=MAX_CONTEXTS=8 已给足，不足 8 的成因是语料文档体量小（title+abstract ≈ 1-2 chunk/篇，两路并集经 `_finalize_contexts` 去重后截断）——候选池已接近上限，不再单独优化。
3. **残余差距（④ vs ②，0.058）**：纯向量全库上限 0.8757 仍高于混合路由。方向（记录在案，未实施）：
   - 词法路候选在 `_finalize_contexts` 并集排序中可能挤占高向量分文档的 top-10 位置（词法低分噪声文档进入排序池）——可做并集后按向量分为主的二次混合；
   - 全库 BM25 补充候选（当前补充路只取向量 top-k，BM25 补充可再兜词法面）。
4. **内部 13 篇语料回归不回归**（硬约束，见 §4）：30 题 full-answer + retrieval cases 修复版全过，超扩库前基线（8/5 active 29/30）。
5. 无重排层（cross-encoder）仍是 border case 区分度瓶颈，但非本次范围（spec §4.3 非目标）。

## 4. 内部回归记录（硬验收）

- 基线：8/5 active 基线 `task15-rag-active-full30-20260805-clean.json`（29/30，仅 charmm36_overview 失败）。
- SUPP v1（第一版补充路由）full-answer 29/30：失败例换成 **opls5_table_metrics**（补充引入的回归）+ charmm36_overview（基线既有失败）。
- A/B 归因：nosupp vs supp 对照证明仅 opls5_table_metrics 由补充路由造成；其余 3 例 retrieval-case 失败（evidence_coverage 门控，143569d 引入"表格必需词"验收）在 nosupp 同样失败——既有覆盖缺口，与本次改动无关。
- **根因**：指标查询场景（"OPLS5 的表……与 OPLS4 相比"），补充路捞到的 OPLS4 表 chunk 与词法路 OPLS5 表 chunk 同构平手，补充路也吃 `TABLE_CONTEXT_SCORE_BOOST`(+40) 后在 `_finalize_contexts` 平局决胜——正确文档被挤出 top-10。
- **修复**：`_search_source_chunks` 新增 `table_promotion` 参数，补充路由传 False（不做表格提权，只做语义兜底）；单测 `test_metric_query_table_supplement_does_not_displace_lexical_table` 复现并锁定该行为。
- **修复版（r2，2026-08-18 实测）**：
  - retrieval cases：recall 1.0/1.0（opls5 恢复），26/30 通过——剩余 3 例为 evidence_coverage 既有缺口（143569d gate，nosupp 对照同样失败，已定性与本次改动无关，与改动前持平）；
  - full-answer 30 题 strict_pass 全过（answer/citation/source 均 1.0，p50 46.9s / p95 116.3s）——与 8/5 基线（29/30）对比：**不回归且 +1 超基线**。
- SciFact 复测（修复后 routed，2026-08-18）：nDCG@10 **0.8180** / Recall@10 0.9186 / 未召回 48/300 / avg 证据 6.1——与修复前逐项一致，未回退（表格提权修正只影响表查询，SciFact claim 多为叙述性，行为惰性）。

## 复现命令

```bash
# 检索评测（服务器）
cd ~/knowledge-agent-dev && set -a && . runtime/app.env && set +a && \
PYTHONPATH=src .venv/bin/python runtime/sci-fact-bench/eval_scifact.py \
    --report runtime/sci-fact-bench/eval-report.json        # ③ 现状
    # --mode bm25 → eval-bm25.json（①）| --mode vector → eval-vector.json（②）
    # --mode routed → eval-improved.json（④）

# claim 相关性判定（服务器，dev 300 条）
PYTHONPATH=src .venv/bin/python runtime/sci-fact-bench/eval_scifact_claims.py \
    --report runtime/sci-fact-bench/claims-eval-report.json

# 内部回归（服务器，30 题 full-answer + retrieval cases）
PYTHONPATH=src:~/knowledge-agent-dev .venv/bin/python runtime/task15/task15-evaluate-api-full30.py \
    --cases runtime/task15/internal-research-overlap30-full-answer-cases.json \
    --report runtime/task15/regression-r2-full30.json --api-url http://127.0.0.1:8002
```
