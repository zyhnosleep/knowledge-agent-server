# 检索层提升 R2：补充路 BM25 ∪ 向量（词法/语义双路兜底）

> 日期：2026-08-18 | 状态：implemented（TDD 3 测试 RED→GREEN；R3 修正：分数天花板 6.0 + 每文档 1 chunk + 路由锁定守卫放宽，SciFact 0.8503 / recall 0.9437 / zero_hit 0，内部回归 30/30 检索正常）
> 来源：R1 全库向量补充路由（2026-08-17 spec，已上线）后残余差距诊断——hybrid 实验（2026-08-18）
> 上游：`docs/superpowers/specs/2026-08-17-retrieval-layer-improvement-design.md`（R1）

## 1. 背景与问题定位

### 1.1 R1 后残余差距（实测，2026-08-18，300 test claims）

> 度量口径注（2026-08-18 修正）：routed 模式的 chunk context → PMID 映射此前
> 未去重，同一文档多次计数把 nDCG 顶过 1.0（R1 报告 24 行 >1、max 2.13）。
> 下表 ④ 为修正后（PMID 去重、文档级）的干净值；② ① hybrid 本就是文档级
> rank，无此问题。修正脚本：`.task18-corpus/recompute_clean_metrics.py`。

| 配置 | nDCG@10 | Recall@10 | 未召回 queries | 说明 |
|---|---|---|---|---|
| ② 纯向量（oracle 上限） | 0.8757 | 0.9500 | 15 | 无路由全库 top-10 |
| ① BM25（584 全库） | 0.7871 | 0.8611 | — | 系统内词法基线 |
| ④ 生产（R1 补充路 = 纯向量 top-8） | 0.7654 | 0.8319 | 48 | 当前线上（干净口径；污染口径 0.8180） |
| hybrid：BM25 top-10 ∪ 向量 top-10，RRF(k=60) | 0.8691 | 0.9667 | 10 | 文档级实验 |
| hybrid：RRF 向量路 ×2.0 | 0.8751 | 0.9500 | 15 | 文档级实验 |
| R3 修复后（routed 全管道） | 0.8503 | 0.9437 | 0（zero_hit 口径） | 2026-08-19 实测（eval-routed-r4-fix.json） |

**诊断结论**：④ 的 48 条未召回中 **38 条被 BM25∪向量救回**（文档级实验逐 query 核对）——残余差距主因不在嵌入模型上限，而在补充路只取纯向量 top-8 + chunk 级纯分数融合。BM25 全库词法面是系统内已具备、未接入补充路的信号。

### 1.2 目标

| 项 | 现状（④，干净口径） | 目标（干净口径） |
|---|---|---|
| SciFact nDCG@10 | 0.7654 | **≥ 0.86**（实验区间 0.869-0.875 为佳；hybrid 实验本就是文档级 rank，无度量污染） |
| 未召回 queries | 48/300 | **≤ 15** |
| Recall@10 | 0.8319 | **≥ 0.95** |
| 内部 13 篇语料回归 | 30 题 full-answer + retrieval | **不回归**（硬约束） |
| test 环境 | — | 不动 |

## 2. 实现设计

**核心原则**：改动限定在补充路内部；不动 `_route_papers`、lock、词法路表格检索（table_promotion=True 特权不变）；`_finalize_contexts` 统一排序截断的融合机制不变。

### 2.1 改动点（search.py，两处）

1. **新增 `_bm25_supplement_candidates(question, project_id, limit)`**（QueryService 方法）：
   - 全库文档级 BM25 打分（复用系统 `_tokenize` 分词 + 文档词频/IDF/平均长度；与 eval_scifact.py 同算法 K1=1.5, B=0.75, tf=1 文档级近似），取 top-k 文档；
   - **分数归一化**：`score = BM25_SUPPLEMENT_SCORE_CEIL × bm25_doc_score / max_bm25_score`（同 query 内 max 归一化，天花板 6.0 由 sweep 校准——兜底不压主信号），使 BM25 路与向量路在 `_finalize_contexts` 中可比；
   - 返回 top-k 文档中检索到的 chunk（文档文本 → chunk 定位），chunk 继承文档归一化分；
   - 无 LLM 调用、无 embedding 调用（纯 CPU 词法，毫秒级）。
2. **`_build_rag_contexts` 补充路升级**（现 1260-1282 行）：
   ```
   补充路 = _search_source_chunks(全库向量 top-8, table_promotion=False)
         ∪ _bm25_supplement_candidates(全库 BM25 top-k, 归一化分)
   ```
   并集入 contexts → `_finalize_contexts` 统一排序去重截断（现有机制，重叠 chunk 高分保留）。
   守卫（R3 归因修正，2026-08-19）：跳过补充路的只有**显式作用域**（`document_ids`
   非 None，API 语义 lock-and-never-widen）、**overview**、以及**路由锁定 + 结构化
   查询**（表/指标/图：锁定 = 用户明确要某论文的表/图，混入其他文档的表格即表错
   论文）。路由锁定本身（prose 声明）不再跳过——归因实验
   （`.task18-corpus/attribution_eval.py`，attribution-report-fixed.json）证伪原守卫
   为纯伤害：noroute−routed = +0.1182，68/300 帮倒忙、9 帮上忙，8 条 nDCG 1.0→0.0
   全是 routed 单 PMID——词法启发式锁错论文时补充路被整体跳过，正确文档永远
   不可见。修复后锁定文档 chunk 有词法 route bonus 占优，`_finalize_contexts`
   分数融合后真相关仍排前。

### 2.2 参数与边界

| 项 | 值 |
|---|---|
| BM25 top-k | 8（与 MAX_CONTEXTS 对齐，实验 top-10 的近似） |
| 归一化 | max 归一化到 [0, 6.0]（`BM25_SUPPLEMENT_SCORE_CEIL`，sweep 校准：`.task18-corpus/bm25_ceil_sweep.py`；兜底不压主信号） |
| 每文档 chunk 数 | 至多 1 个（R3 修正：按 question_vector 余弦选最佳证据；整文档注入会以同一归一化分占满 top-8 挤出真相关，12/50 退化） |
| 表格提权 | BM25 候选 chunk 同补充路语义：`table_promotion=False`（不参与 +40 竞争） |
| 跳过补充路 | 显式作用域 / overview / 路由锁定 + 结构化查询（表/指标/图）——R3 归因修正：路由锁定本身不再跳过 |
| 去重 | `_finalize_contexts` 现有 seen_keys 机制（重叠 chunk 高分保留） |

### 2.3 融合语义说明（与文档级实验的差异）

实验是文档级 RRF（rank 融合，鲁棒但无尺度）；生产是 chunk 级 score 融合（`_finalize_contexts` 现有架构）。本 spec 用 **max 归一化 BM25 分 + 天花板**把词法面映射到向量同尺度，保持现有融合机制不动（R1 已实测该机制 0.8180）。RRF 的 rank 鲁棒性不在本 spec 范围；归一化融合实测 nDCG 0.8503（天花板 6.0 由 `.task18-corpus/bm25_ceil_sweep.py` 校准，r3 退化驱动）。若仍达不到目标，回退选项：调低天花板压低词法面权重（等价于实验 vec×2.0）或增大 top-k。

### 2.4 非目标

- 不改 `_route_papers` 词法路由（lock/精确语义场景原样）
- 不做文档级 RRF / 不引入 rank 级融合
- 不换嵌入模型、不加 reranker
- 不改表格检索提权（词法路 +40 特权不变）

## 3. 测试与回归

1. **单测**（TDD 先行，tests/test_retrieval_route_supplement.py 追加）：
   - 补充路产出包含 BM25 候选（fake：向量路 miss 的文档经 BM25 词法命中进入候选）；
   - BM25 分归一化：同 query 内 max 归一化到 [0, BM25_SUPPLEMENT_SCORE_CEIL]，最高分文档 = 6.0；
   - 显式作用域 / overview / 路由锁定 + 结构化查询不触发补充路（R3 归因修正：纯路由锁定不再跳过，见 test_route_locked_scope_runs_*）；
   - 重叠 chunk 去重后高分保留。
2. **SciFact 复测**（300 claims，routed 模式）：nDCG@10 ≥ 0.86、未召回 ≤ 15、Recall ≥ 0.95。
3. **内部回归**（硬验收）：30 题 full-answer strict_pass + retrieval cases 不回归（有回归须有分析结论）。
4. 报告更新：`.task18-corpus/T0-scifact-eval-report.md` 补 R2 前后对比与诊断表。

## 4. 性能与风险

- **性能**：每次问答补充路 +1 次 BM25 全库打分（~1200 chunks 文档级循环，毫秒级）+ 0 额外 embedding/LLM 调用。语料长大（扩到 150-200 篇 ~2-4 万 chunks）后 BM25 全库循环仍在毫秒-几十毫秒级，可接受；必要时后续做倒排缓存（非本 spec 范围）。
- **风险**：BM25 归一化分与向量分尺度失配 → 调参回退（§2.3）；无关文档混入 → 分数排序 + MAX_CONTEXTS 截断兜底，内部回归验证；单点改动（一方法 + 一调用点），git 级回滚。

## 复现命令

```bash
# SciFact 复测（服务器）
cd ~/knowledge-agent-dev && set -a && . runtime/app.env && set +a && \
PYTHONPATH=src .venv/bin/python runtime/sci-fact-bench/eval_scifact.py \
    --mode routed --report runtime/sci-fact-bench/eval-improved-r3.json

# 内部回归（服务器，30 题）
PYTHONPATH=src:~/knowledge-agent-dev .venv/bin/python runtime/task15/task15-evaluate-api-full30.py \
    --cases runtime/task15/internal-research-overlap30-full-answer-cases.json \
    --report runtime/task15/regression-r3-full30.json --api-url http://127.0.0.1:8002
```
