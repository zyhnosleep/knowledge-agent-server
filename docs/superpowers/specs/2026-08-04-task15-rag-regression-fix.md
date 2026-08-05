# Task 15 RAG 回归修复：表格证据可见性与科学术语覆盖

## Problem Statement

开发版 RAG（GPU0，canonical-v4）在同题 10 题对比中仅 **4/10 严格通过（40%）**，测试版旧基线（GPU1）达 **9/10（90%）**。开发版缺失的关键事实包括精确数值（`21.0`、`95.3`、`12.0`、`11.53`、`25.00`、`13.07`、`RS peptide`、`80`、`1.8`、`1.1`、`0.5`）和科学术语（`FRET`、`GAlib`、`fast-folding`）。citation pass rate 和 source match rate 均为 100%，只能说明引用来源大体正确，不能证明 required facts 已进入 top-k evidence，也不能证明 active/candidate 版本和 table inventory 一致。因此本修复同时覆盖**版本可见性、证据召回/截断、表格结构化和答案事实保真**，不能把问题简单归因于答案组织阶段。

6 道失败题分布在 4 篇文档（charmm36、charmm36idpsff、charmm36m、ff14sb），其中 3 道是表格题（数值缺失），3 道是叙述题（术语缺失），各有不同根因。

## Solution

在不改总体架构、不删测试数据、不单独激活单篇文档的前提下，先执行一个不改变 active pointer 的版本可见性前置检查，再分两轮修复：

**前置检查（P0-A）**：为 candidate 生成并校验 `parse_version_map`，记录 `active_parse_version`、`candidate_parse_version`、typed inventory 和实际评测路由。所有 staged 评测必须显式读取 candidate；这一步不激活、不修改测试环境。

**第一轮（P1-B + P0-E）**：统一 evidence token budget，移除会截断 required anchor 的固定字符窗口。完整 canonical table 由代码扫描，但只将问题相关的完整行组/TableFact 送入回答链；不默认把 300+ 行整表送入 LLM。目标：恢复叙述题和表格题的 required-anchor 可见性。

**第二轮（P0-C + P0-D + P0-B + P1-A）**：重写表格表头/行分类器，支持多级列头和行标签合并；补逐表 typed inventory gate；仅在 deterministic facts 已提取但表达仍缺失时启用受校验的 LLM repair。目标：恢复 3 道表格题的精确数值/标签覆盖。

验收通过且用户确认后，才执行 17 篇文档的原子批量激活；P0-A 前置检查只负责 candidate 可见性和 inventory 审计，不代表已经激活。

## User Stories

### 第一轮：证据窗口修复

1. As a 科研人员, I want 所有科学术语（FRET、GAlib、fast-folding 等）在 evidence 中不被截断, so that 概述类问题的答案包含完整领域术语。

2. As a 科研人员, I want 代码扫描完整 canonical table，并把问题相关的完整表头、行组和 TableFact 传给回答链, so that 目标值位于长表格后部时仍可被确定性提取，而不需要把 300+ 行整表塞进 LLM 上下文。

3. As a 开发人员, I want 所有证据构造路径使用统一的 token budget 而非散落各处的固定字符常量, so that 未来添加新证据类型时不会引入新的截断点。

4. As a 开发人员, I want citation excerpt 在换行处结束且不返回半行, so that 引用片段始终在完整行边界。

### 第二轮：表格解析修复

5. As a 科研人员, I want 多级列头（如 Delta H vap calcd / Delta H vap exptl）被正确识别并生成稳定列名, so that 精确指标值能被确定性路径提取。

6. As a 科研人员, I want 语义表头行（如 Group / Edgewise / Pairwise）不再被当作数据行, so that extract_table_facts 不会拿到错误的 row/column 映射。

7. As a 科研人员, I want 多级行标签（如 RS peptide / C36）按层次合并, so that 行标识在提取事实时保持完整。

8. As a 运维人员, I want activation gate 校验 canonical manifest 的 typed inventory 与 DB inventory 完全一致, so that 缺 table Child 的版本永远不会被激活。

9. As a 科研人员, I want LLM repair 在注入 canonical facts 清单后精确复制数字和单位, so that 修复路径不会虚构证据中不存在的数值。

10. As a 运维人员, I want 候选版本在整批严格通过后执行原子批量激活, so that 不会出现部分文档切换到不完整版本的情况。

11. As a 开发人员, I want staged 评测报告明确记录 active/candidate 版本和 parse_version_map, so that 评测不会误读旧 active 版本而把版本可见性问题误判成检索失败。

### 测试与验收

12. As a 开发人员, I want 新增失败用例（candidate 路由、required-anchor coverage、语义表头、多级表头、长表格尾部事实、inventory 不一致、repair 数字校验）先于修复提交并以失败状态运行, so that 修复后看到它们由红变绿。

13. As a 开发人员, I want 同题 10 题 strict pass 从 4/10 提升到 ≥9/10, so that 开发版 RAG 质量不低于测试版旧基线。

14. As a 开发人员, I want Recall@5/10、required-anchor coverage、missing terms、citation/source pass rate、latency 在修复后重新测量, so that 每个改动的影响可量化。

## Implementation Decisions

### 架构约束

- **不改架构**：本轮不改用外部框架（LlamaIndex 等），在现有代码路径上做局部修复。
- **不动测试环境**：测试版（GPU1）全程只读，不重建数据、不切换 active pointer、不重启。
- **不单独激活**：candidate version 激活是 17 篇文档的批量原子操作，只有整批验收全部通过才执行。
- **candidate 显式路由**：retrieval-only 和 RAG full-answer 验收必须传入与报告一致的 `parse_version_map`；默认 active 路由不能用于候选版本验收。
- **表格不依赖 LLM 完整性**：canonical table 必须由代码扫描全部行；LLM 只接收确定性提取的目标事实和有界行组。

### Implementation Surfaces

- `src/app/services/search.py`：candidate/active 版本选择、required-anchor evidence、科学证据 token budget、表格 facts-first 回答和 bounded citation excerpt。
- `src/app/services/table_evidence.py`：完整 table rows/facts 装配、语义表头和多级表头处理。
- `src/app/services/table_extraction.py`：稳定列名、复合 row label 和 typed table inventory 所需的行列映射。
- `src/app/services/pipeline.py`：activation 前的逐版本 typed inventory gate。
- `scripts/evaluate_canonical_retrieval.py`：读取 candidate `parse_version_map`，输出 active/candidate identity、required-anchor Recall@5/10 和 retrieval/full-answer 分层报告。
- `scripts/rebuild_canonical_index.py`：批量激活前复验 inventory、strict acceptance 和版本映射，保持原子回滚。
- `tests/test_query_service.py`、`tests/test_table_evidence.py`、`tests/test_canonical_retrieval.py`、`tests/test_canonical_acceptance.py`、`tests/test_rebuild_canonical_index.py`：覆盖上述外部行为；不得修改测试环境数据。

### 第一轮改动（P1-B + P0-E）

- **科学证据窗口**：从 `_deterministic_scientific_evidence_answer_if_supported()` 等专用检索函数中移除会静默丢弃 required anchor 的固定字符窗口常量（520、900、1000、1400），改为以 required-anchor 集合 + 统一 token budget 选择完整句子/Child。若 anchor 不在 top-k evidence，先报告 retrieval miss；若已在 evidence 但预算不足，返回 `evidence-insufficient`，不能静默丢弃术语。

- **表格 citation excerpt**：`_table_citation_excerpt()` 和 `_table_block_excerpt()` 改为采用完整行边界结束，优先保留问题命中的完整行、表头和 caption。长度上限只用于 citation 展示；canonical 全表扫描不能依赖该 excerpt，excerpt 不得在半行处结束。

- **表格 answer 路径**：`_generic_table_value_rows()` 在有结构化 `table_context` 时禁止走 2400 字符 excerpt 截断，改为先扫描完整 `TableContext.rows/TableFact`，再按问题要求选择列、目标行和必要的父级行标签；round-robin 只用于多个目标表之间的 bounded evidence 选择。明确的 full-table 请求才允许返回完整 Markdown，并仍受 token budget 控制。

- **LaTeX 锚点归一化（残留 1：χ1）**：新增共享的 LaTeX 命令归一化函数，剥掉 `\mathbb{}`/`\mathrm{}`/`\text{}`/`\mathbf{}` 等命令包装层，用于 `_scientific_anchor_labels_in_text` 的匹配路径。**不改动 evidence 原文**（evidence 原样给 LLM，只在匹配时归一化）。检索侧与验收/评估侧必须共用同一归一化函数，避免一侧归一化、另一侧仍报缺失的漂移。不只修 chi1 单个正则——所有 LaTeX 锚点（alpha、chi2 等）都受同一问题影响，逐个补是打地鼠。

- **表格组覆盖（残留 2：opls5 缺 Table 2/7）**：表格/metric 查询**按 table_id 分组覆盖**，而非简单加大 limit。从 question 识别请求的表 → 用 canonical manifest 的 typed_inventory 枚举全部候选表 → 每个请求表至少取一组完整行 → 再由 token budget 控总量。全局排序后取前 N 个 Child 会天然牺牲排序靠后的请求表，分组覆盖才能保证多表查询不漏表。

### 第二轮改动（P0-C + P0-D + P0-B + P1-A）

- **header-row classifier 重写**：判定顺序为 separator → exact repeated header → semantic header-like row → data row。semantic header-like 判断需考虑非数值比例、与上一层表头的列级对应、`calcd/exptl` 等列标签语义。

- **多级列头合并**：对每列维护最近非空父表头，用列级向前填充生成稳定列名（如 `Delta H vap calcd`），同名列使用确定性后缀而非覆盖。

- **多级行标签保留**：非数值层级按出现顺序合并为复合 row_label（如 `RS peptide / C36`），不把数值单元格当 label。

- **typed inventory gate**：在 `_run_activation_gate_stage()` 中增加版本作用域的逐表等式校验——比较 canonical manifest 与 DB index inventory 的 table ID 集合、每表 Child ID/child_count、source_block_ids、row_count、parent-child 归属、重复/多余项，并校验 figure/formula ID 集合。所有比较必须绑定同一 `document_id + parse_version`；任意缺失、重复、跨版本混入或多余项都阻止激活。

- **LLM repair 加固**：`_repair_missing_table_answer()` 在 prompt 中注入 canonical facts 清单（row_label、column、value、table_id），要求精确复制数字和单位。repair 返回后执行完整校验链（`_answer_lacks_requested_metrics()` → `_unsupported_answer_numbers()` → citation index 支持校验），任一失败回退到 deterministic fallback。

### Agent 部分（Ticket 9，前置依赖 RAG 30/30）

- **定位**：Agent 从"所有查询的包装器"收窄为"会话网关 + 受约束编排"。纯 RAG 29/30 证明答案架构是对的，Agent 的 synthesize 无差别重写是 5/30 的破坏源。

- **路由决策矩阵**：
  - 单轮（simple_rag / evidence_required / table_or_metric）→ 跳过 synthesize，直接透传 rag.answer。
  - 对比（multi_source_compare）→ 透传（QueryService 内部已处理多文档对比，`_is_cross_paper_query` / `_route_papers` 多 match / `_draft_answer` 多文档生成）。
  - 跨轮引用（query 含"之前/刚才/对比/那个/它"指代）→ 启用 synthesize，但以受约束形态。
  - needs_clarification → 保持现状。

- **synthesize 改造为受约束组织器**（三要素）：
  1. 输入收窄：记忆事实 + 当前答案，不是证据全集；
  2. facts 清单注入：从 citations 提取 实体→数值→表号，要求精确复制数字、单位、表号；不允许生成证据里没有的数值；
  3. 输出校验：`_unsupported_answer_numbers()` + `_answer_lacks_requested_metrics()`，证据没有的数字一律打回，失败回退到 rag.answer 原答案。

- **记忆**：追问指代走 `_contextualize_retrieval_query` 扩展（拼最近 N 轮 query）；跨轮事实引用走 conversation_summary + citations 注入 synthesize。单轮不使用 conversation_summary。

- **硬性前提**：synthesize 是组织器不是事实来源——跨轮引用精度依赖 RAG 层事实质量。Agent 部分不得在 RAG 未达标（<30/30）前单独验收。

### 中期路线（Task 16+，本轮不改）

- 评估 LlamaIndex 的 `QueryFusionRetriever`、`SubQuestionQueryEngine`、`FaithfulnessEvaluator` 作为现有组件（paper routing、table assembly、facts extraction）的补充而非替代。

### Schema Changes

- manifest 层：在 `manifest.json` 的 `document` 段扩展 `typed_inventory`，包含 `table_ids`、每个 table 的 `child_ids`、`child_count`、`source_block_ids`、`row_count`，以及 `figure_ids`、`formula_ids` 集合。不涉及数据库 schema 迁移。

## Testing Decisions

### 测试原则

- 只测试外部行为（输入 → 输出），不绑定实现细节（如特定正则、中间变量名）。
- 使用真实 CHARMM36/CHARMM36m 论文 fixture 而非人工构造数据（已有 13 篇文档作为 fixture 来源）。
- 失败测试先提交为红，修复后变绿，不可接受"修复前就通过"的测试。

### 新增测试

- **语义表头识别**：构造 Table 7 的最小复现 fixture，断言 `Group / Edgewise / Pairwise` 不被当作首条数据记录。
- **多级表头展开**：构造 Delta H vap calcd/exptl 最小复现，断言两级列名正确合并。
- **candidate 版本路由**：构造 active 版本只有 narrative、candidate 版本包含 25 个 table Child 的 fixture；断言默认 active 查询不读取 candidate，传入 `parse_version_map` 后能读取 candidate，且报告记录两套版本。
- **required-anchor retrieval**：构造目标术语/数字位于表格后部或相邻 Child 的 fixture，断言 retrieval-only 报告分别记录 Recall@5、Recall@10 和缺失 anchor，不得用空答案或错误版本计为通过。
- **长表格事实边界**：构造 300 行表格且目标值位于最后一行的 fixture，断言 canonical facts 能提取最后一行；LLM 输入只包含选中的完整行组，不要求把整张表放进 prompt；citation excerpt 不以 `| M` 等半行结束。
- **inventory 不一致阻止激活**：分别构造缺 table、少一个 Child、重复 Child、跨 parse_version 混入和多余 Child 的 context，断言每种情况都阻止激活。
- **LLM repair 二次校验**：用 fake Ollama 返回 `999.0`（证据只有 `21.0`），断言最终 answer 不含 `999.0`。

### 回归测试

- 运行专项表格/检索/ingestion 测试和全量项目测试，确保既有回归不退化。
- 使用 candidate `parse_version_map` 重新运行 10 题 RAG 对比评测（retrieval-only → full-answer），记录 active/candidate 版本、required-anchor Recall@5/10、strict pass、answer pass、citation/source、missing terms、p50/p95。

### 验收门槛

- 同题 10 题 strict pass ≥ 9/10
- table/metric 题 required numeric/label facts 覆盖率 100%
- 10 题 retrieval-only 的 required-anchor coverage 必须逐题报告；目标事实不得因 active/candidate 路由错误、空答案或 citation 片段截断而被计为通过。
- citation pass rate 和 source match rate 均为 100%
- Recall@5/10 对每个 required fact 有明确报告
- 0 timeout、0 HTTP error；p95 ≤ 30s
- activation inventory 100% 一致后才允许切 active

## Out of Scope

- 不更换检索/Agent 框架（LlamaIndex、LangChain 等）
- 不修改生成模型（`qwen3.5:9b`）
- 不重建测试环境数据
- 不单独激活单篇文档
- 不调整父子分块策略（`semantic_chunking.py`）
- 不把整张长表默认注入 LLM：完整表格只在 canonical/facts 层扫描；回答链使用问题相关的 bounded facts/row groups。
- 不启动 Agent 评测（另立任务）
- 不删除旧 parse/chunk/vector/artifact
- 不修改 GPU 分配（dev=0, test=1）

## Further Notes

- 候选版本当前停在 `ready_to_activate`，原因是 `include_activation=False` + 整批验收未通过，不是 activation 阶段执行失败。
- 6 道失败题分布在 4 篇文档（charmm36、charmm36idpsff、charmm36m、ff14sb），3 道表格 + 3 道叙述，根因不重叠。
- 当前 candidate 验收必须显式绑定 `parse_version_map`；在 candidate inventory 和 required-anchor retrieval 通过前，不得把旧 active 的结果当作 candidate 结果。
- 开发链路：本地 → push GitHub → pull 服务器 dev → restart systemd service。
- 本 spec 对应的分析报告位于 `docs/superpowers/plans/2026-08-04-task15-rag-bug-analysis-and-upgrade-plan.md`。
