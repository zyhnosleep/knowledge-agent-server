# Task 15 RAG 回归根因与升级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复开发版 RAG 在同题对比中低于测试版的证据可见性、表格结构化和确定性回答回退问题，并在不改测试环境、不删除旧数据的前提下恢复可验收的检索质量。

**Architecture:** 先保证 canonical 解析清单、Child 库存和 active pointer 一致，再让完整 canonical table facts 进入检索和回答链。普通文本可以继续使用 Child + token budget；表格和精确指标必须走 facts-first 的确定性路径，LLM 只负责组织已经通过代码提取和覆盖检查的事实。

**Tech Stack:** Python 3、FastAPI、SQLAlchemy/PostgreSQL、pgvector、MinerU canonical artifacts、Ollama `qwen3.5:9b`、`qwen3-embedding:4b`、pytest。

---

## 1. 当前结论与边界

本计划基于开发/测试同题 10 题 RAG 对比、远程只读版本审计、代码调用链审计和本地回归测试。当前没有修改应用代码、没有改测试环境、没有切换 active pointer，也没有删除旧数据。

开发环境固定使用 GPU0，测试环境固定使用 GPU1；两边生成模型都是 `qwen3.5:9b`，embedding 模型都是 `qwen3-embedding:4b`。因此当前回归不能优先归因于“开发版模型更小”。

| 指标 | 开发 RAG / GPU0 | 测试 RAG / GPU1 |
|---|---:|---:|
| 严格通过 | 4/10 | 9/10 |
| answer pass rate | 40% | 90% |
| citation pass rate | 100% | 100% |
| source match rate | 100% | 100% |
| p50 | 19.5 s | 4.3 s |
| p95 | 162.4 s | 11.2 s |

报告文件：

- `D:\LLM_wiki\runtime\compare10-dev-rag-20260804.json`
- `D:\LLM_wiki\runtime\compare10-test-rag-20260804.json`
- 题集：`D:\LLM_wiki\runtime\internal-research-overlap30-full-answer-cases.json`

开发版缺失的关键事实包括 `21.0`、`95.3`、`12.0`、`11.53`、`25.00`、`13.07`、`RS peptide`、`80`、`1.8`、`1.1`、`0.5`、`FRET`、`GAlib` 和 `fast-folding`。这些题的 citation/source 大多仍然正确，说明主要断点在“完整证据进入回答”和“事实保真组织”，不是来源路由完全失效。

## 2. 当前 RAG 全流程与断点位置

```mermaid
flowchart LR
    Q[问题] --> R[QueryService._build_rag_contexts]
    R --> V[active_parse_version 过滤]
    V --> H[向量/词法 Child 检索]
    H --> E[表格 Child 同表扩展]
    E --> A[assemble_table_context]
    A --> F[extract_table_facts]
    F --> D[确定性 table/metric answer]
    D -->|事实缺失或校验失败| L[Ollama draft/repair]
    L --> C[引用与数字校验]
    D --> C
```

关键实现位置：

- active 版本过滤：`src/app/services/search.py` 的 `_active_child_chunk_condition()`、`_selected_child_chunk_conditions()`。
- 表格同组扩展和完整表组装：`src/app/services/search.py` 的 `_attach_complete_table_evidence()`；`src/app/services/table_evidence.py` 的 `assemble_table_context()`、`extract_table_facts()`。
- 确定性优先：`src/app/services/search.py` 的 `_answer_rag_first()`、`_deterministic_table_answer_if_supported()`、`_extract_requested_metric_values()`。
- LLM 兜底：`_draft_answer()`、`_repair_unsupported_numeric_answer()`、`_repair_missing_table_answer()`。
- 激活门禁：`src/app/services/pipeline.py` 的 `_run_activation_gate_stage()`；阶段状态和 pointer 切换在 `src/app/services/ingestion_stages.py`、`src/app/services/parse_versions.py`。

## 3. 已确认根因

### P0-A：active canonical 版本不完整，完整 table Child 不可见

远程只读审计发现，`charmm36idpsff` 当前 active 是 `canonical-v4-3d766a2ecd4a`，该 active 版本的命中主要是 narrative Child；另一个 shadow/`ready_to_activate` 版本包含约 25 个 table Child。检索按 active 版本严格过滤后，完整表格即使已经生成，也不会进入普通 RAG。

原因有两层：

1. `scripts/rebuild_canonical_index.py` 的重建路径默认 `include_activation=False`；
2. `scripts/evaluate_canonical_retrieval.py` 只有传入 `--activate-on-success` 且严格验收通过时才激活。因此 staged 版本停在 `ready_to_activate` 本身是当前策略允许的，但在开发版对比中造成了“代码已经重建、检索仍读旧 active”的错觉。

代码位置：

- `src/app/services/search.py:_active_child_chunk_condition()`：canonical 只允许 `DocumentChunk.parse_version == Document.active_parse_version`。
- `src/app/services/search.py:_selected_child_chunk_conditions()`：只有显式 `parse_version_map` 才会读取 shadow 版本。
- `src/app/services/ingestion_stages.py:run_until_blocked()`：默认运行到 index，不包含 activate。
- `src/app/services/ingestion_stages.py:activate_batch()`：单独执行激活阶段。
- `src/app/services/pipeline.py:_run_activation_gate_stage()`：激活前只校验数量和 embedding/source span 完整性。
- `src/app/services/parse_versions.py:ParseVersionService.activate()` / `batch_activate()`：只校验状态机、文档归属和 pointer 变更，不校验 canonical 结构库存。

### P0-B：activation gate 只看聚合数量，不看结构化证据库存

`_run_activation_gate_stage()` 会比较 `retrievable`、`embedded`、`indexed`、`valid_source_spans`、`artifact` 的总数，并校验 contextual/plain embedding policy；但没有比较：

- canonical manifest 声明的 `table_id` 是否全部有 table Child；
- 每个 table 的 Child 数是否完整、是否重复或跨版本混入；
- figure/formula/table 的 typed inventory 是否与 canonical artifact 一致；
- parent-child/source block/cell/row 的覆盖是否保持一致。

因此“只剩 narrative、但总 Child 数和 checkpoint 数量一致”的版本仍可能通过聚合 gate。`semantic_split` 的 `source_fidelity_completeness=1.0` 和 `structured_limit_completeness=1.0` 目前只是阶段输出快照，activation 阶段没有重新读取并核对 typed inventory。

代码位置：

- `src/app/services/pipeline.py:_run_semantic_split_stage()`：写入 fidelity 汇总。
- `src/app/services/pipeline.py:_run_index_stage()`：校验角色、数量、embedding 维度和 local ID。
- `src/app/services/pipeline.py:_run_activation_gate_stage()`：当前总量 gate。
- `scripts/rebuild_canonical_index.py:collect_integrity_metrics()`：`table_validation_rate` 校验 canonical tables 的质量状态，但不查询 DocumentChunk 的 table inventory。

### P0-C：canonical table assembly 会把语义表头当成数据行

`assemble_table_context()` 在 `src/app/services/table_evidence.py:52-151` 中只跳过分隔行和“整行与首个表头完全相同”的重复表头。`_is_secondary_header_line()` 只覆盖第一列相同/为空、且满足少数 continuation 条件的两级表头。

最小复现：

```text
Table 7
| Group | OPLS4 | OPLS5 |
| --- | --- | --- |
| Group | Edgewise | Pairwise |
| R-group | 0.93 | 1.06 |
```

当前结果会把 `Group / Edgewise / Pairwise` 作为第一条数据记录，且没有质量告警：

```python
headers == ("Group", "OPLS4", "OPLS5")
rows[0] == {"Group": "Group", "OPLS4": "Edgewise", "OPLS5": "Pairwise"}
```

这会直接污染 `extract_table_facts()` 的 row/column 映射，导致 `_extract_requested_metric_values()` 认为目标事实缺失，随后降级到 LLM 摘要。

### P0-D：多级列头和行头只支持一个狭窄形态

`src/app/services/table_evidence.py:_is_secondary_header_line()` / `_compose_header_lines()` 和 `src/app/services/table_extraction.py:_compose_headers()` 都最多消费一行二级表头，并要求特定的首列相同/为空形态。

CHARMM36/CHARMM36m 这类论文常见的 `System / Simulation`、`Property / force field`、`calcd / exptl` 多级头无法稳定展开。最小复现：

```text
Table 7
|  |  |  | Delta H vap | Delta H vap |
| --- | --- | --- | --- | --- |
| liquid | T | E inter | calcd | exptl |
| methanol | 25.00 | 8.51 | 8.95 | 8.95c |
```

当前结果含空表头、`calcd/exptl` 被当成数据行，无法生成 `Delta H vap calcd` 和 `Delta H vap exptl` 两个稳定列名。现有 OPLS grouped-header 测试覆盖的是“首列为空”的较简单情形，并不能证明真实论文表格可用。

### P0-E：查询时仍存在固定字符窗口，完整组装并未覆盖所有路径

canonical table 主 prompt 已经通过 `_prompt_context_text()` 返回完整 `table_context.markdown`，并由 token budget 控制；但以下路径仍然会截断：

- `src/app/services/search.py:_generic_table_value_rows()`：`_table_block_excerpt(..., max_chars=2400)`；
- `src/app/services/search.py:_table_citation_excerpt()` / `_table_block_excerpt()`：默认 1200，ablation 提高到 2400；
- legacy 文档表格路径：`_search_document_table_contexts()` 的 `prompt_text=block[:4000]`；
- scientific/limitation/parameterization 路径：`_window_text()` 使用 520、900、1000、1400 等窗口。

300 行表格的 ablation citation 已复现为 2400 字符截断，末尾停在半行 `| M`，`Model-299` 丢失。也就是说，“canonical assembly 无 4000 截断”与“最终回答所有路径都无字符截断”并不是一回事。

### P1-A：确定性 table answer 失败后，LLM 会压缩掉精确事实

当前调用顺序为：

```text
_answer_rag_first()
  -> _deterministic_table_answer_if_supported()
  -> _extract_requested_metric_values()
  -> facts 解析失败 / coverage 不足
  -> _draft_answer() 或 _repair_missing_table_answer()
  -> Ollama 摘要
```

因此开发报告中出现了“citation/source 通过，但 `21.0`、`95.3` 等精确值缺失”的结果。模型不是原始根因，而是确定性事实路径被错误禁用后的放大器。

另外，`_repair_missing_table_answer()` 在 `src/app/services/search.py:3460-3509` 中对 LLM 修复结果只检查“是否仍声称缺失”和“是否包含已抽取指标”，没有再次调用 `_unsupported_answer_numbers()`。修复模型如果新增未被证据支持的数字，最终结果可能仍被接受。

### P1-B：科学证据窗口会把命中的术语从最终 evidence 中切掉

`_deterministic_scientific_evidence_answer_if_supported()` 在 `src/app/services/search.py:3342-3395` 中使用 `_window_text()`，英文窗口默认只有 520 字符；其他专用检索路径还有 900/1000/1400 字符窗口。`FRET`、`GAlib`、`fast-folding` 等缺失术语与该窗口/anchor 选择存在直接风险，需要用失败题 fixture 固化后修复。

### P1-C：慢请求和 fallback 互相放大

开发 10 题 p95 为 162.4 秒，测试 p95 为 11.2 秒。表格事实缺失后进入 Ollama，多次重试和 fallback 会把一次本可由代码完成的查询变成长链路。性能问题应在 facts-first 修复后重新测量，不能先通过增大超时掩盖。

## 4. 非根因判断与保留约束

- 不是优先更换生成模型：开发/测试模型配置相同。
- 不是重新把全文切成固定 4000 字符大块：这会重新引入旧版第一页超过窗口后丢文本的问题。
- MinerU 不是当前主要断点：失败报告中的 table citation 已来自 canonical MinerU table Child；需要修的是版本可见性、表格结构化和回答链。
- 不改测试环境、不重建测试数据、不删除旧 parse/chunk/vector/artifact；开发版修复前保留当前 active 作为回滚基线。

## 5. 分阶段升级任务

### Task 0：冻结基线并补最小失败测试

**Files:**

- Modify: `tests/test_table_evidence.py`
- Modify: `tests/test_canonical_retrieval.py`
- Modify: `tests/test_ingestion_stages.py`
- Create: `tests/fixtures/task15_rag_regressions.py`

- [ ] **Step 1: 固化语义重复表头失败用例**

```python
def test_assembly_drops_semantic_secondary_header_row() -> None:
    table = assemble_table_context([
        CanonicalTableChunk(
            chunk_id="c1", document_id="d1", parse_version="v1",
            table_id="t1", ordinal=0,
            text=(
                "Table 7\n"
                "| Group | OPLS4 | OPLS5 |\n"
                "| --- | --- | --- |\n"
                "| Group | Edgewise | Pairwise |\n"
                "| R-group | 0.93 | 1.06 |"
            ),
        )
    ])
    assert table.rows == ({"Group": "R-group", "OPLS4": "0.93", "OPLS5": "1.06"},)
    assert "header_like_data_row" in table.quality_flags
```

- [ ] **Step 2: 固化任意两级/三级表头失败用例**

测试必须断言 `Delta H vap calcd`、`Delta H vap exptl` 存在，且 `calcd`、`exptl` 不作为 row label 或数据行。

- [ ] **Step 3: 固化 2400 字符半行截断失败用例**

构造至少 300 行表格，断言 citation excerpt 要么包含完整目标行，要么在完整行边界结束；不得以 `| M` 这种半行结束。

- [ ] **Step 4: 固化激活库存不一致失败用例**

构造 manifest 声明两个 table、Child 库存只包含一个 table 的 activation context，断言 `_run_activation_gate_stage()` 抛出 `ActivationError`，而不是仅凭总 Child 数通过。

- [ ] **Step 5: 先运行失败测试**

```powershell
pytest -q tests/test_table_evidence.py tests/test_canonical_retrieval.py tests/test_ingestion_stages.py
```

预期：新增用例在修复前失败；当前既有相关回归约 248 项通过，说明新增用例确实补的是覆盖缺口而不是破坏已有契约。

### Task 1：建立 canonical typed inventory 并把它纳入 activation gate（P0）

**Files:**

- Modify: `src/app/services/canonical_artifacts.py`
- Modify: `src/app/services/pipeline.py:_run_semantic_split_stage()`、`_run_index_stage()`、`_run_activation_gate_stage()`
- Modify: `src/app/services/parse_versions.py:ParseVersionService.activate()` / `batch_activate()`（只增加调用前置校验，不改变 pointer 事务语义）
- Modify: `scripts/rebuild_canonical_index.py`
- Modify: `scripts/evaluate_canonical_retrieval.py`
- Test: `tests/test_ingestion_stages.py`、`tests/test_canonical_acceptance.py`

- [ ] **Step 1: 在 canonical manifest 写入 typed inventory**

清单至少包含 `table_ids`、每个 table 的 `child_count`、`source_block_ids`、`row_count`，以及 figure/formula 的 ID 集合。清单必须来自 canonical artifact，不来自 LLM 输出。

- [ ] **Step 2: 在 semantic_split/index 输出同一份 Child inventory**

按 `(document_id, parse_version, block_type, table_id/figure_id/formula_id)` 统计；把缺失、重复、孤儿 Child 和跨版本混入记录为明确错误。

- [ ] **Step 3: 在 activation gate 做逐表等式校验**

除现有总量等式外，逐项比较 canonical manifest 与 DB/index inventory。任意 table 缺 Child、Child 多出未知 table、table row/source block 覆盖不一致，都必须阻止激活并写入 checkpoint 错误。

- [ ] **Step 4: 保持 staged/inactive 策略显式化**

重建脚本默认继续不激活；只有 retrieval/full-answer acceptance 全部通过，显式执行 `--activate-on-success` 才调用 `activate_batch()`。同时输出 `active_parse_version`、`candidate_parse_version`、`activation_block_reason`，避免把“已重建但未激活”误判成检索失败。

- [ ] **Step 5: 运行激活回归**

```powershell
pytest -q tests/test_ingestion_stages.py tests/test_canonical_acceptance.py tests/test_parse_versions.py
```

预期：库存不一致永远不能改变 `Document.active_parse_version`；完整候选版本只有在所有 gate 通过后才切换 pointer。

### Task 2：重写 table header/row classifier，保留完整行层级（P0）

**Files:**

- Modify: `src/app/services/table_evidence.py:assemble_table_context()`、`_is_secondary_header_line()`、`_compose_header_lines()`、`_row_label()`
- Modify: `src/app/services/table_extraction.py:_rows_to_dicts()`、`_compose_headers()`、`_header_index()`
- Test: `tests/test_table_evidence.py`、`tests/test_query_service.py`

- [ ] **Step 1: 统一 header-row classifier**

判定顺序必须是：separator → exact repeated header → semantic header-like row → data row。semantic header-like row 需要同时考虑非数值比例、与上一层表头的列级对应、`calcd/exptl` 等列标签、重复 group 名和 caption/footnote 语义，不能只依赖“首列为空”。

- [ ] **Step 2: 用列级向前填充合并任意深度表头**

对每一列维护最近的非空父表头，生成稳定列名，例如 `Delta H vap calcd`；同名列使用确定性后缀而不是覆盖。合并后若仍有空列名，写入 `blank_headers`/`unresolved_multilevel_header` quality flag。

- [ ] **Step 3: 保留多级行标签**

把 `System`、`Simulation`、`Variant` 等非数值层级按出现顺序合并为 `RS peptide / C36` 形式的 `row_label`，不把数值单元格当作 row label；现有 `TableFact.row_label` 对外保持字符串兼容。

- [ ] **Step 4: 对真实 CHARMM36m fixture 验证**

至少断言 `RS peptide / C36 -> 80 ± 2`、`RS peptide / C36m -> 1.8 ± 0.5`、`FG-nucleoporin peptide / C36m -> 1.1 ± 0.3`、`HEWL19 peptide / C36m -> 0.5 ± 0.4` 能被 `extract_table_facts()` 精确返回。

- [ ] **Step 5: 运行表格单元测试**

```powershell
pytest -q tests/test_table_evidence.py tests/test_query_service.py
```

### Task 3：所有 canonical table facts 走无字符截断路径（P0）

**Files:**

- Modify: `src/app/services/search.py:_generic_table_value_rows()`、`_table_citation_excerpt()`、`_table_block_excerpt()`、`_context_table_evidence_text()`
- Modify: `src/app/services/search.py` 中 scientific/limitation/parameterization 的专用 evidence 构造函数
- Test: `tests/test_canonical_retrieval.py`、`tests/test_query_service.py`

- [ ] **Step 1: generic table answer 改为消费 `TableContext.rows/TableFact`**

当 context 已有 `table_context` 时，禁止先调用 `_table_block_excerpt(..., 2400)`；按完整 rows/facts 做列选择和 round-robin 行选择。只有 legacy、没有结构化 table context 时才允许使用兼容路径。

- [ ] **Step 2: citation excerpt 采用完整行边界**

任何有长度上限的展示 excerpt 必须在换行处结束，并优先保留问题命中的完整行、表头和 caption；不得返回半行。回答事实不能依赖 citation excerpt 的长度上限。

- [ ] **Step 3: 移除 canonical scientific evidence 的固定字符窗口**

以 required-anchor 集合和统一 token budget 选择完整句子/完整 Child；若无法在预算内保留全部 anchor，返回 evidence-insufficient，而不是静默丢掉术语。

- [ ] **Step 4: 覆盖旧 4000 边界和新 2400 边界**

保留已有 500 行 canonical 完整性测试，并新增目标行位于第 2400 字符之后、目标行不被半行截断的测试。

### Task 4：facts-first 确定性回答与 LLM 兜底二次校验（P1）

**Files:**

- Modify: `src/app/services/search.py:_deterministic_table_answer_if_supported()`、`_extract_requested_metric_values()`、`_repair_missing_table_answer()`、`_repair_unsupported_numeric_answer()`
- Test: `tests/test_query_service.py`、`tests/test_canonical_retrieval.py`

- [ ] **Step 1: 明确定义 deterministic success**

只要 canonical `TableFact` 覆盖问题要求的 table/row/column/value，直接返回代码组织的 answer；不能因为 citation excerpt 短或普通摘要缺词而放弃 facts-first。

- [ ] **Step 2: 给 LLM repair 注入 canonical facts 清单**

prompt 中同时提供原始表格事实、row label、column、value、source chunk ID，并要求精确复制数字、单位、表号和技术缩写；LLM 不负责重新解析表格。

- [ ] **Step 3: repair 返回后重复执行完整校验**

顺序必须是：`_answer_lacks_requested_metrics()`、`_unsupported_answer_numbers()`、citation index 支持校验；任一失败都返回 deterministic fallback，并记录 warning，不接受模型新写入的未支持数字。

- [ ] **Step 4: 写入伪造数字回归测试**

用 fake Ollama 返回 `999.0`，证据只包含 `21.0`；断言最终 answer 不含 `999.0`，且包含 `21.0`。用 fake Ollama 删除 `95.3`，断言 coverage retry 或 deterministic fallback 恢复 `95.3`。

### Task 5：科学术语覆盖和性能回归（P1）

**Files:**

- Modify: `src/app/services/search.py:_deterministic_scientific_evidence_answer_if_supported()`、`_scientific_anchor_excerpt_window()`、相关专用检索函数
- Modify: `scripts/evaluate_canonical_retrieval.py`
- Test: `tests/test_canonical_retrieval.py`、`tests/test_query_service.py`

- [ ] **Step 1: 为每个题保存 required anchor coverage**

至少覆盖 `FRET`、`GAlib`、`fast-folding`、`QM`、`NMR`、`Table 1`、`RS peptide` 等术语；报告同时记录 top-5、top-10 evidence 是否包含原始术语和精确数字。

- [ ] **Step 2: 先运行 10 题 staged retrieval-only**

不激活、不改测试环境、不跑 Agent。任何 table inventory、Recall@5/10、citation/source location 失败都停止，不进入 full-answer。

- [ ] **Step 3: 再运行同题 RAG full-answer**

对比开发旧 active、开发修复候选和测试 baseline，分别记录 strict pass、answer pass、citation/source、missing terms、p50/p95、timeout/error 数量。

- [ ] **Step 4: 设定验收门槛**

候选版本必须满足：

1. 同题 10 题 strict pass 至少 9/10；
2. table/metric 题 required numeric/label facts 覆盖率 100%；
3. citation pass rate 和 source match rate 均为 100%；
4. `Recall@5`/`Recall@10` 对每个 required fact 有明确报告，不能用空答案算通过；
5. 0 timeout、0 HTTP error；p95 目标不超过 30 秒，若硬件负载使该指标不可达，必须单独记录慢点而不能放宽事实完整性门槛；
6. activation inventory 100% 一致后才允许切 active。

### Task 6：开发环境部署与停机点

**Files/Commands:**

- Development only: `/home/<user>/knowledge-agent-dev`
- Test environment: read-only `/home/<user>/knowledge-agent-test`
- GPU binding: every development command explicitly uses `CUDA_VISIBLE_DEVICES=0`; test commands, if later approved, use `CUDA_VISIBLE_DEVICES=1`.

- [ ] **Step 1: 只同步已通过本地测试的文件**

同步前记录 commit SHA 和文件 SHA-256；不覆盖 test 版本，不删除旧 active/chunk/vector/artifact。

- [ ] **Step 2: 重启开发 API/worker/Ollama 后做只读 preflight**

确认 API、worker、Ollama、Redis、pgvector 状态；确认 GPU1 没有开发任务占用；确认当前 active pointer 未被测试命令改变。

- [ ] **Step 3: 按 Task 5 顺序运行评测**

先 retrieval-only，再 RAG full-answer；Agent 评测另立任务，不与本次 RAG 根因验收混合。

- [ ] **Step 4: 通过后再由用户确认 activation**

没有 `strict_pass=true`、inventory 一致和用户确认，不执行 `--activate-on-success`，也不执行旧数据删除。

## 6. 预期交付物

- 本计划文件：`docs/superpowers/plans/2026-08-04-task15-rag-bug-analysis-and-upgrade-plan.md`。
- 表格结构化回归测试和 canonical retrieval 回归测试。
- 一份包含 active/candidate version、typed inventory、Recall@5/10、strict pass、citation/source、missing terms、latency 的对比报告。
- 通过验收后再更新 `docs/work.md`，记录实际修复、部署 SHA、评测报告路径和是否激活；本计划本身不改变 active pointer。

## 7. 执行顺序摘要

1. 先补失败测试，证明 active inventory、语义表头、多级表头和 2400 截断缺陷。
2. 修复 inventory gate，保证候选版本不会在缺 table Child 时进入 active。
3. 修复 table header/row assembly，再移除 canonical table 的字符窗口。
4. 恢复 facts-first deterministic answer，并给 LLM repair 加数字/引用二次校验。
5. 修复 scientific anchor coverage，跑 staged retrieval-only，再跑 RAG full-answer。
6. 只有候选版本达到验收门槛并获得确认后，才激活；测试环境和旧数据全程保持不变。

## 8. 2026-08-04 全量 30 条复测后的新结论

GPU0 candidate 的服务器全量 RAG 复测已经证明，检索层不是当前主阻塞：Recall@5/10 均为 1.0，citation/source 校验通过；但 full-answer 只有 21/30，answer pass rate 为 70%。失败集中在表格事实的组织，而不是 candidate 表格 Child 缺失。

已确认的具体表现：

- `charmm36idpsff_table_metrics` 的 evidence 同时包含 Aβ40 的 `12.0/11.53` 和 ACTR 的 `25.00/13.07`，回答却按 context 轮询重复 ACTR，漏掉 Aβ40。
- `charmm36m_table_metrics`、`ff14sb_table_metrics`、`ff99sb_disp_table_metrics` 等多目标表格只输出第一个高分行或错误的表格行，后续请求的 row/property 没有进入最终 answer。
- `opls4_table_metrics`、`opls5_table_metrics` 已检索到目标 Table 8/Table 4/Table 5/Table 7 facts，但回答仍只覆盖其中一个表或一组数值。
- 同一 canonical table 的多个 row-level context 携带相同事实；当前 generic answer 的 round-robin 以 context 为单位，未按 `(document_id, parse_version, table_id, row_index, row_label, values)` 做跨 context 去重，也没有把问题中显式指定的多个 row selector 作为必须覆盖集合。

本轮修复计划：

1. 先添加失败测试，证明多个显式 row selector 都必须保留，重复 table context 不能产生重复答案行，且每个目标 table/property 都能跨 context 保留。
2. 在 `search.py` 的 facts-first generic table answer 中，按 canonical table identity 合并候选行，稳定去重；优先选择显式目标行和目标 property，最后才使用普通相关性排序和行数上限。
3. 保留跨表 round-robin 的 token/行数边界，但不能让同一 row 的重复 context 消耗边界；不改 candidate map、active route、数据库或 GPU1/test。
4. 通过 focused/full local regression 后，重新同步并只重启 GPU0/dev，再按顺序跑 candidate retrieval、candidate RAG、candidate Agent，以及 GPU1/test active RAG/Agent 全量 30 条。
5. 只有 RAG 与 Agent 都达到严格验收门槛，并且 active/candidate 对照报告完整，才结束本轮；否则继续针对失败案例迭代。

### Agent 对照的独立发现

GPU1/test active Agent 全量 30 条复测为 `6/30`，`answer_pass_rate=20%`、`citation_pass_rate=100%`、`synthesis_applied_rate=0%`。这与 candidate RAG 的表格行选择缺陷不同：active 已能返回有 citation 的本地答案，但答案常把英文 required terms、本题的多项数值或完整技术短语翻译/压缩掉，导致严格答案门失败。GPU0 candidate Agent 仍需等当前任务完成后再比较。

因此 Agent 不能只作为 RAG 的附带指标处理，必须单独验证：

- Agent evaluator 的 required-term/answer/citation 评分是否与项目验收契约一致；
- `AgentExecutor`、`RAGAdapter`、`AgentSynthesizer` 和 local fallback 是否在证据已覆盖时保留事实词项和数字；
- active/candidate 两条 Agent 路由是否使用相同的模型、超时和合成配置；
- 任一通用修复必须通过 active 与 candidate 的同题 30 条对照，不能通过硬编码题目术语“刷过”评估。
