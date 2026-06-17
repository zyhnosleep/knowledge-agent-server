# LLM Wiki 工作记录

> 维护说明：这份文档用于持续记录 `LLM Wiki` 项目的阶段性工作、已完成事项、遇到的问题、排查结论和后续计划。每次重要代码改动、服务器验证、重新 ingest、query 结果审查后，都应该追加或更新对应条目。

## 1. 当前状态概览

截至 2026-06-17，项目已经从“能跑通的无 Docker 服务端原型”推进到“具备 wiki-first 查询、MinerU PDF 解析、SAC-KG 启发式 ingest、query/citation 修复和服务器实机验证”的阶段。

当前主线不是继续堆新功能，而是把真实论文场景中的检索、表格证据链和回答质量稳定下来。最近的重点对象是 SAC-KG 论文中的 Table 2 和 Table 5，因为它们暴露了 PDF 表格解析、wiki table 写入、query 上下文组装、answer repair、citation 对齐之间的真实链路问题。

当前最新明确结论：

- MinerU pipeline 已经接入，外部拿到的 MinerU markdown 也证明 Table 2 / Table 5 可以被解析出来。
- wiki/DB 中存在目标表格，但 `/api/query` 的 answer/repair 阶段仍可能只拿到“提到 Table 5 的正文段落”，没有拿到真正 markdown 表格行。
- 第 3 问未达标的核心原因已经从“模型没读懂表格”定位为“query 上下文组装阶段没有把真实 table block 提前送入 answer/repair”。
- 服务器已经确认运行在最新提交 `5df836d`，API 进程和日志正常，因此当前不是部署或重启问题。
- 下一步修复重点应放在 `_build_contexts()`、table-first context promotion、table evidence 判定收紧，而不是继续排查 MinerU。

## 2. 已完成的工作

### 2.1 服务端基础设施

- 建立无 Docker 部署路线：`FastAPI + Redis worker + SQLite + 本地文件 + Ollama + nohup`。
- 修复 `sh` 环境下启动脚本兼容问题，不再依赖 bash-only 的 `source`。
- 增强 `start_api.sh`、`start_worker.sh`、`status.sh`、Redis/Ollama 检查逻辑。
- 明确服务器没有系统 `sqlite3` 命令时，可以使用 Python 标准库 `sqlite3` 操作数据库。
- 引入 `OLLAMA_KEEP_ALIVE` 配置，用于减少 Ollama 和 MinerU 争抢 RTX 3090 显存的风险。
- 服务器验证时确认日志实际写在 `logs/api.log`，不是 `run/api.log`。

### 2.2 Git 与协作流程

- 初始化本地 Git baseline。
- 增加 `.gitignore`，排除 `data/`、`logs/`、`run/`、`tmp/`、PDF 原文、缓存和环境文件。
- 增加 `.gitattributes`，稳定脚本和代码换行风格。
- 形成“每轮改动后提交”的协作习惯，便于服务器 `git pull` 和回滚审查。
- 在 query 表格证据链修复中使用多代理流程：planner / backend worker / reviewer 分工，最终由主线程整合并跑测试。

### 2.3 Wiki-first Query 主链路

- 将 query 主路径从传统 raw chunk-first 调整为 wiki-first。
- 优先检索 `WikiPage`，只有用户明确要求原文证据或 wiki 不足时才 fallback 到 raw chunk。
- 增加 query facets，使多实体问题如 Generator / Verifier / Pruner 能覆盖多个主题。
- 修复 citation 编号归一化与重排问题，避免答案中出现未返回的 `[3]`、`[[0]]`、`[[sources/...]]` 等异常引用。
- 增加普通问题优先返回 wiki citation、原文证据问题才保留 raw chunk citation 的逻辑。
- 通过 SAC-KG 四问脚本持续验证 query 行为：定义、组件、Table 5 指标、Table 2 ablation。

### 2.4 SAC-KG 启发式 ingest 改造

- 将早期“粗摘要式 ingest”推进为更接近 SAC-KG 思路的轻量流程。
- 增加 head-driven generation、tail entity pruner、verifier error types 等机制。
- Verifier 覆盖 quantity too small、format error、head entity error、head-tail contradiction、missing evidence、duplicate、potential conflict 等错误类型。
- Pruner 从笼统保留/删除调整为基于 verified triples 的 tail entity `grow / keep / prune`。
- 明确当前目标不是完整复现论文级 SAC-KG，而是把论文方法中适合本项目的结构化知识整理思路落地。
- 在后续引入 20 篇新文献之前，需要优先验证该轻量化流程是否能泛化，而不是只对当前 SAC-KG 论文有效。

### 2.5 MinerU PDF 解析接入

- 将 PDF 解析主方向从 pypdf / Ollama Vision 逐步调整到 MinerU pipeline。
- 使用可选依赖 `mineru[pipeline]`，并增加相关环境变量：`MINERU_ENABLED`、`MINERU_BIN`、`MINERU_BACKEND`、`MINERU_MODEL_SOURCE`、`MINERU_OUTPUT_DIR`、`MINERU_TIMEOUT`、`MINERU_EXTRA_ARGS`。
- 修复 MinerU CLI 传入相对路径导致找不到 PDF 的问题，改为传绝对路径。
- 支持 MinerU v2 的嵌套 `content_list_v2.json` 结构。
- 增加 MinerU markdown 兜底读取，把 `.md` 中的表格补进 `document_intelligence.tables`、`ParsedChunk` 和 `parsed.text`。
- 对用户提供的 MinerU markdown 文件进行审查，确认 Table 2 / Table 5 内容本身存在，因此当前不再优先怀疑 MinerU 原始输出。

### 2.6 Query 表格证据链修复

已完成多轮 query 层修复，包括：

- 对 metric/table query 增加 table-first 意图识别。
- 对 Table 2 / Table 5 相关问题，优先保留表格 citation，而不是同页 summary citation。
- 对 `citation.excerpt` 与 `prompt_text` 进行合并使用，避免表格内容只出现在 citation excerpt 时 answer repair 看不到。
- 修复 answer citation markup：`[[0]]`、孤立 `[n]`、超界 citation index 会在最终返回前重新编号或删除。
- 增加 deterministic fallback：当模型声称表格缺失，而表格证据存在时，尝试用表格内容直接生成答案。
- 增加 HTML table 到 markdown 的转换测试，覆盖 `colspan` / 多级表头场景。
- 使用逻辑层诊断脚本确认当前第 3 问卡在 `contexts` 没有真实表格行，而不是 API 旧进程或 MinerU 缺表。

相关已提交工作包括：

- `f0f5e63 Generalize table metric answer repair`
- `20b030e Harden storage and ingestion safety`
- `61439a1 Avoid hardcoded ablation table answer`
- `a965123 Add structured table quality and wiki linting`
- `5df836d Use citation excerpts in query evidence repair`

## 3. 遇到的问题与当前结论

### 3.1 Table 5 数值仍不进入答案正文

现象：

- 第 3 问：`SAC-KG 在 OIE2016 或 NYT 数据集上的指标是什么？`
- 返回 citation excerpt 曾经包含完整 Table 5：`74.7 / 73.2 / 88.8 / 87.3`。
- 但 answer markdown 仍只说 Table 5 中有 F1/AUC 指标，没有输出具体数值。

最新诊断输出显示：

```text
contexts count: 5
[1] excerpt has Table 5: True
[1] prompt_text has 74.7: False
[1] prompt_text has 88.8: False
table_indexes: [1]
extracted_metrics: empty
```

观察结论：

- 当前 `contexts` 命中的是 raw chunk / 正文段落，`slug=None`，不是 wiki table block。
- `[1]` 只是 “As shown in Table 5...” 这种正文描述，里面没有实际表格行。
- `_table_citation_indexes()` 将其判为 table context，是因为它包含 `Table 5`，但它不是真正的 table evidence。
- `_extract_requested_metric_values()` 为空，因此 deterministic fallback 只能返回正文片段，无法返回数值。

当前根因：

- 表格证据存在于 wiki/DB 中，但没有在 `_build_contexts()` 阶段进入 answer/repair 使用的 contexts。
- 最终 citation 能找到 Table 5 并不等价于 answer repair 阶段拿到了 Table 5 表格。

### 3.2 `answer_missing=True` 但 `answer_lacks_metrics=False`

现象：

```text
answer_missing: True
answer_lacks_metrics: False
```

解释：

- `answer_missing=True` 说明模型确实说了“提供的上下文没有具体数值”。
- `answer_lacks_metrics=False` 不是说明答案合格，而是因为 `extracted_metrics` 为空，系统没有可比较的指标值。
- 这进一步证明问题在 metrics extraction 的输入上下文，而不是模型最终措辞本身。

### 3.3 Table mention 被误当成 table evidence

现象：

- 只有 “As shown in Table 5...” 的正文段落被选为 `table_indexes=[1]`。

问题：

- 当前 `_context_has_table_data()` 对 `Table 5` 这类文本过于宽松。
- `Table 5` 中的数字 `5` 可能被当作“有数字”，从而误判为表格证据。

应修正方向：

- 对 metric query，真正 table evidence 应优先要求 markdown table 行、目标小数值、指标结构，或来自 wiki `## Tables` block。
- table mention 可以作为回查信号，但不能直接作为 deterministic fallback 的最终证据。

### 3.4 Table 2 ablation 答案基本可用但仍有质量问题

现象：

- 第 4 问已经不再说“未包含 Table 2”。
- citation excerpt 包含 Table 2。
- 但答案中出现了类似 `Iteration rounds: full Model reports recalls Number of recalls, precision Precision.` 的异常句子。

判断：

- 这是 ablation 表格摘要器把表头行误当成数据行的质量问题。
- 优先级低于 Table 5 数值无法进入第 3 问答案，但后续需要修。

### 3.5 编码显示问题

现象：

- 部分 Windows / Codex 附件输出显示为乱码。

当前判断：

- 如果 API 原始 JSON 在浏览器或 `curl` 中是正常 UTF-8，则这是复制/终端显示问题。
- 如果 API 原始响应也是乱码，则需要单独检查 response 编码、终端 locale、日志/文件写入编码。

## 4. 下一步修改计划

### 4.1 第一优先级：修复 table-first context promotion

目标：让真实 Table 5 markdown block 在 answer draft / repair / deterministic fallback 之前进入 `contexts`。

计划：

- 在 `_build_contexts()` 阶段，对 table/metric query 增加强制表格补充逻辑。
- 如果 question 命中 `Table N`、dataset name、metric name，优先从 matched wiki page 的 `## Tables` 中抽取相关 table block。
- 将真实 table block 插入 contexts 前列。
- raw chunk 只作为 fallback，不允许只提到 Table 5 的正文段落抢占 table evidence。

验收：

- 诊断脚本中 `contexts[0]` 或前几个 context 必须包含 `| SAC-KG ChatGPT | 74.7 | 73.2 | ... | 88.8 | 87.3 |`。
- `_extract_requested_metric_values()` 必须抽出 OIE2016 和 NYT 的 F1/AUC。

### 4.2 第二优先级：收紧 table evidence 判定

目标：避免把 table mention 当作 table data。

计划：

- 修改 `_context_has_table_data()`。
- `Table 5` 这种只有表号的文本不能算 table data。
- metric query 下优先要求：
  - markdown table 管道 `|`；或
  - 至少一个非表号的小数值；或
  - 指标词 `F1/AUC/Precision/Recall` 与数值共同出现。

验收：

- “As shown in Table 5...” 不能单独进入 deterministic fallback。
- 真正 markdown table block 必须进入 table indexes。

### 4.3 第三优先级：table mention 回查同页表格

目标：利用 raw chunk 的 “As shown in Table 5” 作为线索，回查 wiki table block。

计划：

- 当 context 是 raw chunk 且提到 `Table N`，但没有表格行时：
  - 根据 document/page 或 matched wiki page 查找对应 source summary。
  - 从 `## Tables` 中找 `Table N` block。
  - 补入一个 table context。

验收：

- 即使向量检索先命中正文段落，也能补到对应 Table 5。

### 4.4 第四优先级：修复 ablation 摘要器误读表头

目标：去除 Table 2 答案中的异常句子。

计划：

- 在 `summarize_ablation_table()` 或表格结构化阶段过滤表头重复行。
- 避免将 `Iteration rounds / Model / Number of recalls / Precision` 当作数据记录。

验收：

- 第 4 问答案不再出现 `full Model reports recalls Number of recalls`。
- Table 2 citation 仍然保留。

### 4.5 后续泛化验证

目标：确保当前修复不是只对 SAC-KG 一篇论文有效。

计划：

- 在一次性加入 20 篇新文献前，建立固定验收表。
- 每篇至少抽查：source summary、tables、figures、metric query、ablation/实验对比 query、citation 对齐。
- 对不同论文中不叫 Table 5 的指标表、不同模型名和不同数据集名做回归测试。

## 5. 服务器验证流程

每次 query 层改动部署后，按以下顺序验收：

```sh
cd ~/llm_wiki_server
git pull
git rev-parse --short HEAD
sh scripts/start_api.sh
bash tmp/sac_kg_query_batch.sh
```

重点检查：

- 第 3 问 answer markdown 必须包含：
  - `OIE2016 F1 74.7 / AUC 73.2`
  - `NYT F1 88.8 / AUC 87.3`
- 第 3 问 citation excerpt 必须包含：
  - `Table 5`
  - `88.8`
- 第 4 问不得再说“未包含 Table 2”。
- 第 4 问 citation excerpt 必须包含 `Table 2`。
- 第 1、2 问 citation 编号不得越界。
- 中文问题默认中文回答。

如果结果异常，优先运行逻辑层诊断脚本，检查：

- `contexts` 是否含真实表格行。
- `table_indexes` 是否命中真实 table block。
- `extracted_metrics` 是否为空。
- `deterministic` 是否能输出具体值。

## 6. 面向未来的维护方式

建议以后每次重要工作后按以下模板追加：

```md
## YYYY-MM-DD 工作记录：主题

### 已完成
- ...

### 遇到的问题
- ...

### 排查结论
- ...

### 后续计划
- ...

### 验证结果
- `pytest -q`: ...
- 服务器四问: ...
```

这样 `docs/work.md` 会逐渐成为项目的长期工程日志，而不是一次性总结。
