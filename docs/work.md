# LLM Wiki 工作记录

> 维护说明：这份文档用于持续记录 `LLM Wiki` 项目的阶段性工作、已完成事项、遇到的问题、排查结论和后续计划。每次重要代码改动、服务器验证、重新 ingest、query 结果审查后，都应该追加或更新对应条目。

## 1. 当前状态概览

截至 2026-06-28，项目已新增：

- **Agent 层** (v3)：确定性策略路由、外部 API 证据合成（可选回退到 local）、answer 验证、有界重试、数据库持久化 trace（AgentTraceRun/AgentTraceStep）、30 天会话 TTL（ConversationSession）、SSE 流式进度推送（`POST /api/agent/query/stream`）、前端流式面板。Agent 层将本地 RAG 作为证据，可选外部 API 用于答案合成，仅从提供的证据中引用。
- **DeepSeek 外部合成已实测接通**：服务器 `.env` 已切到 `https://api.deepseek.com`，真实前端 Agent 面板提交新问题后返回 `Provider: external_api/deepseek-v4-pro`、`Status: COMPLETED`、trace 持久化可用。当前外部 API 只负责基于 RAG 证据合成最终回答，不替代本地检索链路。
- **Agent Evidence Pack (v4, 2026-06-28)**：新增 `EvidenceItem`/`EvidencePack` 模型、`QueryService.retrieve_evidence()` 只读检索路径（不生成答案/不写 QA 记录）、`RAGAdapter.retrieve_evidence()`、`rag.retrieve_evidence` 内置工具注册。Agent 正常流程调整为 `route → retrieve → rag.answer → synthesize → verify → finalize`，retrieve 步骤包含 `evidence_count`、`table_evidence_count`、`evidence_kinds`、`source_stages`、`support_hints` 元数据。`answer.synthesize` 可选接受 evidence pack。详见 `docs/worklogs/2026-06-28-agent-evidence-pack.md`。
- **Quality Dashboard**：只读 API (`GET /api/quality/dashboard`) 展示最近的 loop run manifest 摘要，包括命令状态、query gate、MinerU/service ingest smoke 摘要和失败 case 归因。前端已重设计为 Swiss Pulse 运维控制台风格。dashboard 只读现有 loop artifacts，不会启动新任务。
- **前端重设计**：`index.html` 从过大的米色卡片页升级为紧凑的运维控制台（Swiss Pulse 视觉方向）。

截至 2026-06-17，项目已经从”能跑通的无 Docker 服务端原型”推进到”具备 wiki-first 查询、MinerU PDF 解析、SAC-KG 启发式 ingest、query/citation 修复和服务器实机验证”的阶段。

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

### 2.3 Agent 层与 Quality Dashboard (2026-06-27)

- 新增 Agent v2a 层：确定性策略路由、回答验证、有界重试、会话记忆、只读工具注册。详见 `docs/worklogs/2026-06-27-agent-layer.md`。
- 新增只读 Quality Dashboard：`GET /api/quality/dashboard?limit=5` 读取 `QUALITY_REPORTS_DIR` 下的 loop run manifest，展示命令状态、query gate、MinerU/service ingest smoke 摘要和失败 case 归因。`skipped_reports` 仅统计损坏/不可读的 manifest，不含 limit 截断。支持 `query_eval.json` 备选回退读取。dashboard 不会启动新 loop 或长期任务。
- 前端 `index.html` 完全重设计：从米色卡片页升级为 Swiss Pulse 运维控制台，使用中性调色板、电蓝色强调、网格锁定布局、紧凑卡片（6px 圆角）、只读质量面板通过 `textContent`/DOM 节点渲染（无 `innerHTML` 风险）。

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

## 2026-06-25 工作记录：RAG-first、服务器同步与 profile-term evidence 调试

### 已完成

- 当前方向已经从早期的 wiki-first 查询逐步转向 **RAG-first + Paper Profile 路由**：wiki 暂时保留为阅读层和调试层，但 `/api/query` 的默认证据链应以原文 `DocumentChunk` / table chunk 为准。
- 明确了 Paper Profile 的定位：它只用于“先定位哪篇论文”和“扩展检索词”，不作为最终答案 citation；最终答案仍必须引用原文 chunk 或表格 chunk。
- 本地与服务器代码做过同步核对：最初服务器和本地都在 `e268937`，但本地存在未提交改动；随后本地提交并把相同文件同步到服务器。
- 本地提交记录：
  - `97f2151 Add supplemental profile term evidence retrieval`
  - `a28293b Strengthen supported evidence term repair`
- 服务器因为 bundle/scp 不稳定，采用“同步文件后服务器本地提交”的方式，因此服务器 commit hash 与本地不同：
  - `6fc3e7a Add supplemental profile term evidence retrieval`
  - `2c95d19 Strengthen supported evidence term repair`
- 虽然 commit hash 不同，但已用 `git ls-tree` 核对关键文件 blob hash，本地和服务器的 `src/app/services/search.py`、`tests/test_query_service.py` 内容一致。
- 本地验证通过：
  - `python -m compileall src tests scripts`
  - `D:\Miniconda3\python.exe -m pytest -q --basetemp tmp/pytest-full`
  - 结果：`223 passed`
- 服务器验证通过：
  - `python -m compileall src tests scripts`
  - `pytest tests/test_paper_profile.py tests/test_query_service.py -q`
  - 结果：`142 passed`
- API 已通过 `scripts/start_api.sh` 重启，最近一次确认的新进程 PID 为 `904180`。

### 最近修复内容

- 增加 supplemental profile term evidence retrieval：当 Paper Profile 中的高信号术语没有被当前 RAG 上下文覆盖时，尝试在同一篇论文内部补充对应原文 chunk。
- 增强 supported evidence term repair：答案已经有证据但缺少关键术语时，从已选证据中补充术语说明，避免 benchmark 因“证据在但答案没写出术语”失败。
- 新增回归测试覆盖：
  - profile term evidence chunk 被普通高频 chunk 挤掉时，应该补入对应证据；
  - 第 4/5 个已选 context 中的 `CMAP` / `QM` 也应进入 supported term note，而不是只看前三个 context；
  - MinerU/PDF 文本中类似 `0 . 5 kcal mol` 的 spaced numeric unit 应规范化为 `0.5 kcal/mol`。

### 当前 benchmark 结果

最近一次服务器 4-case 验证命令：

```sh
python scripts/query_eval.py benchmarks/query/internal_research_v1.json \
  --base-url http://127.0.0.1:8000 \
  --timeout 180 \
  --case-id ff14sb_overview \
  --case-id ff14sb_mechanism \
  --case-id ff19sb_overview \
  --case-id ff99sb_ildn_mechanism \
  --output tmp/query_benchmark/after_supported_term_repair.json \
  --markdown tmp/query_benchmark/after_supported_term_repair.md
```

结果：

- `ff14sb_overview`: pass
- `ff14sb_mechanism`: pass
- `ff19sb_overview`: fail，原因为 `missing_expected_answer_text`
- `ff99sb_ildn_mechanism`: fail，原因为 `missing_expected_answer_text`

这说明当前改动已经修复了 ff14SB 的 `GAlib` / fitting / side-chain / backbone 相关覆盖问题，但 ff19SB 与 ff99SB-ILDN 仍有关键证据没有进入最终答案。

### 遇到的问题

- SSH/SCP 通道不稳定：`ssh` 小命令可以成功，但 bundle 传输和某些 heredoc 调试命令会失败；最终采用已验证可用的精确 `scp` 文件同步方式。
- 本地和服务器 commit hash 分叉：这是同步方式导致的历史差异，不代表代码内容不同。后续推 GitHub 时建议以本地 `main` 的提交链为主线，服务器只作为运行验证环境。
- `tmp/remote_query_debug.py` 被临时用于服务器 SQLite / QueryService 调试，不应作为正式项目代码提交。
- 当前 4-case 失败不是 MinerU 解析失败，也不是 Paper Profile 词表完全缺失：
  - ff19SB 的动态 profile terms 中已经包含 `CMAP`、`CMAPs`、`QM-based`、`OPC`、`TIP3P`。
  - ff99SB-ILDN 的动态 profile terms 中已经包含 `Boltzmann`、`500 K`、`barrier` 等。
  - DB 原文 chunk 中也确实存在 ff19SB 的 CMAP 证据和 ff99SB-ILDN 的 `<0.5 kcal mol^-1` 证据。
- 真正问题在 context selection / finalization：
  - ff19SB 的 `CMAP` chunk 能在 `_search_source_chunks()` 中出现，但最终被高分 claim context、figure context、同页 citation cap 或后续排序挤掉，最终 answer contexts 中没有 `CMAP`。
  - ff99SB-ILDN 的 page 5 chunk 含有 `<0.5 kcal mol^-1` 与 barrier 解释，但最终 contexts 仍优先保留 page 3 / page 7 / page 4 等 chunk，导致答案没有 `0.5 kcal/mol`。
- 因此不能用硬编码把 `CMAP` 或 `0.5 kcal/mol` 塞进答案；下一步要修的是“profile-term evidence 的保留与排序”。

### 当前判断

- RAG-first 是正确方向，但现在还缺少“证据保留策略”：只把 Paper Profile 用作路由还不够，必须保证被 profile 命中的关键原文 chunk 不会在最终 context cap 中被挤掉。
- 当前失败已经从“召回不到论文”推进到“召回到论文，但关键 chunk 未进入最终答案”。这是更细的排序和剪枝问题，不是 PDF 重 ingest 问题。
- SAC-KG 的思想仍然有用，但应从“生成 wiki 页面”转为辅助 RAG 的三个能力：
  - routing：帮助先定位论文；
  - verification：检查答案术语/数字是否有原文证据；
  - pruning：在同一篇论文内部剪掉低价值、高重复 chunk，同时保留关键证据 chunk。

### 后续计划

- 暂停继续扩大答案修复模板，优先修 `_build_rag_contexts()` / `_finalize_contexts()` 的 evidence preservation。
- 给 supplemental/profile-term contexts 增加“保留槽位”或更合理的排序策略，确保 `CMAP`、`<0.5 kcal mol^-1` 这类高信号 term chunk 至少有机会进入最终 contexts。
- 避免 broad evidence 再次引入串文献问题：只允许在已路由到的同一篇论文内部补充 profile term evidence。
- 修复后先跑 4-case：
  - `ff14sb_overview`
  - `ff14sb_mechanism`
  - `ff19sb_overview`
  - `ff99sb_ildn_mechanism`
- 4-case 通过后再跑完整 30-case benchmark，并重点观察：
  - `wrong_source_hint`
  - `cross_paper_contamination`
  - `no_citation`
  - `missing_expected_answer_text`
  - table / metric citation 是否包含真实数值。

## 2026-06-26 工作记录：MinerU/RAG Loop、公网远端验证与失败归因

### 已完成

- 按用户要求把 OCR/解析依赖方向明确为 `mineru[all]`，本地 `D:\Miniconda3` 与远端环境均验证 MinerU `3.4.0` 可用。
- 当前 OCR/版面/表格/公式解析基线确定为 MinerU pipeline，而不是直接使用本地 Qwen 做基础 OCR；Qwen 后续更适合放在 MinerU 输出之后做语义增强、事实抽取、答案生成和证据验证。
- 新增并验证真实 MinerU parser smoke gate：`scripts/run_mineru_rag_loop.py` 只有显式传入 `--run-mineru-smoke` 时才调用真实 MinerU CLI，并把 `parser_mode`、chunk 数、page outputs、`content_list_v2.json` 等写入 manifest。
- 新增真实服务 ingest gate：显式传入 `--run-service-ingest-smoke` 时，loop 会上传 PDF 到 `/api/ingest/upload`，轮询 `/api/runs`，再读取 `/api/documents/{id}/quality` 与 `/api/wiki/lint`，把结果写入 manifest。
- 服务 ingest gate 已加固：默认不允许 duplicate-skip 假通过；轮询 runs 时不再只看最新 50 条，而是分页搜索目标 run。
- 新增 query eval + attribution gate：`--run-query-eval` 会生成 `query_eval.json/.md`，并进一步生成 `query_attribution.json`，记录 expected answer/citation term 命中情况、citation sources、source hint match 和 `likely_stage`。
- 新增显式 query gate 阈值：`--min-query-passed`、`--max-query-failed`、`--require-failure-attribution`，让 loop 从“命令成功”升级为“质量达标才算成功”。
- 公网远端 full30 query gate 最新结果为 `25/30 passed`，满足 `min_query_passed=24`、`max_query_failed=6`、`require_failure_attribution=true`。
- 公网远端 target6 attribution run 为 `6/6 passed`。
- 公网远端真实 service ingest gate 通过：`duplicate_skipped=false`、`parser_mode=pdf_mineru`、16 页、16 个 page outputs、10 张表、6 张图，且 `content_list_v2.json` 存在。
- 形成了进入 agent/frontend 前的执行底座：本地 pytest、真实 MinerU smoke、真实服务 ingest、query eval、失败归因和 manifest 报告都可以由同一个 loop 串起来。

### 遇到的问题

- 公网/内网访问方式切换容易造成误判：旧的 `llm-wiki-server` alias 指向内网地址，公网环境下会超时；公网操作应使用公网 host/port/key，但具体连接信息不应写入仓库。
- 真实 service ingest smoke 中 `/api/wiki/lint` 可达，但返回 `missing_index`；当前判断是该 smoke 运行在 RAG-only 模式且 `SAC_KG_ENABLED=false`，因此没有生成 wiki index，不是 MinerU 解析失败。
- 本地隔离 RAG-only ingest smoke 中 Ollama 不可用，因此该结果只能证明 ingest/retrieval/citation 最小链路，不能证明最终模型回答质量。
- full30 剩余失败已经不再主要是 parser 或 citation/source 缺失，而是 answer 阶段漏写已经被证据支持的科学锚点。
- citation marker fallback、profile-term same-source expansion、长 chunk anchor windowing 都提高了召回，但也带来 context budget 和弱引用被掩盖的风险，需要继续用 full30 和跨论文测试守住边界。

### 当前剩余失败

- `charmm36m_overview`：answer 缺 `NMR`，citation/source evidence 已命中。
- `ff99sb_ildn_mechanism`：answer 缺 `0.5 kcal/mol`，citation/source evidence 已命中。
- `opls4_overview`：answer 缺 `van der Waals`、`GLH`，citation/source evidence 已命中。
- `opls4_mechanism`：answer 缺 `GLH`，citation/source evidence 已命中。
- `opls5_mechanism`：answer 缺 `FXA`、`-2.4`，citation/source evidence 已命中。

### 排查结论

- 当前 loop 不是前端，也不是完整 agent 系统本体；它是下一步 agent/frontend 的质量控制流水线。
- 现阶段应先保持 ingestion/query loop 的可复跑和失败归因能力，再把这些结果封装到 agent 或前端里展示和触发。
- 借鉴开源项目的时间点应放在失败 taxonomy 稳定之后：先局部参考 RAGFlow/MinerU 的复杂文档解析，kotaemon 的 citation-first PDF QA，R2R/Onyx 的 API-first 检索和评估纪律；暂不整体替换当前平台。

### 验证结果

- 本地 focused pytest：`53 passed`。
- 远端 focused pytest：`53 passed`。
- 公网远端 target6 attribution：`6/6 passed`。
- 公网远端 full30 query gate：`25/30 passed`。
- 公网远端 service ingest gate：`overall_status=passed`，且确认走 `pdf_mineru`。

### 后续计划

- 优先修 answer-stage completeness：当 citation evidence 已包含科学锚点时，最终答案应稳定带出这些锚点，但不能做 benchmark 硬编码。
- 保持 source/citation gate 严格，不能用答案补全文字掩盖弱 citation。
- 继续保留 full30 gate 作为进入 agent/frontend 的门槛。
- 对 OPLS5 Table 7 做更深的表格语义 hardening：区分模型词、宽泛指标词和具体语义锚点，避免普通 RMSE 表仅凭 `OPLS4/OPLS5` 或 `RMSE` 误通过。

## 2026-06-27 工作记录：Agent Layer v1、审查返工与当前限制

### 已完成

- 在现有 FastAPI/RAG 服务上方实现了一个最小但可扩展的 Agent 编排层。
- 新增 `/api/agent/query` endpoint，并通过 `AGENT_ENABLED` 配置开关控制；关闭时返回 503。
- 新增 `ConversationTurn` 表和 `ConversationMemory`，支持按 session 保存多轮对话，并覆盖隔离、截断、持久化等基础场景。
- 新增 `RAGAdapter`，让 Agent 通过稳定接口调用现有 `QueryService.answer(..., save_answer=False)`，不直接耦合 QueryService 内部实现。
- 新增 `ToolRegistry`，提供工具注册、schema 校验、结构化错误返回，并内置只读 `rag.answer` 工具。
- 新增 `AgentExecutor` v1：当前是确定性两步执行，即 `tool_call -> finalize`，为后续模型驱动 tool selection 留接口。
- 新增 Agent Pydantic schemas：`AgentQueryRequest`、`AgentQueryResponse`、`AgentStep`、`AgentUsage`、`ToolSpec`、`AgentConstraints`。
- 在静态前端增加 Agent panel，能够调用 `/api/agent/query` 并展示最终回答、步骤、citation 和 session 信息。
- `.env.example` 与 `.env.server.example` 增加 Agent 相关配置项。
- 新增 Agent 相关测试：
  - `tests/test_conversation_memory.py`
  - `tests/test_rag_adapter.py`
  - `tests/test_tool_registry.py`
  - `tests/test_agent_executor.py`
  - `tests/test_agent_routes.py`

### 审查后返工

- 修复前端 XSS 风险：不再用 `innerHTML` 渲染 `final_answer` 等动态 API 数据，改为 DOM 构造和 `textContent`。
- 修复会话轮次未提交问题：路由执行完成后显式 `db.commit()`，失败时 rollback 并返回 500。
- 清理未使用 imports 和误导性 docstring。
- 增加 executor error path 测试，覆盖工具失败时返回 `error` 状态。
- 增加 executor timeout path 测试，覆盖步数/耗时控制相关行为。
- 增加跨 DB session 的会话持久化测试，确认 Agent turn 能在文件型 SQLite 中跨 session 读取。

### 验证结果

- Agent focused tests：

```text
D:\Miniconda3\python.exe -m pytest tests/test_agent_executor.py tests/test_conversation_memory.py tests/test_tool_registry.py tests/test_rag_adapter.py tests/test_agent_routes.py -q
36 passed
```

- Query/API 回归：

```text
D:\Miniconda3\python.exe -m pytest tests/test_api_routes.py tests/test_query_service.py tests/test_paper_profile.py -q
199 passed
```

### 遇到的问题

- Agent v1 目前仍是确定性执行器，只会调用 `rag.answer`，还不是模型驱动的多工具、多步规划 Agent。
- 当前 Agent frontend 只是最小面板，能跑通调用和展示，但还没有把 loop manifest、query attribution、service ingest gate 等质量报告整合成完整运维视图。
- 目前没有引入 LangGraph、Dify 等外部 Agent 框架；这是有意保持边界清晰，但意味着后续要自己补 policy router、tool selection、streaming、trace persistence 等能力。
- Agent memory 已有基础持久化，但还没有 TTL、清理任务、长期摘要或更细的权限隔离。
- 这一步主要完成本地实现和本地回归，后续仍需要和服务器同步后做远端 API/Agent endpoint 实机验证。

### 当前判断

- Agent Layer v1 已经能作为 RAG 的外层编排入口，但现在的核心价值是“标准化调用、记录步骤、保留 session、为前端和后续多工具扩展留出接口”。
- 下一步不应急着扩复杂 Agent，而应先把 6.26 的 loop 结果接入 Agent/前端：展示 gates、manifest、query attribution、剩余失败和一键触发入口。
- 只有当 loop 展示和远端验证稳定后，再进入模型驱动 tool selection、SSE streaming、更多工具注册和长期 memory。

### 后续计划

- 把 `run_mineru_rag_loop.py` 的 manifest 和 `query_attribution.json` 设计成 Agent/前端可读取的状态源。
- 增加 Agent/前端页面对 full30 gate、service ingest gate、MinerU smoke 和剩余失败归因的展示。
- 远端同步 Agent 相关文件后，验证 `/api/agent/query` 在服务器环境中可用，并确认不会破坏现有 `/api/query`。
- 再下一步才考虑 model-driven tool selection、SSE step streaming、conversation TTL、trace persistence 和更多只读工具。

## 2026-06-27 Agent v2a final acceptance

- Agent v2a 已按当前 local-first online scope 接受。
- codex-with-cc workflow `agent-v2a-20260627` 已通过 final workflow verification。
- 本地测试通过：Agent focused `109 passed`；regression subset `208 passed`。
- 本地浏览器 UI smoke 通过：Agent panel 渲染 route `needs_clarification`、warnings、steps、answer、usage，且无 console error。
- 远端同步前已备份覆盖文件到 `tmp/backup_agent_v2a_20260627_111038`。
- 远端测试通过：Agent focused `109 passed`；regression subset `207 passed`。
- 远端服务已重启并健康：Redis、Ollama、API、worker 均运行。
- 远端 `/api/agent/query` smoke 通过：空白问题返回 200 + `needs_clarification`；非法 `max_steps=0` 返回 422；真实 RAG 查询 `What is CHARMM36m?` 返回 200 + completed Agent/RAG 响应。
- 远端 full30 query gate 通过：`25/30 passed`，5 个失败均归因为 answer-stage missing expected text。
- 已知非阻塞风险：默认 `max_tool_calls=3` 下 retry 不做二次 verify；routing 仍是确定性关键词；executor 还有少量命名/import 清理项。

## 2026-06-28 工作记录：DeepSeek 外部 API、真实前端验证与 RAG Agent 启动流程

### 已完成

- 将外部合成配置从 OpenAI 示例切换为 DeepSeek OpenAI-compatible API：
  - `EXTERNAL_API_ENABLED=true`
  - `EXTERNAL_API_BASE_URL=https://api.deepseek.com`
  - `EXTERNAL_API_MODEL=deepseek-v4-flash` 作为示例默认；服务器实测使用 `deepseek-v4-pro`
  - `AGENT_SYNTHESIS_PROVIDER=auto`
- 服务器 `.env` 已确认存在 DeepSeek key，但记录时只打印 `key_present=True` 与 key 长度，不把密钥内容写入日志或文档。
- 远端 API/worker clean restart 后，真实 Agent 调用返回：
  - `status=completed`
  - `answer_provider=external_api`
  - `answer_model=deepseek-v4-pro`
  - trace 持久化正常
- 使用远端前端 Agent 面板通过 SSE 提交了 full30 之外的新问题：

```text
请用三点概括：如果企业内部知识库的检索结果互相矛盾，Agent 应该如何判断、追问和记录证据？
```

- 前端真实交互结果：
  - `Status: COMPLETED`
  - `Provider: external_api/deepseek-v4-pro`
  - `Trace: 394f3bd8-3ddb-496b-acf6-a1d005d1ba35`
  - `Steps: route -> rag.answer -> answer.synthesize -> answer.verify -> finalize`
  - 页面 console 无应用错误
- 该问题当前知识库检索到的证据与问题主题不相关，DeepSeek 最终回答为“无法基于当前证据回答”。这是合理表现：外部 API 链路已成功，但 Agent 仍遵守 evidence-grounded 原则，没有脱离 RAG 证据自由发挥。
- 修复了本次真实接入暴露出的两个问题：
  - 前端默认请求的 `AgentConstraints.timeout_seconds` 原本固定为 schema 默认 `45` 秒，没有继承服务器 `AGENT_TIMEOUT_SECONDS=90`。DeepSeek v4-pro 慢一些时会导致最终状态被标记为 `timeout`。已改为在 agent route 执行前把未显式传入的 constraint 字段替换为服务器 `AGENT_*` 配置。
  - 外部模型如果返回空 `answer_markdown`，前端会出现“completed 但答案空白”。已改为：外部合成空答案时优先回退到非空 RAG 答案；如果 RAG 也为空，则返回明确的“证据不足”文本和 warning。
- 修复测试隔离问题：服务器 `.env` 有真实 DeepSeek key 后，`tests/test_agent_routes.py` 曾意外调用真实外部 API，导致单测变成非确定性。已在 route 测试 client 中强制使用 local synthesis，避免测试依赖生产密钥。

### 遇到的问题

- **改 `.env` 后不重启不会生效**：`get_settings()` 有缓存，且 API/worker 启动脚本在进程启动时读取 `.env`。服务器加上 DeepSeek key 后，旧 API 进程仍返回 `answer_provider=local`，重启后才切到 `external_api/deepseek-v4-pro`。
- **默认 timeout 与服务器配置不一致**：请求 schema 先把缺省 constraint 填成 `45` 秒，导致路由层看不到“用户未指定 timeout”。这会让慢模型被错误标记为 timeout。当前修复是把 schema 默认值视为“未指定”，在 route 层用服务器配置覆盖。
- **API 响应字段名容易看错**：Agent v3 的最终答案字段是 `final_answer`，不是旧脚本里误读的 `answer_markdown`。脚本读错字段时会误判为“答案为空”。
- **外部模型会严格根据证据拒答**：当 RAG 召回内容与用户问题不相关时，DeepSeek 会输出低置信或拒答。这里不是 API 失败，而是当前知识库证据不支持该问题。
- **真实密钥会污染测试环境**：如果测试直接读取服务器 `.env`，就可能发起真实 API 请求。测试应显式 mock 或强制 local provider。
- **SSH/SCP 有连接限流风险**：短时间多个 SSH/SCP 连接可能被远端拒绝。同步小修复时优先打包成一个 tgz，单连接上传、单连接解包；避免并发 scp。
- **PowerShell 小坑**：`$PID` 是 PowerShell 保留变量，不能用作普通变量名；`Invoke-WebRequest` 在当前环境对本地隧道偶发 .NET 空引用错误，验证健康接口时 `curl.exe` 更稳定。

### 当前判断

- 当前 RAG Agent 已进入“外部合成可用”的状态：本地 RAG/QueryService 仍负责证据检索，DeepSeek 只作为 evidence synthesis provider。
- `AGENT_SYNTHESIS_PROVIDER=auto` 是合适默认值：有 `EXTERNAL_API_ENABLED=true` 且 key 存在时走外部 API；否则自动回退到 local，不会因为 key 缺失导致 Agent 不可用。
- 当前前端 SSE 是 step-level streaming，不是 token-level streaming。页面会先显示运行状态，执行完成后按 step 渲染 trace、warning、final answer、citations 和 usage。
- 如果想让“企业内部知识库矛盾处理”这类泛 Agent 治理问题得到实质回答，需要先 ingest 对应制度/设计文档，或增加一个明确的非 RAG general-answer 工具。现阶段 Agent 的正确行为是“证据不足则拒答/提示不相关”。

### 验证结果

- 本地相关测试通过：

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_agent_synthesizer.py tests/test_agent_routes.py tests/test_agent_streaming.py -q
```

结果：`24 passed`。

- 远端相关测试通过：

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_synthesizer.py tests/test_agent_routes.py tests/test_agent_streaming.py -q
```

结果：`24 passed`。

- 远端 clean restart 后健康检查通过：`curl -fsS http://127.0.0.1:8000/api/health` 返回 `{"status":"ok","app_name":"LLM Wiki Server"}`。
- 远端同步 API smoke 通过：`status=completed`、`answer_provider=external_api`、`answer_model=deepseek-v4-pro`、`trace_id` 非空。
- 远端前端 Agent 面板真实点击通过：结果区渲染 `Status: COMPLETED`、`Trace`、`Provider: external_api/deepseek-v4-pro`、steps、answer、usage。

### 当前 RAG Agent 启动流程

以下流程描述“现在这套 RAG Agent”的启动方式：FastAPI 服务 + worker + 本地 RAG/Ollama 证据链 + 可选 DeepSeek 外部合成 + 前端 Agent 面板。

#### 1. 检查 `.env`

服务器实际 `.env` 不应提交到 git。必要配置如下：

```env
APP_HOST=0.0.0.0
APP_PORT=8000

DATABASE_URL=sqlite:///./data/app.db
REDIS_URL=redis://127.0.0.1:6379/0

QUERY_MODE=rag
VECTOR_STORE_ENABLED=true
VECTOR_STORE_BACKEND=sqlite-vec

OLLAMA_BASE_URL=http://127.0.0.1:11435
OLLAMA_GENERATION_MODEL=qwen3.6:27b
OLLAMA_BATCH_MODEL=qwen3.6:27b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:8b
OLLAMA_REQUEST_TIMEOUT=600

EXTERNAL_API_ENABLED=true
EXTERNAL_API_BASE_URL=https://api.deepseek.com
EXTERNAL_API_KEY=<只写在服务器 .env，不写入仓库>
EXTERNAL_API_MODEL=deepseek-v4-pro
EXTERNAL_API_TIMEOUT=90

AGENT_ENABLED=true
AGENT_MAX_STEPS=8
AGENT_MAX_TOOL_CALLS=5
AGENT_BUDGET_TOKENS=20000
AGENT_TIMEOUT_SECONDS=90
AGENT_SYNTHESIS_PROVIDER=auto
AGENT_CONVERSATION_TTL_DAYS=30
AGENT_TRACE_RETENTION_DAYS=30
AGENT_STREAM_HEARTBEAT_SECONDS=15

QUALITY_REPORTS_DIR=./tmp
```

说明：

- 想省钱或降延迟时可把 `EXTERNAL_API_MODEL` 改成 `deepseek-v4-flash`。
- 修改 `.env` 后必须重启 API 进程；只改文件不会影响已运行进程。
- `AGENT_ALLOW_EXTERNAL_NETWORK=false` 不影响 DeepSeek synthesis。它用于未来更广义的 agent tool 网络访问约束；当前外部合成由 `EXTERNAL_API_*` 控制。

#### 2. 启动服务器服务

在服务器仓库目录执行：

```sh
cd ~/llm_wiki_server

# 可选：查看 Redis/Ollama/API/worker 当前状态
sh scripts/status.sh

# 启动 API
sh scripts/start_api.sh

# 启动 worker
sh scripts/start_worker.sh

# 健康检查
curl -fsS http://127.0.0.1:8000/api/health
```

如果刚改过 `.env`，建议 clean restart：

```sh
cd ~/llm_wiki_server

if command -v lsof >/dev/null 2>&1; then
  for pid in $(lsof -ti:8000 2>/dev/null || true); do
    kill "$pid" 2>/dev/null || true
  done
fi

for pid_file in run/api.pid run/worker.pid; do
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -n "$pid" ]; then
      kill "$pid" 2>/dev/null || true
    fi
  fi
done

rm -f run/api.pid run/worker.pid
sleep 2

sh scripts/start_api.sh
sh scripts/start_worker.sh
curl -fsS http://127.0.0.1:8000/api/health
```

日志位置：

- API：`logs/api.log`
- worker：`logs/worker.log`
- PID：`run/api.pid`、`run/worker.pid`

#### 3. 验证配置是否被进程读到

不要打印 key 内容，只打印是否存在和长度：

```sh
cd ~/llm_wiki_server
PYTHONPATH=src .venv/bin/python -c "from app.core.config import get_settings; s=get_settings(); print('enabled', s.external_api_enabled); print('base', s.external_api_base_url); print('model', s.external_api_model); print('provider', s.agent_synthesis_provider); print('key_present', bool(s.external_api_key)); print('key_len', len(s.external_api_key or ''))"
```

期望：

```text
enabled True
base https://api.deepseek.com
model deepseek-v4-pro
provider auto
key_present True
key_len <非零>
```

#### 4. 后端同步接口 smoke

```sh
curl -fsS http://127.0.0.1:8000/api/agent/query \
  -H 'Content-Type: application/json' \
  -d '{
    "project_slug": "internal-research",
    "session_id": "server-agent-smoke",
    "query": "请用三点概括：如果企业内部知识库的检索结果互相矛盾，Agent 应该如何判断、追问和记录证据？"
  }'
```

重点检查：

- `status` 应为 `completed`、`max_steps` 或明确错误；正常 DeepSeek smoke 期望 `completed`。
- `answer_provider` 应为 `external_api`。
- `answer_model` 应为 `deepseek-v4-pro` 或当前 `.env` 中配置的模型。
- `trace_id` 应非空。
- 如果答案提示“证据不相关/证据不足”，这不代表 API 失败，而是当前 RAG 证据不支持该问题。

#### 5. SSE streaming smoke

```sh
curl -N http://127.0.0.1:8000/api/agent/query/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "project_slug": "internal-research",
    "session_id": "server-agent-stream-smoke",
    "query": "What is CHARMM36m?"
  }'
```

期望事件顺序大致为：

```text
event: start
event: step
event: step
event: warning   # 仅在有 warning 时出现
event: final
event: done
```

`final` 事件中应包含完整 `AgentQueryResponse`，包括 `trace_id`、`answer_provider`、`answer_model`、`steps`、`warnings`、`final_answer`。

#### 6. Trace 查询

拿到 smoke 的 `trace_id` 后：

```sh
curl -fsS http://127.0.0.1:8000/api/agent/traces/<trace_id>
```

按 session 列表查询：

```sh
curl -fsS 'http://127.0.0.1:8000/api/agent/traces?session_id=server-agent-smoke&limit=5&offset=0'
```

期望：

- detail 返回 `status`、`route`、`provider`、`model`、`steps`。
- list 返回当前 session 下的最新 traces。
- trace 中不应包含 API key 或原始 provider 响应密钥信息。

#### 7. 打开远端前端

如果从本地机器访问远端服务器，建议用 SSH tunnel。不要把真实 host、port、key 路径写入仓库文档，命令模板如下：

```powershell
ssh -N `
  -L 127.0.0.1:8011:127.0.0.1:8000 `
  -p <REMOTE_SSH_PORT> `
  -i <SSH_KEY_PATH> `
  <USER>@<REMOTE_HOST>
```

然后打开：

```text
http://127.0.0.1:8011/
```

前端测试步骤：

1. 找到右侧 Agent 面板。
2. `Project Slug` 使用 `internal-research`。
3. `Session ID` 可填一个临时值，例如 `frontend-deepseek-smoke-YYYYMMDD-HHMM`。
4. `Question` 填入 full30 之外的新问题。
5. 点击 `Execute Agent`。
6. 观察结果区是否出现：
   - `Status: COMPLETED`
   - `Provider: external_api/deepseek-v4-pro`
   - `Trace: <uuid>`
   - `Steps`
   - `Answer`
   - `Usage`

关闭隧道：

```powershell
Get-Process ssh -ErrorAction SilentlyContinue
Stop-Process -Id <PID> -Force
```

#### 8. 本地 Windows 调试启动

本地 `.env` 已被 `.gitignore` 忽略，可以放空 key 或填本地测试 key。key 为空时，`AGENT_SYNTHESIS_PROVIDER=auto` 会自动回退到 local。

```powershell
cd D:\LLM_wiki
$env:PYTHONPATH='D:\LLM_wiki\src'
D:\Miniconda3\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

健康检查：

```powershell
curl.exe -fsS http://127.0.0.1:8010/api/health
```

打开：

```text
http://127.0.0.1:8010/
```

本地调试注意：

- 如果没有 Redis 或 worker，本地只测 `/api/agent/query`、SSE、trace UI 通常仍可进行；真实 ingest/后台任务需要 worker 和 Redis。
- 如果没有 Ollama，本地 RAG 质量不能代表服务器真实回答质量。
- 本地 key 为空时看到 `answer_provider=local` 是正常的。

#### 9. 回归测试建议

改 Agent/DeepSeek/route/streaming 后，至少运行：

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_agent_synthesizer.py tests/test_agent_routes.py tests/test_agent_streaming.py -q
```

关键路径改动后再运行：

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_agent_synthesizer.py tests/test_agent_trace_store.py tests/test_agent_streaming.py tests/test_agent_executor.py tests/test_agent_routes.py tests/test_conversation_memory.py tests/test_tool_registry.py -q
D:\Miniconda3\python.exe -m pytest tests/test_rag_adapter.py tests/test_api_routes.py tests/test_quality_routes.py tests/test_quality_reports.py -q
```

远端同步后至少运行：

```sh
cd ~/llm_wiki_server
PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_synthesizer.py tests/test_agent_routes.py tests/test_agent_streaming.py -q
curl -fsS http://127.0.0.1:8000/api/health
```

需要做最重质量门时，再跑 full30 gate，当前可接受门槛仍是：

- `min_query_passed >= 24`
- `max_query_failed <= 6`
- `require_failure_attribution=true`

## 2026-06-28 Phase 2/3 最新进展补充

### 已完成

- Phase 2 `rag-agent-eval-dashboard-20260628` 已完成并远端验收：
  - `QualityReportsService` 增加 `agent_metrics` 汇总。
  - Dashboard 可从 `query_eval.json` / `query_attribution.json` / trace sidecar 读取 query 通过率、失败归因、工具调用、provider/model 分布。
  - 本地与远端 focused tests 均通过。
  - 远端 full30 dashboard smoke 可读出 `25 passed / 5 failed`，失败归因为 `answer` stage。
- Phase 3 `rag-agent-evidence-aware-synthesis-20260628` 已完成并远端验收：
  - `AgentExecutor` 已把 `rag.retrieve_evidence` 返回的 `evidence_pack` 传入 `answer.synthesize`。
  - `ToolRegistry` 的 `answer.synthesize` schema 已支持可选 `evidence_pack`。
  - `AgentSynthesizer` 外部合成 prompt 已加入 compact evidence-pack section，包含 `doc`、`kind`、`stage`、`support_hint`、excerpt，并通过 `MAX_EVIDENCE_PACK_ITEMS=10`、`MAX_EXCERPT_CHARS=300` 控制 prompt 大小。
  - 增加 bounded coverage retry：仅在 evidence-heavy route 且 evidence pack 存在时触发，最多一次。
  - codex-with-cc 链路完成：implementer、prompt rework、spec review、quality review、final-verifier 均为 `DONE`，`verify_delegate_workflow` 通过。

### 本轮遇到的问题

- Codex child thread 多次因为 `agent thread limit reached` 无法创建；按 codex-with-cc 契约使用了 trusted local terminal fallback，但仍保留 `CODEX_CLAUDE_CHILD_THREAD=1`、TaskFile、WorkflowId、TaskId、SessionKey、Scope、review metadata。
- 初版 Phase 3 worker 漏掉了 prompt 中的 `support_hint`，并且 evidence pack prompt 没有显式 item/excerpt 上限；已通过同一 implementer TaskId 的 rework 修复。
- 远端 SSH 偶发 `Connection closed by ... port 28294`，重连后可继续执行，full30 artifact 未受影响。
- 远端 shell / PowerShell 引号对 `pytest -k "table or evidence or rag_context"`、heredoc JSON 解析不稳定；后续远端复杂命令优先使用脚本文件或 POSIX 安全引号。

### 当前验证结果

本地：

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_agent_synthesizer.py tests/test_tool_registry.py tests/test_agent_executor.py tests/test_agent_streaming.py -q
# 99 passed

D:\Miniconda3\python.exe -m pytest tests/test_agent_routes.py tests/test_agent_trace_store.py tests/test_conversation_memory.py -q
# 33 passed

D:\Miniconda3\python.exe -m pytest tests/test_rag_adapter.py tests/test_quality_reports.py tests/test_quality_routes.py -q
# 63 passed

D:\Miniconda3\python.exe -m pytest tests/test_query_service.py -q -k "table or evidence or rag_context"
# 109 passed, 78 deselected
```

远端：

```text
agent core: 99 passed
agent routes/trace/memory: 33 passed
rag adapter/quality reports/routes: 63 passed
query/evidence focused: 109 passed, 78 deselected
API/worker restart: api_pid 2046358, worker_pid 2046376
agent smoke: sync 200 completed, provider external_api/deepseek-v4-pro
SSE smoke: 200 text/event-stream, events start/step/final/done
trace detail: 200 completed
```

远端 full30：

```text
out_dir: tmp/full30_query_gate_evidence_aware_synthesis_20260628
overall_status: passed
total: 30
completed: 30
passed: 25
failed: 5
failure_reason_counts: {"missing_expected_answer_text": 5}
failed_likely_stage_counts: {"answer": 5}
failed ids:
- charmm36m_overview
- ff99sb_ildn_mechanism
- opls4_overview
- opls4_mechanism
- opls5_mechanism
```

结论：Phase 3 没有让 full30 退步，满足当前阶段 gate；但 5 个 answer-stage 失败没有减少。下一轮若目标是从 `25/30` 提到 `27/30`，应继续聚焦 answer-stage 的 query-aware anchor selection / expected-term-like generic coverage，而不是改 retrieval 或 benchmark。

### 当前远端启动与重测快捷流程

远端服务目录：

```sh
cd ~/llm_wiki_server
```

clean restart：

```sh
sh tmp/remote_restart_api_worker.sh
```

健康检查：

```sh
curl -fsS http://127.0.0.1:8000/api/health
```

Agent smoke：

```sh
PYTHONPATH=src .venv/bin/python tmp/remote_agent_evidence_smoke.py
```

full30 gate：

```sh
sh tmp/remote_full30_evidence_aware_synthesis_20260628.sh

while [ ! -f tmp/full30_evidence_aware_synthesis_20260628.exit ]; do
  sleep 30
done

cat tmp/full30_evidence_aware_synthesis_20260628.exit
ls -lah tmp/full30_query_gate_evidence_aware_synthesis_20260628
```

## 2026-06-28 Phase 4 最新进展补充

### 已完成

- Phase 4 `rag-agent-controlled-complex-agent-20260628` 已完成并远端验收：
  - 新增 `complex_multi_hop` route，用于明显复杂、多步骤、多问题查询。
  - 新增 `ComplexPlan` schema，作为 deterministic / auditable plan metadata。
  - `AgentExecutor` 只在 `complex_multi_hop` 时插入 `plan` step；普通 `simple_rag` 不插入 plan。
  - plan metadata 包含：
    - `plan_type=controlled_complex`
    - `allowed_tools=["rag.retrieve_evidence","rag.answer","answer.synthesize","answer.verify"]`
    - `forbidden_tools=["shell","sql","write","network","file_system_write","arbitrary_code_execution"]`
    - `max_steps`
    - `max_tool_calls`
    - `subtasks`
  - 没有引入 LangGraph / LlamaIndex / 新依赖，也没有开放 shell、SQL、写文件或任意网络工具。
  - codex-with-cc 链路完成：implementer、spec review、quality review、final-verifier 均为 `DONE`，`verify_delegate_workflow` 通过。

### 本轮验证结果

本地：

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_agent_policy.py tests/test_agent_executor.py tests/test_agent_streaming.py -q
# 111 passed

D:\Miniconda3\python.exe -m pytest tests/test_agent_routes.py tests/test_agent_trace_store.py tests/test_conversation_memory.py -q
# 33 passed

D:\Miniconda3\python.exe -m pytest tests/test_agent_synthesizer.py tests/test_tool_registry.py tests/test_agent_executor.py tests/test_agent_streaming.py -q
# 105 passed

D:\Miniconda3\python.exe -m pytest tests/test_rag_adapter.py tests/test_quality_reports.py tests/test_quality_routes.py -q
# 63 passed

D:\Miniconda3\python.exe -m pytest tests/test_query_service.py -q -k "table or evidence or rag_context"
# 109 passed, 78 deselected
```

远端：

```text
agent policy/executor/streaming: 111 passed
agent routes/trace/memory: 33 passed
synthesis/tool/executor/streaming: 105 passed
rag adapter/quality reports/routes: 63 passed
query/evidence focused: 109 passed, 78 deselected
API/worker restart: api_pid 2071499, worker_pid 2071517
ordinary agent smoke: provider external_api/deepseek-v4-pro, trace detail 200
complex plan smoke: route complex_multi_hop, steps route/plan/retrieve/tool_call/synthesis/tool_call/finalize
```

远端 Phase 4 full30：

```text
out_dir: tmp/full30_query_gate_controlled_complex_agent_20260628
overall_status: passed
total: 30
completed: 30
passed: 25
failed: 5
failure_reason_counts: {"missing_expected_answer_text": 5}
failed_likely_stage_counts: {"answer": 5}
failed ids:
- charmm36m_overview
- ff99sb_ildn_mechanism
- opls4_overview
- opls4_mechanism
- opls5_mechanism
```

结论：Phase 4 已把复杂 Agent 的受控 plan 能力接入 route、trace、SSE 和远端服务；普通 benchmark 未被误伤，full30 仍保持 `25/30`。

### Phase 4 残余 follow-up

- `ComplexPlan.max_steps` 可后续加 `gt=0`，保持和 `AgentConstraints` 的 schema 风格一致；当前 executor 只传入已经验证过的 positive constraint，因此不是阻塞。
- 如果中文复杂问题常见，可补充“第一步/第二步”这类中文 step pattern；当前已覆盖英文 complex terms 和 `1.` / `2.` 编号式多问题。

## 2026-06-28 Phase 5/6 最新进展补充

### Phase 5 已完成：Streaming / Trace / TTL Enterprise Hardening

- Workflow：`rag-agent-enterprise-hardening-20260628`。
- codex-with-cc 链路完成：
  - implementer：`20260628_151529_601_22f66b77`，`DONE`。
  - spec review：`20260628_152752_038_e29f8cc1`，`DONE`。
  - quality review：`20260628_153059_038_4ace0def`，`DONE`。
  - final verifier：`20260628_153400_847_831e0394`，`DONE`。
  - `verify_delegate_workflow` 通过。
- SSE streaming：
  - `/api/agent/query/stream` 新增稳定 `heartbeat` 事件，当前为同步 executor 调用前发送一次。
  - `error` event 统一包含 `message` 与 `error_type`。
  - `final` event 保留完整原响应，并新增顶层 `trace_id`、`provider`、`model`、`tool_names`、`step_summary`。
- Trace persistence/listing：
  - `/api/agent/traces` 支持按 `session_id`、`project_slug`、`status`、`provider`、`route` 组合过滤。
  - 只传 `session_id` 的旧调用保持兼容。
  - trace 序列化会对 constraints 与 step metadata 做敏感字段/敏感值脱敏。
- Conversation/Trace TTL：
  - `main._startup_purge()` 已在启动时调用 conversation session 与 trace retention 清理。
  - 新增 `tests/test_app_startup.py` 覆盖成功清理、异常日志、rollback 失败也不阻断启动、lifespan 调用。

### Phase 5 本地验证

```powershell
D:\Miniconda3\python.exe -m pytest tests/test_agent_streaming.py tests/test_agent_routes.py tests/test_agent_trace_store.py tests/test_conversation_memory.py tests/test_app_startup.py -q
# 62 passed

D:\Miniconda3\python.exe -m pytest tests/test_agent_executor.py tests/test_agent_policy.py tests/test_agent_synthesizer.py tests/test_tool_registry.py -q
# 160 passed

D:\Miniconda3\python.exe -m pytest tests/test_rag_adapter.py tests/test_quality_reports.py tests/test_quality_routes.py -q
# 63 passed

D:\Miniconda3\python.exe -m pytest tests/test_query_service.py -q -k "table or evidence or rag_context"
# 109 passed, 78 deselected
```

### Phase 5 远端验证

同步包：

```text
tmp/rag-agent-enterprise-hardening-sync-20260628.tgz
```

远端备份目录：

```text
tmp/backup_rag_agent_enterprise_hardening_20260628_153950
```

远端测试：

```text
agent streaming/routes/trace/memory/startup: 62 passed
executor/policy/synthesizer/tool_registry: 160 passed
rag adapter/quality reports/routes: 63 passed
query/evidence focused: 109 passed, 78 deselected
```

远端服务：

```text
API restart: api_pid 2099163
worker restart: worker_pid 2099177
health: {"status":"ok","app_name":"LLM Wiki Server"}
```

远端 smoke：

```text
remote_agent_evidence_smoke:
- sync 200 completed
- provider external_api/deepseek-v4-pro
- SSE events include start/heartbeat/step/final/done
- trace detail 200 completed

remote_complex_plan_smoke:
- route complex_multi_hop
- steps route/plan/retrieve/tool_call/synthesis/tool_call/finalize
- allowed tools rag.retrieve_evidence/rag.answer/answer.synthesize/answer.verify
- forbidden tools shell/sql/write/network/file_system_write/arbitrary_code_execution

remote_enterprise_hardening_smoke:
- stream 200 text/event-stream
- events include start/heartbeat/final/done
- final includes trace_id/provider/model/tool_names/step_summary
- trace list filters returned the new trace
- trace detail 200 completed
```

远端 full30：

```text
out_dir: tmp/full30_query_gate_enterprise_hardening_20260628
exit_code: 0
total: 30
selected: 30
completed: 30
passed: 25
failed: 5
failure_reason_counts: {"missing_expected_answer_text": 5}
failed_likely_stage_counts: {"answer": 5}
failed ids:
- charmm36m_overview
- ff99sb_ildn_mechanism
- opls4_overview
- opls4_mechanism
- opls5_mechanism
```

结论：Phase 5 没有让普通 query benchmark 退步；full30 仍保持 `25/30`，5 个失败仍集中在 answer-stage expected text 覆盖不足。

### 本轮遇到的问题

- Codex child thread 仍然达到 `agent thread limit reached`，因此按 codex-with-cc 的 trusted local terminal fallback 执行，但保留 `CODEX_CLAUDE_CHILD_THREAD=1`、TaskFile、WorkflowId、TaskId、Role、SessionKey、ReviewKind 等元数据。
- 直接调用 `delegate_to_claude.ps1` 没有 child-thread marker 会被插件拒绝；fallback 必须显式设置 `$env:CODEX_CLAUDE_CHILD_THREAD='1'`。
- 远端 SSH/SCP 偶发 `Connection closed by ... port 28294`，重试可恢复；`scp` 默认模式不稳定时，使用 `scp -O` legacy 模式成功。
- PowerShell 调远端 shell 时，`$(date ...)` 会被本地 PowerShell 抢先展开；远端复杂命令优先使用固定文件脚本或单引号。
- `pytest -k "table or evidence or rag_context"` 在远端引号容易被拆开；稳定写法是外层双引号、远端 `-k 'table or evidence or rag_context'`。
- Python 3.12 远端会提示 `datetime.utcnow()` deprecation warning；不影响本轮验收，后续可统一迁移到 timezone-aware UTC。
- reviewer 标注的非阻塞 follow-up：`agent_routes.py` 里 `asyncio` 未使用、`test_app_startup.py` 里 `MagicMock` 未使用；后续 cleanup 可顺手移除。

### Phase 6 发布门禁已固定

当前不需要新增脚本：`scripts/run_mineru_rag_loop.py` 已经支持 query profile、full30、阈值和 failure attribution；`scripts/query_eval.py` 与 `scripts/query_report_summary.py` 已经负责 case 执行和分层归因。

标准定义位置：

```text
benchmark cases: benchmarks/query/internal_research_v1.json
query executor: scripts/query_eval.py
failure attribution: scripts/query_report_summary.py
loop/gate runner: scripts/run_mineru_rag_loop.py
dashboard parser: src/app/services/quality_reports.py
```

本阶段发布门禁：

```text
Local focused gate:
- Agent streaming/routes/trace/memory/startup tests pass
- executor/policy/synthesizer/tool_registry tests pass
- rag_adapter/quality_reports/quality_routes tests pass
- query_service table/evidence/rag_context focused tests pass

Remote smoke gate:
- /api/health returns ok
- /api/agent/query returns completed or timeout with trace_id
- /api/agent/query/stream returns text/event-stream
- SSE includes heartbeat and final
- trace detail and trace list filters work
- complex_multi_hop still emits plan step and tool allow/forbid metadata

Remote full30 gate:
- selected = 30
- completed = 30
- passed >= 25 for current release baseline
- failed <= 5 for current release baseline
- transition compatibility line may keep max_query_failed=6 during iteration
- require_failure_attribution = true
- unattributed failures = 0
```

远端快捷命令：

```sh
cd ~/llm_wiki_server
sh tmp/remote_restart_api_worker.sh
curl -fsS http://127.0.0.1:8000/api/health
PYTHONPATH=src .venv/bin/python tmp/remote_agent_evidence_smoke.py
PYTHONPATH=src .venv/bin/python tmp/remote_complex_plan_smoke.py
PYTHONPATH=src .venv/bin/python tmp/remote_enterprise_hardening_smoke.py
sh tmp/remote_full30_enterprise_hardening_20260628.sh
```

full30 完成后检查：

```sh
cat tmp/full30_enterprise_hardening_20260628.exit
cat tmp/full30_query_gate_enterprise_hardening_20260628/query_eval.md
cat tmp/full30_query_gate_enterprise_hardening_20260628/query_attribution.json
```

### 下一轮最值得做的优化

当前企业级链路已经闭环，下一轮不应优先扩框架，而应直接打剩余 5 个 answer-stage failure：

- 增强 synthesis 的 query-aware anchor selection，让答案更稳定覆盖 benchmark 中要求的具体术语/数值，例如 `NMR`、`0.5 kcal/mol`、`van der Waals`、`GLH`、`FXA`、`-2.4`。
- 保持 retrieval/source/citation gate 不放松，不改 benchmark，不用硬编码 case 答案。
- 目标从 `25/30` 提升到 `27/30`，并把 answer-stage failure 从 5 降到 3 或更少。

## 2026-07-06 聊天模块隔离与临时附件最新进展补充

### 今日目标

- 回顾 6.27 之后到当前窗口的项目上下文，补齐本轮前端/后端重构进展、同步状态、验证结果和遇到的问题。
- 围绕用户提出的“每个专题卡片进入独立聊天，不被其他卡片知识污染”的目标，落实会话级知识隔离、聊天历史入口、Agent 过程展示和临时文件上传能力。
- 完成本地、GitHub、服务器三端同步，并确认当前是否具备并发操作基础。

### 今日完成

#### 1. 代码/配置变更

- 已完成一次可部署提交：`4b9caea Implement dark chat with session attachments`。
- 后端新增会话级临时附件能力：
  - 新增 `SessionAttachment`、`SessionAttachmentChunk` 数据表模型。
  - 新增会话附件上传、列表、删除接口。
  - 临时附件按 `project_slug + session_id` 隔离，只在当前会话内参与检索与回答。
  - `AgentExecutor` 会先检索正式 RAG 证据，再合并当前会话附件证据；当正式 RAG 不足但附件证据存在时，可以基于附件生成确定性摘要式回答。
  - 会话过期清理时同步删除附件记录与落盘文件。
- 前端完成聊天界面重构：
  - 左侧侧边栏下半区放置历史会话列表，支持点击恢复之前的聊天。
  - 聊天主界面改为更接近图三的大面积深色对话布局。
  - 输入框改为底部大 composer，支持 Enter 发送、Shift+Enter 换行。
  - 发送成功后清空输入框，提示语保持为“问点难的，让我多想一步”。
  - 新增加号按钮，可给当前会话添加临时文件；文件只影响该会话。
  - 新增公开 Agent 过程展示区，展示 route/retrieve/synthesize/verify 等工具步骤摘要，但不暴露模型私有 CoT。
- 服务器连接方式已稳定：
  - SSH alias：`llm-wiki-server`
  - 实际目标：`zhangyh@192.168.31.20`
  - 私钥：`D:/codex_ssh/llm_wiki_server_ed25519`
  - 服务器项目路径：`/home/zhangyh/llm_wiki_server`

#### 2. 验证结果

- 本地测试：

```powershell
D:/Miniconda3/python.exe -m pytest tests/test_session_attachments.py tests/test_agent_executor.py tests/test_agent_routes.py tests/test_agent_streaming.py tests/test_static_frontend.py -q
# 108 passed in 78.63s
```

- 服务器测试：

```text
tests/test_session_attachments.py + tests/test_static_frontend.py
# 30 passed, 117 warnings in 1.93s
```

- 服务器运行状态：

```text
API: 0.0.0.0:8000
worker: running
redis: 127.0.0.1:6379
ollama: http://127.0.0.1:11435
health: {"status":"ok","app_name":"LLM Wiki Server"}
```

- 数据库检查：

```text
session_attachments
session_attachment_chunks
```

- 浏览器验证：
  - 深色聊天界面已渲染。
  - 左侧历史会话区域可见。
  - 底部大输入框可见。
  - Enter 可以发送。
  - 发送后输入框清空。
  - 提示语显示“问点难的，让我多想一步”。
  - Agent 过程步骤会显示在 `thinkingFeed`。

- 三端同步：
  - 本地当前提交：`4b9caea`
  - GitHub 已 push 成功。
  - 服务器已 `git pull --ff-only origin main` 到 `4b9caea`。
  - 服务已重启并通过健康检查。

#### 3. 遇到的问题

- 用户最初在运行页看到的文档标题仍有样例化/slug 化痕迹，例如 `8b4b...-sample`，这说明运行追踪页仍需继续核对“真实标题字段”的来源和展示优先级。
- 运行追踪页显示数量曾只有 2 篇，而用户预期应有 10 篇；本轮主要收敛聊天与会话附件，运行列表的真实数量统计、分页/过滤和项目作用域仍需要单独复查。
- 专题卡片最初统一显示 `Internal Research`，与“上传者希望归入哪个专题就进入哪个专题”的需求不一致；当前会话隔离能力已具备，但专题创建、专题选择和卡片聚合语义还需要继续产品化。
- “明明右侧有原文，但 Agent 回答说没有正文”的根因，是旧链路没有把当前会话/当前卡片里的临时或局部文章证据稳定注入 Agent 检索上下文；本轮通过 session attachment retrieval 解决了会话临时材料的注入问题。
- 前端重构过程中发现 `loadSessionAttachments is not defined`，已补齐会话切换时的附件加载函数并通过浏览器回归。
- codex-with-cc 子线程多次触发 `agent thread limit reached`，因此按插件约定使用 trusted local terminal fallback，并保留 WorkflowId/TaskId/Role/Scope 等元数据完成实现、复核和 final verifier。
- 前端质量复核曾指出长文本可能溢出、输入框缺少可访问性标签；已补充 `overflow-wrap` 与 `aria-label`。
- GitHub 远端查询曾偶发 TLS EOF，但 push 成功且服务器能从 GitHub 拉到 `4b9caea`，因此不影响本轮同步结论。
- 服务器脚本缺少可执行位时，使用 `sh scripts/start_api.sh`、`sh scripts/start_worker.sh`、`sh scripts/status.sh` 可以稳定启动；暂未改动服务器历史遗留的未跟踪文件。
- 当前仓库本地仍有未提交的本地工件：`.codex/`、`output/`；这些不是本轮功能代码，未纳入提交。

#### 4. 解决方案/结论

- 当前已经具备“会话级知识隔离”的核心基础：聊天历史、临时附件、附件分块、附件检索和 Agent 证据合成都按 session/project scoped。
- 当前支持并发操作的基础，但不能直接宣称已经完成生产级高并发：
  - API 层、会话表、附件表和 `session_id` 隔离支持多个会话同时存在。
  - 不同会话的临时附件不会互相污染。
  - Redis worker 架构支持后台任务队列。
  - 但当前服务器仍是 SQLite、本地模型和有限 worker 资源，真实多用户高并发上传/解析/问答还需要压力测试和 worker 并发策略确认。
- 当前“公开 Agent 思考过程”采取的是安全实现：展示工具调用与阶段摘要，不展示模型私有 chain-of-thought。
- 当前更适合定义为“专题独立对话与临时文件 RAG 的可用版本”，下一步应继续补文档库真实上传闭环、运行页真实数据和专题归属流程。

### 关键文件

- `src/app/models/records.py`
- `src/app/schemas/agent.py`
- `src/app/services/session_attachments.py`
- `src/app/services/agent_executor.py`
- `src/app/services/conversation_memory.py`
- `src/app/api/agent_routes.py`
- `src/app/static/index.html`
- `tests/test_session_attachments.py`
- `tests/test_static_frontend.py`

### 当前结论

- 本轮最重要的交付不是单纯改 UI，而是把“一个卡片/一个专题/一个会话内的知识不污染其他会话”这条主线落到了后端数据模型、API、Agent 检索和前端交互里。
- 本地、GitHub、服务器已经对齐到 `4b9caea`，服务器服务已启动并可通过本地转发测试。
- 会话级临时文件已经具备接口和 UI 入口，适合下一轮用真实 PDF/Markdown 做端到端浏览器验收。

### 明日/下一步建议

1. 用真实 PDF 在浏览器里走一遍“专题卡片进入对话 -> 加号上传临时文件 -> 提问 -> 引用临时文件回答”的端到端验收。
2. 单独修运行页：真实文章标题、真实文档数量、运行记录分页/过滤、文档与专题作用域。
3. 单独修文档库：上传后解析 PDF/表格/图片说明并生成完整 Markdown，再切块入库，确保与历史 pipeline 一致。
4. 把专题卡片的“进入对话”路径做成正式产品路径：每个专题有自己的默认 session、历史 session 和隔离证据池。
5. 做一次轻量并发验证：两个浏览器会话同时上传不同临时文件并提问，确认回答不会串证据。

### 备注

- 服务器快捷连接：

```powershell
ssh llm-wiki-server
```

- 如需本地访问服务器服务，仍可使用端口转发；若 `8011` 绑定失败，优先检查本地端口占用或换用新端口。

## 2026-07-13 项目进展与问题汇总

### 1. 当前总体进度

- 当前 Git 基线为 `35aaf02 Improve evidence-backed query answer completion`，远端 `origin/main` 与本地提交基线一致。
- full30 的上一轮稳定结果为 `25/30`。已提交的 answer-stage completeness 改进采用通用证据机制：从当前检索 evidence/context/citation 中识别缩写、专业短语、数值和单位等科学锚点，在答案遗漏且证据明确支持时补充，并保持合法引用。
- 运行时代码未写入 benchmark case id、expected terms、论文 slug 或固定失败样例答案；benchmark 仍只用于评估。
- 当前工作区存在一批尚未提交的后续功能修改，覆盖单篇文献对话、真实文献标题、PDF 原文访问、专题/文献/会话删除、作用域隔离及对应测试。这些改动需要完成整体验收后再形成正式提交。

### 2. 已完成或已落地的功能

#### 2.1 非硬编码答案完整性改进

- `src/app/services/search.py` 已实现基于当前证据的通用锚点补全，不从 benchmark expectation 反向决定答案。
- 补全逻辑要求锚点来自本次有效上下文，并与问题、证据及引用相关；不能跨文档借词，也不能把只出现在 query、但未出现在 evidence 的词补入答案。
- `tests/test_query_service.py` 已增加缩写、数值/单位、专业短语、无证据不补、多来源引用隔离等通用测试。

#### 2.2 单篇文献作用域对话

- Query、RAG adapter、Agent tool、Agent executor、会话模型和前端请求均已增加可选 `document_id`。
- 用户可从单篇文章进入对话；该会话绑定 `project_slug + document_id`，检索和引用被限制在该文献范围内。
- 专题级会话与单篇文献会话分开列出。已有 session 不能被静默重绑定到另一个专题或文献，避免历史对话串库。
- 会话恢复时会恢复文献作用域和真实标题；在单篇文献对话中点击“新聊天”会保留当前文献范围。

#### 2.3 来源标题与引用展示

- RAG 与 Agent 返回内容不再依赖面向用户的技术路由标签来说明来源。
- 前端正文会去除数字 citation marker，并单独渲染“来源文献”列表，优先展示数据库中的真实文献标题。
- citation 仍保留 `document_id` 等结构化字段用于归属和跳转，展示层只呈现用户可理解的真实标题。
- 新增 `scripts/backfill_document_titles.py`，用于从已有元数据回填历史文档标题；脚本保持 source slug 稳定，不以特定论文名硬编码。

#### 2.4 PDF 原文回溯

- 后端新增 `GET /api/documents/{document_id}/file`，按项目作用域校验后返回数据库所记录的原始文件。
- 当原文件是 PDF 时，前端原文抽屉直接嵌入 PDF，并支持页码定位及新窗口打开，不再把 MinerU/Markdown 解析文本作为主要阅读界面。
- 若原文件不存在、路径越界或项目不匹配，接口返回受控错误，不暴露任意服务器文件。

#### 2.5 删除能力与数据清理

- 已增加专题、文献、Agent 会话的删除入口和后端 `DELETE` 接口，并在 UI 中加入确认步骤。
- 删除会话时同步清理 turns、附件、附件分块和相关 trace。
- 删除文献时同步清理向量索引、文档分块、运行记录、claims/reviews、关联问答、文献会话、相关 trace、只属于该文献的 wiki source page，以及未被其他文档共用的原始文件。
- 删除专题时清理其文献及关联数据；文件删除前会检查是否仍被其他 Document 引用。

### 3. 最近故障、排查与结论

#### 3.1 页面“被清空、按钮点不了”

- 现象：`8011` 页面统计变为 0、专题卡片不渲染、按钮全部无法点击，看起来像数据被删除。
- 根因：`src/app/static/index.html` 中 `renderRuns` 的函数声明丢失，后续 `return` 落到函数外，引发浏览器 `SyntaxError: Illegal return statement`。整个内联脚本停止执行，因此数据加载和事件绑定都没有发生。
- 修复：恢复 `function renderRuns(runs) { ... }` 包装，并新增使用 `node --check` 的静态回归测试，防止缺括号、缺函数声明等语法错误再次让整个页面失效。
- 数据结论：数据库没有被清空。排查时 `agent` 专题仍返回 2 篇文档/2 条运行，`internal-research` 仍返回 10 篇文档/10 条运行。`Agent` 作为大小写不同的 slug 查询会返回 0，实际 slug 是小写 `agent`。

#### 3.2 本地与服务器同步容易产生误判

- 本地没有完整数据分块、Ollama 和真实运行数据，因此只靠本地单元测试不能证明真实问答效果；涉及检索质量、模型回答和数据库内容时，必须连接内网服务器验证。
- 当前采用服务器 API/worker/Redis/Ollama 提供真实能力，本地通过 `127.0.0.1:8011` SSH 转发访问。`8012` 按用户要求关闭，不作为当前入口。
- 最近的前端语法修复已同步到服务器，并确认服务器静态测试 `46 passed`；通过 `8011` 读取的 HTML 已包含修复后的 `renderRuns`。
- 但当前大量后续功能仍处于未提交工作区状态。此前执行的是选择性文件同步，不应把“服务器已有部分文件”误认为“本地、GitHub、服务器已形成同一个正式提交”。下一次发布必须重新核对 Git commit、服务器文件哈希、数据库迁移和服务进程版本。

#### 3.3 标题与历史数据问题

- 历史数据中部分 `Document.title` 仍可能是文件名、UUID 或 slug，导致界面即使按 title 展示也不是真实论文标题。
- 需要先以通用元数据优先级回填真实标题，再验证新 ingest 是否从 PDF 元数据/解析结果持续写入正确标题。
- 标题回填必须与 source slug 解耦，避免改显示标题时破坏已有 wiki 路径、引用或检索索引。

#### 3.4 删除操作的风险

- 删除已从“只删卡片”升级为硬删除关联数据，影响范围较大，必须继续验证事务回滚、共享原文件保护、wiki 多来源页面更新、trace/answer 引用清理和删除后的统计刷新。
- UI 确认框只能降低误操作概率，不能替代后端作用域校验。所有删除接口仍需严格校验 `project_slug` 与资源归属。

### 4. 当前验证状态

- 已知通过：前端内联脚本 `node --check`、本地 `tests/test_static_frontend.py`（46 passed）、服务器 `tests/test_static_frontend.py`（46 passed）、本地 `compileall`、`git diff --check`。
- 已完成浏览器/API 核验：`internal-research` 显示 10 篇/10 条运行，`agent` 显示 2 篇/2 条运行；“查看原文”和“对话”按钮可点击。
- 尚不能宣称整批未提交功能已完成发布验收：需要跑 Query、Agent、删除级联、会话附件、文档作用域和 API 的组合回归，并在服务器真实数据上做端到端验证。
- full30 在非硬编码 completeness 提交后的目标仍为至少 `27/30`，但当前上下文没有可信的新一轮 full30 结果，因此仍以最近确认的 `25/30` 为记录值。

### 5. 下一阶段建议顺序

1. 跑完整相关回归，重点覆盖文档级 RAG/Agent 隔离、会话恢复、附件作用域、三个删除接口和 PDF 文件访问安全。
2. 在服务器备份数据库后，用真实专题、真实 PDF 和真实会话做端到端验收；逐项确认删除后的文件、向量、wiki、run、trace、turn 和 attachment 均符合预期。
3. 执行标题回填 dry-run，抽查真实论文标题与 source slug 未发生耦合变化，再正式回填。
4. 将本地改动形成明确提交并推送，服务器按提交同步和重启；记录本地、GitHub、服务器三端 commit 与服务状态。
5. 重新运行 full30，验收 `passed >= 27`、`failed <= 3`、失败均有 attribution，且 source/citation gate 不退化。
