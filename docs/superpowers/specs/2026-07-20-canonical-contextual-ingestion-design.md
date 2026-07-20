# Canonical Contextual Ingestion Upgrade Design

**日期：** 2026-07-20

**代码基线：** `codex/internal-pilot` / `e40425c`

**状态：** 待书面审阅

## 1. 背景

当前文档入库链路已经支持 PDF、DOCX、HTML、TXT/Markdown，能够生成 `DocumentChunk`、Ollama embedding 和 pgvector 索引，并保留表格、图片、公式等部分结构化信息。但现有实现存在几类直接影响 RAG 精度和后续演进的问题：

- PDF 在进入 MinerU 前强依赖 pypdf 文本层，pypdf 单页失败会阻断后续解析器。
- MinerU block、Markdown 补表和 Document Intelligence chunk 存在 `[:4000]` 静默截断。
- 普通文本主要按约 1200 字符合并，没有 token-aware、语义边界或 parent-child 关系。
- 结构块类型依赖 `heading` 字符串判断，缺少稳定的 `block_type` 和来源定位。
- 表格主要以 Markdown 保存，缺少可验证的 cell、rowspan、colspan 和跨页关系。
- 图片和公式的原始证据、AI 分析与检索文本没有清晰分层。
- 解析结果没有不可变的 canonical 版本，重新解析可能覆盖旧结果。
- embedding 直接基于原始 chunk，没有为每个分块添加文档级关系说明。
- DOCX、HTML、TXT/Markdown 的解析能力明显弱于 PDF。

本次升级的目标是建立一个面向所有现有格式的 canonical ingestion pipeline，以 MinerU 为 PDF 主解析器，使用章节结构、语义切割、parent-child chunk、批量 contextualization 和 contextualized embedding 提升召回精度，同时保证原文引用、表格数值、图片、公式和定位信息可审计。

## 2. 已确认决策

本设计基于以下已经逐项确认的约束：

1. PDF 使用 MinerU 作为主解析器。
2. MinerU 失败或质量不合格时，Document Intelligence 负责局部修复或整篇 fallback；pypdf 文本层是最终兜底。
3. 主要输入是出版社或论文官网下载的数字版 PDF，OCR 不作为常规路径。
4. Chunk 使用 parent-child 结构。
5. 章节边界优先，章节内部使用 `qwen3-embedding:4b` 做 embedding-based semantic splitting，并由 token 上下限保护。
6. 每个可检索 Child 都必须由 LLM 生成 1～2 句 contextual prefix。
7. contextual prefix 批量生成，每批默认 12 个 Child，模型当前使用 `qwen3.5:9b`，但必须独立配置。
8. contextual prefix 只参与 embedding，不作为论文原文、citation 或用户可见证据。
9. 正式 embedding 输入为 `contextual_prefix + Child 原文`。
10. contextual prefix 完整率必须为 100%；失败时不允许退化为原文 embedding。
11. Figure 和 Formula 的 AI 深度分析允许失败，但必须记录 warning；原始结构、contextual prefix 和 embedding 仍必须成功。
12. 表格结构损坏必须修复；修复失败会阻止文档进入 `ready`。
13. 只对异常或低置信度表格调用视觉复核。
14. 所有 citation 必须支持打开原文件并定位到段落、表格、图片或公式区域。
15. 每次成功解析生成不可变 canonical 版本；新版全部成功后才激活。
16. 前端只展示当前版本，不提供历史版本和用户回滚。
17. 前端提供查看和下载 `canonical.md` 的入口。
18. canonical Markdown 嵌入原始 Figure 和 caption；AI 图片分析单独保存。
19. canonical Markdown 保存原始 LaTeX 公式和原文说明；AI 公式分析单独保存。
20. References/Bibliography 保留在 canonical 文档中，但默认不进入普通 RAG。
21. Appendix 正常参与 parent-child、contextualization 和 RAG。
22. 升级覆盖现有 PDF、DOCX、HTML/HTM、TXT 和 Markdown，不新增 PPTX/XLSX。
23. 所有格式的表格、图片和公式采用同一结构化、contextualization 和引用原则。
24. 入库继续使用 Redis 后台异步任务，并拆成可断点恢复的阶段。
25. MinerU、contextualization 和 embedding 的 GPU 并发限制为 1。
26. 历史文档全部重新解析并重建索引。
27. 全量迁移期间测试环境暂停使用；全部文档成功后才进入测试。
28. 新版验收成功后删除旧 Chunk、旧向量、旧解析数据和离线旧数据备份；原始文件必须保留。
29. 新 canonical 包生成后删除 MinerU 临时输出。
30. 所有文档重建成功后运行现有 full30 和新增专项题集，全部通过后才重新开放测试环境。

## 3. 方案比较

### 3.1 方案 A：只启用 MinerU，保留现有 Chunk

优点：改动小，部署快。

缺点：`[:4000]`、固定字符切分、弱结构模型和直接 embedding 仍然存在；解析精度提升会在入库转换阶段再次损失。

结论：不采用。

### 3.2 方案 B：MinerU 主解析 + canonical block + parent-child contextual retrieval

优点：保留 MinerU 的结构信息，解决静默截断，支持可审计 Markdown、结构化表格、精确 citation 和 contextualized embedding；可以覆盖所有现有格式。

缺点：需要数据库迁移、全量重建、分阶段队列和更完整的验收。

结论：采用。

### 3.3 方案 C：每篇文档并行运行 MinerU 与 Document Intelligence，再融合

优点：理论上覆盖率最高。

缺点：每篇文档重复消耗解析和模型资源，融合冲突复杂，最终来源不稳定。

结论：不作为常规路径；Document Intelligence 只做异常触发的局部修复或严重失败 fallback。

## 4. 总体架构

```text
Source file
  -> format adapter
  -> CanonicalDocument + CanonicalBlock[]
  -> deterministic quality gate
  -> targeted repair when required
  -> immutable canonical artifact
  -> section-aware semantic Parent construction
  -> semantic Child construction
  -> batched contextual prefix generation
  -> contextualized embedding
  -> pgvector shadow index
  -> validation
  -> atomic activation
```

统一架构分为六层：

1. **Format adapters**：解析 PDF、DOCX、HTML、TXT/Markdown。
2. **Canonical model**：统一 block、结构、来源、定位和资源。
3. **Quality and repair**：检测缺页、结构损坏、表格异常、资源失效并触发定向修复。
4. **Semantic chunking**：构建章节优先的 Parent 和语义 Child。
5. **Contextualization and embedding**：批量生成 prefix，再生成正式向量。
6. **Version activation**：生成不可变解析包和 shadow index，通过验收后原子启用。

## 5. Canonical 数据模型

### 5.1 CanonicalDocument

```python
@dataclass
class CanonicalDocument:
    title: str
    abstract: str | None
    keywords: list[str]
    outline: list[SectionNode]
    blocks: list[CanonicalBlock]
    tables: list[CanonicalTable]
    figures: list[CanonicalFigure]
    formulas: list[CanonicalFormula]
    assets: list[CanonicalAsset]
    metadata: dict[str, Any]
```

Abstract 只允许来自原文解析。若 PDF 原文存在 Abstract 但 MinerU 未识别，前两页进入 Document Intelligence 修复；不允许 LLM 根据全文生成替代摘要。

### 5.2 CanonicalBlock

```python
@dataclass
class CanonicalBlock:
    block_id: str
    block_type: str
    text: str
    section_path: list[str]
    reading_order: int
    page_index: int | None
    page_label: str | None
    source_locator: dict[str, Any]
    parser_source: str
    parser_confidence: float | None
    table_id: str | None
    figure_id: str | None
    formula_id: str | None
```

`block_type` 至少支持：

- `heading`
- `narrative`
- `table`
- `figure`
- `formula`
- `caption`
- `appendix`
- `reference`

结构块类型不能再从 heading 文本推断。

### 5.3 Source locator

不同格式保存不同定位信息：

- PDF：`page_index`、`page_label`、`bbox`、`normalized_bbox`、MinerU block ID。
- DOCX：paragraph ID、table ID、row/column、image relationship ID。
- HTML：XPath/CSS selector、element ID、heading path。
- TXT/Markdown：行号范围、字符偏移、heading path。

一个 Child 可以关联多个 source span，以支持跨 block 或跨页证据。

### 5.4 DocumentChunk 扩展

`DocumentChunk` 增加正式字段：

```text
parse_version
parent_chunk_id
chunk_role
block_type
section_path
source_block_ids
source_spans
contextual_prefix
embedding_text
contextualization_model
contextualization_version
contextualization_prompt_version
contextualized_at
parser_name
parser_version
splitter_name
splitter_version
splitting_model
semantic_boundary_score
token_count
previous_chunk_id
next_chunk_id
```

`text` 始终是原始证据文本；`embedding_text` 是检索派生文本。Citation 只能展示 `text` 或 canonical 原文资产。

## 6. 不可变解析版本

每个成功解析候选版本输出：

```text
data/parsed/<document_id>/<parse_version>/
  canonical.md
  manifest.json
  blocks.jsonl
  tables.json
  figures.json
  formulas.json
  assets/
```

### 6.1 parse_version

版本由以下内容的稳定 hash 生成：

```text
source SHA256
pipeline version
format parser name/version
MinerU backend/config
repair prompt version
canonicalizer version
```

相同输入与配置复用已有 canonical 版本。更换解析器或规范化逻辑会产生新版本。

### 6.2 canonical.md

`canonical.md` 是给人阅读和下载的来源忠实版本，包含：

- 原文标题、Abstract、章节、正文和附录。
- 完整验收后的 Markdown 表格。
- 原始 Figure 图片链接和原始 caption。
- 原始 LaTeX 公式和原文公式说明。
- References/Bibliography。
- 稳定 block anchor。

它不包含：

- contextual prefix。
- AI Figure 总结。
- AI Formula 解释。
- 其他无法直接归因到原文件的生成文字。

### 6.3 manifest.json

manifest 记录解析器、版本、fallback 页面、质量结果、warning、资源清单和状态。它是解析审计与问题定位依据。

### 6.4 blocks.jsonl

blocks 文件是 chunk 重建、citation 定位和 PDF/DOM 高亮的结构化来源。Markdown 是渲染产物，不作为唯一事实来源。

## 7. 格式解析策略

### 7.1 PDF

PDF 调整为：

```text
basic PDF validation/page count
  -> MinerU primary parse
  -> quality gate
  -> targeted Document Intelligence page/region repair
  -> full Document Intelligence fallback on fatal failure
  -> pypdf text-layer fallback as last resort
```

pypdf 单页文本失败只记录 warning，不得阻止 MinerU。文本层用于覆盖率对照和最终 fallback。

MinerU 输出需要保留并归一化：

- page/block reading order
- heading hierarchy
- table HTML/Markdown/cells
- formula LaTeX
- Figure asset/caption/note
- bbox 和 source block ID

所有 `[:4000]` 静默截断必须删除，改为结构感知的 token 分割。

### 7.2 DOCX

DOCX adapter 解析：

- Heading 样式与层级。
- 普通 paragraph。
- table、row、cell、merge 信息。
- inline/anchored images 与 caption。
- 公式和关系 ID。
- 脚注、尾注、页眉页脚的来源分类。

Paragraph ID、table ID、cell 坐标和 relationship ID 用于 source locator。

### 7.3 HTML/HTM

HTML adapter 解析：

- 主正文 heading hierarchy。
- paragraph、list、table、figure、figcaption、code、formula、link。
- DOM selector 和 element ID。
- 导航、广告、脚本、样式和重复模板内容过滤。

Trafilatura 可以辅助确定正文，但不能丢弃 DOM 结构和 selector。

### 7.4 TXT/Markdown

Markdown adapter 识别 heading、table、code fence、formula、image、link 和 reference section。纯 TXT 先按显式标题/空行生成 block，再进入语义切割。

## 8. 质量门控与修复

### 8.1 质量分级

**Fatal：** 缺页、正文大面积为空、reading order 严重错误、content schema 无法解析、解析器失败、资源引用越界。Fatal 触发整篇 fallback。

**Repairable：** 单页表格、Figure、公式、Abstract 或结构缺失。Repairable 触发局部 Document Intelligence。

**Warning：** 个别 Figure/Formula AI 分析失败、少量 caption 缺失、非关键布局信息缺失。Warning 不阻止 canonical 版本。

### 8.2 Abstract 门控

若前两页原文存在 Abstract，但 MinerU 未识别，Document Intelligence 修复前两页。修复结果必须与文本层或页面图像对应，不允许生成替代摘要。

### 8.3 表格强门控

表格必须通过：

- 非空和基本行列结构。
- header/data row 有效。
- 行列对齐或显式 rowspan/colspan。
- caption/Table 编号关联。
- 数字、百分比、单位未被异常拆分。
- 无截断。
- 跨页 continuation 可恢复。
- Markdown、structured cells 和原始区域一致。

异常表格根据 bbox 裁剪区域并调用视觉模型修复；修复后再次执行确定性校验。仍失败时文档进入 `table_repair_failed`，不能生成正式索引。

正常 MinerU 表格不重复调用视觉模型。

### 8.4 Figure 与 Formula

原始 Figure、caption、公式和邻近正文必须保留。AI Figure/Formula 深度分析允许失败并记录 warning，可在后续单独补跑。

视觉模型识别的 Figure 数值可以用于回答，但必须标记为“图像解析结果”，引用原始 Figure 和 bbox；低置信度时只描述趋势，不输出具体数值。

## 9. 结构化表格

完整表格保存在 `tables.json` 和 canonical metadata 中，包含：

```text
table_id
page/source spans
caption
headers
rows
cells
rowspan/colspan
footnotes
source HTML/Markdown
normalized Markdown
validation status
```

Chunk 策略：

- 完整表格是 Parent。
- 按语义相关的数据行组构建 Child。
- 每个 Child 重复 caption 和完整表头。
- Child 按 token 限制，不按固定行数或字符截断。
- 跨页表格先合并，再构建 Parent/Child。

状态至少包括：

- `accepted_mineru`
- `repaired_by_vision`
- `cross_page_merged`
- `validation_failed`

## 10. Figure 与 Formula 派生数据

### 10.1 Figure

canonical Markdown 嵌入复制到版本 `assets/` 的原图和原始 caption。`figures.json` 保存 AI 生成的图类型、坐标轴、图例、趋势、结构化观察和置信度。

Figure 检索文本由以下内容构成：

```text
contextual prefix
original caption
AI visual summary when available
nearby source discussion
```

### 10.2 Formula

canonical Markdown 保存原始 LaTeX、编号和原文说明。`formulas.json` 保存 AI 变量解释、方法作用和置信度。

Formula 检索文本由以下内容构成：

```text
contextual prefix
original LaTeX
nearby source explanation
AI formula analysis when available
```

## 11. Semantic Parent/Child

### 11.1 边界优先级

```text
format structural boundary
  -> section hierarchy
  -> paragraph/sentence units
  -> temporary semantic embedding
  -> similarity breakpoints
  -> token limit protection
```

表格、Figure、Formula 使用专用 chunker，不进入普通正文语义合并。

### 11.2 默认参数

```text
Parent minimum: 500 tokens
Parent target: 1200 tokens
Parent maximum: 1800 tokens
Child minimum: 180 tokens
Child target: 400 tokens
Child maximum: 600 tokens
Child overlap: 50 tokens, whole-sentence only
Semantic break percentile: lowest 20% within current section
```

达到 Parent target 后，在低相似度边界优先切分；达到 maximum 后在最近句子边界强制切分。单个超长段落必须按句子继续拆分，不能保留超限 block。

### 11.3 语义切割模型

当前使用 `qwen3-embedding:4b`，但通过独立配置指定。语义切割 embedding 只用于确定边界；正式检索 embedding 在 contextualization 后重新生成。

### 11.4 回答扩展

检索命中 Child 后：

- 默认加载对应 Parent。
- 根据问题类型和语义连续性补充前后相邻 Child。
- 表格加载 caption、完整表头、命中行组和必要关联行组。
- Figure 加载原图、caption、分析和邻近讨论。
- Formula 加载公式、变量说明和前后推导。
- Citation 始终指向实际原文 source spans。

## 12. Contextualization

### 12.1 输入与输出术语

**Contextualization input：**

```text
document title
source Abstract when available
section outline
current section path
current Parent text
Child ID and original text
```

**Contextual prefix：** LLM 根据以上输入生成的 1～2 句中文关系说明，保留论文名、方法名、模型名、数据集名、指标名和公式英文原名，不生成或改写具体数值。

**Embedding text：**

```text
contextual prefix

Child original text
```

### 12.2 批量处理

- 默认每批 12 个 Child。
- 输出必须使用结构化 schema，并带回每个 `child_id`。
- 校验 ID 集合、唯一性、完整性、句数、长度和实体一致性。
- 当前模型为 `qwen3.5:9b`，配置与回答模型解耦。
- 所有可检索 Child，包括正文、表格、Figure、Formula、caption 和 appendix，都必须生成 prefix。
- References 不进入普通 contextualization 和 embedding。

### 12.3 失败策略

总共三次尝试：

1. 默认 batch size 12。
2. 约 2 秒退避后，用纠错提示重试。
3. 约 8 秒退避，将批次拆半后重试。

OOM 时继续缩小批次。已成功批次保存 checkpoint。任一可检索 Child 最终失败时，文档进入 `contextualization_failed`，不允许使用原文直接生成正式 embedding。

### 12.4 配置

```text
CONTEXTUALIZATION_ENABLED
CONTEXTUALIZATION_MODEL
CONTEXTUALIZATION_BASE_URL
CONTEXTUALIZATION_BATCH_SIZE
CONTEXTUALIZATION_MAX_RETRIES
CONTEXTUALIZATION_TIMEOUT
CONTEXTUALIZATION_MAX_SENTENCES
CONTEXTUALIZATION_PROMPT_VERSION
```

## 13. 向量索引与激活

只有 contextual prefix 完整率达到 100% 后才能生成正式 embedding。每个 Child 保存原始 `text`、`contextual_prefix`、`embedding_text` 和 embedding。

新索引构建在未激活 parse version 下。校验以下条件后原子更新 `Document.active_parse_version`：

- canonical artifact 完整。
- parser quality gate 通过。
- table validation 通过。
- Parent/Child 关系完整。
- contextual prefix 100%。
- embedding 100%，维度正确。
- pgvector row 数与可检索 Child 数一致。
- source span 和 asset 链接有效。

任一步失败时不暴露半成品索引。

## 14. Redis 分阶段任务

入库拆成：

```text
ingest.parse
ingest.repair
ingest.canonicalize
ingest.semantic_split
ingest.contextualize
ingest.embed
ingest.index
ingest.activate
```

每个阶段保存 checkpoint、输入版本、输出版本、状态、错误和进度。Worker 重启后从失败阶段继续。MinerU、contextualization 和 embedding GPU 并发均为 1；文件与数据库校验可并行。

文档状态至少支持：

```text
queued
parsing
quality_checking
repairing
canonicalizing
chunking
contextualizing
embedding
indexing
ready
parse_failed
table_repair_failed
contextualization_failed
embedding_failed
activation_failed
```

## 15. API 与前端

新增或扩展接口：

```text
GET /api/documents/{document_id}/parse
GET /api/documents/{document_id}/parse/markdown
GET /api/documents/{document_id}/parse/download
GET /api/documents/{document_id}/parse/status
GET /api/documents/{document_id}/citations/{chunk_id}/location
```

前端每篇文档提供：

- 查看解析 Markdown。
- 下载当前 `canonical.md`。
- 当前解析版本、解析器、状态和质量摘要。
- MinerU fallback/repair 页面摘要。
- PDF/Markdown 定位切换。
- Citation 点击后打开对应原文件并高亮 source spans。

前端不提供历史版本列表、用户回滚或 contextual prefix 展示。

## 16. References 与 Appendix

References/Bibliography 完整保留在 canonical 文档和 blocks 中，标记 `block_type=reference`，默认不生成普通检索 Child。用户明确查询参考文献、作者或引用关系时，可由独立 reference retrieval 使用。

Appendix 按正常章节处理，生成 Parent/Child、contextual prefix 和 embedding。回答必须标明证据来自附录，不能自动表述成正文核心结论。

## 17. 历史数据迁移

迁移采用测试环境维护模式：

1. 暂停上传和查询。
2. 保留所有原始文件。
3. 执行 schema migration。
4. 全部历史文档使用新流程重新解析。
5. 全部文档必须成功，才进入测试阶段。
6. 运行 full30 和新增专项题集。
7. 所有门槛通过后重新开放测试环境。
8. 删除旧 Chunk、旧 embedding、旧 pgvector rows、旧解析数据和旧离线数据备份。
9. 删除已成功 canonicalize 的 MinerU 临时目录。

这是明确的不可逆清理决策。删除只能在新版本和测试门槛全部通过后执行，删除范围不得包含原始 PDF/DOCX/HTML/TXT/Markdown 或新 canonical 包。

## 18. 验收标准

### 18.1 入库完整性

- 历史文档解析成功率 100%。
- contextual prefix 完整率 100%。
- 正式 embedding 完整率 100%。
- pgvector 行数与可检索 Child 数一致。
- 表格结构验收通过率 100%。
- Citation source span 有效率 100%。
- canonical Markdown 资源链接有效率 100%。
- 任何 `[:4000]` 式静默截断为 0。
- 任一失败文档阻止全量测试开始。

### 18.2 RAG 回归

保留现有 full30，并增加专项题集覆盖：

- 普通正文语义召回。
- 中文问题检索英文论文。
- Parent/Child 动态扩展。
- 长表格行组。
- 跨页表格。
- Figure 原图、caption 和视觉分析。
- Figure/Formula AI 分析 warning fallback。
- Formula 原文和邻近解释。
- contextual prefix 不进入 citation。
- PDF bbox、DOCX paragraph/table、HTML selector 和文本行号定位。
- MinerU 局部修复和整篇 fallback。
- canonical Markdown 完整性。
- 旧索引清理验证。

测试记录 recall@5、recall@10、表格指标准确率、citation 定位正确率、回答完整性、P50/P95 检索时间和单篇处理时间。所有失败必须有阶段归因。

## 19. 安全与来源边界

- contextual prefix、Figure AI 分析和 Formula AI 分析必须标记为生成数据。
- 生成数据不能写入 canonical 原文或 citation excerpt。
- Figure 视觉数值必须标记为图像解析结果并指向原图 bbox。
- 资源路径必须限制在当前 canonical version 的 `assets/`。
- Source locator 不允许构造任意文件路径。
- 新版本激活和旧数据清理必须在数据库事务与文件校验边界内执行。

## 20. 非目标

本次不包含：

- PPTX、XLSX 或其他新格式。
- 前端历史解析版本浏览和用户回滚。
- 每篇文档并行运行完整 MinerU 与 Document Intelligence。
- OCR 作为数字版论文的默认路径。
- References 默认参与普通 RAG。
- contextual prefix 作为用户可见证据。
- 在线无停机历史迁移。
- 保留迁移前旧 RAG 数据或离线旧数据备份。

## 21. 实施分解建议

该设计需要按可独立验收的阶段实施：

1. Canonical schema、artifact 和版本激活基础。
2. PDF MinerU 主解析、质量门控和修复。
3. DOCX、HTML、TXT/Markdown canonical adapters。
4. 结构化 Table/Figure/Formula。
5. Semantic Parent/Child。
6. Batched contextualization。
7. Contextual embedding 与动态 Parent 扩展检索。
8. Redis 阶段队列和断点恢复。
9. Parse API、Markdown 查看/下载和 citation 高亮。
10. 全量迁移、专项评测和不可逆旧数据清理。

每个阶段必须先提供聚焦测试，再进入下一阶段；全量历史数据迁移只能在前九个阶段全部完成后执行。
