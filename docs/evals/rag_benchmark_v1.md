# RAG Benchmark v1（草稿，待评审）

- **benchmark_id**: `rag_benchmark_v1`
- **version**: `v1.1-draft`
- **修订**: 2026-09-22（`eval-fix-001`：修正 Self-RAG 引用页、替换 ColBERT 未被语料支持的术语、统一 JSON/Markdown 引号、明确 CoT-SC 基线为必答，并新增结构化 `scoring` 元数据）
- **status**: `draft_pending_review`
- **project**: `internal-research`
- **生成日期**: 2026-09-22
- **题目数**: 15（`single_paper_fact` 8 / `cross_paper_compare` 4 / `table_metric` 3）
- **机器可读版本**: [`rag_benchmark_v1.json`](./rag_benchmark_v1.json)

## 方法说明（Methodology）

**这是一份等待人工评审的草稿**，不是已验收的评测集。它没有跑过 agent，还没有加入拒答/不可答题，其中结构化 `scoring` 元数据也尚未经人工验证。

- **证据来源**：全部参考答案点都是从 `internal-research` 项目当前 PostgreSQL 语料里逐条探针核对出来的（按 `document_chunks.text` + `page_label` 精确匹配），并与 7 篇源 PDF 做了交叉验证——7 个 PDF 的 SHA-256 与已入库文档完全一致。
- **不使用外部知识**：每个 `expected_answer_points` 都验证过确实出现在所引用的那一页上。凡是语料里没有的内容一律不作为答案点。
- **`page_label` 的含义**：引用里的 `page_label` 是**语料的页码标签**（canonical parse 的 1-based 页号），**不一定等于 PDF 印刷页码**。本批文档里已发现两处页边界归属偏移，详见下面「证据限制」。
- **语言**：问题语言按 `language` 字段区分。本集包含 8 题中文、7 题英文——语料是英文论文，而 `internal-research` 现有查询基准（`internal_research_v1`）用中文提问，两者都需要覆盖。
- **评分元数据**：每道题都带一个 `scoring` 对象，含 `required_point_count`（命中几点算通过）、`required_citation_count`（至少要正确引用几个页）、`numeric_tolerance`（定性题为 `null`，含必答数值/量级的题为 `exact`）、`partial_credit`（统一为 `per_answer_point`，各点独立计分）、`required_terms` 与 `term_aliases`（必答术语及其可接受变体）。所有 `required_terms` 都逐条验证过出现在该题自己引用的语料页上。
- **引用字段**：每条 `expected_citations` 同时给出论文标题（`document_title`）、语料标题（`corpus_title`）、`document_id`、`page_slug`、`page_label` 和 `evidence_anchor`（该页上用于核对的原文字符串）。

### 待评审确认的点

1. 判分方式：`scoring.required_point_count` 已给出「命中几点算通过」，但各点默认等权；若某些点应加权，需要评审时补充。
2. 是否保留 `tm_react_hotpotqa_fever_table`，考虑到它的表体在语料里位于与印刷页码不同的页。
3. 是否需要在 v2 补入拒答题（`unanswerable`）与近重复题去重。
4. `required_terms` / `term_aliases` 的取值是否过严或过松——它们目前是按语料原文挑的，尚未用真实模型输出回放验证。

## 语料（7 篇已入库文档）

| corpus_title | 论文 | 页数 | parse 状态 | parse 版本 |
|---|---|---:|---|---|
| `rag_2005.11401` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 19 | accepted_with_warnings (0.85) | `canonical-v4-23e3249e9a1e-…` |
| `selfrag_2310.11511` | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | 30 | accepted_with_warnings (0.60) | `canonical-v4-d9eaa1398aba-…` |
| `crag_2401.15884` | Corrective Retrieval Augmented Generation | 16 | accepted_with_warnings (0.95) | `canonical-v4-975aa1fd3c1b-…` |
| `colbert_2004.12832` | ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT | 10 | accepted_with_warnings (0.85) | `canonical-v4-2e487d9b96e3-…` |
| `murag_2210.02928` | MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text | 13 | accepted_with_warnings (0.70) | `canonical-v4-dfdfb9bcf1e6-…` |
| `react_2210.03629` | ReAct: Synergizing Reasoning and Acting in Language Models | 33 | **validation_failed** (0.85) | `canonical-v4-textfallback-f285b0971ae4-…` |
| `visrag_2410.10594` | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents | 25 | **validation_failed** (0.55) | `canonical-v4-textfallback-1b952e416611-…` |

---

## A. 单篇事实题（single_paper_fact，8 题）

### 1. `spf_rag_two_components`

- **难度**: easy ｜ **语言**: zh
- **问题**: 在 RAG 论文中，RAG 模型的参数化记忆（parametric memory）和非参数化记忆（non-parametric memory）分别由什么实现？两者由什么组件连接？
- **参考答案点**:
  - 参数化记忆是一个预训练的 seq2seq 模型（BART）
  - 非参数化记忆是 Wikipedia 的稠密向量索引（dense vector index）
  - 该索引由一个预训练的神经检索器访问，即 DPR（Dense Passage Retriever）
  - 检索器给出隐变量文档，seq2seq 生成器在这些文档条件下生成输出，二者端到端联合微调
- **期望引用**:
  - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（`rag_2005.11401`, page_label `1`）— 锚点：`dense vector index of Wikipedia`
  - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（`rag_2005.11401`, page_label `2`）— 锚点：`Dense Passage Retriever [26], henceforth DPR ... seq2seq model (BART [32])`
- **评分元数据**:
  - `required_point_count`: 3（共 4 点）｜ `required_citation_count`: 2（共 2 条引用）｜ `numeric_tolerance`: null ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `BART`、`DPR`、`Wikipedia`
  - `term_aliases`: `BART` → [`BART [32]`]；`DPR` → [`Dense Passage Retriever`]；`Wikipedia` → [`dense vector index of Wikipedia`]
- **备注**: 摘要（page_label 1）只给出总体描述（pre-trained seq2seq + dense vector index of Wikipedia）；BART 与 DPR 的具体命名出现在 page_label 2，完整答案需要引用两页。若答案把非参数化记忆说成“知识图谱”或“搜索引擎”，判为错误。

### 2. `spf_rag_sequence_vs_token`

- **难度**: medium ｜ **语言**: en
- **问题**: What is the difference between the RAG-Sequence and RAG-Token formulations in how they marginalize the retrieved document?
- **参考答案点**:
  - RAG-Sequence uses the same retrieved document to generate the complete output sequence
  - RAG-Token can draw a different latent document for each target token
  - Both marginalize over the top-K retrieved documents returned by the retriever
  - For sequence classification the two formulations are equivalent
- **期望引用**:
  - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（`rag_2005.11401`, page_label `3`）— 锚点：`RAG-Sequence Model / RAG-Token Model`
- **评分元数据**:
  - `required_point_count`: 3（共 4 点）｜ `required_citation_count`: 1（共 1 条引用）｜ `numeric_tolerance`: null ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `RAG-Sequence`、`RAG-Token`、`top-K`
  - `term_aliases`: `RAG-Sequence` → [`RAG-Seq.`, `RAG-Sequence model`]；`RAG-Token` → [`RAG-Tok.`, `RAG-Token model`]；`top-K` → [`top-k`, `top K`]
- **备注**: Both definitions and the classification-equivalence remark sit on page_label 3. An answer that only says 'they marginalize differently' without the per-sequence vs per-token distinction should be scored as incomplete.

### 3. `spf_selfrag_reflection_tokens`

- **难度**: medium ｜ **语言**: en
- **问题**: What are reflection tokens in SELF-RAG, and what two broad categories do they fall into? What do they allow the model to do at inference time?
- **参考答案点**:
  - Special tokens that the language model itself generates to reflect on the retrieved passages and on its own generation
  - They are categorized into retrieval tokens and critique tokens
  - The retrieval token decides whether retrieval is needed, enabling adaptive on-demand retrieval
  - They are learned by expanding the model vocabulary and predicting them as ordinary next tokens
- **期望引用**:
  - Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection（`selfrag_2310.11511`, page_label `1`）— 锚点：`categorized into retrieval and critique tokens`
  - Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection（`selfrag_2310.11511`, page_label `2`）— 锚点：`next token prediction from the expanded model vocabulary`
- **评分元数据**:
  - `required_point_count`: 3（共 4 点）｜ `required_citation_count`: 2（共 2 条引用）｜ `numeric_tolerance`: null ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `reflection tokens`、`retrieval and critique tokens`、`expanded model vocabulary`
  - `term_aliases`: `reflection tokens` → [`reflection token`]；`retrieval and critique tokens` → [`retrieval token`, `critique token`, `retrieval and critique token`]；`expanded model vocabulary` → [`next token prediction from the expanded model vocabulary`]
- **备注**: Page 1 states the two-way categorization (retrieval vs critique tokens) and what they let the model do at inference time. The vocabulary-expansion training mechanism is NOT on page 1: it is stated on page_label 2 (“unifying them as the next token prediction from the expanded model vocabulary”) and recurs on page_label 4. Both pages must be cited for the fourth answer point to be verifiable. The four concrete token types are listed separately in Table 1 on page_label 4 (see case tm_selfrag_reflection_token_table).

### 4. `spf_crag_three_actions`

- **难度**: medium ｜ **语言**: zh
- **问题**: CRAG 的检索评估器（retrieval evaluator）由什么模型初始化？它把检索结果判成哪三个动作，每个动作各自触发什么处理？
- **参考答案点**:
  - 用 T5-large 初始化并微调，参数量远小于当时的 LLM
  - 把相关性置信度量化成三个动作 {Correct, Incorrect, Ambiguous}
  - Correct：把检索文档精炼成 knowledge strips，过程包含 knowledge decomposition、filter、recomposition
  - Incorrect：丢弃检索到的文档，改用大规模网络搜索作为补充知识来源
  - Ambiguous：无法确信地判定对错时触发，同时结合上述两种处理
- **期望引用**:
  - Corrective Retrieval Augmented Generation（`crag_2401.15884`, page_label `4`）— 锚点：`T5-large ... {Correct, Incorrect, Ambiguous}`
  - Corrective Retrieval Augmented Generation（`crag_2401.15884`, page_label `3`）— 锚点：`refined into more precise knowledge strips`
- **评分元数据**:
  - `required_point_count`: 3（共 5 点）｜ `required_citation_count`: 2（共 2 条引用）｜ `numeric_tolerance`: null ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `T5-large`、`Correct`、`Incorrect`、`Ambiguous`、`knowledge strips`
  - `term_aliases`: `T5-large` → [`T5-large (Raffel et al., 2020)`]；`knowledge strips` → [`knowledge strip`]；`Ambiguous` → [`ambiguous action`]
- **备注**: T5-large 与三动作在 page_label 4，各分支处理细节在 page_label 3。注意语料把摘要里的 'decompose-then-recompose' 渲染成 'decompose-thenrecompose'（换行连字符被吞掉），按原文词组做字符串匹配会漏，需要按语义判定。

### 5. `spf_colbert_late_interaction`

- **难度**: easy ｜ **语言**: en
- **问题**: How does ColBERT's late interaction architecture work, and what efficiency claim does the paper make against BERT-based rankers?
- **参考答案点**:
  - Query and document are encoded independently by BERT rather than jointly
  - A cheap interaction step then models fine-grained token-level similarity between the two encodings
  - Because interaction is delayed, document representations can be pre-computed offline
  - ColBERT executes two orders-of-magnitude faster and requires four orders-of-magnitude fewer FLOPs per query than BERT-based rankers
- **期望引用**:
  - ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（`colbert_2004.12832`, page_label `1`）— 锚点：`two orders-of-magnitude faster and requiring four orders-of-magnitude fewer FLOPs`
  - ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（`colbert_2004.12832`, page_label `2`）— 锚点：`MaxSim`
- **评分元数据**:
  - `required_point_count`: 3（共 4 点）｜ `required_citation_count`: 2（共 2 条引用）｜ `numeric_tolerance`: `exact` ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `late interaction`、`BERT`、`two orders-of-magnitude`、`four orders-of-magnitude`
  - `term_aliases`: `late interaction` → [`late interaction architecture`, `late-interaction`]；`two orders-of-magnitude` → [`two orders of magnitude`, `2 orders of magnitude`]；`four orders-of-magnitude` → [`four orders of magnitude`, `4 orders of magnitude`]
- **备注**: Page 1 carries the architecture description and the efficiency claim; the MaxSim operator itself is defined in the model section attributed to page_label 2. The two-orders/four-orders phrasing is also present in the ACM reference block on page 1.

### 6. `spf_murag_design`

- **难度**: medium ｜ **语言**: zh
- **问题**: MuRAG 相对之前的检索增强模型有什么本质区别？它的骨干编码器由什么构成，在哪些数据集上评测，效果如何？
- **参考答案点**:
  - 是第一个能够使用多模态知识（视觉与文本）的检索增强模型，可访问外部非参数化多模态记忆
  - 该记忆包含图像、纯文本或图文对条目
  - 骨干编码器由预训练的 T5 与 ViT 组合而成，把记忆条目与查询编码到同一表示空间
  - 在 WebQA 与 MultimodalQA 两个开放多模态 QA 数据集上评测
  - 相比已有基线绝对提升 10–20%，在 distractor（40+ 候选）与 full-wiki（100 万候选）两种设定下都成立
- **期望引用**:
  - MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text（`murag_2210.02928`, page_label `1`）— 锚点：`first Multimodal Retrieval-Augmented`
  - MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text（`murag_2210.02928`, page_label `2`）— 锚点：`combine pre-trained T5 (Raffel et al., 2020) and ViT`
- **评分元数据**:
  - `required_point_count`: 3（共 5 点）｜ `required_citation_count`: 2（共 2 条引用）｜ `numeric_tolerance`: `exact` ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `T5`、`ViT`、`WebQA`、`MultimodalQA`、`10-20%`
  - `term_aliases`: `T5` → [`T5 (Raffel et al., 2020)`]；`ViT` → [`ViT (Dosovitskiy et al., 2020)`]；`10-20%` → [`10- 20%`, `10–20%`, `10 to 20%`]
- **备注**: page_label 1 的摘要里 '10- 20%' 因换行在语料中带一个空格，按 '10-20%' 精确匹配会漏；page_label 2 的正文写作 '10-20%' 且给出 40+/1M 候选规模。T5+ViT 骨干只在 page_label 2。

### 7. `spf_react_alfworld_webshop`

- **难度**: easy ｜ **语言**: en
- **问题**: On the two interactive decision-making benchmarks, how much does ReAct improve over imitation and reinforcement learning baselines, and how many in-context examples does it use?
- **参考答案点**:
  - ALFWorld: 34% absolute improvement in success rate
  - WebShop: 10% absolute improvement in success rate
  - It is prompted with only one or two in-context examples
  - Both gains are over imitation and reinforcement learning methods
- **期望引用**:
  - ReAct: Synergizing Reasoning and Acting in Language Models（`react_2210.03629`, page_label `1`）— 锚点：`absolute success rate of 34% and 10% respectively`
- **评分元数据**:
  - `required_point_count`: 3（共 4 点）｜ `required_citation_count`: 1（共 1 条引用）｜ `numeric_tolerance`: `exact` ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `ALFWorld`、`WebShop`、`34%`、`10%`
  - `term_aliases`: `34%` → [`34 %`]；`10%` → [`10 %`]
- **备注**: Both numbers and the one-or-two-examples claim are in the abstract on page_label 1. The base model is the frozen PaLM-540B (corpus page_label 3); an answer that additionally names PaLM-540B is more complete but not required.

### 8. `spf_visrag_design`

- **难度**: medium ｜ **语言**: zh
- **问题**: VisRAG 的检索与生成分别由什么组件完成？它为什么声称能避免传统文本 RAG 的信息损失，端到端收益是多少？
- **参考答案点**:
  - 流水线由 VLM 驱动的检索器 VisRAG-Ret 与生成器 VisRAG-Gen 组成
  - 文档不再先解析成文本，而是直接作为图像由 VLM 编码
  - 因此保留原始文档的版式与图像信息，消除解析阶段引入的信息损失
  - 相比传统文本 RAG 流水线取得 20–40% 的端到端性能提升
  - VisRAG-Ret 沿用稠密检索的双塔结构，对最终隐状态做加权平均池化得到嵌入
- **期望引用**:
  - VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents（`visrag_2410.10594`, page_label `1`）— 锚点：`VisRAG-Ret`
  - VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents（`visrag_2410.10594`, page_label `2`）— 锚点：`VisRAG-Gen`
- **评分元数据**:
  - `required_point_count`: 3（共 5 点）｜ `required_citation_count`: 2（共 2 条引用）｜ `numeric_tolerance`: `exact` ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `VisRAG-Ret`、`VisRAG-Gen`、`VLM`、`20–40%`
  - `term_aliases`: `VisRAG-Ret` → [`VisRAG Ret`]；`VisRAG-Gen` → [`VisRAG Gen`]；`20–40%` → [`20-40%`]；`VLM` → [`vision-language model`]
- **备注**: 证据限制：该文档 parse 状态为 validation_failed（score 0.55）。语料把 'VisRAG-Ret' 一词归到 page_label 1（页边界归属偏移），page_label 2 只含 'VisRAG-Gen'，因此引用 page_label 2 时不要断言该页出现 'VisRAG-Ret'。加权平均池化在 page_label 1。

---

## B. 跨论文比较题（cross_paper_compare，4 题）

### 9. `cpc_selfrag_vs_crag`

- **难度**: hard ｜ **语言**: zh
- **问题**: Self-RAG 和 CRAG 都想让 RAG 在检索不可靠时更稳健，两者在机制上有什么本质区别？
- **参考答案点**:
  - Self-RAG 的评判内生于生成模型：同一个 LM 生成 reflection tokens（retrieval 与 critique 两类）来按需决定检索并自我评判
  - CRAG 外挂一个独立的轻量检索评估器（T5-large 初始化），对整批检索文档打分
  - CRAG 由置信度触发 {Correct, Incorrect, Ambiguous} 三个动作；Self-RAG 没有独立的三分支动作
  - CRAG 在 Incorrect 分支引入大规模网络搜索，作为静态语料之外的外部知识；Self-RAG 不引入外部搜索
  - CRAG 的 Correct 分支用 knowledge strips 做分解-过滤-重组的知识精炼；Self-RAG 用 critique token 控制生成质量
- **期望引用**:
  - Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection（`selfrag_2310.11511`, page_label `1`）— 锚点：`categorized into retrieval and critique tokens`
  - Corrective Retrieval Augmented Generation（`crag_2401.15884`, page_label `4`）— 锚点：`T5-large ... {Correct, Incorrect, Ambiguous}`
  - Corrective Retrieval Augmented Generation（`crag_2401.15884`, page_label `3`）— 锚点：`web searches are resorted to and regarded as complementary knowledge sources`
- **评分元数据**:
  - `required_point_count`: 3（共 5 点）｜ `required_citation_count`: 2（共 3 条引用）｜ `numeric_tolerance`: null ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `reflection tokens`、`T5-large`、`Correct`、`Incorrect`、`Ambiguous`
  - `term_aliases`: `reflection tokens` → [`reflection token`]；`T5-large` → [`T5-large (Raffel et al., 2020)`]
- **备注**: 两篇都在处理检索质量问题，落差在于“评判在哪一层”：Self-RAG 是 token 级、内生于生成模型；CRAG 是独立评估器 + 三分支动作 + 外部网络搜索。答案若只说“都做检索纠错”而不区分 evaluator 与 reflection token，判为不足。

### 10. `cpc_rag_vs_colbert_retrieval`

- **难度**: hard ｜ **语言**: en
- **问题**: Compare the retrieval component in the original RAG paper with ColBERT's retrieval design. How does each represent queries and documents, and how is each trained?
- **参考答案点**:
  - RAG uses DPR, a bi-encoder that produces a single dense vector per query and per document
  - RAG retrieves the top-K documents by maximum inner product search (MIPS) over the Wikipedia index
  - ColBERT encodes query and document independently with BERT but keeps per-token embeddings and scores them with late interaction (MaxSim)
  - Because the interaction step is delayed, ColBERT's token-level document representations can be pre-computed offline and served through vector-similarity indexes for end-to-end retrieval
  - RAG's retriever is fine-tuned jointly with the generator, whereas ColBERT is a standalone ranking/retrieval model
- **期望引用**:
  - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（`rag_2005.11401`, page_label `3`）— 锚点：`DPR follows a bi-encoder architecture`
  - ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（`colbert_2004.12832`, page_label `1`）— 锚点：`late interaction architecture`
  - ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（`colbert_2004.12832`, page_label `2`）— 锚点：`MaxSim`
- **评分元数据**:
  - `required_point_count`: 3（共 5 点）｜ `required_citation_count`: 2（共 3 条引用）｜ `numeric_tolerance`: null ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `DPR`、`bi-encoder`、`late interaction`、`MaxSim`
  - `term_aliases`: `DPR` → [`Dense Passage Retriever`]；`bi-encoder` → [`bi-encoder architecture`]；`MaxSim` → [`MaxSim operator`]；`late interaction` → [`late interaction architecture`]
- **备注**: The load-bearing distinction is the single-vector bi-encoder (DPR) against ColBERT's per-token embeddings scored by late interaction (MaxSim). An answer that compares only parameter counts or datasets, without naming the single-vector vs per-token difference, should be scored as incomplete. Terminology note: the ColBERT paper never uses the vector-per-token compound wording anywhere in its 10 pages, so score against the paper's own vocabulary (per-token embeddings, late interaction, MaxSim). Page_label 1 also writes “oﬄine” with an ff ligature, so a literal ASCII match on “offline” fails.

### 11. `cpc_murag_vs_visrag`

- **难度**: hard ｜ **语言**: zh
- **问题**: MuRAG 和 VisRAG 都属于多模态 RAG，它们“检索什么”和“怎么检索”有什么不同？
- **参考答案点**:
  - MuRAG 检索的是外部非参数化多模态记忆中的条目，条目可以是图像、纯文本或图文对
  - MuRAG 用 T5+ViT 组成的骨干把记忆条目与查询编码进同一空间，以对比损失与生成损失联合预训练
  - VisRAG 检索的是整页文档图像，把页面图像直接交给 VLM 编码，不做文本解析
  - VisRAG 的嵌入来自 VLM 末层隐状态的加权平均池化，属于双塔稠密检索
  - 评测取向不同：MuRAG 面向开放多模态 QA（WebQA、MultimodalQA），VisRAG 面向多模态文档的页级 RAG
- **期望引用**:
  - MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text（`murag_2210.02928`, page_label `2`）— 锚点：`nonparametric multimodal memory containing images, text, or image-text pairs`
  - MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text（`murag_2210.02928`, page_label `1`）— 锚点：`WebQA, and MultimodalQA`
  - VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents（`visrag_2410.10594`, page_label `1`）— 锚点：`utilizing the document’s image directly instead of relying on extracted textual content`
- **评分元数据**:
  - `required_point_count`: 3（共 5 点）｜ `required_citation_count`: 2（共 3 条引用）｜ `numeric_tolerance`: null ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `MuRAG`、`VisRAG`、`T5`、`ViT`、`VLM`
  - `term_aliases`: `T5` → [`T5 (Raffel et al., 2020)`]；`ViT` → [`ViT (Dosovitskiy et al., 2020)`]；`VLM` → [`vision-language model`]
- **备注**: 关键差别是检索单元的粒度：MuRAG 检索的是记忆条目（图文对/文本/图像），VisRAG 检索的是整页图像。答案若只笼统说“两者都用图像”而不点出粒度与是否解析，判为不足。证据限制：描述 VisRAG-Ret 直接用文档图像、不走解析的那句话在语料中归属 page_label 1（印刷页为第 2 页），因此本条引用 page_label 1 而非 2。

### 12. `cpc_react_vs_rag_pipeline`

- **难度**: hard ｜ **语言**: en
- **问题**: How does ReAct's use of external knowledge differ from the retrieve-then-generate pipeline of the RAG paper?
- **参考答案点**:
  - ReAct interleaves free-form reasoning traces (thoughts) with task-specific actions, so retrieval is one action among several rather than a fixed first stage
  - ReAct acts on a simple Wikipedia API and reads observations back into the trajectory, letting the model update plans and handle exceptions
  - RAG performs one dense retrieval of top-K Wikipedia passages per query and marginalizes over them during generation
  - ReAct prompts a frozen LLM (PaLM-540B) with one or two in-context examples, while RAG fine-tunes retriever and generator end-to-end
  - ReAct is evaluated on multi-hop QA and fact verification as well as interactive environments (ALFWorld, WebShop); RAG targets knowledge-intensive generation and classification tasks
- **期望引用**:
  - ReAct: Synergizing Reasoning and Acting in Language Models（`react_2210.03629`, page_label `1`）— 锚点：`interacting with a simple Wikipedia API`
  - ReAct: Synergizing Reasoning and Acting in Language Models（`react_2210.03629`, page_label `3`）— 锚点：`frozen large language model, PaLM-540B`
  - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（`rag_2005.11401`, page_label `2`）— 锚点：`marginalize the latent documents with a top-K approximation`
- **评分元数据**:
  - `required_point_count`: 3（共 5 点）｜ `required_citation_count`: 2（共 3 条引用）｜ `numeric_tolerance`: null ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `ReAct`、`Wikipedia API`、`top-K`、`PaLM-540B`
  - `term_aliases`: `Wikipedia API` → [`simple Wikipedia API`]；`top-K` → [`top-k`, `top K`]
- **备注**: 核心是“交织式 推理-行动-观察 轨迹”对比“一次性检索后生成”。答案若只比较数据集或模型规模，而没有提到可以从观察中修正计划，应判为不足。react 文档 parse 状态为 validation_failed，page_label 3 在语料中是一个较短的块。

---

## C. 表格/指标题（table_metric，3 题）

### 13. `tm_selfrag_reflection_token_table`

- **难度**: medium ｜ **语言**: zh
- **问题**: Self-RAG 论文 Table 1 列出了哪四类 reflection token？其中 Retrieve、ISREL、ISSUP、ISUSE 各自的取值集合是什么？
- **参考答案点**:
  - 四类：Retrieve、ISREL、ISSUP、ISUSE
  - Retrieve 取值 {yes, no, continue}，用于决定何时用检索器 R 检索
  - ISREL 取值 {relevant, irrelevant}，判断文档 d 对解决 x 是否有用
  - ISSUP 取值 {fully supported, partially supported, no support}
  - ISUSE 取值 {5, 4, 3, 2, 1}
  - 后三行是三类 critique token，表注指出加粗的是最理想的取值
- **期望引用**:
  - Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection（`selfrag_2310.11511`, page_label `4`）— 锚点：`Table 1: Four types of reflection tokens used in SELF-RAG`
- **评分元数据**:
  - `required_point_count`: 4（共 6 点）｜ `required_citation_count`: 1（共 1 条引用）｜ `numeric_tolerance`: `exact` ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `Retrieve`、`ISREL`、`ISSUP`、`ISUSE`、`{yes, no, continue}`、`{relevant, irrelevant}`、`{fully supported, partially supported, no support}`、`{5, 4, 3, 2, 1}`
  - `term_aliases`: `{5, 4, 3, 2, 1}` → [`{5,4,3,2,1}`, `5, 4, 3, 2, 1`]；`{fully supported, partially supported, no support}` → [`fully supported, partially supported, no support`]
- **备注**: 语料把该表渲染成规整的 markdown 表格，四行齐全，属于本批文档中表格保真度最高的一处。ISSUP 的取值在语料中为一行三值，注意不要只答 'fully supported'。

### 14. `tm_react_hotpotqa_fever_table`

- **难度**: medium ｜ **语言**: en
- **问题**: In ReAct Table 1 (PaLM-540B prompting results), what are the HotpotQA EM and Fever accuracy for ReAct, CoT-SC→ReAct, and ReAct→CoT-SC? The CoT-SC baseline is required as well.
- **参考答案点**:
  - ReAct: HotpotQA EM 27.4, Fever accuracy 60.9
  - CoT-SC→ReAct: HotpotQA EM 34.2, Fever accuracy 64.6
  - ReAct→CoT-SC: HotpotQA EM 35.1, Fever accuracy 62.0
  - CoT-SC baseline: HotpotQA EM 33.4, Fever accuracy 60.4
  - Supervised SoTA (upper reference): HotpotQA EM 67.5, Fever accuracy 89.5
- **期望引用**:
  - ReAct: Synergizing Reasoning and Acting in Language Models（`react_2210.03629`, page_label `4`）— 锚点：`Table 1: PaLM-540B prompting results on HotpotQA and Fever`
- **评分元数据**:
  - `required_point_count`: 3（共 5 点）｜ `required_citation_count`: 1（共 1 条引用）｜ `numeric_tolerance`: `exact` ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `27.4`、`60.9`、`34.2`、`64.6`、`35.1`、`62.0`、`33.4`、`60.4`
  - `term_aliases`: 无（数值题，按 `numeric_tolerance` 直接比对）
- **备注**: 重要证据限制：该表的表体在语料中归属 page_label 4，而不是 PDF 印刷页码 5。react 文档使用 textfallback 解析版本，表格以纯文本行渲染（每行形如 'CoT-SC→ ReAct 34.2 64.6'），没有 markdown 竖线；引用 page_label 5 只会命中 Figure 2 与脚注，会错过表体。

### 15. `tm_colbert_rerank_table`

- **难度**: hard ｜ **语言**: zh
- **问题**: ColBERT 论文 Table 1 的 MS MARCO 重排序结果中，ColBERT（over BERT_base）、BERT_base 与 BERT_large 的 MRR@10（Dev）、重排序延迟和 FLOPs/query 分别是多少？
- **参考答案点**:
  - ColBERT（over BERT_base）：MRR@10 (Dev) 34.9，延迟 61 ms，FLOPs/query 7B（1×）
  - BERT_base：MRR@10 (Dev) 34.7，延迟 10,700 ms，FLOPs/query 97T（13,900×）
  - BERT_large：MRR@10 (Dev) 36.5，延迟 32,900 ms，FLOPs/query 340T（48,600×）
  - 表注说明每个神经模型都对 BM25 官方 top-1000 结果重排序，表中延迟只计重排序部分
- **期望引用**:
  - ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（`colbert_2004.12832`, page_label `7`）— 锚点：`Table 1: “Re-ranking” results on MS MARCO`
- **评分元数据**:
  - `required_point_count`: 3（共 4 点）｜ `required_citation_count`: 1（共 1 条引用）｜ `numeric_tolerance`: `exact` ｜ `partial_credit`: `per_answer_point`
  - `required_terms`: `34.9`、`34.7`、`10,700`、`97T`、`36.5`、`32,900`、`340T`、`7B`
  - `term_aliases`: `10,700` → [`10700`]；`32,900` → [`32900`]
- **备注**: 语料把该表完整渲染成 markdown 表格。BERT 行的方法名在语料里是 LaTeX 形式（$\mathrm{BERT_{base}}$），字面匹配 'BERT_base' 会漏。数字带千分位逗号（10,700 / 32,900），判分时应接受 '10700' / '32900' 写法。
---

## 证据限制与已知歧义（Evidence limits）

评审时请把下面几条当作**已知风险**，而不是当作已被填平的空白：

1. **页边界归属偏移**。语料的 `page_label` 由 canonical parse 生成，与 PDF 印刷页码并不总是一一对应：
   - `react_2210.03629`：Table 1 的表体落在 page_label `4`，而印刷页是第 5 页。
   - `visrag_2410.10594`：`VisRAG-Ret` 一词落在 page_label `1`，而印刷页是第 2 页。
   - 判定原则：本集一律以**语料 page_label** 为准写引用，因为 RAG 系统返回的引用就是这个坐标系。
2. **两篇文档 parse 校验未通过**。`react_2210.03629`（score 0.85，fallback page 33）与 `visrag_2410.10594`（score 0.55，fallback page 22）走的是 textfallback 解析版本，表格与公式保真度低于其余五篇。本集已经据此把这两篇的题目限制在纯文本层可核对的范围内。
3. **连字符换行导致字面匹配失败**。CRAG 摘要的 `decompose-then-recompose`（语料作 `decompose-thenrecompose`）、MuRAG 摘要的 `10- 20%`（语料带空格），都说明不能只靠严格字符串匹配判分，需要正则或语义比对。
4. **RAG 论文 Table 1 在语料中渲染错乱**。该表被拼成 `RAG-Token RAG-Seq. | 44.1.55.2/66.1.44.5.56.8/68.0 | ...`，RAG-Token 与 RAG-Sequence 两行的数值挤在一起，`REALM`/`DPR` 行同样被合并。本集**故意没有**基于这张表出题；如果 v2 要加，必须先修订该表的解析或引入图像通道。
5. **检索召回未验证**。本集只验证了「证据在语料里存在」，没有验证「当前检索器能召回该页」。抽查时曾出现简单问题召回参考文献页的情况，因此本集不应被当成检索质量基准。
6. **判分标准已结构化但未经回放验证**。每题的 `scoring` 给出了通过阈值、必答引用数、数值容差与必答术语，但没有任何模型输出被用来回放验证这些阈值是否合理；`expected_answer_points` 本身仍是核对清单，不是自动评分器。

7. **术语以语料为准**。ColBERT 论文全文 10 页从未出现 “multi-vector” 一词，本集已改用论文自己的说法（per-token embeddings、late interaction、MaxSim）。同类问题若在其他题出现，一律以语料原文为准，不得使用语料外的术语。

## 评审清单（Reviewer checklist）

- [ ] 每题的 `expected_answer_points` 是否确实只依赖所引用页面？
- [ ] 中文题目与英文题目的表述是否会产生歧义？
- [ ] `expected_citations` 的 page_label 是否与语料实际归属一致？（尤其第 14 题的 page_label 4）
- [ ] `difficulty` 分级是否合理（9 / 10 / 11 / 12 / 15 题偏难）？
- [ ] `scoring.required_point_count` 的阈值是否与题目难度匹配？
- [ ] `scoring.required_terms` 是否都能在对应引用页上找到，且不会误伤正确但换词的答案？
- [ ] `scoring.numeric_tolerance` 的 `null` / `exact` 划分是否符合「定性题 vs 含必答数值题」的约定？
- [ ] 是否需要补入拒答题与去重近重复题？
