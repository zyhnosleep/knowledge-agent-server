# 语料扩充 + RAG 能力基准评测（2026-08-17）

> **来源**：2026-08-17 用户要求——"找 internal-research 类型文章，放 1-200 篇顶刊"（力场/分子模拟领域，授权批量下载 PMC OA + arXiv）；随后补充要求——"有没有外部的语料测评级可以测试我的 RAG 的能力，先用这个进行测试吧"——**优先跑外部标准基准（SciFact）**，语料扩充挂起。
> **状态约定**：`[x]` = 已完成；`[ ]` = 待办。
>
> **现状**：internal-research 项目 13 篇文档（CHARMM/AMBER/OPLS/ff 系列力场 PDF），全部 ready。导入管道 = `POST /api/ingest/upload?project_slug=internal-research`（multipart）→ worker 异步 `run_document_ingestion`。评测 = 30 题 full-answer（strict_pass）+ 30 题 retrieval-cases + canary 单题回归。
>
> **关键风险**：扩库后检索分布变化 → 30 题评测必须回归；新增语料需新 CASES 覆盖。

---

## Ticket 0 — SciFact 外部基准评测（RAG 能力客观定位）

**Blocked by:** None — 用户要求优先

**Status:** in-progress

**What to build:** 用 BEIR/SciFact 标准数据集测系统 RAG 能力：下载 SciFact corpus（5,183 篇 PMC 文献）+ claims（1,409 训练 + 300 测试，带 SUPPORT/CONTRADICT/NEI 标签与 cited_doc_ids 证据）→ 下载 claims 引用文献的 PDF（PMC OA）→ 导入系统（复用 upload API）→ 用 claims 做检索评测（nDCG@10/Recall@10，对标 BEIR 榜单 BM25/bge 等模型）→ claim 判定评测（检索证据 + LLM 判 SUPPORT/CONTRADICT/NEI）。

**进展（2026-08-17）：**
- 语料：BEIR scifact corpus/queries/qrels 已拉取；corpus 文本（title+abstract）→ 967 个 `.md`（`<pmid>-<uuid8>.md`）生成完毕（PDF 路线废弃：SciFact PMID 多不在 PMC）
- 导入：584 篇（test-cited 284 + distractor 300）批量上传 0 失败；file_name 前缀映射 PMID 验证 584/584；上传幂等（sha256 去重，重复传返回同一 document_id）
- 并行：8 个 rq worker 消费队列（含临时 w1-w4），吞吐 ~2.5 篇/分钟（GPU0 ollama 为瓶颈）
- 脚本：`eval_scifact.py`（检索评测，smoke test 通过）、`eval_scifact_claims.py`（相关性判定，部署编译通过）在服务器 runtime/sci-fact-bench/
- **label 缺失**：官方 2021 版 claims 的 evidence/label 字段被清空（S3 旧版 3.7KB/s 不可达）→ 判定评测以 cited_doc_ids 为真值（相关性判定），局限写入报告
- **检索评测出数（16:47）**：nDCG@10 = **0.4422**、Recall@10 = **0.4706**、0 空结果（对标 BM25 0.665 / bge 0.72，低于基线）。双峰分布：hit@1 = 40.3%，未召回 56.7%，几乎无 2-10 名散落命中；平均证据数 3.8 < 10（候选池被截断）。子集语料（584 篇 vs 全库 5,183）干扰更少，真实差距比数字更大
- **claim 判定评测运行中**（17:0x 启动，dev 300 claims，GPU0）：generate_structured 首轮 schema 失败率约 9% 自动降级 JSON 模式重试（qwen3.5 对 format=schema 是软约束），重试全成功

**Acceptance criteria:**
- [x] SciFact 语料（claims 引用文献子集）导入系统，文档 ready（584/584，0 失败）
- [x] 检索评测：nDCG@10 / Recall@10 出数（0.4422 / 0.4706），与 BEIR 公开榜单对比定位（低于基线，见报告）
- [ ] claim 判定评测：证据相关性判定（cited_doc_ids 真值）出数（官方三分类 label 不可得，见上）
- [ ] 评测报告存档（`.task18-corpus/T0-scifact-eval-report.md` + 服务器 runtime/sci-fact-bench/）

**数据文件**：服务器 `~/knowledge-agent-dev/runtime/sci-fact-bench/`（clone 自 allenai/scifact）。

## Ticket 1 — 语料批量下载（150-200 篇）

**Blocked by:** None

**Status:** in-progress

**What to build:** 下载脚本（标准库 urllib，服务器上跑）：arXiv API（physics.chem-ph/q-bio.BM/cond-mat.soft/cs.LG × 力场关键词） + PMC OA 子集（E-utilities + oa.fcgi，按期刊矩阵：JCTC/JCP/JPCB/PCCP/Nat Commun/PNAS/eLife/WIREs 等）→ 去重（标题归一化 + DOI）→ 命名对齐现有格式 `<uuid32>-<slug>.pdf` → 落盘 `runtime/data/raw/internal-research/` → 输出 manifest JSON（文件名/arXiv ID 或 PMCID/标题/期刊/年份/DOI/大小）。

**Acceptance criteria:**
- [ ] 下载 150-200 篇力场/分子模拟相关 PDF（PMC 正式版优先保质量，arXiv 补足数量）
- [ ] 命名格式与现有 13 篇一致（`<uuid32>-<slug>.pdf`）
- [ ] 去重有效（arXiv 预印本 vs 期刊版不重复入库）
- [ ] manifest 含每篇元数据（标题/期刊/年份/来源 ID）
- [ ] 脚本幂等可断点续跑

## Ticket 2 — 批量导入 + 解析验证

**Blocked by:** Ticket 1

**Status:** ready-for-agent

**What to build:** 循环调用 `POST /api/ingest/upload?project_slug=internal-research` 分批（如 25 篇/批）导入；每批后查 `GET /api/documents` 确认状态 ready；parse 失败记录原因。

**Acceptance criteria:**
- [ ] 150-200 篇全部导入，文档状态 ready（或失败有明确原因记录）
- [ ] 每批 canary 抽查：chunk 数、嵌入成功、文档可检索
- [ ] 导入耗时/资源（GPU0 嵌入）可控，无 OOM

## Ticket 3 — 评测回归 + 新题集

**Blocked by:** Ticket 2

**Status:** ready-for-agent

**What to build:** 重跑 30 题 full-answer + retrieval（strict_pass 不回归）；混入 5-10 篇故意不相关文章测误检（multi_source_compare 场景）；从新语料生成新 CASES（题集 30 → 60）。

**Acceptance criteria:**
- [ ] 30 题 strict_pass 重跑结果与扩库前对比，不回归（或回归有分析结论）
- [ ] 新 CASES ≥ 30 题覆盖新语料，纳入评测基线
- [ ] 负样本误检报告
- [ ] 产出扩库评测报告（扩库前后对比）

## Ticket 4 — 术语表/规范化更新

**Blocked by:** Ticket 2

**Status:** ready-for-agent

**What to build:** 新文章引入的新术语走 `active-staged-terms` 机制处理；canonical 表/指标规范化重建（如有）走 dry-run → staged → 正式。

**Acceptance criteria:**
- [ ] 术语冲突/重复处理完，无 blocking 项
- [ ] canonical 重建（如触发）dry-run 验证后再正式

---

## 实施顺序

T1 → T2 → T3（T4 可与 T3 并行）。T1 完成后汇报 manifest 统计再进 T2。
