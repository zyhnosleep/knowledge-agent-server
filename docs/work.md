# 当前任务工作记录

这份文档用于集中记录本轮服务器版 `LLM Wiki` 改造过程中已经遇到的问题、已完成的修复方式，以及当前项目推进到的阶段。

记录范围不是单独某一天的日报，而是覆盖当前对话窗口的完整工作链路：从最初阅读 `2026.6.4` 工作进度、切换到无 Docker 服务器部署，到 `wiki-first` 检索、SAC-KG/LLM Wiki 方法论对齐、Obsidian 协作、PDF 多模态解析、MinerU 接入、query/citation 修复、Git baseline 建立，以及当前服务器实机验收中暴露的问题。

如果后续需要快速回顾“这轮对话到底踩过哪些坑、修到了哪里、下一步该做什么”，优先阅读：

- `六、2026-06-12 工作记录`：记录今天集中处理的 MinerU、query、citation 和服务器问题。
- `七、本对话窗口完整问题索引`：记录整个当前对话窗口累计遇到的主要问题，不限于今天。

## 一、当前推进到哪一步

当前已经完成：

- 无 Docker 服务器版基础可运行链路
- `wiki-first` 查询主路径
- SAC-KG 启发式摄入改造
- Obsidian 协作说明
- query citation 收敛
- 上传随机前缀标题净化
- `/api/query` 是否命中 Ollama 的代码与测试验证

当前所处阶段：

- **代码侧核心改造已基本完成**
- **当前进入“服务器实机验收 + 历史数据重建准备”阶段**

这意味着：

- 现在最重要的不是继续快速加新功能
- 而是把已有代码在真实服务器数据上验证清楚
- 并逐步把旧知识库重建成新版结构

## 二、已经遇到的问题与解决方法

### 1. 服务器无法依赖 Docker / sudo

问题：

- 当前服务器环境不适合继续依赖 Docker 或 systemd
- 需要一版用户态可运行方案

解决方法：

- 改为 `FastAPI + Redis worker + SQLite + 本地文件 + Ollama + nohup`
- 增加无 Docker 部署说明与启动脚本
- 所有命令按 POSIX `sh` 兼容方式整理，不依赖 `source`

当前状态：

- 已解决

### 2. Windows / Linux 路径格式不一致，影响 wiki 链接和索引

问题：

- 早期 `index.md` 链接依赖持久化路径字符串
- Windows 路径分隔符和 Linux 不一致，导致生成结果不稳定

解决方法：

- `index.md` 改为基于逻辑 `slug` 生成链接
- 不再依赖持久化的 `markdown_path` 作为展示层路径来源

当前状态：

- 已解决

### 3. 初始查询逻辑过于接近传统 RAG，未体现 `llm_wiki` 的 wiki-first 思路

问题：

- 查询容易直接退回 raw chunk
- wiki 页面虽然存在，但没有成为主知识入口

解决方法：

- 查询逻辑改为优先检索 `WikiPage`
- 增加 wiki 页面打分、key term、verified claim count 等权重
- 只有 wiki 缺失信息或用户明确要求证据时才回退 raw chunk

当前状态：

- 已基本解决

### 4. 文档里明明有事实，但回答仍然说“回答不了”或答偏

问题：

- 不是单纯检索不到，而是 wiki source 页面信息密度不够
- 摄入早期更像“粗摘要”，没把复查时间、用药、诊断等事实写清楚

解决方法：

- source 页面增强为：
  - `Key Facts`
  - `Verified Triples`
  - `Evidence Notes`
- 摄入从文档级一次抽取，调整为更接近 head-driven 的 triple 生成

当前状态：

- 已明显改善，但仍需服务器实机继续验证

### 5. 早期 chunk / prompt 方式不合理，不像 `llm_wiki`

问题：

- 用户明确指出不能只是“截断前 200 个字符”式逻辑
- 当前项目应更接近 `llm_wiki` 的 wiki 编译思路

解决方法：

- 不再把问题归结为简单截断
- 摄入链路改为：
  - 先抽取 candidate heads
  - 再按 head 检索 snippets
  - 再生成 triples
- 查询链路改为：
  - 先问 wiki
  - 再必要时看 raw

当前状态：

- 已完成核心改造

### 6. SAC-KG 的 Generator / Verifier / Pruner 起初只实现了轻量近似版

问题：

- 最初只是“受启发”，并没有真正做到：
  - head-driven generation
  - tail-based pruning
  - 错误类型驱动 verifier

解决方法：

- 增加 head-driven 生成
- 增加本地 open KG 示例检索
- verifier 增加：
  - quantity too small
  - format error
  - head entity error
  - head-tail contradiction
  - missing evidence
  - duplicate
  - potential conflict
- pruner 改为针对 verified triple 的 tail 做 `grow / keep / prune`

当前状态：

- 代码侧已完成第一版
- 仍需真实数据继续验收

### 7. query 没有写回 `wiki/queries/`

问题：

- 早期问答虽然返回结果，但知识没有持续沉淀

解决方法：

- 保持 `save_answer=true` 时写回 `wiki/queries/`
- 同步更新 `index.md` 和 `log.md`

当前状态：

- 已解决

### 8. citations 很乱，夹杂无关页面与空实体页

问题：

- `sample` 页面、空 entity 页、弱相关页混入 prompt
- 导致回答正文和引用一起漂

解决方法：

- 增加 wiki 页真实重叠门槛
- 跳过 `No summary available` / `No claims yet` 这类空页
- 过滤弱相关页面
- 收紧 citations 默认返回数量

当前状态：

- 已解决主要污染问题

### 9. citations 返回 `page_slug: null`，仍以 raw chunk 为主

问题：

- 即使答案更接近 wiki，返回的仍可能是 raw chunk citation

解决方法：

- 当问题不是明确索要原文/证据时：
  - 优先把 raw chunk citation 提升为对应的 `source_summary` wiki citation
- 同时支持从答案正文中的 `[1]` 这类索引反推 citation 选择

当前状态：

- 已解决

### 10. 上传时自动加的随机前缀污染标题、slug、citation title

问题：

- 上传文件名带 `uuid-原文件名`
- 这个前缀一路进入：
  - `Document.title`
  - `WikiPage.title`
  - `citation.page_title`

解决方法：

- 增加统一标题归一化 helper
- 在：
  - 上传记录阶段
  - 解析阶段
  - 查询返回阶段
 统一去除随机前缀

当前状态：

- 已解决
- 新数据会更干净
- 历史 wiki 文件名仍需通过重建进一步清理

### 11. Obsidian 的角色容易和后端处理链路混淆

问题：

- 容易误以为 Obsidian 本身就是知识引擎
- 或误以为上传 Markdown 后完全不需要内部 chunk/snippet

解决方法：

- 明确区分：
  - 服务器负责 ingest / query / verify / writeback
  - `wiki/` 是知识工件
  - Obsidian 是本地查看和策展界面
- 补充 `obsidian-workflow.md`

当前状态：

- 已澄清

### 12. 不清楚 `/api/query` 是否真的调用了 Ollama

问题：

- 因为 API 返回成功并不代表模型一定成功执行
- 代码里有 fallback，容易误判

解决方法：

- 从代码上确认：
  - `/api/query` -> `QueryService.answer()` -> `_draft_answer()` -> `ollama.generate_structured()`
  - raw chunk fallback 时还可能调用 `ollama.embed()`
- 增加测试验证 query 调用链
- 补充“如何判断 query 是否真实命中 Ollama”的命令说明

当前状态：

- 已确认 query 路径会调用 Ollama
- 但单次线上请求是否真的命中，还要结合日志判断是否 fallback

### 13. PDF 能否使用、是否由模型解析这件事没有说清楚

问题：

- 容易误以为 PDF 解析是由本地 `qwen3.6` 完成
- 但实际上我们还没有做 PDF 实机验收

解决方法：

- 明确标注：
  - `pdf` 当前是解析入口已接通
  - 本地文本提取使用 `pypdf.PdfReader.extract_text()`
  - Ollama 只参与 PDF 文本提取之后的知识处理
- 同时明确：
  - 扫描型 PDF / 图片型 PDF / 复杂双栏表格 PDF 风险较高

当前状态：

- 已标注文档
- **尚未完成 PDF 实机测试**

## 三、目前还没有解决、或尚未完成验证的点

### 1. 历史知识库尚未完成系统性重建

- 旧文档还能查询
- 但很多历史 wiki 页面仍然是旧逻辑生成的
- 要真正体现新逻辑效果，需要对历史文档重建

### 2. PDF / DOCX / HTML 尚未完成真实服务器验收

- 代码支持入口不等于已验证稳定
- 当前最稳定输入仍然是 `md` / `txt`

### 3. 历史带前缀 wiki 文件名 / slug 尚未彻底清理

- 查询返回标题已能净化
- 但旧 wiki 文件本身仍可能保留带前缀名字
- 需要靠历史数据重建来彻底处理

## 四、下一步应该做什么

### 第一优先级：历史数据重建

- 选 1 个真实项目做试点
- 重新 ingest 历史文档
- 生成新版 `sources/`、`entities/`、`queries/`
- 清理旧前缀页面

### 第二优先级：服务器实机验收

- 固定问题集验收：
  - 复查时间
  - 主要诊断
  - 当前用药
  - 医生建议
  - 原文/证据/引用
- 验证：
  - 普通问题优先 wiki citation
  - 要原文时才回退 raw chunk
  - citations 数量收敛
  - 页面标题自然

### 第三优先级：多格式输入验收

- 重点先测：
  - 文本型 PDF
  - 扫描型 PDF
  - 简单 DOCX
  - 简单 HTML
- 明确每类格式的真实可用边界

### 第四优先级：Obsidian 联调

- 将 `wiki/` 同步到本地 Vault
- 检查 wikilink、frontmatter、目录跳转是否符合预期

## 五、结论

当前项目已经从“能跑通的服务器骨架”推进到了“核心知识编译逻辑基本成型”的阶段。

现在最重要的不是再继续快速堆功能，而是：

- 用真实历史数据重建知识库
- 做服务器实机验收
- 明确多格式输入的真实边界

等这些稳定以后，再继续往下推进更深入的 KG 化扩展或更复杂的自动知识整理能力。

---

## 六、2026-06-12 工作记录：MinerU PDF 集成、Query 质量修复与服务器实机问题

### 1. 今日整体进展

今天的主要工作从“PDF 解析质量提升”推进到“MinerU 接入 + Query 质量收敛 + 服务器实机排错”。

已经完成：

- 初始化 Git baseline，并形成后续分支 / PR 工作方式的基础。
- 新增 MinerU PDF 解析能力，作为 PDF 解析第一优先级。
- 修复 MinerU CLI 路径问题、MinerU v2 输出解析问题、MinerU markdown 表格兜底问题。
- 修复 query 多意图覆盖、表格/指标召回、数字反幻觉、citation 编号错位等问题。
- 修复服务器 `sh` 环境下启动脚本对 `.venv/bin/activate` 的兼容问题。
- 增加 `OLLAMA_KEEP_ALIVE`，用于减少 Ollama 长时间占用 3090 显存。
- 明确服务器上无需系统 `sqlite3` 命令，可用 Python 标准库清理 SQLite 数据。

当前测试状态：

- 最新本地验证为 `python -m compileall src tests` 通过。
- 最新本地测试为 `pytest -q` 通过，当前为 `56 passed`。

### 2. 今日遇到的问题与处理

#### 2.1 Git baseline 与远端仓库权限

问题：

- 项目此前没有 Git baseline，导致多轮改动后无法清晰审查“这次到底改了哪些文件”。
- GitHub MCP 当前 token 没有创建私有仓库权限。
- 本机未安装 `gh` CLI，无法自动创建 GitHub 私有仓库。

处理：

- 本地初始化 Git 仓库。
- 新增 `.gitignore`，排除 `.env`、`data/`、`logs/`、`run/`、`tmp/`、缓存、PDF 原文等。
- 新增 `.gitattributes`，固定脚本和代码使用 LF，避免 Linux 服务器执行 `.sh` 脚本出问题。
- 完成初始提交：
  - `9958f82 Initial LLM Wiki server baseline`
- 后续形成多个独立提交，方便服务器 `git pull` 更新。

当前状态：

- 本地 Git baseline 已建立。
- GitHub 私有仓库需要用户手动创建后再 push。

#### 2.2 服务器 `sh` 环境下启动脚本报 `OSTYPE: parameter not set`

问题：

- 服务器 shell 是 `sh`，不是 bash。
- `scripts/start_api.sh` 和 `scripts/start_worker.sh` 在 `set -u` 下执行 `.venv/bin/activate`，触发 `OSTYPE: parameter not set`。

处理：

- API / worker 启动脚本不再 `source` 或 `.` activate。
- 改为直接调用 `.venv/bin/python`。
- 同时增强 Redis / Ollama / status 脚本：
  - Redis 已可 `PING` 时视为 running。
  - Ollama `/api/tags` 可访问时视为 running。
  - `status.sh` 不再只依赖 PID 文件。

相关提交：

- `d8cc657 Fix sh-compatible server startup scripts`

当前状态：

- 已解决。

#### 2.3 Redis 显示 stopped，但端口已经被占用

问题：

- `start_redis.sh` 只检查 `run/redis.pid`。
- 实际 Redis 已经监听 `127.0.0.1:6379`，但 PID 文件不匹配，脚本误判为 stopped 并重复启动。

处理：

- `start_redis.sh` 增加 `redis-cli ping` 判断。
- `status.sh` 增加真实连通性检查。

当前状态：

- 已解决。

#### 2.4 MinerU 本地服务器部署可行性

问题：

- 需要判断服务器本地部署 MinerU 是否可行。
- 用户服务器配置为：
  - RTX 3090
  - 24GB 显存
  - CPU 内存 90GB

判断：

- 该硬件足够运行 MinerU pipeline。
- 真正风险不是硬件不足，而是 MinerU 和 Ollama Qwen 模型同时占用 GPU。

处理：

- MinerU 采用 CLI 子进程方式运行。
- 设计为 GPU sequential mode：
  - PDF -> MinerU 解析
  - MinerU 进程退出释放显存
  - 再进入 Ollama 抽取 / Verifier / Pruner / 问答
- 增加 `OLLAMA_KEEP_ALIVE=5m` 配置，必要时可临时改成 `0`。

当前状态：

- 方案可行，已进入代码实现与服务器验收阶段。

#### 2.5 MinerU 依赖计划修正

问题：

- 早期计划里写的是 `magic-pdf[full]>=1.0.0`，偏旧。
- 官方当前更推荐 `mineru` 包和 CLI。

处理：

- 改为可选依赖：
  - `mineru = ["mineru[pipeline]>=2.0.0"]`
- `.env` 新增：
  - `MINERU_ENABLED`
  - `MINERU_BIN`
  - `MINERU_BACKEND`
  - `MINERU_MODEL_SOURCE`
  - `MINERU_OUTPUT_DIR`
  - `MINERU_TIMEOUT`
  - `MINERU_EXTRA_ARGS`
- 服务器模板默认：
  - `MINERU_MODEL_SOURCE=modelscope`

相关提交：

- `c7f62de Add optional MinerU PDF parser integration`

当前状态：

- 已完成。

#### 2.6 MinerU CLI 报 PDF 路径不存在

问题：

服务器日志：

```text
MinerU exited with code 2:
Error: Invalid value for '-p' / '--path':
Path 'data/raw/internal-research/...-Knowledge graph.pdf' does not exist.
```

原因：

- 数据库保存的是相对路径 `data/raw/...pdf`。
- 调用 MinerU 时又把子进程 `cwd` 切到 PDF 所在目录。
- 最终 MinerU 在错误目录下查找相对路径，导致路径重复拼接。

处理：

- `_parse_pdf_with_mineru()` 中把 PDF 输入路径 `resolve()` 成绝对路径。
- MinerU 输出目录也改为绝对路径。
- 增加测试验证传给 CLI 的 `-p` 和 `-o` 都是绝对路径。

相关提交：

- `fd26bf8 Fix MinerU CLI path handling`

当前状态：

- 已解决。

#### 2.7 MinerU 输出为空

问题：

服务器日志：

```text
MinerU output was empty: ..._content_list_v2.json
```

实际原因：

- MinerU v2 的 `content_list_v2.json` 常见结构是按页面分组的二维 list。
- 原解析器只支持扁平 list，因此把页面列表过滤掉，误判为空。

处理：

- `_normalize_mineru_content_list()` 支持：
  - 二维 page list
  - page object 中的 `blocks`
  - `content` 嵌套字段
  - `title_content`
  - `paragraph_content`
  - `table_content`
  - `math_content`
- v2 解析不到时继续 fallback 到旧 `content_list.json`。
- 增加 JSON 结构描述日志，便于后续排查。

相关提交：

- `3e7e882 Support MinerU v2 content list output`

当前状态：

- 已解决。

#### 2.8 Table 2 / Table 5 仍未进入 wiki

问题：

- 重新 ingest 后，问：
  - `ablation studies 得出了什么结论？请引用 Table 2`
  - `SAC-KG 在 OIE2016 或 NYT 数据集上的指标是什么？`
- 系统能找到相关章节文字，但无法返回 Table 2 / Table 5 中的具体数值。

分析：

- 当前已是 MinerU 优先，不是先走 Ollama Vision。
- 真正问题更可能是：
  - MinerU JSON 中表格字段未被现有代码识别。
  - 或 MinerU 把表格写入 `.md`，但我们之前没有读取 markdown 作为兜底。
- 因此不能简单归因于“PyMuPDF -> Ollama Vision 表格识别失败”，因为当前主路径已经是 MinerU。

处理：

- 在读取 MinerU JSON 后，继续查找 MinerU 输出目录中的 `.md`。
- 从 MinerU markdown 中提取标准 markdown table。
- 将补救表格写入：
  - `document_intelligence.tables`
  - `ParsedChunk`
  - `parsed.text`

相关提交：

- `b415f82 Recover MinerU markdown tables and stabilize citations`

当前状态：

- 代码已修复。
- 需要重新清理旧 ingest 结果并重新上传 PDF 才能验证。
- 验证重点：
  - `grep -R "Table 2" data/wiki/internal-research/sources`
  - `grep -R "Table 5" data/wiki/internal-research/sources`
  - `grep -R "OIE2016" data/wiki/internal-research/sources`
  - `grep -R "NYT" data/wiki/internal-research/sources`

#### 2.9 Query 回答里 citation 编号和返回 citations 不一致

问题：

- 回答正文出现 `[1]`、`[2]`、`[3]`、`[4]`。
- 但 API 返回的 `citations` 可能只有 1 条。

原因：

- 模型会在 answer markdown 中生成上下文编号。
- `_infer_citation_indexes()` 会过滤超出 contexts 范围的编号。
- 最终返回 citations 后，答案正文里的编号没有重新映射。

处理：

- 新增 `_choose_citation_indexes()`：
  - 合并模型返回的 citations。
  - 合并答案正文中的 `[n]`。
  - 合并 query facets 对应上下文。
- 新增 `_renumber_answer_citations()`：
  - 将答案中的上下文编号映射到最终返回 citations 的编号。
- 如果最终没有 citations，则去掉答案里的 citation markers。

相关提交：

- `b415f82 Recover MinerU markdown tables and stabilize citations`

当前状态：

- 已修复。

#### 2.10 中文“请引用 Table 2”误触发 raw chunk citation

问题：

- 用户问“请引用 Table 2”时，系统把“引用”理解成用户要求原文证据。
- 因此 `_needs_source_evidence()` 返回 true，导致不能提升到 wiki citation，更容易返回 raw chunk。

处理：

- 从 `_needs_source_evidence()` 的中文 marker 中移除泛化的“引用”。
- 保留更明确的：
  - `原文`
  - `出处`
  - `证据`
  - `摘录`
  - `quote`
  - `exact`
  - `verbatim`
  - `source`

相关提交：

- `b415f82 Recover MinerU markdown tables and stabilize citations`

当前状态：

- 已修复。

#### 2.11 多实体查询只返回 1 个 citation

问题：

- 问：
  - `SAC-KG 的 Generator、Verifier、Pruner 分别做什么？`
- 只返回一个 citation。

原因：

- query 虽然回答了多个组件，但 citation 选择过度依赖模型返回。
- facet context 即使命中，也未必进入最终 citations。

处理：

- `_choose_citation_indexes()` 会把 query facets 对应的上下文也纳入 citation 选择。
- facets 包括：
  - `Generator`
  - `Verifier`
  - `Pruner`
  - `Retriever`
  - `OIE2016`
  - `NYT`
  - `Table N`
  - `Figure N`

相关提交：

- `917ae14 Improve query facet coverage and numeric evidence checks`
- `b415f82 Recover MinerU markdown tables and stabilize citations`

当前状态：

- 代码已修复。
- 需要重新 ingest 后继续服务器验证。

#### 2.12 数字反幻觉修复路径中的 NameError

问题：

- Claude Code 指出 `_repair_unsupported_numeric_answer()` 中仍残留 `supported_contexts`。
- 实际代码确实存在：
  - `citations=list(range(len(supported_contexts)))`
  - `self._build_answer_constraints(question, supported_contexts)`
- 但变量已改名为 `supported_pairs`。

处理：

- 改成基于 `supported_pairs`。
- 增加测试覆盖 unsupported numeric answer repair 路径。

相关提交：

- `71d6987 Complete query quality follow-up fixes`

当前状态：

- 已解决。

#### 2.13 数字正则未识别句末数字

问题：

- `88.8.` 这种句末数字没有被识别为 unsupported number。
- 导致反幻觉修复没有触发。

处理：

- 调整 `_answer_numbers()` 正则，支持句末数字。
- 增加测试：
  - `OIE2016 F1 74.7` 有证据
  - `NYT F1 88.8` 无证据
  - 应识别 `88.8` 为 unsupported

相关提交：

- `71d6987 Complete query quality follow-up fixes`

当前状态：

- 已解决。

#### 2.14 Wiki 渲染上限不足

问题：

- 早期 `wiki.py` 只渲染：
  - `page_outputs[:8]`
  - `tables[:4]`
  - `figures[:6]`
- PDF 论文中 Table 2 / Table 5 / 多图注可能被截断。

处理：

- 改为：
  - `page_outputs[:12]`
  - `tables[:8]`
  - `figures[:12]`
- 增加测试确保 Table 8、Figure 12 可以进入 wiki。

相关提交：

- `71d6987 Complete query quality follow-up fixes`

当前状态：

- 已解决。

#### 2.15 Head 数量与学术概念实体扩展不足

问题：

- `HEAD_MAX_COUNT=6` 对论文类文档偏少。
- `GROW_ENTITY_TYPES` 不包含：
  - `concept`
  - `component`
  - `module`
- 导致 Generator / Verifier / Pruner 这类学术概念不容易生成 entity page。

处理：

- `HEAD_MAX_COUNT=10`
- `GROW_ENTITY_TYPES` 增加：
  - `concept`
  - `component`
  - `module`
- 无 verified claim 支撑的 tail concept 不直接生成独立页面，避免空实体页膨胀。

相关提交：

- `71d6987 Complete query quality follow-up fixes`

当前状态：

- 已解决。

#### 2.16 服务器没有 `sqlite3` 命令

问题：

- 执行清理命令时报：

```text
-sh: sqlite3: not found
```

原因：

- 用户无权限安装系统 sqlite3。

处理：

- 改用 Python 标准库 `sqlite3` 模块清理数据库。
- 不依赖 sudo，不依赖系统命令。

当前状态：

- 已给出替代方案。

### 3. 当前最新代码提交

今天相关关键提交包括：

- `d8cc657 Fix sh-compatible server startup scripts`
- `c7f62de Add optional MinerU PDF parser integration`
- `fd26bf8 Fix MinerU CLI path handling`
- `3e7e882 Support MinerU v2 content list output`
- `917ae14 Improve query facet coverage and numeric evidence checks`
- `71d6987 Complete query quality follow-up fixes`
- `b415f82 Recover MinerU markdown tables and stabilize citations`

### 4. 当前仍需要服务器继续验证的事项

#### 4.1 必须重新清理并重新 ingest PDF

原因：

- Table 2 / Table 5 是否进入 wiki，是 ingest 阶段决定的。
- 只更新 query 代码不会自动补全旧 wiki。
- MinerU markdown table 兜底也需要重新 ingest 才会生效。

建议步骤：

1. 备份 `data/`。
2. 停止 API / worker。
3. 用 Python sqlite3 清理 ingest 相关表。
4. 清理：
   - `data/raw/internal-research`
   - `data/wiki/internal-research`
   - `data/cache/mineru`
5. 重启 API / worker。
6. 重新上传 PDF。
7. 观察 `logs/worker.log`。

#### 4.2 重新 ingest 后优先检查 wiki source 内容

先不要急着问答，先检查 source page 是否含表格：

```sh
grep -R "Table 2" data/wiki/internal-research/sources
grep -R "Table 5" data/wiki/internal-research/sources
grep -R "OIE2016" data/wiki/internal-research/sources
grep -R "NYT" data/wiki/internal-research/sources
```

如果这些仍搜不到，需要进一步打开：

- `data/cache/mineru/.../*.md`
- `data/cache/mineru/.../*content_list_v2.json`

确认 MinerU 原始输出中到底有没有 Table 2 / Table 5。

#### 4.3 重新运行四问脚本

已生成：

```text
tmp/sac_kg_query_batch.sh
```

服务器运行：

```sh
sh tmp/sac_kg_query_batch.sh
```

输出会保存到：

```text
tmp/query_results/
```

重点观察：

- `SAC-KG 是什么？`
  - 不应乱展开缩写。
  - citation 编号应与返回 citations 对齐。
- `Generator / Verifier / Pruner`
  - 应逐项回答。
  - citations 不应只有一个笼统摘要。
- `OIE2016 / NYT 指标`
  - 如果 Table 5 数字进入上下文，则应报告具体数值。
  - 如果没进入上下文，则应明确说当前材料缺少具体数值，不能脑补。
- `ablation studies / Table 2`
  - 如果 Table 2 进入 wiki，应报告具体结论和数值。
  - 如果没进入 wiki，应明确指出 Table 2 缺失。

### 5. 后续工作建议

#### 第一优先级：完成 MinerU 重 ingest 验收

目标：

- 证明 `PDF -> MinerU -> wiki tables -> query` 这条链路真实有效。

验收标准：

- `sources/*.md` 中能看到 Table 2 / Table 5。
- `document_intelligence.tables` 中包含这些表。
- query 能引用 wiki page citation，而不是 raw chunk。

#### 第二优先级：如果 MinerU markdown 也没有 Table 2 / Table 5

需要进一步判断：

- MinerU 是否使用了正确 backend。
- PDF 渲染页码是否正常。
- MinerU 输出目录里是否有其它格式文件，例如中间 JSON、layout JSON、span JSON。
- 是否需要调整 MinerU 参数或启用更强后端。

#### 第三优先级：继续收敛 citation 显示质量

当前已经修复编号错位，但还要观察：

- citation excerpt 是否足够贴近答案依据。
- 多 citation 是否太多或太少。
- wiki page citation 是否仍会退回 raw chunk。

#### 第四优先级：整理服务器运维脚本

可以考虑新增：

- `scripts/clean_project_data.py`
- `scripts/reingest_pdf_checklist.sh`
- `scripts/query_sac_kg_eval.sh`

这样后续不用手写大段清理和验收命令。

### 6. 今日结论

今天项目从“PDF 能跑”推进到了“开始认真处理 PDF 表格质量、citation 对齐和真实服务器验收问题”的阶段。

当前最重要的判断是：

- MinerU 已经接入，但必须确认其原始输出中是否真的包含 Table 2 / Table 5。
- Query 层已经加强，但如果 ingest 阶段没有把表格写进 wiki，query 不能凭空回答。
- 后续关键工作不是再盲目改 search，而是先完成一次干净的重 ingest，并对 `data/cache/mineru` 和 `data/wiki` 做逐层核验。

---

## 七、本对话窗口完整问题索引

本节不是只记录单日工作，而是整理当前对话窗口中从最初无 Docker 服务器版部署到 MinerU PDF 解析、query 验收、Git baseline 的所有主要问题。它用于后续快速回顾“我们到底遇到过哪些坑、修到哪里、还剩什么”。

### 1. 部署与服务器环境类问题

#### 1.1 不能使用 Docker

问题：

- 用户当前服务器暂时不能使用 Docker。
- 原计划里的 Docker Compose / MinIO / PostgreSQL 不适合当前阶段。

处理：

- 固定为无 Docker 双进程版：
  - FastAPI API 单独运行
  - RQ worker 单独运行
  - Redis 本机用户态运行
  - SQLite 本地数据库
  - 文件落本地 `data/`
  - Ollama 使用宿主机本地端口
  - `nohup` 托管进程

当前状态：

- 已形成无 Docker 部署文档和启动脚本。

#### 1.2 服务器 shell 是 `sh`，不支持 `source`

问题：

- 文档和脚本最初容易写成 bash 风格。
- 用户明确指出服务器 shell 是 `sh`。

处理：

- 文档命令统一使用：

```sh
. .venv/bin/activate
```

- 避免使用 `source`。
- 后续 API / worker 启动脚本甚至不再依赖 activate，直接调用 `.venv/bin/python`。

当前状态：

- 已解决。

#### 1.3 `.env` 手动配置与重复变量

问题：

- 用户希望手动改 `.env`。
- `.env` 里曾出现重复：

```dotenv
OLLAMA_REQUEST_TIMEOUT=180
OLLAMA_REQUEST_TIMEOUT=601
```

处理：

- 整理最终 `.env` 推荐值。
- 固定：

```dotenv
OLLAMA_REQUEST_TIMEOUT=600
QUEUE_JOB_TIMEOUT=3600
OLLAMA_KEEP_ALIVE=5m
MINERU_ENABLED=true
```

当前状态：

- 已给出最终配置模板。

#### 1.4 `.env` 中带空格的值在 `sh` 下需要引号

问题：

- `APP_NAME=LLM Wiki Server`
- `DEFAULT_PROJECT_NAME=Internal Research`
- 在 `sh` 读取时容易被拆分。

处理：

```dotenv
APP_NAME="LLM Wiki Server"
DEFAULT_PROJECT_NAME="Internal Research"
```

当前状态：

- `.env.example` 和 `.env.server.example` 已修正。

#### 1.5 Redis 已运行但脚本误判 stopped

问题：

- Redis 实际已占用 `127.0.0.1:6379`。
- `status.sh` 只看 PID 文件，显示 stopped。
- `start_redis.sh` 再次启动时报：

```text
bind: Address already in use
Failed listening on port 6379
```

处理：

- `start_redis.sh` 增加 `redis-cli ping` 检查。
- `status.sh` 增加真实连通性判断。

当前状态：

- 已解决。

#### 1.6 Ollama / MinerU 共享 3090 显存

问题：

- 服务器 GPU 为 RTX 3090 24GB。
- Ollama Qwen 模型和 MinerU 如果同时跑，可能抢显存。

处理：

- 采用 GPU sequential mode：
  - MinerU 先跑 PDF 解析
  - MinerU 子进程退出
  - 再进入 Ollama 抽取 / 问答
- 增加：

```dotenv
OLLAMA_KEEP_ALIVE=5m
```

- 必要时可临时改成：

```dotenv
OLLAMA_KEEP_ALIVE=0
```

当前状态：

- 方案已确定并写入脚本 / 文档。

#### 1.7 服务器没有 `sqlite3` 命令

问题：

```text
-sh: sqlite3: not found
```

- 用户没有权限安装系统命令。

处理：

- 改用 Python 标准库 `sqlite3` 清理数据库。
- 不依赖 sudo、不依赖系统 sqlite3。

当前状态：

- 已给出替代命令。

### 2. API / Worker / Ingest 链路类问题

#### 2.1 Worker 是否正确消费队列

问题：

- 用户发现有文档 `processing`，但 run 已经 failed。
- 早期还出现两份 worker / 任务状态判断不清。

处理：

- 通过：

```sh
curl http://127.0.0.1:8000/api/documents
curl http://127.0.0.1:8000/api/runs
tail -f logs/worker.log
```

确认真实状态。

当前状态：

- 已建立基本排查路径。

#### 2.2 文档状态卡在 processing / run 状态 running

问题：

- 进程异常退出或 worker 报错时，文档可能停留在 processing。

处理：

- 通过 runs 和 worker 日志判断。
- 必要时把文档状态改成 `failed` 后重新上传触发 requeue。

当前状态：

- 已有操作方案。

#### 2.3 重复上传同一个 PDF 被 sha256 去重

问题：

- 同一个文件如果已有 `ready` 文档，重复上传会返回：

```text
Duplicate document skipped.
```

- 因此不会重新解析。

处理：

- 若需要重新 ingest 同一个文件：
  - 将旧 document 状态改为 `failed`
  - 或清理旧数据后重新上传

当前状态：

- 已明确。

#### 2.4 进度不可见

问题：

- 用户等待 PDF ingest 时，不知道是否还在处理。
- 曾讨论“页面进度条”，但用户真正想要的是终端进度显示。

处理：

- 增加/使用终端侧进度查看方式：

```sh
tail -f logs/worker.log
python scripts/watch_ingest_progress.py
```

当前状态：

- 已给出终端观察方式。

#### 2.5 `cosine_similarity` 未定义

问题：

worker 日志报：

```text
NameError: name 'cosine_similarity' is not defined
```

原因：

- pipeline 中调用了相似度函数但未正确 import。

处理：

- 修复 import。

当前状态：

- 已解决。

#### 2.6 `_merge_document_analysis()` 中 keywords 合并报 unhashable list

问题：

worker 日志报：

```text
TypeError: unhashable type: 'list'
```

原因：

- `_derive_keywords(payload.summary)` 返回 list。
- 代码将 list 当成单个元素塞进 `dict.fromkeys()`。

处理：

- 展平 keywords 后再去重。

当前状态：

- 已解决。

### 3. 文件路径与跨平台问题

#### 3.1 Windows 路径导致 wiki 相对路径判断失败

问题：

日志中出现：

```text
'data\\wiki\\internal-research\\sources\\...' is not in the subpath of 'data/wiki/internal-research'
```

原因：

- Windows 风格反斜杠和 Linux 风格路径混用。

处理：

- 改为基于 slug 生成 wiki 链接。
- 避免依赖持久化 `markdown_path` 做相对路径计算。

当前状态：

- 已解决。

#### 3.2 上传文件随机前缀污染标题

问题：

- 上传文件会保存成：

```text
uuid-original-name.pdf
```

- 前缀污染：
  - document title
  - wiki title
  - page slug
  - citation title

处理：

- 增加前缀清理 helper。
- 查询返回 title 也进行清理。

当前状态：

- 新数据已改善。
- 历史页面需要重建才能完全清理。

### 4. PDF 与 MinerU 解析类问题

#### 4.1 早期 PDF 只是文本层抽取

问题：

- 早期 PDF 解析主要依赖 pypdf 文本层。
- 扫描 PDF、复杂表格、双栏论文、公式页质量不足。

处理：

- 先升级为 PyMuPDF 渲染 + Ollama Vision。
- 后续进一步引入 MinerU。

当前状态：

- 当前主计划已转向 MinerU 优先。

#### 4.2 Ollama Vision 表格转 markdown 不可靠

问题：

- Table 1 可能识别成功。
- Table 2 / Table 5 这类复杂表格容易丢失或不完整。

处理：

- 引入 MinerU 专用 PDF 结构化解析。

当前状态：

- MinerU 已接入。
- 仍需确认 MinerU 原始输出是否包含目标表格。

#### 4.3 MinerU 依赖写法从旧 `magic-pdf` 修正为 `mineru`

问题：

- 初始计划写的是：

```sh
pip install "magic-pdf[full]>=1.0.0"
```

- 与当前官方推荐不完全一致。

处理：

- 改为：

```sh
pip install -e ".[mineru]"
```

- optional dependency：

```toml
mineru = ["mineru[pipeline]>=2.0.0"]
```

当前状态：

- 已修正。

#### 4.4 MinerU CLI 输入相对路径错误

问题：

```text
Error: Invalid value for '-p' / '--path': Path 'data/raw/...pdf' does not exist.
```

处理：

- PDF 路径和输出目录都传绝对路径。

当前状态：

- 已解决。

#### 4.5 MinerU v2 输出被误判为空

问题：

```text
MinerU output was empty: ...content_list_v2.json
```

原因：

- v2 输出是 page list / nested content。
- 旧 parser 只支持扁平 dict list。

处理：

- 支持二维 page list、nested blocks、`content.*` 字段。

当前状态：

- 已解决。

#### 4.6 MinerU JSON 没有把 Table 2 / Table 5 写进系统表格

问题：

- Query 已经能定位到 ablation 章节文字。
- 但 Table 2 的具体数值缺失。
- Table 5 的 OIE2016 / NYT 数字缺失。

分析：

- 当前已是 MinerU 优先，不应简单归因为 Ollama Vision。
- 可能是：
  - MinerU JSON 表字段未识别。
  - MinerU markdown 有表，但系统之前没读。
  - MinerU 本身输出中就没有这些表。

处理：

- 新增 MinerU markdown 表格兜底读取。

当前状态：

- 代码已修。
- 需要重新 ingest 后验证。

### 5. Wiki / Obsidian / 知识工件问题

#### 5.1 Obsidian 角色不清

问题：

- 用户关心服务器部署后还能不能用 Obsidian。
- 容易误解为 Obsidian 替代后端处理链路。

处理：

- 明确：
  - 服务器负责 ingest / query / writeback。
  - `wiki/` 是主知识工件。
  - Obsidian 用于本地阅读、策展、双链浏览。

当前状态：

- 已写入 `docs/obsidian-workflow.md`。

#### 5.2 query 没有写回 `wiki/queries`

问题：

- 问答结果没有沉淀成知识工件。

处理：

- `save_answer=true` 时写入 `wiki/queries/`。
- 更新 `index.md` 和 `log.md`。

当前状态：

- 已解决。

#### 5.3 wiki 页面内容密度不足

问题：

- 文档事实存在，但问答说回答不了。
- 原因是 source page 中没有充分保留关键事实、triples、evidence。

处理：

- 增加：
  - key facts
  - verified triples
  - evidence notes
  - source summary

当前状态：

- 已改善。

#### 5.4 wiki table / figure 渲染上限不足

问题：

- 早期只渲染：
  - tables 前 4
  - figures 前 6
  - page_outputs 前 8

处理：

- 提升为：
  - tables 前 8
  - figures 前 12
  - page_outputs 前 12

当前状态：

- 已解决。

### 6. Query / Citation / RAG 行为问题

#### 6.1 早期逻辑过于 raw chunk-first

问题：

- 用户指出项目应参考 `llm_wiki`，不是普通 RAG。

处理：

- 改为 wiki-first：
  - 先检索 wiki page。
  - 再必要时 fallback raw chunk。

当前状态：

- 已基本完成。

#### 6.2 page_slug 为 null

问题：

- citations 返回 raw chunk，`page_slug=null`。

处理：

- 普通问题尽量提升到 wiki page citation。
- 明确要求原文证据时才保留 raw chunk。

当前状态：

- 已改善，但仍需实机观察。

#### 6.3 Figure / Table 检索窗口落在页面开头

问题：

- 问 Figure 1 时，excerpt 可能截到页面开头，而不是图注。

处理：

- `_window_text()` 优先锚定：
  - Figure N
  - Table N
  - dataset name

当前状态：

- 已解决主要问题。

#### 6.4 Table / metric query 命中错误表格

问题：

- 问 OIE2016 / NYT，可能命中 Table 1 或 Page 6，而不是 Table 5。

处理：

- table blocks 按 dataset / metric / number / table anchor 重新排序。

当前状态：

- 代码已修。
- 依赖重新 ingest 后 wiki 中是否真的有 Table 5。

#### 6.5 多实体问题只回答第一个主题

问题：

- 问 Generator / Verifier / Pruner 时，模型容易只围绕 SAC-KG 总体回答。

处理：

- 增加 query facets。
- 对每个 facet 尝试构造上下文。
- citations 选择时纳入 facet coverage。

当前状态：

- 已修。

#### 6.6 数字幻觉

问题：

- 模型可能回答 F1 / AUC 数值，但 citations 中没有这些数字。

处理：

- `_answer_numbers()` 提取答案数字。
- `_unsupported_answer_numbers()` 检查数字是否出现在 citation context。
- 若 unsupported，触发二次回答修复。

当前状态：

- 已修。

#### 6.7 citation 编号错位

问题：

- 答案中出现 `[4]`，但 API 返回只有一个 citation。

处理：

- 返回前重排 citation 编号。
- 没有 citation 时去掉 marker。

当前状态：

- 已修。

#### 6.8 中文“引用”误触发 raw evidence 模式

问题：

- “请引用 Table 2”被误判为要原文证据，导致 raw chunk citation。

处理：

- `_needs_source_evidence()` 不再把中文“引用”作为 raw evidence marker。

当前状态：

- 已修。

### 7. SAC-KG 改造问题

#### 7.1 初版只是 SAC-KG-inspired，不是真正对齐论文

问题：

- 用户指出论文 Generator 并不简单，包含 domain corpora retriever 和 open KG retriever。

处理：

- 重新评估差距。
- 计划修正为：
  - head entity driven generation
  - sentence/snippet-level retriever
  - local open KG examples
  - strict triples
  - verifier error types
  - tail entity pruner

当前状态：

- 第一版轻量实现已完成。
- 不追求论文百万级 KG 完整复现。

#### 7.2 HEAD 数量过少，学术概念不扩展

问题：

- `HEAD_MAX_COUNT=6` 不够。
- `concept/component/module` 不在 grow 类型中。

处理：

- `HEAD_MAX_COUNT=10`
- 增加 grow 类型：
  - concept
  - component
  - module

当前状态：

- 已修。

### 8. 数据清理与历史重建问题

#### 8.1 同一 PDF 多次 ingest 产生多个 source page

问题：

出现多个页面：

```text
sources/knowledge-graph
sources/sac-kg
sources/sac-kg-exploiting-large-language-models...
```

影响：

- 检索被旧页面污染。
- citations 重复。
- 旧页面内容质量不一致。

处理：

- 建议清理旧 ingest 结果后重新上传。

当前状态：

- 已给出清理方案。

#### 8.2 旧 wiki 不会因代码更新自动重写

问题：

- query 代码更新后，旧 source page 仍是旧解析结果。
- MinerU markdown 表格兜底也不会自动作用于旧结果。

处理：

- 必须重新 ingest PDF。

当前状态：

- 已明确。

#### 8.3 清理命令需要避免依赖系统 sqlite3

问题：

- 用户服务器没有 `sqlite3`。

处理：

- 用 Python 标准库清理：
  - question_answers
  - review_items
  - claims
  - entities
  - wiki_pages
  - document_chunks
  - pipeline_runs
  - documents

当前状态：

- 已给出。

### 9. Git 与协作流程问题

#### 9.1 没有 baseline 导致审查困难

问题：

- 多轮功能迭代后，不清楚每次改了哪些文件。

处理：

- 初始化 Git。
- 新增 `.gitignore` / `.gitattributes`。
- 建议后续走：
  - branch
  - commit
  - PR
  - review

当前状态：

- 本地已完成。
- 远端私有仓库创建需要用户手动完成或补权限。

#### 9.2 临时脚本与运行数据不应进入 Git

问题：

- `data/`、`tmp/`、日志、PDF 原文可能污染 Git。

处理：

- `.gitignore` 排除这些目录。

当前状态：

- 已完成。

### 10. 当前最新状态总览

当前代码侧已经完成：

- 无 Docker 服务器部署。
- sh 兼容启动脚本。
- Git baseline。
- wiki-first query。
- SAC-KG inspired head-driven 摄入。
- MinerU 可选 PDF 解析。
- MinerU v2 JSON 兼容。
- MinerU markdown table 兜底。
- query 多 facet 覆盖。
- 数字反幻觉修复。
- citation 编号重排。
- Table/Figure 渲染上限提升。

当前仍要靠服务器实机确认：

- MinerU 原始输出是否包含 Table 2 / Table 5。
- 重新 ingest 后 `sources/*.md` 是否包含这些表。
- 四问脚本返回的答案和 citations 是否稳定对齐。
- 普通问题是否优先 wiki citation。
- “请引用 Table 2”是否不再退回 raw chunk。

### 11. 下一轮建议从哪里继续

下一轮最建议从下面顺序继续：

1. 服务器 `git pull origin main`。
2. `pytest -q` 确认服务器代码。
3. 清理旧 ingest 结果。
4. 重新上传 PDF。
5. 先 grep `Table 2` / `Table 5` / `OIE2016` / `NYT`。
6. 如果 grep 不到，直接检查 `data/cache/mineru` 原始输出。
7. 如果 grep 到，再运行 `tmp/sac_kg_query_batch.sh`。
8. 根据四问结果继续修 citation / search / wiki 渲染。
