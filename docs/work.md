# 当前任务工作记录

这份文档用于集中记录本轮服务器版 `LLM Wiki` 改造过程中已经遇到的问题、已完成的修复方式，以及当前项目推进到的阶段。

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
