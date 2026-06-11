# 当前进展与后续规划

## 已完成

- 完成无 Docker 服务器版基础落地，保留 `FastAPI + Redis worker + SQLite + 本地文件 + Ollama` 的运行方式。
- 完成 `wiki-first` 查询方向，`/api/query` 优先检索 wiki 页面，再按需回退到 raw chunk。
- 完成 SAC-KG 启发式改造：
  - head-driven triple 生成
  - 本地 verifier
  - tail-based pruner
  - wiki 页面 metadata 强化
- 完成 Obsidian 兼容说明：
  - `wiki/` 可作为本地 Vault 同步后直接查看
  - 新增 [obsidian-workflow.md](./obsidian-workflow.md)
- 完成查询引用收敛：
  - 去除无关 `sample` / 空 entity 页污染
  - 优先返回 wiki page citation
  - 对历史上传前缀进行标题归一化
- 完成多类输入解析：
  - `md`
  - `txt`
  - `pdf`
  - `docx`
  - `html`
- 完成一批回归测试补强，当前本地测试通过。

## 当前能力边界

- `md` / `txt` 是当前最稳定、最推荐的输入格式。
- `html` 与 `docx` 支持基础文本提取，但还需要更多真实文档验收。
- `pdf` 当前只是**接通了解析入口**，但**还没有完成服务器实机测试验收**。
- 当前 `pdf` 解析本身不是调用 `qwen3.6` 或 Ollama 完成的，而是：
  - 先用 `pypdf.PdfReader.extract_text()` 提取文本
  - 再把提取出的文本交给 Ollama 做知识抽取、triple 生成、embedding 和问答
- 这意味着：
  - 文本型 PDF 可能可用
  - 扫描版 PDF、图片型 PDF、复杂表格/双栏 PDF 当前风险较高
- 因为还没有完成 PDF 实机验收，现阶段不应把 `pdf` 视为与 `md/txt` 同等级稳定输入。

## 当前状态

- `/api/query` 已确认会调用 Ollama 生成问答内容，并在必要时调用 embedding。
- 服务器端 query 现在可以把 raw chunk citation 提升成对应 wiki page citation。
- 新上传文档的标题已开始自动去掉随机前缀，旧数据也在查询侧做了兼容净化。
- 当前最有价值的知识资产是 `data/wiki/<project>/` 下生成的 Markdown 页面。

## 后续规划

### 1. 历史数据重建

- 选定 1 个真实项目做试点
- 对旧文档重新 ingest，生成新版 `sources/`、`entities/`、`queries/`
- 清理或隔离旧的带前缀 wiki 页面
- 验证新旧知识工件是否一致

### 2. 服务器实机验收

- 设计固定问题集：
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
  - `pdf` / `docx` / `html` 的实际可用性边界

### 3. Obsidian 联调

- 将 `data/wiki/<project>/` 同步到本地 Obsidian Vault
- 检查：
  - `index.md`
  - `log.md`
  - `sources/`
  - `entities/`
  - `queries/`
- 验证 wikilink、frontmatter、页面跳转是否可用

### 4. 运维整理

- 整理服务器更新后命令版 checklist
- 补充“如何判断 query 是否真实命中 Ollama”的操作说明
- 形成可重复执行的更新与验收流程

### 5. 多格式输入验收

- 单独准备 1 份文本型 PDF 做 ingest 测试
- 单独准备 1 份扫描型或图片型 PDF 做失败边界测试
- 检查 `pypdf` 提取结果是否足够支撑后续 wiki 生成
- 如果 PDF 文本提取质量差，再决定是否补 OCR 路线

## 备注

- 目前优先级高于继续加新功能的是“让现有流程在真实服务器数据上稳定、可解释、可回看”。
- 后续如果历史数据重建稳定，再考虑更进一步的 KG 化扩展或外部知识源接入。
