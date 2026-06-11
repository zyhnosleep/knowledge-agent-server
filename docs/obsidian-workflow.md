# Obsidian 协作说明

这份说明面向当前的服务器版 `LLM Wiki Server`。

## 角色分工

- 服务器负责 `ingest / query / verify / writeback`
- `raw/` 保存原始资料
- `wiki/` 保存可读 Markdown Wiki
- Obsidian 作为本地阅读和策展工具，不替代后端处理链路

## 如何配合使用

- 服务器摄入文档后，会把结果写入 `data/wiki/<project>/`
- 这个目录可以同步到本地，直接作为 Obsidian Vault 打开
- 你可以在 Obsidian 中查看 `sources/`、`entities/`、`queries/`、`index.md`、`log.md`

常见做法：

- 在服务器上上传 `.md` / `.txt` / `.pdf`
- 等 worker 完成摄入
- 把 `data/wiki/<project>/` 同步到本地
- 用 Obsidian 打开同步目录，做人类阅读和补充整理

## Markdown 与内部处理

上传 Markdown 很适合这套流程，但要区分两层：

- 对你来说，Markdown 是知识工件
- 对服务器来说，Markdown 仍会进入解析、snippet/chunk、抽取、校验、写回流程

这意味着：

- 可以直接上传 `.md`
- 不需要像传统 RAG 那样手工做分块
- 但服务器内部仍会保留 snippet/chunk 处理，用于证据定位、去重、失败回退和精确 citations

## 查询路径

当前查询已经收敛到 `wiki-first`：

- 优先检索 `index.md` 和 `wiki/*.md`
- 优先引用 source/entity/query wiki 页面
- 只有 wiki 缺少细节，或用户明确要求原文/证据/引用时，才回退到 raw/snippet

## 目录建议

推荐把下面这个目录作为 Obsidian Vault 的同步源：

```text
data/wiki/<project>/
  index.md
  log.md
  sources/
  entities/
  queries/
```

如果你只想查看知识库结果，打开 `wiki/` 就够了。
如果你还想追踪原始输入，再同时保留 `data/raw/<project>/`。
