# PDF 问答核验问题修正计划

## Summary

本计划记录 2026-06-11 对 PDF 问答验收结果的核验结论与后续修正方向。

当前判断是：PDF 解析主链已经生成了可用的 `source_summary` wiki 页面，并且能够抽取摘要、关键事实、表格、图注、相关实体页链接。这说明“页面渲染 + 多模态理解 + wiki 写入”链路已经跑通。

但当前问答链路仍存在明显问题，主要集中在 query/retrieval/rendering，而不是 PDF 主解析完全失败：

- `Figure 1 展示了什么流程？` 回答错误。PDF 和 wiki 中都有 Figure 1 图注，但检索上下文没有命中对应内容。
- `ablation studies` 回答过度保守。PDF 第 4.4 节、Table 2、Appendix D 都有明确结论。
- `使用了哪些数据集或 benchmark` 回答部分幻觉。应包含 `OIE2016 / WEB / NYT / PENN`，以及 rice domain corpora；不应把 `rice pesticide dataset` 当成主要数据集。
- citations 重复严重。同一个 `source_summary` 页面被返回多次，excerpt 也重复截到页面开头。
- `verified_claim_count=0` 导致 entity/wiki 结构化知识不足，回答更多依赖 source summary 和模型推断。
- `page_label` 显示为 5/14/16 等，但 citation excerpt 仍是 source page 开头，页码和证据片段不一致。

目标是让 PDF 解析后的 wiki 内容能稳定支撑 `wiki-first` 查询，尤其是 Figure/Table/指标/消融实验类问题。

## Key Changes

- 修正 wiki context window：
  - `_window_text()` 需要优先命中 `Figure 1`、`Table 2`、`OIE2016`、`NYT`、`ablation` 等精确短语。
  - 对 `Figure X`、`Table X` 做专门正则定位，优先截取对应图注/表格块，而不是总截 source page 开头。
  - 页面级 citation 的 excerpt 应来自命中的局部块，而不是整页 Markdown 的开头。

- 修正 citation 去重与排序：
  - 同一 `page_slug + page_label + excerpt` 只保留一次。
  - 同一 source page 多次命中时，优先保留分数最高且 excerpt 不同的 2 条以内。
  - citations 中不要混用模型生成的 `[0][1]` 和 wiki link；最终回答应统一使用返回的 citation indexes 或统一 wiki citation。

- 强化图表/表格检索：
  - `document_intelligence.figures/tables` 应作为独立检索块参与 query，而不是只嵌在 source summary Markdown 后半段。
  - 对 `Figure 1 展示什么` 优先检索 `Figure Notes`。
  - 对 `OIE2016/NYT 指标` 优先检索 `Tables` 和对应 page chunks。

- 修正回答约束：
  - query prompt 中明确：如果 context 中存在相关 Figure/Table，不得回答“材料中未包含”。
  - 当问题含 `Figure`、`Table`、`指标`、`ablation` 时，要求回答必须引用具体表号/图号和数值。
  - 对数据集类问题，要求区分 `benchmark datasets`、`case study categories`、`domain corpora`，避免混成“数据集”。

- 改善摄入后 wiki 质量：
  - source page 的 `Keywords` 正文复用 frontmatter `key_terms`，避免显示 `None`。
  - 如果 `page_summary` 为空，从 page text/table/figure note 自动生成一句 fallback summary。
  - 对 `source_fact` fallback triples 可以先标记为本地 verified，避免 `verified_claim_count=0` 让 wiki-first scoring 变弱。

## Test Plan

- 回归测试 `Figure 1 展示了什么流程？`
  - 应回答：输入由 text、instruction、examples 三段组成；text 来自 domain corpora retriever；examples 来自 open-source encyclopedia KG；输出包含 generated triples 和 grow/prune 指示。
  - citation excerpt 必须包含 `Figure 1` 或其图注。

- 回归测试 `论文中的 ablation studies 得出了什么结论？`
  - 应提到移除任一组件会降低性能。
  - 应提到 pruner 和 open KG retriever 影响更明显。
  - 应能引用 Table 2 或 Appendix D 的结论。

- 回归测试 `SAC-KG 使用了哪些数据集或 benchmark？`
  - 应包含 `OIE2016 / WEB / NYT / PENN`。
  - 应包含 rice domain case study categories，例如 `Rice variety`、`Rice expert`。
  - 不应无证据输出 `rice pesticide dataset` 作为主要数据集。

- 回归测试 `SAC-KG 在 OIE2016 或 NYT 数据集上的指标是什么？`
  - OIE2016: `F1 74.7`、`AUC 73.2`。
  - NYT: `F1 88.8`、`AUC 87.3`。
  - citation excerpt 必须命中 Table 5 附近。

- citation 测试：
  - 同一 page citation 不应重复 5 次。
  - `page_label` 与 excerpt 内容应一致。
  - query 写回 `wiki/queries/` 仍正常。

## Assumptions

- 不改变外部 API。
- 不重新设计数据库表。
- 继续保持 `wiki-first`，raw chunk 只作为 fallback。
- 当前优先修 query/retrieval/rendering，不重新大改 PDF 多模态解析主链。
- 修复后需要对已 ingest 的 PDF 执行一次重新 ingest，才能让 page summary、keywords、source facts 等 wiki 内容更新。
