"""语义分块(Semantic Chunking)模块。

把 CanonicalDocument(规范化文档)切成 parent/child 两级 chunk,作为
RAG 检索层的检索单元:

- **Parent chunk**:较大的上下文窗口,代表一个语义完整的片段,用于给
  Child 提供上下文;
- **Child chunk**:较小的检索单元,是实际参与向量检索、上下文化与引用的
  最小单元;
- 每个 Child 记录 source_block_ids、source_spans、前后邻接关系、语义
  边界分数等元数据,保证"检索结果可回溯到原文"。

实现要点:
- 对叙述类正文先按句子切成"raw unit",用 embedding 计算相邻句的余弦
  相似度,选取低相似度位置作为语义边界(parent/child 分块);
- 对表格/图/公式等结构化对象,复用 StructuredEvidenceBuilder 生成父子
  chunk;
- 严格的 token 预算控制(min/target/max):超长单元做无损窗口拆分,
  尾部欠长组做重平衡,超过硬上限的单元强制切分;
- 最终执行"源保真审计"(source fidelity audit),逐 chunk 校验能否从
  原文重建,失败则抛出 :class:`SourceFidelityError`,保证分块结果不丢失
  任何源信息。

对外主入口是 :class:`SemanticChunker`,其 :meth:`build` 接收
CanonicalDocument,返回一组 :class:`ChunkDraft`。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.services.canonical_models import CanonicalBlock, CanonicalDocument, SourceSpan
from app.services.canonical_provenance import block_is_generated
from app.services.structured_evidence import (
    StructuredEvidenceBuilder,
    StructuredEvidenceChunk,
)


class ChunkDraft(BaseModel):
    """分块结果的内存表示(尚未做持久化 id 映射)。

    这个 Pydantic 模型承载单个 chunk 的全部信息:
    - 文本与用于向量化的文本(text / embedding_text,通常两者一致);
    - 角色与结构:parent/child、父 chunk、前后邻接 child;
    - 来源追溯:source_block_ids、source_spans、section_path;
    - 分块工具信息:splitter_name/version、splitting_model;
    - 语义边界信息与自定义 metadata。
    禁止额外字段且禁止无穷大/NaN,保证数据可被安全持久化。
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # chunk 的唯一标识(local id,持久化后映射到数据库主键)。
    local_id: str
    # 来源文档的解析版本。
    parse_version: str
    # 角色:parent(父 chunk)或 child(子 chunk)。
    chunk_role: Literal["parent", "child"]
    # 块类型:narrative/table/figure/formula/caption/appendix。
    block_type: Literal["narrative", "table", "figure", "formula", "caption", "appendix"]
    # chunk 的正文文本。
    text: str
    # 用于向量化的文本(默认与 text 一致;上下文化后会变成
    # "{contextual_prefix}\\n\\n{text}")。
    embedding_text: str
    # 文本的 token 数。
    token_count: int = Field(ge=0)
    # 所属父 chunk 的 local_id(None 表示自身就是 parent)。
    parent_local_id: str | None = None
    # 同一结构下前/后一个 child 的 local_id(用于邻接检索)。
    previous_child_local_id: str | None = None
    next_child_local_id: str | None = None
    # 来源 block id 列表(按出现顺序去重)。
    source_block_ids: list[str] = Field(default_factory=list)
    # 来源跨度(SourceSpan),用于回溯到原文页码与字符区间。
    source_spans: list[SourceSpan] = Field(default_factory=list)
    # 章节路径(如 ["Introduction", "Methods"])。
    section_path: list[str] = Field(default_factory=list)
    # 在整体输出中的序号(决定排序)。
    ordinal: int = Field(ge=0)
    # 分块器名称/版本/所用模型(写入元数据用于审计)。
    splitter_name: str
    splitter_version: str
    splitting_model: str
    # 语义边界分数(None 表示非语义边界,如结构化边界)。
    semantic_boundary_score: float | None = None
    # 各类附加元数据(溢出标记、token 窗口标记、来源映射等)。
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceFidelityError(RuntimeError):
    """由最终"源保真审计"抛出的有界、结构化失败异常。

    当一个 chunk 无法从原始文档重建(文本被改写、token 计数不一致、
    超出硬上限、来源块缺失等)时抛出。异常消息是结构化的 JSON,
    便于记录与排查:包含文档 id、页码、块类型、源 id、片段摘录与原因码。
    """

    # 摘录文本的最大长度,防止异常消息过大。
    excerpt_limit = 200

    def __init__(
        self,
        *,
        document_id: str,
        page: int | None,
        block_type: str,
        source_id: str,
        excerpt: str,
        reason: str,
    ) -> None:
        self.document_id = document_id
        self.page = page
        self.block_type = block_type
        self.source_id = source_id
        # 摘录截断到 excerpt_limit 个字符。
        self.excerpt = str(excerpt)[: self.excerpt_limit]
        self.reason = reason
        # 把诊断字段序列化为排序后的紧凑 JSON 作为异常消息。
        super().__init__(
            json.dumps(
                {
                    "document_id": self.document_id,
                    "page": self.page,
                    "block_type": self.block_type,
                    "source_id": self.source_id,
                    "excerpt": self.excerpt,
                    "reason": self.reason,
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


class SemanticChunker:
    """语义分块器:把 CanonicalDocument 切成 parent/child 两级 chunk。

    职责:
    - 把文档块序列化为可检索的 parent/child chunk(叙述类做语义切分,
      结构化对象复用 StructuredEvidenceBuilder);
    - 严格控制 token 预算(min/target/max),处理超长单元与欠长尾部;
    - 在返回前执行源保真审计,保证 chunk 可无损回溯到原文。

    构造参数中所有 token 限制与百分比都可传入覆盖,缺省从配置读取。
    """

    def __init__(
        self,
        embedder: Any,
        token_counter: Any | None = None,
        *,
        parent_min_tokens: int | None = None,
        parent_target_tokens: int | None = None,
        parent_max_tokens: int | None = None,
        child_min_tokens: int | None = None,
        child_target_tokens: int | None = None,
        child_max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        break_percentile: int | None = None,
        splitter_name: str = "section_aware_semantic",
        splitter_version: str = "semantic-v1",
        splitting_model: str | None = None,
    ) -> None:
        """初始化 token 预算、embedder 与结构化证据构建器。

        - ``embedder``:用于给 raw unit 生成 embedding 的对象(可调用或
          有 embed 方法);
        - ``token_counter``:token 计数函数/对象;缺省用 StructuredEvidenceBuilder
          的估计器;
        - min/target/max 三元组分别定义 parent/child 的最小、目标、最大
          token 数(分块时尽量靠近 target,绝不超过 max);
        - ``overlap_tokens``:相邻 child 之间允许重叠的 token 数;
        - ``break_percentile``:从候选边界中取"相似度最低的分位"作为语义
          断点的激进程度(1-99)。
        """
        settings = get_settings()
        self.embedder = embedder
        # 缺省 token 计数器:基于配置的 tokenizer 构建;保留 provider 引用,
        # 以便使用更高效的 offset 范围计数。
        self._default_token_provider: StructuredEvidenceBuilder | None = None
        if token_counter is None:
            self._default_token_provider = StructuredEvidenceBuilder(
                tokenizer_name=settings.semantic_tokenizer_name,
                strict_tokenizer=True,
            )
            self.token_counter = self._default_token_provider.estimate_tokens
        else:
            self.token_counter = token_counter
        # parent 的 (min, target, max) token 限制,未显式传入则用配置值。
        self.parent_token_limits = (
            settings.semantic_parent_min_tokens
            if parent_min_tokens is None
            else parent_min_tokens,
            settings.semantic_parent_target_tokens
            if parent_target_tokens is None
            else parent_target_tokens,
            settings.semantic_parent_max_tokens
            if parent_max_tokens is None
            else parent_max_tokens,
        )
        # child 的 (min, target, max) token 限制。
        self.child_token_limits = (
            settings.semantic_child_min_tokens
            if child_min_tokens is None
            else child_min_tokens,
            settings.semantic_child_target_tokens
            if child_target_tokens is None
            else child_target_tokens,
            settings.semantic_child_max_tokens
            if child_max_tokens is None
            else child_max_tokens,
        )
        # 相邻 child 间的重叠 token 预算。
        self.overlap_tokens = (
            settings.semantic_child_overlap_tokens
            if overlap_tokens is None
            else overlap_tokens
        )
        # 语义断点激进程度(百分位)。
        self.break_percentile = (
            settings.semantic_break_percentile
            if break_percentile is None
            else break_percentile
        )
        self.splitter_name = splitter_name
        self.splitter_version = splitter_version
        # 从 embedder 对象上猜测模型名(model_name / model / name 属性)。
        embedder_model = next(
            (
                value
                for value in (
                    getattr(embedder, "model_name", None),
                    getattr(embedder, "model", None),
                    getattr(embedder, "name", None),
                )
                if isinstance(value, str) and value.strip()
            ),
            None,
        )
        # 记录实际用于分块(语义打分)的模型,便于审计。
        self.splitting_model = (
            splitting_model
            or embedder_model
            or settings.semantic_splitting_model
        )
        # 校验 token 限制配置的合法性。
        self._validate_configuration()
        # 结构化证据构建器:用于表格/图/公式的父子 chunk 生成。
        self._structured_builder = StructuredEvidenceBuilder(token_counter=self._count_tokens)

    def build(self, document: CanonicalDocument) -> list[ChunkDraft]:
        """把一个规范化文档分块为 parent/child 两级 chunk 列表。

        流程:
        1. 把文档块归一化为两种"条目":叙述段(_NarrativeSegment)与
           结构化条目(_StructuredEntry),同时产出 raw unit 列表;
        2. 为所有 raw unit 批量生成 embedding(用于语义边界判定);
        3. 叙述段走语义分块,结构化条目走 StructuredEvidenceBuilder;
        4. 最终执行源保真审计,失败则抛 SourceFidelityError。
        """
        entries, raw_units = self._document_entries(document)
        # 有叙述正文时,先为句子级 raw unit 生成 embedding。
        if raw_units:
            self._attach_embeddings(raw_units)

        drafts: list[ChunkDraft] = []
        for entry in entries:
            if isinstance(entry, _NarrativeSegment):
                # 叙述段:语义分组生成 parent + child。
                self._append_narrative(entry, document.parse_version, drafts)
            else:
                # 结构化对象(表/图/公式):复用证据构建器生成 parent + child。
                self._append_structured(entry, document, drafts)
        # 最终审计:确保每个 child 都能从原文无损重建。
        self._audit_source_fidelity(document, drafts)
        return drafts

    def _audit_source_fidelity(
        self,
        document: CanonicalDocument,
        drafts: list[ChunkDraft],
    ) -> None:
        """执行最终"源保真审计":每个 child 必须能从原文无损重建。

        审计内容分三层:
        1. 文本层:embedding_text 必须与 text 一致(纯文本路径),记录的
           token 数与实际计数一致,且不超过 child 硬上限;
        2. 叙述层:每个展开后的 raw unit 都必须被某个同类 child 的
           source_block_ids + source_unit_ids 覆盖,且文本确实出现在该
           child 中;
        3. 结构化层:委托 _audit_structured_fidelity 校验表/图/公式。
        任一层不满足即抛 SourceFidelityError。
        """
        children = [draft for draft in drafts if draft.chunk_role == "child"]
        child_max = self.child_token_limits[2]
        # 1) 文本层的 child 级校验。
        for child in children:
            source_id = self._draft_source_id(child)
            page = self._first_page(child.source_spans)
            # 纯文本分块路径下 embedding_text 必须等于 text,否则后续向量化
            # 会引入与源不一致的内容。
            if child.embedding_text != child.text:
                raise SourceFidelityError(
                    document_id=document.document_id,
                    page=page,
                    block_type=child.block_type,
                    source_id=source_id,
                    excerpt=child.embedding_text,
                    reason="child_embedding_text_differs_from_source_text",
                )
            # 记录的 token 数必须与实测一致。
            actual_token_count = self._count_tokens(child.text)
            if child.token_count != actual_token_count:
                raise SourceFidelityError(
                    document_id=document.document_id,
                    page=page,
                    block_type=child.block_type,
                    source_id=source_id,
                    excerpt=child.text,
                    reason="child_token_count_mismatch",
                )
            # 不得超过 child 的硬上限。
            if actual_token_count > child_max:
                raise SourceFidelityError(
                    document_id=document.document_id,
                    page=page,
                    block_type=child.block_type,
                    source_id=source_id,
                    excerpt=child.text,
                    reason="child_token_limit_exceeded",
                )

        # 2) 叙述层:每个展开后的 raw unit 都必须被某个 child 覆盖。
        entries, _raw_units = self._document_entries(document)
        for entry in entries:
            if not isinstance(entry, _NarrativeSegment):
                continue
            for unit in self._expanded_narrative_units(entry):
                # 检查是否存在一个同类 child 同时满足:来自同一 block、声明
                # 覆盖该 unit、且文本确实包含 unit 文本。
                represented = any(
                    unit.block_id in child.source_block_ids
                    and unit.unit_id in child.metadata.get("source_unit_ids", [])
                    and unit.text in child.text
                    for child in children
                    if child.block_type == entry.block_type
                )
                if represented:
                    continue
                raise SourceFidelityError(
                    document_id=document.document_id,
                    page=self._first_page(unit.source_spans),
                    block_type=entry.block_type,
                    source_id=unit.block_id,
                    excerpt=unit.text,
                    reason="narrative_source_unit_not_reconstructable",
                )

        # 3) 结构化层(表格/图/公式)。
        self._audit_structured_fidelity(document, drafts)

    def _audit_structured_fidelity(
        self,
        document: CanonicalDocument,
        drafts: list[ChunkDraft],
    ) -> None:
        """审计结构化对象(表格/图/公式)是否被 chunk 图完整覆盖。

        对每个可检索的结构化 block:
        1. 找到与其同一结构(按 *_id 元数据)关联的 draft;
        2. 确认存在父 chunk(来源包含该 block)且存在 child;
        3. 确认相关 draft 的 source_block_ids 都包含该 block(来源完整);
        4. 按类型进一步审计(表/图/公式的细节一致性)。
        """
        tables = {table.table_id: table for table in document.tables}
        figures = {figure.figure_id: figure for figure in document.figures}
        formulas = {formula.formula_id: formula for formula in document.formulas}
        for block, section_path in self._blocks_with_effective_section_paths(
            document.blocks
        ):
            # 只审计可检索的结构化块;参考文献章节跳过。
            if (
                block.block_type not in {"table", "figure", "formula"}
                or not block.retrievable
                or self._is_reference_section(section_path)
            ):
                continue
            # 结构化对象的规范 id(如 table_id)。
            source_id = str(
                {
                    "table": block.table_id,
                    "figure": block.figure_id,
                    "formula": block.formula_id,
                }[block.block_type]
                or ""
            )
            # 通过 metadata 里的 "<type>_id" 找到同一结构的 draft。
            metadata_key = f"{block.block_type}_id"
            identity_related = [
                draft
                for draft in drafts
                if draft.block_type == block.block_type
                and draft.metadata.get(metadata_key) == source_id
            ]
            # 其中来源覆盖该 block 的父 chunk。
            related_parents = [
                draft
                for draft in identity_related
                if draft.chunk_role == "parent"
                and block.block_id in draft.source_block_ids
            ]
            parent_ids = {draft.local_id for draft in related_parents}
            # 父 chunk 本身及其全部 child。
            related = [
                draft
                for draft in identity_related
                if draft.local_id in parent_ids or draft.parent_local_id in parent_ids
            ]
            # 定位结构化源对象(用于取页码)。
            source = {
                "table": tables.get(source_id),
                "figure": figures.get(source_id),
                "formula": formulas.get(source_id),
            }[block.block_type]
            page = self._first_page(
                getattr(source, "source_spans", []) if source is not None else []
            )
            if page is None:
                page = self._first_page(block.source_spans)
            # 源对象缺失,或没有父 chunk/child,说明该结构未进入 chunk 图。
            if source is None or not related_parents or not any(
                draft.chunk_role == "child" for draft in related
            ):
                self._raise_source_fidelity(
                    document,
                    page=page,
                    block_type=block.block_type,
                    source_id=source_id or block.block_id,
                    excerpt=block.text or source_id,
                    reason="structured_source_missing_from_chunk_graph",
                )
            # 任一相关 draft 的来源列表缺少该 block,说明来源追踪不完整。
            if any(block.block_id not in draft.source_block_ids for draft in related):
                self._raise_source_fidelity(
                    document,
                    page=page,
                    block_type=block.block_type,
                    source_id=source_id,
                    excerpt=block.block_id,
                    reason="structured_source_provenance_missing",
                )
            # 按类型执行更细的审计。
            if block.block_type == "table":
                self._audit_table(document, source, related, page=page)
            elif block.block_type == "figure":
                self._audit_figure(document, source, related, page=page)
            else:
                self._audit_formula(document, source, related, page=page)

    def _audit_table(self, document, table, drafts, *, page: int | None) -> None:
        """审计表格的题注、表头、数据行、单元格与脚注能否从 chunk 重建。

        四种来源表示都会接受:文本子串出现,或 metadata 中按结构存储的
        (headers / row_indices+rows / cells)一致。任一缺失即抛错。
        """
        texts = [draft.text for draft in drafts]
        metadata = [draft.metadata for draft in drafts]
        # 生成表头行与完整网格的参考文本,用于子串匹配。
        header_line = StructuredEvidenceBuilder._markdown_grid(table.headers, [])[0]
        grid_text = "\n".join(
            StructuredEvidenceBuilder._markdown_grid(table.headers, table.rows)
        )
        # 1) 题注必须能被表示。
        if table.caption and not self._string_is_represented(
            table.caption, texts, metadata, "caption"
        ):
            self._raise_source_fidelity(
                document,
                page=page,
                block_type="table",
                source_id=table.table_id,
                excerpt=table.caption,
                reason="table_caption_not_reconstructable",
            )
        # 2) 表头必须被完整表示(metadata 中 headers 一致,或表头行文本出现)。
        if table.headers and not any(
            item.get("headers") == table.headers for item in metadata
        ) and not any(header_line in text for text in texts):
            self._raise_source_fidelity(
                document,
                page=page,
                block_type="table",
                source_id=table.table_id,
                excerpt=" | ".join(table.headers),
                reason="table_headers_not_reconstructable",
            )

        # 3) 数据行:从各 draft 的 metadata 收集 (行下标 -> 行内容)。
        rows_by_index: dict[int, list[str]] = {}
        for item in metadata:
            indices = item.get("row_indices")
            rows = item.get("rows")
            # 只接受合法的下标/行内容成对数据。
            if not isinstance(indices, list) or not isinstance(rows, list):
                continue
            if len(indices) != len(rows):
                continue
            for index, row in zip(indices, rows, strict=True):
                if isinstance(index, int) and isinstance(row, list):
                    rows_by_index[index] = row
        # 全部行一致,或完整网格文本出现在某个 draft 文本中,即算可重建。
        rows_reconstructed = all(
            rows_by_index.get(index) == row for index, row in enumerate(table.rows)
        ) or any(grid_text in text for text in texts)
        if not rows_reconstructed:
            missing = next(
                row
                for index, row in enumerate(table.rows)
                if rows_by_index.get(index) != row
            )
            self._raise_source_fidelity(
                document,
                page=page,
                block_type="table",
                source_id=table.table_id,
                excerpt=" | ".join(missing),
                reason="table_rows_not_reconstructable",
            )

        # 4) 单元格:所有期望单元格都必须出现在 metadata 的 cells 中。
        expected_cells = {
            self._stable_json(cell.model_dump(mode="json")) for cell in table.cells
        }
        represented_cells = {
            self._stable_json(cell)
            for item in metadata
            for cell in (item.get("cells") or [])
            if isinstance(cell, dict)
        }
        if not expected_cells.issubset(represented_cells):
            missing_cell = next(
                cell
                for cell in table.cells
                if self._stable_json(cell.model_dump(mode="json"))
                not in represented_cells
            )
            self._raise_source_fidelity(
                document,
                page=page,
                block_type="table",
                source_id=table.table_id,
                excerpt=missing_cell.text,
                reason="table_cells_not_reconstructable",
            )
        # 5) 脚注:每条都必须能被表示。
        for footnote in table.footnotes:
            if self._string_is_represented(footnote, texts, metadata, "footnotes"):
                continue
            self._raise_source_fidelity(
                document,
                page=page,
                block_type="table",
                source_id=table.table_id,
                excerpt=footnote,
                reason="table_footnote_not_reconstructable",
            )

    def _audit_formula(self, document, formula, drafts, *, page: int | None) -> None:
        """审计公式的 LaTeX、题注与描述能否从 chunk 重建。

        接受"整段子串出现"或"按片段(metadata 中的 source_fragment 系列
        字段)拼接重建"两种表示;片段重建由 _fragments_reconstruct 校验。
        """
        texts = [draft.text for draft in drafts]
        # 1) LaTeX 主体:整串或片段重建均可。
        if not any(
            formula.latex in text for text in texts
        ) and not self._fragments_reconstruct(
            formula.latex,
            drafts,
            fragment_kind="latex",
        ):
            self._raise_source_fidelity(
                document,
                page=page,
                block_type="formula",
                source_id=formula.formula_id,
                excerpt=formula.latex,
                reason="formula_latex_not_reconstructable",
            )
        # 2) 题注与描述:空值放行;否则需整串或片段可重建。
        for name, value in (
            ("caption", formula.caption),
            ("description", formula.description),
        ):
            if (
                not value
                or any(value in text for text in texts)
                or self._fragments_reconstruct(
                    value,
                    drafts,
                    fragment_kind=name,
                )
            ):
                continue
            self._raise_source_fidelity(
                document,
                page=page,
                block_type="formula",
                source_id=formula.formula_id,
                excerpt=value,
                reason=f"formula_{name}_not_reconstructable",
            )

    def _audit_figure(self, document, figure, drafts, *, page: int | None) -> None:
        """审计图表的题注、描述、资源路径与资源 id 能否从 chunk 重建。"""
        texts = [draft.text for draft in drafts]
        # 1) 题注与描述:空值放行;题注需整串出现,描述允许片段重建。
        for name, value in (
            ("caption", figure.caption),
            ("description", figure.description),
        ):
            represented = not value or any(value in text for text in texts)
            if name == "description" and value:
                represented = represented or self._fragments_reconstruct(value, drafts)
            if represented:
                continue
            self._raise_source_fidelity(
                document,
                page=page,
                block_type="figure",
                source_id=figure.figure_id,
                excerpt=value,
                reason=f"figure_{name}_not_reconstructable",
            )
        # 2) 资源路径:文本中出现,或某 draft 的 metadata 记录了该路径。
        if figure.asset_path and not (
            any(figure.asset_path in text for text in texts)
            or any(
                draft.metadata.get("asset_path") == figure.asset_path for draft in drafts
            )
        ):
            self._raise_source_fidelity(
                document,
                page=page,
                block_type="figure",
                source_id=figure.figure_id,
                excerpt=figure.asset_path,
                reason="figure_asset_path_not_reconstructable",
            )
        # 3) 资源 id:必须在某个 draft 的 source_spans 中标注了该 asset_id。
        asset_id = self._figure_asset_id(document, figure)
        if asset_id and not any(
            span.metadata.get("asset_id") == asset_id
            for draft in drafts
            for span in draft.source_spans
        ):
            self._raise_source_fidelity(
                document,
                page=page,
                block_type="figure",
                source_id=figure.figure_id,
                excerpt=asset_id,
                reason="figure_asset_id_not_reconstructable",
            )

    @staticmethod
    def _fragments_reconstruct(
        source: str,
        drafts: Iterable[ChunkDraft],
        *,
        fragment_kind: str | None = None,
    ) -> bool:
        """判断若干"来源片段"能否按位置无缝拼接出 source 全文。

        各 draft 的 metadata 中可带 source_fragment_start / end / fragment
        记录一段子串在原文本中的位置。本函数校验:片段不重叠不冲突、位置
        连续覆盖(第 i+1 段的起点 == 第 i 段的终点)、且拼接结果恰好等于
        source。可选按 fragment_kind 过滤(如 latex/caption/description)。
        """
        # 收集所有合法片段:{起点: (终点, 文本)}。
        fragments: dict[int, tuple[int, str]] = {}
        for draft in drafts:
            if (
                fragment_kind is not None
                and draft.metadata.get("source_fragment_kind") != fragment_kind
            ):
                continue
            start = draft.metadata.get("source_fragment_start")
            end = draft.metadata.get("source_fragment_end")
            fragment = draft.metadata.get("source_fragment")
            # 只接受字段齐全且位置/长度自洽的片段。
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not isinstance(fragment, str)
                or start < 0
                or end < start
                or end - start != len(fragment)
            ):
                continue
            # 同一起点出现冲突的片段:不可重建。
            existing = fragments.get(start)
            if existing is not None and existing != (end, fragment):
                return False
            fragments[start] = (end, fragment)
        # 按起点顺序拼接;要求起点连续、终点恰好覆盖 source 全长。
        cursor = 0
        reconstructed: list[str] = []
        for start in sorted(fragments):
            end, fragment = fragments[start]
            if start != cursor:
                return False
            reconstructed.append(fragment)
            cursor = end
        return bool(fragments) and cursor == len(source) and "".join(reconstructed) == source

    @staticmethod
    def _string_is_represented(
        value: str,
        texts: Iterable[str],
        metadata: Iterable[dict[str, Any]],
        metadata_key: str,
    ) -> bool:
        """判断一个字符串是否已在 chunk 集合中被表示。

        满足任一即算表示:作为某 chunk 文本的子串出现;或某 chunk 的
        metadata[metadata_key] 等于该值(或该值是其中的列表元素)。
        """
        if any(value in text for text in texts):
            return True
        for item in metadata:
            candidate = item.get(metadata_key)
            if candidate == value or (
                isinstance(candidate, list) and value in candidate
            ):
                return True
        return False

    @staticmethod
    def _stable_json(value: Any) -> str:
        """把值序列化为"排序键 + 紧凑分隔符"的稳定 JSON,用于集合比较。"""
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _first_page(spans: Iterable[SourceSpan]) -> int | None:
        """返回第一个带页码的 span 的页码(没有则返回 None)。"""
        return next(
            (span.page_index for span in spans if span.page_index is not None),
            None,
        )

    @staticmethod
    def _draft_source_id(draft: ChunkDraft) -> str:
        """取 chunk 的规范来源 id(用于错误消息)。

        优先用 metadata 里的 "<type>_id"(结构化对象 id);其次用第一个
        source_block_id;都没有则退回 local_id。
        """
        metadata_key = f"{draft.block_type}_id"
        structured_id = draft.metadata.get(metadata_key)
        if isinstance(structured_id, str) and structured_id:
            return structured_id
        if draft.source_block_ids:
            return draft.source_block_ids[0]
        return draft.local_id

    @staticmethod
    def _raise_source_fidelity(
        document: CanonicalDocument,
        *,
        page: int | None,
        block_type: str,
        source_id: str,
        excerpt: str,
        reason: str,
    ) -> None:
        """统一抛出 SourceFidelityError 的辅助方法。"""
        raise SourceFidelityError(
            document_id=document.document_id,
            page=page,
            block_type=block_type,
            source_id=source_id,
            excerpt=excerpt,
            reason=reason,
        )

    def _validate_configuration(self) -> None:
        """校验 token 限制与百分比参数是否合法。

        要求 parent/child 各满足 0 < min <= target <= max;重叠 token 非负;
        break_percentile 在 1-99 之间。
        """
        if not (
            0 < self.parent_token_limits[0]
            <= self.parent_token_limits[1]
            <= self.parent_token_limits[2]
        ):
            raise ValueError("parent token limits must satisfy 0 < min <= target <= max")
        if not (
            0 < self.child_token_limits[0]
            <= self.child_token_limits[1]
            <= self.child_token_limits[2]
        ):
            raise ValueError("child token limits must satisfy 0 < min <= target <= max")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if not 1 <= self.break_percentile <= 99:
            raise ValueError("break_percentile must be between 1 and 99")

    def _count_tokens(self, text: str) -> int:
        """统一入口:对文本做 token 计数,并校验结果为非负整数。

        兼容两种计数器接口:可调用对象,或暴露 ``count(text)`` 方法。
        """
        counter = self.token_counter
        if callable(counter):
            count = counter(text)
        elif hasattr(counter, "count") and callable(counter.count):
            count = counter.count(text)
        else:
            raise TypeError("token_counter must be callable or expose count(text)")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("token counter must return a non-negative integer")
        return count

    def _uses_monotonic_prefix_counts(self) -> bool:
        """判断 token 计数是否"前缀单调"。

        单调意味着"某区间的 token 数随区间扩大只增不减",可以启用二分
        查找来快速定位满足/突破预算的边界;UTF-8 字节回退模式计数满足
        该性质。False 时只能线性逐区间计数。
        """
        if getattr(self.token_counter, "monotonic_prefix_counts", False) is True:
            return True
        provider = self._default_token_provider
        return (
            provider is not None
            and provider.token_count_mode == "utf8_bytes_fallback"
        )

    def _offset_range_counter(
        self,
        units: list[_RawUnit],
    ) -> Callable[[int, int], int] | None:
        """为 raw unit 构造一个"按字符区间近似计 token"的高效计数器。

        利用 transformers tokenizer 的 ``offset_mapping``(token -> 字符
        区间)把字符区间映射到 token 区间,从而在 O(log n) 内用二分统计
        任意 unit 子区间的 token 数,大幅加快分块时的预算探测。仅当:
        - 使用默认 token provider 且计数模式为 transformers;
        - tokenizer 可调用且返回单调递增的 offset 映射;
        - 区间映射自洽(已排序、无越界)。
        任一条件不满足时返回 None(调用方回退到逐区间线性计数)。
        """
        provider = self._default_token_provider
        # 只对 transformers 模式且能拿到 tokenizer 的情况生效。
        if provider is None or provider.token_count_mode != "transformers":
            return None
        tokenizer = getattr(provider, "_tokenizer", None)
        if not callable(tokenizer):
            return None
        # 对整个 unit 序列做一次编码,拿到每个 token 的字符偏移。
        text = self._join_units(units)
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoded.get("offset_mapping")
        except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
            return None
        if not isinstance(offsets, list):
            return None
        # 校验并归一化 offsets:丢弃空 token(左 == 右)。
        normalized_offsets: list[tuple[int, int]] = []
        for offset in offsets:
            if (
                not isinstance(offset, (list, tuple))
                or len(offset) != 2
                or not all(isinstance(value, int) for value in offset)
            ):
                return None
            left, right = offset
            if left < 0 or right < left or right > len(text):
                return None
            if right > left:
                normalized_offsets.append((left, right))
        # offsets 必须覆盖全文且单调有序,否则二分假设不成立。
        if (text and not normalized_offsets) or normalized_offsets != sorted(
            normalized_offsets
        ):
            return None

        # 计算每个 unit 的全局字符起始/结束位置(跨 block 时补 2 个换行)。
        unit_starts: list[int] = []
        unit_ends: list[int] = []
        cursor = 0
        for index, unit in enumerate(units):
            # _join_units 在 block 切换处会插入 "\n\n"(2 字符)。
            if index and units[index - 1].block_id != unit.block_id:
                cursor += 2
            unit_starts.append(cursor)
            cursor += len(unit.text)
            unit_ends.append(cursor)
        # token 的起止字符位置数组(供二分查找)。
        token_starts = [left for left, _right in normalized_offsets]
        token_ends = [right for _left, right in normalized_offsets]

        def count_range(start: int, end: int) -> int:
            """统计 units[start:end] 这个连续区间内的 token 数。

            用二分找"字符区间 [char_start, char_end) 覆盖的 token 下标区间"。
            """
            char_start = unit_starts[start]
            char_end = unit_ends[end - 1]
            # 第一个终点 > char_start 的 token(即覆盖区间的第一个 token)。
            first_token = bisect_right(token_ends, char_start)
            # 最后一个起点 < char_end 的 token(即覆盖区间的最后一个 token)。
            last_token = bisect_left(token_starts, char_end)
            return max(0, last_token - first_token)

        return count_range

    def _document_entries(
        self, document: CanonicalDocument
    ) -> tuple[list[_NarrativeSegment | _StructuredEntry], list[_RawUnit]]:
        """把文档块归一化为"分块条目"列表,并产出 raw unit 列表。

        规则:
        - heading 块只用于划分章节,不产生 chunk;
        - 不可检索或位于参考文献章节的块跳过;
        - 表格/图/公式产生一个 _StructuredEntry;
        - 叙述类块按块类型+格式+章节路径聚合成 _NarrativeSegment,其内部
          按句子(_RawUnit)继续细分;代码等预格式化文本整块作为一个 unit。
        返回值:(entries, raw_units),raw_units 是所有叙述单元的展平列表,
        用于批量生成 embedding。
        """
        entries: list[_NarrativeSegment | _StructuredEntry] = []
        raw_units: list[_RawUnit] = []
        current: _NarrativeSegment | None = None

        def flush() -> None:
            """把当前聚合中的叙述段写入 entries,并清空。"""
            nonlocal current
            if current is not None and current.units:
                entries.append(current)
                raw_units.extend(current.units)
            current = None

        for block, section_path in self._blocks_with_effective_section_paths(
            document.blocks
        ):
            # heading 只是章节标记。
            if block.block_type == "heading":
                flush()
                continue
            # 不可检索或参考文献章节的块不参与分块。
            if not block.retrievable or self._is_reference_section(section_path):
                flush()
                continue
            # 结构化对象单独成一个条目。
            if block.block_type in {"table", "figure", "formula"}:
                flush()
                entries.append(_StructuredEntry(block=block, section_path=section_path))
                continue
            # 非可检索叙述(如 caption 位于参考文献区)跳过。
            if not self._is_retrievable_narrative(block, section_path):
                flush()
                continue

            # 判断是否为预格式化文本(代码等)——整块作为一个 unit。
            format_kind = str(block.metadata.get("kind") or "text").strip().casefold()
            preformatted = format_kind in {
                "code",
                "code_block",
                "fenced_code",
                "html_pre",
                "pre",
                "preformatted",
            }
            # 叙述段的聚合键:块类型 + 格式 + (预格式化时)块 id + 章节路径。
            key = (
                block.block_type,
                format_kind,
                block.block_id if preformatted else "",
                tuple(section_path),
            )
            if current is None or current.key != key:
                flush()
                current = _NarrativeSegment(
                    key=key,
                    block_type=block.block_type,
                    section_path=section_path,
                )
            # 预格式化文本整块取;普通文本按句子切分。
            raw_texts = (
                [(0, len(block.text), block.text)]
                if preformatted and block.text
                else self._sentence_ranges(block.text)
            )
            # 把每个句子/整块变成 _RawUnit(含 token 计数与来源信息)。
            for sentence_index, (start, end, sentence) in enumerate(raw_texts):
                current.units.append(
                    _RawUnit(
                        text=sentence,
                        token_count=self._count_tokens(sentence),
                        block_id=block.block_id,
                        block_type=block.block_type,
                        section_path=section_path,
                        source_spans=list(block.source_spans),
                        sentence_id=f"{block.block_id}:{sentence_index}",
                        preformatted=preformatted,
                        block_char_start=start,
                        block_char_end=end,
                    )
                )
        # 处理尾部未 flush 的叙述段。
        flush()
        return entries, raw_units

    @staticmethod
    def _blocks_with_effective_section_paths(
        blocks: Iterable[CanonicalBlock],
    ) -> Iterable[tuple[CanonicalBlock, list[str]]]:
        """按阅读顺序产出 (block, 有效章节路径) 迭代器。

        对 heading 块,用其 section_path(缺失时退回块文本)更新当前章节
        路径;其他块优先用自身的 section_path,缺失时继承最近一次 heading
        的路径。``reference_scope_reset`` 标记会清空路径(表示进入引用域)。
        """
        heading_path: list[str] = []
        for block in sorted(blocks, key=lambda item: item.reading_order):
            # 遇到引用域重置标记则清空章节路径。
            if block.metadata.get("reference_scope_reset") is True:
                heading_path = []
            # heading 块本身更新当前章节路径。
            if block.block_type == "heading":
                heading_path = list(block.section_path) or (
                    [block.text.strip()] if block.text.strip() else []
                )
            # 非 heading 块:自身有路径用自身的,否则继承当前路径。
            yield block, list(block.section_path) or list(heading_path)

    @staticmethod
    def _is_retrievable_narrative(
        block: CanonicalBlock, section_path: list[str]
    ) -> bool:
        """判断叙述类块是否可检索(非参考文献章节的 narrative/caption/appendix)。"""
        if block.block_type not in {"narrative", "caption", "appendix"}:
            return False
        return not SemanticChunker._is_reference_section(section_path)

    @staticmethod
    def _is_reference_section(section_path: list[str]) -> bool:
        """判断章节路径是否属于"参考文献"类章节(中英文名称)。

        匹配前会剥掉路径段开头的编号(如 "1.2"、"[ivxlcdm]+"、中文数字),
        再与参考文献关键词表比对。
        """
        reference_names = {
            "reference",
            "references",
            "bibliography",
            "works cited",
            "literature cited",
            "cited literature",
            "sources",
            "参考文献",
            "引用文献",
            "文献",
            "参考资料",
            "引用资料",
        }
        for part in section_path:
            # 去掉路径段开头的编号/前缀,再统一小写并去尾部冒号。
            normalized = re.sub(
                r"^\s*(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+|[一二三四五六七八九十]+)(?:[.、)]|\s+))\s*",
                "",
                part.strip().casefold(),
            ).rstrip(":：")
            if normalized in reference_names:
                return True
        return False

    @staticmethod
    def _sentence_ranges(text: str) -> list[tuple[int, int, str]]:
        """把一段文本按句边界切成 (起始, 结束, 句子) 列表。

        这是一个启发式句切分器,处理常见的误切场景:
        - 中文句号/感叹号/问号("。！？")始终是边界;
        - 英文 "!" / "?" 后跟引号/右括号且其后是空白/行尾时是边界;
        - "." 需排除:小数("3.14")、非句末缩写(Dr.、Fig. 等)、句末可断
          缩写(etc./e.g.)与首字母缩写(如 "U.S.")——后者要看后续是否跟
          大写字母或中文来判断是否真的是句末。
        """
        if not text:
            return []
        # 常见非句末缩写(其后带点不视为句末)。
        nonterminal_abbreviations = {
            "dr",
            "mr",
            "mrs",
            "ms",
            "prof",
            "sr",
            "jr",
            "st",
            "vs",
            "fig",
            "eq",
            "sec",
            "ref",
        }
        # 可作句末的缩写(etc./e.g./i.e./al 等),需结合后续字符判断。
        terminal_capable_abbreviations = {"al", "etc", "e.g", "i.e"}
        # 句末标点后可跟随的闭合符(引号、括号、中英文右引号等)。
        closers = {'"', "'", "”", "’", "）", ")", "]"}
        ranges: list[tuple[int, int, str]] = []
        start = 0
        index = 0
        while index < len(text):
            character = text[index]
            # 中文句末标点直接视为边界。
            boundary = character in "。！？"
            if character in "!?":
                # 英文感叹/问号:跳过闭合符,其后是空白/行尾才视为边界。
                probe = index + 1
                while probe < len(text) and text[probe] in closers:
                    probe += 1
                boundary = probe == len(text) or text[probe].isspace()
            elif character == ".":
                next_character = text[index + 1] if index + 1 < len(text) else ""
                previous_character = text[index - 1] if index else ""
                # 小数(数字.数字)不是句末。
                decimal = previous_character.isdigit() and next_character.isdigit()
                # 取出句首到当前点之间的"缩写词"候选(如 "...fig.")。
                token_match = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)\.$", text[start : index + 1])
                token = token_match.group(1).casefold() if token_match else ""
                # 首字母缩写(如 u.s./n.y.),每段 1-2 字符。
                initialism = "." in token and all(
                    1 <= len(part) <= 2 for part in token.split(".")
                )
                # 句末标点后应跟空白或闭合符(否则可能是缩写中间)。
                punctuation_context = (
                    not next_character
                    or next_character.isspace()
                    or next_character in closers
                )
                if token in nonterminal_abbreviations:
                    # 非句末缩写:绝不视为边界。
                    boundary = False
                elif token in terminal_capable_abbreviations or initialism:
                    # 句末可断缩写:需结合后续字符(闭合符+空白,再跟大写/
                    # 中文/行尾)判断是否为真正的句末,避免 "etc. and" 误切。
                    probe = index + 1
                    while probe < len(text) and text[probe] in closers:
                        probe += 1
                    if probe < len(text) and not text[probe].isspace():
                        boundary = False
                    else:
                        while probe < len(text) and text[probe].isspace():
                            probe += 1
                        # 跳过空白后看下一字符:大写/中文/行尾 => 句末。
                        following = text[probe] if probe < len(text) else ""
                        boundary = punctuation_context and (
                            not following
                            or following.isupper()
                            or "\u3400" <= following <= "\u9fff"
                        )
                else:
                    # 普通 ".":非小数且后续符合标点上下文时才视为边界。
                    boundary = not decimal and punctuation_context
            if not boundary:
                index += 1
                continue
            # 确定边界:吃掉闭合符与尾部空白,记录句子区间。
            end = index + 1
            while end < len(text) and text[end] in closers:
                end += 1
            while end < len(text) and text[end].isspace():
                end += 1
            ranges.append((start, end, text[start:end]))
            start = end
            index = end
        # 尾部剩余无标点的文本作为最后一个句子。
        if start < len(text):
            ranges.append((start, len(text), text[start:]))
        return ranges

    def _attach_embeddings(self, units: list[_RawUnit]) -> None:
        """为每个 raw unit 批量生成 embedding,并做完整性校验。

        - embedder 支持 ``embed(list[str])`` 或直接可调用;
        - 校验返回向量数量与 unit 数一致、维度一致且均为有限数值;
        - 记录 ``zero_vector`` 标记(全零向量),供后续语义分组降级判断。
        """
        texts = [unit.text for unit in units]
        # 调用 embedder:优先用 embed 方法,其次按可调用对象处理。
        if hasattr(self.embedder, "embed") and callable(self.embedder.embed):
            response = self.embedder.embed(texts)
        elif callable(self.embedder):
            response = self.embedder(texts)
        else:
            raise TypeError("embedder must be callable or expose embed(list[str])")
        try:
            vectors = list(response)
        except TypeError as exc:
            raise ValueError("embedder must return one vector per raw unit") from exc
        # 数量必须一一对应。
        if len(vectors) != len(units):
            raise ValueError("embedding batch length does not match raw unit count")

        expected_dimension: int | None = None
        for unit, raw_vector in zip(units, vectors, strict=True):
            try:
                vector = list(raw_vector)
            except TypeError as exc:
                raise ValueError("embedding vector must be an iterable of finite numbers") from exc
            if not vector:
                raise ValueError("embedding vector dimension must be positive")
            # 所有向量的维度必须一致。
            if expected_dimension is None:
                expected_dimension = len(vector)
            elif len(vector) != expected_dimension:
                raise ValueError("embedding vector dimension mismatch")
            # 元素必须是有限数值(排除 bool、非数值、无穷大/NaN)。
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in vector
            ):
                raise ValueError("embedding vectors must contain finite numbers")
            unit.embedding = [float(value) for value in vector]
            # 记录是否为零向量(范数为 0),语义分组时无法用其判定边界。
            unit.zero_vector = math.sqrt(sum(value * value for value in unit.embedding)) == 0.0

    def _append_narrative(
        self,
        segment: _NarrativeSegment,
        parse_version: str,
        drafts: list[ChunkDraft],
    ) -> None:
        """把一个叙述段切分为 parent + child 两级 chunk 并追加到 drafts。

        流程:
        1. 先做无损单元展开(_expanded_narrative_units,处理超长句子);
        2. 按 parent 的 token 预算做语义分组(_semantic_groups),得到若干
           父组;每个父组生成一个 parent chunk;
        3. 每个父组内部再按 child 预算生成若干 child chunk(带重叠);
        4. 同一结构下的全部 child 建立前后邻接链接。
        """
        expanded = self._expanded_narrative_units(segment)

        # 父级语义分组:每组对应一个 parent chunk。
        parent_groups = self._semantic_groups(expanded, *self.parent_token_limits)
        # 结构 id:由块类型、章节路径与来源块指纹派生,用于 local_id 去重。
        structure_id = "narrative:" + hashlib.sha256(
            json.dumps(
                [segment.block_type, segment.section_path, self._source_ids(expanded)],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        structure_children: list[ChunkDraft] = []
        for parent_index, parent_group in enumerate(parent_groups):
            units = parent_group.units
            parent_text = self._join_units(units)
            parent_id = self._local_id(
                parse_version,
                self._source_ids(units),
                structure_id,
                "parent",
                parent_index,
                parent_text,
            )
            # 构造 parent chunk;metadata 记录分块过程的关键状态
            # (零向量回退、token 窗口拆分、不可避免的溢出、欠长原因、
            # 语义边界分数、来源映射等),供审计与调试。
            parent = self._draft(
                local_id=parent_id,
                parse_version=parse_version,
                chunk_role="parent",
                block_type=segment.block_type,
                text=parent_text,
                parent_local_id=None,
                source_block_ids=self._source_ids(units),
                source_spans=self._source_spans(units),
                section_path=segment.section_path,
                ordinal=len(drafts),
                metadata={
                    # 组内是否有零向量(embedding 退化为 0)的单元。
                    "semantic_zero_vector_fallback": any(unit.zero_vector for unit in units),
                    # 是否有单元因超长被拆成 token 窗口。
                    "token_window_split": any(unit.token_window_split for unit in units),
                    # 是否发生不可避免的 token 溢出(单个字符都超上限)。
                    "unavoidable_token_overflow": any(
                        unit.unavoidable_overflow for unit in units
                    ),
                    "overflow_reason": next(
                        (
                            unit.overflow_reason
                            for unit in units
                            if unit.overflow_reason is not None
                        ),
                        None,
                    ),
                    # 是否低于 min(欠长)及其原因。
                    "parent_min_underflow": self._count_tokens(parent_text)
                    < self.parent_token_limits[0],
                    "undersized_reason": (
                        "section_too_short_or_max_prevents_rebalance"
                        if self._count_tokens(parent_text) < self.parent_token_limits[0]
                        else None
                    ),
                    # 组内覆盖的 raw unit id 列表(用于保真审计)。
                    "source_unit_ids": [unit.unit_id for unit in units],
                    # 语义边界分数与原因(percentile / max_tokens / 等)。
                    "semantic_boundary_score": parent_group.boundary_score,
                    "boundary_reason": parent_group.boundary_reason,
                    # 来源 span 映射类型(exact / approximate)。
                    "source_span_mapping": self._source_span_mapping(units),
                    # 跨 block 拼接时的结构分隔符记录。
                    "structural_separators": self._structural_separators(units),
                },
            )
            drafts.append(parent)
            # 在该父组内部切分 child(带重叠)。
            children = self._child_drafts(
                units,
                parent,
                structure_id=f"{structure_id}:parent:{parent_index}",
                start_ordinal=len(drafts),
            )
            drafts.extend(children)
            structure_children.extend(children)
        # 把同一结构下的所有 child 串成邻接链表。
        self._link_neighbors(structure_children)

    def _expanded_narrative_units(
        self,
        segment: _NarrativeSegment,
    ) -> list[_RawUnit]:
        """对超长 raw unit 做无损窗口拆分,得到"可直接分组的展开单元"列表。

        如果一个句子的 token 数超过窗口上限(取 parent 与 child 上限的较小
        值),就用 _lossless_token_windows 把它无损拆成若干窗口子单元
        (token_window_split=True)。拆出的子单元保持来源 block/span 信息,
        并把 block_char_start/end 调整为窗口对应的字符区间。
        若某个窗口仍然超过上限(说明单个字符都超限),标记 unavoidable_overflow。
        """
        expanded: list[_RawUnit] = []
        parent_max = self.parent_token_limits[2]
        for unit in segment.units:
            # 窗口上限取 parent 与 child 上限的较小者,保证切出的单元
            # 既能参与 child 分组又不突破任一硬上限。
            window_max = min(parent_max, self.child_token_limits[2])
            if unit.token_count <= window_max:
                expanded.append(unit)
                continue
            # 无损拆分:窗口按 token 预算用二分切出,不丢失字符。
            windows = self._lossless_token_windows(unit.text, window_max)
            for window_index, text in enumerate(windows):
                relative_start = sum(len(item) for item in windows[:window_index])
                token_count = self._count_tokens(text)
                expanded.append(
                    unit.copy_with(
                        text=text,
                        token_count=token_count,
                        token_window_split=True,
                        window_index=window_index,
                        # 调整字符区间到窗口对应的相对位置。
                        block_char_start=unit.block_char_start + relative_start,
                        block_char_end=unit.block_char_start + relative_start + len(text),
                        # 极少数情况窗口仍超限:记录为不可避免的溢出。
                        unavoidable_overflow=token_count > window_max,
                        overflow_reason=(
                            "single_source_character_exceeds_max"
                            if token_count > window_max
                            else None
                        ),
                    )
                )
        return expanded

    def _semantic_groups(
        self,
        units: list[_RawUnit],
        minimum: int,
        target: int,
        maximum: int,
    ) -> list[_ChunkGroup]:
        """把 raw unit 序列按 token 预算与语义相似度切成若干组。

        算法概要:
        1. 从当前位置 start 出发,用单调前缀计数(或线性计数)找到"不
           超过 maximum"的最大合法结束位置 max_end;
        2. 在 [max(minimum, target) 达标] 的候选边界中,计算相邻单元
           embedding 的余弦相似度,把最低相似度的"分位"选为语义断点;
        3. 若语义断点会突破 maximum 或无法达标,则退回到 max_end
           (section_end / max_tokens);
        4. 循环直到覆盖全部单元;最后对欠长的尾部组做重平衡。

        返回值是 _ChunkGroup 列表,每组带有边界分数与边界原因
        (semantic_percentile / max_tokens / section_end / min_rebalance)。
        """
        if not units:
            return []
        # token 计数缓存:避免重复统计同一区间。
        token_cache: dict[tuple[int, int], int] = {}
        # 是否可用"前缀单调"计数(可启用二分快速探测)。
        monotonic_prefix_counts = self._uses_monotonic_prefix_counts()

        def range_tokens(start: int, end: int) -> int:
            """统计 units[start:end] 的 token 数(带缓存)。"""
            key = (start, end)
            if key not in token_cache:
                token_cache[key] = self._group_tokens(units[start:end])
            return token_cache[key]

        # search_tokens 是预算探测用的计数函数:优先用基于字符偏移的近似
        # 计数(更快),否则退回精确的 range_tokens。
        search_tokens = range_tokens
        offsets_are_searchable = False
        if not monotonic_prefix_counts:
            offset_range_counter = self._offset_range_counter(units)
            if offset_range_counter is not None:
                # 固定全文偏移是单调的搜索线索。下面 emit 的硬上限仍以精确
                # 区间计数为准(这里只把精确计数结果替换为偏移计数)。
                search_tokens = offset_range_counter
                monotonic_prefix_counts = True
                offsets_are_searchable = True

        def largest_fitting_end(start: int) -> int:
            """从 start 出发,找到不超 maximum 的最大结束位置(不含)。

            非单调计数:线性向后探测。单调计数:先用指数增长找上界,再
            在 [lower, upper] 上二分精确定位。
            """
            if not monotonic_prefix_counts:
                end = start + 1
                while end <= len(units):
                    if range_tokens(start, end) > maximum:
                        return end if end == start + 1 else end - 1
                    end += 1
                return len(units)
            # 单个单元就已超限:只能取 start+1(仍会溢出,留待上层处理)。
            first_end = start + 1
            if search_tokens(start, first_end) > maximum:
                return first_end
            lower = first_end
            distance = 1
            # 指数扩大探测窗口,直到某次超出 maximum,得到上界。
            while lower < len(units):
                distance *= 2
                probe = min(len(units), start + distance)
                if search_tokens(start, probe) > maximum:
                    upper = probe - 1
                    break
                lower = probe
            else:
                # 直到末尾都不超限:直接取末尾。
                return lower
            # 在 (lower, upper] 上二分找最后一个不超限的位置。
            while lower < upper:
                middle = (lower + upper + 1) // 2
                if search_tokens(start, middle) <= maximum:
                    lower = middle
                else:
                    upper = middle - 1
            return lower

        def first_end_reaching(start: int, end: int, threshold: int) -> int | None:
            """在 [start, end] 内二分找第一个使 token 数 >= threshold 的结束位置。

            返回 None 表示整个区间都不达标。
            """
            if search_tokens(start, end) < threshold:
                return None
            lower = start + 1
            upper = end
            while lower < upper:
                middle = (lower + upper) // 2
                if search_tokens(start, middle) >= threshold:
                    upper = middle
                else:
                    lower = middle + 1
            return lower

        groups: list[list[_RawUnit]] = []
        # 记录每组末尾单元对应的 (边界分数, 原因),供最终结果复用。
        boundary_audit: dict[str, tuple[float | None, str]] = {}
        start = 0
        while start < len(units):
            # 1) 找到合法的最大结束位置。
            max_end = largest_fitting_end(start)
            if offsets_are_searchable:
                # 偏移近似计数可能高估;用精确计数收紧到真正不超限的位置。
                while (
                    max_end > start + 1
                    and range_tokens(start, max_end) > maximum
                ):
                    max_end -= 1

            # 2) 收集候选边界:结束位置满足 token >= threshold(= max(min, target)),
            #    计算该边界的语义分数 = 相邻单元 embedding 的余弦相似度
            #    (相似度越低越可能是语义断裂处,越适合做断点)。
            candidates: list[tuple[float, int]] = []
            threshold = max(minimum, target)
            if monotonic_prefix_counts:
                # 前缀单调:从第一个达标位置开始枚举候选,后续都达标。
                first_candidate = first_end_reaching(start, max_end, threshold)
                candidate_ends = (
                    range(first_candidate, min(max_end + 1, len(units)))
                    if first_candidate is not None
                    else ()
                )
            else:
                # 非单调:线性枚举并逐段精确计数判断是否达标。
                candidate_ends = (
                    end
                    for end in range(start + 1, min(max_end + 1, len(units)))
                    if range_tokens(start, end) >= threshold
                )
            for end in candidate_ends:
                left = units[end - 1].embedding
                right = units[end].embedding
                if left is None or right is None:
                    raise RuntimeError("raw unit embedding missing")
                candidates.append((self._cosine(left, right), end))

            # 3) 从候选里挑选语义断点:取"相似度最低的 break_percentile 分位"
            #    中的最低者(即最像断裂处的位置)。
            if candidates:
                percentile_count = max(
                    1,
                    math.ceil(len(candidates) * self.break_percentile / 100),
                )
                # 取相似度最低的前 N% 作为备选(低相似度 = 高断裂可能性)。
                bottom = sorted(candidates, key=lambda item: (item[0], item[1]))[
                    :percentile_count
                ]
                # 从中再选相似度最低、且最靠前的边界。
                score, chosen_end = min(bottom, key=lambda item: item[1])
                chosen_tokens = range_tokens(start, chosen_end)
                max_tokens = range_tokens(start, max_end)
                # 若语义断点本身超限,或选了断点后无法达到 min 而 max_end 能达:
                # 放弃语义断点,退回 max_end。
                if chosen_tokens > maximum or (
                    chosen_tokens < threshold <= max_tokens
                ):
                    chosen_end = max_end
                    score = None
                    reason = (
                        "section_end" if chosen_end >= len(units) else "max_tokens"
                    )
                else:
                    reason = "semantic_percentile"
            else:
                # 没有达标的语义候选:只能切到 max_end。
                chosen_end = max_end
                score = None
                reason = "section_end" if chosen_end >= len(units) else "max_tokens"
            group = units[start:chosen_end]
            groups.append(group)
            boundary_audit[group[-1].unit_id] = (score, reason)
            start = chosen_end

        # 4) 尾部重平衡:最后一个组如果欠长,尝试合并/重分配。
        groups = self._rebalance_tail(groups, minimum, maximum)
        result: list[_ChunkGroup] = []
        for index, group in enumerate(groups):
            # 为每组还原 (边界分数, 原因)。最后一组永远是 section_end;
            # 重平衡可能改变分组,若边界审计里没有对应单元,则重新计算
            # 相邻组边界分数,原因是 min_rebalance。
            if index + 1 == len(groups):
                score, reason = None, "section_end"
            elif group[-1].unit_id in boundary_audit:
                score, reason = boundary_audit[group[-1].unit_id]
            else:
                left = group[-1].embedding
                right = groups[index + 1][0].embedding
                score = (
                    self._cosine(left, right)
                    if left is not None and right is not None
                    else None
                )
                reason = "min_rebalance"
            result.append(
                _ChunkGroup(
                    units=group,
                    boundary_score=score,
                    boundary_reason=reason,
                )
            )
        return result

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        """计算两个 embedding 向量的余弦相似度。

        任一向量的范数为 0 时返回 0.0(无法比较,视为无相似性)。
        """
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)

    def _safe_token_windows(self, text: str, maximum: int) -> list[str]:
        """把文本按 token 上限切成无损窗口(贪婪 + 超长原子细分)。

        按"原子"(单词 + 其后空格)贪心累积;当前窗口再加一个原子会超限时
        先收下当前窗口;单个原子本身超限时用 _split_oversized_atom 按 token
        预算二分细分。保证每个窗口 token 数不超过 maximum(否则抛错)。
        """
        # 以"词 + 其后空白"为原子,保证切分处不丢空格。
        atoms = re.findall(r"\S+\s*", text)
        if not atoms:
            atoms = list(text)
        windows: list[str] = []
        current = ""
        for atom in atoms:
            candidate = current + atom
            # 加入该原子会超限:先把当前窗口收下。
            if current and self._count_tokens(candidate.strip()) > maximum:
                windows.append(current.strip())
                current = ""
            if self._count_tokens(atom.strip()) <= maximum:
                current += atom
                continue
            # 单个原子就超限:先收下当前窗口,再把原子细分。
            if current:
                windows.append(current.strip())
                current = ""
            windows.extend(self._split_oversized_atom(atom.strip(), maximum))
        if current.strip():
            windows.append(current.strip())
        # 兜底校验:必须能产出无损且全部达标(<= maximum)的窗口。
        if not windows or any(self._count_tokens(item) > maximum for item in windows):
            raise ValueError("token counter cannot produce a lossless parent window within max")
        return windows

    def _split_oversized_atom(self, text: str, maximum: int) -> list[str]:
        """把一个超长的"原子"(无空格连续串)按 token 预算无损切分。

        用二分查找每次切出的最长前缀(其 token 数 <= maximum);若单个字符
        都超限(理论上 token 不可能小于等于 maximum 时),则退化为逐字符切。
        """
        windows: list[str] = []
        remaining = text
        while remaining:
            # 二分:找最长的"token <= maximum"前缀长度 best。
            low, high = 1, len(remaining)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if self._count_tokens(remaining[:middle]) <= maximum:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                # 极端情况:单个字符都超限,只能逐字符切(无损但会溢出标记)。
                windows.append(remaining[0])
                remaining = remaining[1:]
                continue
            windows.append(remaining[:best])
            remaining = remaining[best:]
        return windows

    def _lossless_token_windows(self, text: str, maximum: int) -> list[str]:
        """把一段文本无损切成若干个 token 数 <= maximum 的窗口。

        与 _split_oversized_atom 类似,但面向整段文本:整段可放就整段;
        否则二分切最长合法前缀,剩余继续。用于展开超长句子。
        """
        windows: list[str] = []
        remaining = text
        while remaining:
            # 整段已达标:直接作为最后一个窗口。
            if self._count_tokens(remaining) <= maximum:
                windows.append(remaining)
                break
            # 二分找最长合法前缀。
            low, high = 1, len(remaining)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if self._count_tokens(remaining[:middle]) <= maximum:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                # 极端情况:单字符超限,逐字符切。
                windows.append(remaining[0])
                remaining = remaining[1:]
                continue
            windows.append(remaining[:best])
            remaining = remaining[best:]
        return windows

    def _child_drafts(
        self,
        units: list[_RawUnit],
        parent: ChunkDraft,
        *,
        structure_id: str,
        start_ordinal: int,
    ) -> list[ChunkDraft]:
        """在父 chunk 的单元序列内,按 child 预算切分出若干 child chunk。

        流程:
        1. 对 units 做 child 级语义分组(_semantic_groups);
        2. 在相邻组之间注入"整句重叠"(overlap_tokens 预算内),增强相邻
           child 的上下文连贯性;
        3. 每个组生成一个 child chunk(metadata 记录重叠、溢出、欠长等
           状态),并在结尾建立邻接链接。
        """
        minimum, target, maximum = self.child_token_limits
        # 1) 基础分组(不含重叠)。
        base_chunk_groups = self._semantic_groups(units, minimum, target, maximum)
        base_groups = [item.units for item in base_chunk_groups]

        # 2) 为每个组注入前一组尾部的整句重叠。
        child_groups: list[tuple[list[_RawUnit], int]] = []
        for index, group in enumerate(base_groups):
            overlap: list[_RawUnit] = []
            if index > 0 and self.overlap_tokens:
                # 从前一组尾部取若干整句,使重叠 token 数接近预算且不超
                # child 的 maximum。
                overlap = self._whole_sentence_overlap(
                    base_groups[index - 1],
                    group,
                    maximum,
                )
            # 记录 (重叠 + 本组, 重叠句数)。
            child_groups.append(([*overlap, *group], len(overlap)))

        children: list[ChunkDraft] = []
        for child_index, (group, overlap_count) in enumerate(child_groups):
            text = self._join_units(group)
            source_ids = self._source_ids(group)
            local_id = self._local_id(
                parent.parse_version,
                source_ids,
                structure_id,
                "child",
                child_index,
                text,
            )
            base_group = base_groups[child_index]
            # 构造 child chunk,metadata 记录与重叠/溢出/欠长相关的诊断状态。
            children.append(
                self._draft(
                    local_id=local_id,
                    parse_version=parent.parse_version,
                    chunk_role="child",
                    block_type=parent.block_type,
                    text=text,
                    parent_local_id=parent.local_id,
                    source_block_ids=source_ids,
                    source_spans=self._source_spans(group),
                    section_path=parent.section_path,
                    ordinal=start_ordinal + child_index,
                    metadata={
                        # 与前一组重叠的句子数/token 数。
                        "overlap_sentence_count": overlap_count,
                        "overlap_tokens": self._group_tokens(group[:overlap_count]),
                        # 单句就超限(基础组只有一个句子且超 max)。
                        "single_sentence_overflow": len(base_group) == 1
                        and base_group[0].token_count > maximum,
                        # 不可避免的溢出(展开时已标记)。
                        "unavoidable_token_overflow": any(
                            item.unavoidable_overflow for item in base_group
                        ),
                        "overflow_reason": next(
                            (
                                item.overflow_reason
                                for item in base_group
                                if item.overflow_reason is not None
                            ),
                            None,
                        ),
                        # 组内是否有被 token 窗口拆分的单元。
                        "source_sentence_token_window": any(
                            item.token_window_split for item in group
                        ),
                        # 是否低于 min(欠长)及其原因。
                        "child_min_underflow": self._count_tokens(text) < minimum,
                        "undersized_reason": (
                            "structure_too_short_or_max_prevents_rebalance"
                            if self._count_tokens(text) < minimum
                            else None
                        ),
                        # 覆盖的 raw unit id 列表(保真审计用)。
                        "source_unit_ids": [item.unit_id for item in group],
                        # 语义边界分数与原因(继承基础分组)。
                        "semantic_boundary_score": base_chunk_groups[
                            child_index
                        ].boundary_score,
                        "boundary_reason": base_chunk_groups[
                            child_index
                        ].boundary_reason,
                        "source_span_mapping": self._source_span_mapping(group),
                        "structural_separators": self._structural_separators(group),
                    },
                )
            )
        # 把本结构的全部 child 串成邻接链表。
        self._link_neighbors(children)
        return children

    def _rebalance_tail(
        self,
        groups: list[list[_RawUnit]],
        minimum: int,
        maximum: int,
    ) -> list[list[_RawUnit]]:
        """对欠长的尾部组做重平衡,尽量让最后一组达到 min。

        策略依次尝试:
        1. 直接并入前一组的尾部(合并后不超过 maximum);
        2. 否则从前一组尾部逐个搬单元给尾组(搬动过程两组都不超 maximum,
           搬完后两组都 >= minimum 才采纳);
        都不行则保持原样(尾部欠长属于可接受的最小坏情况)。
        """
        # 组数不足 2 或尾部已达标:无需处理。
        if len(groups) < 2 or self._group_tokens(groups[-1]) >= minimum:
            return groups
        previous = groups[-2]
        tail = groups[-1]
        # 方案一:尾组合并进前一组(总量不超 max)。
        if self._group_tokens([*previous, *tail]) <= maximum:
            previous.extend(groups.pop())
            return groups

        # 方案二:逐单元把前一组尾部让渡给尾组,直到尾组达标。
        candidate_previous = list(previous)
        candidate_tail = list(tail)
        while self._group_tokens(candidate_tail) < minimum and len(candidate_previous) > 1:
            candidate_tail.insert(0, candidate_previous.pop())
            # 任一时刻某组超限就放弃本次重平衡。
            if (
                self._group_tokens(candidate_previous) > maximum
                or self._group_tokens(candidate_tail) > maximum
            ):
                return groups
        # 两组都达标才采纳。
        if (
            self._group_tokens(candidate_previous) >= minimum
            and self._group_tokens(candidate_tail) >= minimum
        ):
            groups[-2] = candidate_previous
            groups[-1] = candidate_tail
        return groups

    def _whole_sentence_overlap(
        self,
        previous: list[_RawUnit],
        current: list[_RawUnit],
        maximum: int,
    ) -> list[_RawUnit]:
        """从前一组尾部挑选若干整句,作为与当前组的重叠内容。

        从 previous 的末尾向前逐个累加整句(保持整句完整性),直到:
        - 重叠 token 数超过 overlap_tokens 预算;或
        - 加上当前组后会突破 child 的 maximum。
        返回按原顺序排列的重叠单元列表(为空表示不重叠)。
        """
        selected: list[_RawUnit] = []
        for unit in reversed(previous):
            # 从尾部向前累积:新候选 = [unit, *已选(逆序)]。
            candidate = [unit, *reversed(selected)]
            # 超出重叠预算则停止增加。
            if self._group_tokens(candidate) > self.overlap_tokens:
                break
            # 加上当前组会超 max 则停止增加。
            if self._group_tokens([*candidate, *current]) > maximum:
                break
            selected.append(unit)
        selected.reverse()
        return selected

    def _append_structured(
        self,
        entry: _StructuredEntry,
        document: CanonicalDocument,
        drafts: list[ChunkDraft],
    ) -> None:
        """把结构化对象(表格/图/公式)转换为 parent + child 两级 chunk。

        流程:
        1. 按块类型找到对应的结构化源对象(table/figure/formula),若缺失
           直接抛源保真错误;
        2. 用 StructuredEvidenceBuilder 生成源级 parent/child 证据块
           (source_parent / source_children),以 child 的 maximum 为 token
           上限;
        3. 收集 contributors(邻近的可检索叙述块)并整合来源 span;
        4. 把源块转成 ChunkDraft 并追加到 drafts,子块建立邻接链接。
        """
        block = entry.block
        source_parent: StructuredEvidenceChunk
        source_children: list[StructuredEvidenceChunk]
        structure_id: str
        structure_source_ids = [block.block_id]
        structure_source_spans: list[SourceSpan] = []
        figure_asset_id: str | None = None
        # ---- 表格 ----
        if block.block_type == "table":
            # 定位结构化的表格对象;缺失说明解析产物不一致。
            table = next(
                (item for item in document.tables if item.table_id == block.table_id),
                None,
            )
            if table is None:
                self._raise_source_fidelity(
                    document,
                    page=self._first_page(block.source_spans),
                    block_type="table",
                    source_id=str(block.table_id or block.block_id),
                    excerpt=str(block.table_id or block.text or block.block_id),
                    reason="structured_source_object_missing",
                )
            # 用证据构建器生成表格的 parent/child 源块。
            source_parent, source_children = self._structured_builder.table_chunks(
                table, max_tokens=self.child_token_limits[2]
            )
            structure_id = table.table_id
            # 合并 block 与源块来源 span 并去重。
            structure_source_spans = self._deduplicate_spans(
                [*block.source_spans, *source_parent.source_spans]
            )
            # 在 span 上标注 table_id(便于回溯)。
            structure_source_spans = self._annotate_structure_spans(
                structure_source_spans,
                table_id=table.table_id,
            )
            # 子块 span 同样标注 table_id。
            source_children = [
                child.model_copy(
                    update={
                        "source_spans": self._annotate_structure_spans(
                            child.source_spans,
                            table_id=table.table_id,
                        )
                    }
                )
                for child in source_children
            ]
        # ---- 图 ----
        elif block.block_type == "figure":
            figure = next(
                (item for item in document.figures if item.figure_id == block.figure_id),
                None,
            )
            if figure is None:
                self._raise_source_fidelity(
                    document,
                    page=self._first_page(block.source_spans),
                    block_type="figure",
                    source_id=str(block.figure_id or block.block_id),
                    excerpt=str(block.figure_id or block.text or block.block_id),
                    reason="structured_source_object_missing",
                )
            # 邻近的可检索叙述块作为"贡献者",给图提供上下文。
            contributors = self._structured_contributors(
                figure.nearby_block_ids,
                document.blocks,
            )
            source_parent, source_children = self._structured_builder.figure_chunks(
                figure,
                contributors,
                max_tokens=self.child_token_limits[2],
            )
            structure_id = figure.figure_id
            # 整合 block、源块与贡献者的来源 id/span。
            structure_source_ids, structure_source_spans = self._structured_sources(
                block,
                source_parent.source_spans,
                contributors,
            )
            # 解析资源 asset_id 并标注到 span 上。
            figure_asset_id = self._figure_asset_id(document, figure)
            structure_source_spans = self._annotate_structure_spans(
                structure_source_spans,
                figure_id=figure.figure_id,
                asset_id=figure_asset_id,
            )
        # ---- 公式 ----
        elif block.block_type == "formula":
            formula = next(
                (item for item in document.formulas if item.formula_id == block.formula_id),
                None,
            )
            if formula is None:
                self._raise_source_fidelity(
                    document,
                    page=self._first_page(block.source_spans),
                    block_type="formula",
                    source_id=str(block.formula_id or block.block_id),
                    excerpt=str(block.formula_id or block.text or block.block_id),
                    reason="structured_source_object_missing",
                )
            contributors = self._structured_contributors(
                formula.nearby_block_ids,
                document.blocks,
            )
            source_parent, source_children = self._structured_builder.formula_chunks(
                formula,
                contributors,
                max_tokens=self.child_token_limits[2],
            )
            structure_id = formula.formula_id
            structure_source_ids, structure_source_spans = self._structured_sources(
                block,
                source_parent.source_spans,
                contributors,
            )
            structure_source_spans = self._annotate_structure_spans(
                structure_source_spans,
                formula_id=formula.formula_id,
            )
        else:  # pragma: no cover - guarded by entry construction
            raise ValueError(f"unsupported structured block type: {block.block_type}")

        # 父 chunk:来源于源级 parent 块,来源 span 用合并后的结构 span。
        parent_id = self._local_id(
            document.parse_version,
            structure_source_ids,
            structure_id,
            "parent",
            0,
            source_parent.text,
        )
        parent = self._draft(
            local_id=parent_id,
            parse_version=document.parse_version,
            chunk_role="parent",
            block_type=block.block_type,
            text=source_parent.text,
            embedding_text=source_parent.embedding_text,
            parent_local_id=None,
            source_block_ids=structure_source_ids,
            source_spans=structure_source_spans,
            section_path=entry.section_path,
            ordinal=len(drafts),
            metadata={
                # 继承证据构建器的元数据,并记录结构化 chunk id 与边界原因。
                **source_parent.metadata,
                "structured_chunk_id": source_parent.chunk_id,
                "semantic_boundary_score": None,
                "boundary_reason": "structured_boundary",
            },
        )
        drafts.append(parent)

        # 子 chunk:把源级子块逐个转成 ChunkDraft。
        children: list[ChunkDraft] = []
        for child_index, source_child in enumerate(source_children):
            child_id = self._local_id(
                document.parse_version,
                structure_source_ids,
                structure_id,
                "child",
                child_index,
                source_child.text,
            )
            metadata = {
                **source_child.metadata,
                "structured_chunk_id": source_child.chunk_id,
                "semantic_boundary_score": None,
                "boundary_reason": "structured_boundary",
            }
            # 图/公式只有一个子块时,标记为"源保真的单子块"(整块即子块)。
            if block.block_type in {"figure", "formula"} and len(source_children) == 1:
                metadata["source_faithful_single_child"] = True
            children.append(
                self._draft(
                    local_id=child_id,
                    parse_version=document.parse_version,
                    chunk_role="child",
                    block_type=block.block_type,
                    text=source_child.text,
                    embedding_text=source_child.embedding_text,
                    parent_local_id=parent_id,
                    source_block_ids=structure_source_ids,
                    # 表格子块用其自身的细粒度 span;图/公式用结构级 span。
                    source_spans=(
                        source_child.source_spans
                        if block.block_type == "table"
                        else structure_source_spans
                    ),
                    section_path=entry.section_path,
                    ordinal=len(drafts) + child_index,
                    metadata=metadata,
                )
            )
        self._link_neighbors(children)
        drafts.extend(children)

    @staticmethod
    def _annotate_structure_spans(
        spans: Iterable[SourceSpan],
        *,
        table_id: str | None = None,
        figure_id: str | None = None,
        formula_id: str | None = None,
        asset_id: str | None = None,
    ) -> list[SourceSpan]:
        """给来源 span 批量标注结构元数据(table_id/figure_id/...)。

        对已有的 span,在不覆盖已有字段的前提下把结构 id 写入其 metadata;
        若 span 列表为空,则构造一个仅含元数据的占位 span(便于审计时仍能
        追溯到所属结构)。
        """
        source_spans = list(spans)
        if not source_spans:
            # 无 span 时构造占位 span,只带非空的结构元数据。
            metadata = {
                key: value
                for key, value in {
                    "table_id": table_id,
                    "figure_id": figure_id,
                    "formula_id": formula_id,
                    "asset_id": asset_id,
                }.items()
                if value is not None
            }
            return [SourceSpan(metadata=metadata)] if metadata else []
        annotated: list[SourceSpan] = []
        for span in source_spans:
            metadata = dict(span.metadata)
            # 仅当 span 尚无对应字段时才写入,避免覆盖更精确的信息。
            if table_id is not None and span.table_id is None:
                metadata["table_id"] = table_id
            if figure_id is not None:
                metadata["figure_id"] = figure_id
            if formula_id is not None:
                metadata["formula_id"] = formula_id
            if asset_id is not None:
                metadata["asset_id"] = asset_id
            annotated.append(
                span.model_copy(
                    update={"metadata": metadata}
                )
            )
        return annotated

    @staticmethod
    def _figure_asset_id(document: CanonicalDocument, figure) -> str | None:
        """解析图表对应的资源(asset)id。

        优先用 figure 自身 metadata 里的 asset_id;否则根据 asset_path
        (规范化路径)在文档资产列表中匹配,返回匹配资产的 asset_id。
        """
        # 优先使用显式声明的 asset_id。
        metadata_asset_id = figure.metadata.get("asset_id")
        if isinstance(metadata_asset_id, str) and metadata_asset_id.strip():
            return metadata_asset_id.strip()
        # 否则按路径匹配:统一分隔符为 "/" 并去掉 "./" 前缀。
        asset_path = str(figure.asset_path or "").replace("\\", "/").removeprefix("./")
        if not asset_path:
            return None
        for asset in document.assets:
            candidate_path = str(asset.path or "").replace("\\", "/").removeprefix("./")
            if candidate_path == asset_path:
                return asset.asset_id
        return None

    @classmethod
    def _structured_contributors(
        cls,
        nearby_block_ids: list[str],
        blocks: Iterable[CanonicalBlock],
    ) -> list[CanonicalBlock]:
        """选取结构化对象(图/公式)的"贡献者"叙述块。

        只保留满足全部条件的邻近块:是 near-by 列表中的成员、可检索、
        属于 narrative/appendix、非参考文献章节、非生成内容、且有正文。
        这些块为图/公式 chunk 提供叙述上下文。
        """
        wanted = set(nearby_block_ids)
        return [
            block
            for block, section_path in cls._blocks_with_effective_section_paths(blocks)
            if block.block_id in wanted
            and block.retrievable
            and block.block_type in {"narrative", "appendix"}
            and cls._is_retrievable_narrative(block, section_path)
            and not block_is_generated(block)
            and block.text.strip()
        ]

    @classmethod
    def _structured_sources(
        cls,
        structure_block: CanonicalBlock,
        structure_spans: Iterable[SourceSpan],
        contributors: Iterable[CanonicalBlock],
    ) -> tuple[list[str], list[SourceSpan]]:
        """整合结构化块的来源 id 列表与来源 span 列表。

        来源 id = 结构块自身 + 各贡献者块(去重);span 合并结构块自身、
        结构级 span 与所有贡献者的 span,再去重。
        """
        contributors = list(contributors)
        source_ids = [structure_block.block_id]
        for block in contributors:
            if block.block_id not in source_ids:
                source_ids.append(block.block_id)
        spans = cls._deduplicate_spans(
            [
                *structure_block.source_spans,
                *structure_spans,
                *(span for item in contributors for span in item.source_spans),
            ]
        )
        return source_ids, spans

    def _draft(
        self,
        *,
        local_id: str,
        parse_version: str,
        chunk_role: Literal["parent", "child"],
        block_type: str,
        text: str,
        parent_local_id: str | None,
        source_block_ids: list[str],
        source_spans: Iterable[SourceSpan],
        section_path: list[str],
        ordinal: int,
        metadata: dict[str, Any],
        embedding_text: str | None = None,
    ) -> ChunkDraft:
        """构造一个 ChunkDraft,统一处理默认值、token 计数与 span 去重。

        - embedding_text 缺省与 text 相同;
        - token_count 基于 embedding_text 计算(上下文化后前缀也算在内);
        - source_spans 去重后写入;
        - 记录 splitter 名称/版本/模型与语义边界分数。
        """
        return ChunkDraft(
            local_id=local_id,
            parse_version=parse_version,
            chunk_role=chunk_role,
            block_type=block_type,
            text=text,
            embedding_text=text if embedding_text is None else embedding_text,
            token_count=self._count_tokens(text if embedding_text is None else embedding_text),
            parent_local_id=parent_local_id,
            source_block_ids=source_block_ids,
            source_spans=self._deduplicate_spans(source_spans),
            section_path=section_path,
            ordinal=ordinal,
            splitter_name=self.splitter_name,
            splitter_version=self.splitter_version,
            splitting_model=self.splitting_model,
            semantic_boundary_score=metadata.get("semantic_boundary_score"),
            metadata=metadata,
        )

    @staticmethod
    def _link_neighbors(children: list[ChunkDraft]) -> None:
        """把同一结构下的 child 串成双向邻接链表(前后 sibling 引用)。"""
        for index, child in enumerate(children):
            child.previous_child_local_id = children[index - 1].local_id if index else None
            child.next_child_local_id = (
                children[index + 1].local_id if index + 1 < len(children) else None
            )

    @staticmethod
    def _join_units(units: Iterable[_RawUnit]) -> str:
        """把若干 raw unit 的文本按原顺序拼接;跨 block 时插入双换行。"""
        values = list(units)
        if not values:
            return ""
        parts = [values[0].text]
        for previous, current in zip(values, values[1:]):
            # 跨 block 时用空行分隔,保证还原后的文本结构与源一致。
            if previous.block_id != current.block_id:
                parts.append("\n\n")
            parts.append(current.text)
        return "".join(parts)

    @staticmethod
    def _structural_separators(units: Iterable[_RawUnit]) -> list[dict[str, Any]]:
        """记录拼接时跨 block 插入的结构分隔符(用于还原/审计)。"""
        values = list(units)
        return [
            {
                "text": "\n\n",
                "source_backed": False,
                "between_block_ids": [previous.block_id, current.block_id],
            }
            for previous, current in zip(values, values[1:])
            if previous.block_id != current.block_id
        ]

    def _group_tokens(self, units: Iterable[_RawUnit]) -> int:
        """统计一组 raw unit 拼接后的 token 数。"""
        return self._count_tokens(self._join_units(units))

    @staticmethod
    def _source_ids(units: Iterable[_RawUnit]) -> list[str]:
        """返回一组 unit 覆盖的、按出现顺序去重的 block id 列表。"""
        result: list[str] = []
        seen: set[str] = set()
        for unit in units:
            if unit.block_id not in seen:
                result.append(unit.block_id)
                seen.add(unit.block_id)
        return result

    @classmethod
    def _source_spans(cls, units: Iterable[_RawUnit]) -> list[SourceSpan]:
        """把一组 unit 的来源 span 映射/合并为一个 span 列表。

        当 span 的字符区间足够覆盖 unit 的 block_char_end 时,按 unit 在
        block 内的相对偏移重算 char_start/char_end(得到精确区间);否则
        保留原 span(近似)。随后去重,并把相邻且定位信息一致(仅字符区间
        连续)的 span 合并为更长的连续区间。
        """
        mapped: list[SourceSpan] = []
        for unit in units:
            for span in unit.source_spans:
                if (
                    span.char_start is not None
                    and span.char_end is not None
                    and span.char_end - span.char_start >= unit.block_char_end
                ):
                    # span 覆盖单元区间:按 block_char_start 偏移映射出精确区间。
                    mapped.append(
                        span.model_copy(
                            update={
                                "char_start": span.char_start + unit.block_char_start,
                                "char_end": span.char_start + unit.block_char_end,
                            }
                        )
                    )
                else:
                    # span 不足以映射:原样保留(近似)。
                    mapped.append(span)
        deduplicated = cls._deduplicate_spans(mapped)
        # 合并相邻 span(同一 locator 且字符区间连续)以减小 span 数量。
        merged: list[SourceSpan] = []
        for span in deduplicated:
            if merged and cls._spans_are_adjacent(merged[-1], span):
                merged[-1] = merged[-1].model_copy(update={"char_end": span.char_end})
            else:
                merged.append(span)
        return merged

    @staticmethod
    def _source_span_mapping(units: Iterable[_RawUnit]) -> str:
        """判断一组 unit 的来源 span 映射是 "exact" 还是 "approximate"。

        只要任一 unit 没有 span、或 span 的字符区间不足以精确映射到该
        unit 的字符区间,就返回 "approximate";全部精确才返回 "exact"。
        """
        for unit in units:
            if not unit.source_spans:
                return "approximate"
            for span in unit.source_spans:
                if (
                    span.char_start is None
                    or span.char_end is None
                    or span.char_end - span.char_start < unit.block_char_end
                ):
                    return "approximate"
        return "exact"

    @staticmethod
    def _spans_are_adjacent(left: SourceSpan, right: SourceSpan) -> bool:
        """判断两个 span 是否"定位信息一致且字符区间首尾相接"。

        去掉 char_start/char_end 后其余字段(定位标识)完全一致,且
        left 的 char_end == right 的 char_start,才视为相邻可合并。
        """
        if left.char_end is None or right.char_start is None:
            return False
        left_locator = left.model_dump(mode="json")
        right_locator = right.model_dump(mode="json")
        left_locator.pop("char_start", None)
        left_locator.pop("char_end", None)
        right_locator.pop("char_start", None)
        right_locator.pop("char_end", None)
        return left_locator == right_locator and left.char_end == right.char_start

    @staticmethod
    def _deduplicate_spans(spans: Iterable[SourceSpan]) -> list[SourceSpan]:
        """按序列化后的完整内容去重 span,保持原顺序。"""
        result: list[SourceSpan] = []
        seen: set[str] = set()
        for span in spans:
            key = json.dumps(
                span.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            if key not in seen:
                result.append(span)
                seen.add(key)
        return result

    @staticmethod
    def _local_id(
        parse_version: str,
        source_block_ids: list[str],
        structure_id: str,
        role: str,
        ordinal: int,
        text: str,
    ) -> str:
        """由分块内容与上下文派生一个确定性的本地 id。

        输入包括:解析版本、来源 block id 列表、结构 id、角色、序号,以及
        文本的 SHA-256 指纹(保证"同内容同 id",防止重复)。输出形如
        "draft-<24位十六进制>"。持久化后 id 会映射到数据库主键。
        """
        payload = json.dumps(
            {
                "parse_version": parse_version,
                "source_block_ids": source_block_ids,
                "structure_id": structure_id,
                "role": role,
                "ordinal": ordinal,
                "source_fingerprint": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"draft-{hashlib.sha256(payload).hexdigest()[:24]}"


@dataclass
class _RawUnit:
    """分块的原子单元:通常是"一个句子"(预格式化文本则是一整块)。

    - text / token_count:单元文本与其 token 数;
    - block_id / block_type / section_path / source_spans:来源信息;
    - sentence_id:形如 "{block_id}:{句内序号}" 的标识;window 拆分后
      通过 unit_id 追加 ":window:{index}";
    - embedding / zero_vector:句向量及其"是否零向量"标记(语义分组用);
    - token_window_split / window_index:是否因超长被拆为 token 窗口;
    - preformatted:是否预格式化文本(整块一个单元);
    - block_char_start / block_char_end:单元在来源 block 内的字符区间;
    - unavoidable_overflow / overflow_reason:是否发生不可避免的溢出。
    """

    text: str
    token_count: int
    block_id: str
    block_type: str
    section_path: list[str]
    source_spans: list[SourceSpan]
    sentence_id: str
    embedding: list[float] | None = None
    zero_vector: bool = False
    token_window_split: bool = False
    window_index: int | None = None
    preformatted: bool = False
    block_char_start: int = 0
    block_char_end: int = 0
    unavoidable_overflow: bool = False
    overflow_reason: str | None = None

    @property
    def unit_id(self) -> str:
        """单元唯一标识:句子 id,若为 token 窗口子单元则附加窗口下标。"""
        suffix = "" if self.window_index is None else f":window:{self.window_index}"
        return f"{self.sentence_id}{suffix}"

    def copy_with(self, **changes: Any) -> _RawUnit:
        """基于自身复制一个 _RawUnit,并可覆盖任意字段。

        用于 token 窗口拆分:复制来源信息的同时修改 text/token_count 等。
        """
        values = {
            "text": self.text,
            "token_count": self.token_count,
            "block_id": self.block_id,
            "block_type": self.block_type,
            "section_path": list(self.section_path),
            "source_spans": list(self.source_spans),
            "sentence_id": self.sentence_id,
            "embedding": self.embedding,
            "zero_vector": self.zero_vector,
            "token_window_split": self.token_window_split,
            "window_index": self.window_index,
            "preformatted": self.preformatted,
            "block_char_start": self.block_char_start,
            "block_char_end": self.block_char_end,
            "unavoidable_overflow": self.unavoidable_overflow,
            "overflow_reason": self.overflow_reason,
        }
        values.update(changes)
        return _RawUnit(**values)


@dataclass
class _ChunkGroup:
    """一次语义分组的结果:一组 raw unit + 边界分数与原因。

    - ``boundary_score``:组末尾处的语义边界分数(相邻单元余弦相似度,
      None 表示非语义边界,如 section_end / max_tokens);
    - ``boundary_reason``:semantic_percentile / max_tokens / section_end
      / min_rebalance 之一。
    """

    units: list[_RawUnit]
    boundary_score: float | None
    boundary_reason: str


@dataclass
class _NarrativeSegment:
    """一段连续的叙述性正文(同块类型、同格式、同章节路径)。

    - ``key``:聚合键(块类型, 格式类型, 预格式化时的块 id, 章节路径);
    - ``block_type`` / ``section_path``:本段统一的块类型与章节路径;
    - ``units``:按句子切分出的 raw unit 列表。
    """

    key: tuple[str, str, str, tuple[str, ...]]
    block_type: str
    section_path: list[str]
    units: list[_RawUnit] = field(default_factory=list)


@dataclass
class _StructuredEntry:
    """一个结构化对象(表格/图/公式)对应的分块条目。"""

    block: CanonicalBlock
    section_path: list[str]
