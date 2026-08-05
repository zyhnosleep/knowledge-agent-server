"""Canonical 数据模型：定义文档解析后的统一中间表示（Canonical Document）。

本模块是 "Canonical" 层的核心 Schema，所有解析器（PDF、DOCX、HTML、Markdown、TXT）
最终都要把输出转换成这里的 Pydantic 模型。目标是让下游的检索、质量门、持久化、渲染
只依赖一种数据结构，而不用关心原始格式。

主要概念：
- CanonicalDocument：整份文档，包含 blocks（段落/标题/表格块等）、tables、figures、
  formulas、assets、outline（章节树）、quality（质量报告）等。
- CanonicalBlock：文档中的一个逻辑块，例如标题、正文、表格引用、图片引用、公式引用。
  每个 block 都有 block_id、block_type、reading_order（阅读顺序）、source_spans（源位置）。
- SourceSpan：块/结构在原始文档中的定位信息，例如页码、bbox、行号、xpath 等。
- CanonicalTable / CanonicalFigure / CanonicalFormula / CanonicalAsset：
  表格、图片、公式、附件的独立结构。
- CanonicalQualityReport / CanonicalQualityIssue：质量门评估结果。

所有模型继承自 CanonicalModel，它会递归校验字段值必须是 JSON 兼容的（字符串、数字、
布尔、列表、字典），并且浮点数必须是有限值，保证后续序列化到 manifest / JSONL 不会失败。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


# 边界框：x0, y0, x1, y1（左上角 + 右下角，可以是像素或归一化到 0..1）
BBox = tuple[float, float, float, float]

# block 类型枚举：标题、正文、表格、图片、公式、标题说明、附录、参考文献
BlockType = Literal[
    "heading",
    "narrative",
    "table",
    "figure",
    "formula",
    "caption",
    "appendix",
    "reference",
]
TableStatus = Literal[
    "parsed",
    "accepted_mineru",
    "repaired_by_vision",
    "cross_page_merged",
    "validation_failed",
]
AnalysisStatus = Literal["pending", "complete", "failed", "skipped"]
QualitySeverity = Literal["info", "warning", "error", "fatal"]
QualityStatus = Literal[
    "pending",
    "accepted",
    "accepted_with_warnings",
    "rejected",
    "validation_failed",
]
CanonicalDocumentStatus = Literal["draft", "staged", "ready", "failed"]


class CanonicalModel(BaseModel):
    """所有 Canonical 模型的基类。

    - 禁止额外字段（extra="forbid"），避免解析器塞入未声明的数据。
    - 禁止 Inf/NaN（allow_inf_nan=False）。
    - 在模型校验后递归检查所有值都是 JSON 兼容的，确保能安全落盘。
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_finite_json_values(self) -> CanonicalModel:
        self.ensure_json_compatible()
        return self

    def ensure_json_compatible(self) -> None:
        self._validate_json_value(self.model_dump(mode="python"))

    @classmethod
    def _validate_json_value(cls, value: Any) -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("canonical models require finite JSON numbers")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("canonical models require string JSON object keys")
                cls._validate_json_value(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                cls._validate_json_value(item)
            return
        raise ValueError(
            f"canonical models require JSON-compatible values, got {type(value).__name__}"
        )


class SourceSpan(CanonicalModel):
    """块或结构在原始来源中的定位信息。

    不同解析器能提供的粒度不同：PDF 通常有 page_index + bbox；HTML 有 xpath /
    css_selector；DOCX 有 paragraph_id；纯文本有 line_start / line_end /
    char_start / char_end。所有字段都是可选的，但至少应该有一种可定位方式。
    """

    page_index: int | None = Field(default=None, ge=0)
    page_label: str | None = None
    bbox: BBox | None = None
    normalized_bbox: BBox | None = None
    source_block_id: str | None = None

    paragraph_id: str | None = None
    table_id: str | None = None
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    image_relationship_id: str | None = None

    xpath: str | None = None
    css_selector: str | None = None
    element_id: str | None = None
    heading_path: list[str] = Field(default_factory=list)

    line_start: int | None = Field(default=None, ge=0)
    line_end: int | None = Field(default=None, ge=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox", "normalized_bbox")
    @classmethod
    def validate_coordinate_order(cls, value: BBox | None) -> BBox | None:
        if value is None:
            return None
        x0, y0, x1, y1 = value
        if x1 < x0 or y1 < y0:
            raise ValueError("bbox coordinates must satisfy x1 >= x0 and y1 >= y0")
        return value

    @field_validator("normalized_bbox")
    @classmethod
    def validate_normalized_bbox_range(cls, value: BBox | None) -> BBox | None:
        if value is not None and any(coordinate < 0.0 or coordinate > 1.0 for coordinate in value):
            raise ValueError("normalized_bbox coordinates must be within 0..1")
        return value

    @model_validator(mode="after")
    def validate_source_ranges(self) -> SourceSpan:
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("line_start and line_end must be provided together")
        if self.line_start is not None and self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be provided together")
        if self.char_start is not None and self.char_end < self.char_start:
            raise ValueError("char_end must be >= char_start")
        return self


class CanonicalBlock(CanonicalModel):
    """文档中的一个逻辑块，是检索和阅读顺序的基本单位。

    - block_type 决定语义：heading（标题）、narrative（正文）、table（表格块）、
      figure（图片块）、formula（公式块）、caption、appendix、reference。
    - section_path 表示该块所在的标题路径（例如 ["1 引言", "1.1 背景"]）。
    - reading_order 是块在阅读顺序中的位置，必须是 0..N 且唯一。
    - table_id / figure_id / formula_id 用于把块关联到对应的结构化对象。
    - retrievable 表示该块是否应进入向量检索索引（默认标题和参考文献不可检索）。
    """

    block_id: str
    block_type: BlockType
    text: str
    section_path: list[str] = Field(default_factory=list)
    reading_order: int = Field(ge=0)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    parser_source: str
    parser_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    table_id: str | None = None
    figure_id: str | None = None
    formula_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrievable: bool = True

    @model_validator(mode="before")
    @classmethod
    def default_retrievability(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "retrievable" not in value:
            data = dict(value)
            data["retrievable"] = data.get("block_type") not in {
                "heading",
                "reference",
            }
            return data
        return value


class CanonicalCell(CanonicalModel):
    """表格中的一个单元格，包含文本、行列索引、跨行跨列信息。"""

    text: str
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    is_header: bool = False
    bbox: BBox | None = None
    source_spans: list[SourceSpan] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox")
    @classmethod
    def validate_coordinate_order(cls, value: BBox | None) -> BBox | None:
        return SourceSpan.validate_coordinate_order(value)


class CanonicalTable(CanonicalModel):
    """表格的结构化表示。

    提供三种视角：headers + rows（简单网格）、cells（精确单元格，支持 rowspan/colspan）、
    source_html / source_markdown / normalized_markdown（原始与规范化文本）。
    status 描述表格的来源/处理状态，例如 parsed、accepted_mineru、
    repaired_by_vision、cross_page_merged、validation_failed。
    """

    table_id: str
    caption: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    cells: list[CanonicalCell] = Field(default_factory=list)
    source_html: str | None = None
    source_markdown: str | None = None
    normalized_markdown: str | None = None
    footnotes: list[str] = Field(default_factory=list)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    status: TableStatus = "parsed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalFigure(CanonicalModel):
    """图片/图形的结构化表示。

    caption 是标题，description 是可选描述，asset_path 指向持久化后的资源文件。
    analysis_status / ai_* 字段预留用于后续 AI 对图片内容的分析（类型、坐标轴、图例、
    趋势、观察结论等）。
    """

    figure_id: str
    caption: str | None = None
    description: str | None = None
    asset_path: str | None = None
    source_spans: list[SourceSpan] = Field(default_factory=list)
    nearby_block_ids: list[str] = Field(default_factory=list)
    analysis_status: AnalysisStatus = "pending"
    ai_figure_type: str | None = None
    ai_axes: list[JsonValue] | dict[str, JsonValue] = Field(default_factory=dict)
    ai_legend: list[str] = Field(default_factory=list)
    ai_trends: list[str] = Field(default_factory=list)
    ai_observations: list[str] = Field(default_factory=list)
    ai_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    generated_summary: str | None = None
    analysis_model: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalFormula(CanonicalModel):
    """数学公式的结构化表示，统一保存为 LaTeX。"""

    formula_id: str
    latex: str
    caption: str | None = None
    description: str | None = None
    source_spans: list[SourceSpan] = Field(default_factory=list)
    nearby_block_ids: list[str] = Field(default_factory=list)
    analysis_status: AnalysisStatus = "pending"
    ai_variable_explanations: (
        list[dict[str, JsonValue]]
        | dict[str, JsonValue]
    ) = Field(default_factory=list)
    ai_method_role: str | None = None
    ai_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    generated_explanation: str | None = None
    analysis_model: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalAsset(CanonicalModel):
    """文档附件（图片、字体、嵌入文件等）的元数据。

    path 是相对于 canonical bundle 的相对路径（通常以 assets/ 开头），
    source_path 是解析过程中的原始路径，sha256 用于完整性校验。
    """

    asset_id: str
    path: str
    media_type: str
    sha256: str | None = None
    source_path: str | None = None
    source_spans: list[SourceSpan] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalQualityIssue(CanonicalModel):
    """质量门发现的一个问题。

    code 是机器可读的问题类型；severity 是严重程度 info/warning/error/fatal；
    repairable 表示该问题是否可以通过重试/修复流程处理；repair_scope 描述
    需要修复的页面范围（例如 page:3 或 pages:1-2）。
    """

    code: str
    severity: QualitySeverity
    message: str
    block_ids: list[str] = Field(default_factory=list)
    repairable: bool = False
    repair_scope: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalQualityReport(CanonicalModel):
    """质量门对一份 CanonicalDocument 的评估结果。"""

    accepted: bool = False
    status: QualityStatus = "pending"
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    issues: list[CanonicalQualityIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fallback_pages: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionNode(CanonicalModel):
    """章节树中的一个节点，用于构建文档大纲。"""

    title: str
    level: int = Field(default=1, ge=1)
    block_id: str | None = None
    children: list[SectionNode] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalDocument(CanonicalModel):
    """整份文档的 Canonical 表示，是解析 pipeline 的最终输出。

    字段分为几类：
    1. 身份与来源：document_id、source_path、source_media_type、parser_source、parse_version。
    2. 元数据：title、abstract、keywords、outline。
    3. 内容容器：blocks（阅读顺序块）、tables、figures、formulas、assets。
    4. 解析与质量：source_metadata、parser_metadata、quality、warnings、status。
    """

    document_id: str = ""
    source_path: str | None = None
    source_media_type: str | None = None
    parser_source: str = ""
    parse_version: str = ""

    title: str = ""
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    outline: list[SectionNode] = Field(default_factory=list)
    blocks: list[CanonicalBlock] = Field(default_factory=list)
    tables: list[CanonicalTable] = Field(default_factory=list)
    figures: list[CanonicalFigure] = Field(default_factory=list)
    formulas: list[CanonicalFormula] = Field(default_factory=list)
    assets: list[CanonicalAsset] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    source_metadata: dict[str, Any] = Field(default_factory=dict)
    parser_metadata: dict[str, Any] = Field(default_factory=dict)
    quality: CanonicalQualityReport = Field(default_factory=CanonicalQualityReport)
    warnings: list[str] = Field(default_factory=list)
    status: CanonicalDocumentStatus = "draft"
