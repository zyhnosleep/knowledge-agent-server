from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BBox = tuple[float, float, float, float]
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


class CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SourceSpan(CanonicalModel):
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


class CanonicalBlock(CanonicalModel):
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
    status: str = "parsed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalFigure(CanonicalModel):
    figure_id: str
    caption: str | None = None
    asset_path: str | None = None
    source_spans: list[SourceSpan] = Field(default_factory=list)
    nearby_block_ids: list[str] = Field(default_factory=list)
    generated_summary: str | None = None
    analysis_model: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalFormula(CanonicalModel):
    formula_id: str
    latex: str
    caption: str | None = None
    description: str | None = None
    source_spans: list[SourceSpan] = Field(default_factory=list)
    nearby_block_ids: list[str] = Field(default_factory=list)
    generated_explanation: str | None = None
    analysis_model: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalAsset(CanonicalModel):
    asset_id: str
    path: str
    media_type: str
    sha256: str
    source_path: str | None = None
    source_spans: list[SourceSpan] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalQualityIssue(CanonicalModel):
    code: str
    severity: str
    message: str
    block_ids: list[str] = Field(default_factory=list)
    repairable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalQualityReport(CanonicalModel):
    accepted: bool = False
    status: str = "pending"
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    issues: list[CanonicalQualityIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fallback_pages: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalDocument(CanonicalModel):
    document_id: str = ""
    source_path: str | None = None
    source_media_type: str | None = None
    parser_source: str = ""
    parse_version: str = ""

    title: str = ""
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)
    outline: list[str] = Field(default_factory=list)
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
    status: str = "draft"
