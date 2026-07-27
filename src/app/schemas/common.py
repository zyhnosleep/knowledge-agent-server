from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProjectCreate(BaseModel):
    slug: str
    name: str
    description: str | None = None


class ProjectRead(BaseModel):
    id: str
    slug: str
    name: str
    description: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    project_id: str
    project_slug: str
    run_id: str
    status: str
    document_title: str


class DocumentRead(BaseModel):
    id: str
    title: str
    file_name: str
    status: str
    sha256: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class ParseProgress(BaseModel):
    stage: str
    percent: int = Field(ge=0, le=100)


class ParseQualitySummary(BaseModel):
    status: str | None = None
    accepted: bool | None = None
    score: float | None = None


class CanonicalParseRead(BaseModel):
    document_id: str
    version: str
    parser: str | None = None
    parser_version: str | None = None
    progress: ParseProgress
    quality: ParseQualitySummary
    repair_pages: list[int] = Field(default_factory=list)
    warning_count: int = Field(ge=0)
    download_available: bool


class CanonicalMarkdownRead(BaseModel):
    document_id: str
    version: str
    markdown: str


class PublicSourceMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    table_id: str | None = Field(default=None, max_length=512)
    figure_id: str | None = Field(default=None, max_length=512)
    formula_id: str | None = Field(default=None, max_length=512)
    asset_id: str | None = Field(default=None, max_length=512)
    source_role: str | None = Field(default=None, max_length=128)
    structure_type: str | None = Field(default=None, max_length=128)


class PublicSourceSpan(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    page_index: int | None = Field(default=None, ge=0)
    page_label: str | None = Field(default=None, max_length=128)
    bbox: tuple[float, float, float, float] | None = None
    normalized_bbox: tuple[float, float, float, float] | None = None
    source_block_id: str | None = Field(default=None, max_length=512)
    paragraph_id: str | None = Field(default=None, max_length=512)
    table_id: str | None = Field(default=None, max_length=512)
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    image_relationship_id: str | None = Field(default=None, max_length=512)
    xpath: str | None = Field(default=None, max_length=4096)
    css_selector: str | None = Field(default=None, max_length=4096)
    element_id: str | None = Field(default=None, max_length=512)
    heading_path: list[str] = Field(default_factory=list, max_length=64)
    line_start: int | None = Field(default=None, ge=0)
    line_end: int | None = Field(default=None, ge=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    metadata: PublicSourceMetadata = Field(default_factory=PublicSourceMetadata)

    @field_validator("bbox", "normalized_bbox", mode="before")
    @classmethod
    def validate_bbox(cls, value):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("bbox must contain exactly four coordinates")
        if any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(coordinate)
            for coordinate in value
        ):
            raise ValueError("bbox coordinates must be finite numbers")
        x0, y0, x1, y1 = (float(coordinate) for coordinate in value)
        if x1 < x0 or y1 < y0:
            raise ValueError("bbox coordinates are out of order")
        return (x0, y0, x1, y1)

    @field_validator("normalized_bbox")
    @classmethod
    def validate_normalized_bbox(cls, value):
        if value is not None and any(coordinate < 0 or coordinate > 1 for coordinate in value):
            raise ValueError("normalized bbox coordinates must be within 0..1")
        return value

    @field_validator("heading_path")
    @classmethod
    def validate_heading_path(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or len(item) > 512 for item in value):
            raise ValueError("heading path entries must be bounded strings")
        return value

    @model_validator(mode="after")
    def validate_ranges(self):
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("line range must contain both endpoints")
        if self.line_start is not None and self.line_end < self.line_start:
            raise ValueError("line range is out of order")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("character range must contain both endpoints")
        if self.char_start is not None and self.char_end < self.char_start:
            raise ValueError("character range is out of order")
        return self


class CitationLocationRead(BaseModel):
    document_id: str
    chunk_id: str
    parse_version: str
    source_type: str
    source_url: str
    source_spans: list[PublicSourceSpan] = Field(default_factory=list)


class QueryRequest(BaseModel):
    project_slug: str
    question: str
    save_answer: bool = True
    document_id: str | None = None


class Citation(BaseModel):
    document_id: str | None = None
    chunk_id: str | None = None
    attachment_id: str | None = None
    page_slug: str | None = None
    page_title: str | None = None
    page_kind: str | None = None
    score: float
    page_label: str | None = None
    excerpt: str
    parse_version: str | None = None
    parent_chunk_id: str | None = None
    block_type: str | None = None
    source_spans: list[dict[str, Any]] = Field(default_factory=list)
    asset_id: str | None = None
    table_id: str | None = None
    figure_id: str | None = None
    formula_id: str | None = None


class QueryResponse(BaseModel):
    answer_markdown: str
    citations: list[Citation]
    verification_status: str


class ReviewItemRead(BaseModel):
    id: str
    title: str
    detail: str
    severity: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    app_name: str
    api_status: str = "ok"
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
    queues: dict[str, dict[str, int]] = Field(default_factory=dict)
