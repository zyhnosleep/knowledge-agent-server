from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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


class CitationLocationRead(BaseModel):
    document_id: str
    chunk_id: str
    parse_version: str
    source_type: str
    source_url: str
    source_spans: list[dict[str, Any]] = Field(default_factory=list)


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
