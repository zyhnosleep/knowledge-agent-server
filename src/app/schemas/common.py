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
