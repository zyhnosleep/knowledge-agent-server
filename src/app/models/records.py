from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DocumentStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class PageKind(str, Enum):
    source_summary = "source_summary"
    entity = "entity"
    concept = "concept"
    synthesis = "synthesis"
    comparison = "comparison"
    query_answer = "query_answer"


class ReviewSeverity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ReviewStatus(str, Enum):
    pending = "pending"
    resolved = "resolved"
    ignored = "ignored"


class RunType(str, Enum):
    ingest = "ingest"
    query = "query"
    rebuild = "rebuild"
    verify = "verify"


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    documents: Mapped[list["Document"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    wiki_pages: Mapped[list["WikiPage"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    entities: Mapped[list["Entity"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    file_name: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    source_type: Mapped[str] = mapped_column(String(40), default="file")
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_path: Mapped[str] = mapped_column(Text)
    object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(40), default=DocumentStatus.pending.value)

    project: Mapped["Project"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    runs: Mapped[list["PipelineRun"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base, TimestampMixin):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class WikiPage(Base, TimestampMixin):
    __tablename__ = "wiki_pages"
    __table_args__ = (UniqueConstraint("project_id", "slug", name="uq_wiki_pages_project_slug"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    slug: Mapped[str] = mapped_column(String(255), index=True)
    title: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(40), default=PageKind.source_summary.value)
    markdown_path: Mapped[str] = mapped_column(Text)
    markdown_content: Mapped[str] = mapped_column(Text)
    source_document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    project: Mapped["Project"] = relationship(back_populates="wiki_pages")


class Entity(Base, TimestampMixin):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_entities_project_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    entity_type: Mapped[str] = mapped_column(String(80), default="concept")
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    canonical_page_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    project: Mapped["Project"] = relationship(back_populates="entities")


class Claim(Base, TimestampMixin):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    subject: Mapped[str] = mapped_column(String(255))
    predicate: Mapped[str] = mapped_column(String(255))
    object_text: Mapped[str] = mapped_column(Text)
    evidence_chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    verification_status: Mapped[str] = mapped_column(String(40), default="unverified")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ReviewItem(Base, TimestampMixin):
    __tablename__ = "review_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True)
    wiki_page_id: Mapped[str | None] = mapped_column(ForeignKey("wiki_pages.id"), nullable=True)
    claim_id: Mapped[str | None] = mapped_column(ForeignKey("claims.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(255))
    detail: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(20), default=ReviewSeverity.medium.value)
    status: Mapped[str] = mapped_column(String(20), default=ReviewStatus.pending.value)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


class PipelineRun(Base, TimestampMixin):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True)
    run_type: Mapped[str] = mapped_column(String(40), default=RunType.ingest.value)
    status: Mapped[str] = mapped_column(String(20), default=RunStatus.queued.value)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider_report: Mapped[dict] = mapped_column(JSON, default=dict)

    document: Mapped["Document | None"] = relationship(back_populates="runs")


class QuestionAnswer(Base, TimestampMixin):
    __tablename__ = "question_answers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer_markdown: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[dict]] = mapped_column(JSON, default=list)
    risk_level: Mapped[str] = mapped_column(String(20), default="normal")
    verification_status: Mapped[str] = mapped_column(String(40), default="local-only")
