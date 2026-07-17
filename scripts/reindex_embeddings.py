from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.records import Document, DocumentChunk, Project
from app.services.ai import OllamaClient
from app.services.vector_store import ChunkVector, get_vector_store


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reindex_documents(
    db: Session,
    ollama: OllamaClient,
    *,
    vector_store: Any | None = None,
    dimensions: int,
    maintenance_confirmed: bool,
    batch_size: int = 16,
    project_slug: str | None = None,
    resume_after_document: str | None = None,
) -> dict[str, Any]:
    if not maintenance_confirmed:
        raise ValueError("Embedding reindex requires explicit maintenance confirmation")
    if dimensions <= 0 or batch_size <= 0:
        raise ValueError("dimensions and batch_size must be positive")

    statement = select(Document).join(Project, Project.id == Document.project_id)
    if project_slug:
        statement = statement.where(Project.slug == project_slug)
    if resume_after_document:
        statement = statement.where(Document.id > resume_after_document)
    documents = list(db.scalars(statement.order_by(Document.id)))
    store = vector_store or get_vector_store(db)
    report: dict[str, Any] = {
        "started_at": _utc_now(),
        "model": get_settings().ollama_embedding_model,
        "dimensions": dimensions,
        "documents_total": len(documents),
        "documents_succeeded": 0,
        "documents_failed": [],
        "chunks_updated": 0,
    }

    for document in documents:
        try:
            chunks = list(
                db.scalars(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == document.id)
                    .order_by(DocumentChunk.ordinal)
                )
            )
            embeddings: list[list[float]] = []
            for offset in range(0, len(chunks), batch_size):
                texts = [chunk.text for chunk in chunks[offset : offset + batch_size]]
                embeddings.extend(ollama.embed(texts))
            if len(embeddings) != len(chunks):
                raise ValueError(
                    f"Embedding count mismatch for {document.id}: "
                    f"expected {len(chunks)}, got {len(embeddings)}"
                )
            if any(len(embedding) != dimensions for embedding in embeddings):
                raise ValueError(f"Embedding dimension mismatch for {document.id}")
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk.embedding = embedding
            db.flush()
            store.replace_document_chunks(
                document.id,
                [
                    ChunkVector(
                        chunk_id=chunk.id,
                        document_id=document.id,
                        embedding=embedding,
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )
            db.commit()
            report["documents_succeeded"] += 1
            report["chunks_updated"] += len(chunks)
        except Exception:  # noqa: BLE001 - failures are recorded per document
            db.rollback()
            report["documents_failed"].append(document.id)

    report["finished_at"] = _utc_now()
    return report


def verify_embeddings(db: Session, *, dimensions: int) -> dict[str, Any]:
    chunks_total = int(db.scalar(select(func.count()).select_from(DocumentChunk)) or 0)
    chunks_valid = sum(
        1
        for embedding in db.scalars(select(DocumentChunk.embedding))
        if isinstance(embedding, list) and len(embedding) == dimensions
    )
    pgvector_rows: int | None = None
    if db.get_bind().dialect.name == "postgresql":
        pgvector_rows = int(
            db.execute(text("SELECT COUNT(*) FROM document_chunk_pgvector_index")).scalar_one()
        )
    return {
        "dimensions": dimensions,
        "chunks_total": chunks_total,
        "chunks_valid": chunks_valid,
        "pgvector_rows": pgvector_rows,
        "valid": chunks_total == chunks_valid
        and (pgvector_rows is None or pgvector_rows == chunks_total),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild document embeddings")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--project")
    scope.add_argument("--all-projects", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume-after-document")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--maintenance-confirmed", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.maintenance_confirmed:
        _parser().error("--maintenance-confirmed is required")
    settings = get_settings()
    with SessionLocal() as db:
        if args.verify_only:
            report = verify_embeddings(
                db, dimensions=settings.ollama_embedding_dimensions
            )
        else:
            report = reindex_documents(
                db,
                OllamaClient(),
                dimensions=settings.ollama_embedding_dimensions,
                maintenance_confirmed=True,
                batch_size=args.batch_size,
                project_slug=args.project,
                resume_after_document=args.resume_after_document,
            )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report.get("documents_failed") or report.get("valid") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
