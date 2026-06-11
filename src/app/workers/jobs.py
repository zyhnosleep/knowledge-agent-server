from __future__ import annotations

from app.db.session import SessionLocal
from app.services.pipeline import IngestionPipeline


def run_document_ingestion(document_id: str) -> str:
    db = SessionLocal()
    try:
        pipeline = IngestionPipeline(db)
        run = pipeline.process_document(document_id)
        return run.id
    finally:
        db.close()
