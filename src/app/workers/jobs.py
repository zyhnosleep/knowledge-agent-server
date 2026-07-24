from __future__ import annotations

from app.db.session import SessionLocal
from app.services.ingestion_stages import IngestionStageRunner
from app.services.pipeline import IngestionPipeline
from app.services.queue import JobDispatcher


def run_ingestion_stage(document_id: str, version_key: str, stage: str) -> str:
    db = SessionLocal()
    try:
        pipeline = IngestionPipeline(db)
        runner = IngestionStageRunner(
            db,
            handlers=pipeline.ingestion_stage_handlers(),
            dispatcher=JobDispatcher(),
        )
        version = runner.run_stage(document_id, version_key, stage)
        return version.id
    finally:
        db.close()


def run_document_ingestion(document_id: str) -> str:
    db = SessionLocal()
    try:
        pipeline = IngestionPipeline(db)
        run = pipeline.process_document(document_id)
        return run.id
    finally:
        db.close()
