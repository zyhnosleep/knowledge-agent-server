from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.records import Document, DocumentParseVersion


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "queued": {"parsing", "parse_failed"},
    "parsing": {"quality_checking", "parse_failed"},
    "quality_checking": {"repairing", "canonicalizing", "parse_failed"},
    "repairing": {"canonicalizing", "table_repair_failed", "parse_failed"},
    "canonicalizing": {"chunking", "parse_failed"},
    "chunking": {"contextualizing", "parse_failed"},
    "contextualizing": {"embedding", "contextualization_failed"},
    "embedding": {"indexing", "embedding_failed"},
    "indexing": {"ready_to_activate", "embedding_failed"},
    "ready_to_activate": {"active", "activation_failed"},
}

FAILED_STAGE_RETRIES: dict[str, str] = {
    "parse_failed": "parsing",
    "table_repair_failed": "repairing",
    "contextualization_failed": "contextualizing",
    "embedding_failed": "embedding",
    "activation_failed": "ready_to_activate",
}


class ParseVersionService:
    def __init__(self, db: Session | None) -> None:
        self.db = db

    def create(
        self,
        document_id: str,
        version_key: str,
        artifact_dir: str,
        parser_name: str | None = None,
        parser_version: str | None = None,
    ) -> DocumentParseVersion:
        if self.db is None:
            raise RuntimeError("A database session is required to create a parse version.")
        version = DocumentParseVersion(
            document_id=document_id,
            version_key=version_key,
            artifact_dir=artifact_dir,
            parser_name=parser_name,
            parser_version=parser_version,
        )
        self.db.add(version)
        self.db.flush()
        return version

    def transition(
        self, version: DocumentParseVersion, target: str
    ) -> DocumentParseVersion:
        allowed = ALLOWED_TRANSITIONS.get(version.status, set())
        if target not in allowed:
            raise ValueError(
                f"Invalid parse-version transition from {version.status!r} to {target!r}."
            )
        version.status = target
        return version

    def retry_failed_stage(
        self, version: DocumentParseVersion
    ) -> DocumentParseVersion:
        target = FAILED_STAGE_RETRIES.get(version.status)
        if target is None:
            raise ValueError(
                f"Parse version in status {version.status!r} is not retryable."
            )
        version.status = target
        return version

    def activate(
        self, document: Document, version: DocumentParseVersion
    ) -> DocumentParseVersion:
        if self.db is None:
            raise RuntimeError("A database session is required to activate a parse version.")
        if version.document_id != document.id:
            raise ValueError("Parse version and document must belong to the same document.")
        if version.status != "ready_to_activate":
            raise ValueError(
                "Parse version must be in ready_to_activate status before activation."
            )

        with self.db.no_autoflush:
            locked_document = self.db.scalar(
                select(Document).where(Document.id == document.id).with_for_update()
            )
            if locked_document is None:
                raise ValueError(f"Document {document.id!r} does not exist.")
            previous_versions = self.db.scalars(
                select(DocumentParseVersion)
                .where(
                    DocumentParseVersion.document_id == document.id,
                    DocumentParseVersion.status == "active",
                    DocumentParseVersion.id != version.id,
                )
                .with_for_update()
            ).all()

        activated_at = datetime.utcnow()
        previous_pointer = document.active_parse_version
        previous_statuses = [(previous, previous.status) for previous in previous_versions]
        version_status = version.status
        version_activated_at = version.activated_at
        for previous in previous_versions:
            previous.status = "superseded"
        document.active_parse_version = version.version_key
        version.status = "active"
        version.activated_at = activated_at
        try:
            self.db.flush()
        except Exception:
            document.active_parse_version = previous_pointer
            for previous, status in previous_statuses:
                previous.status = status
            version.status = version_status
            version.activated_at = version_activated_at
            raise
        return version
