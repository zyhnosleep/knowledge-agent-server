from __future__ import annotations

from datetime import datetime

from sqlalchemy import inspect as sa_inspect, select
from sqlalchemy.orm import Session, object_session

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
        if object_session(document) is not self.db:
            raise ValueError(
                "Document must be attached to the current session before activation."
            )
        if object_session(version) is not self.db:
            raise ValueError(
                "Parse version must be attached to the current session before activation."
            )

        with self.db.no_autoflush:
            locked_document = self.db.scalar(
                select(Document).where(Document.id == document.id).with_for_update()
            )
            if locked_document is None:
                raise ValueError(f"Document {document.id!r} does not exist.")
            locked_version = self.db.scalar(
                select(DocumentParseVersion)
                .where(DocumentParseVersion.id == version.id)
                .with_for_update()
            )
            if locked_version is None:
                raise ValueError(f"Parse version {version.id!r} does not exist.")
            database_pointer = self.db.execute(
                select(Document.active_parse_version)
                .where(Document.id == locked_document.id)
                .with_for_update()
            ).scalar_one()
            database_status = self.db.execute(
                select(DocumentParseVersion.status)
                .where(DocumentParseVersion.id == locked_version.id)
                .with_for_update()
            ).scalar_one()
            if locked_version.document_id != locked_document.id:
                raise ValueError(
                    "Parse version and document must belong to the same document."
                )
            self._validate_activation_status(locked_version, database_status)
            previous_versions = self.db.scalars(
                select(DocumentParseVersion)
                .where(
                    DocumentParseVersion.document_id == locked_document.id,
                    DocumentParseVersion.status == "active",
                    DocumentParseVersion.id != locked_version.id,
                )
                .with_for_update()
            ).all()

        activated_at = datetime.utcnow()
        previous_statuses = [(previous, "active") for previous in previous_versions]
        version_status = locked_version.status
        version_activated_at = locked_version.activated_at
        for previous in previous_versions:
            previous.status = "superseded"
        locked_document.active_parse_version = locked_version.version_key
        locked_version.status = "active"
        locked_version.activated_at = activated_at
        try:
            self.db.flush()
        except Exception:
            locked_document.active_parse_version = database_pointer
            for previous, status in previous_statuses:
                previous.status = status
            locked_version.status = version_status
            locked_version.activated_at = version_activated_at
            raise
        return locked_version

    @staticmethod
    def _validate_activation_status(
        version: DocumentParseVersion, database_status: str
    ) -> None:
        target = "ready_to_activate"
        if version.status != target:
            raise ValueError(
                "Parse version must be in ready_to_activate status before activation."
            )

        history = sa_inspect(version).attrs.status.history
        if not history.has_changes():
            if database_status != target:
                raise ValueError(
                    f"Parse version database status is {database_status!r}, not {target!r}; "
                    "activation aborted."
                )
            return

        sources = list(history.deleted)
        targets = list(history.added)
        is_valid_local_transition = (
            len(sources) == 1
            and targets == [target]
            and database_status == sources[0]
            and target in ALLOWED_TRANSITIONS.get(sources[0], set())
        )
        if not is_valid_local_transition:
            source = sources[0] if len(sources) == 1 else None
            raise ValueError(
                "Parse version local transition conflicts with database status "
                f"{database_status!r} (local source {source!r}); activation aborted."
            )
