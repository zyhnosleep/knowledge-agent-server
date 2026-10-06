"""Read-only production checks. A failed check never repairs or relabels data."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.records import Document, DocumentParseVersion


class RuntimeContractError(RuntimeError):
    """Public reason codes deliberately exclude driver messages and credentials."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class EmbeddingIdentity:
    provider: str
    model: str
    revision: str
    processor_hash: str
    dimensions: int

    def __post_init__(self):
        if any(type(value) is not str or not value.strip() for value in
               (self.provider, self.model, self.revision, self.processor_hash)):
            raise RuntimeContractError("embedding_identity_unverified")
        if type(self.dimensions) is not int or self.dimensions <= 0:
            raise RuntimeContractError("embedding_identity_unverified")

    @classmethod
    def from_settings(cls, settings: Settings) -> EmbeddingIdentity:
        return cls(settings.active_embedding_provider, settings.active_embedding_model,
            settings.embedding_revision or "", settings.embedding_processor_hash or "",
            settings.active_embedding_dimensions)

    @classmethod
    def from_mapping(cls, value: Any) -> EmbeddingIdentity:
        if not isinstance(value, Mapping):
            raise RuntimeContractError("embedding_identity_unverified")
        return cls(**{field: value.get(field) for field in cls.__dataclass_fields__})


def check_embedding_contract(db: Session, identity: EmbeddingIdentity,
                             parse_version_map: Mapping[str, str]) -> None:
    """Validate the exact frozen pairs, including versions since superseded."""
    if not parse_version_map:
        return
    versions = {(v.document_id, v.version_key): v for v in db.scalars(
        select(DocumentParseVersion).where(DocumentParseVersion.document_id.in_(parse_version_map)))}
    for document_id, version_key in parse_version_map.items():
        version = versions.get((document_id, version_key))
        if version is None:
            raise RuntimeContractError("parse_version_missing")
        manifest = version.manifest_json or {}
        config = manifest.get("ingestion_config")
        stored = EmbeddingIdentity.from_mapping(config.get("embedding") if isinstance(config, dict) else None)
        if stored != identity:
            raise RuntimeContractError("embedding_identity_mismatch")


def check_pgvector_configuration(db: Session, settings: Settings) -> None:
    if not settings.vector_store_enabled or settings.vector_store_backend != "pgvector":
        raise RuntimeContractError("pgvector_config_invalid")
    dialect = db.get_bind().dialect
    if dialect.name != "postgresql" or dialect.driver != "psycopg":
        raise RuntimeContractError("pgvector_config_invalid")


def check_pgvector_contract(db: Session, settings: Settings) -> dict[str, Any]:
    """Inspect extension, physical dimension, active provenance and index coverage."""
    check_pgvector_configuration(db, settings)
    identity = EmbeddingIdentity.from_settings(settings)
    try:
        extension = db.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar_one_or_none()
        if not extension:
            raise RuntimeContractError("pgvector_extension_missing")
        column_type = db.execute(text("""
            SELECT format_type(atttypid, atttypmod) FROM pg_attribute
            WHERE attrelid=to_regclass('public.document_chunk_pgvector_index')
            AND attname='embedding' AND NOT attisdropped
        """)).scalar_one_or_none()
        if column_type != f"vector({identity.dimensions})":
            raise RuntimeContractError("pgvector_dimension_mismatch")
        active = dict(db.execute(select(Document.id, Document.active_parse_version).where(
            Document.active_parse_version.is_not(None))).all())
        check_embedding_contract(db, identity, active)
        for document_id, version_key in active.items():
            version = db.scalar(select(DocumentParseVersion).where(
                DocumentParseVersion.document_id == document_id, DocumentParseVersion.version_key == version_key))
            if version.status != "active":
                raise RuntimeContractError("active_version_inconsistent")
            coverage = db.execute(text("""
                SELECT count(*) AS expected, count(idx.chunk_id) AS indexed
                FROM document_chunks AS c LEFT JOIN document_chunk_pgvector_index AS idx
                ON idx.chunk_id=c.id AND idx.document_id=c.document_id AND idx.parse_version=c.parse_version
                WHERE c.document_id=:document_id AND c.parse_version=:version_key AND c.chunk_role='child'
            """), {"document_id": document_id, "version_key": version_key}).one()
            if coverage.expected <= 0 or coverage.indexed != coverage.expected:
                raise RuntimeContractError("active_index_incomplete")
    except RuntimeContractError:
        raise
    except Exception:
        raise RuntimeContractError("pgvector_inspection_failed") from None
    return {"status": "ready", "backend": "pgvector", "extension": extension,
        "column_type": column_type, "embedding_identity": asdict(identity), "active_documents": len(active)}
