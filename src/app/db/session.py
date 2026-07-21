from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from datetime import datetime

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import get_settings

settings = get_settings()

connect_args = (
    {"check_same_thread": False, "timeout": 30}
    if settings.database_url.startswith("sqlite")
    else {}
)
engine = create_engine(settings.database_url, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app.models import records  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_columns()
    _ensure_sqlite_unique_indexes()


def _ensure_sqlite_columns() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    columns = (
        ("conversation_sessions", "document_id", "TEXT"),
        ("conversation_turns", "citations", "TEXT"),
        ("documents", "active_parse_version", "VARCHAR(128)"),
        ("document_chunks", "parse_version", "VARCHAR(128)"),
        ("document_chunks", "parent_chunk_id", "VARCHAR(36)"),
        ("document_chunks", "chunk_role", "VARCHAR(20)"),
        ("document_chunks", "block_type", "VARCHAR(40)"),
        ("document_chunks", "section_path", "JSON"),
        ("document_chunks", "source_block_ids", "JSON"),
        ("document_chunks", "source_spans", "JSON"),
        ("document_chunks", "contextual_prefix", "TEXT"),
        ("document_chunks", "embedding_text", "TEXT"),
        ("document_chunks", "contextualization_model", "VARCHAR(120)"),
        ("document_chunks", "contextualization_version", "VARCHAR(120)"),
        (
            "document_chunks",
            "contextualization_prompt_version",
            "VARCHAR(120)",
        ),
        ("document_chunks", "contextualized_at", "DATETIME"),
        ("document_chunks", "parser_name", "VARCHAR(120)"),
        ("document_chunks", "parser_version", "VARCHAR(120)"),
        ("document_chunks", "splitter_name", "VARCHAR(120)"),
        ("document_chunks", "splitter_version", "VARCHAR(120)"),
        ("document_chunks", "splitting_model", "VARCHAR(120)"),
        ("document_chunks", "semantic_boundary_score", "FLOAT"),
        ("document_chunks", "token_count", "INTEGER"),
        ("document_chunks", "previous_chunk_id", "VARCHAR(36)"),
        ("document_chunks", "next_chunk_id", "VARCHAR(36)"),
    )
    with engine.begin() as connection:
        table_names = set(inspect(connection).get_table_names())
        for table_name, column_name, column_type in columns:
            if table_name not in table_names:
                continue
            try:
                connection.execute(
                    text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                )
            except OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
    _backfill_sqlite_parse_versions()
    _ensure_sqlite_indexes()


def _backfill_sqlite_parse_versions() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    empty_json = json.dumps([])
    with engine.begin() as connection:
        table_names = set(inspect(connection).get_table_names())
        if not {"document_chunks", "document_parse_versions"} <= table_names:
            return
        connection.execute(
            text(
                "UPDATE document_chunks SET "
                "parse_version = COALESCE(parse_version, 'legacy'), "
                "chunk_role = COALESCE(chunk_role, 'child'), "
                "block_type = COALESCE(block_type, 'narrative'), "
                "section_path = COALESCE(section_path, :empty_json), "
                "source_block_ids = COALESCE(source_block_ids, :empty_json), "
                "source_spans = COALESCE(source_spans, :empty_json), "
                "embedding_text = COALESCE(embedding_text, text), "
                "token_count = COALESCE(token_count, token_estimate, 0)"
            ),
            {"empty_json": empty_json},
        )
        document_ids = connection.execute(
            text(
                "SELECT DISTINCT document_id FROM document_chunks "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM document_parse_versions "
                "WHERE document_parse_versions.document_id = document_chunks.document_id "
                "AND document_parse_versions.version_key = 'legacy')"
            )
        ).scalars()
        now = datetime.utcnow()
        for document_id in document_ids:
            connection.execute(
                text(
                    "INSERT INTO document_parse_versions "
                    "(id, document_id, version_key, status, artifact_dir, parser_name, "
                    "parser_version, manifest_json, quality_json, stage_state, activated_at, "
                    "created_at, updated_at) VALUES "
                    "(:id, :document_id, 'legacy', 'quarantined', 'legacy', NULL, NULL, "
                    ":empty_object, :empty_object, :stage_state, NULL, :now, :now)"
                ),
                {
                    "id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "empty_object": json.dumps({}),
                    "stage_state": json.dumps({"migration": "legacy"}),
                    "now": now,
                },
            )


def _ensure_sqlite_indexes() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    statements = (
        (
            "conversation_sessions",
            "CREATE INDEX IF NOT EXISTS ix_conversation_sessions_document_id "
            "ON conversation_sessions (document_id)",
        ),
        (
            "document_chunks",
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_document_parse_version_role "
            "ON document_chunks (document_id, parse_version, chunk_role)",
        ),
    )
    with engine.begin() as connection:
        table_names = set(inspect(connection).get_table_names())
        for table_name, statement in statements:
            if table_name not in table_names:
                continue
            connection.execute(text(statement))


def _ensure_sqlite_unique_indexes() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    statements = (
        (
            "uq_entities_project_name",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_entities_project_name ON entities (project_id, name)",
        ),
    )
    with engine.begin() as connection:
        for index_name, statement in statements:
            try:
                connection.execute(text(statement))
            except IntegrityError as exc:
                raise RuntimeError(f"Cannot create unique index {index_name}; duplicate rows already exist.") from exc
            except OperationalError as exc:
                raise RuntimeError(f"Cannot create unique index {index_name}: {exc}") from exc
