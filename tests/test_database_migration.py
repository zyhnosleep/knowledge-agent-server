from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from app.db.session import Base
from app.models.records import AgentTraceRun, ConversationSession, Document, DocumentChunk, Project
from scripts.migrate_sqlite_to_postgres import MigrationError, migrate_database


def _url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def _prepare_source(path: Path) -> None:
    engine = create_engine(_url(path), future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(Project(id="p1", slug="research", name="Research"))
        db.add(
            Document(
                id="d1",
                project_id="p1",
                title="Paper",
                file_name="paper.pdf",
                sha256="abc",
                raw_path="data/raw/paper.pdf",
                metadata_json={"source_slug": "sources/paper", "nested": {"value": 3}},
                status="ready",
            )
        )
        db.add(
            DocumentChunk(
                id="c1",
                document_id="d1",
                ordinal=0,
                text="Evidence text",
                page_label="1",
                embedding=[1.0, 0.0],
            )
        )
        db.commit()


def _prepare_target(path: Path) -> None:
    engine = create_engine(_url(path), future=True)
    Base.metadata.create_all(engine)


def test_migration_preserves_ids_json_and_is_repeatable(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _prepare_source(source)
    _prepare_target(target)

    first = migrate_database(_url(source), _url(target), reindex_vectors=False)
    second = migrate_database(_url(source), _url(target), reindex_vectors=False)

    assert first["projects"] == 1
    assert first["documents"] == 1
    assert first["document_chunks"] == 1
    assert second == first

    with Session(create_engine(_url(target), future=True)) as db:
        document = db.get(Document, "d1")
        chunk = db.get(DocumentChunk, "c1")
        assert document is not None
        assert document.metadata_json == {
            "source_slug": "sources/paper",
            "nested": {"value": 3},
        }
        assert chunk is not None
        assert chunk.embedding == [1.0, 0.0]


def test_migration_dry_run_does_not_write_target(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _prepare_source(source)
    _prepare_target(target)

    counts = migrate_database(
        _url(source), _url(target), dry_run=True, reindex_vectors=False
    )

    assert counts["projects"] == 1
    with Session(create_engine(_url(target), future=True)) as db:
        assert db.scalar(select(func.count()).select_from(Project)) == 0


def test_migration_treats_tables_missing_from_legacy_source_as_empty(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-source.db"
    target = tmp_path / "target.db"
    _prepare_source(source)
    with create_engine(_url(source), future=True).begin() as connection:
        connection.execute(text("DROP TABLE auth_sessions"))
        connection.execute(text("DROP TABLE users"))
    _prepare_target(target)

    counts = migrate_database(_url(source), _url(target), reindex_vectors=False)

    assert counts["users"] == 0
    assert counts["auth_sessions"] == 0
    assert counts["projects"] == 1


def test_migration_fills_new_nullable_columns_missing_from_legacy_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-columns.db"
    target = tmp_path / "target.db"
    _prepare_source(source)
    with Session(create_engine(_url(source), future=True)) as db:
        db.add(
            ConversationSession(
                id="legacy-session",
                project_slug="research",
                expires_at=datetime.utcnow(),
            )
        )
        db.add(
            AgentTraceRun(
                id="legacy-trace",
                request_id="legacy-request",
                session_id="legacy-session",
                project_slug="research",
                query="legacy",
                final_answer="legacy",
                status="completed",
            )
        )
        db.commit()
    with create_engine(_url(source), future=True).begin() as connection:
        connection.execute(text("DROP INDEX ix_conversation_sessions_owner_user_id"))
        connection.execute(text("DROP INDEX ix_agent_trace_runs_owner_user_id"))
        connection.execute(text("ALTER TABLE conversation_sessions DROP COLUMN owner_user_id"))
        connection.execute(text("ALTER TABLE agent_trace_runs DROP COLUMN owner_user_id"))
    _prepare_target(target)

    first = migrate_database(_url(source), _url(target), reindex_vectors=False)
    second = migrate_database(_url(source), _url(target), reindex_vectors=False)

    assert first == second


def test_migration_rejects_extra_target_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _prepare_source(source)
    _prepare_target(target)
    target_engine = create_engine(_url(target), future=True)
    with Session(target_engine) as db:
        db.add(Project(id="extra", slug="extra", name="Extra"))
        db.commit()

    with pytest.raises(MigrationError, match="unexpected rows"):
        migrate_database(_url(source), _url(target), reindex_vectors=False)
