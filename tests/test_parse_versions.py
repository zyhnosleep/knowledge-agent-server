from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import (
    Document,
    DocumentChunk,
    DocumentParseVersion,
    DocumentStatus,
    Project,
)
from app.services.parse_versions import ALLOWED_TRANSITIONS, ParseVersionService


@pytest.fixture
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with factory() as session:
        yield session


@pytest.fixture
def document(db: Session) -> Document:
    project = Project(id="project-1", slug="research", name="Research")
    row = Document(
        id="document-1",
        project_id=project.id,
        title="Paper",
        file_name="paper.pdf",
        sha256="abc",
        raw_path="raw/paper.pdf",
    )
    db.add_all([project, row])
    db.flush()
    return row


def test_document_status_keeps_legacy_values_and_adds_ingestion_stages() -> None:
    assert {status.value for status in DocumentStatus} == {
        "pending",
        "processing",
        "ready",
        "failed",
        "parsing",
        "quality_checking",
        "repairing",
        "canonicalizing",
        "chunking",
        "contextualizing",
        "embedding",
        "indexing",
        "parse_failed",
        "table_repair_failed",
        "contextualization_failed",
        "embedding_failed",
        "activation_failed",
    }


def test_create_parse_version_sets_defaults_without_committing(
    db: Session, document: Document
) -> None:
    version = ParseVersionService(db).create(
        document.id,
        "canonical-v1-abcd",
        "runtime/data/parsed/document-1/canonical-v1-abcd",
        parser_name="mineru",
        parser_version="2.5.0",
    )

    assert version.id is not None
    assert version.status == "queued"
    assert version.manifest_json == {}
    assert version.quality_json == {}
    assert version.stage_state == {}
    assert version.parser_name == "mineru"
    assert version.parser_version == "2.5.0"
    assert db.in_transaction()


def test_parse_version_is_unique_per_document(db: Session, document: Document) -> None:
    service = ParseVersionService(db)
    service.create(document.id, "v1", "runtime/data/parsed/document-1/v1")

    with pytest.raises(IntegrityError):
        service.create(document.id, "v1", "runtime/data/parsed/document-1/v1-copy")


def test_transition_map_is_exact() -> None:
    assert ALLOWED_TRANSITIONS == {
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


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (source, target)
        for source, targets in {
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
        }.items()
        for target in targets
    ],
)
def test_transition_accepts_each_declared_edge(source: str, target: str) -> None:
    version = DocumentParseVersion(
        document_id="document-1",
        version_key="v1",
        artifact_dir="parsed/document-1/v1",
        status=source,
    )

    ParseVersionService(None).transition(version, target)

    assert version.status == target


@pytest.mark.parametrize(
    ("source", "target"),
    [("queued", "active"), ("parsing", "embedding"), ("active", "queued")],
)
def test_transition_rejects_undeclared_edges(source: str, target: str) -> None:
    version = DocumentParseVersion(
        document_id="document-1",
        version_key="v1",
        artifact_dir="parsed/document-1/v1",
        status=source,
    )

    with pytest.raises(ValueError, match=f"{source}.*{target}"):
        ParseVersionService(None).transition(version, target)

    assert version.status == source


@pytest.mark.parametrize(
    ("failed_status", "retry_status"),
    [
        ("parse_failed", "parsing"),
        ("table_repair_failed", "repairing"),
        ("contextualization_failed", "contextualizing"),
        ("embedding_failed", "embedding"),
        ("activation_failed", "ready_to_activate"),
    ],
)
def test_retry_failed_stage_uses_controlled_mapping(
    failed_status: str, retry_status: str
) -> None:
    version = DocumentParseVersion(
        document_id="document-1",
        version_key="v1",
        artifact_dir="parsed/document-1/v1",
        status=failed_status,
    )

    ParseVersionService(None).retry_failed_stage(version)

    assert version.status == retry_status


def test_retry_failed_stage_rejects_non_failure_status() -> None:
    version = DocumentParseVersion(
        document_id="document-1",
        version_key="v1",
        artifact_dir="parsed/document-1/v1",
        status="parsing",
    )

    with pytest.raises(ValueError, match="parsing"):
        ParseVersionService(None).retry_failed_stage(version)


def test_activate_updates_pointer_and_version_in_one_uncommitted_transaction(
    db: Session, document: Document
) -> None:
    version = ParseVersionService(db).create(document.id, "v1", "parsed/document-1/v1")
    version.status = "ready_to_activate"

    ParseVersionService(db).activate(document, version)

    assert document.active_parse_version == "v1"
    assert version.status == "active"
    assert version.activated_at is not None
    assert db.in_transaction()


def test_same_session_activation_persists_locked_instances(
    db: Session, document: Document
) -> None:
    version = ParseVersionService(db).create(document.id, "v1", "parsed/document-1/v1")
    version.status = "ready_to_activate"
    db.flush()

    activated = ParseVersionService(db).activate(document, version)
    db.commit()

    with Session(db.get_bind()) as verification:
        stored_document = verification.get(Document, document.id)
        stored_version = verification.get(DocumentParseVersion, version.id)
        assert stored_document is not None
        assert stored_version is not None
        assert stored_document.active_parse_version == "v1"
        assert stored_version.status == "active"
        assert stored_version.activated_at is not None
    assert activated is version


def test_activate_supersedes_previous_active_version(
    db: Session, document: Document
) -> None:
    service = ParseVersionService(db)
    previous = service.create(document.id, "v1", "parsed/document-1/v1")
    previous.status = "active"
    current = service.create(document.id, "v2", "parsed/document-1/v2")
    current.status = "ready_to_activate"
    document.active_parse_version = previous.version_key
    db.flush()

    service.activate(document, current)

    assert previous.status == "superseded"
    assert current.status == "active"
    assert document.active_parse_version == "v2"


def test_activate_rejects_cross_document_without_partial_changes(
    db: Session, document: Document
) -> None:
    other = Document(
        id="document-2",
        project_id=document.project_id,
        title="Other",
        file_name="other.pdf",
        sha256="def",
        raw_path="raw/other.pdf",
    )
    db.add(other)
    db.flush()
    version = ParseVersionService(db).create(other.id, "v1", "parsed/document-2/v1")
    version.status = "ready_to_activate"

    with pytest.raises(ValueError, match="same document"):
        ParseVersionService(db).activate(document, version)

    assert document.active_parse_version is None
    assert version.status == "ready_to_activate"
    assert version.activated_at is None


def test_activate_rejects_non_ready_version_without_partial_changes(
    db: Session, document: Document
) -> None:
    version = ParseVersionService(db).create(document.id, "v1", "parsed/document-1/v1")
    version.status = "embedding"

    with pytest.raises(ValueError, match="ready_to_activate"):
        ParseVersionService(db).activate(document, version)

    assert document.active_parse_version is None
    assert version.status == "embedding"


def _prepare_active_and_ready_versions(
    db: Session, document: Document
) -> tuple[DocumentParseVersion, DocumentParseVersion]:
    service = ParseVersionService(db)
    previous = service.create(document.id, "v1", "parsed/document-1/v1")
    previous.status = "active"
    current = service.create(document.id, "v2", "parsed/document-1/v2")
    current.status = "ready_to_activate"
    document.active_parse_version = "v1"
    db.commit()
    return previous, current


def _assert_activation_was_not_partially_applied(
    db: Session,
    document_id: str,
    previous_id: str,
    current_id: str,
) -> None:
    db.expire_all()
    stored_document = db.get(Document, document_id)
    stored_previous = db.get(DocumentParseVersion, previous_id)
    stored_current = db.get(DocumentParseVersion, current_id)
    assert stored_document is not None
    assert stored_previous is not None
    assert stored_current is not None
    assert stored_document.active_parse_version == "v1"
    assert stored_previous.status == "active"
    assert stored_current.status == "ready_to_activate"
    assert stored_current.activated_at is None


def test_activate_rejects_detached_document_before_database_changes(
    db: Session, document: Document
) -> None:
    previous, current = _prepare_active_and_ready_versions(db, document)
    document_id = document.id
    previous_id = previous.id
    current_id = current.id
    db.expunge(document)

    with pytest.raises(ValueError, match="current session"):
        ParseVersionService(db).activate(document, current)

    _assert_activation_was_not_partially_applied(
        db, document_id, previous_id, current_id
    )


def test_activate_rejects_detached_version_before_database_changes(
    db: Session, document: Document
) -> None:
    previous, current = _prepare_active_and_ready_versions(db, document)
    document_id = document.id
    previous_id = previous.id
    current_id = current.id
    db.expunge(current)

    with pytest.raises(ValueError, match="current session"):
        ParseVersionService(db).activate(document, current)

    _assert_activation_was_not_partially_applied(
        db, document_id, previous_id, current_id
    )


def test_activate_rejects_version_from_another_session_before_database_changes(
    db: Session, document: Document
) -> None:
    previous, current = _prepare_active_and_ready_versions(db, document)
    document_id = document.id
    previous_id = previous.id
    current_id = current.id
    with Session(db.get_bind()) as other_session:
        other_current = other_session.get(DocumentParseVersion, current_id)
        assert other_current is not None

        with pytest.raises(ValueError, match="current session"):
            ParseVersionService(db).activate(document, other_current)

    _assert_activation_was_not_partially_applied(
        db, document_id, previous_id, current_id
    )


def test_activate_restores_in_memory_state_when_flush_fails(
    db: Session, document: Document, monkeypatch
) -> None:
    service = ParseVersionService(db)
    previous = service.create(document.id, "v1", "parsed/document-1/v1")
    previous.status = "active"
    current = service.create(document.id, "v2", "parsed/document-1/v2")
    current.status = "ready_to_activate"
    document.active_parse_version = "v1"
    db.flush()

    def _fail_flush() -> None:
        raise RuntimeError("simulated flush failure")

    monkeypatch.setattr(db, "flush", _fail_flush)

    with pytest.raises(RuntimeError, match="simulated flush failure"):
        service.activate(document, current)

    assert document.active_parse_version == "v1"
    assert previous.status == "active"
    assert current.status == "ready_to_activate"
    assert current.activated_at is None


def test_parent_child_keeps_original_and_embedding_text_separate(
    db: Session, document: Document
) -> None:
    parent = DocumentChunk(
        document_id=document.id,
        parse_version="v1",
        ordinal=0,
        chunk_role="parent",
        block_type="narrative",
        text="Parent source text",
        embedding_text="Parent source text",
    )
    db.add(parent)
    db.flush()
    child = DocumentChunk(
        document_id=document.id,
        parse_version="v1",
        parent_chunk_id=parent.id,
        ordinal=1,
        chunk_role="child",
        block_type="narrative",
        section_path=["Methods"],
        source_block_ids=["block-7"],
        source_spans=[{"page_index": 3, "bbox": [1, 2, 3, 4]}],
        text="Original evidence",
        contextual_prefix="This chunk describes the method used by the paper.",
        embedding_text=(
            "This chunk describes the method used by the paper.\n\nOriginal evidence"
        ),
        token_count=9,
    )
    db.add(child)
    db.flush()

    assert child.parent_chunk_id == parent.id
    assert child.text == "Original evidence"
    assert child.embedding_text != child.text
    assert child.parent is parent
    assert child in parent.children


def test_parent_foreign_key_cascades_at_database_level(
    db: Session, document: Document
) -> None:
    parent = DocumentChunk(
        id="parent",
        document_id=document.id,
        parse_version="v1",
        ordinal=0,
        chunk_role="parent",
        block_type="narrative",
        text="Parent",
        embedding_text="Parent",
    )
    child = DocumentChunk(
        id="child",
        document_id=document.id,
        parse_version="v1",
        parent_chunk_id=parent.id,
        ordinal=1,
        chunk_role="child",
        block_type="narrative",
        text="Child",
        embedding_text="Child",
    )
    db.add_all([parent, child])
    db.commit()

    db.execute(text("DELETE FROM document_chunks WHERE id = :id"), {"id": parent.id})
    db.commit()
    db.expire_all()

    assert db.get(DocumentChunk, "child") is None


def test_chunk_schema_has_version_role_index_and_self_foreign_key() -> None:
    table = DocumentChunk.__table__
    index_columns = {tuple(column.name for column in index.columns) for index in table.indexes}
    parent_fk = next(iter(table.c.parent_chunk_id.foreign_keys))
    previous_fk = next(iter(table.c.previous_chunk_id.foreign_keys))
    next_fk = next(iter(table.c.next_chunk_id.foreign_keys))

    assert ("document_id", "parse_version", "chunk_role") in index_columns
    assert parent_fk.ondelete == "CASCADE"
    assert previous_fk.ondelete == "SET NULL"
    assert next_fk.ondelete == "SET NULL"
    assert inspect(DocumentChunk).relationships["parent"].direction.name == "MANYTOONE"


def test_document_delete_cascades_parse_versions(
    db: Session, document: Document
) -> None:
    version = ParseVersionService(db).create(document.id, "v1", "parsed/document-1/v1")
    version_id = version.id
    db.commit()

    db.delete(document)
    db.commit()

    assert db.scalar(
        select(DocumentParseVersion).where(DocumentParseVersion.id == version_id)
    ) is None


def test_init_db_upgrades_legacy_sqlite_chunk_columns_and_indexes(
    tmp_path, monkeypatch
) -> None:
    from app.db import session as session_module

    database_path = tmp_path / "legacy-init.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    engine = create_engine(database_url, future=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE projects ("
                "id VARCHAR(36) PRIMARY KEY, slug VARCHAR(120) NOT NULL UNIQUE, "
                "name VARCHAR(255) NOT NULL, description TEXT, "
                "created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE documents ("
                "id VARCHAR(36) PRIMARY KEY, project_id VARCHAR(36) NOT NULL, "
                "title VARCHAR(255) NOT NULL, file_name VARCHAR(255) NOT NULL, "
                "sha256 VARCHAR(64) NOT NULL, source_type VARCHAR(40) NOT NULL, "
                "source_uri TEXT, raw_path TEXT NOT NULL, object_key TEXT, raw_text TEXT, "
                "metadata_json JSON NOT NULL, status VARCHAR(40) NOT NULL, "
                "created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, "
                "FOREIGN KEY(project_id) REFERENCES projects(id))"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE document_chunks ("
                "id VARCHAR(36) PRIMARY KEY, document_id VARCHAR(36) NOT NULL, "
                "ordinal INTEGER NOT NULL, heading VARCHAR(255), page_label VARCHAR(32), "
                "text TEXT NOT NULL, token_estimate INTEGER NOT NULL, embedding JSON, "
                "created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, "
                "FOREIGN KEY(document_id) REFERENCES documents(id))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO projects VALUES "
                "('p1', 'research', 'Research', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO documents VALUES "
                "('d1', 'p1', 'Paper', 'paper.pdf', 'abc', 'file', NULL, "
                "'raw/paper.pdf', NULL, NULL, '{}', 'ready', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO document_chunks VALUES "
                "('c1', 'd1', 0, NULL, '1', 'Legacy evidence', 4, NULL, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )

    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module.settings, "database_url", database_url)

    session_module.init_db()

    inspector = inspect(engine)
    document_columns = {column["name"] for column in inspector.get_columns("documents")}
    chunk_columns = {column["name"] for column in inspector.get_columns("document_chunks")}
    chunk_indexes = {
        index["name"] for index in inspector.get_indexes("document_chunks")
    }
    with engine.connect() as connection:
        chunk = connection.execute(
            text(
                "SELECT parse_version, chunk_role, block_type, embedding_text, token_count "
                "FROM document_chunks WHERE id = 'c1'"
            )
        ).mappings().one()
        version = connection.execute(
            text(
                "SELECT version_key, status FROM document_parse_versions "
                "WHERE document_id = 'd1'"
            )
        ).mappings().one()

    assert "active_parse_version" in document_columns
    assert set(CHUNK_SCHEMA_COLUMNS) <= chunk_columns
    assert "ix_document_chunks_document_parse_version_role" in chunk_indexes
    assert dict(chunk) == {
        "parse_version": "legacy",
        "chunk_role": "child",
        "block_type": "narrative",
        "embedding_text": "Legacy evidence",
        "token_count": 4,
    }
    assert dict(version) == {"version_key": "legacy", "status": "quarantined"}


CHUNK_SCHEMA_COLUMNS = {
    "parse_version",
    "parent_chunk_id",
    "chunk_role",
    "block_type",
    "section_path",
    "source_block_ids",
    "source_spans",
    "contextual_prefix",
    "embedding_text",
    "contextualization_model",
    "contextualization_version",
    "contextualization_prompt_version",
    "contextualized_at",
    "parser_name",
    "parser_version",
    "splitter_name",
    "splitter_version",
    "splitting_model",
    "semantic_boundary_score",
    "token_count",
    "previous_chunk_id",
    "next_chunk_id",
}
