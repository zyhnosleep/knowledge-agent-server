from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from app.db.session import Base


ROOT = Path(__file__).resolve().parents[1]


def _config(database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_alembic_upgrade_builds_current_schema_on_sqlite(tmp_path: Path) -> None:
    database_path = tmp_path / "migration.db"
    database_url = f"sqlite:///{database_path.as_posix()}"

    command.upgrade(_config(database_url), "head")

    inspector = inspect(create_engine(database_url))
    table_names = set(inspector.get_table_names())
    expected_tables = set(Base.metadata.tables)
    assert expected_tables <= table_names
    assert "alembic_version" in table_names
    session_columns = {
        column["name"]: column for column in inspector.get_columns("conversation_sessions")
    }
    assert "answer_mode" not in session_columns
    for table_name, table in Base.metadata.tables.items():
        assert {column.name for column in table.columns} == {
            column["name"] for column in inspector.get_columns(table_name)
        }


def test_canonical_migration_quarantines_legacy_chunks_on_sqlite(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy-migration.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    config = _config(database_url)
    command.upgrade(config, "81f6b74cc203")
    engine = create_engine(database_url, future=True)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO projects "
                "(id, slug, name, description, created_at, updated_at) "
                "VALUES ('p1', 'research', 'Research', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO documents "
                "(id, project_id, title, file_name, sha256, source_type, source_uri, "
                "raw_path, object_key, raw_text, metadata_json, status, created_at, updated_at) "
                "VALUES ('d1', 'p1', 'Paper', 'paper.pdf', 'abc', 'file', NULL, "
                "'raw/paper.pdf', NULL, NULL, '{}', 'ready', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO document_chunks "
                "(id, document_id, ordinal, heading, page_label, text, token_estimate, "
                "embedding, created_at, updated_at) "
                "VALUES ('c1', 'd1', 0, NULL, '1', 'Legacy evidence', 3, NULL, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )

    command.upgrade(config, "head")

    with engine.connect() as connection:
        chunk = connection.execute(
            text(
                "SELECT parse_version, chunk_role, block_type, embedding_text "
                "FROM document_chunks WHERE id = 'c1'"
            )
        ).mappings().one()
        version = connection.execute(
            text(
                "SELECT version_key, status FROM document_parse_versions "
                "WHERE document_id = 'd1'"
            )
        ).mappings().one()
        active_parse_version = connection.scalar(
            text("SELECT active_parse_version FROM documents WHERE id = 'd1'")
        )

    assert dict(chunk) == {
        "parse_version": "legacy",
        "chunk_role": "child",
        "block_type": "narrative",
        "embedding_text": "Legacy evidence",
    }
    assert dict(version) == {"version_key": "legacy", "status": "quarantined"}
    assert active_parse_version is None


def test_canonical_migration_downgrades_on_sqlite(tmp_path: Path) -> None:
    database_path = tmp_path / "downgrade.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    config = _config(database_url)
    command.upgrade(config, "head")

    command.downgrade(config, "81f6b74cc203")

    inspector = inspect(create_engine(database_url, future=True))
    assert "document_parse_versions" not in inspector.get_table_names()
    assert "active_parse_version" not in {
        column["name"] for column in inspector.get_columns("documents")
    }
    assert "parse_version" not in {
        column["name"] for column in inspector.get_columns("document_chunks")
    }


def test_alembic_history_has_one_head() -> None:
    script = ScriptDirectory.from_config(_config("sqlite:///:memory:"))

    assert len(script.get_heads()) == 1


def test_postgresql_migration_resizes_pgvector_table_to_2560_dimensions() -> None:
    migration_path = ROOT / "src/app/db/alembic/versions/81f6b74cc203_resize_embeddings_to_2560.py"
    migration_text = migration_path.read_text(encoding="utf-8")

    assert "document_chunk_pgvector_index" in migration_text
    assert "vector(2560)" in migration_text
    assert "vector(4096)" in migration_text
    assert "USING hnsw (embedding vector_cosine_ops)" not in migration_text


def test_canonical_migration_versions_postgresql_pgvector_rows() -> None:
    migration_path = (
        ROOT
        / "src/app/db/alembic/versions/a4e2c7f90120_add_canonical_parse_versions.py"
    )
    migration_text = migration_path.read_text(encoding="utf-8")

    assert 'revision: str = "a4e2c7f90120"' in migration_text
    assert 'down_revision: Union[str, Sequence[str], None] = "81f6b74cc203"' in migration_text
    assert "document_chunk_pgvector_index" in migration_text
    assert 'dialect.name == "postgresql"' in migration_text
    assert "ADD COLUMN parse_version" in migration_text
    assert "SET parse_version = 'legacy'" in migration_text
    assert "ALTER COLUMN parse_version SET NOT NULL" in migration_text
    assert "ix_document_chunk_pgvector_index_document_parse_version" in migration_text


def test_pgvector_parse_version_keeps_legacy_default_during_writer_transition() -> None:
    migration_path = (
        ROOT
        / "src/app/db/alembic/versions/a4e2c7f90120_add_canonical_parse_versions.py"
    )
    migration_text = migration_path.read_text(encoding="utf-8")

    not_null_position = migration_text.index(
        "ALTER COLUMN parse_version SET NOT NULL"
    )
    default_position = migration_text.index(
        "ALTER COLUMN parse_version SET DEFAULT 'legacy'"
    )
    assert default_position > not_null_position
