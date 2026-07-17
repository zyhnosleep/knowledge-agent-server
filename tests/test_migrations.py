from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

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
