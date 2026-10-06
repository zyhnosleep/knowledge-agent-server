from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import Entity, Project
from app.core.config import Settings


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()



def test_entity_name_is_unique_per_project() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.add_all(
        [
            Entity(project_id="p1", name="SAC-KG", entity_type="concept", aliases=[], summary=""),
            Entity(project_id="p1", name="SAC-KG", entity_type="concept", aliases=[], summary=""),
        ]
    )

    with pytest.raises(IntegrityError):
        db.commit()


def test_sqlite_schema_upgrade_adds_unique_indexes(tmp_path, monkeypatch) -> None:
    from app.db import session as session_module

    db_path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    session_module.Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS uq_entities_project_name"))

    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module.settings, "database_url", f"sqlite:///{db_path}")

    session_module._ensure_sqlite_unique_indexes()

    with engine.connect() as connection:
        entity_indexes = {row[1] for row in connection.execute(text("PRAGMA index_list('entities')"))}
    assert "uq_entities_project_name" in entity_indexes


def test_default_generation_stack_is_single_qwen_profile() -> None:
    settings = Settings(_env_file=None)

    assert settings.ollama_generation_base_url == "http://localhost:11435"
    assert settings.ollama_embedding_base_url == "http://localhost:11435"
    assert settings.ollama_generation_model == "qwen3.5:9b"
    assert settings.ollama_generation_context_length == 32768
    assert settings.ollama_generation_parallelism == 1
    assert settings.ollama_embedding_model == "qwen3-embedding:4b"
    assert settings.ollama_embedding_dimensions == 2560
    assert settings.embedding_provider == "ollama"
    assert settings.embedding_api_model == "Qwen/Qwen3-Embedding-4B"
    assert settings.active_embedding_provider == "ollama"
    assert settings.active_embedding_model == "qwen3-embedding:4b"
    assert settings.active_embedding_dimensions == 2560


def test_remote_embedding_configuration_selects_qwen_api_identity() -> None:
    settings = Settings(
        _env_file=None,
        EMBEDDING_PROVIDER="openai-compatible",
        EMBEDDING_API_MODEL="Qwen/Qwen3-Embedding-4B",
        EMBEDDING_DIMENSIONS=2560,
    )

    assert settings.active_embedding_provider == "openai-compatible"
    assert settings.active_embedding_model == "Qwen/Qwen3-Embedding-4B"
    assert settings.active_embedding_dimensions == 2560


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("OLLAMA_GENERATION_CONTEXT_LENGTH", 0),
        ("OLLAMA_GENERATION_CONTEXT_LENGTH", -1),
        ("OLLAMA_GENERATION_PARALLELISM", 0),
        ("OLLAMA_GENERATION_PARALLELISM", -1),
    ],
)
def test_generation_numeric_settings_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
