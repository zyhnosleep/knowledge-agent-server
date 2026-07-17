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


def test_dual_ollama_profile_defaults_are_safe() -> None:
    settings = Settings(_env_file=None)

    assert settings.ollama_fast_base_url == "http://localhost:11435"
    assert settings.ollama_deep_base_url == "http://localhost:11436"
    assert settings.ollama_embedding_base_url == "http://localhost:11435"
    assert settings.ollama_fast_model == "qwen3:14b"
    assert settings.ollama_deep_model == "qwen3.6:27b"
    assert settings.ollama_fast_context_length == 16384
    assert settings.ollama_deep_context_length == 32768


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("OLLAMA_FAST_CONTEXT_LENGTH", 0),
        ("OLLAMA_DEEP_CONTEXT_LENGTH", -1),
        ("OLLAMA_FAST_PARALLELISM", 0),
        ("OLLAMA_DEEP_PARALLELISM", -1),
    ],
)
def test_dual_ollama_numeric_settings_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
