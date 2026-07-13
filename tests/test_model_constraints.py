from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import Entity, Project


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
