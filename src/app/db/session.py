from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import get_settings

settings = get_settings()

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
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
    _ensure_sqlite_unique_indexes()


def _ensure_sqlite_unique_indexes() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    statements = (
        (
            "uq_wiki_pages_project_slug",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_wiki_pages_project_slug ON wiki_pages (project_id, slug)",
        ),
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
