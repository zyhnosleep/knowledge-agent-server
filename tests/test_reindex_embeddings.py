from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.session import Base
from app.models.records import Document, DocumentChunk, Project
from scripts.reindex_embeddings import reindex_documents


class FakeOllama:
    def __init__(self, dimensions: int, *, fail_text: str | None = None) -> None:
        self.dimensions = dimensions
        self.fail_text = fail_text

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail_text and self.fail_text in texts:
            raise RuntimeError("embedding failed")
        return [[float(index + 1)] * self.dimensions for index, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents: list[str] = []

    def replace_document_chunks(self, document_id, vectors) -> None:
        assert all(len(vector.embedding) == 3 for vector in vectors)
        self.documents.append(document_id)


def make_db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, future=True)()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.add_all(
        [
            Document(id="d1", project_id="p1", title="One", file_name="one.pdf", sha256="1", raw_path="raw/one.pdf"),
            Document(id="d2", project_id="p1", title="Two", file_name="two.pdf", sha256="2", raw_path="raw/two.pdf"),
        ]
    )
    db.add_all(
        [
            DocumentChunk(id="c1", document_id="d1", ordinal=0, text="alpha", embedding=[9.0]),
            DocumentChunk(id="c2", document_id="d1", ordinal=1, text="beta", embedding=[9.0]),
            DocumentChunk(id="c3", document_id="d2", ordinal=0, text="fail", embedding=[9.0]),
        ]
    )
    db.commit()
    return db


def test_reindex_refuses_without_maintenance_confirmation() -> None:
    db = make_db()

    try:
        reindex_documents(
            db,
            FakeOllama(3),
            vector_store=FakeVectorStore(),
            dimensions=3,
            maintenance_confirmed=False,
        )
    except ValueError as exc:
        assert "maintenance" in str(exc).lower()
    else:
        raise AssertionError("reindex must require explicit maintenance confirmation")


def test_reindex_updates_each_document_and_reports_failures() -> None:
    db = make_db()
    store = FakeVectorStore()

    report = reindex_documents(
        db,
        FakeOllama(3, fail_text="fail"),
        vector_store=store,
        dimensions=3,
        maintenance_confirmed=True,
        batch_size=2,
    )

    assert report["documents_succeeded"] == 1
    assert report["documents_failed"] == ["d2"]
    assert report["chunks_updated"] == 2
    assert store.documents == ["d1"]
    assert len(db.get(DocumentChunk, "c1").embedding) == 3
    assert db.get(DocumentChunk, "c3").embedding == [9.0]
