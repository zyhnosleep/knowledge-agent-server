from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
import pytest

from app.db.session import Base
from app.models.records import Document, DocumentChunk, Project
from app.schemas.common import Citation
from app.services.search import PaperMatch, QueryService, RetrievedContext
from app.services.vector_store import ChunkVector, SQLiteVecStore, VectorHit


class EmbedOnlyOllama:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vector for _ in texts]


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def test_search_source_chunks_prefers_sqlite_vec_hits(monkeypatch) -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.add(Document(id="d1", project_id="p1", title="Paper", file_name="paper.pdf", sha256="abc", raw_path="raw/paper.pdf", status="ready"))
    db.add_all(
        [
            DocumentChunk(
                id="lexical",
                document_id="d1",
                ordinal=0,
                text="unmatched query lexical distractor",
                page_label="1",
                embedding=None,
            ),
            DocumentChunk(
                id="semantic",
                document_id="d1",
                ordinal=1,
                text="The architecture uses a retriever, verifier, and pruner.",
                page_label="2",
                embedding=None,
            ),
        ]
    )
    db.commit()

    class FakeVectorStore:
        def __init__(self, db: Session) -> None:
            self.db = db

        def search(self, embedding: list[float], *, limit: int, document_ids: list[str] | None = None) -> list[VectorHit]:
            return [VectorHit(chunk_id="semantic", distance=0.0)]

    import app.services.search as search_module

    monkeypatch.setattr(search_module, "SQLiteVecStore", FakeVectorStore)
    service = QueryService(db)
    service.ollama = EmbedOnlyOllama([1.0, 0.0])

    contexts = service._search_source_chunks("unmatched query", "p1", ["d1"], limit=1)

    assert [context.citation.chunk_id for context in contexts] == ["semantic"]


def test_sqlite_vec_failure_does_not_roll_back_written_chunks(monkeypatch) -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.add(Document(id="d1", project_id="p1", title="Paper", file_name="paper.pdf", sha256="abc", raw_path="raw/paper.pdf", status="ready"))
    db.add(
        DocumentChunk(
            id="c1",
            document_id="d1",
            ordinal=0,
            text="Chunk with JSON embedding.",
            page_label="1",
            embedding=[1.0, 0.0],
        )
    )
    db.flush()

    store = SQLiteVecStore(db)
    monkeypatch.setattr(store, "available", lambda: True)

    store.replace_document_chunks("d1", [ChunkVector(chunk_id="c1", document_id="d1", embedding=[1.0, 0.0])])
    db.commit()

    chunk = db.get(DocumentChunk, "c1")
    assert chunk is not None
    assert chunk.embedding == [1.0, 0.0]


def test_sqlite_vec_replace_removes_stale_chunk_vector_rows(monkeypatch) -> None:
    db = make_session()
    store = SQLiteVecStore(db)
    monkeypatch.setattr(store, "available", lambda: True)
    monkeypatch.setattr(
        store,
        "_ensure_vector_table",
        lambda dimensions: db.execute(text(f"CREATE TABLE IF NOT EXISTS document_chunk_vec_{dimensions} (embedding BLOB NOT NULL)")),
    )
    monkeypatch.setattr(store, "_serialize", lambda embedding: bytes(str(embedding), encoding="utf-8"))

    store.replace_document_chunks("d1", [ChunkVector(chunk_id="shared", document_id="d1", embedding=[1.0, 0.0])])
    store.replace_document_chunks("d2", [ChunkVector(chunk_id="shared", document_id="d2", embedding=[1.0, 0.0, 0.0])])

    mapping_rows = db.execute(text("SELECT chunk_id, document_id, dimensions FROM document_chunk_vector_index")).all()
    old_vector_count = db.execute(text("SELECT COUNT(*) FROM document_chunk_vec_2")).scalar_one()
    new_vector_count = db.execute(text("SELECT COUNT(*) FROM document_chunk_vec_3")).scalar_one()

    assert [(row.chunk_id, row.document_id, row.dimensions) for row in mapping_rows] == [("shared", "d2", 3)]
    assert old_vector_count == 0
    assert new_vector_count == 1


def test_sqlite_vec_normalizes_embeddings_before_serialization_and_query(monkeypatch) -> None:
    db = make_session()
    store = SQLiteVecStore(db)
    monkeypatch.setattr(store, "available", lambda: True)
    monkeypatch.setattr(
        store,
        "_ensure_vector_table",
        lambda dimensions: db.execute(text(f"CREATE TABLE IF NOT EXISTS document_chunk_vec_{dimensions} (embedding BLOB NOT NULL)")),
    )
    serialized_embeddings: list[list[float]] = []

    def fake_serialize(embedding: list[float]) -> bytes:
        serialized_embeddings.append(list(embedding))
        return bytes(str(embedding), encoding="utf-8")

    monkeypatch.setattr(store, "_serialize", fake_serialize)

    store.replace_document_chunks("d1", [ChunkVector(chunk_id="c1", document_id="d1", embedding=[3.0, 4.0])])

    assert serialized_embeddings[0] == pytest.approx([0.6, 0.8])

    queried_embeddings: list[list[float]] = []

    def fake_search_vector_rows(dimensions: int, embedding: list[float], limit: int) -> list[SimpleNamespace]:
        queried_embeddings.append(list(embedding))
        return []

    monkeypatch.setattr(store, "_search_vector_rows", fake_search_vector_rows)

    assert store.search([10.0, 0.0], limit=1) == []
    assert queried_embeddings == [[1.0, 0.0]]


def test_sqlite_vec_scoped_search_expands_past_global_candidate_window(monkeypatch) -> None:
    db = make_session()
    db.add_all(
        [
            Project(id="p1", slug="demo", name="Demo"),
            Project(id="p2", slug="other", name="Other"),
            Document(id="scoped", project_id="p1", title="Scoped", file_name="scoped.pdf", sha256="abc", raw_path="raw/scoped.pdf", status="ready"),
            Document(id="distractor", project_id="p2", title="Distractor", file_name="distractor.pdf", sha256="def", raw_path="raw/distractor.pdf", status="ready"),
            *[
                DocumentChunk(
                    id=f"distractor-{index}",
                    document_id="distractor",
                    ordinal=index,
                    text=f"Distractor chunk {index}.",
                    page_label="1",
                    embedding=[1.0, 0.0],
                )
                for index in range(60)
            ],
            DocumentChunk(
                id="scoped-hit",
                document_id="scoped",
                ordinal=0,
                text="Scoped semantic evidence.",
                page_label="1",
                embedding=[0.9, 0.1],
            ),
        ]
    )
    db.commit()

    store = SQLiteVecStore(db)
    monkeypatch.setattr(store, "available", lambda: True)
    monkeypatch.setattr(store, "_ensure_vector_table", lambda dimensions: None)
    store._ensure_meta_table()
    for index in range(60):
        store._insert_mapping(f"distractor-{index}", "distractor", 2)
    store._insert_mapping("scoped-hit", "scoped", 2)
    db.commit()

    requested_limits: list[int] = []

    def fake_search_vector_rows(dimensions: int, embedding: list[float], limit: int) -> list[SimpleNamespace]:
        requested_limits.append(limit)
        return [SimpleNamespace(rowid=row_id, distance=float(row_id)) for row_id in range(1, limit + 1)]

    monkeypatch.setattr(store, "_search_vector_rows", fake_search_vector_rows)

    hits = store.search([1.0, 0.0], limit=1, document_ids=["scoped"])

    assert [hit.chunk_id for hit in hits] == ["scoped-hit"]
    assert requested_limits == [50, 61]


def test_sqlite_vec_store_loads_real_extension_when_installed() -> None:
    pytest.importorskip("sqlite_vec")
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.add(Document(id="d1", project_id="p1", title="Paper", file_name="paper.pdf", sha256="abc", raw_path="raw/paper.pdf", status="ready"))
    db.add_all(
        [
            DocumentChunk(
                id="near",
                document_id="d1",
                ordinal=0,
                text="Near vector evidence.",
                page_label="1",
                embedding=[1.0, 0.0],
            ),
            DocumentChunk(
                id="far",
                document_id="d1",
                ordinal=1,
                text="Far vector evidence.",
                page_label="2",
                embedding=[0.0, 1.0],
            ),
        ]
    )
    db.commit()

    store = SQLiteVecStore(db)
    assert store.available()

    store.replace_document_chunks(
        "d1",
        [
            ChunkVector(chunk_id="near", document_id="d1", embedding=[1.0, 0.0]),
            ChunkVector(chunk_id="far", document_id="d1", embedding=[0.0, 1.0]),
        ],
    )
    db.commit()

    assert [hit.chunk_id for hit in store.search([1.0, 0.0], limit=2)] == ["near", "far"]


def test_search_source_chunks_falls_back_to_json_embeddings_when_sqlite_vec_unavailable(monkeypatch) -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.add(Document(id="d1", project_id="p1", title="Paper", file_name="paper.pdf", sha256="abc", raw_path="raw/paper.pdf", status="ready"))
    db.add_all(
        [
            DocumentChunk(
                id="json-best",
                document_id="d1",
                ordinal=0,
                text="Embedding-backed source evidence.",
                page_label="1",
                embedding=[1.0, 0.0],
            ),
            DocumentChunk(
                id="json-worse",
                document_id="d1",
                ordinal=1,
                text="Less similar source evidence.",
                page_label="2",
                embedding=[0.0, 1.0],
            ),
        ]
    )
    db.commit()

    class UnavailableVectorStore:
        def __init__(self, db: Session) -> None:
            self.db = db

        def search(self, embedding: list[float], *, limit: int, document_ids: list[str] | None = None) -> list[VectorHit]:
            return []

    import app.services.search as search_module

    monkeypatch.setattr(search_module, "SQLiteVecStore", UnavailableVectorStore)
    service = QueryService(db)
    service.ollama = EmbedOnlyOllama([1.0, 0.0])

    contexts = service._search_source_chunks("source evidence", "p1", ["d1"], limit=1)

    assert [context.citation.chunk_id for context in contexts] == ["json-best"]


def test_search_source_chunks_ignores_vector_hits_outside_scope(monkeypatch) -> None:
    db = make_session()
    db.add_all(
        [
            Project(id="p1", slug="demo", name="Demo"),
            Project(id="p2", slug="other", name="Other"),
            Document(id="d1", project_id="p1", title="Paper", file_name="paper.pdf", sha256="abc", raw_path="raw/paper.pdf", status="ready"),
            Document(id="d2", project_id="p2", title="Other", file_name="other.pdf", sha256="def", raw_path="raw/other.pdf", status="ready"),
            DocumentChunk(
                id="in-scope",
                document_id="d1",
                ordinal=0,
                text="Scoped JSON embedding evidence.",
                page_label="1",
                embedding=[1.0, 0.0],
            ),
            DocumentChunk(
                id="outside-scope",
                document_id="d2",
                ordinal=0,
                text="Out of scope semantic vector evidence.",
                page_label="1",
                embedding=[1.0, 0.0],
            ),
        ]
    )
    db.commit()

    class LeakyVectorStore:
        def __init__(self, db: Session) -> None:
            self.db = db

        def search(self, embedding: list[float], *, limit: int, document_ids: list[str] | None = None) -> list[VectorHit]:
            return [VectorHit(chunk_id="outside-scope", distance=0.0)]

    import app.services.search as search_module

    monkeypatch.setattr(search_module, "SQLiteVecStore", LeakyVectorStore)
    service = QueryService(db)
    service.ollama = EmbedOnlyOllama([1.0, 0.0])

    contexts = service._search_source_chunks("scoped evidence", "p1", ["d1"], limit=1)

    assert [context.citation.chunk_id for context in contexts] == ["in-scope"]


def test_search_source_chunks_passes_document_scope_to_vector_store(monkeypatch) -> None:
    db = make_session()
    db.add_all(
        [
            Project(id="p1", slug="demo", name="Demo"),
            Document(id="d1", project_id="p1", title="Scoped", file_name="scoped.pdf", sha256="abc", raw_path="raw/scoped.pdf", status="ready"),
            Document(id="d2", project_id="p1", title="Distractor", file_name="distractor.pdf", sha256="def", raw_path="raw/distractor.pdf", status="ready"),
            DocumentChunk(
                id="in-scope",
                document_id="d1",
                ordinal=0,
                text="Scoped semantic vector evidence.",
                page_label="1",
                embedding=[0.6, 0.4],
            ),
            DocumentChunk(
                id="outside-scope-near",
                document_id="d2",
                ordinal=0,
                text="Near vector distractor outside the locked document.",
                page_label="1",
                embedding=[1.0, 0.0],
            ),
        ]
    )
    db.commit()

    class ScopedVectorStore:
        def __init__(self, db: Session) -> None:
            self.db = db

        def search(self, embedding: list[float], *, limit: int, document_ids: list[str] | None = None) -> list[VectorHit]:
            assert document_ids == ["d1"]
            return [VectorHit(chunk_id="in-scope", distance=0.3)]

    import app.services.search as search_module

    monkeypatch.setattr(search_module, "SQLiteVecStore", ScopedVectorStore)
    service = QueryService(db)
    service.ollama = EmbedOnlyOllama([1.0, 0.0])

    contexts = service._search_source_chunks("semantic vector evidence", "p1", ["d1"], limit=1)

    assert [context.citation.chunk_id for context in contexts] == ["in-scope"]


def test_search_source_chunks_keeps_json_fallback_when_vector_hits_are_partial(monkeypatch) -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.add(Document(id="d1", project_id="p1", title="Paper", file_name="paper.pdf", sha256="abc", raw_path="raw/paper.pdf", status="ready"))
    db.add_all(
        [
            DocumentChunk(
                id="vector-only",
                document_id="d1",
                ordinal=0,
                text="Weak vector hit.",
                page_label="1",
                embedding=[0.1, 0.9],
            ),
            DocumentChunk(
                id="json-best",
                document_id="d1",
                ordinal=1,
                text="Strong scoped JSON fallback evidence.",
                page_label="2",
                embedding=[1.0, 0.0],
            ),
        ]
    )
    db.commit()

    class PartialVectorStore:
        def __init__(self, db: Session) -> None:
            self.db = db

        def search(self, embedding: list[float], *, limit: int, document_ids: list[str] | None = None) -> list[VectorHit]:
            return [VectorHit(chunk_id="vector-only", distance=0.0)]

    import app.services.search as search_module

    monkeypatch.setattr(search_module, "SQLiteVecStore", PartialVectorStore)
    service = QueryService(db)
    service.ollama = EmbedOnlyOllama([1.0, 0.0])

    contexts = service._search_source_chunks("scoped JSON fallback evidence", "p1", ["d1"], limit=2)

    assert {context.citation.chunk_id for context in contexts} == {"vector-only", "json-best"}


def test_rag_contexts_retrieve_document_figure_metadata_before_generic_chunks() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="SAC-KG",
        file_name="sac-kg.pdf",
        sha256="abc",
        raw_path="raw/sac-kg.pdf",
        metadata_json={
            "source_slug": "sources/sac-kg",
            "document_intelligence": {
                "figures": [
                    {
                        "page_label": "3",
                        "caption": "Figure 1 shows the SAC-KG workflow.",
                        "note": "Figure 1 shows the SAC-KG architecture with Generator, Verifier, and Pruner components.",
                        "image_path": "images/figure-1.png",
                    }
                ]
            },
        },
        status="ready",
    )
    db.add_all([project, document])
    db.commit()

    service = QueryService(db)
    service.ollama = EmbedOnlyOllama([])

    contexts = service._build_rag_contexts(
        "What does Figure 1 show?",
        "p1",
        [PaperMatch(document=document, score=20, exact_alias=True)],
    )

    assert contexts
    assert contexts[0].citation.document_id == "d1"
    assert contexts[0].citation.page_slug == "sources/sac-kg"
    assert contexts[0].citation.page_label == "3"
    assert "Generator, Verifier, and Pruner" in contexts[0].citation.excerpt
    assert "images/figure-1.png" in contexts[0].citation.excerpt


def test_finalize_contexts_preserves_table_figure_and_profile_evidence() -> None:
    service = QueryService(make_session())
    generic_contexts = [
        RetrievedContext(
            citation=Citation(document_id="d1", chunk_id=f"generic-{index}", score=100 - index, excerpt=f"Generic source evidence {index}."),
            prompt_text=f"Generic source evidence {index}.",
            score=100 - index,
        )
        for index in range(10)
    ]
    table_context = RetrievedContext(
        citation=Citation(
            document_id="d1",
            chunk_id="table",
            score=1,
            excerpt="Table 2: metrics\n| Model | F1 |\n| --- | --- |\n| SAC-KG | 88.8 |",
        ),
        prompt_text="Table 2: metrics\n| Model | F1 |\n| --- | --- |\n| SAC-KG | 88.8 |",
        score=1,
        evidence_kind="table",
    )
    figure_context = RetrievedContext(
        citation=Citation(document_id="d1", chunk_id="figure", score=0.9, page_label="3", excerpt="Figure 1 shows Generator, Verifier, and Pruner components."),
        prompt_text="Figure 1 shows Generator, Verifier, and Pruner components.",
        score=0.9,
        evidence_kind="figure",
    )
    profile_context = RetrievedContext(
        citation=Citation(document_id="d1", chunk_id="profile", score=0.8, excerpt="CMAP torsion fitting protocol profile-term evidence."),
        prompt_text="CMAP torsion fitting protocol profile-term evidence.",
        score=0.8,
        evidence_kind="profile-term",
    )

    finalized = service._finalize_contexts([*generic_contexts, table_context, figure_context, profile_context])
    chunk_ids = {context.citation.chunk_id for context in finalized}

    assert len(finalized) == 8
    assert {"table", "figure", "profile"}.issubset(chunk_ids)
