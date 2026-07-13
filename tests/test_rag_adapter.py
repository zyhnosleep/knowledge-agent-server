from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base
from app.models.records import Project
from app.schemas.common import QueryResponse
from app.services.rag_adapter import RAGAdapter


def make_db() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def test_rag_adapter_returns_query_response(monkeypatch) -> None:
    """RAGAdapter.answer() returns a QueryResponse when the project exists."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    def fake_answer(self, project_slug, question, save_answer=False, document_id=None):
        return QueryResponse(
            answer_markdown="test answer",
            citations=[],
            verification_status="local-only",
        )

    monkeypatch.setattr(
        "app.services.rag_adapter.QueryService.answer", fake_answer
    )

    adapter = RAGAdapter()
    result = adapter.answer(db, "demo", "what is this?")

    assert isinstance(result, QueryResponse)
    assert result.answer_markdown == "test answer"
    assert result.verification_status == "local-only"


def test_rag_adapter_missing_project_returns_empty(monkeypatch) -> None:
    """RAGAdapter.answer() returns empty response for unknown project slug."""
    db = make_db()

    def fake_answer(self, project_slug, question, save_answer=False, document_id=None):
        raise ValueError("Project not found")

    monkeypatch.setattr(
        "app.services.rag_adapter.QueryService.answer", fake_answer
    )

    adapter = RAGAdapter()
    result = adapter.answer(db, "nonexistent", "what?")

    assert isinstance(result, QueryResponse)
    assert result.answer_markdown == ""
    assert result.citations == []
    assert result.verification_status == "project_not_found"


# ------------------------------------------------------------------
# Evidence pack schema tests
# ------------------------------------------------------------------


def test_evidence_item_has_required_fields() -> None:
    """EvidenceItem model exposes all required fields."""
    from app.schemas.agent import EvidenceItem

    item = EvidenceItem(
        index=0,
        document_id="d1",
        chunk_id="c1",
        page_slug="page/slug",
        page_title="Page Title",
        page_kind="source_summary",
        page_label="Table 1",
        score=0.95,
        excerpt="excerpt text",
        evidence_kind="table",
        source_stage="document_table",
        support_hint="direct",
    )
    assert item.index == 0
    assert item.document_id == "d1"
    assert item.chunk_id == "c1"
    assert item.page_slug == "page/slug"
    assert item.page_title == "Page Title"
    assert item.page_kind == "source_summary"
    assert item.page_label == "Table 1"
    assert item.score == 0.95
    assert item.excerpt == "excerpt text"
    assert item.evidence_kind == "table"
    assert item.source_stage == "document_table"
    assert item.support_hint == "direct"


def test_evidence_pack_has_status_and_items() -> None:
    """EvidencePack holds status string and a list of EvidenceItem."""
    from app.schemas.agent import EvidenceItem, EvidencePack

    item = EvidenceItem(
        index=0,
        document_id="d1",
        score=0.95,
        excerpt="test",
        evidence_kind="table",
        source_stage="document_table",
        support_hint="direct",
    )
    pack = EvidencePack(status="ok", items=[item])
    assert pack.status == "ok"
    assert len(pack.items) == 1
    assert pack.items[0].document_id == "d1"


def test_evidence_item_serialization_is_json_safe() -> None:
    """EvidenceItem serializes to JSON-safe dict without secrets or raw provider responses."""
    import json
    from app.schemas.agent import EvidenceItem

    item = EvidenceItem(
        index=0,
        document_id="d1",
        score=0.95,
        excerpt="test",
        evidence_kind="table",
        source_stage="document_table",
        support_hint="direct",
    )
    data = item.model_dump()
    json_str = json.dumps(data)
    assert isinstance(json_str, str)
    # No secrets in serialized output
    assert "api_key" not in json_str.lower()
    assert "authorization" not in json_str.lower()
    assert "bearer" not in json_str.lower()


def test_evidence_pack_empty_is_valid() -> None:
    """An empty EvidencePack with status is valid."""
    from app.schemas.agent import EvidencePack

    pack = EvidencePack(status="project_not_found", items=[])
    assert pack.status == "project_not_found"
    assert len(pack.items) == 0
    data = pack.model_dump()
    assert data["status"] == "project_not_found"
    assert data["items"] == []


# ------------------------------------------------------------------
# retrieve_evidence tests
# ------------------------------------------------------------------


def test_rag_adapter_retrieve_evidence_returns_evidence_pack(monkeypatch) -> None:
    """RAGAdapter.retrieve_evidence() returns an EvidencePack when project exists."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.schemas.agent import EvidencePack

    def fake_retrieve_evidence(self, project_slug, question, limit=15, document_id=None):
        from app.schemas.agent import EvidenceItem
        return EvidencePack(
            status="ok",
            items=[
                EvidenceItem(
                    index=0,
                    document_id="d1",
                    chunk_id="c1",
                    score=0.92,
                    excerpt="test excerpt",
                    evidence_kind="table",
                    source_stage="document_table",
                    support_hint="direct",
                )
            ],
        )

    monkeypatch.setattr(
        "app.services.rag_adapter.QueryService.retrieve_evidence",
        fake_retrieve_evidence,
    )

    adapter = RAGAdapter()
    result = adapter.retrieve_evidence(db, "demo", "what is this?")

    assert isinstance(result, EvidencePack)
    assert result.status == "ok"
    assert len(result.items) == 1
    assert result.items[0].document_id == "d1"
    assert result.items[0].evidence_kind == "table"
    assert result.items[0].source_stage == "document_table"
    assert result.items[0].support_hint == "direct"


def test_rag_adapter_retrieve_evidence_missing_project_returns_empty(monkeypatch) -> None:
    """RAGAdapter.retrieve_evidence() returns empty EvidencePack for unknown project slug."""
    db = make_db()

    def fake_retrieve_evidence(self, project_slug, question, limit=15, document_id=None):
        raise ValueError("Project not found")

    monkeypatch.setattr(
        "app.services.rag_adapter.QueryService.retrieve_evidence",
        fake_retrieve_evidence,
    )

    adapter = RAGAdapter()
    result = adapter.retrieve_evidence(db, "nonexistent", "what?")

    from app.schemas.agent import EvidencePack

    assert isinstance(result, EvidencePack)
    assert result.status == "project_not_found"
    assert len(result.items) == 0


def test_rag_adapter_retrieve_evidence_calls_query_service_with_limit(monkeypatch) -> None:
    """RAGAdapter.retrieve_evidence() passes limit through to QueryService."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    call_args = {}

    def fake_retrieve_evidence(self, project_slug, question, limit=15, document_id=None):
        call_args["project_slug"] = project_slug
        call_args["question"] = question
        call_args["limit"] = limit
        from app.schemas.agent import EvidencePack
        return EvidencePack(status="ok", items=[])

    monkeypatch.setattr(
        "app.services.rag_adapter.QueryService.retrieve_evidence",
        fake_retrieve_evidence,
    )

    adapter = RAGAdapter()
    result = adapter.retrieve_evidence(db, "demo", "test?", limit=5)

    assert result.status == "ok"
    assert call_args["project_slug"] == "demo"
    assert call_args["question"] == "test?"
    assert call_args["limit"] == 5
