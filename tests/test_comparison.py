from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base
from app.models.records import (
    Claim,
    Document,
    DocumentChunk,
    DocumentStatus,
    Entity,
    KnowledgeEdge,
    Project,
)
from app.schemas.agent import (
    ComparisonEvidenceCell,
    ComparisonPack,
    ComparisonPaper,
    EvidenceItem,
    EvidencePack,
)
from app.services.agent_synthesizer import AgentSynthesizer
from app.services.comparison import ComparisonService
from app.services.search import PaperMatch, QueryService, RetrievedContext
from app.schemas.common import Citation
from app.services.agent_executor import AgentExecutor


def make_db() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def seed_documents(db: Session) -> None:
    db.add(Project(id="p1", slug="demo", name="Demo"))
    for document_id, title in (("d1", "Paper A"), ("d2", "Paper B"), ("d3", "Paper C")):
        db.add(
            Document(
                id=document_id,
                project_id="p1",
                title=title,
                file_name=f"{document_id}.pdf",
                sha256=document_id.ljust(64, "0"),
                raw_path=f"{document_id}.pdf",
                raw_text="method dataset training metrics limitations",
                status=DocumentStatus.ready.value,
                active_parse_version="v1",
                metadata_json={},
            )
        )
    db.commit()


def patch_cell_retrieval(monkeypatch):
    def build_contexts(self, question, project_id, paper_matches, *, document_ids=None, **kwargs):
        document_id = document_ids[0]
        return [
            RetrievedContext(
                citation=Citation(
                    document_id=document_id,
                    chunk_id=f"chunk-{document_id}",
                    page_label="1",
                    page_title="Page 1",
                    score=1.0,
                    excerpt=f"Evidence for {document_id}: {question}",
                ),
                prompt_text=f"Evidence for {document_id}",
                score=1.0,
                context_text=f"Evidence for {document_id}",
            )
        ]

    def evidence_items(self, contexts, limit, *, question=""):
        context = contexts[0]
        return [
            EvidenceItem(
                index=0,
                document_id=context.citation.document_id,
                chunk_id=context.citation.chunk_id,
                page_label=context.citation.page_label,
                page_title=context.citation.page_title,
                score=context.citation.score,
                excerpt=context.citation.excerpt,
            )
        ], []

    monkeypatch.setattr(QueryService, "_build_rag_contexts", build_contexts)
    monkeypatch.setattr(QueryService, "_evidence_items_and_facts", evidence_items)


def test_explicit_scope_is_strict_and_matrix_is_balanced(monkeypatch) -> None:
    db = make_db()
    seed_documents(db)
    patch_cell_retrieval(monkeypatch)

    pack = ComparisonService(db).build(
        "demo",
        "compare methods",
        document_ids=["d1", "d2"],
        dimensions=["method"],
    )

    assert pack.comparison is not None
    assert pack.comparison.mode == "cross_paper"
    assert {paper.document_id for paper in pack.comparison.papers} == {"d1", "d2"}
    assert {item.document_id for item in pack.items} == {"d1", "d2"}
    assert all(cell.status == "supported" for cell in pack.comparison.cells)
    assert all(item.comparison_paper_id in {"d1", "d2"} for item in pack.items)


def test_auto_route_with_one_candidate_requests_selection(monkeypatch) -> None:
    db = make_db()
    seed_documents(db)

    def route_papers(self, question, project_id, limit=3, document_id=None):
        return [PaperMatch(document=db.get(Document, "d1"), score=1.0)]

    monkeypatch.setattr(QueryService, "_route_papers", route_papers)
    pack = ComparisonService(db).build("demo", "compare methods")

    assert pack.comparison is not None
    assert pack.comparison.mode == "candidate_selection"
    assert pack.comparison.status == "needs_selection"
    assert pack.comparison.candidate_document_ids == ["d1"]
    assert pack.items == []


def test_auto_comparison_router_prefers_named_papers(monkeypatch) -> None:
    db = make_db()
    seed_documents(db)
    db.get(Document, "d1").title = "selfrag_2310.11511"
    db.get(Document, "d2").title = "crag_2401.15884"
    db.commit()

    from app.services.search import QueryService

    matches = QueryService(db)._route_comparison_papers(
        "selfrag_2310.11511 vs crag_2401.15884", "p1"
    )

    assert [match.document.id for match in matches] == ["d1", "d2"]
    assert all(match.locked and match.exact_alias for match in matches)


def test_graph_edges_are_parse_version_scoped_and_align_entities() -> None:
    db = make_db()
    seed_documents(db)
    db.get(Document, "d1").active_parse_version = "a1"
    db.get(Document, "d2").active_parse_version = "b1"
    db.add_all(
        [
            Entity(id="e1", project_id="p1", name="Method A", aliases=["solver"], summary=""),
            Entity(id="e2", project_id="p1", name="Method B", aliases=["solver"], summary=""),
            DocumentChunk(
                id="c1", document_id="d1", parse_version="a1", ordinal=0, text="claim A",
                embedding_text="claim A", source_block_ids=[], source_spans=[], section_path=[],
            ),
            DocumentChunk(
                id="c2", document_id="d2", parse_version="b1", ordinal=0, text="claim B",
                embedding_text="claim B", source_block_ids=[], source_spans=[], section_path=[],
            ),
            Claim(
                id="cl1", project_id="p1", document_id="d1", subject="solver",
                predicate="uses", object_text="Method A", evidence_chunk_id="c1",
                confidence=0.95, verification_status="verified", metadata_json={},
            ),
            Claim(
                id="cl2", project_id="p1", document_id="d2", subject="solver",
                predicate="uses", object_text="Method B", evidence_chunk_id="c2",
                confidence=0.95, verification_status="verified", metadata_json={},
            ),
        ]
    )
    db.commit()

    service = ComparisonService(db, sac_kg_enabled=True)
    edges_v1 = service._ensure_graph_edges("p1", [db.get(Document, "d1"), db.get(Document, "d2")])
    assert any(edge.relation_type == "same_as" for edge in edges_v1)
    assert any(edge.relation_type == "compared_with" and edge.parse_version == "compare:a1|b1" for edge in edges_v1)
    assert any(edge.relation_type == "contradicts" and edge.parse_version == "compare:a1|b1" for edge in edges_v1)

    db.get(Document, "d2").active_parse_version = "b2"
    db.commit()
    edges_v2 = service._ensure_graph_edges("p1", [db.get(Document, "d1"), db.get(Document, "d2")])
    assert any(edge.relation_type == "compared_with" and edge.parse_version == "compare:a1|b2" for edge in edges_v2)
    stored_versions = {
        edge.parse_version
        for edge in db.scalars(select(KnowledgeEdge)).all()
        if edge.relation_type == "compared_with"
    }
    assert {"compare:a1|b1", "compare:a1|b2"}.issubset(stored_versions)


def test_sac_kg_disabled_keeps_matrix_and_only_persists_pair_edge(monkeypatch) -> None:
    db = make_db()
    seed_documents(db)
    patch_cell_retrieval(monkeypatch)
    # These rows deliberately exist to prove that the comparison path does
    # not consume SAC-KG data when the feature is disabled.
    db.add_all(
        [
            Entity(id="e1", project_id="p1", name="Method A", aliases=["solver"], summary=""),
            Claim(
                id="cl1", project_id="p1", document_id="d1", subject="solver",
                predicate="uses", object_text="Method A", evidence_chunk_id="c1",
                confidence=0.95, verification_status="verified", metadata_json={},
            ),
        ]
    )
    db.commit()

    service = ComparisonService(db, sac_kg_enabled=False)
    pack = service.build("demo", "compare methods", document_ids=["d1", "d2"], dimensions=["method"])

    assert pack.status == "ok"
    assert pack.comparison is not None
    assert pack.comparison.conflict_cells == []
    assert {item.document_id for item in pack.items} == {"d1", "d2"}
    assert {edge.relation_type for edge in pack.comparison.edges} == {"compared_with"}
    assert {
        edge.relation_type for edge in db.scalars(select(KnowledgeEdge)).all()
    } == {"compared_with"}


def test_deepseek_receives_matrix_and_failure_returns_structured_fallback(monkeypatch) -> None:
    pack = ComparisonPack(
        mode="cross_paper",
        explicit_scope=True,
        papers=[
            ComparisonPaper(document_id="d1", title="Paper A", ordinal=0),
            ComparisonPaper(document_id="d2", title="Paper B", ordinal=1),
        ],
        dimensions=["method/architecture"],
        cells=[
            ComparisonEvidenceCell(
                paper_id="d1", paper_title="Paper A", dimension="method/architecture",
                status="supported", evidence_indexes=[0], citation_indexes=[0], confidence=0.8,
            ),
            ComparisonEvidenceCell(
                paper_id="d2", paper_title="Paper B", dimension="method/architecture",
                status="missing", evidence_indexes=[], citation_indexes=[], confidence=0.0,
            ),
        ],
        missing_cells=["d2:method/architecture"],
        status="partial",
    )
    items = [
        {
            "index": 0,
            "document_id": "d1",
            "chunk_id": "c1",
            "page_label": "1",
            "excerpt": "Paper A uses a graph encoder.",
        }
    ]
    evidence_pack = {"items": items, "comparison": pack.model_dump()}
    citations = [{"document_id": "d1", "chunk_id": "c1", "excerpt": items[0]["excerpt"]}]
    captured: list[str] = []

    def success(self, messages, **kwargs):
        captured.append("\n".join(message["content"] for message in messages))
        return {"content": '{"answer_markdown":"Paper A uses a graph encoder [0]. Paper B: evidence missing.","cited_indexes":[0],"warnings":[],"confidence":0.9}', "model": "deepseek-chat"}

    monkeypatch.setattr("app.services.agent_synthesizer.DeepSeekClient.generate_chat", success)
    synthesizer = AgentSynthesizer()
    synthesizer._settings = SimpleNamespace(
        deepseek_api_key="test-key",
        deepseek_base_url="https://example.test",
        deepseek_model="deepseek-chat",
        generation_timeout_seconds=1,
        generation_max_retries=0,
        generation_retry_backoff_seconds=0,
        generation_max_output_tokens=500,
    )
    result = synthesizer._deepseek_synthesize(
        query="compare the methods",
        route="multi_source_compare",
        conversation_summary="",
        rag_answer="",
        citations=citations,
        evidence_pack=evidence_pack,
    )
    assert captured and "Comparison evidence matrix" in captured[0]
    assert "paper=d1" in captured[0] and "paper=d2" in captured[0]
    assert result["answer_markdown"].startswith("Paper A")

    def fail(self, messages, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr("app.services.agent_synthesizer.DeepSeekClient.generate_chat", fail)
    fallback = synthesizer._deepseek_synthesize(
        query="compare the methods",
        route="multi_source_compare",
        conversation_summary="",
        rag_answer="LLM generation temporarily failed; following is raw retrieval evidence",
        citations=citations,
        evidence_pack=evidence_pack,
    )
    assert "Comparison evidence matrix" in fallback["answer_markdown"]
    assert "raw retrieval evidence" not in fallback["answer_markdown"]

    # A non-empty legacy RAG draft must not bypass the structured fallback.
    # This is the production failure mode when a gateway returns a JSON array
    # (or otherwise malformed JSON) after ``rag.answer`` already produced a
    # long flat draft.
    def malformed(self, messages, **kwargs):
        return {"content": "[not an object]", "model": "deepseek-chat"}

    monkeypatch.setattr("app.services.agent_synthesizer.DeepSeekClient.generate_chat", malformed)
    malformed_fallback = synthesizer._deepseek_synthesize(
        query="compare the methods",
        route="multi_source_compare",
        conversation_summary="",
        rag_answer="A long flat RAG draft that must not be returned as the comparison result.",
        citations=citations,
        evidence_pack=evidence_pack,
    )
    assert "Comparison evidence matrix" in malformed_fallback["answer_markdown"]
    assert "long flat RAG draft" not in malformed_fallback["answer_markdown"]
    assert any("structured comparison matrix fallback" in warning for warning in malformed_fallback["warnings"])


def test_synthesis_indexes_preserve_both_paper_sides_and_remap_cells() -> None:
    pack = {
        "comparison": {
            "papers": [{"document_id": "d1"}, {"document_id": "d2"}],
            "cells": [
                {"paper_id": "d1", "citation_indexes": [0]},
                {"paper_id": "d2", "citation_indexes": [1]},
            ],
        }
    }
    citations = [
        Citation(document_id="d1", chunk_id="c1", score=1.0, excerpt="a"),
        Citation(document_id="d2", chunk_id="c2", score=1.0, excerpt="b"),
    ]
    selected = AgentExecutor._normalized_synthesis_indexes([0], citations, pack)
    assert selected == [0, 1]
    AgentExecutor._remap_comparison_citation_indexes(pack, selected)
    assert pack["comparison"]["cells"][0]["citation_indexes"] == [0]
    assert pack["comparison"]["cells"][1]["citation_indexes"] == [1]
