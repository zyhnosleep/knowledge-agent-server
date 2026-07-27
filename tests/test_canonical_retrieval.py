from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import Document, DocumentChunk, Project
from app.schemas.agent import EvidenceItem
from app.schemas.common import Citation
from app.services.ai import QueryAnswerPayload
from app.services.search import QueryService, RetrievedContext


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)()


class NoVectorOllama:
    def embed(self, _texts: list[str]) -> list[list[float]]:
        return [[]]


def add_document(
    db: Session,
    *,
    document_id: str = "d1",
    active_parse_version: str | None = "canonical-v1",
) -> Document:
    if db.get(Project, "p1") is None:
        db.add(Project(id="p1", slug="project", name="Project"))
    document = Document(
        id=document_id,
        project_id="p1",
        title="SAC-KG",
        file_name=f"{document_id}.pdf",
        sha256=f"sha-{document_id}",
        raw_path=f"raw/{document_id}.pdf",
        status="ready",
        active_parse_version=active_parse_version,
    )
    db.add(document)
    return document


def chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "d1",
    parse_version: str = "canonical-v1",
    parent_chunk_id: str | None = None,
    chunk_role: str = "child",
    block_type: str = "narrative",
    ordinal: int = 0,
    previous_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
    contextual_prefix: str | None = None,
    source_spans: list[dict] | None = None,
    source_block_ids: list[str] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id=document_id,
        parse_version=parse_version,
        parent_chunk_id=parent_chunk_id,
        chunk_role=chunk_role,
        block_type=block_type,
        ordinal=ordinal,
        text=text,
        embedding_text=(
            f"{contextual_prefix}\n\n{text}" if contextual_prefix else text
        ),
        contextual_prefix=contextual_prefix,
        previous_chunk_id=previous_chunk_id,
        next_chunk_id=next_chunk_id,
        source_spans=source_spans or [{"page_index": 0, "page_label": "1"}],
        source_block_ids=source_block_ids or [chunk_id],
    )


def service_for(db: Session) -> QueryService:
    service = QueryService(db)
    service.ollama = NoVectorOllama()
    return service


def test_active_retrieval_returns_only_children_and_excludes_references() -> None:
    db = make_session()
    add_document(db)
    db.add_all(
        [
            chunk("active", "unique canonical result"),
            chunk("appendix", "unique appendix result", block_type="appendix", ordinal=1),
            chunk("reference", "unique reference result", block_type="reference", ordinal=2),
            chunk("parent", "unique parent result", chunk_role="parent", ordinal=3),
            chunk("old", "unique old result", parse_version="old-v", ordinal=4),
        ]
    )
    db.commit()

    contexts = service_for(db)._search_source_chunks(
        "unique result", "p1", ["d1"], limit=10
    )

    assert {item.citation.chunk_id for item in contexts} == {"active", "appendix"}


def test_child_hit_expands_parent_but_cites_only_original_child_text() -> None:
    db = make_session()
    add_document(db)
    prefix = "This chunk reports the main SAC-KG experiment."
    parent = chunk(
        "parent",
        "Experiments\n\nThe section compares SAC-KG with multiple baselines.",
        chunk_role="parent",
    )
    child = chunk(
        "child",
        "SAC-KG reaches 91.2 F1 on OIE2016.",
        parent_chunk_id="parent",
        contextual_prefix=prefix,
        ordinal=1,
        source_spans=[{"page_index": 4, "page_label": "5", "char_start": 10, "char_end": 49}],
    )
    db.add_all([parent, child])
    db.commit()

    context = service_for(db)._search_source_chunks(
        "SAC-KG OIE2016 F1", "p1", ["d1"], limit=1
    )[0]

    assert context.parent_chunk_id == "parent"
    assert "Experiments" in context.context_text
    assert context.citation.excerpt == child.text
    assert prefix not in context.citation.excerpt
    assert prefix not in context.prompt_text
    assert context.citation.source_spans == child.source_spans


@pytest.mark.parametrize(
    "question",
    ["compare SAC-KG approaches", "summarize current child"],
)
def test_overview_and_comparison_neighbor_expansion_is_bounded_by_token_budget(
    question: str,
) -> None:
    db = make_session()
    add_document(db)
    db.add_all(
        [
            chunk("parent", "Results section context", chunk_role="parent"),
            chunk("previous", "Previous child introduces the baseline.", parent_chunk_id="parent", ordinal=1, next_chunk_id="hit"),
            chunk(
                "hit",
                "Current child compares SAC-KG approaches.",
                parent_chunk_id="parent",
                ordinal=2,
                previous_chunk_id="previous",
                next_chunk_id="next",
            ),
            chunk("next", "Next child explains the observed gain.", parent_chunk_id="parent", ordinal=3, previous_chunk_id="hit"),
        ]
    )
    db.commit()
    service = service_for(db)
    service.NEIGHBOR_EXPANSION_TOKEN_BUDGET = 6
    service._retrieval_token_counter = lambda text: len(text.split())

    context = service._search_source_chunks(
        question, "p1", ["d1"], limit=1
    )[0]

    assert "Previous child" in context.prompt_text or "Next child" in context.prompt_text
    assert service._retrieval_token_counter(context.neighbor_text) <= 6


def test_legacy_child_without_parent_remains_retrievable() -> None:
    db = make_session()
    add_document(db, active_parse_version=None)
    db.add(
        chunk(
            "legacy-child",
            "legacy unique evidence",
            parse_version="legacy",
            parent_chunk_id=None,
        )
    )
    db.commit()

    context = service_for(db)._search_source_chunks(
        "legacy unique evidence", "p1", ["d1"], limit=1
    )[0]

    assert context.citation.chunk_id == "legacy-child"
    assert context.citation.excerpt == "legacy unique evidence"


def test_table_hit_contains_caption_header_row_group_and_table_id() -> None:
    db = make_session()
    add_document(db)
    table_text = (
        "Table 5: Main results\n\n"
        "| Dataset | Model | F1 | AUC |\n"
        "| --- | --- | --- | --- |\n"
        "| OIE2016 | SAC-KG | 91.2 | 94.0 |"
    )
    db.add_all(
        [
            chunk("table-parent", table_text, chunk_role="parent", block_type="table"),
            chunk(
                "table-child",
                table_text,
                parent_chunk_id="table-parent",
                block_type="table",
                ordinal=1,
                source_spans=[{"page_index": 6, "page_label": "7", "table_id": "table-5", "row_index": 2}],
                source_block_ids=["table-block-5"],
            ),
        ]
    )
    db.commit()

    context = service_for(db)._search_source_chunks(
        "OIE2016 results", "p1", ["d1"], limit=1
    )[0]

    assert "Table 5" in context.prompt_text
    assert "Dataset | Model | F1 | AUC" in context.prompt_text
    assert "OIE2016" in context.prompt_text
    assert context.citation.table_id == "table-5"


def test_figure_and_formula_hits_keep_typed_ids_assets_and_source_spans() -> None:
    db = make_session()
    add_document(db)
    db.add_all(
        [
            chunk("figure-parent", "Figure 2 shows accuracy trends.", chunk_role="parent", block_type="figure"),
            chunk(
                "figure-child",
                "Figure 2 shows accuracy trends.",
                parent_chunk_id="figure-parent",
                block_type="figure",
                ordinal=1,
                source_block_ids=["figure-2"],
                source_spans=[{"page_index": 2, "page_label": "3", "image_relationship_id": "asset-rId7"}],
            ),
            chunk("formula-parent", "Equation 4: L = L_r + lambda L_c", chunk_role="parent", block_type="formula", ordinal=2),
            chunk(
                "formula-child",
                "Equation 4: L = L_r + lambda L_c",
                parent_chunk_id="formula-parent",
                block_type="formula",
                ordinal=3,
                source_block_ids=["formula-4"],
                source_spans=[{"page_index": 3, "page_label": "4", "metadata": {"formula_id": "formula-4"}}],
            ),
        ]
    )
    db.commit()
    service = service_for(db)

    figure = service._search_source_chunks("Figure 2 trends", "p1", ["d1"], limit=1)[0]
    formula = service._search_source_chunks("Equation 4 lambda", "p1", ["d1"], limit=1)[0]

    assert figure.citation.block_type == "figure"
    assert figure.citation.figure_id == "figure-2"
    assert figure.citation.asset_id == "asset-rId7"
    assert figure.citation.source_spans[0]["page_index"] == 2
    assert formula.citation.block_type == "formula"
    assert formula.citation.formula_id == "formula-4"
    assert formula.citation.source_spans[0]["page_index"] == 3


def test_canonical_schema_fields_are_optional_and_backward_compatible() -> None:
    legacy_citation = Citation(score=1.0, excerpt="legacy")
    legacy_item = EvidenceItem(index=0, score=1.0, excerpt="legacy")

    assert legacy_citation.parse_version is None
    assert legacy_item.parent_chunk_id is None

    citation = Citation(
        score=1.0,
        excerpt="source",
        parse_version="canonical-v1",
        parent_chunk_id="parent",
        block_type="figure",
        source_spans=[{"page_index": 1}],
        asset_id="asset-1",
        table_id=None,
        figure_id="figure-1",
        formula_id=None,
    )
    item = EvidenceItem(
        index=0,
        score=1.0,
        excerpt="source",
        context_text="parent context",
        **citation.model_dump(exclude={"score", "excerpt"}),
    )

    assert item.parse_version == "canonical-v1"
    assert item.context_text == "parent context"
    assert item.figure_id == "figure-1"


def test_answer_enforces_context_token_budget_before_drafting(monkeypatch) -> None:
    db = make_session()
    add_document(db)
    db.commit()
    service = service_for(db)
    service.DRAFT_CONTEXT_TOKEN_BUDGET = 5
    service._retrieval_token_counter = lambda text: len(text.split())
    contexts = [
        RetrievedContext(
            citation=Citation(score=10.0, excerpt="one two three four"),
            prompt_text="one two three four",
            score=10.0,
        ),
        RetrievedContext(
            citation=Citation(score=9.0, excerpt="five six seven four"),
            prompt_text="five six seven four",
            score=9.0,
        ),
    ]
    service._route_papers = lambda *_args, **_kwargs: []
    service._build_rag_contexts = lambda *_args, **_kwargs: contexts
    monkeypatch.setattr(service, "_deterministic_table_answer_if_supported", lambda *_args: None)

    def draft(_question: str, _index_context: str | None, received: list[RetrievedContext]) -> QueryAnswerPayload:
        assert sum(service._retrieval_token_counter(item.prompt_text) for item in received) <= 5
        return QueryAnswerPayload(answer_markdown="Grounded answer [0]", citations=[0], risk_level="normal")

    monkeypatch.setattr(service, "_draft_answer", draft)

    service.answer("project", "ordinary grounded question", save_answer=False)
