from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event, literal, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import Claim, Document, DocumentChunk, Project
from app.schemas.agent import EvidenceItem
from app.schemas.common import Citation
from app.services.ai import QueryAnswerPayload
from app.services.canonical_models import (
    CanonicalAsset,
    CanonicalBlock,
    CanonicalCell,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalTable,
    SourceSpan,
)
from app.services.pipeline import IngestionPipeline
from app.services.paper_profile import ensure_paper_profile
from app.services.semantic_chunking import SemanticChunker
from app.services.search import QueryService, RetrievedContext
from app.services.structured_evidence import StructuredEvidenceBuilder


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


def test_shadow_parse_version_map_queries_staged_children_without_switching_pointer() -> None:
    db = make_session()
    document = add_document(db, active_parse_version="old-v")
    db.add_all(
        [
            chunk(
                "old-child",
                "old baseline evidence",
                parse_version="old-v",
            ),
            chunk(
                "staged-child",
                "staged unique evidence",
                parse_version="staged-v",
            ),
        ]
    )
    db.commit()
    service = QueryService(db, parse_version_map={document.id: "staged-v"})
    service.ollama = NoVectorOllama()

    contexts = service._search_source_chunks(
        "staged unique evidence",
        "p1",
        [document.id],
        limit=10,
    )

    assert [item.citation.chunk_id for item in contexts] == ["staged-child"]
    assert db.get(Document, document.id).active_parse_version == "old-v"


def test_shadow_map_is_mixed_with_active_versions_without_leaking_old_chunks() -> None:
    db = make_session()
    staged_document = add_document(db, document_id="d1", active_parse_version="old-v")
    active_document = add_document(
        db,
        document_id="d2",
        active_parse_version="active-v",
    )
    active_document.title = "Second Paper"
    db.add_all(
        [
            chunk(
                "old-child",
                "old staged target evidence",
                document_id=staged_document.id,
                parse_version="old-v",
            ),
            chunk(
                "staged-child",
                "staged target evidence",
                document_id=staged_document.id,
                parse_version="staged-v",
            ),
            chunk(
                "active-child",
                "active target evidence",
                document_id=active_document.id,
                parse_version="active-v",
            ),
            chunk(
                "inactive-child",
                "inactive target evidence",
                document_id=active_document.id,
                parse_version="inactive-v",
            ),
        ]
    )
    db.commit()
    service = QueryService(db, parse_version_map={staged_document.id: "staged-v"})
    service.ollama = NoVectorOllama()

    contexts = service._search_source_chunks(
        "staged active target evidence",
        "p1",
        [staged_document.id, active_document.id],
        limit=10,
    )

    assert {item.citation.chunk_id for item in contexts} == {
        "staged-child",
        "active-child",
    }
    assert db.get(Document, staged_document.id).active_parse_version == "old-v"


def test_shadow_retrieve_evidence_routes_with_staged_child_text() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="project", name="Project"))
    staged_document = Document(
        id="d1",
        project_id="p1",
        title="First Paper",
        file_name="first.pdf",
        sha256="sha-d1",
        raw_path="raw/first.pdf",
        raw_text="The old active version discusses unrelated evidence.",
        metadata_json={
            "source_slug": "sources/first-paper",
            "paper_profile": {
                "profile_version": "paper-profile-v1",
                "source_text_sha256": "stale",
                "aliases": [],
                "key_terms": ["OldOnlyTerm"],
            },
        },
        status="ready",
        active_parse_version="old-v",
    )
    distractor = Document(
        id="d2",
        project_id="p1",
        title="Second Paper",
        file_name="second.pdf",
        sha256="sha-d2",
        raw_path="raw/second.pdf",
        raw_text="NMX is mentioned as background, not introduced here.",
        metadata_json={"source_slug": "sources/second-paper"},
        status="ready",
        active_parse_version="active-v",
    )
    db.add_all(
        [
            staged_document,
            distractor,
            chunk(
                "staged-child",
                "This paper introduces NMX architecture for the target task evidence.",
                document_id="d1",
                parse_version="staged-v",
            ),
            chunk(
                "old-child",
                "Unrelated old active evidence.",
                document_id="d1",
                parse_version="old-v",
            ),
            chunk(
                "distractor-child",
                "NMX is background in this other paper.",
                document_id="d2",
                parse_version="active-v",
            ),
        ]
    )
    db.commit()
    service = QueryService(db, parse_version_map={"d1": "staged-v"})
    service.ollama = NoVectorOllama()

    evidence = service.retrieve_evidence(
        "project",
        "What NMX architecture is used for the target task evidence?",
        limit=5,
    )

    assert evidence.items
    assert evidence.items[0].document_id == "d1"
    assert evidence.items[0].chunk_id == "staged-child"
    assert db.get(Document, "d1").active_parse_version == "old-v"
    assert db.get(Document, "d1").raw_text.startswith("The old active version")


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
                source_block_ids=["figure-block-not-id"],
                source_spans=[
                    {
                        "page_index": 2,
                        "page_label": "3",
                        "image_relationship_id": "relationship-not-asset-id",
                        "metadata": {
                            "figure_id": "figure-2",
                            "asset_id": "asset-rId7",
                        },
                    }
                ],
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


def test_claim_search_correlates_each_chunk_to_its_own_active_document() -> None:
    db = make_session()
    add_document(db, document_id="d1", active_parse_version="v1")
    add_document(db, document_id="d2", active_parse_version="v2")
    active = chunk(
        "active-claim-child",
        "SAC-KG reports OIE2016 evidence.",
        document_id="d1",
        parse_version="v1",
    )
    inactive = chunk(
        "inactive-claim-child",
        "SAC-KG reports OIE2016 evidence with many extra matching SAC-KG OIE2016 terms.",
        document_id="d2",
        parse_version="v1",
    )
    db.add_all(
        [
            active,
            inactive,
            Claim(
                id="active-claim",
                project_id="p1",
                document_id="d1",
                subject="SAC-KG",
                predicate="reports",
                object_text="OIE2016 evidence",
                evidence_chunk_id=active.id,
                confidence=0.1,
            ),
            Claim(
                id="inactive-claim",
                project_id="p1",
                document_id="d2",
                subject="SAC-KG OIE2016",
                predicate="reports",
                object_text="SAC-KG OIE2016 evidence",
                evidence_chunk_id=inactive.id,
                confidence=1.0,
            ),
        ]
    )
    db.commit()

    contexts = service_for(db)._search_claim_evidence_contexts(
        "Compare SAC-KG OIE2016 results", "p1", ["d1", "d2"], limit=1
    )

    assert [context.citation.chunk_id for context in contexts] == [active.id]


def test_source_search_bulk_loads_only_top_hit_relationships() -> None:
    db = make_session()
    add_document(db)
    rows: list[DocumentChunk] = []
    for index in range(30):
        parent_id = f"parent-{index}"
        child_id = f"child-{index}"
        rows.extend(
            [
                chunk(
                    parent_id,
                    f"Parent section {index} contains shared evidence.",
                    chunk_role="parent",
                    ordinal=index * 2,
                ),
                chunk(
                    child_id,
                    f"Child {index} contains shared evidence.",
                    parent_chunk_id=parent_id,
                    ordinal=index * 2 + 1,
                ),
            ]
        )
    db.add_all(rows)
    db.commit()
    db.expunge_all()
    service = service_for(db)
    statements: list[str] = []

    def record_statement(_connection, _cursor, statement, _parameters, _context, _executemany) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    engine = db.get_bind()
    event.listen(engine, "before_cursor_execute", record_statement)
    try:
        contexts = service._search_source_chunks(
            "shared evidence", "p1", ["d1"], limit=3
        )
    finally:
        event.remove(engine, "before_cursor_execute", record_statement)

    assert len(contexts) == 3
    assert len(statements) <= 4


def test_structured_ids_survive_semantic_chunking_persistence_and_retrieval() -> None:
    table_span = SourceSpan(page_index=1, source_block_id="table-source")
    figure_span = SourceSpan(page_index=2, source_block_id="figure-source")
    formula_span = SourceSpan(page_index=3, source_block_id="formula-source")
    canonical = CanonicalDocument(
        document_id="d1",
        parser_source="test",
        parse_version="canonical-v1",
        blocks=[
            CanonicalBlock(
                block_id="table-block-not-id",
                block_type="table",
                text="",
                reading_order=0,
                parser_source="test",
                table_id="table-real",
                source_spans=[table_span],
            ),
            CanonicalBlock(
                block_id="figure-block-not-id",
                block_type="figure",
                text="",
                reading_order=1,
                parser_source="test",
                figure_id="figure-real",
                source_spans=[figure_span],
            ),
            CanonicalBlock(
                block_id="formula-block-not-id",
                block_type="formula",
                text="",
                reading_order=2,
                parser_source="test",
                formula_id="formula-real",
                source_spans=[formula_span],
            ),
        ],
        tables=[
            CanonicalTable(
                table_id="table-real",
                caption="Table 5: Results",
                headers=["Dataset", "F1"],
                rows=[["OIE2016", "91.2"]],
                cells=[
                    CanonicalCell(text="Dataset", row_index=0, column_index=0, is_header=True),
                    CanonicalCell(text="F1", row_index=0, column_index=1, is_header=True),
                    CanonicalCell(text="OIE2016", row_index=1, column_index=0),
                    CanonicalCell(text="91.2", row_index=1, column_index=1),
                ],
                source_spans=[table_span],
            )
        ],
        figures=[
            CanonicalFigure(
                figure_id="figure-real",
                caption="Figure 2: Accuracy",
                asset_path="assets/figure-two.png",
                source_spans=[figure_span],
            )
        ],
        formulas=[
            CanonicalFormula(
                formula_id="formula-real",
                latex="L=L_r+lambda L_c",
                caption="Equation 4",
                source_spans=[formula_span],
            )
        ],
        assets=[
            CanonicalAsset(
                asset_id="asset-real",
                path="assets/figure-two.png",
                media_type="image/png",
            )
        ],
    )
    drafts = SemanticChunker(
        embedder=NoVectorOllama(),
        token_counter=lambda text: len(text.split()),
        parent_min_tokens=1,
        parent_target_tokens=20,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=20,
        child_max_tokens=100,
        overlap_tokens=0,
    ).build(canonical)
    context = type(
        "PipelineContext",
        (),
        {
            "document": type("PipelineDocument", (), {"id": "d1"})(),
            "version": type("PipelineVersion", (), {"version_key": "canonical-v1"})(),
        },
    )()
    persisted = {
        draft.block_type: IngestionPipeline._document_chunk_from_draft(
            context,
            draft,
            embedding=None,
            parent_chunk_id=draft.parent_local_id,
        )
        for draft in drafts
        if draft.chunk_role == "child"
    }

    retrieval = service_for(make_session())
    table_ids = retrieval._expand_child_hit(
        persisted["table"],
        question="OIE2016 results",
        score=1.0,
        page_fields={},
        evidence_kind="table",
        related_chunks={},
    ).citation
    figure_ids = retrieval._expand_child_hit(
        persisted["figure"],
        question="Figure 2 accuracy",
        score=1.0,
        page_fields={},
        evidence_kind="figure",
        related_chunks={},
    ).citation
    formula_ids = retrieval._expand_child_hit(
        persisted["formula"],
        question="Equation 4",
        score=1.0,
        page_fields={},
        evidence_kind="formula",
        related_chunks={},
    ).citation

    assert table_ids.table_id == "table-real"
    assert table_ids.asset_id is None
    assert figure_ids.figure_id == "figure-real"
    assert figure_ids.asset_id == "asset-real"
    assert formula_ids.formula_id == "formula-real"


def test_untyped_source_and_relationship_ids_never_become_typed_ids() -> None:
    untyped = chunk(
        "figure-child",
        "Figure source text",
        block_type="figure",
        source_block_ids=["block-id-is-not-figure-id"],
        source_spans=[
            {
                "page_index": 1,
                "image_relationship_id": "relationship-id-is-not-asset-id",
            }
        ],
    )

    assert QueryService._canonical_chunk_identifiers(untyped) == {
        "asset_id": None,
        "table_id": None,
        "figure_id": None,
        "formula_id": None,
    }


def test_context_budget_counts_the_exact_draft_representation() -> None:
    db = make_session()
    service = service_for(db)
    service.DRAFT_CONTEXT_TOKEN_BUDGET = 5
    service._retrieval_token_counter = lambda text: len(text.split())
    context = RetrievedContext(
        citation=Citation(score=1.0, excerpt="fallback has three"),
        prompt_text="surface has four tokens",
        score=1.0,
    )

    fitted = service._fit_contexts_to_token_budget(
        [context], question="ordinary question"
    )

    assert len(fitted) == 1
    assert fitted[0].prompt_text == "fallback has three"
    rendered = f"[0] {service._prompt_context_text('ordinary question', fitted[0])}"
    assert service._retrieval_token_counter(rendered) <= 5


def test_prompt_context_preserves_complete_parent_without_character_window() -> None:
    evidence = "prefix " + ("x" * 2600) + " exact-tail"
    context = RetrievedContext(
        citation=Citation(score=1.0, excerpt="matched child"),
        prompt_text=evidence,
        score=1.0,
        parent_chunk_id="parent-long",
    )

    rendered = QueryService._prompt_context_text("ordinary question", context)

    assert rendered == evidence + "\n\nmatched child"
    assert rendered.endswith("exact-tail\n\nmatched child")


def test_answer_budget_expands_at_most_six_unique_parents_without_reranking() -> None:
    db = make_session()
    service = service_for(db)
    service.DRAFT_CONTEXT_TOKEN_BUDGET = 10_000
    service._retrieval_token_counter = lambda text: len(text.split())
    contexts = [
        RetrievedContext(
            citation=Citation(
                score=float(10 - index),
                chunk_id=f"child-{index}",
                excerpt=f"matched child {index}",
            ),
            prompt_text=f"complete parent evidence {index}",
            context_text=f"complete parent evidence {index}",
            score=float(10 - index),
            parent_chunk_id=f"parent-{index}",
        )
        for index in range(10)
    ]
    original_ranking = [item.citation.chunk_id for item in contexts]

    fitted = service._fit_contexts_to_token_budget(
        contexts,
        question="ordinary question",
    )

    assert [item.citation.chunk_id for item in contexts] == original_ranking
    assert [item.citation.chunk_id for item in fitted] == original_ranking
    expanded = [
        item for item in fitted if item.prompt_text.startswith("complete parent")
    ]
    assert len({item.parent_chunk_id for item in expanded}) == 6
    assert all(item.prompt_text == item.citation.excerpt for item in fitted[6:])


def test_table_answer_expansion_surfaces_adjacent_children_as_separate_citations() -> None:
    db = make_session()
    add_document(db)
    db.add_all(
        [
            chunk(
                "table-parent",
                "Complete oversized table parent",
                chunk_role="parent",
                block_type="table",
            ),
            chunk(
                "table-prev",
                "Table 2\n| Method | F1 |\n| Baseline | 80.0 |",
                parent_chunk_id="table-parent",
                block_type="table",
                ordinal=1,
                next_chunk_id="table-hit",
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 0}],
            ),
            chunk(
                "table-hit",
                "Table 2\n| Method | F1 |\n| SAC-KG | 91.2 |",
                parent_chunk_id="table-parent",
                block_type="table",
                ordinal=2,
                previous_chunk_id="table-prev",
                next_chunk_id="table-next",
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 1}],
            ),
            chunk(
                "table-next",
                "Table 2\n| Method | F1 |\n| Ablation | 87.4 |",
                parent_chunk_id="table-parent",
                block_type="table",
                ordinal=3,
                previous_chunk_id="table-hit",
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 2}],
            ),
        ]
    )
    db.commit()
    service = service_for(db)
    service.DRAFT_CONTEXT_TOKEN_BUDGET = 10_000
    service._retrieval_token_counter = lambda text: len(text.split())
    hit = db.get(DocumentChunk, "table-hit")
    context = service._expand_child_hit(
        hit,
        question="What are the Table 2 F1 results?",
        score=10.0,
        page_fields={},
        evidence_kind="table",
    )

    fitted = service._fit_contexts_to_token_budget(
        [context],
        question="What are the Table 2 F1 results?",
    )

    assert [item.citation.chunk_id for item in fitted] == [
        "table-hit",
        "table-prev",
        "table-next",
    ]
    assert [item.citation.excerpt for item in fitted] == [
        hit.text,
        db.get(DocumentChunk, "table-prev").text,
        db.get(DocumentChunk, "table-next").text,
    ]
    assert [item.citation.source_spans[0]["row_index"] for item in fitted] == [1, 0, 2]


def test_table_sibling_aggregation_is_scoped_by_table_id_and_parse_version() -> None:
    db = make_session()
    add_document(db, active_parse_version="staged-v")
    db.add_all(
        [
            chunk(
                "table-2-hit",
                "Table 2\n| Method | F1 |\n| SAC-KG | 91.2 |",
                parse_version="staged-v",
                block_type="table",
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 1}],
            ),
            chunk(
                "table-2-sibling",
                "Table 2\n| Method | F1 |\n| Baseline | 80.0 |",
                parse_version="staged-v",
                block_type="table",
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 0}],
            ),
            chunk(
                "table-3-sibling",
                "Table 3\n| Method | F1 |\n| Other | 70.0 |",
                parse_version="staged-v",
                block_type="table",
                source_spans=[{"page_index": 2, "table_id": "table-3", "row_index": 0}],
            ),
            chunk(
                "table-2-old-version",
                "Table 2\n| Method | F1 |\n| Old | 60.0 |",
                parse_version="old-v",
                block_type="table",
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 0}],
            ),
        ]
    )
    db.commit()

    service = service_for(db)
    contexts = service._search_source_chunks(
        "What are the Table 2 F1 results?", "p1", ["d1"], limit=1
    )

    assert {item.citation.chunk_id for item in contexts} == {
        "table-2-hit",
        "table-2-sibling",
    }
    assert all(item.citation.table_id == "table-2" for item in contexts)
    assert all(item.citation.parse_version == "staged-v" for item in contexts)


def test_canonical_table_context_attaches_complete_facts_to_row_citations() -> None:
    db = make_session()
    add_document(db, active_parse_version="staged-v")
    db.add_all(
        [
            chunk(
                "table-2-hit",
                "Table 2\n| Method | F1 |\n| SAC-KG | 91.2 |",
                parse_version="staged-v",
                block_type="table",
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 1}],
            ),
            chunk(
                "table-2-sibling",
                "Table 2\n| Method | F1 |\n| Baseline | 80.0 |",
                parse_version="staged-v",
                block_type="table",
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 0}],
            ),
        ]
    )
    db.commit()

    contexts = service_for(db)._search_source_chunks(
        "What are the Table 2 F1 results?", "p1", ["d1"], limit=1
    )

    assert contexts
    assert all(context.table_context is not None for context in contexts)
    facts = {fact.value for context in contexts for fact in context.table_facts}
    assert facts == {"91.2", "80.0"}
    assert {item.citation.chunk_id for item in contexts} == {
        "table-2-hit",
        "table-2-sibling",
    }


def test_table_query_prioritizes_all_table_children_before_profile_terms() -> None:
    db = make_session()
    add_document(db)
    table_chunks = [
        chunk(
            f"table-row-{index}",
            f"Table 1 | Model-{index} | {index}.0",
            block_type="table",
            ordinal=index,
            source_spans=[{"page_index": 1, "table_id": "table-1", "row_index": index}],
        )
        for index in range(25)
    ]
    profile_chunks = [
        chunk(
            f"profile-{index}",
            f"Profile evidence {index} includes SPARTA and QM-MM",
            block_type="narrative",
            ordinal=10 + index,
        )
        for index in range(2)
    ]
    db.add_all([*table_chunks, *profile_chunks])
    db.commit()
    service = service_for(db)
    question = "What are the Table 1 model values?"
    contexts = [
        *[
            service._expand_child_hit(
                item,
                question=question,
                score=10.0 - item.ordinal,
                page_fields={},
                evidence_kind="table",
            )
            for item in table_chunks
        ],
        *[
            service._expand_child_hit(
                item,
                question=question,
                score=50.0 - item.ordinal,
                page_fields={},
                evidence_kind="profile-term",
            )
            for item in profile_chunks
        ],
    ]

    finalized = service._finalize_contexts(contexts, question=question)

    assert len(finalized) == 24
    assert all(item.citation.chunk_id.startswith("table-row-") for item in finalized)
    assert {item.citation.chunk_id for item in finalized} == {
        f"table-row-{index}" for index in range(24)
    }


def test_metric_query_without_canonical_table_keeps_profile_priority() -> None:
    service = service_for(make_session())
    generic_contexts = [
        RetrievedContext(
            citation=Citation(
                chunk_id=f"generic-{index}",
                score=100.0 - index,
                excerpt=f"Generic RMSE narrative evidence {index}.",
            ),
            prompt_text=f"Generic RMSE narrative evidence {index}.",
            score=100.0 - index,
        )
        for index in range(9)
    ]
    profile_context = RetrievedContext(
        citation=Citation(
            chunk_id="profile-sparta",
            score=0.1,
            excerpt="SPARTA profile evidence for the paper.",
        ),
        prompt_text="SPARTA profile evidence for the paper.",
        score=0.1,
        evidence_kind="profile-term",
    )

    finalized = service._finalize_contexts(
        [*generic_contexts, profile_context],
        question="What is the RMSE formula?",
    )

    assert len(finalized) == 8
    assert any(item.citation.chunk_id == "profile-sparta" for item in finalized)


def test_retrieve_evidence_keeps_canonical_table_children_ahead_of_profile_terms() -> None:
    db = make_session()
    document = add_document(db)
    document.raw_text = "SPARTA"
    ensure_paper_profile(document)
    table_chunks = [
        chunk(
            f"retrieval-table-row-{index}",
            "Table 1\n| Model | Value |\n| --- | --- |\n"
            f"| Model-{index} | {index}.0 |",
            block_type="table",
            ordinal=index,
            source_spans=[{"page_index": 1, "table_id": "table-1", "row_index": index}],
        )
        for index in range(25)
    ]
    profile_chunk = chunk(
        "retrieval-profile-sparta",
        "SPARTA profile evidence outside the requested table.",
        ordinal=100,
    )
    db.add_all([*table_chunks, profile_chunk])
    db.commit()

    evidence = service_for(db).retrieve_evidence(
        "project",
        "What are the Table 1 model values?",
        limit=10,
        document_id=document.id,
    )

    assert evidence.status == "ok"
    assert len(evidence.items) == 10
    assert all(item.evidence_kind == "table" for item in evidence.items)
    assert all(item.chunk_id.startswith("retrieval-table-row-") for item in evidence.items)


def test_shadow_table_retrieval_bypasses_legacy_table_metadata_and_exposes_facts() -> None:
    db = make_session()
    document = add_document(db, active_parse_version=None)
    document.metadata_json = {
        "document_intelligence": {
            "tables": [
                {
                    "page_label": "1",
                    "markdown": (
                        "Table 5\n| Model | pKa |\n| --- | --- |\n"
                        "| OPLS4 | -0.26 |"
                    ),
                }
            ]
        }
    }
    db.add_all(
        [
            chunk(
                "staged-table-row",
                "Table 5\n| Model | pKa |\n| --- | --- |\n| OPLS5 | -0.14 |",
                parse_version="staged-v",
                block_type="table",
                source_spans=[{"page_index": 2, "page_label": "3", "table_id": "table-5"}],
            ),
        ]
    )
    db.commit()

    evidence = QueryService(
        db,
        parse_version_map={document.id: "staged-v"},
    ).retrieve_evidence(
        "project",
        "What is the OPLS5 pKa value in Table 5?",
        limit=5,
        document_id=document.id,
    )

    assert evidence.items
    assert all(item.parse_version == "staged-v" for item in evidence.items)
    assert any(fact.value == "-0.14" for fact in evidence.table_facts)
    assert all("-0.26" not in item.excerpt for item in evidence.items)


def test_table_query_does_not_let_profile_table_mentions_outrank_table_children() -> None:
    db = make_session()
    add_document(db)
    table_chunks = [
        chunk(
            f"adversarial-table-row-{index}",
            "Table 1\n| Model | Value |\n| --- | --- |\n"
            f"| Model-{index} | {index}.0 |",
            block_type="table",
            ordinal=index,
            source_spans=[{"page_index": 1, "table_id": "table-1", "row_index": index}],
        )
        for index in range(25)
    ]
    profile_chunk = chunk(
        "adversarial-profile-table-mention",
        "Table 1 model values profile summary for SPARTA.",
        ordinal=100,
    )
    db.add_all([*table_chunks, profile_chunk])
    db.commit()
    service = service_for(db)
    question = "What are the Table 1 model values?"
    contexts = [
        *[
            service._expand_child_hit(
                item,
                question=question,
                score=10.0 - item.ordinal,
                page_fields={},
                evidence_kind="table",
            )
            for item in table_chunks
        ],
        service._expand_child_hit(
            profile_chunk,
            question=question,
            score=100.0,
            page_fields={},
            evidence_kind="profile-term",
        ),
    ]

    finalized = service._finalize_contexts(contexts, question=question)

    assert finalized[0].citation.chunk_id.startswith("adversarial-table-row-")


def test_metric_table_query_preserves_explicit_figure_anchor() -> None:
    db = make_session()
    add_document(db)
    table_chunks = [
        chunk(
            f"figure-metric-table-{index}",
            "Table 1\n| Model | Accuracy |\n| --- | --- |\n"
            f"| Model-{index} | {index}.0 |",
            block_type="table",
            ordinal=index,
            source_spans=[{"page_index": 1, "table_id": "table-1", "row_index": index}],
        )
        for index in range(24)
    ]
    figure_chunk = chunk(
        "figure-metric-target",
        "Figure 2 shows accuracy across the evaluation settings.",
        block_type="figure",
        ordinal=100,
        source_spans=[{"page_index": 2, "figure_id": "figure-2"}],
    )
    db.add_all([*table_chunks, figure_chunk])
    db.commit()
    service = service_for(db)
    question = "What does Figure 2 show about accuracy?"
    contexts = [
        *[
            service._expand_child_hit(
                item,
                question=question,
                score=10.0 - item.ordinal,
                page_fields={},
                evidence_kind="table",
            )
            for item in table_chunks
        ],
        service._expand_child_hit(
            figure_chunk,
            question=question,
            score=100.0,
            page_fields={},
            evidence_kind="figure",
        ),
    ]

    finalized = service._finalize_contexts(contexts, question=question)

    assert finalized[0].citation.chunk_id == "figure-metric-target"


def test_table_query_keeps_profile_summary_below_requested_table_rows() -> None:
    db = make_session()
    add_document(db)
    table_chunks = [
        chunk(
            f"profile-priority-table-{index}",
            "Table 1\n| System | AlphaL probability |\n| --- | --- |\n"
            f"| System-{index} | {index}.0 |",
            block_type="table",
            ordinal=index,
            source_spans=[{"page_index": 1, "table_id": "table-1", "row_index": index}],
        )
        for index in range(25)
    ]
    profile_chunk = chunk(
        "profile-priority-summary",
        "Table 1 compares RS peptide, FG-nucleoporin peptide, and HEWL19 "
        "alphaL probability values in the C36m summary.",
        ordinal=100,
    )
    db.add_all([*table_chunks, profile_chunk])
    db.commit()
    service = service_for(db)
    question = (
        "What are the Table 1 alphaL probability values for RS peptide, "
        "FG-nucleoporin peptide, and HEWL19?"
    )
    contexts = [
        *[
            service._expand_child_hit(
                item,
                question=question,
                score=10.0 - item.ordinal,
                page_fields={},
                evidence_kind="table",
            )
            for item in table_chunks
        ],
        service._expand_child_hit(
            profile_chunk,
            question=question,
            score=100.0,
            page_fields={},
            evidence_kind="profile-term",
        ),
    ]

    finalized = service._finalize_contexts(contexts, question=question)

    assert finalized[0].citation.chunk_id.startswith("profile-priority-table-")


def test_canonical_table_prompt_keeps_complete_child_text_without_character_window() -> None:
    db = make_session()
    service = service_for(db)
    table_text = (
        "Table 9\n| Method | F1 |\n| --- | --- |\n"
        + "\n".join(f"| Model-{index} | {index}.0 |" for index in range(500))
    )
    context = RetrievedContext(
        citation=Citation(
            score=1.0,
            excerpt=table_text,
            parse_version="canonical-v1",
            block_type="table",
        ),
        prompt_text=table_text,
        score=1.0,
        evidence_kind="table",
    )

    rendered = service._prompt_context_text("What are the Table 9 F1 results?", context)

    assert rendered == table_text
    assert "Model-499" in rendered


def test_table_sibling_expansion_obeys_token_budget_without_truncating_child() -> None:
    db = make_session()
    service = service_for(db)
    service.DRAFT_CONTEXT_TOKEN_BUDGET = 10
    service._retrieval_token_counter = lambda text: len(text.split())
    contexts = [
        RetrievedContext(
            citation=Citation(score=3.0, chunk_id="table-hit", excerpt="Table 2 hit"),
            prompt_text="Table 2 hit",
            score=3.0,
            evidence_kind="table",
        ),
        RetrievedContext(
            citation=Citation(score=2.0, chunk_id="table-long", excerpt=" ".join(["long"] * 20)),
            prompt_text=" ".join(["long"] * 20),
            score=2.0,
            evidence_kind="table",
        ),
        RetrievedContext(
            citation=Citation(score=1.0, chunk_id="table-next", excerpt="Table 2 next"),
            prompt_text="Table 2 next",
            score=1.0,
            evidence_kind="table",
        ),
    ]

    fitted = service._fit_contexts_to_token_budget(
        contexts, question="What are the Table 2 F1 results?"
    )

    assert [item.citation.chunk_id for item in fitted] == ["table-hit", "table-next"]
    assert all(item.prompt_text == item.citation.excerpt for item in fitted)


def test_table_candidate_assembly_includes_same_table_sibling_rows() -> None:
    db = make_session()
    add_document(db)
    db.add_all(
        [
            chunk(
                "table-parent-assembly",
                "Complete Table 2 parent",
                chunk_role="parent",
                block_type="table",
                ordinal=0,
            ),
            chunk(
                "table-row-a",
                "Table 2\n| Method | F1 |\n| --- | --- |\n| Baseline | 80.0 |",
                parent_chunk_id="table-parent-assembly",
                block_type="table",
                ordinal=1,
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 0}],
            ),
            chunk(
                "table-row-b",
                "Table 2\n| Method | F1 |\n| --- | --- |\n| SAC-KG | 91.2 |",
                parent_chunk_id="table-parent-assembly",
                block_type="table",
                ordinal=2,
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 1}],
            ),
            chunk(
                "table-row-c",
                "Table 2\n| Method | F1 |\n| --- | --- |\n| Ablation | 87.4 |",
                parent_chunk_id="table-parent-assembly",
                block_type="table",
                ordinal=3,
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 2}],
            ),
        ]
    )
    db.commit()
    service = service_for(db)
    hit = db.get(DocumentChunk, "table-row-b")

    contexts = service._expand_source_candidates(
        [(hit, 10.0, "table")],
        question="What are all Table 2 F1 results?",
        project_id="p1",
    )

    assert [item.citation.chunk_id for item in contexts] == [
        "table-row-b",
        "table-row-a",
        "table-row-c",
    ]
    assert [item.citation.excerpt for item in contexts] == [
        db.get(DocumentChunk, chunk_id).text
        for chunk_id in ("table-row-b", "table-row-a", "table-row-c")
    ]
    assert all(item.citation.table_id == "table-2" for item in contexts)
    assert all(item.citation.parse_version == "canonical-v1" for item in contexts)
    assert all(item.citation.source_spans for item in contexts)


def test_table_candidate_assembly_covers_requested_tables_in_one_question() -> None:
    db = make_session()
    add_document(db)
    db.add_all(
        [
            chunk(
                "table-one-row",
                "Table 1\n| Dataset | F1 |\n| --- | --- |\n| OIE2016 | 91.2 |",
                block_type="table",
                ordinal=1,
                source_spans=[{"page_index": 1, "table_id": "table-1", "row_index": 0}],
            ),
            chunk(
                "table-two-row",
                "Table 2\n| Dataset | F1 |\n| --- | --- |\n| NYT | 88.4 |",
                block_type="table",
                ordinal=2,
                source_spans=[{"page_index": 2, "table_id": "table-2", "row_index": 0}],
            ),
        ]
    )
    db.commit()
    service = service_for(db)
    first_hit = db.get(DocumentChunk, "table-one-row")

    contexts = service._expand_source_candidates(
        [(first_hit, 10.0, "table")],
        question="Compare Table 1 OIE2016 with Table 2 NYT F1 results.",
        project_id="p1",
    )

    assert {item.citation.table_id for item in contexts} == {"table-1", "table-2"}
    assert {item.citation.chunk_id for item in contexts} == {"table-one-row", "table-two-row"}


def test_table_candidate_assembly_matches_requested_table_group_across_children() -> None:
    db = make_session()
    add_document(db)
    db.add_all(
        [
            chunk(
                "table-one-hit",
                "Table 1\n| molecule | energy |\n| --- | --- |\n| butane | 6.04 |",
                block_type="table",
                ordinal=1,
                source_spans=[{"page_index": 1, "table_id": "table-1", "row_index": 0}],
            ),
            chunk(
                "table-seven-header",
                "Table 7\n| molecule | Hvap |",
                block_type="table",
                ordinal=2,
                source_spans=[{"page_index": 2, "table_id": "table-7", "row_index": 0}],
            ),
            chunk(
                "table-seven-row",
                "| methanol | 8.95 |",
                block_type="table",
                ordinal=3,
                source_spans=[{"page_index": 2, "table_id": "table-7", "row_index": 1}],
            ),
        ]
    )
    db.commit()
    service = service_for(db)
    hit = db.get(DocumentChunk, "table-one-hit")

    contexts = service._expand_source_candidates(
        [(hit, 10.0, "table")],
        question="How do butane energy and methanol Hvap compare?",
        project_id="p1",
    )

    assert {item.citation.table_id for item in contexts} == {"table-1", "table-7"}
    assert "table-seven-row" in {item.citation.chunk_id for item in contexts}


def test_finalized_table_contexts_reserve_one_child_per_requested_table_group() -> None:
    db = make_session()
    add_document(db)
    service = service_for(db)
    contexts: list[RetrievedContext] = []
    for index in range(10):
        contexts.append(
            RetrievedContext(
                citation=Citation(
                    document_id="d1",
                    chunk_id=f"table-one-{index}",
                    parse_version="canonical-v1",
                    block_type="table",
                    table_id="table-1",
                    score=100.0 - index,
                    excerpt=f"Table 1 | butane | methanol | Hvap | energy | {index}",
                ),
                prompt_text=f"Table 1 | butane | methanol | Hvap | energy | {index}",
                score=100.0 - index,
                evidence_kind="table",
            )
        )
    contexts.append(
        RetrievedContext(
            citation=Citation(
                document_id="d1",
                chunk_id="table-seven-methanol",
                parse_version="canonical-v1",
                block_type="table",
                table_id="table-7",
                score=1.0,
                excerpt="Table 7 | methanol | Hvap | 8.95",
            ),
            prompt_text="Table 7 | methanol | Hvap | 8.95",
            score=1.0,
            evidence_kind="table",
        )
    )

    finalized = service._finalize_contexts(
        contexts,
        question="How do butane energy and methanol Hvap compare?",
    )

    assert any(item.citation.table_id == "table-7" for item in finalized[:10])


def test_finalized_table_reservation_uses_entity_and_metric_group_matches() -> None:
    db = make_session()
    add_document(db)
    service = service_for(db)
    contexts: list[RetrievedContext] = []
    # An unrelated interaction-energy table has many high-scoring rows and
    # would otherwise crowd out the requested HFE/pKa/binding tables.
    for index in range(12):
        contexts.append(
            RetrievedContext(
                citation=Citation(
                    document_id="d1",
                    chunk_id=f"table-three-{index}",
                    parse_version="canonical-v1",
                    block_type="table",
                    table_id="table-3",
                    score=100.0 - index,
                    excerpt="Table 3 | interaction energy | OPLS5",
                ),
                prompt_text="Table 3 | interaction energy | OPLS5",
                score=100.0 - index,
                evidence_kind="table",
            )
        )
    for table_id, text in (
        ("table-2", "Table 2 | hydration free energies | aromatic | 0.76 | 0.46"),
        ("table-5", "Table 5 | pKa | GLU | 0.70 | 0.61"),
        ("table-7", "Table 7 | relative binding free energy | RMSE | 1.18 | 1.12"),
    ):
        contexts.append(
            RetrievedContext(
                citation=Citation(
                    document_id="d1",
                    chunk_id=f"{table_id}-row",
                    parse_version="canonical-v1",
                    block_type="table",
                    table_id=table_id,
                    score=1.0,
                    excerpt=text,
                ),
                prompt_text=text,
                score=1.0,
                evidence_kind="table",
            )
        )

    finalized = service._finalize_contexts(
        contexts,
        question="OPLS5 HFE, GLU pKa and binding RMSE compared with OPLS4",
    )

    assert {item.citation.table_id for item in finalized[:10]} >= {
        "table-2",
        "table-5",
        "table-7",
    }
    assert "table-3" not in {
        item.citation.table_id for item in finalized[:3]
    }


def test_table_shadow_assembly_keeps_staged_siblings_and_active_pointer() -> None:
    db = make_session()
    document = add_document(db, active_parse_version="active-v")
    db.add_all(
        [
            chunk(
                "active-table-row",
                "Table 2\n| Method | F1 |\n| --- | --- |\n| Active | 80.0 |",
                parse_version="active-v",
                block_type="table",
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 0}],
            ),
            chunk(
                "staged-table-row",
                "Table 2\n| Method | F1 |\n| --- | --- |\n| Staged | 91.2 |",
                parse_version="staged-v",
                block_type="table",
                source_spans=[{"page_index": 1, "table_id": "table-2", "row_index": 0}],
            ),
        ]
    )
    db.commit()
    service = QueryService(db, parse_version_map={document.id: "staged-v"})
    service.ollama = NoVectorOllama()
    staged_hit = db.get(DocumentChunk, "staged-table-row")

    contexts = service._expand_source_candidates(
        [(staged_hit, 10.0, "table")],
        question="What is Table 2 F1?",
        project_id="p1",
    )

    assert [item.citation.chunk_id for item in contexts] == ["staged-table-row"]
    assert all("Active" not in item.citation.excerpt for item in contexts)
    assert db.get(Document, document.id).active_parse_version == "active-v"


def test_canonical_table_prompt_preserves_complete_evidence_without_character_window() -> None:
    evidence = "Table 9\n| Dataset | Value |\n| --- | --- |\n" + "\n".join(
        f"| Dataset-{index} | {index}.123 |" for index in range(180)
    )
    context = RetrievedContext(
        citation=Citation(
            document_id="d1",
            chunk_id="table-long-child",
            parse_version="canonical-v1",
            block_type="table",
            table_id="table-9",
            score=1.0,
            excerpt=evidence,
        ),
        prompt_text=evidence,
        score=1.0,
        evidence_kind="table",
    )

    rendered = QueryService._prompt_context_text("What are Table 9 values?", context)

    assert rendered == evidence
    assert rendered.endswith("| Dataset-179 | 179.123 |")


def test_answer_prompt_deduplicates_parent_overlap_without_changing_citations() -> None:
    db = make_session()
    service = service_for(db)
    service.DRAFT_CONTEXT_TOKEN_BUDGET = 10_000
    service._retrieval_token_counter = lambda text: len(text.split())
    contexts = [
        RetrievedContext(
            citation=Citation(
                score=2.0,
                chunk_id="child-a",
                document_id="d1",
                block_type="narrative",
                excerpt="first exact child",
            ),
            prompt_text="alpha beta shared one two",
            context_text="alpha beta shared one two",
            score=2.0,
            parent_chunk_id="parent-a",
        ),
        RetrievedContext(
            citation=Citation(
                score=1.0,
                chunk_id="child-b",
                document_id="d1",
                block_type="narrative",
                excerpt="shared one two gamma delta",
            ),
            prompt_text="shared one two gamma delta",
            context_text="shared one two gamma delta",
            score=1.0,
            parent_chunk_id="parent-b",
        ),
    ]

    fitted = service._fit_contexts_to_token_budget(
        contexts,
        question="ordinary question",
    )

    assert [item.citation.chunk_id for item in fitted] == ["child-a", "child-b"]
    assert fitted[1].prompt_text == "gamma delta"
    assert fitted[1].citation.excerpt == "shared one two gamma delta"
    rendered = "\n\n".join(
        service._prompt_context_text("ordinary question", item) for item in fitted
    )
    assert rendered.count("shared one two") == 1
    assert "gamma delta" in rendered


def test_answer_prompt_deduplicates_cjk_overlap_with_real_token_counter() -> None:
    db = make_session()
    service = service_for(db)
    service.DRAFT_CONTEXT_TOKEN_BUDGET = 10_000
    service._retrieval_token_counter = lambda text: len(text)
    overlap = "实验结果显示该方法在全部数据集上稳定提升"
    contexts = [
        RetrievedContext(
            citation=Citation(
                score=2.0,
                chunk_id="cjk-a",
                document_id="d1",
                block_type="narrative",
                excerpt=f"前一分块{overlap}",
            ),
            prompt_text=f"前一分块{overlap}",
            context_text=f"前一分块{overlap}",
            score=2.0,
        ),
        RetrievedContext(
            citation=Citation(
                score=1.0,
                chunk_id="cjk-b",
                document_id="d1",
                block_type="narrative",
                excerpt=f"{overlap}并验证了鲁棒性",
            ),
            prompt_text=f"{overlap}并验证了鲁棒性",
            context_text=f"{overlap}并验证了鲁棒性",
            score=1.0,
        ),
    ]

    fitted = service._fit_contexts_to_token_budget(
        contexts,
        question="该方法的实验结果如何？",
    )
    rendered = "\n\n".join(
        service._prompt_context_text("该方法的实验结果如何？", item)
        for item in fitted
    )

    assert rendered.count(overlap) == 1
    assert fitted[1].citation.excerpt == f"{overlap}并验证了鲁棒性"



def test_answer_budget_fails_when_strict_tokenizer_snapshot_is_unavailable() -> None:
    def unavailable(_name: str):
        raise OSError("pinned tokenizer snapshot missing")

    db = make_session()
    service = service_for(db)
    service._retrieval_token_counter = StructuredEvidenceBuilder(
        tokenizer_name="Qwen/Qwen3-Embedding-4B",
        tokenizer_loader=unavailable,
        strict_tokenizer=True,
    ).estimate_tokens
    context = RetrievedContext(
        citation=Citation(score=1.0, excerpt="matched child"),
        prompt_text="complete parent evidence",
        score=1.0,
        parent_chunk_id="parent-1",
    )

    with pytest.raises(RuntimeError, match="cannot use a fallback token count"):
        service._fit_contexts_to_token_budget(
            [context],
            question="ordinary question",
        )


def test_active_child_condition_is_strict_for_canonical_and_nullable_for_legacy() -> None:
    db = make_session()
    canonical_null_role = QueryService._active_child_chunk_condition(
        document_active_version=literal("canonical-v1"),
        chunk_parse_version=literal("canonical-v1"),
        chunk_role=literal(None),
        block_type=literal("narrative"),
    )
    legacy_null_fields = QueryService._active_child_chunk_condition(
        document_active_version=literal(None),
        chunk_parse_version=literal("legacy"),
        chunk_role=literal(None),
        block_type=literal(None),
    )

    assert db.scalar(select(literal(1)).where(canonical_null_role)) is None
    assert db.scalar(select(literal(1)).where(legacy_null_fields)) == 1


def test_context_budget_counts_join_separators_in_complete_draft_prompt() -> None:
    db = make_session()
    service = service_for(db)
    service.DRAFT_CONTEXT_TOKEN_BUDGET = 11
    service._retrieval_token_counter = (
        lambda text: len(text.split()) + text.count("\n\n") * 2
    )
    contexts = [
        RetrievedContext(
            citation=Citation(score=2.0, excerpt="alpha beta gamma delta"),
            prompt_text="alpha beta gamma delta",
            score=2.0,
        ),
        RetrievedContext(
            citation=Citation(score=1.0, excerpt="epsilon zeta eta theta"),
            prompt_text="epsilon zeta eta theta",
            score=1.0,
        ),
    ]

    fitted = service._fit_contexts_to_token_budget(
        contexts,
        question="ordinary question",
    )

    individual_cost = sum(
        service._retrieval_token_counter(
            f"[{index}] {service._prompt_context_text('ordinary question', context)}"
        )
        for index, context in enumerate(contexts)
    )
    complete_cost = service._retrieval_token_counter(
        "\n\n".join(
            f"[{index}] {service._prompt_context_text('ordinary question', context)}"
            for index, context in enumerate(contexts)
        )
    )
    assert individual_cost == 10
    assert complete_cost == 12
    assert len(fitted) == 1


def test_metadata_only_figure_span_preserves_typed_id_without_fake_location() -> None:
    canonical = CanonicalDocument(
        document_id="d1",
        parser_source="test",
        parse_version="canonical-v1",
        blocks=[
            CanonicalBlock(
                block_id="figure-block-not-id",
                block_type="figure",
                text="",
                reading_order=0,
                parser_source="test",
                figure_id="figure-real",
                source_spans=[],
            )
        ],
        figures=[
            CanonicalFigure(
                figure_id="figure-real",
                caption="Figure 7: No locator",
                asset_path="assets/figure-seven.png",
                source_spans=[],
            )
        ],
        assets=[
            CanonicalAsset(
                asset_id="asset-real",
                path="assets/figure-seven.png",
                media_type="image/png",
            )
        ],
    )
    draft = next(
        item
        for item in SemanticChunker(
            embedder=NoVectorOllama(),
            token_counter=lambda text: len(text.split()),
            parent_min_tokens=1,
            parent_target_tokens=20,
            parent_max_tokens=100,
            child_min_tokens=1,
            child_target_tokens=20,
            child_max_tokens=100,
            overlap_tokens=0,
        ).build(canonical)
        if item.chunk_role == "child" and item.block_type == "figure"
    )
    pipeline_context = type(
        "PipelineContext",
        (),
        {
            "document": type("PipelineDocument", (), {"id": "d1"})(),
            "version": type("PipelineVersion", (), {"version_key": "canonical-v1"})(),
        },
    )()
    persisted = IngestionPipeline._document_chunk_from_draft(
        pipeline_context,
        draft,
        embedding=None,
        parent_chunk_id=draft.parent_local_id,
    )
    citation = service_for(make_session())._expand_child_hit(
        persisted,
        question="Figure 7",
        score=1.0,
        page_fields={},
        evidence_kind="figure",
        related_chunks={},
    ).citation

    assert citation.figure_id == "figure-real"
    assert citation.asset_id == "asset-real"
    assert len(citation.source_spans) == 1
    derived_span = citation.source_spans[0]
    assert derived_span["metadata"] == {
        "figure_id": "figure-real",
        "asset_id": "asset-real",
    }
    assert all(
        derived_span.get(field) is None
        for field in (
            "page_index",
            "source_block_id",
            "paragraph_id",
            "table_id",
            "image_relationship_id",
            "xpath",
            "css_selector",
            "element_id",
            "line_start",
            "char_start",
        )
    )
    assert IngestionPipeline._chunk_has_valid_source_spans(persisted) is False


def test_metadata_only_table_span_preserves_typed_id_without_fake_location() -> None:
    annotated = SemanticChunker._annotate_structure_spans(
        [SourceSpan(metadata={"parser": "mineru"})],
        table_id="table-real",
    )
    record = chunk(
        "table-child",
        "Table 5: Results\n\n| Dataset | F1 |\n| --- | --- |\n| OIE2016 | 91.2 |",
        block_type="table",
        source_spans=[span.model_dump(mode="json") for span in annotated],
    )
    citation = service_for(make_session())._expand_child_hit(
        record,
        question="OIE2016 results",
        score=1.0,
        page_fields={},
        evidence_kind="table",
        related_chunks={},
    ).citation

    assert citation.table_id == "table-real"
    assert record.source_spans[0]["table_id"] is None
    assert record.source_spans[0]["metadata"] == {
        "parser": "mineru",
        "table_id": "table-real",
    }
    assert IngestionPipeline._chunk_has_valid_source_spans(record) is False


def test_existing_table_locator_is_preserved_and_remains_valid() -> None:
    annotated = SemanticChunker._annotate_structure_spans(
        [
            SourceSpan(
                table_id="source-table-locator",
                row_index=2,
                column_index=1,
                metadata={"parser": "mineru"},
            )
        ],
        table_id="table-real",
    )
    record = chunk(
        "table-child",
        "OIE2016 | 91.2",
        block_type="table",
        source_spans=[span.model_dump(mode="json") for span in annotated],
    )

    assert record.source_spans[0]["table_id"] == "source-table-locator"
    assert record.source_spans[0]["row_index"] == 2
    assert record.source_spans[0]["column_index"] == 1
    assert record.source_spans[0]["metadata"] == {"parser": "mineru"}
    assert IngestionPipeline._chunk_has_valid_source_spans(record) is True
