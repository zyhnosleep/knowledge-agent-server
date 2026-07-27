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
from app.services.semantic_chunking import SemanticChunker
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
