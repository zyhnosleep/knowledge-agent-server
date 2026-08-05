from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

from app.schemas.agent import EvidenceItem, EvidencePack, TableFactEvidence
from app.schemas.common import Citation, QueryResponse
from app.services.rag_adapter import RAGAdapter
from scripts import evaluate_canonical_retrieval as acceptance


def _item(
    *,
    document_id: str = "d1",
    chunk_id: str = "c1",
    block_type: str = "narrative",
    excerpt: str = "SAC-KG reports OIE2016 F1 74.7 and AUC 73.2.",
    context_text: str = "Generated contextual prefix\n\nSource parent text",
    parent_chunk_id: str | None = None,
    page_label: str = "8",
    source_spans: list[dict] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        index=0,
        document_id=document_id,
        chunk_id=chunk_id,
        block_type=block_type,
        excerpt=excerpt,
        context_text=context_text,
        parent_chunk_id=parent_chunk_id,
        page_label=page_label,
        parse_version="canonical-v1-abc",
        source_spans=source_spans
        or [{"page_index": 7, "page_label": page_label, "normalized_bbox": [0.1, 0.2, 0.8, 0.3]}],
    )


def test_run_acceptance_routes_shadow_versions_to_integrity_retrieval_and_answer(
    monkeypatch,
) -> None:
    observed: list[tuple[str, dict[str, str] | None]] = []
    integrity_kwargs: dict[str, object] = {}

    class FakeRAGAdapter:
        def retrieve_evidence(self, *_args, parse_version_map=None, **_kwargs):
            observed.append(("retrieve", parse_version_map))
            return EvidencePack(status="ok", items=[_item(document_id="d1")])

        def answer(self, *_args, parse_version_map=None, **_kwargs):
            observed.append(("answer", parse_version_map))
            return QueryResponse(
                answer_markdown="staged answer",
                citations=[
                    Citation(
                        document_id="d1",
                        score=1.0,
                        excerpt="staged evidence",
                    )
                ],
                verification_status="ok",
            )

    def fake_integrity(*_args, parse_version_map=None, **kwargs):
        observed.append(("integrity", parse_version_map))
        integrity_kwargs.update(kwargs)
        return {name: 1.0 for name in acceptance.STRICT_INTEGRITY_GATES}

    monkeypatch.setattr(acceptance, "RAGAdapter", FakeRAGAdapter)
    monkeypatch.setattr(acceptance, "collect_integrity_metrics", fake_integrity)
    monkeypatch.setattr(
        acceptance,
        "_observe_active_parse_version_map",
        lambda _db, _document_ids: {"d1": "active-v"},
    )
    shadow = {"d1": "staged-v"}
    cases = [
        {
            "id": "shadow",
            "category": "narrative",
            "project_slug": "demo",
            "question": "What changed?",
            "expected_document_id": "d1",
            "require_answer": True,
            "answer_required_terms": ["staged answer"],
        }
    ]

    report = acceptance.run_acceptance(
        db=object(),  # type: ignore[arg-type]
        cases=cases,
        parse_version_map=shadow,
        expected_ingestion_config={"config": "test"},
        expected_ingestion_config_sha256="a" * 64,
    )

    assert report["parse_version_map"] == shadow
    assert report["active_parse_version_map"] == {"d1": "active-v"}
    assert report["route_identity"] == "candidate"
    assert report["retrieval"]["summary"]["route_identity"] == "candidate"
    assert integrity_kwargs["document_ids"] == {"d1"}
    assert integrity_kwargs["expected_ingestion_config"] == {"config": "test"}
    integrity_call = next(item for item in observed if item[0] == "integrity")
    assert integrity_call == ("integrity", shadow)
    assert observed == [
        ("integrity", shadow),
        ("retrieve", shadow),
        ("answer", shadow),
    ]


def test_run_acceptance_validates_active_versions_against_live_ingestion_config(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}
    snapshot = {"semantic_splitting": {"child_tokens": {"max": 600}}}

    class FakeRAGAdapter:
        pass

    def fake_integrity(*_args, **kwargs):
        observed.update(kwargs)
        return {name: 1.0 for name in acceptance.STRICT_INTEGRITY_GATES}

    monkeypatch.setattr(acceptance, "RAGAdapter", FakeRAGAdapter)
    monkeypatch.setattr(acceptance, "build_ingestion_config_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        acceptance,
        "canonical_ingestion_config_hash",
        lambda value: "b" * 64 if value is snapshot else "unexpected",
    )
    monkeypatch.setattr(acceptance, "collect_integrity_metrics", fake_integrity)
    monkeypatch.setattr(
        acceptance,
        "_observe_active_parse_version_map",
        lambda _db, _document_ids: {"d1": "active-v"},
    )
    monkeypatch.setattr(
        acceptance,
        "evaluate_cases",
        lambda *_args, **_kwargs: {"summary": {"strict_pass": True}, "cases": []},
    )

    report = acceptance.run_acceptance(db=object(), cases=[])

    assert observed["expected_ingestion_config"] is snapshot
    assert observed["expected_ingestion_config_sha256"] == "b" * 64
    assert report["strict_pass"] is True
    assert report["route_identity"] == "active"
    assert report["active_parse_version_map"] == {"d1": "active-v"}
    assert report["parse_version_map"] == {}


def test_observe_active_parse_version_map_reads_database_pointers() -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db.session import Base
    from app.models.records import Document, Project

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, expire_on_commit=False, future=True)()
    db.add(Project(id="p1", slug="project", name="Project"))
    db.add_all(
        [
            Document(
                id="d1",
                project_id="p1",
                title="A",
                file_name="a.pdf",
                sha256="s1",
                raw_path="raw/a.pdf",
                status="ready",
                active_parse_version="active-v",
            ),
            Document(
                id="d2",
                project_id="p1",
                title="B",
                file_name="b.pdf",
                sha256="s2",
                raw_path="raw/b.pdf",
                status="ready",
                active_parse_version="other-v",
            ),
            Document(
                id="d3",
                project_id="p1",
                title="C",
                file_name="c.pdf",
                sha256="s3",
                raw_path="raw/c.pdf",
                status="ready",
                active_parse_version=None,
            ),
        ]
    )
    db.commit()

    assert acceptance._observe_active_parse_version_map(db) == {
        "d1": "active-v",
        "d2": "other-v",
    }
    assert acceptance._observe_active_parse_version_map(db, {"d2"}) == {
        "d2": "other-v"
    }


def test_evaluate_cases_reports_required_anchor_recall_and_missing_anchors() -> None:
    cases = [
        {
            "id": "anchor-recall",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "Find the anchors.",
            "expected_document_id": "d1",
            "required_terms": ["ALPHA", "BETA", "GAMMA"],
        }
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        items = [_item(excerpt="ALPHA BETA present.")]
        items += [
            _item(excerpt="unrelated.", chunk_id=f"c{index}")
            for index in range(1, 6)
        ]
        items.append(_item(excerpt="GAMMA present.", chunk_id="c6"))
        return EvidencePack(status="ok", items=items)

    report = acceptance.evaluate_cases(cases, retrieve)
    row = report["cases"][0]
    summary = report["summary"]

    assert row["required_anchor_coverage_at_5"] == 2 / 3
    assert row["required_anchor_coverage_at_10"] == 1.0
    assert row["missing_required_terms_at_5"] == ["GAMMA"]
    assert row["missing_required_terms_at_10"] == []
    assert row["recall_at_5"] is True
    assert summary["retrieval_only"]["required_anchor_recall_at_5"] == round(
        2 / 3, 6
    )
    assert summary["retrieval_only"]["required_anchor_recall_at_10"] == 1.0
    assert summary["retrieval_only"]["required_anchor_case_count"] == 1
    assert summary["retrieval_only"]["full_anchor_recall_at_5_cases"] == 0
    assert summary["retrieval_only"]["full_anchor_recall_at_10_cases"] == 1


def test_required_anchor_recall_excludes_cases_without_anchors() -> None:
    cases = [
        {
            "id": "anchor-miss",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "Find the anchors.",
            "expected_document_id": "d1",
            "required_terms": ["MISSING_ANCHOR"],
        },
        {
            "id": "no-anchor",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "No anchors requested.",
            "expected_document_id": "d1",
        },
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        return EvidencePack(
            status="ok", items=[_item(excerpt="Plain narrative evidence.")]
        )

    report = acceptance.evaluate_cases(cases, retrieve)
    anchor_miss, no_anchor = report["cases"]
    summary = report["summary"]["retrieval_only"]

    assert anchor_miss["required_anchor_count"] == 1
    assert anchor_miss["required_anchor_checked"] is True
    assert anchor_miss["required_anchor_coverage_at_5"] == 0.0
    assert anchor_miss["required_anchor_coverage_at_10"] == 0.0
    # A case without required anchors is represented as unchecked/null rather
    # than as a fabricated perfect recall.
    assert no_anchor["required_anchor_count"] == 0
    assert no_anchor["required_anchor_checked"] is False
    assert no_anchor["required_anchor_coverage_at_5"] is None
    assert no_anchor["required_anchor_coverage_at_10"] is None
    # Aggregate recall is computed only over anchor-bearing cases.
    assert summary["required_anchor_case_count"] == 1
    assert summary["required_anchor_recall_at_5"] == 0.0
    assert summary["required_anchor_recall_at_10"] == 0.0
    assert summary["full_anchor_recall_at_5_cases"] == 0
    assert summary["full_anchor_recall_at_10_cases"] == 0


def test_required_anchor_recall_is_null_when_no_anchors_checked() -> None:
    cases = [
        {
            "id": "no-anchor-a",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "No anchors A.",
            "expected_document_id": "d1",
        },
        {
            "id": "no-anchor-b",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "No anchors B.",
            "expected_document_id": "d1",
        },
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        return EvidencePack(status="ok", items=[_item(excerpt="Plain narrative.")])

    summary = acceptance.evaluate_cases(cases, retrieve)["summary"]["retrieval_only"]

    assert summary["required_anchor_case_count"] == 0
    assert summary["required_anchor_recall_at_5"] is None
    assert summary["required_anchor_recall_at_10"] is None
    assert summary["full_anchor_recall_at_5_cases"] == 0
    assert summary["full_anchor_recall_at_10_cases"] == 0


def test_evaluate_cases_separates_retrieval_and_full_answer_layers() -> None:
    cases = [
        {
            "id": "answer-gap",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "What component?",
            "expected_document_id": "d1",
            "required_terms": ["Generator"],
            "require_answer": True,
            "answer_required_terms": ["ANSWER_WORD"],
        }
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        return EvidencePack(
            status="ok", items=[_item(excerpt="Generator is a component.")]
        )

    def answer(_case: dict) -> QueryResponse:
        return QueryResponse(
            answer_markdown="The answer omits the required word.",
            verification_status="verified",
            citations=[
                Citation(
                    document_id="d1",
                    score=1.0,
                    excerpt="Generator is a component.",
                )
            ],
        )

    report = acceptance.evaluate_cases(cases, retrieve, answer)
    row = report["cases"][0]
    summary = report["summary"]

    assert row["retrieval_gate_passed"] is True
    assert row["answer_passed"] is False
    assert row["citation_passed"] is True
    assert row["missing_answer_terms"] == ["ANSWER_WORD"]
    assert row["failed_layers"] == ["full_answer"]
    assert row["answer_latency_ms"] >= 0
    assert summary["retrieval_only"]["strict_retrieval_pass"] is True
    assert summary["retrieval_only"]["retrieval_gate_pass_rate"] == 1.0
    assert summary["full_answer"]["strict_full_answer_pass"] is False
    assert summary["full_answer"]["case_count"] == 1
    assert summary["full_answer"]["answer_pass_rate"] == 0.0
    assert summary["full_answer"]["citation_pass_rate"] == 1.0
    assert summary["full_answer"]["missing_answer_terms"] == ["ANSWER_WORD"]
    assert summary["full_answer"]["p50_answer_latency_ms"] >= 0
    assert summary["full_answer"]["p95_answer_latency_ms"] >= 0
    assert summary["strict_pass"] is False
    assert summary["failed_layers"] == ["full_answer"]


def test_strict_acceptance_fails_when_citation_fails_despite_retrieval_passing() -> None:
    cases = [
        {
            "id": "wrong-source-citation",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "What component?",
            "expected_document_id": "d1",
            "required_terms": ["Generator"],
            "require_answer": True,
            "answer_required_terms": ["Generator"],
        }
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        return EvidencePack(
            status="ok", items=[_item(excerpt="Generator is a component.")]
        )

    def answer(_case: dict) -> QueryResponse:
        return QueryResponse(
            answer_markdown="Generator is a component.",
            verification_status="verified",
            citations=[
                Citation(
                    document_id="d2",
                    score=0.9,
                    excerpt="Generator is a component.",
                )
            ],
        )

    report = acceptance.evaluate_cases(cases, retrieve, answer)
    row = report["cases"][0]

    assert row["retrieval_gate_passed"] is True
    assert row["answer_passed"] is True
    assert row["citation_passed"] is False
    assert row["failed_layers"] == ["full_answer"]
    assert report["summary"]["strict_pass"] is False
    assert report["summary"]["failed_layers"] == ["full_answer"]


def test_evaluate_cases_rejects_empty_answer_despite_retrieval_passing() -> None:
    cases = [
        {
            "id": "empty-answer",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "What component?",
            "expected_document_id": "d1",
            "require_answer": True,
        }
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        return EvidencePack(
            status="ok", items=[_item(excerpt="Generator is a component.")]
        )

    def answer(_case: dict) -> QueryResponse:
        return QueryResponse(
            answer_markdown="",
            verification_status="verified",
            citations=[
                Citation(document_id="d1", score=1.0, excerpt="component")
            ],
        )

    row = acceptance.evaluate_cases(cases, retrieve, answer)["cases"][0]

    assert row["retrieval_gate_passed"] is True
    assert row["answer_passed"] is False
    assert row["citation_passed"] is True
    assert "answer" in row["failures"]
    assert row["failed_layers"] == ["full_answer"]


def test_evaluate_cases_rejects_wrong_version_evidence_in_candidate_route() -> None:
    cases = [
        {
            "id": "wrong-version",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "What changed?",
            "expected_document_id": "d1",
            "required_terms": ["Generator"],
        }
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        item = _item(excerpt="Generator staged evidence.")
        item.parse_version = "old-v"
        return EvidencePack(status="ok", items=[item])

    report = acceptance.evaluate_cases(
        cases,
        retrieve,
        route_identity="candidate",
        effective_parse_version_map={"d1": "staged-v"},
    )
    row = report["cases"][0]

    assert row["recall_at_10"] is True
    assert "parse_version" in row["failures"]
    assert row["retrieval_gate_passed"] is False
    assert row["failed_layers"] == ["retrieval"]
    assert report["summary"]["strict_pass"] is False


def test_evaluate_cases_accepts_candidate_route_evidence_from_the_right_version() -> None:
    cases = [
        {
            "id": "right-version",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "What changed?",
            "expected_document_id": "d1",
            "required_terms": ["Generator"],
        }
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        item = _item(excerpt="Generator staged evidence.")
        item.parse_version = "staged-v"
        return EvidencePack(status="ok", items=[item])

    report = acceptance.evaluate_cases(
        cases,
        retrieve,
        route_identity="candidate",
        effective_parse_version_map={"d1": "staged-v"},
    )
    row = report["cases"][0]

    assert row["retrieval_gate_passed"] is True
    assert "parse_version" not in row["failures"]
    assert report["summary"]["route_identity"] == "candidate"
    assert report["summary"]["strict_pass"] is True


def test_apply_strict_gates_surfaces_failed_layers() -> None:
    retrieval = {
        "summary": {
            "strict_pass": False,
            "failed_layers": ["full_answer"],
        },
        "cases": [],
    }
    integrity = {name: 1.0 for name in acceptance.STRICT_INTEGRITY_GATES}

    result = acceptance.apply_strict_gates(integrity, retrieval)

    assert result["failed_gates"] == ["retrieval_cases"]
    assert result["failed_layers"] == ["full_answer"]
    assert result["strict_pass"] is False


def test_rag_adapter_constructs_query_service_with_shadow_versions(
    monkeypatch,
) -> None:
    observed: list[dict[str, str] | None] = []

    class FakeQueryService:
        def __init__(self, _db, *, parse_version_map=None) -> None:
            observed.append(parse_version_map)

        def retrieve_evidence(self, *_args, **_kwargs):
            return EvidencePack(status="ok", items=[])

        def answer(self, *_args, **_kwargs):
            return QueryResponse(
                answer_markdown="",
                citations=[],
                verification_status="ok",
            )

    import app.services.rag_adapter as adapter_module

    monkeypatch.setattr(adapter_module, "QueryService", FakeQueryService)
    shadow = {"d1": "staged-v"}
    adapter = RAGAdapter()

    adapter.retrieve_evidence(
        object(),  # type: ignore[arg-type]
        "demo",
        "question",
        parse_version_map=shadow,
    )
    adapter.answer(
        object(),  # type: ignore[arg-type]
        "demo",
        "question",
        parse_version_map=shadow,
    )

    assert observed == [shadow, shadow]


def test_acceptance_measures_recall_location_and_prefix_exclusion() -> None:
    cases = [
        {
            "id": "table-metrics",
            "category": "long_table",
            "project_slug": "research",
            "question": "SAC-KG 在 OIE2016 上的指标是什么？",
            "expected_source_identity": "sources/sac-kg",
            "expected_block_type": "table",
            "expected_page_label": "8",
            "required_terms": ["OIE2016", "74.7", "73.2"],
            "forbidden_excerpt_terms": ["Generated contextual prefix"],
            "location_requirement": "bbox",
        }
    ]

    def retrieve(_case: dict, limit: int) -> EvidencePack:
        assert limit == 10
        item = _item(block_type="table")
        item.page_slug = "sources/sac-kg"
        return EvidencePack(status="ok", items=[item])

    report = acceptance.evaluate_cases(cases, retrieve)

    assert report["summary"]["recall_at_5"] == 1.0
    assert report["summary"]["recall_at_10"] == 1.0
    assert report["summary"]["source_location_validity"] == 1.0
    assert report["summary"]["contextual_prefix_citation_exclusion"] == 1.0
    assert report["summary"]["p50_retrieval_latency_ms"] >= 0
    assert report["summary"]["p95_retrieval_latency_ms"] >= 0
    assert report["cases"][0]["passed"] is True


def test_acceptance_gates_table_evidence_coverage_and_records_narrative_misses() -> None:
    cases = [
        {
            "id": "narrative-broad",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "Summarize the paper.",
            "expected_document_id": "d1",
            "expected_block_type": "narrative",
            "required_terms": ["ABSENT_NARRATIVE_TERM"],
        },
        {
            "id": "table-strict",
            "category": "long_table",
            "project_slug": "research",
            "question": "What is in Table 1?",
            "expected_document_id": "d1",
            "expected_block_type": "table",
            "required_terms": ["ABSENT_TABLE_TERM"],
        },
    ]

    def retrieve(case: dict, _limit: int) -> EvidencePack:
        return EvidencePack(
            status="ok",
            items=[
                _item(
                    document_id=case["expected_document_id"],
                    block_type=case["expected_block_type"],
                    excerpt="Source evidence without the requested strict term.",
                )
            ],
        )

    report = acceptance.evaluate_cases(cases, retrieve)
    narrative, table = report["cases"]

    # A narrative coverage miss is diagnostic-only: it stays visible in the
    # evidence-coverage fields but does not fail the retrieval gate.
    assert narrative["evidence_coverage_passed"] is False
    assert narrative["required_terms_passed"] is False
    assert "evidence_coverage" not in narrative["failures"]
    assert narrative["retrieval_gate_passed"] is True
    assert narrative["passed"] is True
    # Table evidence coverage is a real gate and fails the retrieval layer.
    assert table["required_terms_passed"] is False
    assert "evidence_coverage" in table["failures"]
    assert table["retrieval_gate_passed"] is False
    assert table["passed"] is False


def test_retrieval_gate_records_narrative_coverage_without_blocking_gate() -> None:
    cases = [
        {
            "id": "narrative-observational",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "Summarize the paper.",
            "expected_document_id": "d1",
            "expected_block_type": "narrative",
            "required_terms": ["ABSENT_NARRATIVE_TERM"],
        },
        {
            "id": "table-gated",
            "category": "long_table",
            "project_slug": "research",
            "question": "What is in Table 1?",
            "expected_document_id": "d1",
            "expected_block_type": "table",
            "required_terms": ["Table 1", "91.2"],
        },
    ]

    def retrieve(case: dict, _limit: int) -> EvidencePack:
        excerpt = (
            "Table 1 | Model-A | 91.2"
            if case["id"] == "table-gated"
            else "The source narrative is present."
        )
        return EvidencePack(
            status="ok",
            items=[
                _item(
                    document_id="d1",
                    block_type=case["expected_block_type"],
                    excerpt=excerpt,
                )
            ],
        )

    report = acceptance.evaluate_cases(cases, retrieve)
    narrative, table = report["cases"]

    assert narrative["evidence_coverage_passed"] is False
    assert "evidence_coverage" not in narrative["failures"]
    assert narrative["passed"] is True
    assert narrative["retrieval_gate_passed"] is True
    assert table["retrieval_gate_passed"] is True
    assert report["summary"]["strict_pass"] is True
    assert report["summary"]["passed_cases"] == 2
    assert "evidence_coverage" not in report["summary"]["failure_attribution"]


def test_acceptance_gates_figure_and_formula_evidence_coverage_by_default() -> None:
    cases = [
        {
            "id": "figure-strict",
            "category": "figure",
            "project_slug": "research",
            "question": "What does Figure 1 show?",
            "expected_document_id": "d1",
            "expected_block_type": "figure",
            "required_terms": ["ABSENT_FIGURE_TERM"],
        },
        {
            "id": "formula-strict",
            "category": "formula",
            "project_slug": "research",
            "question": "What does the formula include?",
            "expected_document_id": "d1",
            "expected_block_type": "formula",
            "required_terms": ["ABSENT_FORMULA_TERM"],
        },
    ]

    def retrieve(case: dict, _limit: int) -> EvidencePack:
        return EvidencePack(
            status="ok",
            items=[
                _item(
                    document_id="d1",
                    block_type=case["expected_block_type"],
                    excerpt="Structured evidence without the requested term.",
                )
            ],
        )

    figure, formula = acceptance.evaluate_cases(cases, retrieve)["cases"]

    for row in (figure, formula):
        assert row["evidence_coverage_passed"] is False
        assert row["required_terms_passed"] is False
        assert row["evidence_coverage_gate_required"] is True
        assert "evidence_coverage" in row["failures"]
        assert row["retrieval_gate_passed"] is False
        assert row["failed_layers"] == ["retrieval"]
        assert row["passed"] is False


def test_acceptance_explicit_flag_gates_narrative_while_unflagged_stays_diagnostic() -> None:
    cases = [
        {
            "id": "narrative-explicit-gate",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "Summarize the paper.",
            "expected_document_id": "d1",
            "expected_block_type": "narrative",
            "required_terms": ["ABSENT_TERM"],
            "evidence_coverage_required": True,
        },
        {
            "id": "narrative-diagnostic",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "Summarize the paper.",
            "expected_document_id": "d1",
            "expected_block_type": "narrative",
            "required_terms": ["ABSENT_TERM"],
        },
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        return EvidencePack(
            status="ok",
            items=[
                _item(
                    document_id="d1",
                    block_type="narrative",
                    excerpt="Source narrative without the requested term.",
                )
            ],
        )

    gated, diagnostic = acceptance.evaluate_cases(cases, retrieve)["cases"]

    # The explicit ``evidence_coverage_required`` flag turns the narrative
    # coverage miss into a retrieval gate.
    assert gated["evidence_coverage_gate_required"] is True
    assert gated["evidence_coverage_passed"] is False
    assert "evidence_coverage" in gated["failures"]
    assert gated["retrieval_gate_passed"] is False
    assert gated["failed_layers"] == ["retrieval"]
    assert gated["passed"] is False
    # Without the flag the same narrative miss stays diagnostic-only.
    assert diagnostic["evidence_coverage_gate_required"] is False
    assert diagnostic["evidence_coverage_passed"] is False
    assert "evidence_coverage" not in diagnostic["failures"]
    assert diagnostic["retrieval_gate_passed"] is True
    assert diagnostic["failed_layers"] == []
    assert diagnostic["passed"] is True


def test_acceptance_attributes_reference_and_location_failures() -> None:
    cases = [
        {
            "id": "reference-exclusion",
            "category": "reference_exclusion",
            "project_slug": "research",
            "question": "Which paper is cited?",
            "expected_source_identity": "sources/sac-kg",
            "expected_block_type": "narrative",
            "required_terms": ["SAC-KG"],
            "location_requirement": "bbox",
        }
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        item = _item(block_type="reference", source_spans=[{"page_index": 7}])
        item.page_slug = "sources/sac-kg"
        return EvidencePack(status="ok", items=[item])

    report = acceptance.evaluate_cases(cases, retrieve)

    assert report["summary"]["strict_pass"] is False
    assert report["summary"]["failure_attribution"]["reference_exclusion"] == 1
    assert "reference_exclusion" in report["cases"][0]["failures"]
    assert "source_location" in report["cases"][0]["failures"]


def test_acceptance_uses_excerpt_not_context_text_for_required_terms() -> None:
    cases = [
        {
            "id": "prefix-only-hit",
            "category": "contextual_prefix_citation_exclusion",
            "project_slug": "research",
            "question": "What metric was reported?",
            "expected_document_id": "d1",
            "required_terms": ["SECRET_PREFIX_TERM"],
            "forbidden_excerpt_terms": ["SECRET_PREFIX_TERM"],
        }
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        return EvidencePack(
            status="ok",
            items=[
                _item(
                    excerpt="The source contains no requested metric.",
                    context_text="SECRET_PREFIX_TERM\n\nThe source contains no requested metric.",
                )
            ],
        )

    report = acceptance.evaluate_cases(cases, retrieve)
    row = report["cases"][0]

    assert row["required_terms_passed"] is False
    assert row["evidence_coverage_passed"] is False
    # The narrative prefix-only miss is diagnostic-only; the prefix-exclusion
    # gate itself still passes and the case passes.
    assert "evidence_coverage" not in row["failures"]
    assert row["passed"] is True
    assert row["prefix_exclusion_passed"] is True


def test_acceptance_separates_parent_evidence_coverage_from_citation_terms() -> None:
    cases = [
        {
            "id": "parent-evidence",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "What strategy was used?",
            "expected_document_id": "d1",
            "expected_block_type": "narrative",
            "required_terms": ["ParentStrategy"],
            "citation_required_terms": ["DirectChildClaim"],
        }
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        return EvidencePack(
            status="ok",
            items=[
                _item(
                    excerpt="DirectChildClaim is supported here.",
                    context_text="The source parent explains ParentStrategy in detail.",
                    parent_chunk_id="parent-1",
                )
            ],
        )

    row = acceptance.evaluate_cases(cases, retrieve)["cases"][0]

    assert row["citation_required_terms_passed"] is True
    assert row["evidence_coverage_passed"] is True
    assert row["required_terms_passed"] is True
    assert row["passed"] is True


def test_acceptance_counts_only_source_linked_table_facts_as_evidence() -> None:
    cases = [
        {
            "id": "structured-table-evidence",
            "category": "long_table",
            "project_slug": "research",
            "question": "What does Table 1 report?",
            "expected_document_id": "d1",
            "expected_block_type": "table",
            "required_terms": ["Model-A", "91.2"],
            "citation_required_terms": ["Table 1"],
        },
        {
            "id": "unlinked-structured-table-evidence",
            "category": "long_table",
            "project_slug": "research",
            "question": "What does Table 1 report?",
            "expected_document_id": "d1",
            "expected_block_type": "table",
            "required_terms": ["Wrong-Model", "99.9"],
            "citation_required_terms": ["Table 1"],
        },
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        item = _item(
            block_type="table",
            excerpt="Table 1 benchmark results.",
        )
        item.table_id = "table-1"
        return EvidencePack(
            status="ok",
            items=[item],
            table_facts=[
                TableFactEvidence(
                    table_id="table-1",
                    document_id="d1",
                    row_label="Model-A",
                    column="F1",
                    value="91.2",
                    row_index=1,
                    source_chunk_ids=["table-row-a"],
                ),
                TableFactEvidence(
                    table_id="table-wrong",
                    document_id="d2",
                    row_label="Wrong-Model",
                    column="F1",
                    value="99.9",
                    row_index=1,
                    source_chunk_ids=["wrong-row"],
                ),
            ],
        )

    linked, unlinked = acceptance.evaluate_cases(cases, retrieve)["cases"]

    assert linked["citation_required_terms_passed"] is True
    assert linked["evidence_coverage_passed"] is True
    assert linked["passed"] is True
    assert unlinked["citation_required_terms_passed"] is True
    assert unlinked["evidence_coverage_passed"] is False
    assert unlinked["missing_required_terms"] == ["Wrong-Model", "99.9"]


def test_acceptance_matches_latex_greek_subscripts_without_relaxing_numeric_terms() -> None:
    cases = [
        {
            "id": "latex-equivalence",
            "category": "long_table",
            "project_slug": "research",
            "question": "What does Table I report?",
            "expected_document_id": "d1",
            "expected_block_type": "table",
            "required_terms": ["χ1", "χ2", "0.0"],
        },
        {
            "id": "numeric-boundary",
            "category": "long_table",
            "project_slug": "research",
            "question": "What does Table I report?",
            "expected_document_id": "d1",
            "expected_block_type": "table",
            "required_terms": ["2.35"],
        },
    ]

    def retrieve(case: dict, _limit: int) -> EvidencePack:
        excerpt = (
            r"Table I | $\chi _ { 1 }$ and $\chi _ { 2 }$ | theta0 | 0.0 |"
            if case["id"] == "latex-equivalence"
            else "Table I | value | 2.350"
        )
        return EvidencePack(
            status="ok",
            items=[_item(document_id="d1", block_type="table", excerpt=excerpt)],
        )

    latex, numeric = acceptance.evaluate_cases(cases, retrieve)["cases"]

    assert latex["evidence_coverage_passed"] is True
    assert latex["passed"] is True
    assert numeric["evidence_coverage_passed"] is False
    assert numeric["missing_required_terms"] == ["2.35"]


def test_compact_formula_matching_requires_numeric_index_boundary() -> None:
    cases = [
        {
            "id": "chi1-vs-chi10",
            "category": "long_table",
            "project_slug": "research",
            "question": "χ1?",
            "expected_document_id": "d1",
            "expected_block_type": "table",
            "required_terms": ["χ1"],
        },
        {
            "id": "c1-vs-c10",
            "category": "long_table",
            "project_slug": "research",
            "question": "C1?",
            "expected_document_id": "d1",
            "expected_block_type": "table",
            "required_terms": ["C1"],
        },
        {
            "id": "chi10-matches",
            "category": "long_table",
            "project_slug": "research",
            "question": "χ10?",
            "expected_document_id": "d1",
            "expected_block_type": "table",
            "required_terms": ["χ10"],
        },
    ]

    def retrieve(case: dict, _limit: int) -> EvidencePack:
        excerpt = {
            "chi1-vs-chi10": r"Table I | $\chi _ { 10 }$ | 0.0",
            "c1-vs-c10": "Table I | C10 | 0.0",
            "chi10-matches": r"Table I | $\chi _ { 10 }$ | 0.0",
        }[case["id"]]
        return EvidencePack(
            status="ok",
            items=[_item(document_id="d1", block_type="table", excerpt=excerpt)],
        )

    chi1, c1, chi10 = acceptance.evaluate_cases(cases, retrieve)["cases"]

    # ``χ1`` must not be satisfied by the ``chi1`` prefix of ``χ10``.
    assert chi1["evidence_coverage_passed"] is False
    assert chi1["missing_required_terms"] == ["χ1"]
    # ``C1`` must not be satisfied by the ``C1`` prefix of ``C10``.
    assert c1["evidence_coverage_passed"] is False
    assert c1["missing_required_terms"] == ["C1"]
    # The exact indexed form still matches its LaTeX rendering.
    assert chi10["evidence_coverage_passed"] is True
    assert chi10["missing_required_terms"] == []


def test_acceptance_matches_wrapped_latex_formula_subscripts() -> None:
    """Formula anchors survive a font wrapper around the Greek symbol."""

    wrapped_chi1 = r"$\mathbb { \chi } _ { 1 }$"
    wrapped_chi10 = r"$\mathbb { \chi } _ { 10 }$"

    assert acceptance._term_matches_text("\u03c71", wrapped_chi1) is True
    assert acceptance._term_matches_text("\u03c71", wrapped_chi10) is False


def test_acceptance_matches_all_common_latex_font_wrappers() -> None:
    for wrapper in ("mathrm", "mathbf", "mathsf", "text"):
        wrapped_chi2 = rf"$\{wrapper} {{ \chi }} _ {{ 2 }}$"
        assert acceptance._term_matches_text("\u03c72", wrapped_chi2) is True


def test_acceptance_matches_safe_table_formula_and_residue_aliases() -> None:
    cases = [
        {
            "id": "table-subscript-alias",
            "category": "long_table",
            "project_slug": "research",
            "question": "What is C6 in Table 1?",
            "expected_document_id": "d1",
            "expected_block_type": "table",
            "required_terms": ["C6"],
        },
        {
            "id": "residue-ocr-alias",
            "category": "long_table",
            "project_slug": "research",
            "question": "Which residue is listed?",
            "expected_document_id": "d1",
            "expected_block_type": "table",
            "required_terms": ["Ile"],
        },
    ]

    def retrieve(case: dict, _limit: int) -> EvidencePack:
        excerpt = (
            r"Table 1 | $C _ { 6 }$ | 900"
            if case["id"] == "table-subscript-alias"
            else "Table I | Res. | lle | parameters"
        )
        return EvidencePack(
            status="ok",
            items=[_item(document_id="d1", block_type="table", excerpt=excerpt)],
        )

    rows = acceptance.evaluate_cases(cases, retrieve)["cases"]

    assert all(row["evidence_coverage_passed"] for row in rows)
    assert all(row["passed"] for row in rows)


def test_acceptance_supports_stable_source_identity_aliases() -> None:
    cases = [
        {
            "id": "source-alias",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "What is SAC-KG?",
            "expected_source_identities": [
                "sources/sac-kg",
                "sources/knowledge-graph-sac-kg-framework-overview",
            ],
            "required_terms": ["SAC-KG"],
        }
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        item = _item(excerpt="SAC-KG has three components.")
        item.page_slug = "sources/sac-kg"
        return EvidencePack(status="ok", items=[item])

    report = acceptance.evaluate_cases(cases, retrieve)

    assert report["cases"][0]["passed"] is True


def test_answer_acceptance_rejects_citation_from_wrong_source() -> None:
    cases = [
        {
            "id": "grounded-answer",
            "category": "narrative_retrieval",
            "project_slug": "research",
            "question": "What are the components?",
            "expected_source_identity": "sources/sac-kg",
            "required_terms": ["Generator"],
            "require_answer": True,
            "answer_required_terms": ["Generator"],
        }
    ]

    def retrieve(_case: dict, _limit: int) -> EvidencePack:
        item = _item(excerpt="Generator is one component.")
        item.page_slug = "sources/sac-kg"
        return EvidencePack(status="ok", items=[item])

    def answer(_case: dict) -> QueryResponse:
        return QueryResponse(
            answer_markdown="Generator is one component.",
            verification_status="verified",
            citations=[
                Citation(
                    page_slug="sources/other-paper",
                    score=0.9,
                    excerpt="Generator is one component.",
                )
            ],
        )

    report = acceptance.evaluate_cases(cases, retrieve, answer)

    assert report["cases"][0]["answer_passed"] is True
    assert report["cases"][0]["citation_passed"] is False
    assert report["summary"]["answer_citation_result"] == 0.0
    assert "citation" in report["cases"][0]["failures"]


def test_strict_gate_requires_all_integrity_metrics_and_cases() -> None:
    integrity = {
        **{name: 1.0 for name in acceptance.STRICT_INTEGRITY_GATES},
        "embedding_completeness": 0.9,
        "pgvector_completeness": 1.0,
    }
    retrieval = {"summary": {"strict_pass": True}, "cases": []}

    result = acceptance.apply_strict_gates(integrity, retrieval)

    assert result["strict_pass"] is False
    assert result["failed_gates"] == ["embedding_completeness"]


def test_strict_gate_rejects_incomplete_plain_embedding_policy() -> None:
    integrity = {
        **{name: 1.0 for name in acceptance.STRICT_INTEGRITY_GATES},
        "plain_embedding_completeness": 0.0,
        "pgvector_completeness": 1.0,
    }

    result = acceptance.apply_strict_gates(
        integrity,
        {"summary": {"strict_pass": True}, "cases": []},
    )

    assert result["strict_pass"] is False
    assert result["failed_gates"] == ["plain_embedding_completeness"]


def test_case_file_contains_every_required_category() -> None:
    path = Path("docs/query_acceptance/canonical_ingestion_v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = {case["category"] for case in payload["cases"]}

    assert {
        "narrative_retrieval",
        "chinese_to_english",
        "long_table",
        "cross_page_table",
        "figure",
        "formula",
        "appendix",
        "reference_exclusion",
        "contextual_prefix_citation_exclusion",
        "bbox_location",
    } <= categories


def test_main_returns_nonzero_when_any_strict_gate_fails(
    tmp_path: Path, monkeypatch
) -> None:
    cases_path = tmp_path / "cases.json"
    report_path = tmp_path / "report.json"
    cases_path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    monkeypatch.setattr(
        acceptance,
        "run_acceptance",
        lambda **_kwargs: {
            "strict_pass": False,
            "failed_gates": ["table_validation_rate"],
        },
    )

    exit_code = acceptance.main(
        ["--cases", str(cases_path), "--report", str(report_path)]
    )

    assert exit_code == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["strict_pass"] is False


def test_main_can_atomically_activate_the_exact_shadow_batch_after_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cases_path = tmp_path / "cases.json"
    map_path = tmp_path / "staged-map.json"
    report_path = tmp_path / "report.json"
    cases_path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    shadow = {"d1": "staged-v"}
    map_path.write_text(json.dumps(shadow), encoding="utf-8")
    fake_db = object()
    observed: dict[str, object] = {}

    def fake_acceptance(**kwargs):
        observed["acceptance_kwargs"] = kwargs
        return {
            "strict_pass": True,
            "failed_gates": [],
            "parse_version_map": shadow,
        }

    def fake_activate(db, **kwargs):
        observed["activation_db"] = db
        observed["activation_kwargs"] = kwargs
        return [object()]

    monkeypatch.setattr(acceptance, "SessionLocal", lambda: nullcontext(fake_db))
    monkeypatch.setattr(acceptance, "run_acceptance", fake_acceptance)
    monkeypatch.setattr(acceptance, "activate_rebuild_batch", fake_activate)

    exit_code = acceptance.main(
        [
            "--cases",
            str(cases_path),
            "--report",
            str(report_path),
            "--parse-version-map",
            str(map_path),
            "--activate-on-success",
        ]
    )

    assert exit_code == 0
    assert observed["acceptance_kwargs"]["parse_version_map"] == shadow
    assert observed["activation_db"] is fake_db
    assert observed["activation_kwargs"]["parse_version_map"] == shadow
    assert observed["activation_kwargs"]["acceptance_report"]["strict_pass"] is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["activation"] == {
        "status": "completed",
        "activated_document_ids": ["d1"],
    }
