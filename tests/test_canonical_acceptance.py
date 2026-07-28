from __future__ import annotations

import json
from pathlib import Path

from app.schemas.agent import EvidenceItem, EvidencePack
from app.schemas.common import Citation, QueryResponse
from scripts import evaluate_canonical_retrieval as acceptance


def _item(
    *,
    document_id: str = "d1",
    chunk_id: str = "c1",
    block_type: str = "narrative",
    excerpt: str = "SAC-KG reports OIE2016 F1 74.7 and AUC 73.2.",
    context_text: str = "Generated contextual prefix\n\nSource parent text",
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
        page_label=page_label,
        parse_version="canonical-v1-abc",
        source_spans=source_spans
        or [{"page_index": 7, "page_label": page_label, "normalized_bbox": [0.1, 0.2, 0.8, 0.3]}],
    )


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

    assert report["cases"][0]["required_terms_passed"] is False
    assert "required_terms" in report["cases"][0]["failures"]
    assert report["cases"][0]["prefix_exclusion_passed"] is True


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
        "parse_completeness": 1.0,
        "contextual_prefix_completeness": 1.0,
        "embedding_completeness": 0.9,
        "pgvector_completeness": 1.0,
        "table_validation_rate": 1.0,
        "source_span_validity": 1.0,
        "artifact_link_validity": 1.0,
    }
    retrieval = {"summary": {"strict_pass": True}, "cases": []}

    result = acceptance.apply_strict_gates(integrity, retrieval)

    assert result["strict_pass"] is False
    assert result["failed_gates"] == ["embedding_completeness"]


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
