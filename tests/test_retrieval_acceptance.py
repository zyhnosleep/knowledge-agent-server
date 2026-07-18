from __future__ import annotations

from app.schemas.agent import EvidenceItem, EvidencePack
from scripts import build_retrieval_acceptance as acceptance
from scripts.build_retrieval_acceptance import build_report


def test_run_cases_measures_retrieval_without_generating_answers(monkeypatch) -> None:
    limits: list[int] = []

    class StubRAG:
        def answer(self, *args, **kwargs):
            raise AssertionError("retrieval acceptance must not generate answers")

        def retrieve_evidence(
            self, db, project_slug, question, *, limit, document_id=None
        ) -> EvidencePack:
            limits.append(limit)
            return EvidencePack(
                status="ok",
                items=[
                    EvidenceItem(
                        index=0,
                        document_id=question,
                        page_label="1",
                    )
                ],
            )

    monkeypatch.setattr(acceptance, "RAGAdapter", StubRAG)
    cases = [
        {
            "id": f"q{index}",
            "question": f"document-{index}",
            "expected_document_ids": [f"document-{index}"],
        }
        for index in range(20)
    ]

    report = acceptance.run_cases(object(), cases, project_slug="demo", top_k=5)

    assert report["summary"]["recall_at_k"] == 1.0
    assert report["summary"]["citation_validity"] == 1.0
    assert limits == [5] * 20


def test_report_calculates_recall_citation_validity_and_p95() -> None:
    cases = [
        {
            "id": f"q{index}",
            "question": f"question {index}",
            "expected_document_ids": ["d1" if index < 10 else "d2"],
        }
        for index in range(20)
    ]

    def query(case):
        if int(case["id"][1:]) < 10:
            return {
                "retrieved_document_ids": ["d1", "d3"],
                "citations": [{"document_id": "d1", "page_label": "2"}],
                "latency_ms": 100,
            }
        return {
            "retrieved_document_ids": ["d3"],
            "citations": [{"document_id": "d3", "page_label": None}],
            "latency_ms": 300,
        }

    report = build_report(cases, query, top_k=5)

    assert report["summary"]["recall_at_k"] == 0.5
    assert report["summary"]["citation_validity"] == 0.5
    assert report["summary"]["p95_latency_ms"] == 300
    assert report["cases"][0]["passed"] is True
    assert report["cases"][10]["passed"] is False


def test_report_requires_exactly_twenty_fixed_cases() -> None:
    try:
        build_report([], lambda case: {}, top_k=5)
    except ValueError as exc:
        assert "20" in str(exc)
    else:
        raise AssertionError("acceptance report must require exactly 20 cases")
