from __future__ import annotations

from scripts.build_retrieval_acceptance import build_report


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
