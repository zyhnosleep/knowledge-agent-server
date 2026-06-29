from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import query_report_summary


def test_json_output_includes_term_presence_sources_and_likely_stage(tmp_path, capsys) -> None:
    report_path = tmp_path / "report.json"
    benchmark_path = tmp_path / "benchmark.json"
    json_output_path = tmp_path / "attribution.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": {"total": 2, "selected": 2, "completed": 2, "remaining": 0, "passed": 0, "failed": 2},
                "cases": [
                    {
                        "id": "case-answer",
                        "project_slug": "paper-a",
                        "status": "fail",
                        "failure_reasons": ["missing_expected_answer_text", "missing_citation_text"],
                        "answer": "The answer includes Alpha [0].",
                        "citations": [
                            {
                                "page_slug": "sources/paper-a",
                                "page_title": "Paper A",
                                "page_kind": "source_summary",
                                "document_id": "doc-1",
                                "chunk_id": "chunk-1",
                                "page_label": "5",
                                "score": 0.9,
                                "excerpt": "Table 5 reports 88.8.",
                            }
                        ],
                    },
                    {
                        "id": "case-source",
                        "project_slug": "paper-b",
                        "status": "fail",
                        "failure_reasons": ["wrong_source_hint", "cross_paper_contamination"],
                        "answer": "Answer [0]",
                        "citations": [
                            {
                                "page_slug": "sources/actual-paper",
                                "page_title": "Actual Paper",
                                "page_kind": "source_summary",
                                "page_label": "2",
                                "excerpt": "Evidence.",
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    benchmark_path.write_text(
        json.dumps(
            [
                {
                    "id": "case-answer",
                    "project_slug": "paper-a",
                    "question": "q",
                    "answer_must_include": ["Alpha", "Beta"],
                    "citation_must_include": "Table 5",
                    "citation_excerpt_must_include": ["88.8", "SAC-KG ChatGPT"],
                    "citation_excerpt_groups_must_include": [["Table 5", "88.8"]],
                    "citation_source_hint": "paper-a",
                },
                {
                    "id": "case-source",
                    "project_slug": "paper-b",
                    "question": "q",
                    "citation_source_hint": "expected-paper",
                },
            ]
        ),
        encoding="utf-8",
    )

    exit_code = query_report_summary.main(
        [str(report_path), "--benchmark", str(benchmark_path), "--json-output", str(json_output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Summary:" in captured.out
    assert "Failure reasons:" in captured.out
    assert "case-answer: fail" in captured.out
    assert "missing_answer: ['Beta']" in captured.out
    assert "missing_citation_excerpt: ['SAC-KG ChatGPT']" in captured.out

    attribution = json.loads(json_output_path.read_text(encoding="utf-8"))
    assert attribution["summary"]["failed"] == 2
    assert attribution["failure_reason_counts"] == {
        "missing_expected_answer_text": 1,
        "missing_citation_text": 1,
        "wrong_source_hint": 1,
        "cross_paper_contamination": 1,
    }

    answer_case = attribution["cases"][0]
    assert answer_case["likely_stage"] == "citation"
    assert answer_case["answer_expected_terms"] == {"present": ["Alpha"], "missing": ["Beta"]}
    assert answer_case["citation_expected_terms"]["present"] == ["Table 5", "88.8"]
    assert answer_case["citation_expected_terms"]["missing"] == ["SAC-KG ChatGPT"]
    assert answer_case["citation_excerpt_groups"] == [
        {"terms": ["Table 5", "88.8"], "matched": True, "matching_citation_indexes": [0]}
    ]
    assert answer_case["citation_sources"][0]["page_slug"] == "sources/paper-a"
    assert answer_case["citation_sources"][0]["page_label"] == "5"
    assert answer_case["citation_source_hint"]["matched"] is True
    assert "sources/paper-a" in answer_case["citation_source_hint"]["matching_values"]

    source_case = attribution["cases"][1]
    assert source_case["likely_stage"] == "source"
    assert source_case["citation_source_hint"]["matched"] is False
    assert source_case["citation_source_hint"]["expected"] == ["expected-paper"]
    assert "sources/actual-paper" in source_case["citation_source_hint"]["available_values"]


def test_citation_group_attribution_requires_one_excerpt_to_contain_group(tmp_path) -> None:
    report = {
        "summary": {"failed": 1},
        "cases": [
            {
                "id": "split-group",
                "status": "fail",
                "failure_reasons": ["missing_citation_text"],
                "answer": "answer",
                "citations": [
                    {"excerpt": "Table 7 reports an aggregate row."},
                    {"excerpt": "The values include 1.18 and 1.12."},
                ],
            }
        ],
    }
    expected = {
        "split-group": {
            "citation_excerpt_must_include": ["Table 7", "1.18", "1.12"],
            "citation_excerpt_groups_must_include": [["Table 7", "1.18", "1.12"]],
        }
    }

    attribution = query_report_summary.build_attribution(report, expected)
    case = attribution["cases"][0]

    assert case["citation_expected_terms"]["missing"] == []
    assert case["citation_excerpt_groups"] == [
        {"terms": ["Table 7", "1.18", "1.12"], "matched": False, "matching_citation_indexes": []}
    ]


def test_term_presence_is_case_insensitive() -> None:
    report = {
        "summary": {},
        "cases": [
            {
                "id": "case-folding",
                "status": "pass",
                "failure_reasons": [],
                "answer": "The answer mentions Alpha and beta.",
                "citations": [{"excerpt": "table 1 reports Gamma."}],
            }
        ],
    }
    expected = {
        "case-folding": {
            "answer_must_include": ["alpha", "BETA"],
            "citation_excerpt_must_include": ["Table 1", "gamma"],
        }
    }

    case = query_report_summary.build_attribution(report, expected)["cases"][0]

    assert case["answer_expected_terms"] == {"present": ["alpha", "BETA"], "missing": []}
    assert case["citation_expected_terms"] == {"present": ["Table 1", "gamma"], "missing": []}


def test_build_attribution_likely_stage_mapping() -> None:
    report = {
        "summary": {},
        "cases": [
            {"id": "http", "status": "fail", "failure_reasons": ["http_error"]},
            {"id": "source", "status": "fail", "failure_reasons": ["wrong_source_hint"]},
            {"id": "table", "status": "fail", "failure_reasons": ["citation_not_table"]},
            {"id": "answer", "status": "fail", "failure_reasons": ["forbidden_answer_text"]},
            {"id": "chinese", "status": "fail", "failure_reasons": ["non_chinese_answer"]},
            {"id": "unsupported", "status": "fail", "failure_reasons": ["unsupported_claim"]},
            {"id": "unknown", "status": "pass", "failure_reasons": []},
        ],
    }

    attribution = query_report_summary.build_attribution(report, {})

    assert [case["likely_stage"] for case in attribution["cases"]] == [
        "http",
        "source",
        "table",
        "answer",
        "chinese",
        "unsupported",
        "unknown",
    ]


def test_passed_cases_are_still_hidden_from_stdout_by_default(tmp_path, capsys) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": {"passed": 1, "failed": 0},
                "cases": [{"id": "case-pass", "status": "pass", "failure_reasons": [], "answer": "ok", "citations": []}],
            }
        ),
        encoding="utf-8",
    )

    query_report_summary.main([str(report_path)])

    captured = capsys.readouterr()
    assert "Summary:" in captured.out
    assert "case-pass" not in captured.out
