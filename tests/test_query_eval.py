from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import query_eval


def test_evaluate_cases_reports_expected_failure_reasons(monkeypatch) -> None:
    responses = {
        "ok": query_eval.QueryHTTPResult(
            status_code=200,
            payload={
                "answer_markdown": "\u7b54\u6848\u5305\u542b Alpha\uff0c\u4f46\u662f\u5f15\u7528\u4e86\u8d8a\u754c\u6765\u6e90 [2]\u3002",
                "citations": [
                    {
                        "page_slug": "sources/paper-a",
                        "page_title": "Paper A",
                        "page_kind": "source_summary",
                        "score": 1.0,
                        "excerpt": "This excerpt mentions Table 1 and Alpha evidence.",
                    }
                ],
                "verification_status": "local-only",
            },
            error="",
        ),
        "bad": query_eval.QueryHTTPResult(status_code=503, payload={}, error="service unavailable"),
    }

    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        assert base_url == "http://api.example"
        assert save_answer is False
        return responses[project_slug]

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    cases = [
        {
            "id": "ok-case",
            "project_slug": "ok",
            "question": "question",
            "answer_must_include": ["Alpha", "Beta"],
            "answer_must_not_include": ["forbidden"],
            "citation_must_include": ["Table 2"],
            "citation_source_hint": "paper-b",
            "require_citation": True,
            "require_chinese": True,
        },
        {
            "id": "http-case",
            "project_slug": "bad",
            "question": "question",
            "require_citation": True,
        },
    ]

    report = query_eval.evaluate_cases(cases, base_url="http://api.example", save_answer=False)

    assert report["summary"] == {"total": 2, "selected": 2, "completed": 2, "remaining": 0, "passed": 0, "failed": 2}
    assert report["cases"][0]["failure_reasons"] == [
        "missing_expected_answer_text",
        "missing_citation_text",
        "wrong_source_hint",
        "cross_paper_contamination",
        "citation_index_mismatch",
        "citation_not_table",
    ]
    assert report["cases"][1]["failure_reasons"] == ["http_error", "no_citation"]


def test_evaluate_cases_detects_forbidden_text_missing_citation_and_non_chinese(monkeypatch) -> None:
    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={"answer_markdown": "This answer contains a forbidden claim.", "citations": []},
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    report = query_eval.evaluate_cases(
        [
            {
                "id": "case-1",
                "project_slug": "demo",
                "question": "question",
                "answer_must_not_include": "forbidden claim",
                "require_citation": True,
                "require_chinese": True,
            }
        ],
        base_url="http://127.0.0.1:8000",
        save_answer=False,
    )

    assert report["cases"][0]["failure_reasons"] == [
        "no_citation",
        "forbidden_answer_text",
        "non_chinese_answer",
    ]


def test_require_chinese_rejects_mostly_english_answer_with_one_chinese_character(monkeypatch) -> None:
    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={"answer_markdown": "答案: This is otherwise an English answer with many words.", "citations": []},
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    report = query_eval.evaluate_cases(
        [
            {
                "id": "case-1",
                "project_slug": "demo",
                "question": "question",
                "require_chinese": True,
            }
        ],
        base_url="http://127.0.0.1:8000",
        save_answer=False,
    )

    assert report["cases"][0]["failure_reasons"] == ["non_chinese_answer"]


def test_require_citation_marks_answer_without_marker_as_unsupported_claim(monkeypatch) -> None:
    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={
                "answer_markdown": "答案给出了事实，但正文没有引用编号。",
                "citations": [
                    {
                        "page_slug": "sources/paper-a",
                        "page_title": "Paper A",
                        "page_kind": "source_summary",
                        "score": 1.0,
                        "excerpt": "Evidence.",
                    }
                ],
            },
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    report = query_eval.evaluate_cases(
        [
            {
                "id": "case-1",
                "project_slug": "demo",
                "question": "question",
                "require_citation": True,
            }
        ],
        base_url="http://127.0.0.1:8000",
        save_answer=False,
    )

    assert report["cases"][0]["failure_reasons"] == ["unsupported_claim"]


def test_source_hint_checks_citation_source_fields_not_excerpt(monkeypatch) -> None:
    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={
                "answer_markdown": "Answer [0]",
                "citations": [
                    {
                        "page_slug": "sources/actual-paper",
                        "page_title": "Actual Paper",
                        "page_kind": "source_summary",
                        "score": 1.0,
                        "excerpt": "The text mentions expected-paper as a comparison baseline.",
                    }
                ],
            },
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    report = query_eval.evaluate_cases(
        [
            {
                "id": "case-1",
                "project_slug": "demo",
                "question": "question",
                "citation_source_hint": "expected-paper",
            }
        ],
        base_url="http://127.0.0.1:8000",
        save_answer=False,
    )

    assert report["cases"][0]["failure_reasons"] == ["wrong_source_hint", "cross_paper_contamination"]


def test_source_hint_accepts_any_candidate(monkeypatch) -> None:
    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={
                "answer_markdown": "Answer [0]",
                "citations": [
                    {
                        "page_slug": "sources/actual-paper",
                        "page_title": "Actual Paper",
                        "page_kind": "source_summary",
                        "score": 1.0,
                        "excerpt": "Evidence.",
                    }
                ],
            },
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    report = query_eval.evaluate_cases(
        [
            {
                "id": "case-1",
                "project_slug": "demo",
                "question": "question",
                "citation_source_hint": ["missing-paper", "actual-paper"],
            }
        ],
        base_url="http://127.0.0.1:8000",
        save_answer=False,
    )

    assert report["cases"][0]["failure_reasons"] == []


def test_source_hint_does_not_accept_longer_slug_alias(monkeypatch) -> None:
    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={
                "answer_markdown": "Answer [0]",
                "citations": [
                    {
                        "page_slug": "sources/opls4-force-field-development-and-validation",
                        "page_title": "OPLS4 force field development and validation",
                        "page_kind": "source_summary",
                        "score": 1.0,
                        "excerpt": "Evidence.",
                    }
                ],
            },
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    report = query_eval.evaluate_cases(
        [
            {
                "id": "case-1",
                "project_slug": "demo",
                "question": "question",
                "citation_source_hint": "sources/opls4",
            }
        ],
        base_url="http://127.0.0.1:8000",
        save_answer=False,
    )

    assert report["cases"][0]["failure_reasons"] == ["wrong_source_hint", "source_alias_confusion"]


def test_wrong_source_hint_marks_cross_paper_contamination(monkeypatch) -> None:
    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={
                "answer_markdown": "Answer [0]",
                "citations": [
                    {
                        "page_slug": "sources/opls5-force-field-development-and-validation",
                        "page_title": "OPLS5",
                        "page_kind": "source_summary",
                        "score": 1.0,
                        "excerpt": "Evidence.",
                    }
                ],
            },
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    report = query_eval.evaluate_cases(
        [
            {
                "id": "case-1",
                "project_slug": "demo",
                "question": "question",
                "citation_source_hint": "sources/opls4",
            }
        ],
        base_url="http://127.0.0.1:8000",
        save_answer=False,
    )

    assert report["cases"][0]["failure_reasons"] == ["wrong_source_hint", "cross_paper_contamination"]


def test_citation_excerpt_must_include_checks_excerpt_only(monkeypatch) -> None:
    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={
                "answer_markdown": "The answer reports 88.8 [0]",
                "citations": [
                    {
                        "page_slug": "sources/paper-a",
                        "page_title": "Paper A 88.8",
                        "page_kind": "source_summary",
                        "score": 1.0,
                        "excerpt": "Table 5 caption without the required value.",
                    }
                ],
            },
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    report = query_eval.evaluate_cases(
        [
            {
                "id": "case-1",
                "project_slug": "demo",
                "question": "question",
                "citation_excerpt_must_include": ["Table 5", "88.8"],
            }
        ],
        base_url="http://127.0.0.1:8000",
        save_answer=False,
    )

    assert report["cases"][0]["failure_reasons"] == ["missing_citation_text"]


def test_citation_excerpt_groups_require_each_group_in_one_excerpt(monkeypatch) -> None:
    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={
                "answer_markdown": "The answer reports both values [0][1]",
                "citations": [
                    {
                        "page_slug": "sources/paper-a",
                        "page_title": "Paper A",
                        "page_kind": "source_summary",
                        "score": 1.0,
                        "excerpt": "Table 2 reports 0.76.",
                    },
                    {
                        "page_slug": "sources/paper-a",
                        "page_title": "Paper A",
                        "page_kind": "source_summary",
                        "score": 1.0,
                        "excerpt": "A nearby paragraph mentions 0.46 without Table 2.",
                    },
                ],
            },
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    report = query_eval.evaluate_cases(
        [
            {
                "id": "case-1",
                "project_slug": "demo",
                "question": "question",
                "citation_excerpt_groups_must_include": [["Table 2", "0.76", "0.46"]],
            }
        ],
        base_url="http://127.0.0.1:8000",
        save_answer=False,
    )

    assert report["cases"][0]["failure_reasons"] == ["missing_citation_text"]


def test_table_case_marks_citation_not_table(monkeypatch) -> None:
    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={
                "answer_markdown": "The answer reports 88.8 [0]",
                "citations": [
                    {
                        "page_slug": "sources/paper-a",
                        "page_title": "Paper A",
                        "page_kind": "source_summary",
                        "score": 1.0,
                        "excerpt": "Nearby prose without a table label or row.",
                    }
                ],
            },
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    report = query_eval.evaluate_cases(
        [
            {
                "id": "case-1",
                "project_slug": "demo",
                "question": "question",
                "citation_must_include": "Table 5",
                "citation_excerpt_must_include": ["Table 5", "88.8"],
            }
        ],
        base_url="http://127.0.0.1:8000",
        save_answer=False,
    )

    assert report["cases"][0]["failure_reasons"] == ["missing_citation_text", "citation_not_table"]


def test_table_case_marks_false_negative_answer(monkeypatch) -> None:
    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={
                "answer_markdown": "未包含相关表格数值 [0]",
                "citations": [
                    {
                        "page_slug": "sources/paper-a",
                        "page_title": "Paper A",
                        "page_kind": "source_summary",
                        "score": 1.0,
                        "excerpt": "Table 5 reports 88.8.",
                    }
                ],
            },
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    report = query_eval.evaluate_cases(
        [
            {
                "id": "case-1",
                "project_slug": "demo",
                "question": "question",
                "answer_must_not_include": "未包含",
                "citation_must_include": "Table 5",
            }
        ],
        base_url="http://127.0.0.1:8000",
        save_answer=False,
    )

    assert report["cases"][0]["failure_reasons"] == ["forbidden_answer_text", "table_false_negative"]


def test_post_query_posts_to_api_query_with_save_answer_false(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"answer_markdown": "ok", "citations": []}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(query_eval.urllib.request, "urlopen", fake_urlopen)

    result = query_eval.post_query("http://host.test/", "demo", "What?", save_answer=False, timeout=12)

    assert result.status_code == 200
    assert captured["url"] == "http://host.test/api/query"
    assert captured["method"] == "POST"
    assert captured["body"] == {"project_slug": "demo", "question": "What?", "save_answer": False}
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["timeout"] == 12


def test_post_query_accepts_api_base_url(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"answer_markdown": "ok", "citations": []}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr(query_eval.urllib.request, "urlopen", fake_urlopen)

    query_eval.post_query("http://host.test/api", "demo", "What?", save_answer=False, timeout=12)

    assert captured["url"] == "http://host.test/api/query"


def test_cli_writes_json_and_markdown_and_fails_on_error(tmp_path, monkeypatch) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    json_report_path = tmp_path / "report.json"
    markdown_report_path = tmp_path / "report.md"
    benchmark_path.write_text(
        json.dumps(
            [
                {
                    "id": "case-1",
                    "project_slug": "demo",
                    "question": "question",
                    "answer_must_include": "expected",
                }
            ]
        ),
        encoding="utf-8",
    )

    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(
            status_code=200,
            payload={"answer_markdown": "missing", "citations": []},
            error="",
        )

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    exit_code = query_eval.main(
        [
            str(benchmark_path),
            "--output",
            str(json_report_path),
            "--markdown",
            str(markdown_report_path),
            "--fail-on-error",
        ]
    )

    assert exit_code == 1
    json_report = json.loads(json_report_path.read_text(encoding="utf-8"))
    assert json_report["summary"]["failed"] == 1
    markdown = markdown_report_path.read_text(encoding="utf-8")
    assert "| case-1 | fail | missing_expected_answer_text |" in markdown


def test_cli_filters_cases_by_offset_limit_and_repeated_case_id(tmp_path, monkeypatch) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    report_path = tmp_path / "report.json"
    benchmark_path.write_text(
        json.dumps(
            [
                {"id": "case-1", "project_slug": "p1", "question": "q1"},
                {"id": "case-2", "project_slug": "p2", "question": "q2"},
                {"id": "case-3", "project_slug": "p3", "question": "q3"},
                {"id": "case-4", "project_slug": "p4", "question": "q4"},
            ]
        ),
        encoding="utf-8",
    )
    seen_project_slugs = []

    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        seen_project_slugs.append(project_slug)
        return query_eval.QueryHTTPResult(status_code=200, payload={"answer_markdown": "ok", "citations": []}, error="")

    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    exit_code = query_eval.main(
        [
            str(benchmark_path),
            "--case-id",
            "case-2",
            "--case-id",
            "case-4",
            "--offset",
            "1",
            "--limit",
            "1",
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert seen_project_slugs == ["p4"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["total"] == 4
    assert report["summary"]["selected"] == 1
    assert report["summary"]["completed"] == 1
    assert report["summary"]["remaining"] == 0
    assert [case["id"] for case in report["cases"]] == ["case-4"]


def test_cli_incrementally_writes_reports_after_each_case(tmp_path, monkeypatch) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    json_report_path = tmp_path / "report.json"
    markdown_report_path = tmp_path / "report.md"
    benchmark_path.write_text(
        json.dumps(
            [
                {"id": "case-1", "project_slug": "p1", "question": "q1"},
                {"id": "case-2", "project_slug": "p2", "question": "q2"},
            ]
        ),
        encoding="utf-8",
    )
    snapshots = []

    def fake_write_report(report, output_path, markdown_path):
        snapshots.append(
            {
                "summary": dict(report["summary"]),
                "cases": [case["id"] for case in report["cases"]],
                "output_path": output_path,
                "markdown_path": markdown_path,
            }
        )
        query_eval._write_report_file(report, output_path, markdown_path)

    def fake_post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float):
        return query_eval.QueryHTTPResult(status_code=200, payload={"answer_markdown": "ok", "citations": []}, error="")

    monkeypatch.setattr(query_eval, "write_report", fake_write_report)
    monkeypatch.setattr(query_eval, "post_query", fake_post_query)

    exit_code = query_eval.main(
        [
            str(benchmark_path),
            "--output",
            str(json_report_path),
            "--markdown",
            str(markdown_report_path),
        ]
    )

    assert exit_code == 0
    assert [snapshot["cases"] for snapshot in snapshots] == [["case-1"], ["case-1", "case-2"]]
    assert snapshots[0]["summary"]["completed"] == 1
    assert snapshots[0]["summary"]["remaining"] == 1
    assert snapshots[1]["summary"]["completed"] == 2
    assert snapshots[1]["summary"]["remaining"] == 0
    assert json.loads(json_report_path.read_text(encoding="utf-8"))["summary"]["completed"] == 2
    assert "| case-2 | pass | - |" in markdown_report_path.read_text(encoding="utf-8")


def test_write_report_replaces_existing_json_atomically(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    output_path.write_text("old", encoding="utf-8")
    markdown_path.write_text("old", encoding="utf-8")
    seen_replacements = []

    real_replace = Path.replace

    def tracking_replace(self, target):
        seen_replacements.append((self.name, Path(target).name))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", tracking_replace)

    query_eval.write_report(
        {
            "base_url": "http://127.0.0.1:8000",
            "save_answer": False,
            "summary": {"total": 1, "selected": 1, "completed": 1, "remaining": 0, "passed": 1, "failed": 0},
            "cases": [{"id": "case-1", "status": "pass", "failure_reasons": []}],
        },
        output_path,
        markdown_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8"))["summary"]["passed"] == 1
    assert output_path.with_name(output_path.name + ".tmp").exists() is False
    assert markdown_path.with_name(markdown_path.name + ".tmp").exists() is False
    assert ("report.json.tmp", "report.json") in seen_replacements
    assert ("report.md.tmp", "report.md") in seen_replacements


def test_load_benchmark_accepts_cases_wrapper(tmp_path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps({"cases": [{"id": "case-1", "project_slug": "demo", "question": "q"}]}), encoding="utf-8")

    assert query_eval.load_benchmark(benchmark_path) == [{"id": "case-1", "project_slug": "demo", "question": "q"}]


def test_load_benchmark_rejects_missing_required_fields(tmp_path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps([{"id": "case-1", "project_slug": "demo"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="question"):
        query_eval.load_benchmark(benchmark_path)
