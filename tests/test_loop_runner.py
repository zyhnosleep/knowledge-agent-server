from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_mineru_rag_loop


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_profile_command_sets(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        calls.append(argv)
        if argv == ["git", "status", "--short"]:
            return _completed(stdout=" M README.md\n")
        assert encoding == "utf-8"
        assert errors == "replace"
        return _completed(stdout="ok\n")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)

    expected_pytest_counts = {"quick": 1, "query": 2, "full": 3}
    expected_last_tests = {
        "quick": ["tests/test_parser_document_intelligence.py", "tests/test_query_eval.py"],
        "query": ["tests/test_query_service.py", "tests/test_paper_profile.py"],
        "full": ["tests/test_pipeline_sac_kg.py", "tests/test_vector_retrieval.py", "tests/test_wiki_quality.py"],
    }

    for profile, expected_count in expected_pytest_counts.items():
        calls.clear()
        args = run_mineru_rag_loop.parse_args(
            ["--profile", profile, "--python", "py", "--out-dir", str(tmp_path / profile)]
        )
        manifest = run_mineru_rag_loop.run_loop(args)
        command_argvs = [command["argv"] for command in manifest["commands"]]

        assert manifest["overall_status"] == "passed"
        assert manifest["git_status_short"] == "M README.md"
        assert len(command_argvs) == expected_count
        assert all(argv[:3] == ["py", "-m", "pytest"] for argv in command_argvs)
        assert command_argvs[-1][-1] == "-q"
        for test_path in expected_last_tests[profile]:
            assert test_path in command_argvs[-1]
        assert calls[0] == ["git", "status", "--short"]


def test_query_eval_is_only_run_when_explicitly_enabled(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text("[]", encoding="utf-8")

    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        calls.append(argv)
        if argv == ["git", "status", "--short"]:
            return _completed()
        if argv[1:] and argv[1] == "scripts/query_eval.py":
            output_path = Path(argv[argv.index("--output") + 1])
            output_path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "total": 2,
                            "selected": 1,
                            "completed": 1,
                            "remaining": 0,
                            "passed": 1,
                            "failed": 0,
                        },
                        "cases": [],
                    }
                ),
                encoding="utf-8",
            )
            markdown_path = Path(argv[argv.index("--markdown") + 1])
            markdown_path.write_text("# report\n", encoding="utf-8")
            return _completed(stdout="query eval ok\n")
        if argv[1:] and argv[1] == "scripts/query_report_summary.py":
            json_output = Path(argv[argv.index("--json-output") + 1])
            json_output.write_text(
                json.dumps(
                    {
                        "summary": {"passed": 1, "failed": 0},
                        "failure_reason_counts": {},
                        "cases": [{"id": "case-a", "status": "pass", "likely_stage": "unknown"}],
                    }
                ),
                encoding="utf-8",
            )
            return _completed(stdout="Summary: {'passed': 1}\n", stderr="summary stderr\n")
        return _completed(stdout="pytest ok\n")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)

    no_eval_args = run_mineru_rag_loop.parse_args(["--out-dir", str(tmp_path / "no-eval"), "--python", "py"])
    no_eval_manifest = run_mineru_rag_loop.run_loop(no_eval_args)
    assert [command["name"] for command in no_eval_manifest["commands"]] == ["pytest_quick_1"]
    assert no_eval_manifest["mineru_smoke"] is None

    eval_args = run_mineru_rag_loop.parse_args(
        [
            "--out-dir",
            str(tmp_path / "eval"),
            "--python",
            "py",
            "--run-query-eval",
            "--base-url",
            "http://api.example",
            "--benchmark",
            str(benchmark),
            "--case-id",
            "case-a",
            "--case-id",
            "case-b",
            "--offset",
            "2",
            "--limit",
            "3",
            "--timeout",
            "12",
            "--fail-on-query-error",
        ]
    )
    eval_manifest = run_mineru_rag_loop.run_loop(eval_args)

    query_eval = next(command for command in eval_manifest["commands"] if command["name"] == "query_eval")
    summary = next(command for command in eval_manifest["commands"] if command["name"] == "query_report_summary")
    assert query_eval["argv"] == [
        "py",
        "scripts/query_eval.py",
        str(benchmark),
        "--base-url",
        "http://api.example",
        "--output",
        str(tmp_path / "eval" / "query_eval.json"),
        "--markdown",
        str(tmp_path / "eval" / "query_eval.md"),
        "--timeout",
        "12",
        "--offset",
        "2",
        "--limit",
        "3",
        "--case-id",
        "case-a",
        "--case-id",
        "case-b",
        "--fail-on-error",
    ]
    assert summary["argv"] == [
        "py",
        "scripts/query_report_summary.py",
        str(tmp_path / "eval" / "query_eval.json"),
        "--benchmark",
        str(benchmark),
        "--json-output",
        str(tmp_path / "eval" / "query_attribution.json"),
    ]
    assert eval_manifest["query_eval"]["summary"]["passed"] == 1
    assert eval_manifest["query_eval"]["attribution_summary"] == {
        "case_count": 1,
        "failure_reason_counts": {},
        "likely_stage_counts": {"unknown": 1},
        "failed_likely_stage_counts": {},
    }
    assert eval_manifest["query_eval"]["gate"] == {
        "enabled": False,
        "min_query_passed": None,
        "max_query_failed": None,
        "require_failure_attribution": False,
        "passed": None,
        "checks": {},
    }
    assert Path(eval_manifest["query_eval"]["summary_stdout_path"]).read_text(encoding="utf-8").startswith("Summary:")
    assert Path(eval_manifest["query_eval"]["summary_stderr_path"]).read_text(encoding="utf-8") == "summary stderr\n"


def test_query_eval_gate_thresholds_can_fail_or_pass_loop(monkeypatch, tmp_path) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text("[]", encoding="utf-8")

    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        if argv == ["git", "status", "--short"]:
            return _completed()
        if argv[1:] and argv[1] == "scripts/query_eval.py":
            output_path = Path(argv[argv.index("--output") + 1])
            output_path.write_text(
                json.dumps({"summary": {"total": 3, "selected": 3, "completed": 3, "remaining": 0, "passed": 2, "failed": 1}, "cases": []}),
                encoding="utf-8",
            )
            markdown_path = Path(argv[argv.index("--markdown") + 1])
            markdown_path.write_text("# report\n", encoding="utf-8")
            return _completed()
        if argv[1:] and argv[1] == "scripts/query_report_summary.py":
            output_path = Path(argv[argv.index("--json-output") + 1])
            output_path.write_text(
                json.dumps(
                    {
                        "summary": {"passed": 2, "failed": 1},
                        "failure_reason_counts": {"missing_expected_answer_text": 1},
                        "cases": [
                            {
                                "id": "case-fail",
                                "status": "fail",
                                "likely_stage": "answer",
                                "stage_reasons": {"missing_expected_answer_text": "answer"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return _completed()
        return _completed()

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    failing_args = run_mineru_rag_loop.parse_args(
        [
            "--out-dir",
            str(tmp_path / "failing"),
            "--python",
            "py",
            "--run-query-eval",
            "--benchmark",
            str(benchmark),
            "--min-query-passed",
            "3",
            "--max-query-failed",
            "0",
            "--require-failure-attribution",
        ]
    )
    passing_args = run_mineru_rag_loop.parse_args(
        [
            "--out-dir",
            str(tmp_path / "passing"),
            "--python",
            "py",
            "--run-query-eval",
            "--benchmark",
            str(benchmark),
            "--min-query-passed",
            "2",
            "--max-query-failed",
            "1",
            "--require-failure-attribution",
        ]
    )

    failing_manifest = run_mineru_rag_loop.run_loop(failing_args)
    passing_manifest = run_mineru_rag_loop.run_loop(passing_args)

    assert failing_manifest["overall_status"] == "failed"
    assert failing_manifest["query_eval"]["gate"]["checks"] == {
        "min_query_passed": False,
        "max_query_failed": False,
        "require_failure_attribution": True,
    }
    assert passing_manifest["overall_status"] == "passed"
    assert passing_manifest["query_eval"]["gate"]["passed"] is True


def test_mineru_smoke_is_only_run_when_explicitly_enabled(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        calls.append(argv)
        if argv == ["git", "status", "--short"]:
            return _completed()
        if argv[1:] and argv[1] == "scripts/mineru_parser_smoke.py":
            output_path = Path(argv[argv.index("--output") + 1])
            output_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "parser_mode": "pdf_mineru",
                        "chunks": 3,
                        "page_outputs": 2,
                        "tables": 1,
                        "figures": 1,
                        "formulas": 1,
                        "content_list_path": "mineru/auto/content_list_v2.json",
                        "output_dir": "mineru/auto",
                    }
                ),
                encoding="utf-8",
            )
            return _completed(stdout="smoke ok\n")
        return _completed(stdout="pytest ok\n")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)

    default_args = run_mineru_rag_loop.parse_args(["--out-dir", str(tmp_path / "default"), "--python", "py"])
    default_manifest = run_mineru_rag_loop.run_loop(default_args)
    assert all(command["name"] != "mineru_parser_smoke" for command in default_manifest["commands"])

    smoke_args = run_mineru_rag_loop.parse_args(
        [
            "--out-dir",
            str(tmp_path / "smoke"),
            "--python",
            "py",
            "--run-mineru-smoke",
            "--mineru-smoke-pdf",
            str(pdf_path),
            "--mineru-smoke-start",
            "1",
            "--mineru-smoke-end",
            "2",
            "--mineru-bin",
            "mineru",
            "--mineru-output-dir",
            str(tmp_path / "mineru-output"),
            "--mineru-model-source",
            "modelscope",
            "--timeout",
            "90",
        ]
    )
    smoke_manifest = run_mineru_rag_loop.run_loop(smoke_args)

    smoke_command = next(command for command in smoke_manifest["commands"] if command["name"] == "mineru_parser_smoke")
    assert smoke_command["argv"] == [
        "py",
        "scripts/mineru_parser_smoke.py",
        "--pdf",
        str(pdf_path),
        "--output",
        str(tmp_path / "smoke" / "mineru_parser_smoke.json"),
        "--mineru-output-dir",
        str(tmp_path / "mineru-output"),
        "--timeout",
        "90",
        "--start",
        "1",
        "--end",
        "2",
        "--mineru-bin",
        "mineru",
        "--mineru-model-source",
        "modelscope",
    ]
    assert smoke_manifest["overall_status"] == "passed"
    assert smoke_manifest["mineru_smoke"]["summary"]["parser_mode"] == "pdf_mineru"
    assert smoke_manifest["mineru_smoke"]["summary"]["tables"] == 1


def test_mineru_smoke_failure_marks_loop_failed(monkeypatch, tmp_path) -> None:
    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        if argv == ["git", "status", "--short"]:
            return _completed()
        if argv[1:] and argv[1] == "scripts/mineru_parser_smoke.py":
            output_path = Path(argv[argv.index("--output") + 1])
            output_path.write_text(
                json.dumps({"status": "failed", "parser_mode": None, "error": "MinerU parser returned no parsed document."}),
                encoding="utf-8",
            )
            return _completed(returncode=1, stderr="smoke failed\n")
        return _completed(stdout="pytest ok\n")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    args = run_mineru_rag_loop.parse_args(
        ["--out-dir", str(tmp_path), "--python", "py", "--run-mineru-smoke", "--mineru-smoke-pdf", str(tmp_path / "paper.pdf")]
    )

    manifest = run_mineru_rag_loop.run_loop(args)

    assert manifest["overall_status"] == "failed"
    assert manifest["mineru_smoke"]["summary"]["status"] == "failed"
    assert "no parsed document" in manifest["mineru_smoke"]["summary"]["error"]


def test_mineru_smoke_failed_summary_marks_loop_failed_even_with_zero_exit(monkeypatch, tmp_path) -> None:
    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        if argv == ["git", "status", "--short"]:
            return _completed()
        if argv[1:] and argv[1] == "scripts/mineru_parser_smoke.py":
            output_path = Path(argv[argv.index("--output") + 1])
            output_path.write_text(
                json.dumps({"status": "failed", "parser_mode": None, "error": "summary-level failure"}),
                encoding="utf-8",
            )
            return _completed(returncode=0, stdout="bad summary\n")
        return _completed(stdout="pytest ok\n")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    args = run_mineru_rag_loop.parse_args(
        ["--out-dir", str(tmp_path), "--python", "py", "--run-mineru-smoke", "--mineru-smoke-pdf", str(tmp_path / "paper.pdf")]
    )

    manifest = run_mineru_rag_loop.run_loop(args)

    assert manifest["overall_status"] == "failed"
    assert manifest["mineru_smoke"]["summary"]["error"] == "summary-level failure"


def test_mineru_smoke_missing_status_marks_loop_failed(monkeypatch, tmp_path) -> None:
    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        if argv == ["git", "status", "--short"]:
            return _completed()
        if argv[1:] and argv[1] == "scripts/mineru_parser_smoke.py":
            output_path = Path(argv[argv.index("--output") + 1])
            output_path.write_text(json.dumps({}), encoding="utf-8")
            return _completed(returncode=0, stdout="empty summary\n")
        return _completed(stdout="pytest ok\n")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    args = run_mineru_rag_loop.parse_args(
        ["--out-dir", str(tmp_path), "--python", "py", "--run-mineru-smoke", "--mineru-smoke-pdf", str(tmp_path / "paper.pdf")]
    )

    manifest = run_mineru_rag_loop.run_loop(args)

    assert manifest["overall_status"] == "failed"
    assert manifest["mineru_smoke"]["summary"]["status"] is None


def test_mineru_smoke_missing_summary_marks_loop_failed(monkeypatch, tmp_path) -> None:
    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        if argv == ["git", "status", "--short"]:
            return _completed()
        return _completed(returncode=0, stdout="no summary written\n")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    args = run_mineru_rag_loop.parse_args(
        ["--out-dir", str(tmp_path), "--python", "py", "--run-mineru-smoke", "--mineru-smoke-pdf", str(tmp_path / "paper.pdf")]
    )

    manifest = run_mineru_rag_loop.run_loop(args)

    assert manifest["overall_status"] == "failed"
    assert manifest["mineru_smoke"]["summary"]["error"] == "MinerU smoke summary file was not written."


def test_mineru_smoke_requires_pdf_argument() -> None:
    try:
        run_mineru_rag_loop.parse_args(["--run-mineru-smoke"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("--run-mineru-smoke should require --mineru-smoke-pdf")


def test_service_ingest_smoke_is_only_run_when_explicitly_enabled(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    calls: list[str] = []

    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        if argv == ["git", "status", "--short"]:
            return _completed()
        return _completed(stdout="pytest ok\n")

    def fake_service_ingest(args):
        calls.append(str(args.service_ingest_pdf))
        return {
            "status": "passed",
            "evidence": {
                "document_id": "d1",
                "run_id": "r1",
                "document_status": "ready",
                "parser_mode": "pdf_mineru",
                "page_output_count": 1,
                "content_list_path": "mineru/auto/content_list_v2.json",
            },
            "checks": {
                "document_ready": True,
                "mineru_parser": True,
                "page_outputs_present": True,
                "content_list_present": True,
                "lint_reachable": True,
            },
        }

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    monkeypatch.setattr(run_mineru_rag_loop, "_run_service_ingest_smoke", fake_service_ingest)

    default_args = run_mineru_rag_loop.parse_args(["--out-dir", str(tmp_path / "default"), "--python", "py"])
    default_manifest = run_mineru_rag_loop.run_loop(default_args)
    assert default_manifest["service_ingest"] is None
    assert calls == []

    service_args = run_mineru_rag_loop.parse_args(
        [
            "--out-dir",
            str(tmp_path / "service"),
            "--python",
            "py",
            "--run-service-ingest-smoke",
            "--service-ingest-pdf",
            str(pdf_path),
            "--service-ingest-project-slug",
            "mineru-rag-smoke",
            "--service-ingest-project-name",
            "MinerU RAG Smoke",
            "--service-ingest-timeout",
            "120",
            "--service-ingest-poll-interval",
            "1",
            "--service-ingest-http-timeout",
            "7",
        ]
    )
    service_manifest = run_mineru_rag_loop.run_loop(service_args)

    assert calls == [str(pdf_path)]
    assert service_manifest["overall_status"] == "passed"
    assert service_manifest["service_ingest"]["summary"]["status"] == "passed"
    assert service_manifest["service_ingest"]["summary"]["evidence"]["parser_mode"] == "pdf_mineru"
    assert (tmp_path / "service" / "service_ingest_smoke.json").is_file()
    assert "Service Ingest Smoke" in (tmp_path / "service" / "manifest.md").read_text(encoding="utf-8")


def test_service_ingest_smoke_can_run_without_pytest(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        if argv == ["git", "status", "--short"]:
            return _completed()
        raise AssertionError("pytest should be skipped")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    monkeypatch.setattr(run_mineru_rag_loop, "_run_service_ingest_smoke", lambda args: {"status": "passed"})
    args = run_mineru_rag_loop.parse_args(
        [
            "--skip-pytest",
            "--out-dir",
            str(tmp_path),
            "--run-service-ingest-smoke",
            "--service-ingest-pdf",
            str(pdf_path),
        ]
    )

    manifest = run_mineru_rag_loop.run_loop(args)

    assert manifest["commands"] == []
    assert manifest["overall_status"] == "passed"
    assert manifest["service_ingest"]["summary"]["status"] == "passed"


def test_service_ingest_dry_run_does_not_upload(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        raise AssertionError("dry-run must not execute subprocesses")

    def fake_service_ingest(args):
        raise AssertionError("dry-run must not upload to the service")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    monkeypatch.setattr(run_mineru_rag_loop, "_run_service_ingest_smoke", fake_service_ingest)
    args = run_mineru_rag_loop.parse_args(
        [
            "--skip-pytest",
            "--dry-run",
            "--out-dir",
            str(tmp_path),
            "--run-service-ingest-smoke",
            "--service-ingest-pdf",
            str(pdf_path),
        ]
    )

    manifest = run_mineru_rag_loop.run_loop(args)

    assert manifest["overall_status"] == "dry_run"
    assert manifest["service_ingest"]["summary"]["status"] == "dry_run"


def test_service_ingest_failed_summary_marks_loop_failed(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        if argv == ["git", "status", "--short"]:
            return _completed()
        return _completed(stdout="pytest ok\n")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_mineru_rag_loop,
        "_run_service_ingest_smoke",
        lambda args: {"status": "failed", "error": "Service ingest smoke failed checks: mineru_parser"},
    )
    args = run_mineru_rag_loop.parse_args(
        ["--out-dir", str(tmp_path), "--python", "py", "--run-service-ingest-smoke", "--service-ingest-pdf", str(pdf_path)]
    )

    manifest = run_mineru_rag_loop.run_loop(args)

    assert manifest["overall_status"] == "failed"
    assert manifest["service_ingest"]["summary"]["status"] == "failed"
    assert "mineru_parser" in manifest["service_ingest"]["summary"]["error"]


def test_service_ingest_requires_pdf_argument() -> None:
    try:
        run_mineru_rag_loop.parse_args(["--run-service-ingest-smoke"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("--run-service-ingest-smoke should require --service-ingest-pdf")


def test_run_service_ingest_smoke_collects_quality_and_lint(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    calls: list[str] = []

    def fake_upload(base_url, pdf, *, project_slug, project_name, timeout):
        assert base_url == "http://api.example/api"
        assert pdf == pdf_path
        assert project_slug == "mineru-rag-smoke"
        assert project_name == "MinerU RAG Smoke"
        assert timeout == 8
        return {"status_code": 200, "payload": {"document_id": "d1", "run_id": "r1", "status": "queued"}, "error": ""}

    def fake_wait(base_url, *, run_id, timeout, interval, http_timeout):
        assert base_url == "http://api.example/api"
        assert run_id == "r1"
        assert timeout == 120
        assert interval == 1
        assert http_timeout == 8
        return {"id": "r1", "status": "completed", "document_id": "d1", "history": [{"status": "completed"}]}

    def fake_get(url, *, timeout):
        calls.append(url)
        assert timeout == 8
        if url.endswith("/documents/d1"):
            return {
                "status_code": 200,
                "payload": {
                    "id": "d1",
                    "status": "ready",
                    "metadata_json": {
                        "parser_mode": "pdf_mineru",
                        "document_intelligence": {"content_list_path": "mineru/auto/content_list_v2.json"},
                    },
                },
                "error": "",
            }
        if url.endswith("/documents/d1/quality"):
            return {
                "status_code": 200,
                "payload": {
                    "parser_mode": "pdf_mineru",
                    "page_output_count": 2,
                    "table_count": 1,
                    "structured_table_count": 1,
                    "formula_count": 1,
                    "figure_count": 1,
                    "warnings": [],
                },
                "error": "",
            }
        if url.endswith("/wiki/lint?project_slug=mineru-rag-smoke&limit=20&offset=0"):
            return {
                "status_code": 200,
                "payload": {"issue_count": 0, "returned_issue_count": 0, "issues": []},
                "error": "",
            }
        raise AssertionError(url)

    monkeypatch.setattr(run_mineru_rag_loop, "_upload_service_ingest_pdf", fake_upload)
    monkeypatch.setattr(run_mineru_rag_loop, "_wait_for_service_ingest_run", fake_wait)
    monkeypatch.setattr(run_mineru_rag_loop, "_http_get_json", fake_get)
    args = run_mineru_rag_loop.parse_args(
        [
            "--run-service-ingest-smoke",
            "--service-ingest-pdf",
            str(pdf_path),
            "--base-url",
            "http://api.example/api",
            "--service-ingest-timeout",
            "120",
            "--service-ingest-poll-interval",
            "1",
            "--service-ingest-http-timeout",
            "8",
        ]
    )

    summary = run_mineru_rag_loop._run_service_ingest_smoke(args)

    assert summary["status"] == "passed"
    assert summary["checks"] == {
        "document_ready": True,
        "mineru_parser": True,
        "page_outputs_present": True,
        "content_list_present": True,
        "lint_reachable": True,
    }
    assert summary["evidence"]["document_id"] == "d1"
    assert summary["evidence"]["page_output_count"] == 2
    assert summary["evidence"]["lint_issue_count"] == 0
    assert len(calls) == 3


def test_run_service_ingest_smoke_fails_when_parser_falls_back(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(
        run_mineru_rag_loop,
        "_upload_service_ingest_pdf",
        lambda *args, **kwargs: {"status_code": 200, "payload": {"document_id": "d1", "run_id": "r1"}, "error": ""},
    )
    monkeypatch.setattr(
        run_mineru_rag_loop,
        "_wait_for_service_ingest_run",
        lambda *args, **kwargs: {"id": "r1", "status": "completed", "document_id": "d1", "history": [{"status": "completed"}]},
    )

    def fake_get(url, *, timeout):
        if url.endswith("/documents/d1"):
            return {"status_code": 200, "payload": {"id": "d1", "status": "ready", "metadata_json": {}}, "error": ""}
        if url.endswith("/documents/d1/quality"):
            return {"status_code": 200, "payload": {"parser_mode": "pdf_vision", "page_output_count": 0}, "error": ""}
        if "/wiki/lint" in url:
            return {"status_code": 200, "payload": {"issue_count": 0, "returned_issue_count": 0}, "error": ""}
        raise AssertionError(url)

    monkeypatch.setattr(run_mineru_rag_loop, "_http_get_json", fake_get)
    args = run_mineru_rag_loop.parse_args(["--run-service-ingest-smoke", "--service-ingest-pdf", str(pdf_path)])

    summary = run_mineru_rag_loop._run_service_ingest_smoke(args)

    assert summary["status"] == "failed"
    assert summary["checks"]["mineru_parser"] is False
    assert summary["checks"]["page_outputs_present"] is False
    assert summary["checks"]["content_list_present"] is False
    assert "mineru_parser" in summary["error"]


def test_run_service_ingest_smoke_fails_duplicate_skip_by_default(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(
        run_mineru_rag_loop,
        "_upload_service_ingest_pdf",
        lambda *args, **kwargs: {"status_code": 200, "payload": {"document_id": "d1", "run_id": "r1"}, "error": ""},
    )
    monkeypatch.setattr(
        run_mineru_rag_loop,
        "_wait_for_service_ingest_run",
        lambda *args, **kwargs: {
            "id": "r1",
            "status": "completed",
            "document_id": "d1",
            "notes": "Duplicate document skipped.",
            "history": [{"status": "completed"}],
        },
    )
    monkeypatch.setattr(
        run_mineru_rag_loop,
        "_http_get_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("duplicate skip should stop before quality fetch")),
    )
    args = run_mineru_rag_loop.parse_args(["--run-service-ingest-smoke", "--service-ingest-pdf", str(pdf_path)])

    summary = run_mineru_rag_loop._run_service_ingest_smoke(args)

    assert summary["status"] == "failed"
    assert summary["duplicate_skipped"] is True
    assert "duplicate-skip" in summary["error"]


def test_run_service_ingest_smoke_can_allow_duplicate_skip(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(
        run_mineru_rag_loop,
        "_upload_service_ingest_pdf",
        lambda *args, **kwargs: {"status_code": 200, "payload": {"document_id": "d1", "run_id": "r1"}, "error": ""},
    )
    monkeypatch.setattr(
        run_mineru_rag_loop,
        "_wait_for_service_ingest_run",
        lambda *args, **kwargs: {
            "id": "r1",
            "status": "completed",
            "document_id": "d1",
            "notes": "Duplicate document skipped.",
            "history": [{"status": "completed"}],
        },
    )

    def fake_get(url, *, timeout):
        if url.endswith("/documents/d1"):
            return {
                "status_code": 200,
                "payload": {
                    "id": "d1",
                    "status": "ready",
                    "metadata_json": {
                        "parser_mode": "pdf_mineru",
                        "document_intelligence": {"content_list_path": "mineru/auto/content_list_v2.json"},
                    },
                },
                "error": "",
            }
        if url.endswith("/documents/d1/quality"):
            return {"status_code": 200, "payload": {"parser_mode": "pdf_mineru", "page_output_count": 1}, "error": ""}
        if "/wiki/lint" in url:
            return {"status_code": 200, "payload": {"issue_count": 0, "returned_issue_count": 0}, "error": ""}
        raise AssertionError(url)

    monkeypatch.setattr(run_mineru_rag_loop, "_http_get_json", fake_get)
    args = run_mineru_rag_loop.parse_args(
        ["--run-service-ingest-smoke", "--service-ingest-pdf", str(pdf_path), "--allow-service-ingest-duplicate-skip"]
    )

    summary = run_mineru_rag_loop._run_service_ingest_smoke(args)

    assert summary["status"] == "passed"
    assert summary["duplicate_skipped"] is True
    assert summary["evidence"]["duplicate_skipped"] is True


def test_wait_for_service_ingest_run_searches_later_run_pages(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get(url, *, timeout):
        calls.append(url)
        if "offset=0" in url:
            return {
                "status_code": 200,
                "payload": [{"id": f"old-{index}", "status": "completed"} for index in range(50)],
                "error": "",
            }
        if "offset=50" in url:
            return {
                "status_code": 200,
                "payload": [{"id": "target-run", "status": "completed", "document_id": "d1", "notes": None, "provider_report": {}}],
                "error": "",
            }
        raise AssertionError(url)

    monkeypatch.setattr(run_mineru_rag_loop, "_http_get_json", fake_get)

    run = run_mineru_rag_loop._wait_for_service_ingest_run(
        "http://api.example",
        run_id="target-run",
        timeout=1,
        interval=0.1,
        http_timeout=3,
    )

    assert run["status"] == "completed"
    assert run["document_id"] == "d1"
    assert any("offset=50" in call for call in calls)


def test_dry_run_writes_manifest_and_does_not_execute(monkeypatch, tmp_path, capsys) -> None:
    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        raise AssertionError("dry-run must not execute subprocesses")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    args = run_mineru_rag_loop.parse_args(
        ["--profile", "query", "--python", "py", "--out-dir", str(tmp_path), "--run-query-eval", "--dry-run"]
    )

    manifest = run_mineru_rag_loop.run_loop(args)

    captured = capsys.readouterr()
    assert "py -m pytest tests/test_parser_document_intelligence.py tests/test_query_eval.py -q" in captured.out
    assert "py scripts/query_eval.py" in captured.out
    assert manifest["overall_status"] == "dry_run"
    assert manifest["git_status_short"] == ""
    assert all(command["exit_code"] is None for command in manifest["commands"])
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["overall_status"] == "dry_run"
    assert (tmp_path / "manifest.md").is_file()


def test_failure_status_is_recorded(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        calls.append(argv)
        if argv == ["git", "status", "--short"]:
            return _completed()
        if argv[:3] == ["py", "-m", "pytest"] and "tests/test_query_service.py" in argv:
            return _completed(returncode=2, stderr="pytest failed\n")
        return _completed(stdout="ok\n")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    args = run_mineru_rag_loop.parse_args(["--profile", "query", "--python", "py", "--out-dir", str(tmp_path)])

    manifest = run_mineru_rag_loop.run_loop(args)

    assert manifest["overall_status"] == "failed"
    assert [command["exit_code"] for command in manifest["commands"]] == [0, 2]
    failed_command = manifest["commands"][1]
    assert Path(failed_command["stderr_path"]).read_text(encoding="utf-8") == "pytest failed\n"
    assert run_mineru_rag_loop.main(["--profile", "query", "--python", "py", "--out-dir", str(tmp_path / "main")]) == 1


def test_no_command_loop_is_not_marked_passed(tmp_path) -> None:
    args = run_mineru_rag_loop.parse_args(["--skip-pytest", "--out-dir", str(tmp_path)])

    manifest = run_mineru_rag_loop.run_loop(args)

    assert manifest["commands"] == []
    assert manifest["overall_status"] == "no_commands"
    assert run_mineru_rag_loop.main(["--skip-pytest", "--out-dir", str(tmp_path / "main")]) == 1


def test_subprocess_start_failure_is_recorded(monkeypatch, tmp_path) -> None:
    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        if argv == ["git", "status", "--short"]:
            return _completed()
        raise FileNotFoundError("missing executable")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    args = run_mineru_rag_loop.parse_args(["--python", "missing-python", "--out-dir", str(tmp_path)])

    manifest = run_mineru_rag_loop.run_loop(args)

    assert manifest["overall_status"] == "failed"
    command = manifest["commands"][0]
    assert command["exit_code"] == 127
    assert "missing executable" in Path(command["stderr_path"]).read_text(encoding="utf-8")
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["overall_status"] == "failed"


def test_git_status_failure_is_recorded_without_aborting(monkeypatch, tmp_path) -> None:
    def fake_run(argv, capture_output, text, check, encoding=None, errors=None):
        if argv == ["git", "status", "--short"]:
            raise FileNotFoundError("missing git")
        return _completed(stdout="ok\n")

    monkeypatch.setattr(run_mineru_rag_loop.subprocess, "run", fake_run)
    args = run_mineru_rag_loop.parse_args(["--python", "py", "--out-dir", str(tmp_path)])

    manifest = run_mineru_rag_loop.run_loop(args)

    assert manifest["overall_status"] == "passed"
    assert "missing git" in manifest["git_status_short"]
