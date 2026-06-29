from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK = Path("benchmarks/query/internal_research_v1.json")
DEFAULT_OUT_ROOT = Path("tmp/loop_runs")


PYTEST_COMMANDS = {
    "quick": [
        ["-m", "pytest", "tests/test_parser_document_intelligence.py", "tests/test_query_eval.py", "-q"],
    ],
    "query": [
        ["-m", "pytest", "tests/test_parser_document_intelligence.py", "tests/test_query_eval.py", "-q"],
        ["-m", "pytest", "tests/test_query_service.py", "tests/test_paper_profile.py", "-q"],
    ],
    "full": [
        ["-m", "pytest", "tests/test_parser_document_intelligence.py", "tests/test_query_eval.py", "-q"],
        ["-m", "pytest", "tests/test_query_service.py", "tests/test_paper_profile.py", "-q"],
        ["-m", "pytest", "tests/test_pipeline_sac_kg.py", "tests/test_vector_retrieval.py", "tests/test_wiki_quality.py", "-q"],
    ],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a thin MinerU/RAG regression loop.")
    parser.add_argument("--profile", choices=["quick", "query", "full"], default="quick")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child commands.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-query-eval", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--fail-on-query-error", action="store_true")
    parser.add_argument("--min-query-passed", type=int, default=None)
    parser.add_argument("--max-query-failed", type=int, default=None)
    parser.add_argument("--require-failure-attribution", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-mineru-smoke", action="store_true")
    parser.add_argument("--mineru-smoke-pdf", type=Path, default=None)
    parser.add_argument("--mineru-smoke-start", type=int, default=None)
    parser.add_argument("--mineru-smoke-end", type=int, default=None)
    parser.add_argument("--mineru-bin", default=None)
    parser.add_argument("--mineru-output-dir", type=Path, default=None)
    parser.add_argument("--mineru-model-source", default=None)
    parser.add_argument("--run-service-ingest-smoke", action="store_true")
    parser.add_argument("--service-ingest-pdf", type=Path, default=None)
    parser.add_argument("--service-ingest-project-slug", default="mineru-rag-smoke")
    parser.add_argument("--service-ingest-project-name", default="MinerU RAG Smoke")
    parser.add_argument("--service-ingest-timeout", type=float, default=900.0)
    parser.add_argument("--service-ingest-poll-interval", type=float, default=5.0)
    parser.add_argument("--service-ingest-http-timeout", type=float, default=30.0)
    parser.add_argument("--allow-service-ingest-duplicate-skip", action="store_true")
    args = parser.parse_args(argv)
    if args.run_mineru_smoke and args.mineru_smoke_pdf is None:
        parser.error("--mineru-smoke-pdf is required when --run-mineru-smoke is used.")
    if args.run_service_ingest_smoke and args.service_ingest_pdf is None:
        parser.error("--service-ingest-pdf is required when --run-service-ingest-smoke is used.")
    return args


def build_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    if not args.skip_pytest:
        for index, pytest_args in enumerate(PYTEST_COMMANDS[args.profile], start=1):
            commands.append({"name": f"pytest_{args.profile}_{index}", "argv": [args.python, *pytest_args]})

    if args.run_mineru_smoke:
        smoke_output = args.out_dir / "mineru_parser_smoke.json"
        mineru_output_dir = args.mineru_output_dir or args.out_dir / "mineru_smoke_output"
        smoke_argv = [
            args.python,
            "scripts/mineru_parser_smoke.py",
            "--pdf",
            str(args.mineru_smoke_pdf),
            "--output",
            str(smoke_output),
            "--mineru-output-dir",
            str(mineru_output_dir),
            "--timeout",
            str(int(args.timeout)),
        ]
        if args.mineru_smoke_start is not None:
            smoke_argv.extend(["--start", str(args.mineru_smoke_start)])
        if args.mineru_smoke_end is not None:
            smoke_argv.extend(["--end", str(args.mineru_smoke_end)])
        if args.mineru_bin:
            smoke_argv.extend(["--mineru-bin", args.mineru_bin])
        if args.mineru_model_source:
            smoke_argv.extend(["--mineru-model-source", args.mineru_model_source])
        commands.append({"name": "mineru_parser_smoke", "argv": smoke_argv})

    if args.run_query_eval:
        json_output = args.out_dir / "query_eval.json"
        markdown_output = args.out_dir / "query_eval.md"
        query_eval_argv = [
            args.python,
            "scripts/query_eval.py",
            str(args.benchmark),
            "--base-url",
            args.base_url,
            "--output",
            str(json_output),
            "--markdown",
            str(markdown_output),
            "--timeout",
            _format_number(args.timeout),
            "--offset",
            str(args.offset),
        ]
        if args.limit is not None:
            query_eval_argv.extend(["--limit", str(args.limit)])
        for case_id in args.case_id:
            query_eval_argv.extend(["--case-id", case_id])
        if args.fail_on_query_error:
            query_eval_argv.append("--fail-on-error")
        commands.append({"name": "query_eval", "argv": query_eval_argv})

        summary_argv = [
            args.python,
            "scripts/query_report_summary.py",
            str(json_output),
            "--benchmark",
            str(args.benchmark),
            "--json-output",
            str(args.out_dir / "query_attribution.json"),
        ]
        commands.append({"name": "query_report_summary", "argv": summary_argv})

    return commands


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    started_at = _utc_now()
    out_dir = _resolve_out_dir(args.out_dir)
    args.out_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    command_specs = build_commands(args)
    commands: list[dict[str, Any]] = []
    git_status_short = "" if args.dry_run else _git_status_short()

    for spec in command_specs:
        command_record = _initial_command_record(spec["name"], spec["argv"], logs_dir)
        if args.dry_run:
            command_record["exit_code"] = None
            command_record["duration_seconds"] = 0.0
            _write_text(Path(command_record["stdout_path"]), "")
            _write_text(Path(command_record["stderr_path"]), "")
            print(_quote_argv(spec["argv"]))
        else:
            _run_command(command_record)
        commands.append(command_record)

    query_eval_summary = _query_eval_summary(args, commands)
    mineru_smoke_summary = _mineru_smoke_summary(args, commands)
    service_ingest_summary = _service_ingest_summary(args)
    overall_status = _overall_status(args.dry_run, commands, query_eval_summary, mineru_smoke_summary, service_ingest_summary)
    manifest = {
        "started_at": started_at,
        "finished_at": _utc_now(),
        "profile": args.profile,
        "base_url": args.base_url,
        "benchmark": str(args.benchmark),
        "out_dir": str(out_dir),
        "dry_run": args.dry_run,
        "git_status_short": git_status_short,
        "commands": commands,
        "query_eval": query_eval_summary,
        "mineru_smoke": mineru_smoke_summary,
        "service_ingest": service_ingest_summary,
        "overall_status": overall_status,
    }
    _write_manifest(out_dir, manifest)
    return manifest


def _initial_command_record(name: str, argv: list[str], logs_dir: Path) -> dict[str, Any]:
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return {
        "name": name,
        "argv": argv,
        "exit_code": None,
        "duration_seconds": None,
        "stdout_path": str(logs_dir / f"{safe_name}.stdout.txt"),
        "stderr_path": str(logs_dir / f"{safe_name}.stderr.txt"),
    }


def _run_command(command_record: dict[str, Any]) -> None:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command_record["argv"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        command_record["exit_code"] = 127
        command_record["duration_seconds"] = round(time.perf_counter() - started, 3)
        _write_text(Path(command_record["stdout_path"]), "")
        _write_text(Path(command_record["stderr_path"]), str(exc) + "\n")
        return
    duration = time.perf_counter() - started
    command_record["duration_seconds"] = round(duration, 3)
    command_record["exit_code"] = completed.returncode
    _write_text(Path(command_record["stdout_path"]), completed.stdout or "")
    _write_text(Path(command_record["stderr_path"]), completed.stderr or "")


def _git_status_short() -> str:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return f"git status failed: {exc}"
    if completed.returncode != 0:
        return (completed.stderr or completed.stdout or "").strip()
    return completed.stdout.strip()


def _query_eval_summary(args: argparse.Namespace, commands: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not args.run_query_eval:
        return None
    report_path = args.out_dir / "query_eval.json"
    markdown_path = args.out_dir / "query_eval.md"
    attribution_path = args.out_dir / "query_attribution.json"
    report_summary: dict[str, Any] | None = None
    attribution_summary: dict[str, Any] | None = None
    attribution: dict[str, Any] | None = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary = report.get("summary") if isinstance(report, dict) else None
            if isinstance(summary, dict):
                report_summary = summary
        except json.JSONDecodeError:
            report_summary = {"error": "query_eval report was not valid JSON"}
    if attribution_path.exists():
        try:
            attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
            cases = attribution.get("cases") if isinstance(attribution, dict) else None
            if isinstance(cases, list):
                stage_counts = Counter(
                    str(case.get("likely_stage"))
                    for case in cases
                    if isinstance(case, dict) and case.get("likely_stage")
                )
                failed_stage_counts = Counter(
                    str(case.get("likely_stage"))
                    for case in cases
                    if isinstance(case, dict) and case.get("status") != "pass" and case.get("likely_stage")
                )
                attribution_summary = {
                    "case_count": len(cases),
                    "failure_reason_counts": attribution.get("failure_reason_counts", {}),
                    "likely_stage_counts": dict(stage_counts),
                    "failed_likely_stage_counts": dict(failed_stage_counts),
                }
        except json.JSONDecodeError:
            attribution_summary = {"error": "query attribution report was not valid JSON"}

    summary_command = next((command for command in commands if command["name"] == "query_report_summary"), None)
    return {
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
        "attribution_path": str(attribution_path),
        "summary": report_summary,
        "attribution_summary": attribution_summary,
        "gate": _query_gate_summary(args, report_summary, attribution),
        "summary_stdout_path": summary_command.get("stdout_path") if summary_command else None,
        "summary_stderr_path": summary_command.get("stderr_path") if summary_command else None,
    }


def _query_gate_summary(args: argparse.Namespace, report_summary: dict[str, Any] | None, attribution: dict[str, Any] | None) -> dict[str, Any]:
    enabled = args.min_query_passed is not None or args.max_query_failed is not None or args.require_failure_attribution
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {
        "enabled": enabled,
        "min_query_passed": args.min_query_passed,
        "max_query_failed": args.max_query_failed,
        "require_failure_attribution": args.require_failure_attribution,
    }
    if not enabled:
        return {**details, "passed": None, "checks": checks}
    if not isinstance(report_summary, dict):
        return {**details, "passed": False, "checks": checks, "error": "query_eval summary is unavailable"}
    passed_count = _safe_int(report_summary.get("passed"))
    failed_count = _safe_int(report_summary.get("failed"))
    details.update({"passed_count": passed_count, "failed_count": failed_count})
    if args.min_query_passed is not None:
        checks["min_query_passed"] = "passed" in report_summary and passed_count >= args.min_query_passed
    if args.max_query_failed is not None:
        checks["max_query_failed"] = "failed" in report_summary and failed_count <= args.max_query_failed
    if args.require_failure_attribution:
        cases = attribution.get("cases") if isinstance(attribution, dict) else None
        if not isinstance(cases, list):
            checks["require_failure_attribution"] = False
        else:
            failed_cases = [case for case in cases if isinstance(case, dict) and case.get("status") != "pass"]
            checks["require_failure_attribution"] = all(
                case.get("likely_stage") not in (None, "", "unknown") and bool(case.get("stage_reasons") or {})
                for case in failed_cases
            )
            details["unattributed_case_ids"] = [
                case.get("id")
                for case in failed_cases
                if case.get("likely_stage") in (None, "", "unknown") or not bool(case.get("stage_reasons") or {})
            ]
    return {**details, "passed": all(checks.values()), "checks": checks}


def _mineru_smoke_summary(args: argparse.Namespace, commands: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not args.run_mineru_smoke:
        return None
    summary_path = args.out_dir / "mineru_parser_smoke.json"
    summary: dict[str, Any] | None = None
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = {
                    "status": loaded.get("status"),
                    "parser_mode": loaded.get("parser_mode"),
                    "chunks": loaded.get("chunks"),
                    "page_outputs": loaded.get("page_outputs"),
                    "tables": loaded.get("tables"),
                    "figures": loaded.get("figures"),
                    "formulas": loaded.get("formulas"),
                    "content_list_path": loaded.get("content_list_path"),
                    "output_dir": loaded.get("output_dir"),
                    "error": loaded.get("error"),
                }
        except json.JSONDecodeError:
            summary = {"status": "failed", "error": "MinerU smoke summary was not valid JSON."}
    else:
        summary = {"status": "failed", "error": "MinerU smoke summary file was not written."}

    smoke_command = next((command for command in commands if command["name"] == "mineru_parser_smoke"), None)
    return {
        "summary_path": str(summary_path),
        "summary": summary,
        "stdout_path": smoke_command.get("stdout_path") if smoke_command else None,
        "stderr_path": smoke_command.get("stderr_path") if smoke_command else None,
    }


def _service_ingest_summary(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.run_service_ingest_smoke:
        return None
    summary_path = args.out_dir / "service_ingest_smoke.json"
    summary = _dry_run_service_ingest_summary(args) if args.dry_run else _run_service_ingest_smoke(args)
    _write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return {
        "summary_path": str(summary_path),
        "summary": summary,
    }


def _dry_run_service_ingest_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "dry_run",
        "base_url": args.base_url,
        "pdf": str(args.service_ingest_pdf),
        "project_slug": args.service_ingest_project_slug,
        "project_name": args.service_ingest_project_name,
    }


def _run_service_ingest_smoke(args: argparse.Namespace) -> dict[str, Any]:
    pdf_path = args.service_ingest_pdf
    summary: dict[str, Any] = {
        "status": "failed",
        "base_url": args.base_url,
        "pdf": str(pdf_path),
        "project_slug": args.service_ingest_project_slug,
        "project_name": args.service_ingest_project_name,
        "upload": None,
        "run": None,
        "document": None,
        "quality": None,
        "lint": None,
        "error": None,
    }
    if pdf_path is None or not pdf_path.is_file():
        summary["error"] = f"Service ingest PDF does not exist: {pdf_path}"
        return summary

    upload = _upload_service_ingest_pdf(
        args.base_url,
        pdf_path,
        project_slug=args.service_ingest_project_slug,
        project_name=args.service_ingest_project_name,
        timeout=args.service_ingest_http_timeout,
    )
    summary["upload"] = upload
    if upload["status_code"] != 200 or not isinstance(upload.get("payload"), dict):
        summary["error"] = upload.get("error") or f"Upload failed with HTTP {upload['status_code']}"
        return summary

    upload_payload = upload["payload"]
    document_id = str(upload_payload.get("document_id") or "")
    run_id = str(upload_payload.get("run_id") or "")
    if not document_id or not run_id:
        summary["error"] = "Upload response did not include document_id and run_id."
        return summary

    run = _wait_for_service_ingest_run(
        args.base_url,
        run_id=run_id,
        timeout=args.service_ingest_timeout,
        interval=args.service_ingest_poll_interval,
        http_timeout=args.service_ingest_http_timeout,
    )
    summary["run"] = run
    if run.get("status") != "completed":
        summary["error"] = run.get("error") or f"Ingest run ended with status {run.get('status')!r}."
        return summary
    duplicate_skipped = _is_duplicate_ingest_skip(run)
    summary["duplicate_skipped"] = duplicate_skipped
    if duplicate_skipped and not args.allow_service_ingest_duplicate_skip:
        summary["error"] = "Service ingest produced a duplicate-skip run; use a fresh project/file or pass --allow-service-ingest-duplicate-skip."
        return summary

    document = _http_get_json(_api_url(args.base_url, f"documents/{urllib.parse.quote(document_id, safe='')}"), timeout=args.service_ingest_http_timeout)
    quality = _http_get_json(_api_url(args.base_url, f"documents/{urllib.parse.quote(document_id, safe='')}/quality"), timeout=args.service_ingest_http_timeout)
    lint = _http_get_json(
        _api_url(
            args.base_url,
            "wiki/lint",
            {
                "project_slug": args.service_ingest_project_slug,
                "limit": "20",
                "offset": "0",
            },
        ),
        timeout=args.service_ingest_http_timeout,
    )
    summary["document"] = document
    summary["quality"] = quality
    summary["lint"] = lint

    document_payload = document.get("payload") if isinstance(document.get("payload"), dict) else {}
    quality_payload = quality.get("payload") if isinstance(quality.get("payload"), dict) else {}
    lint_payload = lint.get("payload") if isinstance(lint.get("payload"), dict) else {}
    metadata = document_payload.get("metadata_json") if isinstance(document_payload.get("metadata_json"), dict) else {}
    intelligence = metadata.get("document_intelligence") if isinstance(metadata.get("document_intelligence"), dict) else {}

    parser_mode = quality_payload.get("parser_mode") or metadata.get("parser_mode")
    page_output_count = _safe_int(quality_payload.get("page_output_count"))
    content_list_path = intelligence.get("content_list_path")
    checks = {
        "document_ready": document_payload.get("status") == "ready",
        "mineru_parser": parser_mode == "pdf_mineru",
        "page_outputs_present": page_output_count > 0,
        "content_list_present": bool(content_list_path),
        "lint_reachable": lint.get("status_code") == 200,
    }
    summary["checks"] = checks
    summary["evidence"] = {
        "document_id": document_id,
        "run_id": run_id,
        "document_status": document_payload.get("status"),
        "parser_mode": parser_mode,
        "page_output_count": page_output_count,
        "table_count": quality_payload.get("table_count"),
        "structured_table_count": quality_payload.get("structured_table_count"),
        "formula_count": quality_payload.get("formula_count"),
        "figure_count": quality_payload.get("figure_count"),
        "content_list_path": content_list_path,
        "lint_issue_count": lint_payload.get("issue_count"),
        "lint_returned_issue_count": lint_payload.get("returned_issue_count"),
        "warnings": quality_payload.get("warnings"),
        "duplicate_skipped": duplicate_skipped,
    }
    if all(checks.values()):
        summary["status"] = "passed"
        summary["error"] = None
    else:
        failed_checks = [name for name, passed in checks.items() if not passed]
        summary["error"] = "Service ingest smoke failed checks: " + ", ".join(failed_checks)
    return summary


def _upload_service_ingest_pdf(
    base_url: str,
    pdf_path: Path,
    *,
    project_slug: str,
    project_name: str,
    timeout: float,
) -> dict[str, Any]:
    boundary = f"----llmwiki-{uuid.uuid4().hex}"
    body = _multipart_file_body("file", pdf_path, boundary)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    url = _api_url(base_url, "ingest/upload", {"project_slug": project_slug, "project_name": project_name})
    return _http_json(url, data=body, headers=headers, method="POST", timeout=timeout)


def _multipart_file_body(field_name: str, path: Path, boundary: str) -> bytes:
    file_name = path.name
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{file_name}"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + path.read_bytes() + footer


def _wait_for_service_ingest_run(
    base_url: str,
    *,
    run_id: str,
    timeout: float,
    interval: float,
    http_timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(timeout, 0.0)
    history: list[dict[str, Any]] = []
    while True:
        run_lookup = _find_service_ingest_run(base_url, run_id=run_id, http_timeout=http_timeout)
        if run_lookup.get("error"):
            return {
                "id": run_id,
                "status": "failed",
                "history": history,
                "error": run_lookup["error"],
            }
        run = run_lookup.get("run")
        if run is not None:
            status = str(run.get("status") or "unknown")
            history.append(
                {
                    "status": status,
                    "notes": run.get("notes"),
                    "provider_report": run.get("provider_report"),
                }
            )
            if status in {"completed", "failed"}:
                return {
                    "id": run_id,
                    "status": status,
                    "document_id": run.get("document_id"),
                    "notes": run.get("notes"),
                    "provider_report": run.get("provider_report"),
                    "history": history,
                }
        else:
            history.append({"status": "missing"})

        if time.monotonic() >= deadline:
            return {
                "id": run_id,
                "status": "timeout",
                "history": history,
                "error": f"Ingest run did not reach a terminal state within {timeout:g}s.",
            }
        time.sleep(max(interval, 0.1))


def _find_service_ingest_run(base_url: str, *, run_id: str, http_timeout: float, page_size: int = 50, max_pages: int = 20) -> dict[str, Any]:
    for page_index in range(max_pages):
        offset = page_index * page_size
        response = _http_get_json(_api_url(base_url, "runs", {"limit": str(page_size), "offset": str(offset)}), timeout=http_timeout)
        if response["status_code"] != 200:
            return {"run": None, "error": response.get("error") or f"Run polling failed with HTTP {response['status_code']}"}
        runs = response.get("payload") if isinstance(response.get("payload"), list) else []
        run = next((item for item in runs if isinstance(item, dict) and item.get("id") == run_id), None)
        if run is not None:
            return {"run": run, "error": ""}
        if len(runs) < page_size:
            return {"run": None, "error": ""}
    return {"run": None, "error": f"Ingest run {run_id} was not found in the newest {page_size * max_pages} runs."}


def _is_duplicate_ingest_skip(run: dict[str, Any]) -> bool:
    notes = str(run.get("notes") or "")
    return "duplicate document skipped" in notes.lower()


def _http_get_json(url: str, *, timeout: float) -> dict[str, Any]:
    return _http_json(url, timeout=timeout)


def _http_json(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
            return {
                "status_code": getattr(response, "status", 200),
                "payload": json.loads(raw_body) if raw_body else {},
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        return {
            "status_code": exc.code,
            "payload": _decode_json(raw_body),
            "error": raw_body or str(exc),
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {
            "status_code": 0,
            "payload": {},
            "error": str(exc),
        }


def _api_url(base_url: str, path: str, query: dict[str, str] | None = None) -> str:
    normalized_base = base_url.rstrip("/")
    api_base = normalized_base if normalized_base.endswith("/api") else f"{normalized_base}/api"
    url = f"{api_base}/{path.lstrip('/')}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    return url


def _decode_json(raw_body: str) -> Any:
    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        return {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _overall_status(
    dry_run: bool,
    commands: list[dict[str, Any]],
    query_eval_summary: dict[str, Any] | None = None,
    mineru_smoke_summary: dict[str, Any] | None = None,
    service_ingest_summary: dict[str, Any] | None = None,
) -> str:
    if not commands and not mineru_smoke_summary and not service_ingest_summary:
        return "no_commands"
    if dry_run:
        return "dry_run"
    if any(command.get("exit_code") not in (0, None) for command in commands):
        return "failed"
    if query_eval_summary:
        gate = query_eval_summary.get("gate")
        if isinstance(gate, dict) and gate.get("enabled") and gate.get("passed") is not True:
            return "failed"
    if mineru_smoke_summary:
        summary = mineru_smoke_summary.get("summary")
        if not isinstance(summary, dict) or summary.get("status") != "passed":
            return "failed"
    if service_ingest_summary:
        summary = service_ingest_summary.get("summary")
        if not isinstance(summary, dict) or summary.get("status") != "passed":
            return "failed"
    return "passed"


def _write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    _write_text(out_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _write_text(out_dir / "manifest.md", _render_manifest_markdown(manifest))


def _render_manifest_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# MinerU/RAG Loop Manifest",
        "",
        f"- Started: `{manifest['started_at']}`",
        f"- Finished: `{manifest['finished_at']}`",
        f"- Profile: `{manifest['profile']}`",
        f"- Overall status: `{manifest['overall_status']}`",
        "",
        "## Commands",
        "",
        "| Name | Exit code | Duration |",
        "| --- | ---: | ---: |",
    ]
    for command in manifest["commands"]:
        exit_code = "-" if command["exit_code"] is None else str(command["exit_code"])
        duration = "-" if command["duration_seconds"] is None else str(command["duration_seconds"])
        lines.append(f"| {command['name']} | {exit_code} | {duration} |")
    query_eval = manifest.get("query_eval")
    if query_eval:
        lines.extend(
            [
                "",
                "## Query Eval",
                "",
                f"- Report: `{query_eval['report_path']}`",
                f"- Markdown: `{query_eval['markdown_path']}`",
                f"- Attribution: `{query_eval['attribution_path']}`",
            ]
        )
        if query_eval.get("summary") is not None:
            lines.append(f"- Summary: `{json.dumps(query_eval['summary'], ensure_ascii=False)}`")
        if query_eval.get("attribution_summary") is not None:
            lines.append(f"- Attribution summary: `{json.dumps(query_eval['attribution_summary'], ensure_ascii=False)}`")
        if query_eval.get("gate") is not None:
            lines.append(f"- Gate: `{json.dumps(query_eval['gate'], ensure_ascii=False)}`")
    mineru_smoke = manifest.get("mineru_smoke")
    if mineru_smoke:
        lines.extend(["", "## MinerU Parser Smoke", "", f"- Summary: `{mineru_smoke['summary_path']}`"])
        if mineru_smoke.get("summary") is not None:
            lines.append(f"- Result: `{json.dumps(mineru_smoke['summary'], ensure_ascii=False)}`")
    service_ingest = manifest.get("service_ingest")
    if service_ingest:
        lines.extend(["", "## Service Ingest Smoke", "", f"- Summary: `{service_ingest['summary_path']}`"])
        if service_ingest.get("summary") is not None:
            lines.append(f"- Result: `{json.dumps(service_ingest['summary'], ensure_ascii=False)}`")
    lines.append("")
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_out_dir(out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return DEFAULT_OUT_ROOT / f"mineru_rag_{stamp}"


def _format_number(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _quote_argv(argv: list[str]) -> str:
    return " ".join(_quote_arg(arg) for arg in argv)


def _quote_arg(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_loop(args)
    return 0 if manifest["overall_status"] in {"passed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
