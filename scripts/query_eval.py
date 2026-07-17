from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any
import urllib.error
import urllib.request


FAILURE_REASON_ORDER = [
    "http_error",
    "no_citation",
    "missing_expected_answer_text",
    "forbidden_answer_text",
    "missing_citation_text",
    "wrong_source_hint",
    "source_alias_confusion",
    "cross_paper_contamination",
    "non_chinese_answer",
    "citation_index_mismatch",
    "table_false_negative",
    "citation_not_table",
    "unsupported_claim",
]


@dataclass(frozen=True)
class QueryHTTPResult:
    status_code: int
    payload: dict[str, Any]
    error: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate /api/query answers against a JSON benchmark.")
    parser.add_argument("benchmark", type=Path, help="Path to a JSON benchmark file.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Knowledge Agent API base URL.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report output path.")
    parser.add_argument("--markdown", type=Path, default=None, help="Optional Markdown report output path.")
    parser.add_argument("--save-answer", action="store_true", help="Ask the API to persist query answers.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many selected cases before running.")
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many selected cases.")
    parser.add_argument("--case-id", action="append", default=[], help="Run only matching case id. Repeatable.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with code 1 when any case fails.")
    return parser.parse_args(argv)


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        cases = data.get("cases")
    else:
        cases = data

    if not isinstance(cases, list):
        raise ValueError("Benchmark JSON must be a list or an object with a 'cases' list.")

    required_fields = ["id", "project_slug", "question"]
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Case at index {index} must be an object.")
        missing = [field for field in required_fields if not case.get(field)]
        if missing:
            raise ValueError(f"Case at index {index} is missing required field(s): {', '.join(missing)}.")
    return cases


def post_query(base_url: str, project_slug: str, question: str, save_answer: bool, timeout: float) -> QueryHTTPResult:
    normalized_base = base_url.rstrip("/")
    url = f"{normalized_base}/query" if normalized_base.endswith("/api") else f"{normalized_base}/api/query"
    body = json.dumps(
        {"project_slug": project_slug, "question": question, "save_answer": save_answer},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
            payload = json.loads(raw_body) if raw_body else {}
            return QueryHTTPResult(status_code=getattr(response, "status", 200), payload=payload)
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        payload = _decode_json_object(raw_body)
        return QueryHTTPResult(status_code=exc.code, payload=payload, error=raw_body or str(exc))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return QueryHTTPResult(status_code=0, payload={}, error=str(exc))


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    base_url: str,
    save_answer: bool,
    timeout: float = 60.0,
    total_cases: int | None = None,
    on_case_complete=None,
) -> dict[str, Any]:
    results = []
    selected_count = len(cases)
    total_count = selected_count if total_cases is None else total_cases
    for index, case in enumerate(cases, start=1):
        result = post_query(
            base_url,
            str(case["project_slug"]),
            str(case["question"]),
            save_answer,
            timeout,
        )
        results.append(evaluate_case(case, result))
        if on_case_complete is not None:
            on_case_complete(_build_report(base_url, save_answer, results, total_count, selected_count))

    return _build_report(base_url, save_answer, results, total_count, selected_count)


def evaluate_case(case: dict[str, Any], result: QueryHTTPResult) -> dict[str, Any]:
    payload = result.payload if isinstance(result.payload, dict) else {}
    answer = str(payload.get("answer_markdown") or payload.get("answer") or "")
    citations = payload.get("citations") if isinstance(payload.get("citations"), list) else []
    failure_reasons: list[str] = []

    if result.status_code != 200:
        failure_reasons.append("http_error")

    needs_citation = (
        bool(case.get("require_citation"))
        or bool(case.get("citation_must_include"))
        or bool(case.get("citation_excerpt_must_include"))
        or bool(case.get("citation_source_hint"))
    )
    if needs_citation and not citations:
        failure_reasons.append("no_citation")

    if result.status_code != 200:
        ordered_reasons = [reason for reason in FAILURE_REASON_ORDER if reason in set(failure_reasons)]
        return _case_result(case, result, answer, citations, ordered_reasons)

    if _missing_any(answer, _as_string_list(case.get("answer_must_include"))):
        failure_reasons.append("missing_expected_answer_text")

    if _contains_any(answer, _as_string_list(case.get("answer_must_not_include"))):
        failure_reasons.append("forbidden_answer_text")

    citation_text = _joined_citation_text(citations)
    if _missing_any(citation_text, _as_string_list(case.get("citation_must_include"))):
        failure_reasons.append("missing_citation_text")

    citation_excerpt_text = _joined_citation_excerpt_text(citations)
    if _missing_any(citation_excerpt_text, _as_string_list(case.get("citation_excerpt_must_include"))):
        failure_reasons.append("missing_citation_text")
    if _missing_any_citation_excerpt_group(citations, case.get("citation_excerpt_groups_must_include")):
        failure_reasons.append("missing_citation_text")

    source_hints = _as_string_list(case.get("citation_source_hint"))
    source_values = _citation_source_values(citations)
    if source_hints and not _matches_any_source_hint(source_values, source_hints):
        failure_reasons.append("wrong_source_hint")
        if _has_source_alias_confusion(source_values, source_hints):
            failure_reasons.append("source_alias_confusion")
        else:
            failure_reasons.append("cross_paper_contamination")

    if case.get("require_chinese") and not _looks_like_chinese_answer(answer):
        failure_reasons.append("non_chinese_answer")

    if _has_citation_index_mismatch(answer, len(citations)):
        failure_reasons.append("citation_index_mismatch")
    elif needs_citation and citations and not _has_valid_citation_marker(answer, len(citations)):
        failure_reasons.append("unsupported_claim")

    if _is_table_case(case):
        if _contains_any(answer, _false_negative_phrases(case)):
            failure_reasons.append("table_false_negative")
        if not _has_table_citation(citations, case):
            failure_reasons.append("citation_not_table")

    ordered_reasons = [reason for reason in FAILURE_REASON_ORDER if reason in set(failure_reasons)]
    return _case_result(case, result, answer, citations, ordered_reasons)


def _case_result(
    case: dict[str, Any],
    result: QueryHTTPResult,
    answer: str,
    citations: list[Any],
    ordered_reasons: list[str],
) -> dict[str, Any]:
    passed = not ordered_reasons
    return {
        "id": case["id"],
        "project_slug": case["project_slug"],
        "question": case["question"],
        "status": "pass" if passed else "fail",
        "passed": passed,
        "failure_reasons": ordered_reasons,
        "http_status": result.status_code,
        "error": result.error,
        "answer": answer,
        "citations": citations,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Query Evaluation Report",
        "",
        f"- Base URL: `{report['base_url']}`",
        f"- Total: {summary['total']}",
        f"- Selected: {summary['selected']}",
        f"- Completed: {summary['completed']}",
        f"- Remaining: {summary['remaining']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        "",
        "| Case | Status | Failure reasons |",
        "| --- | --- | --- |",
    ]
    for case in report["cases"]:
        reasons = ", ".join(case["failure_reasons"]) if case["failure_reasons"] else "-"
        lines.append(f"| {_escape_markdown_table(str(case['id']))} | {case['status']} | {_escape_markdown_table(reasons)} |")
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_path: Path | None, markdown_path: Path | None) -> None:
    _write_report_file(report, output_path, markdown_path)


def _write_report_file(report: dict[str, Any], output_path: Path | None, markdown_path: Path | None) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(output_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(markdown_path, render_markdown(report))


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        "Query eval: "
        f"{summary['passed']}/{summary['completed']} passed, "
        f"{summary['failed']} failed, "
        f"{summary['remaining']} remaining "
        f"({summary['selected']} selected of {summary['total']} total)."
    )
    for case in report["cases"]:
        if case["passed"]:
            continue
        reasons = ", ".join(case["failure_reasons"])
        print(f"- {case['id']}: {reasons}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cases = load_benchmark(args.benchmark)
        selected_cases = select_cases(cases, case_ids=args.case_id, offset=args.offset, limit=args.limit)
        report = evaluate_cases(
            selected_cases,
            base_url=args.base_url,
            save_answer=args.save_answer,
            timeout=args.timeout,
            total_cases=len(cases),
            on_case_complete=lambda partial_report: write_report(partial_report, args.output, args.markdown),
        )
        if not selected_cases:
            write_report(report, args.output, args.markdown)
        if args.output is None and args.markdown is None:
            print_summary(report)
        if args.fail_on_error and report["summary"]["failed"]:
            return 1
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"query_eval error: {exc}", file=sys.stderr)
        return 2


def select_cases(
    cases: list[dict[str, Any]],
    *,
    case_ids: list[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("--offset must be >= 0.")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0.")

    selected = cases
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in selected if str(case.get("id")) in wanted]
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _build_report(
    base_url: str,
    save_answer: bool,
    results: list[dict[str, Any]],
    total_count: int,
    selected_count: int,
) -> dict[str, Any]:
    passed = sum(1 for result in results if result["passed"])
    completed = len(results)
    return {
        "base_url": base_url,
        "save_answer": save_answer,
        "summary": {
            "total": total_count,
            "selected": selected_count,
            "completed": completed,
            "remaining": max(selected_count - completed, 0),
            "passed": passed,
            "failed": completed - passed,
        },
        "cases": results,
    }


def _decode_json_object(raw_body: str) -> dict[str, Any]:
    try:
        decoded = json.loads(raw_body)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _as_string_list(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _missing_any(haystack: str, needles: list[str]) -> bool:
    text = haystack.lower()
    return any(needle.lower() not in text for needle in needles)


def _contains_any(haystack: str, needles: list[str]) -> bool:
    text = haystack.lower()
    return any(needle.lower() in text for needle in needles)


def _joined_citation_text(citations: list[Any]) -> str:
    parts: list[str] = []
    for citation in citations:
        if isinstance(citation, dict):
            for key in ("page_slug", "page_title", "page_kind", "document_id", "chunk_id", "page_label", "excerpt"):
                value = citation.get(key)
                if value is not None:
                    parts.append(str(value))
        else:
            parts.append(str(citation))
    return "\n".join(parts)


def _joined_citation_excerpt_text(citations: list[Any]) -> str:
    parts: list[str] = []
    for citation in citations:
        if isinstance(citation, dict):
            value = citation.get("excerpt")
            if value is not None:
                parts.append(str(value))
        else:
            parts.append(str(citation))
    return "\n".join(parts)


def _citation_source_values(citations: list[Any]) -> list[str]:
    values: list[str] = []
    for citation in citations:
        if isinstance(citation, dict):
            for key in ("page_slug", "page_title", "page_kind", "document_id", "chunk_id", "page_label"):
                value = citation.get(key)
                if value is not None:
                    values.append(str(value))
        else:
            values.append(str(citation))
    return values


def _matches_any_source_hint(source_values: list[str], hints: list[str]) -> bool:
    normalized_values = [_normalize_source_value(value) for value in source_values]
    for hint in hints:
        normalized_hint = _normalize_source_value(hint)
        if normalized_hint in normalized_values:
            return True
        if "/" not in normalized_hint and _contains_any("\n".join(normalized_values), [normalized_hint]):
            return True
    return False


def _has_source_alias_confusion(source_values: list[str], hints: list[str]) -> bool:
    normalized_values = [_normalize_source_value(value) for value in source_values]
    for hint in hints:
        normalized_hint = _normalize_source_value(hint)
        for value in normalized_values:
            if value.startswith(normalized_hint + "-") or value.startswith(normalized_hint + "_"):
                return True
    return False


def _normalize_source_value(value: str) -> str:
    return value.strip().strip("/").lower()


def _is_table_case(case: dict[str, Any]) -> bool:
    checks = (
        case.get("citation_must_include"),
        case.get("citation_excerpt_must_include"),
        case.get("citation_excerpt_groups_must_include"),
        case.get("question"),
        case.get("id"),
    )
    return any(_contains_any(" ".join(_as_string_list(value)), ["table", "表格"]) for value in checks)


def _false_negative_phrases(case: dict[str, Any]) -> list[str]:
    defaults = [
        "未包含",
        "无法提取",
        "无法基于现有材料",
        "not present",
        "not found",
        "not included",
        "does not contain",
        "not included in the provided text",
    ]
    return defaults + _as_string_list(case.get("answer_must_not_include"))


def _has_table_citation(citations: list[Any], case: dict[str, Any]) -> bool:
    excerpts = _citation_excerpt_values(citations)
    table_needles = [needle for needle in _as_string_list(case.get("citation_must_include")) if "table" in needle.lower()]
    if not table_needles:
        return _contains_any("\n".join(excerpts), ["table"])
    return any(_contains_any(excerpt, table_needles) for excerpt in excerpts)


def _missing_any_citation_excerpt_group(citations: list[Any], groups_value: Any) -> bool:
    if not groups_value:
        return False
    if not isinstance(groups_value, list):
        raise ValueError("citation_excerpt_groups_must_include must be a list of string lists.")

    excerpts = _citation_excerpt_values(citations)
    for group in groups_value:
        needles = _as_string_list(group)
        if needles and not any(not _missing_any(excerpt, needles) for excerpt in excerpts):
            return True
    return False


def _citation_excerpt_values(citations: list[Any]) -> list[str]:
    values: list[str] = []
    for citation in citations:
        if isinstance(citation, dict):
            value = citation.get("excerpt")
            if value is not None:
                values.append(str(value))
        else:
            values.append(str(citation))
    return values


def _looks_like_chinese_answer(text: str) -> bool:
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in text if ("a" <= char.lower() <= "z"))
    if cjk_count < 2:
        return False
    if cjk_count + latin_count == 0:
        return False
    return cjk_count / (cjk_count + latin_count) >= 0.2


def _has_citation_index_mismatch(answer: str, citation_count: int) -> bool:
    for match in re.finditer(r"(?<!\[)\[(\d+)\](?!\])", answer):
        if int(match.group(1)) >= citation_count:
            return True
    return False


def _has_valid_citation_marker(answer: str, citation_count: int) -> bool:
    for match in re.finditer(r"(?<!\[)\[(\d+)\](?!\])", answer):
        if int(match.group(1)) < citation_count:
            return True
    return False


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
