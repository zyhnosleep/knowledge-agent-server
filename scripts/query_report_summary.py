from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a scripts/query_eval.py JSON report.")
    parser.add_argument("report", type=Path, help="Path to a query_eval JSON report.")
    parser.add_argument("--benchmark", type=Path, default=None, help="Optional benchmark JSON for missing expected terms.")
    parser.add_argument("--passed", action="store_true", help="Also print passing case ids.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional structured attribution JSON output path.")
    return parser.parse_args(argv)


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise ValueError(f"{path} does not contain a cases list")
    return [case for case in cases if isinstance(case, dict)]


def expected_by_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {str(case.get("id")): case for case in load_cases(path)}


def as_string_list(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def build_attribution(report: dict[str, Any], expected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cases = report.get("cases", [])
    if not isinstance(cases, list):
        cases = []
    reason_counts = Counter(reason for case in cases if isinstance(case, dict) for reason in case.get("failure_reasons", []))
    return {
        "summary": report.get("summary", {}),
        "failure_reason_counts": dict(reason_counts),
        "cases": [_case_attribution(case, expected.get(str(case.get("id")), {})) for case in cases if isinstance(case, dict)],
    }


def _case_attribution(case: dict[str, Any], benchmark_case: dict[str, Any]) -> dict[str, Any]:
    answer = str(case.get("answer") or "")
    citations = case.get("citations") if isinstance(case.get("citations"), list) else []
    citation_excerpt_text = "\n".join(
        str(citation.get("excerpt") or "")
        for citation in citations
        if isinstance(citation, dict)
    )
    answer_terms = as_string_list(benchmark_case.get("answer_must_include"))
    citation_terms = _citation_expected_terms(benchmark_case)
    citation_groups = _citation_expected_groups(benchmark_case)
    source_hints = as_string_list(benchmark_case.get("citation_source_hint"))
    citation_sources = [_citation_source(citation) for citation in citations]
    available_source_values = _citation_source_values(citations)
    matched_sources = _matching_source_values(available_source_values, source_hints)
    stage_reasons = _stage_reasons(case.get("failure_reasons", []))
    likely_stages = list(dict.fromkeys(stage_reasons.values()))
    return {
        "id": case.get("id"),
        "status": case.get("status"),
        "project_slug": case.get("project_slug"),
        "failure_reasons": case.get("failure_reasons", []),
        "likely_stage": likely_stages[0] if likely_stages else "unknown",
        "likely_stages": likely_stages or ["unknown"],
        "stage_reasons": stage_reasons,
        "answer_expected_terms": _term_presence(answer, answer_terms),
        "citation_expected_terms": _term_presence(citation_excerpt_text, citation_terms),
        "citation_excerpt_groups": _citation_group_presence(citations, citation_groups),
        "citation_sources": citation_sources,
        "citation_source_hint": {
            "expected": source_hints,
            "matched": bool(matched_sources) if source_hints else None,
            "matching_values": matched_sources,
            "available_values": available_source_values,
        },
        "error": case.get("error") or "",
    }


def _citation_expected_terms(benchmark_case: dict[str, Any]) -> list[str]:
    terms = []
    terms.extend(as_string_list(benchmark_case.get("citation_must_include")))
    terms.extend(as_string_list(benchmark_case.get("citation_excerpt_must_include")))
    groups = benchmark_case.get("citation_excerpt_groups_must_include")
    if isinstance(groups, list):
        for group in groups:
            terms.extend(as_string_list(group))
    return list(dict.fromkeys(term for term in terms if term))


def _citation_expected_groups(benchmark_case: dict[str, Any]) -> list[list[str]]:
    groups = benchmark_case.get("citation_excerpt_groups_must_include")
    if not isinstance(groups, list):
        return []
    normalized: list[list[str]] = []
    for group in groups:
        terms = as_string_list(group)
        if terms:
            normalized.append(terms)
    return normalized


def _term_presence(text: str, terms: list[str]) -> dict[str, Any]:
    normalized_text = text.lower()
    present = [term for term in terms if term.lower() in normalized_text]
    missing = [term for term in terms if term.lower() not in normalized_text]
    return {"present": present, "missing": missing}


def _citation_group_presence(citations: list[Any], groups: list[list[str]]) -> list[dict[str, Any]]:
    excerpts = [
        str(citation.get("excerpt") or "")
        for citation in citations
        if isinstance(citation, dict)
    ]
    results = []
    for group in groups:
        matching_indexes = [
            index
            for index, excerpt in enumerate(excerpts)
            if not _term_presence(excerpt, group)["missing"]
        ]
        results.append(
            {
                "terms": group,
                "matched": bool(matching_indexes),
                "matching_citation_indexes": matching_indexes,
            }
        )
    return results


def _citation_source(citation: Any) -> dict[str, Any]:
    if not isinstance(citation, dict):
        return {"raw": str(citation)}
    return {
        "page_slug": citation.get("page_slug"),
        "page_title": citation.get("page_title"),
        "page_kind": citation.get("page_kind"),
        "document_id": citation.get("document_id"),
        "chunk_id": citation.get("chunk_id"),
        "page_label": citation.get("page_label"),
        "score": citation.get("score"),
    }


def _citation_source_values(citations: list[Any]) -> list[str]:
    values: list[str] = []
    for citation in citations:
        if not isinstance(citation, dict):
            values.append(str(citation))
            continue
        for key in ("page_slug", "page_title", "page_kind", "document_id", "chunk_id", "page_label"):
            value = citation.get(key)
            if value is not None:
                values.append(str(value))
    return values


def _matching_source_values(source_values: list[str], hints: list[str]) -> list[str]:
    if not hints:
        return []
    matches = []
    normalized_values = [(_normalize_source_value(value), value) for value in source_values]
    for hint in hints:
        normalized_hint = _normalize_source_value(hint)
        for normalized_value, raw_value in normalized_values:
            if normalized_hint == normalized_value or ("/" not in normalized_hint and normalized_hint in normalized_value):
                matches.append(raw_value)
    return list(dict.fromkeys(matches))


def _normalize_source_value(value: str) -> str:
    return value.strip().strip("/").lower()


def _stage_reasons(reasons: Any) -> dict[str, str]:
    reason_set = set(as_string_list(reasons))
    mapped: dict[str, str] = {}
    if "http_error" in reason_set:
        mapped["http_error"] = "http"
    for reason in ("wrong_source_hint", "source_alias_confusion", "cross_paper_contamination"):
        if reason in reason_set:
            mapped[reason] = "source"
    for reason in ("table_false_negative", "citation_not_table"):
        if reason in reason_set:
            mapped[reason] = "table"
    for reason in ("no_citation", "missing_citation_text"):
        if reason in reason_set:
            mapped[reason] = "citation"
    for reason in ("missing_expected_answer_text", "forbidden_answer_text"):
        if reason in reason_set:
            mapped[reason] = "answer"
    if "non_chinese_answer" in reason_set:
        mapped["non_chinese_answer"] = "chinese"
    for reason in ("citation_index_mismatch", "unsupported_claim"):
        if reason in reason_set:
            mapped[reason] = "unsupported"
    return mapped


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    cases = report.get("cases", [])
    summary = report.get("summary", {})
    expected = expected_by_id(args.benchmark)
    attribution = build_attribution(report, expected)

    print("Summary:", summary)
    reason_counts = Counter(reason for case in cases for reason in case.get("failure_reasons", []))
    if reason_counts:
        print("Failure reasons:", dict(reason_counts))

    for case in cases:
        status = case.get("status")
        if status == "pass" and not args.passed:
            continue
        case_id = str(case.get("id"))
        reasons = case.get("failure_reasons", [])
        print(f"\n{case_id}: {status} {reasons}")
        if status == "pass":
            continue
        benchmark_case = expected.get(case_id, {})
        answer = str(case.get("answer") or "")
        citations = case.get("citations") or []
        citation_text = "\n".join(
            str(citation.get("excerpt") or "")
            for citation in citations
            if isinstance(citation, dict)
        )
        missing_answer = _term_presence(answer, as_string_list(benchmark_case.get("answer_must_include")))["missing"]
        missing_citation = _term_presence(citation_text, as_string_list(benchmark_case.get("citation_excerpt_must_include")))["missing"]
        if missing_answer:
            print("  missing_answer:", missing_answer)
        if missing_citation:
            print("  missing_citation_excerpt:", missing_citation)
        if case.get("error"):
            print("  error:", case.get("error"))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(attribution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
