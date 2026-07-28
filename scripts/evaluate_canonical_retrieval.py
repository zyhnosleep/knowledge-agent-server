from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.schemas.agent import EvidencePack
from app.services.rag_adapter import RAGAdapter
from scripts.rebuild_canonical_index import collect_integrity_metrics


STRICT_INTEGRITY_GATES = (
    "parse_completeness",
    "contextual_prefix_completeness",
    "embedding_completeness",
    "table_validation_rate",
    "source_span_validity",
    "artifact_link_validity",
)

RetrieveFunction = Callable[[dict[str, Any], int], EvidencePack]
AnswerFunction = Callable[[dict[str, Any]], Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _identity_matches(item: Any, case: dict[str, Any]) -> bool:
    expected_id = str(case.get("expected_document_id") or "").strip()
    expected_identities = {
        str(value).strip()
        for value in [
            case.get("expected_source_identity"),
            *case.get("expected_source_identities", []),
        ]
        if str(value or "").strip()
    }
    return bool(
        (expected_id and str(item.document_id) == expected_id)
        or (expected_identities and str(item.page_slug) in expected_identities)
    )


def _has_location(item: Any, requirement: str | None) -> bool:
    if not requirement:
        return True
    spans = item.source_spans if isinstance(item.source_spans, list) else []
    if requirement == "bbox":
        return any(span.get("normalized_bbox") or span.get("bbox") for span in spans)
    if requirement == "text":
        return any(
            span.get("line_start") is not None or span.get("char_start") is not None
            for span in spans
        )
    if requirement == "any":
        return bool(spans)
    raise ValueError(f"Unknown location requirement {requirement!r}.")


def evaluate_cases(
    cases: list[dict[str, Any]],
    retrieve: RetrieveFunction,
    answer: AnswerFunction | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[int] = []
    failure_counts: Counter[str] = Counter()
    location_passes = 0
    location_required = 0
    prefix_passes = 0
    prefix_checked = 0

    for case in cases:
        started = time.perf_counter()
        evidence = retrieve(case, 10)
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        latencies.append(latency_ms)
        items = list(evidence.items)
        matching_indexes = [
            index for index, item in enumerate(items) if _identity_matches(item, case)
        ]
        recall_at_5 = any(index < 5 for index in matching_indexes)
        recall_at_10 = any(index < 10 for index in matching_indexes)
        candidates = [items[index] for index in matching_indexes[:10]]
        expected_block_type = case.get("expected_block_type")
        block_type_passed = not expected_block_type or any(
            item.block_type == expected_block_type for item in candidates
        )
        expected_page = str(case.get("expected_page_label") or "").strip()
        page_passed = not expected_page or any(
            str(item.page_label or "") == expected_page for item in candidates
        )
        qualified_candidates = [
            item
            for item in candidates
            if (not expected_block_type or item.block_type == expected_block_type)
            and (not expected_page or str(item.page_label or "") == expected_page)
        ]
        excerpts = "\n".join(item.excerpt or "" for item in qualified_candidates)
        required_terms = [str(term) for term in case.get("required_terms", [])]
        required_terms_passed = all(
            term.casefold() in excerpts.casefold() for term in required_terms
        )
        forbidden_terms = [
            str(term) for term in case.get("forbidden_excerpt_terms", [])
        ]
        prefix_exclusion_passed = all(
            term.casefold() not in excerpts.casefold() for term in forbidden_terms
        )
        if forbidden_terms:
            prefix_checked += 1
            prefix_passes += int(prefix_exclusion_passed)

        location_requirement = case.get("location_requirement")
        location_passed = not location_requirement or any(
            _has_location(item, str(location_requirement))
            for item in qualified_candidates
        )
        if location_requirement:
            location_required += 1
            location_passes += int(location_passed)

        reference_exclusion_passed = not any(
            str(item.block_type or "").casefold() in {"reference", "references"}
            for item in items
        )
        failures: list[str] = []
        checks = (
            ("retrieval", recall_at_10),
            ("block_type", block_type_passed),
            ("page_label", page_passed),
            ("required_terms", required_terms_passed),
            ("contextual_prefix_citation_exclusion", prefix_exclusion_passed),
            ("source_location", location_passed),
        )
        for label, passed in checks:
            if not passed:
                failures.append(label)
        if case.get("category") == "reference_exclusion" and not reference_exclusion_passed:
            failures.append("reference_exclusion")

        answer_passed: bool | None = None
        citation_passed: bool | None = None
        if case.get("require_answer"):
            if answer is None:
                answer_passed = False
                citation_passed = False
            else:
                response = answer(case)
                answer_text = str(response.answer_markdown or "")
                answer_passed = all(
                    str(term).casefold() in answer_text.casefold()
                    for term in case.get("answer_required_terms", required_terms)
                )
                citation_passed = any(
                    _identity_matches(citation, case)
                    and all(
                        forbidden.casefold()
                        not in str(citation.excerpt or "").casefold()
                        for forbidden in forbidden_terms
                    )
                    for citation in response.citations
                )
            if not answer_passed:
                failures.append("answer")
            if not citation_passed:
                failures.append("citation")

        for failure in set(failures):
            failure_counts[failure] += 1
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "latency_ms": latency_ms,
                "retrieved_document_ids": [item.document_id for item in items[:10]],
                "retrieved_chunk_ids": [item.chunk_id for item in items[:10]],
                "recall_at_5": recall_at_5,
                "recall_at_10": recall_at_10,
                "block_type_passed": block_type_passed,
                "page_label_passed": page_passed,
                "required_terms_passed": required_terms_passed,
                "prefix_exclusion_passed": prefix_exclusion_passed,
                "source_location_passed": location_passed,
                "reference_exclusion_passed": reference_exclusion_passed,
                "answer_passed": answer_passed,
                "citation_passed": citation_passed,
                "failures": failures,
                "passed": not failures,
            }
        )

    total = len(rows)
    answer_rows = [row for row in rows if row["answer_passed"] is not None]
    summary = {
        "case_count": total,
        "passed_cases": sum(row["passed"] for row in rows),
        "recall_at_5": (
            round(sum(row["recall_at_5"] for row in rows) / total, 6)
            if total
            else 0.0
        ),
        "recall_at_10": (
            round(sum(row["recall_at_10"] for row in rows) / total, 6)
            if total
            else 0.0
        ),
        "source_location_validity": (
            round(location_passes / location_required, 6)
            if location_required
            else 1.0
        ),
        "contextual_prefix_citation_exclusion": (
            round(prefix_passes / prefix_checked, 6)
            if prefix_checked
            else 1.0
        ),
        "answer_citation_result": round(
            sum(
                row["answer_passed"] is True and row["citation_passed"] is True
                for row in answer_rows
            )
            / len(answer_rows),
            6,
        ) if answer_rows else 1.0,
        "p50_retrieval_latency_ms": _percentile(latencies, 0.50),
        "p95_retrieval_latency_ms": _percentile(latencies, 0.95),
        "failure_attribution": dict(sorted(failure_counts.items())),
        "strict_pass": bool(rows) and all(row["passed"] for row in rows),
    }
    return {"created_at": _utc_now(), "cases": rows, "summary": summary}


def apply_strict_gates(
    integrity: dict[str, Any], retrieval: dict[str, Any]
) -> dict[str, Any]:
    failed = [
        name for name in STRICT_INTEGRITY_GATES if integrity.get(name) != 1.0
    ]
    if integrity.get("pgvector_completeness") not in {None, 1.0}:
        failed.append("pgvector_completeness")
    if retrieval.get("summary", {}).get("strict_pass") is not True:
        failed.append("retrieval_cases")
    return {
        "created_at": _utc_now(),
        "integrity": integrity,
        "retrieval": retrieval,
        "failed_gates": failed,
        "strict_pass": not failed,
    }


def run_acceptance(
    *,
    db: Session,
    cases: list[dict[str, Any]],
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    rag = RAGAdapter()

    def retrieve(case: dict[str, Any], limit: int) -> EvidencePack:
        return rag.retrieve_evidence(
            db,
            case["project_slug"],
            case["question"],
            limit=limit,
            document_id=case.get("query_document_id"),
        )

    def answer(case: dict[str, Any]):
        return rag.answer(
            db,
            case["project_slug"],
            case["question"],
            document_id=case.get("query_document_id"),
        )

    integrity = collect_integrity_metrics(db, artifact_root=artifact_root)
    retrieval = evaluate_cases(cases, retrieve, answer)
    return apply_strict_gates(integrity, retrieval)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate canonical retrieval acceptance")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Acceptance case file must contain a cases array.")
    with SessionLocal() as db:
        report = run_acceptance(db=db, cases=cases)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if report.get("strict_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
