from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.records import Document
from app.schemas.agent import EvidencePack
from app.services.rag_adapter import RAGAdapter
from app.services.ingestion_identity import (
    build_ingestion_config_snapshot,
    canonical_ingestion_config_hash,
)
from app.services.scientific_normalization import normalize_scientific_selector
from scripts.rebuild_canonical_index import (
    activate_rebuild_batch,
    collect_integrity_metrics,
)


STRICT_INTEGRITY_GATES = (
    "parse_completeness",
    "source_fidelity_completeness",
    "structured_limit_completeness",
    "contextual_prefix_completeness",
    "plain_embedding_completeness",
    "embedding_completeness",
    "table_validation_rate",
    "source_span_validity",
    "artifact_link_validity",
    "config_identity_completeness",
    "source_version_identity_completeness",
    "child_token_limit_completeness",
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


def _compact_formula_text(value: str, *, preserve_boundaries: bool = False) -> str:
    return normalize_scientific_selector(
        value,
        preserve_boundaries=preserve_boundaries,
    )


def _term_matches_text(term: str, text: str) -> bool:
    """Match a required term while keeping numeric values exact.

    Formula symbols may arrive as Unicode (``χ1``) or LaTeX
    (``\\chi _ { 1 }``); only those symbols receive structural normalization.
    Numeric terms retain boundaries so ``2.35`` does not match ``2.350``, and
    a compact formula subscript keeps its boundary so ``χ1`` does not match
    ``χ10``.
    """
    term_text = str(term or "")
    text_value = str(text or "")
    if re.search(r"(?:\\(?:chi|alpha|beta)|[χαβ])", term_text, re.IGNORECASE):
        compact_term = _compact_formula_text(term_text)
        compact_text = _compact_formula_text(text_value, preserve_boundaries=True)
        if not compact_term:
            return False
        # The subscript index must sit at a numeric boundary: ``χ1`` cannot
        # match ``χ10`` or ``χ12``, while a bare symbol (``χ``) still matches
        # an indexed rendering (``χ10``) because it has no index to bound.
        if re.search(r"\d", compact_term):
            pattern = rf"(?<!\d){re.escape(compact_term)}(?!\d)"
        else:
            pattern = rf"(?<!\d){re.escape(compact_term)}"
        return re.search(pattern, compact_text) is not None
    folded_term = term_text.casefold()
    folded_text = text_value.casefold()
    # Canonical table extraction can preserve a chemical subscript as LaTeX
    # (``C _ { 6 }``) while the acceptance case uses the compact label ``C6``.
    # Match only the safe one-letter numeric form and keep token boundaries so
    # values such as ``C60`` cannot satisfy ``C6`` accidentally.
    subscript = re.fullmatch(r"([a-z])\s*(\d+)", folded_term)
    if subscript:
        letter, index = subscript.groups()
        notation_pattern = rf"(?<![a-z0-9]){re.escape(letter)}(?:\s*[_^]\s*\{{?\s*{index}\s*\}}?|\s*{index})(?![a-z0-9])"
        if re.search(notation_pattern, folded_text):
            return True
    # MinerU OCR occasionally renders the capital ``I`` in amino-acid residue
    # ``Ile`` as a lowercase ``l``.  This alias is deliberately limited to the
    # exact residue token; it is not a general fuzzy matcher.
    if folded_term == "ile" and re.search(r"(?<![a-z])lle(?![a-z])", folded_text):
        return True
    if re.search(r"\d", term_text):
        # A trailing sentence period is valid; reject only a decimal
        # continuation (``2.350``) or a digit-adjacent larger number.
        pattern = rf"(?<!\d)(?<!\.\d){re.escape(folded_term)}(?!\d|\.\d)"
        return re.search(pattern, folded_text) is not None
    return folded_term in folded_text


def _evidence_text(items: list[Any], table_facts: list[Any]) -> str:
    """Join the answerable evidence surface for a set of retrieved items.

    Mirrors the query-side evidence assembly: excerpts plus trusted parent
    contexts plus source-linked table facts.  Scoping to a prefix of the
    retrieved list is what turns this into Recall@5 / Recall@10.
    """
    excerpts = "\n".join(item.excerpt or "" for item in items)
    trusted_parent_contexts = "\n".join(
        item.context_text or ""
        for item in items
        if item.parent_chunk_id and item.context_text
    )
    chunk_ids = {
        str(item.chunk_id) for item in items if item.chunk_id
    }
    table_keys = {
        (str(item.document_id), str(item.table_id))
        for item in items
        if item.document_id and item.table_id
    }
    linked_table_fact_text = "\n".join(
        " | ".join(
            part
            for part in (
                str(fact.table_id or ""),
                str(fact.row_label or ""),
                str(fact.column or ""),
                str(fact.value or ""),
            )
            if part
        )
        for fact in table_facts
        if (
            (str(fact.document_id), str(fact.table_id))
            in table_keys
            or bool(chunk_ids & set(fact.source_chunk_ids))
        )
    )
    return "\n".join(
        part
        for part in (excerpts, trusted_parent_contexts, linked_table_fact_text)
        if part
    )


def _observe_active_parse_version_map(
    db: Session, document_ids: set[str] | None = None
) -> dict[str, str]:
    """Observe the active parse versions currently pointed to by the database.

    Returns ``{document_id: active_parse_version}`` for documents with a
    non-null active pointer.  When ``document_ids`` is given the observation is
    bounded to that set; otherwise every active document is observed.
    """
    statement = select(Document.id, Document.active_parse_version).where(
        Document.active_parse_version.is_not(None)
    )
    if document_ids:
        statement = statement.where(Document.id.in_(document_ids))
    rows = db.execute(statement).all()
    return {
        str(row.id): str(row.active_parse_version)
        for row in rows
        if row.active_parse_version
    }


def evaluate_cases(
    cases: list[dict[str, Any]],
    retrieve: RetrieveFunction,
    answer: AnswerFunction | None = None,
    *,
    route_identity: str = "active",
    effective_parse_version_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[int] = []
    answer_latencies: list[int] = []
    failure_counts: Counter[str] = Counter()
    location_passes = 0
    location_required = 0
    prefix_passes = 0
    prefix_checked = 0
    citation_passes = 0
    citation_checked = 0

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
        # Narrative required terms are retained as an evidence-coverage
        # diagnostic, but they are not a retrieval gate: a narrative Child
        # can be a valid hit while the requested synthesis term lives in its
        # trusted Parent context or is intentionally answered later.
        # Structured evidence (table/figure/formula) is different because the
        # extraction and assembly contract promises complete, source-linked
        # coverage, so a missing required term there fails the retrieval gate.
        # An explicit ``evidence_coverage_required: true`` forces the gate for
        # any block type, including narrative.
        evidence_gate_required = (
            expected_block_type in {"table", "figure", "formula"}
            or bool(case.get("evidence_coverage_required"))
        )
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
        qualified_excerpts = "\n".join(
            item.excerpt or "" for item in qualified_candidates
        )
        evidence_text = _evidence_text(qualified_candidates, evidence.table_facts)
        required_terms = [str(term) for term in case.get("required_terms", [])]
        missing_required_terms = [
            term
            for term in required_terms
            if not _term_matches_text(term, evidence_text)
        ]
        evidence_coverage_passed = not missing_required_terms
        # Backwards-compatible report field: ``required_terms_passed`` aliases
        # the answerable evidence-coverage result (``evidence_coverage_passed``)
        # for existing consumers.  Whether that coverage is an actual retrieval
        # gate or a diagnostic is captured by ``evidence_coverage_gate_required``:
        # when it is false a coverage miss stays visible in these fields but
        # never enters ``failures``.
        required_terms_passed = evidence_coverage_passed
        citation_required_terms = [
            str(term) for term in case.get("citation_required_terms", [])
        ]
        missing_citation_required_terms = [
            term
            for term in citation_required_terms
            if not _term_matches_text(term, qualified_excerpts)
        ]
        citation_required_terms_passed = not missing_citation_required_terms
        if citation_required_terms:
            citation_checked += 1
            citation_passes += int(citation_required_terms_passed)
        forbidden_terms = [
            str(term) for term in case.get("forbidden_excerpt_terms", [])
        ]
        prefix_exclusion_passed = all(
            term.casefold() not in qualified_excerpts.casefold()
            for term in forbidden_terms
        )
        if forbidden_terms:
            prefix_checked += 1
            prefix_passes += int(prefix_exclusion_passed)

        # Required-anchor recall over the evidence actually retrieved: top-5
        # and top-10 prefixes are scored independently so a missing anchor
        # below rank 5 cannot be hidden by a later rank-10 hit.
        anchor_text_at_5 = _evidence_text(items[:5], evidence.table_facts)
        anchor_text_at_10 = _evidence_text(items[:10], evidence.table_facts)
        missing_required_terms_at_5 = [
            term
            for term in required_terms
            if not _term_matches_text(term, anchor_text_at_5)
        ]
        missing_required_terms_at_10 = [
            term
            for term in required_terms
            if not _term_matches_text(term, anchor_text_at_10)
        ]
        required_anchor_count = len(required_terms)
        required_anchor_coverage_at_5 = (
            (required_anchor_count - len(missing_required_terms_at_5))
            / required_anchor_count
            if required_anchor_count
            else None
        )
        required_anchor_coverage_at_10 = (
            (required_anchor_count - len(missing_required_terms_at_10))
            / required_anchor_count
            if required_anchor_count
            else None
        )

        # Route version scoping: evidence from a version other than the one
        # the route points at cannot satisfy the retrieval gate.
        expected_version_by_document = effective_parse_version_map or {}
        wrong_version_evidence = any(
            item.parse_version is not None
            and str(item.document_id) in expected_version_by_document
            and str(item.parse_version)
            != str(expected_version_by_document[str(item.document_id)])
            for item in candidates
        )

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
        checks = [
            ("retrieval", recall_at_10),
            ("block_type", block_type_passed),
            ("page_label", page_passed),
        ]
        # Evidence coverage is a retrieval gate only for structured evidence
        # (table/figure/formula) or when the case explicitly requires it.  For
        # narrative cases a coverage miss is a diagnostic that stays visible in
        # ``evidence_coverage_passed`` / ``missing_required_terms`` and never
        # enters ``failures``, so ``passed`` keeps the meaning of the gates.
        if evidence_gate_required:
            checks.append(("evidence_coverage", evidence_coverage_passed))
        if citation_required_terms:
            checks.append(
                ("citation_required_terms", citation_required_terms_passed)
            )
        checks += [
            ("contextual_prefix_citation_exclusion", prefix_exclusion_passed),
            ("source_location", location_passed),
            ("parse_version", not wrong_version_evidence),
        ]
        for label, passed in checks:
            if not passed:
                failures.append(label)
        if case.get("category") == "reference_exclusion" and not reference_exclusion_passed:
            failures.append("reference_exclusion")

        answer_passed: bool | None = None
        citation_passed: bool | None = None
        answer_latency_ms: int | None = None
        missing_answer_terms: list[str] = []
        if case.get("require_answer"):
            if answer is None:
                answer_passed = False
                citation_passed = False
            else:
                answer_started = time.perf_counter()
                response = answer(case)
                answer_latency_ms = max(
                    0, int((time.perf_counter() - answer_started) * 1000)
                )
                answer_latencies.append(answer_latency_ms)
                answer_text = str(response.answer_markdown or "")
                answer_required_terms = [
                    str(term)
                    for term in case.get("answer_required_terms", required_terms)
                ]
                missing_answer_terms = [
                    term
                    for term in answer_required_terms
                    if term.casefold() not in answer_text.casefold()
                ]
                answer_passed = bool(answer_text) and not missing_answer_terms
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

        # Retrieval-only gate is deliberately disjoint from the full-answer
        # gate: answer/citation failures belong to the ``full_answer`` layer
        # and must not make the retrieval-only gate fail.
        retrieval_gate_failures = set(failures) - {"answer", "citation"}
        retrieval_gate_passed = not retrieval_gate_failures
        answer_gate_required = answer_passed is not None
        full_answer_gate_passed = (
            not answer_gate_required
            or (answer_passed is True and citation_passed is True)
        )

        failed_layers: list[str] = []
        if not retrieval_gate_passed:
            failed_layers.append("retrieval")
        if answer_gate_required and not full_answer_gate_passed:
            failed_layers.append("full_answer")

        for failure in set(failures):
            failure_counts[failure] += 1
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "latency_ms": latency_ms,
                "answer_latency_ms": answer_latency_ms,
                "retrieved_document_ids": [item.document_id for item in items[:10]],
                "retrieved_chunk_ids": [item.chunk_id for item in items[:10]],
                "recall_at_5": recall_at_5,
                "recall_at_10": recall_at_10,
                "required_anchor_count": required_anchor_count,
                "required_anchor_checked": bool(required_anchor_count),
                "required_anchor_coverage_at_5": required_anchor_coverage_at_5,
                "required_anchor_coverage_at_10": required_anchor_coverage_at_10,
                "missing_required_terms_at_5": missing_required_terms_at_5,
                "missing_required_terms_at_10": missing_required_terms_at_10,
                "block_type_passed": block_type_passed,
                "page_label_passed": page_passed,
                "required_terms_passed": required_terms_passed,
                "evidence_coverage_passed": evidence_coverage_passed,
                "evidence_coverage_gate_required": evidence_gate_required,
                "citation_required_terms_passed": citation_required_terms_passed,
                "missing_required_terms": missing_required_terms,
                "missing_citation_required_terms": missing_citation_required_terms,
                "prefix_exclusion_passed": prefix_exclusion_passed,
                "source_location_passed": location_passed,
                "reference_exclusion_passed": reference_exclusion_passed,
                "answer_passed": answer_passed,
                "citation_passed": citation_passed,
                "missing_answer_terms": missing_answer_terms,
                "failures": failures,
                "failed_layers": failed_layers,
                "retrieval_gate_passed": retrieval_gate_passed,
                "full_answer_gate_passed": full_answer_gate_passed,
                "passed": not failures,
            }
        )

    total = len(rows)
    answer_rows = [row for row in rows if row["answer_passed"] is not None]
    # Required-anchor recall is computed only over the cases that declare
    # required terms; the denominator is exposed as ``required_anchor_case_count``.
    anchor_rows = [row for row in rows if row["required_anchor_checked"]]
    anchor_case_count = len(anchor_rows)
    full_answer_required = bool(answer_rows)
    retrieval_strict = bool(rows) and all(
        row["retrieval_gate_passed"] for row in rows
    )
    full_answer_strict = (
        not full_answer_required
        or all(
            row["answer_passed"] is True and row["citation_passed"] is True
            for row in answer_rows
        )
    )
    strict_pass = bool(rows) and retrieval_strict and full_answer_strict
    failed_layers = sorted(
        {layer for row in rows for layer in row["failed_layers"]}
    )
    answer_citation_result = (
        round(
            sum(
                row["answer_passed"] is True and row["citation_passed"] is True
                for row in answer_rows
            )
            / len(answer_rows),
            6,
        )
        if answer_rows
        else 1.0
    )
    retrieval_only = {
        "case_count": total,
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
        "required_anchor_recall_at_5": (
            round(
                sum(
                    row["required_anchor_coverage_at_5"]
                    for row in anchor_rows
                )
                / anchor_case_count,
                6,
            )
            if anchor_case_count
            else None
        ),
        "required_anchor_recall_at_10": (
            round(
                sum(
                    row["required_anchor_coverage_at_10"]
                    for row in anchor_rows
                )
                / anchor_case_count,
                6,
            )
            if anchor_case_count
            else None
        ),
        "required_anchor_case_count": anchor_case_count,
        "full_anchor_recall_at_5_cases": sum(
            row["required_anchor_coverage_at_5"] == 1.0 for row in anchor_rows
        ),
        "full_anchor_recall_at_10_cases": sum(
            row["required_anchor_coverage_at_10"] == 1.0 for row in anchor_rows
        ),
        "missing_required_terms": sorted(
            {term for row in rows for term in row["missing_required_terms"]}
        ),
        "missing_required_terms_at_5": sorted(
            {term for row in rows for term in row["missing_required_terms_at_5"]}
        ),
        "missing_required_terms_at_10": sorted(
            {term for row in rows for term in row["missing_required_terms_at_10"]}
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
        "citation_required_terms_validity": (
            round(citation_passes / citation_checked, 6)
            if citation_checked
            else 1.0
        ),
        "retrieval_gate_pass_rate": (
            round(
                sum(row["retrieval_gate_passed"] for row in rows) / total,
                6,
            )
            if total
            else 0.0
        ),
        "strict_retrieval_pass": retrieval_strict,
        "p50_retrieval_latency_ms": _percentile(latencies, 0.50),
        "p95_retrieval_latency_ms": _percentile(latencies, 0.95),
    }
    full_answer = {
        "case_count": len(answer_rows),
        "answer_pass_rate": (
            round(
                sum(row["answer_passed"] is True for row in answer_rows)
                / len(answer_rows),
                6,
            )
            if answer_rows
            else 1.0
        ),
        "citation_pass_rate": (
            round(
                sum(row["citation_passed"] is True for row in answer_rows)
                / len(answer_rows),
                6,
            )
            if answer_rows
            else 1.0
        ),
        "answer_citation_result": answer_citation_result,
        "missing_answer_terms": sorted(
            {term for row in answer_rows for term in row["missing_answer_terms"]}
        ),
        "strict_full_answer_pass": full_answer_strict,
        "p50_answer_latency_ms": _percentile(answer_latencies, 0.50),
        "p95_answer_latency_ms": _percentile(answer_latencies, 0.95),
    }
    summary = {
        "case_count": total,
        "passed_cases": sum(row["passed"] for row in rows),
        "route_identity": route_identity,
        "retrieval_only": retrieval_only,
        "full_answer": full_answer,
        # Backwards-compatible top-level fields; the layered blocks above are
        # the authoritative per-gate view.
        "recall_at_5": retrieval_only["recall_at_5"],
        "recall_at_10": retrieval_only["recall_at_10"],
        "source_location_validity": retrieval_only["source_location_validity"],
        "contextual_prefix_citation_exclusion": (
            retrieval_only["contextual_prefix_citation_exclusion"]
        ),
        "answer_citation_result": answer_citation_result,
        "p50_retrieval_latency_ms": retrieval_only["p50_retrieval_latency_ms"],
        "p95_retrieval_latency_ms": retrieval_only["p95_retrieval_latency_ms"],
        "failure_attribution": dict(sorted(failure_counts.items())),
        "failed_layers": failed_layers,
        "strict_pass": strict_pass,
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
        "failed_layers": sorted(
            set(retrieval.get("summary", {}).get("failed_layers", []))
        ),
        "strict_pass": not failed,
    }


def run_acceptance(
    *,
    db: Session,
    cases: list[dict[str, Any]],
    artifact_root: Path | None = None,
    parse_version_map: dict[str, str] | None = None,
    expected_ingestion_config: dict[str, Any] | None = None,
    expected_ingestion_config_sha256: str | None = None,
) -> dict[str, Any]:
    rag = RAGAdapter()
    if expected_ingestion_config is None:
        expected_ingestion_config = build_ingestion_config_snapshot()
    if expected_ingestion_config_sha256 is None:
        expected_ingestion_config_sha256 = canonical_ingestion_config_hash(
            expected_ingestion_config
        )

    def retrieve(case: dict[str, Any], limit: int) -> EvidencePack:
        return rag.retrieve_evidence(
            db,
            case["project_slug"],
            case["question"],
            limit=limit,
            document_id=case.get("query_document_id"),
            parse_version_map=parse_version_map,
        )

    def answer(case: dict[str, Any]):
        return rag.answer(
            db,
            case["project_slug"],
            case["question"],
            document_id=case.get("query_document_id"),
            parse_version_map=parse_version_map,
        )

    route_identity = "candidate" if parse_version_map else "active"
    candidate_document_ids = set(parse_version_map) if parse_version_map else set()
    case_document_ids = {
        str(case.get("query_document_id") or case.get("expected_document_id") or "")
        for case in cases
    }
    case_document_ids.discard("")
    observed_document_ids = candidate_document_ids | case_document_ids
    active_parse_version_map = _observe_active_parse_version_map(
        db, observed_document_ids or None
    )
    # Effective routing map: candidate versions win for the documents the
    # candidate map covers; every other document keeps its active version.
    effective_parse_version_map = {
        **active_parse_version_map,
        **(parse_version_map or {}),
    }

    integrity = collect_integrity_metrics(
        db,
        artifact_root=artifact_root,
        document_ids=set(parse_version_map) if parse_version_map else None,
        expected_ingestion_config=expected_ingestion_config,
        expected_ingestion_config_sha256=expected_ingestion_config_sha256,
        parse_version_map=parse_version_map,
    )
    retrieval = evaluate_cases(
        cases,
        retrieve,
        answer,
        route_identity=route_identity,
        effective_parse_version_map=effective_parse_version_map,
    )
    report = apply_strict_gates(integrity, retrieval)
    report["parse_version_map"] = dict(sorted((parse_version_map or {}).items()))
    report["active_parse_version_map"] = dict(sorted(active_parse_version_map.items()))
    report["route_identity"] = route_identity
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate canonical retrieval acceptance")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--parse-version-map", type=Path)
    parser.add_argument("--activate-on-success", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.activate_on_success and args.parse_version_map is None:
        _parser().error("--activate-on-success requires --parse-version-map")
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Acceptance case file must contain a cases array.")
    parse_version_map: dict[str, str] | None = None
    if args.parse_version_map is not None:
        map_payload = json.loads(args.parse_version_map.read_text(encoding="utf-8"))
        if (
            isinstance(map_payload, dict)
            and isinstance(map_payload.get("parse_version_map"), dict)
        ):
            map_payload = map_payload["parse_version_map"]
        if not isinstance(map_payload, dict):
            raise ValueError("Parse version map file must contain a JSON object.")
        parse_version_map = {
            str(document_id): str(version_key)
            for document_id, version_key in map_payload.items()
            if str(document_id) and str(version_key)
        }
        if not parse_version_map or len(parse_version_map) != len(map_payload):
            raise ValueError("Parse version map contains an empty document or version key.")
    with SessionLocal() as db:
        report = run_acceptance(
            db=db,
            cases=cases,
            parse_version_map=parse_version_map,
        )
        if args.activate_on_success and report.get("strict_pass") is True:
            assert parse_version_map is not None
            activate_rebuild_batch(
                db,
                parse_version_map=parse_version_map,
                acceptance_report=report,
            )
            report["activation"] = {
                "status": "completed",
                "activated_document_ids": sorted(parse_version_map),
            }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if report.get("strict_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
