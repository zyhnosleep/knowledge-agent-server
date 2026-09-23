#!/usr/bin/env python3
"""Replay the 15-case RAG benchmark against the running project.

This is deliberately a deterministic first-pass evaluator.  Retrieval is
measured against the benchmark's expected ``document_id`` + ``page_label``
citations and their evidence anchors.  Answer quality is reported with two
signals: a conservative required-term score and a point-level lexical
heuristic.  The latter is a diagnostic, not an LLM judge; the report records
the exact answer and citations so a later human/LLM adjudication can replace
it without re-running retrieval.

Run from the repository root, for example:

    .venv\\Scripts\\python.exe scripts\\evaluate_rag_benchmark.py \\
      --cases docs\\evals\\rag_benchmark_v1.json \\
      --report runtime\\evals\\rag_benchmark_v1_replay_20260923.json \\
      --api-url http://127.0.0.1:8002
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.db.session import SessionLocal
from app.services.rag_adapter import RAGAdapter


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "then", "these", "this", "to", "two", "uses", "what",
    "when", "which", "with", "where", "why", "both", "same", "more", "than",
    "into", "over", "under", "such", "also", "only", "each", "one", "via",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.casefold().strip()


def compact(value: Any) -> str:
    return re.sub(r"[^\w]+", "", normalize(value), flags=re.UNICODE)


def variants(term: str, aliases: dict[str, list[str]]) -> list[str]:
    return [term, *[str(x) for x in aliases.get(term, [])]]


def contains_variant(text: str, term: str, aliases: dict[str, list[str]]) -> bool:
    folded = normalize(text)
    return any(normalize(candidate) in folded for candidate in variants(term, aliases))


def item_dump(item: Any) -> dict[str, Any]:
    """Keep enough evidence for audit without serialising model internals."""
    if hasattr(item, "model_dump"):
        value = item.model_dump()
    else:
        value = dict(item)
    keep = (
        "index", "document_id", "chunk_id", "page_slug", "page_label", "score",
        "excerpt", "context_text", "parse_version", "block_type", "table_id",
        "figure_id", "formula_id", "source_stage", "support_hint",
    )
    return {key: value.get(key) for key in keep if key in value}


def citation_pair(citation: dict[str, Any]) -> tuple[str, str]:
    return (str(citation.get("document_id") or ""), str(citation.get("page_label") or ""))


def expected_anchor_match(expected: dict[str, Any], item: dict[str, Any]) -> bool:
    if citation_pair(expected) != citation_pair(item):
        return False
    anchor = normalize(expected.get("evidence_anchor"))
    if not anchor:
        return True
    surface = normalize(
        f"{item.get('excerpt') or ''}\n{item.get('context_text') or ''}"
    )
    # A strict phrase check is preferred.  The compact fallback handles line
    # breaks and MinerU's occasional punctuation/whitespace variation.
    if anchor in surface or compact(anchor) in compact(surface):
        return True
    # Benchmark anchors sometimes use ``...`` as an intentional omission
    # between two quoted fragments.  Treat it as a wildcard, while requiring
    # every non-empty fragment to occur on the same retrieved page.
    fragments = [part.strip() for part in re.split(r"(?:\.\.\.|…)", anchor) if part.strip()]
    if len(fragments) > 1:
        return all(
            fragment in surface or compact(fragment) in compact(surface)
            for fragment in fragments
        )
    # A heading anchor may have punctuation or inserted words (for example,
    # ``RAG-Sequence Model / RAG-Token Model``).  For short heading-like
    # anchors, require all distinctive alphanumeric components rather than
    # requiring one uninterrupted phrase.
    tokens = [token for token in re.findall(r"[a-z0-9]+", anchor) if len(token) >= 4]
    return len(tokens) >= 2 and all(token in compact(surface) for token in tokens)


def retrieval_metrics(case: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    expected = list(case.get("expected_citations") or [])
    ranks: list[dict[str, Any]] = []
    for target in expected:
        pair_rank = None
        anchor_rank = None
        for index, item in enumerate(items, start=1):
            if citation_pair(target) == citation_pair(item) and pair_rank is None:
                pair_rank = index
            if expected_anchor_match(target, item) and anchor_rank is None:
                anchor_rank = index
        ranks.append({
            "document_id": target.get("document_id"),
            "page_label": target.get("page_label"),
            "evidence_anchor": target.get("evidence_anchor"),
            "pair_rank": pair_rank,
            "anchor_rank": anchor_rank,
        })

    def recall(field: str, cutoff: int) -> float | None:
        if not ranks:
            return None
        return round(sum(1 for row in ranks if row[field] is not None and row[field] <= cutoff) / len(ranks), 6)

    evidence_text = "\n".join(
        f"{item.get('excerpt') or ''}\n{item.get('context_text') or ''}" for item in items
    )
    scoring = case.get("scoring") or {}
    aliases = scoring.get("term_aliases") or {}
    terms = [str(term) for term in scoring.get("required_terms") or []]
    term_rows = [
        {"term": term, "hit": contains_variant(evidence_text, term, aliases)}
        for term in terms
    ]
    return {
        "expected_citation_count": len(expected),
        "citation_ranks": ranks,
        "pair_recall_at_5": recall("pair_rank", 5),
        "pair_recall_at_10": recall("pair_rank", 10),
        "anchor_recall_at_5": recall("anchor_rank", 5),
        "anchor_recall_at_10": recall("anchor_rank", 10),
        "required_terms": term_rows,
        "required_term_recall_at_10": (
            round(sum(row["hit"] for row in term_rows) / len(term_rows), 6)
            if term_rows else None
        ),
    }


def significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9][a-zA-Z0-9_-]*", normalize(text))
    return {word for word in words if word not in STOP_WORDS and len(word) >= 3}


def point_hit(point: str, answer: str, case: dict[str, Any]) -> tuple[bool, str]:
    """Conservative lexical point check; return (hit, evidence)."""
    scoring = case.get("scoring") or {}
    aliases = scoring.get("term_aliases") or {}
    terms = [str(term) for term in scoring.get("required_terms") or []]
    linked = [
        candidate
        for term in terms
        for candidate in variants(term, aliases)
        if normalize(candidate) in normalize(point)
    ]
    answer_folded = normalize(answer)
    if linked:
        hits = [candidate for candidate in linked if normalize(candidate) in answer_folded]
        if hits:
            return True, f"term:{hits[0]}"
        # When a benchmark point contains a required term, do not let generic
        # word overlap make an answer that omitted that term look correct.
        return False, "required_term_missing"

    point_words = significant_words(point)
    answer_words = significant_words(answer)
    overlap = point_words & answer_words
    if len(point_words) >= 4 and len(overlap) >= 2:
        return True, f"word_overlap:{','.join(sorted(overlap))}"

    # Chinese points have no reliable word boundary.  Two shared 3-char
    # shingles is a useful conservative signal for copied/paraphrased facts.
    point_compact = compact(point)
    answer_compact = compact(answer)
    shingles = {
        point_compact[index : index + 3]
        for index in range(max(0, len(point_compact) - 2))
        if len(point_compact[index : index + 3]) == 3
    }
    shared = [shingle for shingle in shingles if shingle in answer_compact]
    if len(shared) >= 2:
        return True, f"char_shingles:{len(shared)}"

    ratio = SequenceMatcher(None, point_compact, answer_compact).ratio() if point_compact else 0.0
    if ratio >= 0.55:
        return True, f"sequence_ratio:{ratio:.3f}"
    return False, f"sequence_ratio:{ratio:.3f}"


def answer_metrics(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer = str(response.get("final_answer") or "")
    scoring = case.get("scoring") or {}
    aliases = scoring.get("term_aliases") or {}
    terms = [str(term) for term in scoring.get("required_terms") or []]
    term_rows = [
        {"term": term, "hit": contains_variant(answer, term, aliases)}
        for term in terms
    ]
    points = []
    for point in case.get("expected_answer_points") or []:
        hit, evidence = point_hit(str(point), answer, case)
        points.append({"point": point, "hit": hit, "evidence": evidence})

    returned_citations = [dict(item) for item in response.get("citations") or []]
    expected_pairs = {
        citation_pair(item) for item in case.get("expected_citations") or []
    }
    returned_pairs = {citation_pair(item) for item in returned_citations}
    matched_pairs = sorted(expected_pairs & returned_pairs)
    required_citation_count = int(scoring.get("required_citation_count") or 0)
    required_point_count = int(scoring.get("required_point_count") or 0)
    point_hits = sum(row["hit"] for row in points)
    return {
        "status": response.get("status"),
        "answer_provider": response.get("answer_provider"),
        "answer_model": response.get("answer_model"),
        "answer_text": answer,
        "warnings": response.get("warnings") or [],
        "required_terms": term_rows,
        "required_term_recall": round(sum(row["hit"] for row in term_rows) / len(term_rows), 6) if term_rows else None,
        "answer_points": points,
        "answer_point_hit_count": point_hits,
        "answer_point_count": len(points),
        "answer_point_hit_rate": round(point_hits / len(points), 6) if points else None,
        "required_point_count": required_point_count,
        "required_point_threshold_pass": point_hits >= required_point_count,
        "returned_citation_count": len(returned_citations),
        "matched_expected_citation_pairs": matched_pairs,
        "citation_pair_recall": round(len(matched_pairs) / len(expected_pairs), 6) if expected_pairs else None,
        "required_citation_count": required_citation_count,
        "required_citation_threshold_pass": len(matched_pairs) >= required_citation_count,
        "citations": returned_citations,
        "threshold_pass": (
            response.get("status") == "completed"
            and point_hits >= required_point_count
            and len(matched_pairs) >= required_citation_count
        ),
    }


def call_agent(api_url: str, project_slug: str, case: dict[str, Any], timeout: int) -> tuple[dict[str, Any], int]:
    payload = json.dumps(
        {
            "project_slug": project_slug,
            "query": case["question"],
            "session_id": f"benchmark-{case['id']}-{int(time.time() * 1000)}",
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/agent/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        body = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return body, max(0, int((time.perf_counter() - started) * 1000))


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8002")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark must contain a non-empty cases array")
    project_slug = str(payload.get("project_slug") or "internal-research")
    report_cases: list[dict[str, Any]] = []
    retrieval_latencies: list[int] = []
    generation_latencies: list[int] = []

    with SessionLocal() as db:
        rag = RAGAdapter()
        for position, case in enumerate(cases, start=1):
            print(f"[{position}/{len(cases)}] {case['id']} retrieval", flush=True)
            started = time.perf_counter()
            try:
                pack = rag.retrieve_evidence(db, project_slug, case["question"], limit=10)
                # The comparison workbench materializes Graph-lite edges on
                # first access.  This replay intentionally reuses one
                # SessionLocal for all cases, so commit each case before the
                # subsequent HTTP request opens another transaction; leaving
                # the edge inserts uncommitted can hold PostgreSQL uniqueness
                # locks and make the API-side comparison appear to hang.
                db.commit()
                retrieval_error = None
                items = [item_dump(item) for item in pack.items]
            except Exception as exc:  # keep the remaining cases running
                db.rollback()
                pack = None
                retrieval_error = f"{type(exc).__name__}: {exc}"
                items = []
            retrieval_ms = max(0, int((time.perf_counter() - started) * 1000))
            retrieval_latencies.append(retrieval_ms)
            row: dict[str, Any] = {
                "position": position,
                "id": case["id"],
                "category": case.get("category"),
                "difficulty": case.get("difficulty"),
                "retrieval_latency_ms": retrieval_ms,
                "retrieval_status": getattr(pack, "status", "error"),
                "retrieval_error": retrieval_error,
                "retrieved_items": items,
                "retrieval": retrieval_metrics(case, items),
            }
            if not args.skip_generation:
                print(f"[{position}/{len(cases)}] {case['id']} generation", flush=True)
                response, generation_ms = call_agent(args.api_url, project_slug, case, args.timeout)
                generation_latencies.append(generation_ms)
                row["generation_latency_ms"] = generation_ms
                row["generation"] = answer_metrics(case, response)
            report_cases.append(row)

    def avg(path: tuple[str, ...]) -> float | None:
        values: list[float] = []
        for row in report_cases:
            value: Any = row
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                values.append(float(value))
        return round(sum(values) / len(values), 6) if values else None

    retrieval_rows = [row["retrieval"] for row in report_cases]
    summary: dict[str, Any] = {
        "case_count": len(report_cases),
        "category_counts": dict(Counter(row["category"] for row in report_cases)),
        "retrieval": {
            "pair_recall_at_5": avg(("retrieval", "pair_recall_at_5")),
            "pair_recall_at_10": avg(("retrieval", "pair_recall_at_10")),
            "anchor_recall_at_5": avg(("retrieval", "anchor_recall_at_5")),
            "anchor_recall_at_10": avg(("retrieval", "anchor_recall_at_10")),
            "required_term_recall_at_10": avg(("retrieval", "required_term_recall_at_10")),
            "case_full_pair_recall_at_5": sum(row.get("pair_recall_at_5") == 1.0 for row in retrieval_rows),
            "case_full_pair_recall_at_10": sum(row.get("pair_recall_at_10") == 1.0 for row in retrieval_rows),
            "p50_latency_ms": percentile(retrieval_latencies, 0.50),
            "p95_latency_ms": percentile(retrieval_latencies, 0.95),
        },
        "generation": {
            "completed_cases": sum(row.get("generation", {}).get("status") == "completed" for row in report_cases),
            "answer_point_hit_rate": avg(("generation", "answer_point_hit_rate")),
            "required_term_recall": avg(("generation", "required_term_recall")),
            "citation_pair_recall": avg(("generation", "citation_pair_recall")),
            "required_point_threshold_pass_rate": avg(("generation", "required_point_threshold_pass")),
            "required_citation_threshold_pass_rate": avg(("generation", "required_citation_threshold_pass")),
            "threshold_pass_rate": avg(("generation", "threshold_pass")),
            "p50_latency_ms": percentile(generation_latencies, 0.50),
            "p95_latency_ms": percentile(generation_latencies, 0.95),
        } if not args.skip_generation else None,
        "method": {
            "retrieval_recall_definition": "fraction of expected document_id/page_label citations present in top-k retrieved items; anchor recall additionally requires evidence_anchor text",
            "answer_point_definition": "deterministic lexical diagnostic using required terms, significant-word overlap, Chinese character shingles, and sequence similarity; not an LLM judge",
            "project_scope": project_slug,
            "api_url": args.api_url,
            "generated_at": now_iso(),
        },
    }
    output = {
        "benchmark_id": payload.get("benchmark_id"),
        "benchmark_version": payload.get("version"),
        "created_at": now_iso(),
        "summary": summary,
        "cases": report_cases,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
