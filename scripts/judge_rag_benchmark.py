#!/usr/bin/env python3
"""LLM-based semantic adjudication for a RAG benchmark replay.

The deterministic evaluator in ``evaluate_rag_benchmark.py`` is useful for
regression gates, but it cannot reliably recognise paraphrases, partial
answers, or grounded abstentions.  This script sends each replay case to the
configured DeepSeek-compatible generation endpoint and writes an auditable
case-level JSON report.

Run from the repository root::

    $env:PYTHONPATH = "src"
    .venv\\Scripts\\python.exe scripts\\judge_rag_benchmark.py `
      --benchmark docs\\evals\\rag_benchmark_v1.json `
      --replay runtime\\evals\\rag_benchmark_v1_replay_api_20260923.json `
      --output runtime\\evals\\rag_benchmark_v1_semantic_judge_20260923.json

The API key is read from the server-side ``.env`` through the application
settings and is never printed or persisted in the output report.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.agent_synthesizer import AgentSynthesizer
from app.services.ai import DeepSeekClient


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _evidence_for_prompt(replay_case: dict[str, Any], *, per_item_limit: int) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in replay_case.get("retrieved_items") or []:
        evidence.append(
            {
                "index": item.get("index"),
                "document_id": item.get("document_id"),
                "page_label": item.get("page_label"),
                "page_slug": item.get("page_slug"),
                "block_type": item.get("block_type"),
                "source_stage": item.get("source_stage"),
                "excerpt": _clip(item.get("excerpt"), per_item_limit),
            }
        )
    return evidence


def _build_prompt(
    benchmark_case: dict[str, Any],
    replay_case: dict[str, Any],
    *,
    evidence_item_limit: int,
) -> tuple[str, str]:
    scoring = benchmark_case.get("scoring") or {}
    expected_points = [str(point) for point in benchmark_case.get("expected_answer_points") or []]
    expected_citations = [
        {
            "document_id": citation.get("document_id"),
            "page_label": citation.get("page_label"),
            "evidence_anchor": citation.get("evidence_anchor"),
        }
        for citation in benchmark_case.get("expected_citations") or []
    ]
    generation = replay_case.get("generation") or {}
    retrieved = _evidence_for_prompt(replay_case, per_item_limit=evidence_item_limit)

    system = """You are a strict but fair evaluator for a scientific-paper RAG system.
Judge the answer against the supplied question, benchmark rubric, and retrieved
evidence only. Do not use outside knowledge to fill missing evidence. A concise
answer may receive full credit if it covers the required points. A grounded
abstention is preferable to an invented fact, but it is incomplete when the
provided evidence actually supports the requested point. Distinguish these
cases:

1. retrieval-limited: the necessary page or fact is absent from the supplied
   evidence;
2. generation-limited: the evidence contains the fact but the answer omits,
   distorts, or invents it;
3. pipeline-fallback: the answer is explicitly a raw-evidence/system-failure
   fallback rather than a synthesized response.

Return exactly one JSON object, with no Markdown fences and no commentary,
using this schema:
{
  "score_0_100": integer,
  "correctness_0_4": integer,
  "completeness_0_4": integer,
  "groundedness_0_4": integer,
  "citation_quality_0_4": integer,
  "style_0_4": integer,
  "point_judgments": [
    {"point": string, "status": "full|partial|missing|contradicted|not_verifiable", "reason": string}
  ],
  "failure_class": "none|retrieval-limited|generation-limited|pipeline-fallback|mixed",
  "major_errors": [string],
  "summary": string
}

Scoring guidance: score_0_100 should reflect factual correctness (35%),
completeness (25%), grounding (20%), citations (10%), and clarity (10%).
Use the full 0-100 range. Do not award completeness for merely repeating
retrieved text without answering the question."""

    user_payload = {
        "case_id": benchmark_case.get("id"),
        "category": benchmark_case.get("category"),
        "difficulty": benchmark_case.get("difficulty"),
        "language": benchmark_case.get("language"),
        "question": benchmark_case.get("question"),
        "benchmark_notes": benchmark_case.get("notes"),
        "required_answer_points": expected_points,
        "required_terms": scoring.get("required_terms") or [],
        "term_aliases": scoring.get("term_aliases") or {},
        "required_point_count": scoring.get("required_point_count"),
        "required_citation_count": scoring.get("required_citation_count"),
        "expected_citations": expected_citations,
        "retrieval_metrics": replay_case.get("retrieval"),
        "answer_provider": generation.get("answer_provider"),
        "answer_model": generation.get("answer_model"),
        "answer_warnings": generation.get("warnings") or [],
        "answer": generation.get("answer_text") or "",
        "answer_citations": generation.get("citations") or [],
        "retrieved_evidence": retrieved,
    }
    user = "Evaluate this replay case. Treat the required answer points as the rubric, not as text that the answer may claim without evidence.\n\n" + json.dumps(
        user_payload,
        ensure_ascii=False,
        indent=2,
    )
    return system, user


def _parse_json(content: str) -> dict[str, Any]:
    parsed = AgentSynthesizer._parse_deepseek_json(content)
    if not isinstance(parsed, dict):
        raise ValueError("judge response must be a JSON object")
    return parsed


def _normalise_judgment(value: dict[str, Any], *, case_id: str) -> dict[str, Any]:
    def integer(key: str, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value.get(key, 0))
        except (TypeError, ValueError):
            parsed = 0
        return max(minimum, min(maximum, parsed))

    score = integer("score_0_100", 0, 100)
    rows: list[dict[str, str]] = []
    raw_rows = value.get("point_judgments")
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "not_verifiable").strip().lower()
            if status not in {"full", "partial", "missing", "contradicted", "not_verifiable"}:
                status = "not_verifiable"
            rows.append(
                {
                    "point": str(row.get("point") or ""),
                    "status": status,
                    "reason": str(row.get("reason") or ""),
                }
            )
    failure_class = str(value.get("failure_class") or "mixed").strip().lower()
    if failure_class not in {
        "none",
        "retrieval-limited",
        "generation-limited",
        "pipeline-fallback",
        "mixed",
    }:
        failure_class = "mixed"
    errors = value.get("major_errors")
    if isinstance(errors, str):
        errors = [errors]
    if not isinstance(errors, list):
        errors = []
    return {
        "case_id": case_id,
        "score_0_100": score,
        "correctness_0_4": integer("correctness_0_4", 0, 4),
        "completeness_0_4": integer("completeness_0_4", 0, 4),
        "groundedness_0_4": integer("groundedness_0_4", 0, 4),
        "citation_quality_0_4": integer("citation_quality_0_4", 0, 4),
        "style_0_4": integer("style_0_4", 0, 4),
        "point_judgments": rows,
        "failure_class": failure_class,
        "major_errors": [str(error) for error in errors],
        "summary": str(value.get("summary") or ""),
    }


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [int(row["score_0_100"]) for row in results]
    by_category: dict[str, list[int]] = {}
    for row in results:
        by_category.setdefault(str(row.get("category") or "unknown"), []).append(
            int(row["score_0_100"])
        )
    return {
        "judged_cases": len(results),
        "average_score_0_100": round(sum(scores) / len(scores), 2) if scores else None,
        "median_score_0_100": sorted(scores)[len(scores) // 2] if scores else None,
        "pass_rate_at_70": round(sum(score >= 70 for score in scores) / len(scores), 6)
        if scores
        else None,
        "failure_class_counts": dict(Counter(str(row.get("failure_class")) for row in results)),
        "category_average_score_0_100": {
            key: round(sum(values) / len(values), 2)
            for key, values in by_category.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-item-limit", type=int, default=2400)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    benchmark_cases = {
        str(case.get("id")): case for case in benchmark.get("cases") or []
    }
    settings = get_settings()
    api_key = getattr(settings, "deepseek_api_key", None)
    if not api_key or str(api_key).strip().upper() in {"CHANGE_ME", "YOUR_API_KEY"}:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    client = DeepSeekClient(
        base_url=getattr(settings, "deepseek_base_url", None),
        api_key=api_key,
        model=getattr(settings, "deepseek_model", "deepseek-chat"),
        timeout=getattr(settings, "generation_timeout_seconds", 90),
        max_retries=getattr(settings, "generation_max_retries", 1),
        retry_backoff_seconds=getattr(settings, "generation_retry_backoff_seconds", 0.5),
    )

    judgments: list[dict[str, Any]] = []
    for position, replay_case in enumerate(replay.get("cases") or [], start=1):
        case_id = str(replay_case.get("id") or "")
        benchmark_case = benchmark_cases.get(case_id)
        if benchmark_case is None:
            print(f"[{position}] {case_id}: skipped (not in benchmark)", flush=True)
            continue
        print(f"[{position}/{len(replay.get('cases') or [])}] {case_id} semantic judge", flush=True)
        system, user = _build_prompt(
            benchmark_case,
            replay_case,
            evidence_item_limit=max(200, args.evidence_item_limit),
        )
        started = time.perf_counter()
        raw_content = ""
        error: str | None = None
        response_meta: dict[str, Any] = {}
        try:
            generated = client.generate_chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_output_tokens=args.max_output_tokens,
                response_format={"type": "json_object"},
            )
            raw_content = str(generated.get("content") or "")
            response_meta = {
                "provider": generated.get("provider"),
                "model": generated.get("model"),
                "usage": generated.get("usage"),
                "usage_source": generated.get("usage_source"),
            }
            judgment = _normalise_judgment(_parse_json(raw_content), case_id=case_id)
        except Exception as exc:  # noqa: BLE001 - preserve one failed case and continue
            error = f"{type(exc).__name__}: {exc}"
            judgment = {
                "case_id": case_id,
                "score_0_100": None,
                "correctness_0_4": None,
                "completeness_0_4": None,
                "groundedness_0_4": None,
                "citation_quality_0_4": None,
                "style_0_4": None,
                "point_judgments": [],
                "failure_class": "mixed",
                "major_errors": [error],
                "summary": "Semantic judge failed for this case.",
            }
        judgment.update(
            {
                "category": benchmark_case.get("category"),
                "difficulty": benchmark_case.get("difficulty"),
                "latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                "error": error,
                "response": response_meta,
            }
        )
        judgments.append(judgment)
        # Checkpoint after every case so a transient provider failure never
        # discards already-paid-for judgments.
        output = {
            "benchmark_id": benchmark.get("benchmark_id"),
            "benchmark_version": benchmark.get("version"),
            "created_at": now_iso(),
            "source_replay": str(args.replay),
            "judge_model": client.model,
            "summary": _summary([row for row in judgments if row.get("score_0_100") is not None]),
            "judgments": judgments,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    final = json.loads(args.output.read_text(encoding="utf-8"))
    final["summary"] = _summary(
        [row for row in final.get("judgments") or [] if row.get("score_0_100") is not None]
    )
    final["completed_at"] = now_iso()
    args.output.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
