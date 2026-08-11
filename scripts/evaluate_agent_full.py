#!/usr/bin/env python3
"""Evaluate Agent full-pipeline answers against acceptance cases.

Usage:
  python scripts/evaluate_agent_full.py \
    --cases runtime/task15/internal-research-overlap30-full-answer-cases.json \
    --report runtime/task15/dev-new-agent-full30-v1.json \
    --api-url http://127.0.0.1:8002
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def call_agent_api(
    api_url: str,
    case: dict[str, Any],
    timeout: int = 120,
    *,
    project_scope: bool = False,
) -> dict[str, Any]:
    """Call the Agent API endpoint and return the response.

    :param project_scope: True 时强制项目级会话（不传 document_id），
        用于验证项目级弱指代查询的检索锚定回退（T1）；默认取 case 的
        query_document_id（回退 expected_document_id），模拟"进文章提问"
        的锁定场景。
    """
    document_id = None
    if not project_scope:
        document_id = case.get("query_document_id") or case.get(
            "expected_document_id"
        )
    payload = json.dumps({
        "project_slug": case["project_slug"],
        "query": case["question"],
        "session_id": f"eval-{case['id']}-{int(time.time())}",
        "document_id": document_id,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{api_url}/api/agent/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = json.loads(resp.read())
    except Exception as exc:
        data = {"status": "error", "error": str(exc)}
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    data["_elapsed_ms"] = elapsed_ms
    return data


def evaluate_agent_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Score an Agent response against the acceptance case."""
    failures: list[str] = []

    # Check status
    if response.get("status") != "completed":
        failures.append("agent_status")

    # Check answer contains required terms
    answer_text = str(response.get("final_answer") or "")
    required_terms = [str(t) for t in case.get("answer_required_terms", case.get("required_terms", []))]

    answer_passed = None
    missing_terms: list[str] = []
    if required_terms and answer_text:
        answer_lower = answer_text.casefold()
        missing_terms = [
            str(term) for term in required_terms
            if str(term).casefold() not in answer_lower
        ]
        answer_passed = not missing_terms
        if not answer_passed:
            failures.append("answer")

    # Check citations
    citations = response.get("citations") or []
    citation_passed = None
    if citations:
        citation_passed = len(citations) > 0
        if not citation_passed:
            failures.append("citation")

    # rag-direct 和 local-fallback 都没有经过 Agent 二次综合。
    # 其他非空模型（包括真实调用本地 Ollama）才算执行了综合。
    answer_model = response.get("answer_model")
    synthesis_applied = answer_model not in (None, "local-fallback", "rag-direct")

    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "latency_ms": response.get("_elapsed_ms", 0),
        "answer_passed": answer_passed,
        "citation_passed": citation_passed,
        "synthesis_applied": synthesis_applied,
        "answer_provider": response.get("answer_provider"),
        "answer_model": answer_model,
        "missing_terms": missing_terms,
        "answer_text_preview": answer_text[:300],
        "failures": failures,
        "passed": not failures,
    }


def run_agent_evaluation(
    cases: list[dict[str, Any]],
    api_url: str,
    *,
    project_scope: bool = False,
) -> dict[str, Any]:
    """Run all cases through the Agent API."""
    rows = []
    latencies: list[int] = []
    failure_counts: dict[str, int] = {}

    for i, case in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] {case['id']} ...", end=" ", flush=True)
        response = call_agent_api(api_url, case, project_scope=project_scope)
        row = evaluate_agent_case(case, response)
        rows.append(row)

        if row["latency_ms"] > 0:
            latencies.append(row["latency_ms"])

        for f in row["failures"]:
            failure_counts[f] = failure_counts.get(f, 0) + 1

        status = "PASS" if row["passed"] else f"FAIL ({','.join(row['failures'])})"
        print(f"{status} ({row['latency_ms']}ms)")

    total = len(rows)
    answer_rows = [r for r in rows if r["answer_passed"] is not None]
    synthesis_applied = sum(1 for r in rows if r["synthesis_applied"])

    summary = {
        "case_count": total,
        "passed_cases": sum(r["passed"] for r in rows),
        "answer_pass_rate": (
            round(sum(r["answer_passed"] for r in answer_rows) / len(answer_rows), 4)
            if answer_rows else 1.0
        ),
        "total_missing_terms": sum(len(r["missing_terms"]) for r in rows),
        "citation_pass_rate": (
            round(sum(r["citation_passed"] is True for r in rows if r["citation_passed"] is not None) /
                  sum(1 for r in rows if r["citation_passed"] is not None), 4)
            if any(r["citation_passed"] is not None for r in rows) else 1.0
        ),
        "synthesis_applied_rate": round(synthesis_applied / total, 4) if total else 1.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "failure_attribution": dict(sorted(failure_counts.items())),
        "strict_pass": bool(rows) and all(r["passed"] for r in rows),
    }

    return {"created_at": _utc_now(), "cases": rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Agent full-pipeline answers")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8002")
    parser.add_argument(
        "--project-scope",
        action="store_true",
        help="强制项目级会话（不传 document_id），验证弱指代查询的检索锚定回退（T1）",
    )
    args = parser.parse_args()

    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload if isinstance(payload, list) else [])
    if not isinstance(cases, list):
        raise ValueError("Cases file must contain a cases array")

    report = run_agent_evaluation(cases, args.api_url, project_scope=args.project_scope)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {args.report}")
    print(f"Passed: {report['summary']['passed_cases']}/{report['summary']['case_count']}")
    print(f"Synthesis applied: {report['summary']['synthesis_applied_rate']:.0%}")

    return 0 if report["summary"]["strict_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
