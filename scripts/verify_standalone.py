#!/usr/bin/env python
"""Smoke the official APIs. HTTP/structure success never becomes accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_adaptive_agent import (
    CONSTRAINTS, Case, assert_loopback_url, capture_context, private_json)
import httpx


def check_agent_result(body, *, require_pgvector=False, require_pixels=False):
    if body.get("status") != "completed" or not body.get("final_answer"):
        raise ValueError("execution_not_completed")
    metadata = body.get("metadata", {})
    backends = [metadata.get("retrieval_backend")]
    backends.extend(step.get("metadata", {}).get("retrieval_backend") for step in body.get("steps", []))
    if require_pgvector and ("pgvector" not in backends or any(b in ("json_cosine", "sqlite-vec") for b in backends)):
        raise ValueError("pgvector_not_executed")
    visual = metadata.get("visual_evidence", {})
    if require_pixels and not visual.get("sent"):
        raise ValueError("pixels_not_executed")
    return {"backend": metadata.get("retrieval_backend"), "pixels_sent": len(visual.get("sent", [])),
        "execution_mode": metadata.get("execution_mode"), "fact_correct": None}


def parse_sse_final(lines):
    event, data, finals = "", [], []
    for line in [*lines, ""]:
        if line == "":
            if event == "final":
                finals.append(json.loads("\n".join(data)))
            event, data = "", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if len(finals) != 1:
        raise ValueError("sse_final_missing" if not finals else "sse_multiple_finals")
    return finals[0]


def _error_code(exc):
    known = {"execution_not_completed", "pgvector_not_executed", "pixels_not_executed",
        "sse_final_missing", "sse_multiple_finals", "plain_answer_missing", "readiness_failed"}
    if isinstance(exc, ValueError) and str(exc) in known:
        return str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_status_{exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "request_timeout"
    return "smoke_check_failed"


def run_smoke(*, base_url, project, cases=None):
    assert_loopback_url(base_url)
    context = capture_context(project)
    checks = []
    if not cases:
        raise ValueError("reviewed_smoke_cases_required")
    with httpx.Client(timeout=135) as client:
        health = client.get(base_url.rstrip("/") + "/api/health")
        health.raise_for_status()
        body = health.json()
        if body.get("status") != "ok" or body.get("models", {}).get("vector_store", {}).get("status") != "ready":
            raise ValueError("readiness_failed")
        for case in cases:
            row = {"id": case.id, "fact_correct": None}
            try:
                pixel = case.group == "pixels"
                if case.setup_questions:
                    row["plain_skipped"] = "stateless_endpoint"
                else:
                    plain = client.post(base_url.rstrip("/") + "/api/query", json={"project_slug": project,
                        "question": case.question, "document_id": case.document_scope, "save_answer": False})
                    plain.raise_for_status()
                    row["plain_response"] = plain.json()
                    if not row["plain_response"].get("answer_markdown"):
                        raise ValueError("plain_answer_missing")
                    row['plain'] = check_agent_result({
                        'status': 'completed', 'final_answer': row['plain_response']['answer_markdown'],
                        'metadata': row['plain_response'].get('metadata', {}),
                    }, require_pgvector=not case.unanswerable, require_pixels=pixel)
                for kind in ("json", "sse"):
                    session_id = None
                    row[kind + "_setup_responses"] = []
                    for index, question in enumerate([*case.setup_questions, case.question]):
                        payload = {"project_slug": project, "query": question, "document_id": case.document_scope,
                            "constraints": dict(CONSTRAINTS)}
                        if session_id:
                            payload["session_id"] = session_id
                        if kind == "json":
                            response = client.post(base_url.rstrip("/") + "/api/agent/query", json=payload)
                            response.raise_for_status()
                            body = response.json()
                        else:
                            with client.stream("POST", base_url.rstrip("/") + "/api/agent/query/stream", json=payload) as stream:
                                stream.raise_for_status()
                                body = parse_sse_final(list(stream.iter_lines()))
                        if index < len(case.setup_questions):
                            row[kind + "_setup_responses"].append(body)
                            check_agent_result(body)
                            session_id = body.get("session_id")
                            if not session_id:
                                raise ValueError("execution_not_completed")
                        else:
                            row[kind + "_response"] = body
                            row[kind] = check_agent_result(body, require_pgvector=not case.unanswerable, require_pixels=pixel)
                row["passed"] = True
            except Exception as exc:
                row.update({"passed": False, "error": _error_code(exc)})
            checks.append(row)
    stable = context == capture_context(project)
    return {"passed": stable and all(c["passed"] for c in checks), "context_stable": stable,
        "context": context, "checks": checks, "accuracy": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        cases = [Case.model_validate(json.loads(line)) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()] if args.cases else None
        report = run_smoke(base_url=args.base_url, project=args.project, cases=cases)
    except Exception:
        report = {"passed": False, "checks": [], "error": "smoke_failed"}
    private_json(args.output / "report.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
