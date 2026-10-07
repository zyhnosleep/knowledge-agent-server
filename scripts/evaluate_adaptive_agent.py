#!/usr/bin/env python
"""Private, sequential, one-arm collection; the operator switches mode locally."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import ipaddress
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool

CONSTRAINTS = {"allow_external_network": False, "max_steps": 12, "max_tool_calls": 6,
    "budget_tokens": 60000, "timeout_seconds": 120}


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    document_scope: str | None
    reference_facts: list[str]
    required_evidence: list[dict]
    unanswerable: StrictBool
    setup_questions: list[str] = Field(default_factory=list)
    group: str = ""


def _aggregate(records: list[dict]) -> dict:
    result = {"run_count": len(records), "error_count": sum(bool(r.get("error")) for r in records),
        "ungraded_count": sum(all(r.get(k) is None for k in ("fact_correct", "citation_correct", "abstention_appropriate")) for r in records)}
    for field, label in (("fact_correct", "fact"), ("citation_correct", "citation"), ("abstention_appropriate", "abstention")):
        values = [r.get(field) for r in records]
        if any(v is not None and type(v) is not bool for v in values):
            raise ValueError("judgment_type_invalid")
        judged = [v for v in values if v is not None]
        result[f"{label}_accuracy"] = sum(judged) / len(judged) if judged else None
        result[f"{label}_judged_count"] = len(judged)
        result[f"{label}_ungraded_count"] = len(values) - len(judged)
    return result


def assert_loopback_url(url: str):
    parsed = urlparse(url)
    try:
        valid = parsed.scheme == "http" and ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        valid = False
    if not valid or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("loopback_url_required")


def capture_context(project: str) -> dict:
    """Hash the actual selected index, not just its configured backend name."""
    from sqlalchemy import select, text
    from app.core.config import get_settings
    from app.db.session import SessionLocal
    from app.models.records import Document, Project
    from app.services.runtime_contract import EmbeddingIdentity, check_pgvector_contract
    settings = get_settings()
    with SessionLocal() as db:
        check_pgvector_contract(db, settings)
        project_id = db.scalar(select(Project.id).where(Project.slug == project))
        if project_id is None:
            raise ValueError("project_missing")
        versions = dict(db.execute(select(Document.id, Document.active_parse_version).where(
            Document.project_id == project_id, Document.active_parse_version.is_not(None))).all())
        digest = hashlib.sha256()
        count = 0
        for row in db.execute(text("""
            SELECT i.chunk_id, i.parse_version, i.embedding::text FROM document_chunk_pgvector_index i
            JOIN documents d ON d.id=i.document_id AND d.active_parse_version=i.parse_version
            WHERE d.project_id=:project ORDER BY i.chunk_id
        """), {"project": project_id}):
            digest.update(json.dumps(list(row), separators=(",", ":")).encode())
            count += 1
        return {"model": settings.ollama_generation_model,
            "embedding_identity": asdict(EmbeddingIdentity.from_settings(settings)),
            "parse_version_map": versions, "index_sha256": digest.hexdigest(), "index_count": count}


def collect_arm(cases, *, client, base_url, project, mode, context):
    assert_loopback_url(base_url)
    if mode not in ("static", "adaptive"):
        raise ValueError("mode_invalid")
    records = []
    for case in cases:
        row = {"id": case.id, "mode": mode, "case": case.model_dump(), "context": context,
            "constraints": dict(CONSTRAINTS), "fact_correct": None, "citation_correct": None,
            "abstention_appropriate": None, "error": None}
        started = time.monotonic()
        session_id = None
        try:
            for question in [*case.setup_questions, case.question]:
                payload = {"project_slug": project, "query": question, "document_id": case.document_scope,
                    "constraints": dict(CONSTRAINTS)}
                if session_id:
                    payload["session_id"] = session_id
                response = client.post(base_url.rstrip("/") + "/api/agent/query", json=payload)
                response.raise_for_status()
                body = response.json()
                row["response"] = body
                session_id = body.get("session_id")
                if body.get("status") != "completed":
                    raise ValueError("execution_not_completed")
            route = (body.get("route") or {}).get("route")
            expected = "static" if route and route != "complex_multi_hop" else mode
            if body.get("metadata", {}).get("execution_mode") != expected:
                raise ValueError("execution_mode_mismatch")
            if context.get("model"):
                targets = {step.get('metadata', {}).get('inference_model')
                    for step in body.get('steps', []) if step.get('metadata', {}).get('inference_model')}
                label = body.get('answer_model')
                # Deterministic/direct finalizers use a strategy label, not a
                # model name. Require their trusted inference target explicitly.
                if targets - {context['model']} or (label != context['model'] and not (
                    label in {'rag-direct', 'local-fallback'} and targets == {context['model']})):
                    raise ValueError("generation_model_mismatch")
            row.update({"response": body, "usage": body.get("usage", {}),
                "stop_reason": body.get("metadata", {}).get("stop_reason"),
                "verification_status": body.get("metadata", {}).get("verification_status")})
        except Exception as exc:
            codes = {"execution_mode_mismatch", "execution_not_completed", "generation_model_mismatch"}
            row["error"] = str(exc) if isinstance(exc, ValueError) and str(exc) in codes else "request_failed"
        row["latency_ms"] = round((time.monotonic() - started) * 1000)
        records.append(row)
    return records


def compare_arms(static, adaptive):
    if len(static) != len(adaptive) or len({r["id"] for r in static}) != len(static):
        raise ValueError("paired_cases_mismatch")
    by_id = {r["id"]: r for r in adaptive}
    for left in static:
        right = by_id.get(left["id"])
        if not right or any(left.get(k) != right.get(k) for k in ("case", "context", "constraints")):
            raise ValueError("paired_context_mismatch")
    return {"static": _aggregate(static), "adaptive": _aggregate(adaptive)}


def private_json(path: Path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    path.chmod(0o600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--project", default="multimodal-pilot")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "adaptive"), required=True)
    args = parser.parse_args(argv)
    assert_loopback_url(args.base_url)
    cases = [Case.model_validate(json.loads(line)) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("case_ids_invalid")
    from app.core.config import get_settings
    if get_settings().agent_adaptive_enabled != (args.mode == "adaptive"):
        raise ValueError("operator_mode_mismatch")
    context = capture_context(args.project)
    if context["model"] != "qwen3-vl:4b":
        raise ValueError("base_model_required")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    with httpx.Client(timeout=135) as client:
        records = collect_arm(cases, client=client, base_url=args.base_url, project=args.project, mode=args.mode, context=context)
    stable = context == capture_context(args.project)
    private_json(args.output / "records.json", records)
    private_json(args.output / "report.json", {"context_stable": stable, "metrics": _aggregate(records),
        "grading": "ungraded; structure passed is not factual correctness"})
    return 0 if stable and not any(r["error"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
