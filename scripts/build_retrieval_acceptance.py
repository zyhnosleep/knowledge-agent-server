from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.records import Document, DocumentChunk, Project
from app.services.rag_adapter import RAGAdapter


QueryFunction = Callable[[dict[str, Any]], dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def build_report(
    cases: list[dict[str, Any]], query: QueryFunction, *, top_k: int
) -> dict[str, Any]:
    if len(cases) != 20:
        raise ValueError("Retrieval acceptance requires exactly 20 fixed cases")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    rows: list[dict[str, Any]] = []
    valid_citations = 0
    total_citations = 0
    for case in cases:
        result = query(case)
        expected = {str(value) for value in case["expected_document_ids"]}
        retrieved = [
            str(value) for value in result.get("retrieved_document_ids", [])
        ][:top_k]
        citations = list(result.get("citations", []))
        citation_pages = [citation.get("page_label") for citation in citations]
        total_citations += len(citations)
        valid_citations += sum(
            1
            for citation in citations
            if str(citation.get("document_id")) in expected
            and bool(citation.get("page_label"))
        )
        passed = bool(expected.intersection(retrieved))
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_document_ids": sorted(expected),
                "retrieved_document_ids": retrieved,
                "top_k": top_k,
                "citation_pages": citation_pages,
                "latency_ms": int(result.get("latency_ms", 0)),
                "passed": passed,
            }
        )

    return {
        "created_at": _utc_now(),
        "cases": rows,
        "summary": {
            "recall_at_k": round(
                sum(1 for row in rows if row["passed"]) / len(rows), 4
            ),
            "citation_validity": round(
                valid_citations / total_citations if total_citations else 0.0, 4
            ),
            "p95_latency_ms": _p95([row["latency_ms"] for row in rows]),
        },
    }


def capture_cases(db: Session, *, project_slug: str) -> list[dict[str, Any]]:
    rows = db.execute(
        select(Document, DocumentChunk)
        .join(DocumentChunk, DocumentChunk.document_id == Document.id)
        .join(Project, Project.id == Document.project_id)
        .where(Project.slug == project_slug)
        .order_by(Document.id, DocumentChunk.ordinal)
        .limit(20)
    ).all()
    if len(rows) != 20:
        raise ValueError(
            f"Project {project_slug!r} needs at least 20 chunks for acceptance capture"
        )
    return [
        {
            "id": f"retrieval-{index + 1:02d}",
            "question": (
                f"请根据《{document.title}》说明第 "
                f"{chunk.page_label or chunk.ordinal + 1} 页所讨论的核心内容。"
            ),
            "expected_document_ids": [document.id],
            "source_page_label": chunk.page_label,
        }
        for index, (document, chunk) in enumerate(rows)
    ]


def run_cases(
    db: Session, cases: list[dict[str, Any]], *, project_slug: str, top_k: int
) -> dict[str, Any]:
    rag = RAGAdapter()

    def query(case: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        response = rag.answer(db, project_slug, case["question"])
        citations = [citation.model_dump() for citation in response.citations]
        return {
            "retrieved_document_ids": [
                citation["document_id"] for citation in citations
            ],
            "citations": citations,
            "latency_ms": int((time.monotonic() - started) * 1000),
        }

    return build_report(cases, query, top_k=top_k)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and run retrieval acceptance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--project", required=True)
    capture.add_argument("--out", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--project", required=True)
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with SessionLocal() as db:
        if args.command == "capture":
            _write_json(args.out, {"cases": capture_cases(db, project_slug=args.project)})
            return 0
        payload = json.loads(args.cases.read_text(encoding="utf-8"))
        report = run_cases(
            db,
            payload["cases"],
            project_slug=args.project,
            top_k=args.top_k,
        )
        _write_json(args.out, report)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
