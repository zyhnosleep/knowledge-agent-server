#!/usr/bin/env python3
"""BM25 补充路 A/B 对照（2026-08-19）：内部 30 题检索层。

生产现状（有 BM25）vs 关掉 _bm25_supplement_candidates（无 BM25），
同一进程同一 DB 状态双跑（embedding 按文本缓存共享），判定口径 =
内部回归验收口径：expected_document_id 是否在 retrieve_evidence 的证据里。
基线对照：task16-agent-full30-20260819-route-lock-fix.json（28/30 全答，
检索 30/30 正常）。

回答的问题：BM25 补充路在内部语料上有没有存在必要——
without_bm25 全命中 → 无必要（可删）；有题掉 → 有硬证据保留。

用法（服务器）:
    cd ~/knowledge-agent-dev && set -a && . runtime/app.env && set +a && \
    PYTHONPATH=src .venv/bin/python -u runtime/sci-fact-bench/bm25_ab_test.py \
        --report runtime/sci-fact-bench/bm25-ab-report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.records import Project
from app.services.ai import OllamaClient
from app.services.search import QueryService

CASES = Path.home() / "knowledge-agent-dev/runtime/task15/internal-research-overlap30-full-answer-cases.json"

_ORIG_BM25 = QueryService._bm25_supplement_candidates


def _no_bm25(self, question, project_id, limit, question_vector=None):
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("runtime/sci-fact-bench/bm25-ab-report.json"))
    args = parser.parse_args()

    payload = json.loads(CASES.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload)
    if not cases:
        print("no cases", file=sys.stderr)
        return 2
    db = SessionLocal()
    project = db.scalar(select(Project).where(Project.slug == cases[0]["project_slug"]))
    if project is None:
        print(f"project {cases[0]['project_slug']} not found", file=sys.stderr)
        return 2

    service = QueryService(db)
    # embedding 按文本缓存：每题两个变体共享一次 embed（qwen3-embedding 确定性）
    _embed_cache: dict[tuple, list] = {}
    _orig_embed = OllamaClient.embed

    def _cached_embed(self, texts):
        key = tuple(texts) if isinstance(texts, (list, tuple)) else texts
        if key not in _embed_cache:
            _embed_cache[key] = _orig_embed(self, texts)
        return _embed_cache[key]

    OllamaClient.embed = _cached_embed

    rows = []
    for idx, case in enumerate(cases, 1):
        q = case["question"]
        expected = case.get("expected_document_id")
        pack_with = service.retrieve_evidence(case["project_slug"], q, limit=10)
        QueryService._bm25_supplement_candidates = _no_bm25
        try:
            pack_without = service.retrieve_evidence(case["project_slug"], q, limit=10)
        finally:
            QueryService._bm25_supplement_candidates = _ORIG_BM25
        docs_with = [it.document_id for it in pack_with.items]
        docs_without = [it.document_id for it in pack_without.items]
        rows.append(
            {
                "id": case["id"],
                "expected_document_id": expected,
                "with_bm25": {
                    "hit": expected in docs_with,
                    "n": len(docs_with),
                    "docs": docs_with,
                },
                "without_bm25": {
                    "hit": expected in docs_without,
                    "n": len(docs_without),
                    "docs": docs_without,
                },
                "changed": docs_with != docs_without,
            }
        )
        r = rows[-1]
        print(
            f"[{idx}/{len(cases)}] {case['id']}: with={r['with_bm25']['hit']} "
            f"without={r['without_bm25']['hit']} changed={r['changed']} "
            f"(n {r['with_bm25']['n']}/{r['without_bm25']['n']})",
            flush=True,
        )

    hits_with = sum(1 for r in rows if r["with_bm25"]["hit"])
    hits_without = sum(1 for r in rows if r["without_bm25"]["hit"])
    lost = [r["id"] for r in rows if r["with_bm25"]["hit"] and not r["without_bm25"]["hit"]]
    gained = [r["id"] for r in rows if not r["with_bm25"]["hit"] and r["without_bm25"]["hit"]]
    changed_ids = [r["id"] for r in rows if r["changed"]]
    summary = {
        "cases": len(rows),
        "with_bm25_hit": hits_with,
        "without_bm25_hit": hits_without,
        "lost_without_bm25": lost,
        "gained_without_bm25": gained,
        "evidence_changed_cases": len(changed_ids),
        "evidence_changed_ids": changed_ids,
    }
    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline_report": "task16-agent-full30-20260819-route-lock-fix.json",
        "summary": summary,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
