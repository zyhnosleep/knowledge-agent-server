#!/usr/bin/env python3
"""SciFact claim-证据相关性判定评测（train/dev 的 cited_doc_ids 作真值）。

每个 claim：retrieve_evidence top-k → LLM 判定每个证据文档与 claim 的
关系（SUPPORT/CONTRADICT → related；NEI → not_related）→ 与 cited 集合
对比（cited 文档应被判定为 related）。

指标（文档级）：accuracy、related 二分类 macro-F1、precision/recall。
指标（claim 级）：判定命中率（判定出的 related 集合与 cited 集合的交集）。

说明：官方 claims 文件（2021 版）不含 SUPPORT/CONTRADICT/NEI label
（evidence 字段被清空），故本评测以 cited_doc_ids 为"相关"真值，
测的是证据相关性判定能力而非三分类；局限在报告中标明。

运行方式（服务器）：
    cd ~/knowledge-agent-dev && set -a && . runtime/app.env && set +a && \
    PYTHONPATH=src .venv/bin/python runtime/sci-fact-bench/eval_scifact_claims.py \
        --report runtime/sci-fact-bench/claims-eval-report.json [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.records import Document, Project
from app.services.ai import OllamaClient
from app.services.search import QueryService

BASE = Path.home() / "knowledge-agent-dev/runtime/sci-fact-bench/beir-scifact/scifact"
CLAIMS_DIR = Path.home() / "knowledge-agent-dev/runtime/sci-fact-bench/data"
PROJECT_SLUG = "sci-fact-bench"

MAX_EXCERPT_CHARS = 1500


class Verdict(BaseModel):
    """LLM 对 claim-证据关系的判定。"""

    relation: Literal["SUPPORT", "CONTRADICT", "NEI"] = Field(
        description="证据与该科学声明的关系：SUPPORT=支持声明，CONTRADICT=反驳声明，NEI=无信息/不相关"
    )


def load_claims(split: str) -> list[dict]:
    out = []
    for line in (CLAIMS_DIR / f"claims_{split}.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        out.append({"id": str(d["id"]), "claim": d["claim"], "cited": [str(c) for c in d.get("cited_doc_ids", [])]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "train"])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 = 全部 claims")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    claims = load_claims(args.split)
    if args.limit > 0:
        claims = claims[: args.limit]
    print(f"[plan] split={args.split} claims={len(claims)} topk={args.topk}")

    with SessionLocal() as db:
        project = db.scalar(select(Project).where(Project.slug == PROJECT_SLUG))
        if project is None:
            print(f"project {PROJECT_SLUG} not found", file=sys.stderr)
            return 2
        docs = list(db.scalars(select(Document).where(Document.project_id == project.id)))
    doc_to_pmid = {}
    for d in docs:
        name = (d.file_name or "").rsplit("/", 1)[-1]
        pmid = name.split("-")[0]
        if pmid.isdigit():
            doc_to_pmid[d.id] = pmid
    print(f"[plan] project docs={len(docs)} mapped={len(doc_to_pmid)}")

    service = QueryService(SessionLocal())
    client = OllamaClient()

    rows = []
    total_doc_pred = 0
    total_doc_hit = 0  # predicted related == actually cited
    total_cited_docs = 0
    tp = fp = fn = 0  # related 二分类
    claims_with_hit = 0
    t0 = time.monotonic()

    for i, c in enumerate(claims, 1):
        pack = service.retrieve_evidence(PROJECT_SLUG, c["claim"], limit=args.topk)
        cited_set = set(c["cited"])
        judged: list[dict] = []
        predicted_related: set[str] = set()
        for it in pack.items:
            pmid = doc_to_pmid.get(it.document_id or "")
            if not pmid or pmid in {j["pmid"] for j in judged}:
                continue
            excerpt = (it.excerpt or "")[:MAX_EXCERPT_CHARS]
            try:
                v = client.generate_structured(
                    Verdict,
                    system_prompt=(
                        "你是科学事实核查助手。判断提供的证据内容与科学声明的关系："
                        "SUPPORT=证据支持声明；CONTRADICT=证据反驳声明；"
                        "NEI=证据未提及声明内容或与声明无关。只输出 JSON。"
                    ),
                    user_prompt=f"声明: {c['claim']}\n\n证据: {excerpt}",
                )
                relation = v.relation
            except Exception as exc:  # noqa: BLE001
                relation = "NEI"  # 判定失败按 NEI 处理（保守）
                print(f"  [warn] claim={c['id']} doc={pmid} judge failed: {str(exc)[:120]}", flush=True)
            related = relation in ("SUPPORT", "CONTRADICT")
            if related:
                predicted_related.add(pmid)
            judged.append({"pmid": pmid, "relation": relation, "related": related, "cited": pmid in cited_set})
        # 文档级统计
        for j in judged:
            total_doc_pred += 1
            if j["cited"]:
                total_cited_docs += 1
            if j["related"] == j["cited"]:
                total_doc_hit += 1
            if j["related"] and j["cited"]:
                tp += 1
            elif j["related"] and not j["cited"]:
                fp += 1
            elif not j["related"] and j["cited"]:
                fn += 1
        # claim 级：检索+判定是否命中至少一个 cited 文档
        if predicted_related & cited_set:
            claims_with_hit += 1
        rows.append({
            "claim_id": c["id"], "claim": c["claim"][:120], "cited": sorted(cited_set),
            "predicted_related": sorted(predicted_related), "judged": judged,
            "n_evidence": len(pack.items), "status": pack.status,
        })
        if i % 25 == 0:
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            print(
                f"[{i}/{len(claims)}] doc_acc={total_doc_hit}/{total_doc_pred}"
                f" F1={2*prec*rec/(prec+rec) if prec+rec else 0:.3f}"
                f" claim_hit={claims_with_hit}/{i} elapsed={time.monotonic()-t0:.0f}s", flush=True)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    summary = {
        "split": args.split, "topk": args.topk, "claims": len(claims),
        "doc_accuracy": round(total_doc_hit / total_doc_pred, 4) if total_doc_pred else 0.0,
        "related_precision": round(prec, 4), "related_recall": round(rec, 4),
        "related_f1": round(f1, 4),
        "docs_judged": total_doc_pred, "docs_cited_judged": total_cited_docs,
        "claim_hit_rate": round(claims_with_hit / len(claims), 4) if claims else 0.0,
        "note": "真值为 cited_doc_ids（相关判定），非官方三分类 label",
    }
    report = {"summary": summary, "rows": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
