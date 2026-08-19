#!/usr/bin/env python3
"""BM25 补充路分数天花板校准 sweep（2026-08-18，R3 退化修复）。

SciFact r3 实测：BM25 归一化到 10.0 时，词法像但无关的文档以满分挤掉强向量
命中（12/50 退化，8 条 nDCG 1.0→0.0）。修复引入 BM25_SUPPLEMENT_SCORE_CEIL
（每文档 1 chunk + 天花板归一化）。本脚本对前 50 条 test claims（与 r3 同序）
在多个 CEIL 值下测干净 nDCG@10 / Recall@10，选出不退化 R1（0.7178/0.8067）
且保留抢救能力的值。

用法（服务器）：
    cd ~/knowledge-agent-dev && set -a && . runtime/app.env && set +a && \
    PYTHONPATH=src .venv/bin/python runtime/sci-fact-bench/bm25_ceil_sweep.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import app.services.search as search_mod
from app.db.session import SessionLocal
from app.services.search import QueryService
from sqlalchemy import select
from app.models.records import Document, Project

BASE = Path.home() / "knowledge-agent-dev/runtime/sci-fact-bench/beir-scifact/scifact"
PROJECT_SLUG = "sci-fact-bench"


def load_queries() -> dict[str, str]:
    out = {}
    for line in (BASE / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        out[d["_id"]] = d["text"]
    return out


def load_qrels(split: str) -> dict[str, dict[str, int]]:
    rel: dict[str, dict[str, int]] = {}
    path = BASE / "qrels" / f"{split}.tsv"
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        q, c, s = line.split("\t")
        rel.setdefault(q, {})[c] = int(s)
    return rel


def dcg_at_k(rels: list[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))


def main() -> int:
    queries = load_queries()
    qrels = load_qrels("test")
    db = SessionLocal()
    proj = db.scalar(select(Project).where(Project.slug == PROJECT_SLUG))
    if proj is None:
        print("project not found", file=__import__("sys").stderr)
        return 2
    doc_to_pmid: dict[str, str] = {}
    for d in db.scalars(select(Document).where(Document.project_id == proj.id)):
        name = (d.file_name or "").rsplit("/", 1)[-1]
        pmid = name.split("-")[0]
        if pmid.isdigit():
            doc_to_pmid[d.id] = pmid
    order = sorted(qrels.keys())[:50]

    service = QueryService(db)
    # 4 个 CEIL 值共享同一批 claim 嵌入（服务器 GPU 与外部任务共用时 embed 很慢，
    # 缓存避免每个 CEIL 各 embed 一遍——50 claims 总共只 embed 50 次）。
    # 注意：必须先保存原始 embed 再替换（2026-08-18 教训——直接替换后
    # cached_embed 内调用的是自己，无限递归，sweep 全程空向量跑垃圾数据）。
    original_embed = service.ollama.embed
    embed_cache: dict[str, list[float]] = {}

    def cached_embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = embed_cache.get(t)
            if v is None:
                v = original_embed([t])[0]
                embed_cache[t] = v
            out.append(v)
        return out

    for ceil in (4.0, 5.0, 6.0, 8.0):
        search_mod.BM25_SUPPLEMENT_SCORE_CEIL = ceil
        service.ollama.embed = cached_embed
        ndcgs, recalls = [], []
        for idx, qid in enumerate(order, 1):
            claim = queries.get(qid, "")
            expected = {p for p, s in qrels[qid].items() if s >= 1}
            pack = service.retrieve_evidence(PROJECT_SLUG, claim, limit=10)
            ranked_pmids = list(
                dict.fromkeys(
                    p
                    for p in (doc_to_pmid.get(it.document_id or "") for it in pack.items)
                    if p
                )
            )
            rel = [1 if p in expected else 0 for p in ranked_pmids]
            ndcgs.append(dcg_at_k(rel, 10) / (dcg_at_k([1] * len(expected), 10) or 1.0))
            recalls.append(sum(rel) / (len(expected) or 1))
        n = len(ndcgs)
        print(
            f"CEIL={ceil:.1f}: ndcg@10={sum(ndcgs)/n:.4f} recall@10={sum(recalls)/n:.4f}"
            f"  (R1 对照 0.7178/0.8067)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
