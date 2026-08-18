#!/usr/bin/env python3
"""SciFact (BEIR) 检索评测：test claims → 检索 → nDCG@10 / Recall@10。

--mode 四路对照（spec 2026-08-17-retrieval-layer-improvement-design.md §3）：
    bm25   : 标准 BM25（自实现，系统 _tokenize 分词），584 全库文档级检索
    vector : 纯向量全库检索（无路由、无词法 bonus），top-k 文档聚合
    routed : 系统检索链路（retrieve_evidence）——现状（旧代码）/ 改进后（新代码），
             即四路对照中的 ③ 与 ④（词法路由 ∪ 全库向量补充）

运行方式（服务器）：
    cd ~/knowledge-agent-dev && set -a && . runtime/app.env && set +a && \
    PYTHONPATH=src .venv/bin/python runtime/sci-fact-bench/eval_scifact.py \
        --mode bm25 --report runtime/sci-fact-bench/eval-bm25.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.records import Document, DocumentChunk, Project
from app.services.ai import OllamaClient
from app.services.search import QueryService
from app.services.vector_store import get_vector_store

BASE = Path.home() / "knowledge-agent-dev/runtime/sci-fact-bench/beir-scifact/scifact"
PROJECT_SLUG = "sci-fact-bench"

TOKENIZE = QueryService._tokenize
K1 = 1.5
B = 0.75


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


def load_corpus(db) -> dict[str, str]:
    """project 全部 ready 文档 → {document_id: 全文文本}。"""
    project = db.scalar(select(Project).where(Project.slug == PROJECT_SLUG))
    if project is None:
        raise SystemExit(f"project {PROJECT_SLUG} not found")
    docs = list(db.scalars(select(Document).where(Document.project_id == project.id)))
    doc_text: dict[str, str] = {}
    for doc in docs:
        chunks = db.scalars(
            select(DocumentChunk.text).where(DocumentChunk.document_id == doc.id)
        ).all()
        doc_text[doc.id] = " ".join(
            part for part in [doc.title or "", *[str(t or "") for t in chunks]] if part
        )
    return doc_text


def build_bm25_index(doc_text: dict[str, str]) -> tuple[dict[str, Counter[str]], dict[str, float], float]:
    """文档词频表 + IDF + 平均文档长度。"""
    tokenized = {doc_id: TOKENIZE(text) for doc_id, text in doc_text.items()}
    term_doc_count: Counter[str] = Counter()
    for terms in tokenized.values():
        term_doc_count.update(terms)
    n_docs = len(tokenized) or 1
    idf = {
        term: math.log((n_docs - count + 0.5) / (count + 0.5) + 1.0)
        for term, count in term_doc_count.items()
    }
    doc_lens = {doc_id: len(terms) for doc_id, terms in tokenized.items()}
    avgdl = (sum(doc_lens.values()) / len(doc_lens)) if doc_lens else 1.0
    return tokenized, idf, avgdl


def bm25_rank(
    query: str,
    tokenized: dict[str, Counter[str]],
    idf: dict[str, float],
    avgdl: float,
    topk: int = 10,
) -> list[str]:
    query_terms = TOKENIZE(query)
    scored: list[tuple[float, str]] = []
    for doc_id, terms in tokenized.items():
        score = 0.0
        dl = len(terms)
        for term in query_terms & terms:
            tf = 1  # _tokenize 返回集合，无词频；文档级近似 tf=1
            score += idf.get(term, 0.0) * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avgdl))
        if score > 0:
            scored.append((score, doc_id))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc_id for _score, doc_id in scored[:topk]]


def vector_rank(db, client: OllamaClient, query: str, topk: int = 10) -> list[str]:
    """纯向量全库 top-k 文档：top-100 chunks → 按文档取最佳距离 → top-k。"""
    question_vector = client.embed([query])[0]
    vector_store = get_vector_store(db)
    hits = vector_store.search(question_vector, limit=max(topk * 20, 50))
    if not hits:
        return []
    chunk_ids = [hit.chunk_id for hit in hits]
    rows = db.execute(
        select(DocumentChunk.id, DocumentChunk.document_id).where(
            DocumentChunk.id.in_(chunk_ids)
        )
    ).all()
    chunk_to_doc = {str(row[0]): str(row[1]) for row in rows}
    best_by_doc: dict[str, float] = {}
    for hit in hits:
        doc_id = chunk_to_doc.get(hit.chunk_id)
        if not doc_id:
            continue
        if doc_id not in best_by_doc or hit.distance < best_by_doc[doc_id]:
            best_by_doc[doc_id] = hit.distance
    ranked = sorted(best_by_doc.items(), key=lambda item: item[1])
    return [doc_id for doc_id, _distance in ranked[:topk]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 = 全部 claims")
    parser.add_argument("--mode", default="routed", choices=["bm25", "vector", "routed"])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    queries = load_queries()
    qrels = load_qrels(args.split)

    db = SessionLocal()
    doc_to_pmid: dict[str, str] = {}
    project = db.scalar(select(Project).where(Project.slug == PROJECT_SLUG))
    if project is None:
        print(f"project {PROJECT_SLUG} not found", file=sys.stderr)
        return 2
    docs = list(db.scalars(select(Document).where(Document.project_id == project.id)))
    for d in docs:
        name = (d.file_name or "").rsplit("/", 1)[-1]
        pmid = name.split("-")[0]
        if pmid.isdigit():
            doc_to_pmid[d.id] = pmid
    print(f"project docs={len(docs)} mapped_to_pmid={len(doc_to_pmid)} mode={args.mode}")

    service = QueryService(db)
    client = OllamaClient()
    if args.mode == "bm25":
        doc_text = load_corpus(db)
        tokenized, idf, avgdl = build_bm25_index(doc_text)
    order = sorted(qrels.keys())
    if args.limit > 0:
        order = order[: args.limit]

    rows, ndcgs, recalls = [], [], []
    for idx, qid in enumerate(order, 1):
        claim = queries.get(qid, "")
        if not claim:
            continue
        expected = {pmid for pmid, score in qrels[qid].items() if score >= 1}
        if args.mode == "bm25":
            ranked_pmids = [doc_to_pmid.get(doc_id, "") for doc_id in bm25_rank(claim, tokenized, idf, avgdl, args.topk)]
            ranked_pmids = [p for p in ranked_pmids if p]
            n_evidence = len(ranked_pmids)
        elif args.mode == "vector":
            ranked_pmids = [doc_to_pmid.get(doc_id, "") for doc_id in vector_rank(db, client, claim, args.topk)]
            ranked_pmids = [p for p in ranked_pmids if p]
            n_evidence = len(ranked_pmids)
        else:  # routed：系统链路（旧代码=现状 ③，新代码=改进后 ④）
            pack = service.retrieve_evidence(PROJECT_SLUG, claim, limit=args.topk)
            ranked_pmids = [doc_to_pmid.get(it.document_id or "") for it in pack.items]
            ranked_pmids = [p for p in ranked_pmids if p]
            n_evidence = len(pack.items)
        rel_scores = [1 if p in expected else 0 for p in ranked_pmids]
        hits = sum(rel_scores)
        ndcg = dcg_at_k(rel_scores, args.topk) / (dcg_at_k([1] * len(expected), args.topk) or 1.0)
        recall = hits / (len(expected) or 1)
        ndcgs.append(ndcg)
        recalls.append(recall)
        rows.append({"query_id": qid, "expected": sorted(expected), "retrieved": ranked_pmids, "ndcg@k": round(ndcg, 4), "recall@k": round(recall, 4), "n_evidence": n_evidence})
        if idx % 25 == 0:
            print(f"[{idx}/{len(order)}] running ndcg@10={sum(ndcgs)/len(ndcgs):.4f} recall@10={sum(recalls)/len(recalls):.4f}", flush=True)

    summary = {
        "split": args.split,
        "topk": args.topk,
        "mode": args.mode,
        "queries": len(ndcgs),
        "ndcg@k": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
        "recall@k": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "zero_hit_queries": sum(1 for r in rows if not r["retrieved"]),
        "expected_docs_available": len(doc_to_pmid),
    }
    report = {"summary": summary, "rows": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
