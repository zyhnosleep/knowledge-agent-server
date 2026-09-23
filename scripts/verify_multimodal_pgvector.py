"""Verify the actual PostgreSQL backend, cosine semantics, and frozen dev retrieval."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from sqlalchemy import select, text

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.records import DocumentChunk
from app.services.multimodal_rag import MultimodalRAG
from app.services.vector_store import PGVectorStore, get_vector_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite verification results")
    args.output.mkdir(parents=True)
    traces = [json.loads(line) for line in (args.baseline / "retrievals.jsonl").read_text().splitlines()]
    predictions = {row["id"]: row for row in [json.loads(line) for line in
                   (args.baseline / "predictions.jsonl").read_text().splitlines()]}
    with SessionLocal() as db:
        store = get_vector_store(db)
        assert db.get_bind().dialect.name == "postgresql"
        assert get_settings().vector_store_backend == "pgvector"
        assert isinstance(store, PGVectorStore) and store.available()
        children = db.scalars(select(DocumentChunk).where(DocumentChunk.chunk_role == "child")).all()
        assert len(children) == 363
        assert db.scalar(text("SELECT count(*) FROM document_chunk_pgvector_index")) == len(children)
        probe = children[0]
        hits = store.search(probe.embedding, limit=10)
        assert hits and hits[0].chunk_id == probe.id and abs(hits[0].distance) < 1e-5
        errors = []
        for hit in hits:
            candidate = db.get(DocumentChunk, hit.chunk_id)
            a, b = probe.embedding, candidate.embedding
            cosine = sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
            errors.append(abs(hit.distance - (1-cosine)))
        assert max(errors) < 1e-5, errors
        service = MultimodalRAG(db)
        results = []
        with (args.output / "retrievals.jsonl").open("x", encoding="utf-8") as out:
            for trace in traces:
                retrieval = service.retrieve("multimodal-pilot", trace["question"], 3)
                out.write(json.dumps({"id": trace["id"], **retrieval}, ensure_ascii=False) + "\n")
                row = predictions[trace["id"]]
                page_ids = [page["page_id"] for page in retrieval["pages"]]
                results.append({"id": trace["id"], "original_answerability": row["original_answerability"],
                                "gold_page": row["original_gold_page"], "selected_pages": page_ids,
                                "gold_page_hit": row["original_gold_page"] in page_ids})
        answerable = [r for r in results if r["original_answerability"] == "answerable"]
        report = {"backend": type(store).__name__, "database_dialect": db.get_bind().dialect.name,
                  "pgvector_version": db.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'")),
                  "indexed_children": len(children), "cosine_distance_max_error": max(errors),
                  "answerable_selected_gold_page_hits": sum(r["gold_page_hit"] for r in answerable),
                  "answerable_questions": len(answerable), "retrieval_questions": len(results),
                  "generation_rerun": False, "rows": results,
                  "source_hashes": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    for name in ("src/app/services/search.py", "src/app/services/vector_store.py")}}
    (args.output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
