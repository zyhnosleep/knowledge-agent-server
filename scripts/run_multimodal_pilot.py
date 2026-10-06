"""Ingest public PDFs with the project's canonical pipeline, then evaluate dev.

Gold labels are used only AFTER retrieval/generation for page-hit diagnostics.
The original supplied-page refusal labels are not valid gold labels for a
multi-page retrieval run, so this runner never assigns answer/refusal accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from sqlalchemy import select

from app.db.session import SessionLocal, init_db
from app.models.records import DocumentChunk
from app.services.ingestion_stages import IngestionStageRunner, INGESTION_STAGES
from app.services.multimodal_rag import MultimodalRAG, PROMPT_VERSION
from app.services.pipeline import IngestionPipeline
from app.services.paper_profile import ensure_paper_profile, ensure_source_identity


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def status(root, stage, **fields):
    record = {"stage": stage, "pid": os.getpid(), "time": time.time(), **fields}
    write_json(root / "status.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "predictions.jsonl"
    if predictions_path.exists():
        raise RuntimeError("Refusing to overwrite or mix an existing predictions file")
    deadline = time.monotonic() + 3600
    while True:
        try:
            response = httpx.get("http://127.0.0.1:18080/health", timeout=3, trust_env=False)
            response.raise_for_status()
            if response.json().get("embedding_loaded"):
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("Embedding backend did not become ready within one hour")
        status(args.output, "waiting_for_embedding_backend")
        time.sleep(10)
    init_db()
    papers = json.loads((args.corpus / "papers.json").read_text(encoding="utf-8"))
    inventory = []
    with SessionLocal() as db:
        pipeline = IngestionPipeline(db)
        runner = IngestionStageRunner(db, handlers=pipeline.ingestion_stage_handlers(),
                                     pre_stage_validator=pipeline.validate_ingestion_identity)
        for paper in papers:
            source = args.corpus / paper["pdf_path"]
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != paper["sha256"]:
                raise ValueError(f"PDF checksum mismatch: {source.name}")
            _, document, _ = pipeline.register_document(args.project, "Multimodal Paper Pilot", source)
            # Identity/title metadata is source catalog metadata, not QA labels.
            document.title = paper["title"]
            document.metadata_json = {**(document.metadata_json or {}),
                                      "evaluation_paper_id": paper["paper_id"]}
            db.commit()
            version = pipeline._get_or_create_parse_version(document)
            db.commit()
            for stage in INGESTION_STAGES:
                status(args.output, "ingesting", paper=paper["paper_id"], ingestion_stage=stage)
                runner.run_stage(document.id, version.version_key, stage, enqueue_next=False)
            db.refresh(document)
            document.title = paper["title"]
            document.source_uri = paper["source_url"]
            metadata = dict(document.metadata_json or {})
            metadata.pop("paper_profile", None)
            document.metadata_json = metadata
            ensure_source_identity(document, preferred_title=paper["title"])
            ensure_paper_profile(document)
            db.commit()
            children = db.scalars(select(DocumentChunk).where(
                DocumentChunk.document_id == document.id,
                DocumentChunk.parse_version == version.version_key,
                DocumentChunk.chunk_role == "child")).all()
            if not children or any(len(child.embedding or []) != 2560 for child in children):
                raise RuntimeError("Incomplete 2560-dimensional child embeddings")
            inventory.append({"paper_id": paper["paper_id"], "document_id": document.id,
                              "sha256": digest, "pages": paper["pages"],
                              "child_chunks": len(children), "parse_version": version.version_key,
                              "status": document.status})
            write_json(args.output / "ingestion.json", inventory)

        questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line]
        questions = [q for q in questions if q["split"] == "dev" and q.get("evaluation_subset") == "main"]
        if len(questions) != 18:
            raise ValueError("Expected exactly 18 main dev questions")
        manifest = {"project": args.project, "split": "dev", "questions": 18, "requests": 36,
                    "modes": ["text", "text_image"], "max_pages": 3,
                    "prompt_version": PROMPT_VERSION, "generator": "Qwen/Qwen3-VL-8B-Instruct",
                    "embedding": "Qwen/Qwen3-Embedding-4B", "dimensions": 2560,
                    "ingestion": "IngestionPipeline canonical stages, synchronous",
                    "pdf_parser": "pypdf_text_layer; MinerU/DI disabled for initial baseline",
                    "retriever": "QueryService.retrieve_evidence; no gold document/page scope",
                    "source_questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                    "answer_scoring": "pending manual reassessment against retrieved multi-page evidence",
                    "performance_note": "GPU allocation includes resident embedding and generation models"}
        write_json(args.output / "manifest.json", manifest)
        service = MultimodalRAG(db)
        with predictions_path.open("x", encoding="utf-8") as out, (args.output / "retrievals.jsonl").open("x", encoding="utf-8") as traces:
            completed = 0
            for row in questions:
                # The original question falsely presupposes one supplied page.
                # Change only that generic scope phrase, preserving question content.
                question = row["question"].replace("的给定证据页面：", "，请依据检索到的论文页面回答：")
                question = question.replace("该页", "检索到的页面")
                status(args.output, "retrieving", completed=completed, total=36, current_id=row["id"])
                retrieved = service.retrieve(args.project, question, max_pages=3)
                traces.write(json.dumps({"id": row["id"], **retrieved}, ensure_ascii=False) + "\n")
                traces.flush()
                for mode in ("text", "text_image"):
                    status(args.output, "inference_started", completed=completed, total=36,
                           current_id=row["id"], mode=mode,
                           retrieved_page_ids=[p["page_id"] for p in retrieved["pages"]])
                    result = service.generate(retrieved, mode)
                    record = {"id": row["id"], "split": "dev", **result,
                              "retrieved_page_ids": [p["page_id"] for p in retrieved["pages"]],
                              "retrieval_seconds": retrieved["retrieval_seconds"],
                              "original_gold_page": row["evidence_page_id"],
                              "original_gold_page_hit": row["evidence_page_id"] in [p["page_id"] for p in retrieved["pages"]],
                              "original_answerability": row["answerability"],
                              "answer_correctness": "unscored_retrieved_scope"}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    completed += 1
                    status(args.output, "running", completed=completed, total=36)
    status(args.output, "complete", completed=36, total=36)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", default="multimodal-pilot")
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as exc:
        status(arguments.output, "failed", error_type=type(exc).__name__, error=str(exc))
        raise
