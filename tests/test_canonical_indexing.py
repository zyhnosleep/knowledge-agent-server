from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import Document, DocumentChunk, DocumentParseVersion, Project
from app.services.ingestion_stages import IngestionStageRunner
from app.services.pipeline import IngestionPipeline


@pytest.fixture
def db() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False, future=True)()
    session.add(Project(id="p1", slug="research", name="Research"))
    session.add(
        Document(
            id="d1",
            project_id="p1",
            title="Paper",
            file_name="paper.pdf",
            sha256="abc",
            raw_path="raw/paper.pdf",
            status="ready",
            active_parse_version="old-v",
        )
    )
    session.add_all(
        [
            DocumentParseVersion(
                id="old-version",
                document_id="d1",
                version_key="old-v",
                artifact_dir="old",
                status="active",
            ),
            DocumentParseVersion(
                id="new-version",
                document_id="d1",
                version_key="new-v",
                artifact_dir="new",
                status="embedding",
            ),
        ]
    )
    session.commit()
    return session


def _record(local_id: str, *, embedding: list[float] | None) -> dict:
    role = "parent" if embedding is None else "child"
    return {
        "chunk": {
            "local_id": local_id,
            "parse_version": "new-v",
            "chunk_role": role,
            "block_type": "narrative",
            "text": f"Source text for {local_id}",
            "embedding_text": (
                f"Context for {local_id}.\n\nSource text for {local_id}"
                if role == "child"
                else f"Source text for {local_id}"
            ),
            "token_count": 8,
            "parent_local_id": "parent-1" if role == "child" else None,
            "previous_child_local_id": None,
            "next_child_local_id": None,
            "source_block_ids": ["block-1"],
            "source_spans": [{"page_index": 0, "page_label": "1"}],
            "section_path": ["Methods"],
            "ordinal": 0 if role == "parent" else int(local_id[-1]),
            "splitter_name": "section_aware_semantic",
            "splitter_version": "semantic-v1",
            "splitting_model": "test-model",
            "semantic_boundary_score": None,
            "metadata": {},
            **(
                {
                    "contextual_prefix": f"Context for {local_id}.",
                    "contextualization_model": "context-model",
                    "contextualization_version": "context-v1",
                    "contextualization_prompt_version": "prompt-v1",
                    "contextualized_at": "2026-07-24T12:00:00",
                }
                if role == "child"
                else {}
            ),
        },
        "embedding": embedding,
    }


def _write_artifact(directory: Path, name: str, payload: object) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    return {
        "artifact_path": str(path),
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _prepare_index_runner(
    db: Session,
    tmp_path: Path,
    monkeypatch,
    records: list[dict],
    *,
    dimensions: int = 2,
):
    from app.services import pipeline as pipeline_module

    root = tmp_path / "artifacts"
    monkeypatch.setattr(pipeline_module.settings, "canonical_artifacts_dir", root)
    monkeypatch.setattr(pipeline_module.settings, "ollama_embedding_dimensions", dimensions)
    directory = root / "d1" / "new-v.pipeline"
    embed_output = {
        **_write_artifact(directory, "embedded_chunks.json", records),
        "chunk_count": len(records),
        "child_count": sum(item["chunk"]["chunk_role"] == "child" for item in records),
        "embedded_count": sum(item["embedding"] is not None for item in records),
        "dimensions": dimensions,
    }
    version = db.get(DocumentParseVersion, "new-version")
    version.stage_state = {
        stage: {"status": "completed", "output": {}}
        for stage in ("parse", "repair", "canonicalize", "semantic_split", "contextualize")
    }
    version.stage_state["embed"] = {"status": "completed", "output": embed_output}
    db.commit()
    pipeline = IngestionPipeline(db)
    return IngestionStageRunner(db, handlers=pipeline.ingestion_stage_handlers())


def test_index_writes_only_complete_contextual_child_vectors(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    records = [_record("parent-1", embedding=None), _record("child-1", embedding=[1.0, 0.0])]
    runner = _prepare_index_runner(db, tmp_path, monkeypatch, records)

    runner.run_stage("d1", "new-v", "index", enqueue_next=False)

    rows = db.query(DocumentChunk).filter_by(document_id="d1", parse_version="new-v").all()
    assert {row.chunk_role for row in rows} == {"parent", "child"}
    child = next(row for row in rows if row.chunk_role == "child")
    parent = next(row for row in rows if row.chunk_role == "parent")
    assert child.parent_chunk_id == parent.id
    assert child.contextual_prefix
    assert child.embedding_text == f"{child.contextual_prefix}\n\n{child.text}"
    assert child.embedding == [1.0, 0.0]
    assert parent.embedding is None


def test_activation_rejects_incomplete_contextual_embeddings_and_keeps_old_active(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    records = [
        _record("parent-1", embedding=None),
        _record("child-1", embedding=[1.0, 0.0]),
        _record("child-2", embedding=[0.0, 1.0]),
    ]
    runner = _prepare_index_runner(db, tmp_path, monkeypatch, records)
    runner.run_stage("d1", "new-v", "index", enqueue_next=False)
    db.get(DocumentChunk, "child-2").embedding = None
    db.commit()

    with pytest.raises(RuntimeError, match="embedding completeness"):
        runner.run_stage("d1", "new-v", "activate", enqueue_next=False)

    db.expire_all()
    assert db.get(Document, "d1").active_parse_version == "old-v"
    assert db.get(DocumentParseVersion, "old-version").status == "active"
    assert db.get(DocumentParseVersion, "new-version").status == "activation_failed"


def test_activation_atomically_switches_only_after_complete_valid_index(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    records = [_record("parent-1", embedding=None), _record("child-1", embedding=[1.0, 0.0])]
    runner = _prepare_index_runner(db, tmp_path, monkeypatch, records)
    runner.run_stage("d1", "new-v", "index", enqueue_next=False)

    runner.run_stage("d1", "new-v", "activate", enqueue_next=False)

    db.expire_all()
    assert db.get(Document, "d1").active_parse_version == "new-v"
    assert db.get(DocumentParseVersion, "old-version").status == "superseded"
    assert db.get(DocumentParseVersion, "new-version").status == "active"
