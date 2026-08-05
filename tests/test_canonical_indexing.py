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
from app.services.paper_profile import paper_profile_retrieval_terms
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


def _record(
    local_id: str,
    *,
    embedding: list[float] | None,
    block_type: str = "table",
) -> dict:
    role = "parent" if embedding is None else "child"
    effective_block_type = "narrative" if role == "parent" else block_type
    contextualized = False
    return {
        "chunk": {
            "local_id": local_id,
            "parse_version": "new-v",
            "chunk_role": role,
            "block_type": effective_block_type,
            "text": f"Source text for {local_id}",
            "embedding_text": (
                f"Context for {local_id}.\n\nSource text for {local_id}"
                if contextualized
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
                if contextualized
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
    # 这些测试聚焦 activate 阶段的嵌入完整性/原子切换/检索画像行为，不构造
    # canonical bundle。typed inventory gate 已在 tests/test_ingestion_stages.py
    # 中被专项覆盖（含缺失 bundle 失败关闭）；此处提供空库存让 gate 在合成
    # 路径上合法通过，避免把缺库误判为 gate 拒绝。
    monkeypatch.setattr(
        pipeline_module.IngestionPipeline,
        "_verify_typed_inventory_gate",
        lambda self, context, version_key: None,
    )
    pipeline = IngestionPipeline(db)
    return IngestionStageRunner(db, handlers=pipeline.ingestion_stage_handlers())


def test_index_writes_mixed_policy_child_vectors(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    records = [
        _record("parent-1", embedding=None),
        _record("child-1", embedding=[1.0, 0.0], block_type="narrative"),
        _record("child-2", embedding=[0.0, 1.0], block_type="table"),
        _record("child-3", embedding=[0.5, 0.5], block_type="figure"),
    ]
    runner = _prepare_index_runner(db, tmp_path, monkeypatch, records)

    runner.run_stage("d1", "new-v", "index", enqueue_next=False)

    rows = db.query(DocumentChunk).filter_by(document_id="d1", parse_version="new-v").all()
    assert {row.chunk_role for row in rows} == {"parent", "child"}
    narrative = db.get(DocumentChunk, "child-1")
    table = db.get(DocumentChunk, "child-2")
    figure = db.get(DocumentChunk, "child-3")
    parent = next(row for row in rows if row.chunk_role == "parent")
    assert narrative.parent_chunk_id == parent.id
    assert narrative.contextual_prefix is None
    assert narrative.embedding_text == narrative.text
    assert narrative.embedding == [1.0, 0.0]
    assert table.parent_chunk_id == parent.id
    assert table.contextual_prefix is None
    assert table.embedding_text == table.text
    assert table.embedding == [0.0, 1.0]
    assert figure.parent_chunk_id == parent.id
    assert figure.contextual_prefix is None
    assert figure.embedding_text == figure.text
    assert figure.embedding == [0.5, 0.5]
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
    records = [
        _record("parent-1", embedding=None),
        _record("child-1", embedding=[1.0, 0.0], block_type="narrative"),
        _record("child-2", embedding=[0.0, 1.0], block_type="table"),
        _record("child-3", embedding=[0.5, 0.5], block_type="figure"),
    ]
    runner = _prepare_index_runner(db, tmp_path, monkeypatch, records)
    runner.run_stage("d1", "new-v", "index", enqueue_next=False)

    runner.run_stage("d1", "new-v", "activate", enqueue_next=False)

    db.expire_all()
    assert db.get(Document, "d1").active_parse_version == "new-v"
    assert db.get(DocumentParseVersion, "old-version").status == "superseded"
    assert db.get(DocumentParseVersion, "new-version").status == "active"
    output = db.get(DocumentParseVersion, "new-version").stage_state["activate"][
        "output"
    ]
    assert output["retrievable"] == 3
    assert output["contextualization_eligible"] == 0
    assert output["contextualized"] == 0
    assert output["plain"] == 3
    assert output["plain_embedded"] == 3


def test_activation_builds_document_retrieval_profile_from_active_children(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    document = db.get(Document, "d1")
    document.raw_text = None
    document.metadata_json = {
        "source_slug": "sources/stable-paper",
        "source_title": "Stable Paper",
        "document_intelligence": {
            "tables": [{"caption": "StaleOnlyTable"}],
            "figures": [{"caption": "StaleOnlyFigure"}],
        },
    }
    records = [
        _record("parent-1", embedding=None),
        _record("child-1", embedding=[1.0, 0.0], block_type="narrative"),
        _record("child-2", embedding=[0.0, 1.0], block_type="table"),
    ]
    runner = _prepare_index_runner(db, tmp_path, monkeypatch, records)
    runner.run_stage("d1", "new-v", "index", enqueue_next=False)

    runner.run_stage("d1", "new-v", "activate", enqueue_next=False)

    db.expire_all()
    refreshed = db.get(Document, "d1")
    assert refreshed.raw_text == (
        "Source text for child-1\n\nSource text for child-2"
    )
    assert refreshed.metadata_json["source_slug"] == "sources/stable-paper"
    assert "document_intelligence" not in refreshed.metadata_json
    assert "StaleOnlyTable" not in refreshed.metadata_json["paper_profile"]["routing_summary"]
    assert refreshed.metadata_json["paper_profile"]["source_slug"] == (
        "sources/stable-paper"
    )
    assert refreshed.metadata_json["paper_profile"]["routing_summary"]


def test_activation_rebuilds_stale_profile_when_active_source_text_changes(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    document = db.get(Document, "d1")
    document.raw_text = "Old active source text."
    document.metadata_json = {
        "source_slug": "sources/stable-paper",
        "source_title": "Stable Paper",
        "paper_profile": {
            "profile_version": "paper-profile-v1",
            "source_text_sha256": hashlib.sha256(
                document.raw_text.encode("utf-8")
            ).hexdigest(),
            "routing_summary": "stale profile from old active version",
            "one_sentence": "Old active source text.",
        },
    }
    records = [
        _record("parent-1", embedding=None),
        _record("child-1", embedding=[1.0, 0.0], block_type="narrative"),
        _record("child-2", embedding=[0.0, 1.0], block_type="table"),
    ]
    runner = _prepare_index_runner(db, tmp_path, monkeypatch, records)
    runner.run_stage("d1", "new-v", "index", enqueue_next=False)

    runner.run_stage("d1", "new-v", "activate", enqueue_next=False)

    refreshed = db.get(Document, "d1")
    expected_fingerprint = hashlib.sha256(
        refreshed.raw_text.encode("utf-8")
    ).hexdigest()
    profile = refreshed.metadata_json["paper_profile"]
    assert profile["source_text_sha256"] == expected_fingerprint
    assert profile["source_text_sha256"] != hashlib.sha256(
        "Old active source text.".encode("utf-8")
    ).hexdigest()
    assert profile["one_sentence"] != "Old active source text."


def test_profile_retrieval_terms_ignore_stale_fingerprint_cache(db: Session) -> None:
    document = db.get(Document, "d1")
    document.raw_text = "FreshUniqueRoutingTerm appears in the active source."
    document.metadata_json = {
        "paper_profile": {
            "profile_version": "paper-profile-v1",
            "source_text_sha256": hashlib.sha256(b"old source").hexdigest(),
            "routing_summary": "stale",
            "aliases": [],
            "key_terms": ["StaleOnlyTerm"],
        }
    }

    terms = paper_profile_retrieval_terms(document)

    assert "FreshUniqueRoutingTerm" in terms
    assert "StaleOnlyTerm" not in terms


def test_activation_rejects_plain_child_whose_embedding_text_was_altered(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    records = [
        _record("parent-1", embedding=None),
        _record("child-1", embedding=[1.0, 0.0], block_type="narrative"),
    ]
    runner = _prepare_index_runner(db, tmp_path, monkeypatch, records)
    runner.run_stage("d1", "new-v", "index", enqueue_next=False)
    db.get(DocumentChunk, "child-1").embedding_text = "Altered context.\n\nSource text"
    db.commit()

    with pytest.raises(RuntimeError, match="embedding completeness"):
        runner.run_stage("d1", "new-v", "activate", enqueue_next=False)

    db.expire_all()
    assert db.get(Document, "d1").active_parse_version == "old-v"
