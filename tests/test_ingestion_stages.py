from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import (
    Document,
    DocumentChunk,
    DocumentParseVersion,
    PipelineRun,
    Project,
    RunStatus,
    RunType,
)
from app.services.pipeline import (
    IngestionPipeline,
    compare_typed_inventory,
    derive_child_inventory_from_payload,
    derive_db_typed_inventory,
)
from app.services.ingestion_stages import (
    INGESTION_STAGES,
    STAGE_QUEUES,
    IngestionStageRunner,
    StageAlreadyClaimed,
    StageCheckpointTooLarge,
)
from app.services import queue as queue_module
from app.services.contextualization import ContextualizedChunk
from app.services.canonical_models import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalQualityIssue,
    CanonicalQualityReport,
    SourceSpan,
)
from app.services.parse_versions import ActivationError
from app.services.semantic_chunking import ChunkDraft


def _create_directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"directory symlinks are unavailable: {exc}")

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"directory links are unavailable: {result.stderr}")


@pytest.fixture
def db() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    session = factory()
    session.add(Project(id="p1", slug="research", name="Research"))
    session.add(
        Document(
            id="d1",
            project_id="p1",
            title="Paper",
            file_name="paper.pdf",
            sha256="abc",
            raw_path="raw/paper.pdf",
        )
    )
    session.add(
        DocumentParseVersion(
            id="pv1",
            document_id="d1",
            version_key="v1",
            artifact_dir="parsed/d1/v1",
        )
    )
    session.commit()
    return session


class RecordingDispatcher:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.fail = fail

    def enqueue_stage(self, document_id: str, version_key: str, stage: str):
        self.calls.append((document_id, version_key, stage))
        if self.fail:
            raise RuntimeError("queue unavailable")


def handlers(calls: list[str], failures: Counter[str] | None = None):
    remaining = failures or Counter()

    def build(stage: str):
        def handle(context):
            calls.append(stage)
            if remaining[stage]:
                remaining[stage] -= 1
                raise RuntimeError(f"{stage} failed")
            return {
                "stage": stage,
                "at": datetime(2026, 7, 24, 12, 0),
                "artifact": Path(f"artifacts/{stage}"),
            }

        return handle

    return {stage: build(stage) for stage in INGESTION_STAGES}


def _ingestion_config_fixture(*, content_sha256: str = "a" * 64) -> dict[str, object]:
    return {
        "algorithm_revisions": {
            "parser": "canonical-parser-v1",
            "pdf_recovery": "pdf-recovery-v1",
            "structured_splitting": "structured-splitting-v1",
            "source_fidelity_algorithm": "source-fidelity-v1",
            "source_fidelity_schema": "source-fidelity-schema-v1",
        },
        "tokenizer": {
            "name": "Qwen/Qwen3-Embedding-4B",
            "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
            "content_sha256": content_sha256,
        },
        "embedding": {"model": "qwen3-embedding:4b", "dimensions": 2560},
        "semantic_splitting": {
            "model": "qwen3-embedding:4b",
            "break_percentile": 20,
            "parent_tokens": {"min": 500, "target": 1200, "max": 1800},
            "child_tokens": {"min": 180, "target": 400, "max": 600},
            "overlap_tokens": 50,
        },
    }


def _ingestion_config_hash(snapshot: dict[str, object] | None = None) -> str:
    return hashlib.sha256(
        json.dumps(
            snapshot or _ingestion_config_fixture(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@pytest.fixture(autouse=True)
def stub_pipeline_ingestion_identity(monkeypatch) -> None:
    from app.services import pipeline as pipeline_module

    snapshot = _ingestion_config_fixture()
    config_hash = _ingestion_config_hash(snapshot)
    monkeypatch.setattr(
        pipeline_module, "build_ingestion_config_snapshot", lambda: snapshot
    )
    monkeypatch.setattr(
        pipeline_module,
        "canonical_ingestion_config_hash",
        lambda _snapshot: config_hash,
    )


def _canonical_test_version_key(pipeline_version: str = "canonical-v4") -> str:
    return f"{pipeline_version}-abc-{_ingestion_config_hash()[:12]}"


def _set_test_ingestion_manifest(version: DocumentParseVersion) -> None:
    snapshot = _ingestion_config_fixture()
    version.manifest_json = {
        "ingestion_config": snapshot,
        "ingestion_config_sha256": _ingestion_config_hash(snapshot),
    }


def test_stage_contract_and_named_queues_are_exact() -> None:
    assert INGESTION_STAGES == (
        "parse",
        "repair",
        "canonicalize",
        "semantic_split",
        "contextualize",
        "embed",
        "index",
        "activate",
    )
    assert STAGE_QUEUES == {
        stage: f"ingest.{stage}" for stage in INGESTION_STAGES
    }


def test_production_registers_every_stage_handler(db: Session) -> None:
    assert set(IngestionPipeline(db).ingestion_stage_handlers()) == set(
        INGESTION_STAGES
    )


def _prepare_two_document_activation_batch(
    db: Session,
) -> dict[str, str]:
    first = db.get(Document, "d1")
    old_first = db.get(DocumentParseVersion, "pv1")
    assert first is not None and old_first is not None
    first.active_parse_version = "old-1"
    first.raw_text = "old text 1"
    first.metadata_json = {"profile": "old-1"}
    first.status = "ready"
    old_first.version_key = "old-1"
    old_first.status = "active"

    second = Document(
        id="d2",
        project_id="p1",
        title="Second Paper",
        file_name="second.pdf",
        sha256="def",
        raw_path="raw/second.pdf",
        raw_text="old text 2",
        metadata_json={"profile": "old-2"},
        status="ready",
        active_parse_version="old-2",
    )
    prior_stages = {
        stage: {"status": "completed", "output": {"stage": stage}}
        for stage in INGESTION_STAGES[:-1]
    }
    staged = [
        DocumentParseVersion(
            id="new-pv1",
            document_id="d1",
            version_key="new-1",
            artifact_dir="parsed/d1/new-1",
            status="ready_to_activate",
            stage_state=prior_stages,
        ),
        DocumentParseVersion(
            id="old-pv2",
            document_id="d2",
            version_key="old-2",
            artifact_dir="parsed/d2/old-2",
            status="active",
        ),
        DocumentParseVersion(
            id="new-pv2",
            document_id="d2",
            version_key="new-2",
            artifact_dir="parsed/d2/new-2",
            status="ready_to_activate",
            stage_state=prior_stages,
        ),
    ]
    runs = [
        PipelineRun(
            id=f"run-{document_id}",
            project_id="p1",
            document_id=document_id,
            run_type=RunType.rebuild.value,
            status=RunStatus.running.value,
            provider_report={"progress": {"stage": "index"}},
        )
        for document_id in ("d1", "d2")
    ]
    db.add_all([second, *staged, *runs])
    db.commit()
    return {"d1": "new-1", "d2": "new-2"}


def test_batch_activation_commits_all_side_effects_together(db: Session) -> None:
    version_map = _prepare_two_document_activation_batch(db)

    def activate(context):
        suffix = context.document.id[-1]
        context.document.raw_text = f"new text {suffix}"
        context.document.metadata_json = {"profile": f"new-{suffix}"}
        return {"parse_version": context.version.version_key, "validated": True}

    runner = IngestionStageRunner(db, handlers={"activate": activate})

    activated = runner.activate_batch(version_map)

    assert [version.version_key for version in activated] == ["new-1", "new-2"]
    db.expire_all()
    for document_id, old_key, new_key in (
        ("d1", "old-1", "new-1"),
        ("d2", "old-2", "new-2"),
    ):
        document = db.get(Document, document_id)
        assert document is not None
        assert document.active_parse_version == new_key
        assert document.raw_text == f"new text {document_id[-1]}"
        assert document.metadata_json == {"profile": f"new-{document_id[-1]}"}
        assert document.status == "ready"
        old = db.scalar(
            select(DocumentParseVersion).where(
                DocumentParseVersion.document_id == document_id,
                DocumentParseVersion.version_key == old_key,
            )
        )
        new = db.scalar(
            select(DocumentParseVersion).where(
                DocumentParseVersion.document_id == document_id,
                DocumentParseVersion.version_key == new_key,
            )
        )
        assert old is not None and old.status == "superseded"
        assert new is not None and new.status == "active"
        assert new.stage_state["activate"]["status"] == "completed"
        assert new.stage_state["activate"]["output"]["validated"] is True
        run = db.get(PipelineRun, f"run-{document_id}")
        assert run is not None and run.status == RunStatus.completed.value
        assert run.provider_report["progress"]["stage"] == "completed"


def test_batch_activation_validation_failure_rolls_back_every_side_effect(
    db: Session,
) -> None:
    version_map = _prepare_two_document_activation_batch(db)

    def activate(context):
        context.document.raw_text = f"new text {context.document.id[-1]}"
        context.document.metadata_json = {"profile": "new"}
        if context.document.id == "d2":
            raise RuntimeError("second document validation failed")
        return {"validated": True}

    runner = IngestionStageRunner(db, handlers={"activate": activate})

    with pytest.raises(RuntimeError, match="second document validation failed"):
        runner.activate_batch(version_map)

    db.expire_all()
    for document_id, old_key, new_key in (
        ("d1", "old-1", "new-1"),
        ("d2", "old-2", "new-2"),
    ):
        document = db.get(Document, document_id)
        assert document is not None
        assert document.active_parse_version == old_key
        assert document.raw_text == f"old text {document_id[-1]}"
        assert document.metadata_json == {"profile": old_key}
        assert document.status == "ready"
        staged = db.scalar(
            select(DocumentParseVersion).where(
                DocumentParseVersion.document_id == document_id,
                DocumentParseVersion.version_key == new_key,
            )
        )
        assert staged is not None and staged.status == "ready_to_activate"
        assert "activate" not in staged.stage_state
        run = db.get(PipelineRun, f"run-{document_id}")
        assert run is not None and run.status == RunStatus.running.value
        assert run.provider_report["progress"]["stage"] == "index"


def test_batch_activation_flush_failure_rolls_back_every_side_effect(
    db: Session,
    monkeypatch,
) -> None:
    version_map = _prepare_two_document_activation_batch(db)

    def activate(context):
        context.document.raw_text = f"new text {context.document.id[-1]}"
        context.document.metadata_json = {"profile": "new"}
        return {"validated": True}

    runner = IngestionStageRunner(db, handlers={"activate": activate})
    original_flush = db.flush
    monkeypatch.setattr(
        db,
        "flush",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated batch flush failure")),
    )

    with pytest.raises(RuntimeError, match="simulated batch flush failure"):
        runner.activate_batch(version_map)

    monkeypatch.setattr(db, "flush", original_flush)
    db.expire_all()
    for document_id, old_key, new_key in (
        ("d1", "old-1", "new-1"),
        ("d2", "old-2", "new-2"),
    ):
        document = db.get(Document, document_id)
        assert document is not None
        assert document.active_parse_version == old_key
        assert document.raw_text == f"old text {document_id[-1]}"
        assert document.metadata_json == {"profile": old_key}
        staged = db.scalar(
            select(DocumentParseVersion).where(
                DocumentParseVersion.document_id == document_id,
                DocumentParseVersion.version_key == new_key,
            )
        )
        assert staged is not None and staged.status == "ready_to_activate"
        assert "activate" not in staged.stage_state


def test_process_document_enqueues_first_failed_or_incomplete_stage(
    db: Session, monkeypatch
) -> None:
    version = db.get(DocumentParseVersion, "pv1")
    version.version_key = _canonical_test_version_key()
    _set_test_ingestion_manifest(version)
    version.status = "parse_failed"
    version.stage_state = {
        "parse": {"status": "completed", "next_enqueued": True},
        "repair": {"status": "completed", "next_enqueued": True},
        "canonicalize": {"status": "failed", "attempts": 1},
    }
    db.add(
        PipelineRun(
            id="resume-run",
            project_id="p1",
            document_id="d1",
            run_type=RunType.ingest.value,
            status=RunStatus.failed.value,
            provider_report={},
        )
    )
    db.commit()
    calls: list[tuple[str, str, str]] = []

    class FakeDispatcher:
        def enqueue_stage(self, document_id, version_key, stage):
            calls.append((document_id, version_key, stage))

    from app.services import pipeline as pipeline_module
    from app.services import queue as queue_module

    monkeypatch.setattr(pipeline_module.settings, "redis_url", "redis://local")
    monkeypatch.setattr(queue_module, "JobDispatcher", FakeDispatcher)

    IngestionPipeline(db).process_document("d1")

    assert calls == [("d1", _canonical_test_version_key(), "canonicalize")]


def test_process_document_keeps_completed_version_ready_without_enqueue(
    db: Session, monkeypatch
) -> None:
    version = db.get(DocumentParseVersion, "pv1")
    version.version_key = _canonical_test_version_key()
    _set_test_ingestion_manifest(version)
    version.status = "active"
    version.stage_state = {
        stage: {"status": "completed"} for stage in INGESTION_STAGES
    }
    document = db.get(Document, "d1")
    document.active_parse_version = version.version_key
    document.status = "ready"
    db.add(
        PipelineRun(
            id="complete-run",
            project_id="p1",
            document_id="d1",
            run_type=RunType.ingest.value,
            status=RunStatus.failed.value,
            provider_report={},
        )
    )
    db.commit()
    calls: list[object] = []

    class FakeDispatcher:
        def enqueue_stage(self, *args):
            calls.append(args)

    from app.services import pipeline as pipeline_module
    from app.services import queue as queue_module

    monkeypatch.setattr(pipeline_module.settings, "redis_url", "redis://local")
    monkeypatch.setattr(queue_module, "JobDispatcher", FakeDispatcher)

    run = IngestionPipeline(db).process_document("d1")

    assert calls == []
    assert run.status == RunStatus.completed.value
    assert db.get(Document, "d1").status == "ready"


def test_process_document_preserves_active_version_until_activation_is_available(
    db: Session, monkeypatch
) -> None:
    document = db.get(Document, "d1")
    document.status = "ready"
    document.active_parse_version = "legacy"
    version = db.get(DocumentParseVersion, "pv1")
    version.version_key = _canonical_test_version_key("canonical-v1")
    _set_test_ingestion_manifest(version)
    db.add(
        DocumentParseVersion(
            id="legacy-version",
            document_id="d1",
            version_key="legacy",
            artifact_dir="legacy",
            status="active",
        )
    )
    db.commit()
    enqueued: list[tuple[str, str, str]] = []

    class FakeDispatcher:
        def enqueue_stage(self, document_id, version_key, stage):
            enqueued.append((document_id, version_key, stage))

    from app.services import pipeline as pipeline_module
    from app.services import queue as queue_module

    monkeypatch.setattr(pipeline_module.settings, "redis_url", "redis://local")
    monkeypatch.setattr(
        pipeline_module.settings, "canonical_pipeline_version", "canonical-v1"
    )
    monkeypatch.setattr(queue_module, "JobDispatcher", FakeDispatcher)
    pipeline = IngestionPipeline(db)

    pipeline.process_document("d1")

    assert enqueued == [("d1", _canonical_test_version_key("canonical-v1"), "parse")]
    assert db.get(Document, "d1").status == "ready"
    assert db.get(Document, "d1").active_parse_version == "legacy"

    stage_handlers = handlers([])
    stage_handlers["activate"] = pipeline.ingestion_stage_handlers()["activate"]
    runner = IngestionStageRunner(db, handlers=stage_handlers)
    runner.run_until_blocked("d1", _canonical_test_version_key("canonical-v1"))
    with pytest.raises(RuntimeError, match="artifact reference"):
        runner.run_stage(
            "d1", _canonical_test_version_key("canonical-v1"), "activate", enqueue_next=False
        )

    db.expire_all()
    assert db.get(Document, "d1").status == "ready"
    assert db.get(Document, "d1").active_parse_version == "legacy"
    assert db.get(DocumentParseVersion, "pv1").status == "activation_failed"


def test_process_document_commits_retryable_state_before_dispatch(
    db: Session, monkeypatch
) -> None:
    document = db.get(Document, "d1")
    document.status = "pending"
    version = db.get(DocumentParseVersion, "pv1")
    version.version_key = _canonical_test_version_key("canonical-v1")
    _set_test_ingestion_manifest(version)
    db.add(
        PipelineRun(
            id="retry-run",
            project_id="p1",
            document_id="d1",
            run_type=RunType.ingest.value,
            status=RunStatus.failed.value,
            provider_report={"old": True},
        )
    )
    db.commit()
    calls: list[tuple[str, str, str]] = []
    transaction_states: list[tuple[bool, bool, bool]] = []

    class FlakyDispatcher:
        def enqueue_stage(self, document_id, version_key, stage):
            transaction_states.append(
                (db.in_transaction(), bool(db.dirty), bool(db.new))
            )
            calls.append((document_id, version_key, stage))
            if len(calls) == 1:
                raise RuntimeError("queue unavailable")

    from app.services import pipeline as pipeline_module
    from app.services import queue as queue_module

    monkeypatch.setattr(pipeline_module.settings, "redis_url", "redis://local")
    monkeypatch.setattr(
        pipeline_module.settings, "canonical_pipeline_version", "canonical-v1"
    )
    monkeypatch.setattr(queue_module, "JobDispatcher", FlakyDispatcher)
    pipeline = IngestionPipeline(db)

    with pytest.raises(RuntimeError, match="queue unavailable"):
        pipeline.process_document("d1")
    assert transaction_states == [(False, False, False)]
    db.rollback()
    db.expire_all()

    stored_run = db.get(PipelineRun, "retry-run")
    assert stored_run.status == RunStatus.queued.value
    assert stored_run.provider_report["progress"]["stage"] == "queued"
    assert db.get(Document, "d1").status == "processing"

    pipeline.process_document("d1")
    db.rollback()
    db.expire_all()
    assert calls == [
        ("d1", _canonical_test_version_key("canonical-v1"), "parse"),
        ("d1", _canonical_test_version_key("canonical-v1"), "parse"),
    ]
    assert transaction_states == [
        (False, False, False),
        (False, False, False),
    ]
    assert db.get(PipelineRun, "retry-run").status == RunStatus.queued.value


def test_malicious_pipeline_version_is_rejected_before_path_construction(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    from app.services import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module.settings, "redis_url", "redis://local")
    monkeypatch.setattr(pipeline_module.settings, "canonical_pipeline_version", "../escape")
    monkeypatch.setattr(
        pipeline_module.settings, "canonical_artifacts_dir", tmp_path / "artifacts"
    )

    with pytest.raises(ValueError, match="CANONICAL_PIPELINE_VERSION"):
        IngestionPipeline(db).process_document("d1")

    assert not (tmp_path / "escape").exists()


def test_parse_version_key_includes_canonical_ingestion_config_hash(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    from app.services import pipeline as pipeline_module

    snapshot = _ingestion_config_fixture()
    monkeypatch.setattr(
        pipeline_module,
        "build_ingestion_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "canonical_ingestion_config_hash",
        lambda _snapshot: "1234567890abcdef" + "0" * 48,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module.settings, "canonical_artifacts_dir", tmp_path / "artifacts"
    )
    document = db.get(Document, "d1")

    version = IngestionPipeline(db)._get_or_create_parse_version(document)

    assert version.version_key == "canonical-v4-abc-1234567890ab"
    assert version.manifest_json["ingestion_config"] == snapshot
    assert (
        version.manifest_json["ingestion_config_sha256"]
        == "1234567890abcdef" + "0" * 48
    )


def test_existing_checkpoint_rejects_changed_live_ingestion_config(
    db: Session, monkeypatch
) -> None:
    from app.services import pipeline as pipeline_module

    snapshot = _ingestion_config_fixture()
    config_hash = "1234567890abcdef" + "0" * 48
    version = db.get(DocumentParseVersion, "pv1")
    version.version_key = f"canonical-v4-abc-{config_hash[:12]}"
    version.stage_state = {"parse": {"status": "completed"}}
    version.manifest_json = {
        "ingestion_config": _ingestion_config_fixture(content_sha256="b" * 64),
        "ingestion_config_sha256": "b" * 64,
    }
    db.commit()
    monkeypatch.setattr(
        pipeline_module,
        "build_ingestion_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "canonical_ingestion_config_hash",
        lambda _snapshot: config_hash,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="ingestion configuration.*checkpoint"):
        IngestionPipeline(db)._get_or_create_parse_version(db.get(Document, "d1"))


def test_semantic_split_rejects_config_changed_after_queueing(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    from app.services import canonical_artifacts as artifacts_module
    from app.services import pipeline as pipeline_module
    from app.services import semantic_chunking as semantic_chunking_module

    live_snapshot = _ingestion_config_fixture()
    live_hash = "1234567890abcdef" + "0" * 48
    version = db.get(DocumentParseVersion, "pv1")
    version.manifest_json = {
        "ingestion_config": _ingestion_config_fixture(content_sha256="b" * 64),
        "ingestion_config_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        pipeline_module,
        "build_ingestion_config_snapshot",
        lambda: live_snapshot,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "canonical_ingestion_config_hash",
        lambda _snapshot: live_hash,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module.settings, "canonical_artifacts_dir", tmp_path / "artifacts"
    )
    monkeypatch.setattr(
        artifacts_module.CanonicalArtifactStore,
        "load",
        lambda *_args: SimpleNamespace(),
    )

    class FakeSemanticChunker:
        def __init__(self, _embedder) -> None:
            pass

        def build(self, _canonical):
            return [SimpleNamespace(model_dump=lambda **_kwargs: {"local_id": "one"})]

    monkeypatch.setattr(
        semantic_chunking_module, "SemanticChunker", FakeSemanticChunker
    )
    context = SimpleNamespace(
        document=db.get(Document, "d1"),
        version=version,
        stage="semantic_split",
        input={},
    )

    with pytest.raises(RuntimeError, match="ingestion configuration.*queued"):
        IngestionPipeline(db)._run_semantic_split_stage(context)


def test_pdf_targeted_repair_is_not_called_until_repair_phase(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-test")
    issue = CanonicalQualityIssue(
        code="table_invalid",
        severity="error",
        message="repair table",
        repairable=True,
        repair_scope="page:1",
    )
    primary = CanonicalDocument(
        document_id="d1",
        parse_version="v1",
        source_path=str(source),
        source_media_type="application/pdf",
        parser_source="mineru",
        title="Paper",
        blocks=[
            CanonicalBlock(
                block_id="b1",
                block_type="narrative",
                text="Evidence",
                reading_order=0,
                parser_source="mineru",
                source_spans=[SourceSpan(page_index=0, page_label="1")],
            )
        ],
        metadata={
            "expected_page_count": 1,
            "parsed_page_indices": [0],
            "text_layer_pages": ["Evidence"],
        },
        quality=CanonicalQualityReport(
            accepted=False,
            status="rejected",
            issues=[issue],
        ),
    )
    calls: list[object] = []
    from app.services import canonical_adapters as adapters
    from app.services import parser as parser_module
    from app.services import canonical_quality as quality_module

    monkeypatch.setattr(adapters, "_validate_path", lambda path: path)
    monkeypatch.setattr(parser_module, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(adapters, "run_mineru", lambda _path, _count: primary)
    monkeypatch.setattr(
        adapters, "_read_text_layer_for_audit", lambda _path, _count: (["Evidence"], [])
    )
    monkeypatch.setattr(adapters, "_attach_pdf_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(adapters, "_finalize_structured_evidence", lambda document: None)
    monkeypatch.setattr(adapters, "_finalize_pdf_audit", lambda document: document)
    monkeypatch.setattr(
        quality_module.CanonicalQualityGate,
        "evaluate",
        lambda _self, document: document.quality,
    )
    monkeypatch.setattr(
        adapters,
        "run_document_intelligence",
        lambda *args, **kwargs: calls.append(kwargs.get("page_indices")) or primary,
    )
    monkeypatch.setattr(
        adapters, "_targeted_repair_has_complete_coverage", lambda *_args: False
    )

    pipeline = IngestionPipeline(db)
    parsed = pipeline._parse_canonical_phase(source)
    assert calls == []

    pipeline._repair_canonical_phase(parsed, source)
    assert calls == [{0}]


def test_pdf_repair_phase_supplements_mineru_page_gaps_from_text_layer(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-test")
    primary = CanonicalDocument(
        document_id="d1",
        parse_version="v1",
        source_path=str(source),
        source_media_type="application/pdf",
        parser_source="mineru",
        title="Paper",
        blocks=[
            CanonicalBlock(
                block_id="b1",
                block_type="narrative",
                text="MinerU page one",
                reading_order=0,
                parser_source="mineru",
                source_spans=[SourceSpan(page_index=0, page_label="1")],
            )
        ],
        metadata={
            "expected_page_count": 2,
            "parsed_page_indices": [0],
            "text_layer_pages": ["MinerU page one", "Recovered page two"],
            "text_layer_warnings": [],
        },
        quality=CanonicalQualityReport(
            accepted=False,
            status="rejected",
            issues=[
                CanonicalQualityIssue(
                    code="page_missing",
                    severity="fatal",
                    message="page two is missing",
                    metadata={"missing_pages": [2], "expected_page_count": 2},
                )
            ],
        ),
    )
    from app.services import parser as parser_module

    monkeypatch.setattr(parser_module.settings, "document_intelligence_enabled", False)

    result = IngestionPipeline(db)._repair_canonical_phase(primary, source)

    assert result.parser_source == "mineru"
    assert result.metadata["parsed_page_indices"] == [0, 1]
    assert result.metadata["fallback_pages"] == [2]
    assert result.metadata["text_layer_fallback_page_indices"] == [1]
    assert result.quality.accepted is True
    recovered = [
        block
        for block in result.blocks
        if any(span.page_index == 1 for span in block.source_spans)
    ]
    assert [block.text for block in recovered] == ["Recovered page two"]
    assert recovered[0].parser_source == "pypdf_text_layer"
    assert recovered[0].metadata["fallback_reason"] == "mineru_page_missing"


def test_staged_pdf_parse_supplements_completely_missing_mineru_page(
    db: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-test")
    primary = CanonicalDocument(
        source_path=str(source),
        source_media_type="application/pdf",
        parser_source="mineru",
        title="Paper",
        blocks=[
            CanonicalBlock(
                block_id="mineru-page-1",
                block_type="narrative",
                text="MinerU page one",
                reading_order=0,
                parser_source="mineru",
                source_spans=[SourceSpan(page_index=0, page_label="1")],
            )
        ],
        metadata={"parsed_page_indices": [0]},
    )
    from app.services import canonical_adapters as adapters
    from app.services import parser as parser_module

    monkeypatch.setattr(adapters, "_validate_path", lambda path: path)
    monkeypatch.setattr(parser_module, "_validate_pdf_basic", lambda _path: 2)
    monkeypatch.setattr(parser_module.settings, "mineru_enabled", True)
    monkeypatch.setattr(adapters, "run_mineru", lambda *_args: primary)
    monkeypatch.setattr(
        adapters,
        "_read_text_layer_for_audit",
        lambda *_args: (["MinerU page one", "Complete pypdf page two"], []),
    )

    result = IngestionPipeline(db)._parse_canonical_phase(source)

    fallback = [
        block
        for block in result.blocks
        if block.metadata.get("fallback_reason") == "mineru_page_missing"
    ]
    assert [block.text for block in fallback] == ["Complete pypdf page two"]
    assert fallback[0].parser_source == "pypdf_text_layer"
    assert fallback[0].source_spans[0].page_index == 1
    assert result.metadata["parsed_page_indices"] == [0, 1]
    assert result.metadata["text_layer_fallback_page_indices"] == [1]
    assert result.metadata["fallback_pages"] == [2]
    assert "pypdf_text_layer:missing_pages:2" in result.parser_metadata[
        "parser_attempts"
    ]


def test_staged_pdf_parse_recovers_partial_mineru_page_omission(
    db: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-test")
    prefix = "Shared experimental background remains present in MinerU. " * 20
    omitted = (
        "The omitted passage reports a distinct calibration protocol with "
        "temperature controls, replicate counts, and uncertainty estimates."
    )
    suffix = "Shared conclusions and limitations remain present in MinerU. " * 20
    primary = CanonicalDocument(
        source_path=str(source),
        source_media_type="application/pdf",
        parser_source="mineru",
        title="Paper",
        blocks=[
            CanonicalBlock(
                block_id="mineru-page-1",
                block_type="narrative",
                text=prefix + suffix,
                reading_order=0,
                parser_source="mineru",
                source_spans=[SourceSpan(page_index=0, page_label="1")],
            )
        ],
        metadata={"parsed_page_indices": [0]},
    )
    from app.services import canonical_adapters as adapters
    from app.services import parser as parser_module

    monkeypatch.setattr(adapters, "_validate_path", lambda path: path)
    monkeypatch.setattr(parser_module, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(parser_module.settings, "mineru_enabled", True)
    monkeypatch.setattr(adapters, "run_mineru", lambda *_args: primary)
    monkeypatch.setattr(
        adapters,
        "_read_text_layer_for_audit",
        lambda *_args: ([prefix + omitted + suffix], []),
    )

    result = IngestionPipeline(db)._parse_canonical_phase(source)

    recovered = [
        block
        for block in result.blocks
        if block.metadata.get("source") == "pdf_text_recovery"
    ]
    assert [block.text for block in recovered] == [omitted]
    assert recovered[0].parser_source == "pypdf_text_layer"
    assert recovered[0].metadata["fallback_reason"] == "mineru_text_omission"
    assert result.metadata["text_layer_recovery_page_indices"] == [0]
    assert "pypdf_text_layer:page_recovery:1" in result.parser_metadata[
        "parser_attempts"
    ]


def test_staged_pdf_parse_does_not_recover_reordered_mineru_content(
    db: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-test")
    first = (
        "Alpha reports the complete experimental setup, calibration sequence, "
        "temperature controls, replicate counts, and uncertainty estimates."
    )
    second = (
        "Beta presents the complete observations, comparison groups, sensitivity "
        "analysis, confidence intervals, and validation outcomes."
    )
    third = (
        "Gamma summarizes the complete interpretation, limitations, practical "
        "implications, and directions for follow-up research."
    )
    primary = CanonicalDocument(
        source_path=str(source),
        source_media_type="application/pdf",
        parser_source="mineru",
        title="Paper",
        blocks=[
            CanonicalBlock(
                block_id="mineru-page-1",
                block_type="narrative",
                text=f"{second}\n\n{first}\n\n{third}",
                reading_order=0,
                parser_source="mineru",
                source_spans=[SourceSpan(page_index=0, page_label="1")],
            )
        ],
        metadata={"parsed_page_indices": [0]},
    )
    from app.services import canonical_adapters as adapters
    from app.services import parser as parser_module

    monkeypatch.setattr(adapters, "_validate_path", lambda path: path)
    monkeypatch.setattr(parser_module, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(parser_module.settings, "mineru_enabled", True)
    monkeypatch.setattr(adapters, "run_mineru", lambda *_args: primary)
    monkeypatch.setattr(
        adapters,
        "_read_text_layer_for_audit",
        lambda *_args: ([f"{first}\n\n{second}\n\n{third}"], []),
    )

    result = IngestionPipeline(db)._parse_canonical_phase(source)

    recovered = [
        block
        for block in result.blocks
        if block.metadata.get("source") == "pdf_text_recovery"
    ]
    assert recovered == []
    assert result.metadata["text_layer_recovery_page_indices"] == []
    assert "pypdf_text_layer:page_recovery:1" not in result.parser_metadata[
        "parser_attempts"
    ]


@pytest.mark.parametrize(
    "persisted_fallback",
    [False, True],
    ids=["newly-supplemented", "persisted-fallback"],
)
def test_repair_excludes_supplemented_pages_from_targeted_visual_scope(
    db: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persisted_fallback: bool,
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-test")
    blocks = [
        CanonicalBlock(
            block_id="mineru-page-1",
            block_type="narrative",
            text="MinerU page one",
            reading_order=0,
            parser_source="mineru",
            source_spans=[SourceSpan(page_index=0, page_label="1")],
        )
    ]
    metadata = {
        "expected_page_count": 2,
        "parsed_page_indices": [0],
        "text_layer_pages": ["MinerU page one", "Recovered page two"],
        "text_layer_warnings": [],
    }
    if persisted_fallback:
        blocks.append(
            CanonicalBlock(
                block_id="pypdf-page-2",
                block_type="narrative",
                text="Recovered page two",
                reading_order=1,
                parser_source="pypdf_text_layer",
                source_spans=[SourceSpan(page_index=1, page_label="2")],
                metadata={"source": "pypdf_text_layer"},
            )
        )
        metadata["parsed_page_indices"] = [0, 1]
        metadata["text_layer_fallback_page_indices"] = [1]
    primary = CanonicalDocument(
        source_path=str(source),
        source_media_type="application/pdf",
        parser_source="mineru",
        title="Paper",
        blocks=blocks,
        metadata=metadata,
        parser_metadata={"parser_attempts": ["mineru:success"]},
    )
    remaining_issue = CanonicalQualityIssue(
        code="abstract_missing",
        severity="warning",
        message="page one still needs repair",
        repairable=True,
        repair_scope="pages:1-2",
    )
    targeted_calls: list[set[int] | None] = []
    from app.services import canonical_adapters as adapters
    from app.services import canonical_quality as quality_module
    from app.services import parser as parser_module

    def evaluate(_self, document: CanonicalDocument) -> CanonicalQualityReport:
        document.quality = CanonicalQualityReport(
            accepted=False,
            status="rejected",
            issues=[remaining_issue],
        )
        return document.quality

    monkeypatch.setattr(parser_module.settings, "document_intelligence_enabled", True)
    monkeypatch.setattr(quality_module.CanonicalQualityGate, "evaluate", evaluate)
    monkeypatch.setattr(
        adapters,
        "run_document_intelligence",
        lambda *_args, **kwargs: targeted_calls.append(kwargs.get("page_indices")),
    )

    result = IngestionPipeline(db)._repair_canonical_phase(primary, source)

    assert targeted_calls == [{0}]
    attempts = result.parser_metadata["parser_attempts"]
    assert attempts[0] == "mineru:success"
    assert attempts[-1] == "document_intelligence:targeted:unavailable"
    if persisted_fallback:
        assert "pypdf_text_layer:missing_pages:2" not in attempts
    else:
        assert "pypdf_text_layer:missing_pages:2" in attempts


def test_run_until_blocked_completes_in_order_and_stops_ready_to_activate(
    db: Session,
) -> None:
    calls: list[str] = []
    runner = IngestionStageRunner(db, handlers=handlers(calls))

    version = runner.run_until_blocked("d1", "v1")

    assert calls == list(INGESTION_STAGES[:-1])
    assert version.status == "ready_to_activate"
    assert list(version.stage_state) == list(INGESTION_STAGES[:-1])
    assert all(value["status"] == "completed" for value in version.stage_state.values())
    assert all(value["attempts"] == 1 for value in version.stage_state.values())
    assert all(value["progress"] == 100 for value in version.stage_state.values())
    json.dumps(version.stage_state, allow_nan=False)

    runner.run_stage("d1", "v1", "activate", enqueue_next=False)
    assert version.status == "active"
    assert db.get(Document, "d1").active_parse_version == "v1"
    assert db.get(Document, "d1").status == "ready"


def test_completed_stages_are_idempotent_and_do_not_repeat_handlers(db: Session) -> None:
    calls: list[str] = []
    runner = IngestionStageRunner(db, handlers=handlers(calls))
    runner.run_until_blocked("d1", "v1")

    runner.run_until_blocked("d1", "v1")
    runner.run_stage("d1", "v1", "parse", enqueue_next=False)

    assert Counter(calls) == Counter(INGESTION_STAGES[:-1])


@pytest.mark.parametrize("stage", INGESTION_STAGES)
def test_pre_stage_validator_runs_before_completed_checkpoint_return(
    db: Session, stage: str
) -> None:
    version = db.get(DocumentParseVersion, "pv1")
    version.stage_state = {stage: {"status": "completed", "attempts": 1}}
    db.commit()
    validated: list[str] = []

    def validate(_document, _version, current_stage: str) -> None:
        validated.append(current_stage)
        raise RuntimeError(f"configuration drift before {current_stage}")

    runner = IngestionStageRunner(
        db,
        handlers={},
        pre_stage_validator=validate,
    )

    with pytest.raises(RuntimeError, match=f"configuration drift before {stage}"):
        runner.run_stage("d1", "v1", stage, enqueue_next=False)

    assert validated == [stage]
    assert db.in_transaction() is False


def test_failed_contextualization_resumes_only_that_stage(db: Session) -> None:
    calls: list[str] = []
    runner = IngestionStageRunner(
        db,
        handlers=handlers(calls, Counter({"contextualize": 1})),
    )

    with pytest.raises(RuntimeError, match="contextualize failed"):
        runner.run_until_blocked("d1", "v1")

    version = db.get(DocumentParseVersion, "pv1")
    assert version.status == "contextualization_failed"
    assert db.get(Document, "d1").status == "contextualization_failed"
    assert version.stage_state["contextualize"]["status"] == "failed"
    assert version.stage_state["contextualize"]["attempts"] == 1

    runner.run_until_blocked("d1", "v1")
    assert calls.count("parse") == 1
    assert calls.count("contextualize") == 2
    assert version.status == "ready_to_activate"


def test_parse_failed_mapping_resumes_semantic_split_without_replaying_parse(
    db: Session,
) -> None:
    calls: list[str] = []
    runner = IngestionStageRunner(
        db,
        handlers=handlers(calls, Counter({"semantic_split": 1})),
    )

    with pytest.raises(RuntimeError, match="semantic_split failed"):
        runner.run_until_blocked("d1", "v1")
    assert db.get(DocumentParseVersion, "pv1").status == "parse_failed"

    runner.run_until_blocked("d1", "v1")

    assert calls.count("parse") == 1
    assert calls.count("repair") == 1
    assert calls.count("canonicalize") == 1
    assert calls.count("semantic_split") == 2
    assert db.get(DocumentParseVersion, "pv1").status == "ready_to_activate"


def test_repair_failure_is_terminal_and_does_not_dispatch_next(db: Session) -> None:
    calls: list[str] = []
    dispatcher = RecordingDispatcher()
    runner = IngestionStageRunner(
        db,
        handlers=handlers(calls, Counter({"repair": 1})),
        dispatcher=dispatcher,
    )
    runner.run_stage("d1", "v1", "parse", enqueue_next=False)

    with pytest.raises(RuntimeError, match="repair failed"):
        runner.run_stage("d1", "v1", "repair")

    version = db.get(DocumentParseVersion, "pv1")
    assert version.status == "table_repair_failed"
    assert dispatcher.calls == []


def test_checkpoint_is_committed_before_enqueue_and_enqueue_failure_is_repaired(
    db: Session,
) -> None:
    calls: list[str] = []
    dispatcher = RecordingDispatcher(fail=True)
    runner = IngestionStageRunner(db, handlers=handlers(calls), dispatcher=dispatcher)

    with pytest.raises(RuntimeError, match="queue unavailable"):
        runner.run_stage("d1", "v1", "parse")

    db.expire_all()
    version = db.get(DocumentParseVersion, "pv1")
    assert version.stage_state["parse"]["status"] == "completed"
    assert version.stage_state["parse"]["next_enqueued"] is False
    assert calls == ["parse"]

    dispatcher.fail = False
    runner.run_stage("d1", "v1", "parse")
    assert calls == ["parse"]
    assert dispatcher.calls[-1] == ("d1", "v1", "repair")
    assert version.stage_state["parse"]["next_enqueued"] is True


def test_checkpoint_commit_failure_never_enqueues(db: Session, monkeypatch) -> None:
    calls: list[str] = []
    dispatcher = RecordingDispatcher()
    runner = IngestionStageRunner(db, handlers=handlers(calls), dispatcher=dispatcher)
    real_commit = db.commit
    commit_calls = 0

    def fail_completed_checkpoint() -> None:
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 2:
            raise RuntimeError("commit failed")
        real_commit()

    monkeypatch.setattr(db, "commit", fail_completed_checkpoint)
    with pytest.raises(RuntimeError, match="commit failed"):
        runner.run_stage("d1", "v1", "parse")

    assert dispatcher.calls == []


def test_unknown_out_of_order_and_concurrent_claim_are_rejected(db: Session) -> None:
    runner = IngestionStageRunner(db, handlers=handlers([]))
    with pytest.raises(ValueError, match="Unknown ingestion stage"):
        runner.run_stage("d1", "v1", "unknown")
    with pytest.raises(ValueError, match="out of order"):
        runner.run_stage("d1", "v1", "embed")

    version = db.get(DocumentParseVersion, "pv1")
    version.stage_state = {
        "parse": {
            "status": "running",
            "attempts": 1,
            "progress": 0,
            "input": {},
        }
    }
    db.commit()
    with pytest.raises(StageAlreadyClaimed):
        runner.run_stage("d1", "v1", "parse")


def test_expired_claim_is_taken_over_without_replaying_completed_stages(
    db: Session,
) -> None:
    now = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    calls: list[str] = []
    version = db.get(DocumentParseVersion, "pv1")
    version.status = "parsing"
    version.stage_state = {
        "parse": {
            "status": "running",
            "attempts": 1,
            "progress": 0,
            "input": {},
            "claim_owner": "dead-worker",
            "lease_expires_at": (now - timedelta(seconds=1)).isoformat(),
        }
    }
    db.commit()

    runner = IngestionStageRunner(
        db,
        handlers=handlers(calls),
        clock=lambda: now,
        claim_owner="replacement-worker",
        claim_ttl_seconds=60,
    )
    runner.run_stage("d1", "v1", "parse", enqueue_next=False)

    checkpoint = version.stage_state["parse"]
    assert calls == ["parse"]
    assert checkpoint["attempts"] == 2
    assert checkpoint["claim_owner"] == "replacement-worker"


def test_checkpoint_payload_larger_than_cap_fails_before_completion_commit(
    db: Session,
) -> None:
    runner = IngestionStageRunner(
        db,
        handlers={"parse": lambda _context: {"payload": "x" * 4096}},
        max_checkpoint_bytes=1024,
    )

    with pytest.raises(StageCheckpointTooLarge):
        runner.run_stage("d1", "v1", "parse", enqueue_next=False)

    version = db.get(DocumentParseVersion, "pv1")
    assert version.status == "parse_failed"
    assert version.stage_state["parse"]["status"] == "failed"


def test_oversized_stage_error_is_bounded_and_durably_recorded(
    db: Session,
) -> None:
    message = "sensitive failure\x00" + "x" * (300 * 1024)

    def fail(_context):
        raise RuntimeError(message)

    runner = IngestionStageRunner(db, handlers={"parse": fail})

    with pytest.raises(RuntimeError) as raised:
        runner.run_stage("d1", "v1", "parse", enqueue_next=False)
    assert not isinstance(raised.value, StageCheckpointTooLarge)
    db.rollback()

    version = db.get(DocumentParseVersion, "pv1")
    checkpoint = version.stage_state["parse"]
    assert version.status == "parse_failed"
    assert checkpoint["status"] == "failed"
    assert len(checkpoint["error"].encode("utf-8")) <= 4096
    assert "\x00" not in checkpoint["error"]
    assert checkpoint["error_sha256"] == hashlib.sha256(
        message.encode("utf-8")
    ).hexdigest()


def test_default_claim_lease_covers_rq_timeout_plus_safety_margin(
    db: Session,
) -> None:
    runner = IngestionStageRunner(db, handlers=handlers([]))
    assert runner.claim_ttl_seconds >= queue_module.settings.queue_job_timeout + 300


def test_dispatcher_uses_named_queue_and_deterministic_job_id(monkeypatch) -> None:
    enqueued: list[tuple[str, tuple, dict]] = []

    class FakeQueue:
        def __init__(self, name: str, connection) -> None:
            self.name = name

        def enqueue_call(self, *, func, args, **kwargs):
            enqueued.append((self.name, (func, *args), kwargs))
            return "job"

    monkeypatch.setattr(queue_module, "redis_connection", lambda: object())
    monkeypatch.setattr(queue_module, "Queue", FakeQueue)

    result = queue_module.JobDispatcher().enqueue_stage("d1", "v1", "embed")

    assert result == "job"
    assert enqueued == [
        (
            "ingest.embed",
            ("app.workers.jobs.run_ingestion_stage", "d1", "v1", "embed"),
            {
                "timeout": queue_module.settings.queue_job_timeout,
                "job_id": "ingestion-d1-v1-embed",
                "unique": True,
            },
        )
    ]


def test_dispatching_same_stage_twice_uses_unique_rq_contract(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeQueue:
        def __init__(self, name: str, connection) -> None:
            self.name = name

        def enqueue_call(self, **kwargs):
            calls.append(kwargs)
            return kwargs["job_id"]

    monkeypatch.setattr(queue_module, "redis_connection", lambda: object())
    monkeypatch.setattr(queue_module, "Queue", FakeQueue)
    dispatcher = queue_module.JobDispatcher()

    first = dispatcher.enqueue_stage("d1", "v1", "parse")
    second = dispatcher.enqueue_stage("d1", "v1", "parse")

    assert first == second == "ingestion-d1-v1-parse"
    assert len(calls) == 2
    assert all(call["unique"] is True for call in calls)


@pytest.mark.parametrize(
    ("status", "expected_action"),
    [
        ("queued", "returned"),
        ("started", "returned"),
        ("deferred", "returned"),
        ("scheduled", "returned"),
        ("failed", "requeued"),
        ("finished", "reenqueued"),
        ("stopped", "reenqueued"),
        ("canceled", "reenqueued"),
    ],
)
def test_duplicate_stage_job_is_reconciled_by_rq_status(
    monkeypatch, status: str, expected_action: str
) -> None:
    from rq.exceptions import DuplicateJobError

    events: list[str] = []

    class FakeJob:
        def get_status(self):
            return status

        def requeue(self):
            events.append("requeued")
            return self

        def delete(self, *, remove_from_queue):
            assert remove_from_queue is True
            events.append("deleted")

    existing = FakeJob()

    class FakeQueue:
        def __init__(self, name: str, connection) -> None:
            self.name = name
            self.attempts = 0

        def enqueue_call(self, **_kwargs):
            self.attempts += 1
            if status in {"finished", "stopped", "canceled"} and self.attempts > 1:
                events.append("reenqueued")
                return "replacement"
            raise DuplicateJobError("duplicate")

        def fetch_job(self, _job_id):
            return existing

    monkeypatch.setattr(queue_module, "redis_connection", lambda: object())
    monkeypatch.setattr(queue_module, "Queue", FakeQueue)

    result = queue_module.JobDispatcher().enqueue_stage("d1", "v1", "parse")

    if expected_action == "returned":
        assert result is existing
        assert events == []
    elif expected_action == "requeued":
        assert result is existing
        assert events == ["requeued"]
    else:
        assert result == "replacement"
        assert events == ["deleted", "reenqueued"]


def test_worker_consumes_all_stage_queues_in_order_then_legacy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeWorker:
        def __init__(self, queues, connection) -> None:
            captured["queues"] = queues
            captured["connection"] = connection

    connection = object()
    monkeypatch.setattr(queue_module, "redis_connection", lambda: connection)
    monkeypatch.setattr(queue_module, "Worker", FakeWorker)

    worker = queue_module.create_worker()

    assert isinstance(worker, FakeWorker)
    assert captured["queues"] == [
        *(STAGE_QUEUES[stage] for stage in INGESTION_STAGES),
        "ingest",
    ]
    assert captured["connection"] is connection


def test_worker_rejects_invalid_queue_or_concurrency_environment(monkeypatch) -> None:
    monkeypatch.setattr(queue_module, "redis_connection", lambda: object())
    monkeypatch.setenv("INGESTION_WORKER_CONCURRENCY", "2")
    with pytest.raises(RuntimeError, match="CONCURRENCY.*1"):
        queue_module.create_worker()

    monkeypatch.setenv("INGESTION_WORKER_CONCURRENCY", "1")
    monkeypatch.setenv("INGESTION_WORKER_QUEUES", "ingest.parse,ingest")
    with pytest.raises(RuntimeError, match="INGESTION_WORKER_QUEUES"):
        queue_module.create_worker()


def test_run_ingestion_stage_always_closes_session(monkeypatch) -> None:
    from app.workers import jobs

    events: list[object] = []

    class FakeSession:
        def close(self) -> None:
            events.append("closed")

    class FailingRunner:
        def __init__(self, db, **kwargs) -> None:
            events.append(db)

        def run_stage(self, document_id, version_key, stage):
            raise RuntimeError("stage failed")

    session = FakeSession()
    monkeypatch.setattr(jobs, "SessionLocal", lambda: session)
    monkeypatch.setattr(jobs, "IngestionStageRunner", FailingRunner)

    with pytest.raises(RuntimeError, match="stage failed"):
        jobs.run_ingestion_stage("d1", "v1", "parse")

    assert events == [session, "closed"]


def test_worker_wires_pipeline_identity_validation_before_stage_execution(
    monkeypatch,
) -> None:
    from app.workers import jobs

    captured: dict[str, object] = {}
    validator = object()

    class FakeSession:
        def close(self) -> None:
            pass

    class FakePipeline:
        def __init__(self, _db) -> None:
            self.validate_ingestion_identity = validator

        def ingestion_stage_handlers(self):
            return {}

    class CapturingRunner:
        def __init__(self, _db, **kwargs) -> None:
            captured.update(kwargs)

        def run_stage(self, _document_id, _version_key, _stage):
            return SimpleNamespace(id="version-id")

    monkeypatch.setattr(jobs, "SessionLocal", FakeSession)
    monkeypatch.setattr(jobs, "IngestionPipeline", FakePipeline)
    monkeypatch.setattr(jobs, "IngestionStageRunner", CapturingRunner)
    monkeypatch.setattr(jobs, "JobDispatcher", lambda: object())

    assert jobs.run_ingestion_stage("d1", "v1", "parse") == "version-id"
    assert captured["pre_stage_validator"] is validator


def _selective_contextualization_drafts(
    block_types: tuple[str, ...],
) -> list[ChunkDraft]:
    parent = ChunkDraft(
        local_id="parent-1",
        parse_version="v1",
        chunk_role="parent",
        block_type="narrative",
        text="Parent evidence",
        embedding_text="Parent evidence",
        token_count=2,
        source_block_ids=["block-parent"],
        source_spans=[{"page_index": 0, "page_label": "1"}],
        section_path=["Results"],
        ordinal=0,
        splitter_name="test",
        splitter_version="v1",
        splitting_model="test-model",
    )
    children = [
        ChunkDraft(
            local_id=f"child-{block_type}",
            parse_version="v1",
            chunk_role="child",
            block_type=block_type,
            text=f"{block_type} evidence",
            embedding_text=f"{block_type} evidence",
            token_count=2,
            parent_local_id=parent.local_id,
            source_block_ids=[f"block-{block_type}"],
            source_spans=[{"page_index": 0, "page_label": "1"}],
            section_path=["Results", block_type],
            ordinal=index + 1,
            splitter_name="test",
            splitter_version="v1",
            splitting_model="test-model",
        )
        for index, block_type in enumerate(block_types)
    ]
    return [parent, *children]


def _selective_contextualization_stage_context(
    db: Session,
    tmp_path: Path,
    drafts: list[ChunkDraft],
) -> SimpleNamespace:
    directory = tmp_path / "artifacts" / "d1" / "v1.pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        [draft.model_dump(mode="json") for draft in drafts],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    input_path = directory / "semantic_chunks.json"
    input_path.write_bytes(encoded)
    return SimpleNamespace(
        document=db.get(Document, "d1"),
        version=db.get(DocumentParseVersion, "pv1"),
        stage="contextualize",
        input={
            "previous_output": {
                "artifact_path": str(input_path),
                "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        },
    )


def test_contextualize_stage_never_calls_llm_for_any_child(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    from app.services import canonical_artifacts as artifacts_module
    from app.services import contextualization as contextualization_module
    from app.services import pipeline as pipeline_module

    block_types = ("narrative", "table", "figure", "formula", "caption", "appendix")
    drafts = _selective_contextualization_drafts(block_types)
    context = _selective_contextualization_stage_context(db, tmp_path, drafts)
    monkeypatch.setattr(
        pipeline_module.settings, "canonical_artifacts_dir", tmp_path / "artifacts"
    )
    monkeypatch.setattr(
        artifacts_module.CanonicalArtifactStore,
        "load",
        lambda _self, _document_id, _version_key: SimpleNamespace(
            title="Paper", abstract="Abstract", outline=[]
        ),
    )
    class FakeContextualizationService:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("ingestion must not construct the LLM service")

    monkeypatch.setattr(
        contextualization_module,
        "ContextualizationService",
        FakeContextualizationService,
    )

    output = IngestionPipeline(db)._run_contextualize_stage(context)

    assert output["child_count"] == 6
    assert output["contextualized_child_count"] == 0
    assert output["plain_child_count"] == 6
    artifact = json.loads(Path(output["artifact_path"]).read_text("utf-8"))
    by_type = {
        item["block_type"]: item
        for item in artifact
        if item["chunk_role"] == "child"
    }
    for block_type in block_types:
        assert by_type[block_type]["embedding_text"] == by_type[block_type]["text"]
        assert "contextual_prefix" not in by_type[block_type]


def test_contextualize_stage_skips_llm_when_no_structured_children(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    from app.services import canonical_artifacts as artifacts_module
    from app.services import contextualization as contextualization_module
    from app.services import pipeline as pipeline_module

    drafts = _selective_contextualization_drafts(
        ("narrative", "caption", "appendix", "table", "figure", "formula")
    )
    context = _selective_contextualization_stage_context(db, tmp_path, drafts)
    monkeypatch.setattr(
        pipeline_module.settings, "canonical_artifacts_dir", tmp_path / "artifacts"
    )
    monkeypatch.setattr(
        artifacts_module.CanonicalArtifactStore,
        "load",
        lambda _self, _document_id, _version_key: SimpleNamespace(
            title="Paper", abstract=None, outline=[]
        ),
    )

    class ForbiddenContextualizationService:
        def __init__(self) -> None:
            raise AssertionError("plain Children must not construct the LLM service")

    monkeypatch.setattr(
        contextualization_module,
        "ContextualizationService",
        ForbiddenContextualizationService,
    )

    output = IngestionPipeline(db)._run_contextualize_stage(context)

    assert output["child_count"] == 6
    assert output["contextualized_child_count"] == 0
    assert output["plain_child_count"] == 6


def test_embed_stage_vectors_mixed_children_and_keeps_parents(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    from app.services import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module.settings, "canonical_artifacts_dir", tmp_path / "artifacts"
    )
    monkeypatch.setattr(pipeline_module.settings, "ollama_embedding_dimensions", 2)
    pipeline = IngestionPipeline(db)
    directory = tmp_path / "artifacts" / "d1" / "v1.pipeline"
    directory.mkdir(parents=True)
    payload = [
        {
            "local_id": "parent-1",
            "chunk_role": "parent",
            "block_type": "narrative",
            "text": "Parent evidence",
            "embedding_text": "Parent evidence",
        },
        {
            "local_id": "child-narrative",
            "chunk_role": "child",
            "block_type": "narrative",
            "text": "Raw narrative",
            "embedding_text": "Raw narrative",
        },
        {
            "local_id": "child-table",
            "chunk_role": "child",
            "block_type": "table",
            "text": "| A | B |",
            "embedding_text": "| A | B |",
        },
        {
            "local_id": "child-figure",
            "chunk_role": "child",
            "block_type": "figure",
            "text": "Figure 1",
            "embedding_text": "Figure 1",
        },
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    input_path = directory / "contextualized_chunks.json"
    input_path.write_bytes(encoded)
    context = SimpleNamespace(
        document=db.get(Document, "d1"),
        version=db.get(DocumentParseVersion, "pv1"),
        stage="embed",
        input={
            "previous_output": {
                "artifact_path": str(input_path),
                "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        },
    )
    calls: list[list[str]] = []

    class FakeOllama:
        def embed(self, texts):
            calls.append(list(texts))
            return [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]

    pipeline.ollama = FakeOllama()

    output = pipeline._run_embed_stage(context)

    assert calls == [["Raw narrative", "| A | B |", "Figure 1"]]
    assert output["chunk_count"] == 4
    assert output["child_count"] == 3
    assert output["embedded_count"] == 3
    records = json.loads(Path(output["artifact_path"]).read_text("utf-8"))
    assert records == [
        {"chunk": payload[0], "embedding": None},
        {"chunk": payload[1], "embedding": [0.25, 0.75]},
        {"chunk": payload[2], "embedding": [0.5, 0.5]},
        {"chunk": payload[3], "embedding": [0.75, 0.25]},
    ]


@pytest.mark.parametrize(
    "child",
    [
        {
            "local_id": "child-narrative",
            "chunk_role": "child",
            "block_type": "narrative",
            "text": "Raw narrative",
            "embedding_text": "Unexpected context.\n\nRaw narrative",
        },
        {
            "local_id": "child-figure",
            "chunk_role": "child",
            "block_type": "figure",
            "text": "Figure 1",
            "contextual_prefix": "Figure context.",
            "contextualization_model": "context-model",
            "contextualization_version": "context-v1",
            "contextualization_prompt_version": "prompt-v1",
            "contextualized_at": "2026-07-28T12:00:00",
            "embedding_text": "Figure context.\n\nFigure 1",
        },
    ],
)
def test_embed_stage_rejects_children_that_violate_context_policy(
    db: Session, tmp_path: Path, monkeypatch, child: dict[str, object]
) -> None:
    from app.services import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module.settings, "canonical_artifacts_dir", tmp_path / "artifacts"
    )
    pipeline = IngestionPipeline(db)
    directory = tmp_path / "artifacts" / "d1" / "v1.pipeline"
    directory.mkdir(parents=True)
    payload = [child]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    input_path = directory / "contextualized_chunks.json"
    input_path.write_bytes(encoded)
    context = SimpleNamespace(
        document=db.get(Document, "d1"),
        version=db.get(DocumentParseVersion, "pv1"),
        stage="embed",
        input={
            "previous_output": {
                "artifact_path": str(input_path),
                "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        },
    )

    with pytest.raises(RuntimeError, match="contextualization policy"):
        pipeline._run_embed_stage(context)


@pytest.mark.parametrize(
    ("document_id", "version_key"),
    [("../escape", "v1"), ("d1", "CON")],
)
def test_stage_artifact_dir_rejects_nonportable_components(
    db: Session,
    tmp_path: Path,
    monkeypatch,
    document_id: str,
    version_key: str,
) -> None:
    from app.services import pipeline as pipeline_module

    root = tmp_path / "artifacts"
    monkeypatch.setattr(pipeline_module.settings, "canonical_artifacts_dir", root)
    context = SimpleNamespace(
        document=SimpleNamespace(id=document_id),
        version=SimpleNamespace(version_key=version_key, artifact_dir=None),
    )

    with pytest.raises(ValueError, match="path component"):
        IngestionPipeline(db)._stage_artifact_dir(context)

    assert not (tmp_path / "escape").exists()


def test_stage_artifact_dir_rejects_linked_document_root(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    from app.services import pipeline as pipeline_module

    root = tmp_path / "artifacts"
    external = tmp_path / "external-document"
    root.mkdir()
    external.mkdir()
    linked_document = root / "d1"
    _create_directory_link(linked_document, external)
    monkeypatch.setattr(pipeline_module.settings, "canonical_artifacts_dir", root)
    context = SimpleNamespace(
        document=SimpleNamespace(id="d1"),
        version=SimpleNamespace(version_key="v1", artifact_dir=None),
    )

    try:
        with pytest.raises(ValueError, match="symbolic link"):
            IngestionPipeline(db)._stage_artifact_dir(context)
        assert list(external.iterdir()) == []
    finally:
        if linked_document.is_symlink():
            linked_document.unlink()
        elif linked_document.exists():
            os.rmdir(linked_document)


def test_production_staged_parse_writes_artifact_without_legacy_side_effects(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "paper.md"
    source.write_text("# Evidence\n\nA durable source paragraph.", encoding="utf-8")
    document = db.get(Document, "d1")
    version = db.get(DocumentParseVersion, "pv1")
    document.raw_path = str(source)
    document.title = "Published title"
    document.metadata_json = {"published": "metadata"}
    document.status = "ready"
    document.active_parse_version = "legacy"
    version.artifact_dir = str(tmp_path / "artifacts" / "d1" / "v1")
    ingestion_config = _ingestion_config_fixture()
    ingestion_config_hash = hashlib.sha256(
        json.dumps(
            ingestion_config,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    version.manifest_json = {
        "ingestion_config": ingestion_config,
        "ingestion_config_sha256": ingestion_config_hash,
    }
    db.add(
        DocumentParseVersion(
            id="legacy-version",
            document_id="d1",
            version_key="legacy",
            artifact_dir="legacy",
            status="active",
        )
    )
    db.add(
        PipelineRun(
            id="run1",
            project_id="p1",
            document_id="d1",
            run_type=RunType.ingest.value,
            status=RunStatus.queued.value,
            provider_report={"existing": True},
        )
    )
    db.commit()

    from app.services import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module.settings, "canonical_artifacts_dir", tmp_path / "artifacts"
    )
    monkeypatch.setattr(pipeline_module.settings, "ollama_embedding_dimensions", 2)
    monkeypatch.setattr(
        pipeline_module,
        "build_ingestion_config_snapshot",
        lambda: ingestion_config,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "canonical_ingestion_config_hash",
        lambda _snapshot: ingestion_config_hash,
        raising=False,
    )
    pipeline = IngestionPipeline(db)
    pipeline.ollama = type(
        "FakeEmbeddingClient",
        (),
        {"embed": lambda _self, texts: [[1.0, float(index + 1)] for index, _ in enumerate(texts)]},
    )()

    parent = ChunkDraft(
        local_id="parent-1",
        parse_version="v1",
        chunk_role="parent",
        block_type="narrative",
        text="Parent evidence",
        embedding_text="Parent evidence",
        token_count=2,
        source_block_ids=["block-1"],
        source_spans=[{"page_index": 0, "page_label": "1"}],
        section_path=["Evidence"],
        ordinal=0,
        splitter_name="test",
        splitter_version="v1",
        splitting_model="test-model",
    )
    child = ChunkDraft(
        local_id="child-1",
        parse_version="v1",
        chunk_role="child",
        block_type="narrative",
        text="Child evidence",
        embedding_text="Child evidence",
        token_count=2,
        parent_local_id="parent-1",
        source_block_ids=["block-1"],
        source_spans=[{"page_index": 0, "page_label": "1"}],
        section_path=["Evidence"],
        ordinal=1,
        splitter_name="test",
        splitter_version="v1",
        splitting_model="test-model",
    )

    class FakeSemanticChunker:
        def __init__(self, _embedder) -> None:
            pass

        def build(self, _canonical):
            return [parent, child]

    class FakeContextualizationService:
        def contextualize(self, *, document, children, parents):
            assert document.title
            assert set(parents) == {"parent-1"}
            return [
                ContextualizedChunk.model_validate(
                    {
                        **item.model_dump(mode="json"),
                        "contextual_prefix": "Context for child.",
                        "embedding_text": f"Context for child.\n\n{item.text}",
                        "contextualization_model": "test-context-model",
                        "contextualization_version": "context-v1",
                        "contextualization_prompt_version": "prompt-v1",
                        "contextualized_at": datetime(2026, 7, 24, 12, 0),
                    }
                )
                for item in children
            ]

    from app.services import contextualization as contextualization_module
    from app.services import semantic_chunking as semantic_chunking_module

    monkeypatch.setattr(semantic_chunking_module, "SemanticChunker", FakeSemanticChunker)
    monkeypatch.setattr(
        contextualization_module,
        "ContextualizationService",
        FakeContextualizationService,
    )
    runner = IngestionStageRunner(db, handlers=pipeline.ingestion_stage_handlers())

    runner.run_stage("d1", "v1", "parse", enqueue_next=False)

    db.expire_all()
    stored = db.get(Document, "d1")
    stored_version = db.get(DocumentParseVersion, "pv1")
    output = stored_version.stage_state["parse"]["output"]
    assert stored.status == "ready"
    assert stored.active_parse_version == "legacy"
    assert stored.title == "Published title"
    assert stored.metadata_json == {"published": "metadata"}
    assert db.query(DocumentChunk).filter_by(document_id="d1").count() == 0
    assert output["artifact_path"].endswith(".canonical.json")
    assert Path(output["artifact_path"]).is_file()
    parse_artifact = json.loads(Path(output["artifact_path"]).read_text("utf-8"))
    assert parse_artifact["title"] == "Evidence"
    assert "metadata" in parse_artifact
    assert "provider_report" not in output
    assert stored_version.manifest_json["ingestion_config"] == ingestion_config
    assert (
        stored_version.manifest_json["ingestion_config_sha256"]
        == ingestion_config_hash
    )
    run = db.get(PipelineRun, "run1")
    assert run.status == RunStatus.running.value
    assert run.provider_report["existing"] is True
    assert run.provider_report["progress"]["stage"] == "parse"

    runner.run_stage("d1", "v1", "repair", enqueue_next=False)
    repair_output = db.get(DocumentParseVersion, "pv1").stage_state["repair"][
        "output"
    ]
    assert repair_output["artifact_path"].endswith("manifest.json")
    repaired_version = db.get(DocumentParseVersion, "pv1")
    assert repaired_version.manifest_json["ingestion_config"] == ingestion_config
    assert (
        repaired_version.manifest_json["ingestion_config_sha256"]
        == ingestion_config_hash
    )
    runner.run_stage("d1", "v1", "canonicalize", enqueue_next=False)
    runner.run_stage("d1", "v1", "semantic_split", enqueue_next=False)
    runner.run_stage("d1", "v1", "contextualize", enqueue_next=False)
    runner.run_stage("d1", "v1", "embed", enqueue_next=False)
    db.add(
        DocumentChunk(
            id="legacy-chunk",
            document_id="d1",
            parse_version="legacy",
            ordinal=0,
            text="Previously active evidence",
            embedding_text="Previously active evidence",
            embedding=[0.5, 0.5],
        )
    )
    db.commit()
    runner.run_stage("d1", "v1", "index", enqueue_next=False)

    db.expire_all()
    assert db.get(Document, "d1").status == "ready"
    assert db.get(Document, "d1").active_parse_version == "legacy"
    assert db.get(Document, "d1").title == "Published title"
    assert db.get(Document, "d1").metadata_json == {"published": "metadata"}
    assert db.get(DocumentParseVersion, "pv1").status == "ready_to_activate"
    rows = db.query(DocumentChunk).filter_by(document_id="d1", parse_version="v1").all()
    assert {row.chunk_role for row in rows} == {"parent", "child"}
    assert db.get(DocumentChunk, "legacy-chunk") is not None
    index_output = db.get(DocumentParseVersion, "pv1").stage_state["index"]["output"]
    assert Path(index_output["artifact_path"]).is_file()

    runner.run_stage("d1", "v1", "activate", enqueue_next=False)
    db.expire_all()
    assert db.get(Document, "d1").status == "ready"
    assert db.get(Document, "d1").active_parse_version == "v1"
    assert db.get(DocumentParseVersion, "pv1").status == "active"
    assert db.get(DocumentParseVersion, "legacy-version").status == "superseded"
    run = db.get(PipelineRun, "run1")
    assert run.status == RunStatus.completed.value
    assert run.provider_report["progress"]["stage"] == "completed"


# ---------------------------------------------------------------------------
# Typed inventory contract（Task 15：结构化库存与激活门禁）
# ---------------------------------------------------------------------------
def _manifest_inventory(
    *,
    tables=(),
    figure_ids=(),
    formula_ids=(),
    orphans=(),
    document_id: str = "d1",
    version: str = "v1",
) -> dict[str, object]:
    return {
        "document_id": document_id,
        "version": version,
        "tables": [
            {
                "table_id": table["table_id"],
                "row_count": table.get("row_count", 0),
                "source_block_ids": sorted(table.get("source_block_ids", [])),
                "child_ids": sorted(table.get("child_ids", [])),
                "child_count": len(table.get("child_ids", [])),
                "parent_ids": sorted(table.get("parent_ids", [])),
                "row_indices": sorted(table.get("row_indices", [])),
            }
            for table in tables
        ],
        "figure_ids": sorted(figure_ids),
        "formula_ids": sorted(formula_ids),
        "orphan_structured_chunks": sorted(orphans),
    }


def _observed_inventory(
    *,
    tables=(),
    figure_ids=(),
    formula_ids=(),
    orphans=(),
    cross_version=(),
) -> dict[str, object]:
    return {
        "tables": {
            table["table_id"]: {
                "parent_ids": sorted(table.get("parent_ids", [])),
                "child_ids": sorted(table.get("child_ids", [])),
                "source_block_ids": sorted(table.get("source_block_ids", [])),
                "row_indices": sorted(table.get("row_indices", [])),
            }
            for table in tables
        },
        "figure_ids": sorted(figure_ids),
        "formula_ids": sorted(formula_ids),
        "orphan_structured_chunks": sorted(orphans),
        "cross_version_chunk_ids": sorted(cross_version),
    }


def test_typed_inventory_matches_when_manifest_and_db_agree() -> None:
    manifest = _manifest_inventory(
        tables=[
            {
                "table_id": "t1",
                "row_count": 1,
                "source_block_ids": ["b1"],
                "child_ids": ["c1"],
                "parent_ids": ["p1"],
                "row_indices": [0],
            }
        ],
        figure_ids=["f1"],
        formula_ids=["m1"],
    )
    observed = _observed_inventory(
        tables=[
            {
                "table_id": "t1",
                "source_block_ids": ["b1"],
                "child_ids": ["c1"],
                "parent_ids": ["p1"],
                "row_indices": [0],
            }
        ],
        figure_ids=["f1"],
        formula_ids=["m1"],
    )

    assert compare_typed_inventory(manifest, observed) == []


@pytest.mark.parametrize(
    ("manifest_tables", "observed_tables", "expected_fragment"),
    [
        (
            [{"table_id": "t1", "row_count": 1, "source_block_ids": ["b1"], "child_ids": ["c1"], "parent_ids": ["p1"], "row_indices": [0]}],
            [],
            "missing tables: t1",
        ),
        (
            [{"table_id": "t1", "row_count": 1, "source_block_ids": ["b1"], "child_ids": ["c1", "c2"], "parent_ids": ["p1"], "row_indices": [0]}],
            [{"table_id": "t1", "source_block_ids": ["b1"], "child_ids": ["c1"], "parent_ids": ["p1"], "row_indices": [0]}],
            "missing children: c2",
        ),
        (
            [{"table_id": "t1", "row_count": 1, "source_block_ids": ["b1"], "child_ids": ["c1"], "parent_ids": ["p1"], "row_indices": [0]}],
            [{"table_id": "t1", "source_block_ids": ["b1"], "child_ids": ["c1", "c2"], "parent_ids": ["p1"], "row_indices": [0]}],
            "extra children: c2",
        ),
        (
            [{"table_id": "t1", "row_count": 1, "source_block_ids": ["b1"], "child_ids": ["c1"], "parent_ids": ["p1"], "row_indices": [0]}],
            [{"table_id": "t1", "source_block_ids": ["b1"], "child_ids": ["c1"], "parent_ids": ["p2"], "row_indices": [0]}],
            "parent ownership mismatch",
        ),
        (
            [{"table_id": "t1", "row_count": 1, "source_block_ids": ["b1"], "child_ids": ["c1"], "parent_ids": ["p1"], "row_indices": [0]}],
            [{"table_id": "t1", "source_block_ids": ["b2"], "child_ids": ["c1"], "parent_ids": ["p1"], "row_indices": [0]}],
            "source block mismatch",
        ),
        (
            [{"table_id": "t1", "row_count": 3, "source_block_ids": ["b1"], "child_ids": ["c1"], "parent_ids": ["p1"], "row_indices": [0, 1]}],
            [{"table_id": "t1", "source_block_ids": ["b1"], "child_ids": ["c1"], "parent_ids": ["p1"], "row_indices": [0, 1]}],
            "row count mismatch",
        ),
    ],
)
def test_typed_inventory_rejects_table_and_row_mismatches(
    manifest_tables,
    observed_tables,
    expected_fragment: str,
) -> None:
    manifest = _manifest_inventory(tables=manifest_tables)
    observed = _observed_inventory(tables=observed_tables)

    mismatches = compare_typed_inventory(manifest, observed)

    assert any(expected_fragment in message for message in mismatches)


def test_typed_inventory_rejects_duplicate_child_entries_in_manifest() -> None:
    manifest = _manifest_inventory(
        tables=[
            {
                "table_id": "t1",
                "row_count": 1,
                "source_block_ids": ["b1"],
                "child_ids": ["c1", "c1"],
                "child_count": 2,
                "parent_ids": ["p1"],
                "row_indices": [0],
            }
        ]
    )
    observed = _observed_inventory(
        tables=[
            {
                "table_id": "t1",
                "source_block_ids": ["b1"],
                "child_ids": ["c1"],
                "parent_ids": ["p1"],
                "row_indices": [0],
            }
        ]
    )

    mismatches = compare_typed_inventory(manifest, observed)

    assert any("duplicate child entries" in message for message in mismatches)


def test_typed_inventory_rejects_figure_and_formula_id_sets() -> None:
    manifest = _manifest_inventory(
        tables=[],
        figure_ids=["f1"],
        formula_ids=["m1"],
    )
    observed = _observed_inventory(
        tables=[],
        figure_ids=["f1"],
        formula_ids=[],
    )

    mismatches = compare_typed_inventory(manifest, observed)

    assert any("formula ID mismatch" in message for message in mismatches)

    mismatches = compare_typed_inventory(manifest, _observed_inventory(tables=[]))
    assert any("figure ID mismatch" in message for message in mismatches)


def test_typed_inventory_rejects_orphan_and_cross_version_records() -> None:
    manifest = _manifest_inventory(
        tables=[
            {
                "table_id": "t1",
                "row_count": 1,
                "source_block_ids": ["b1"],
                "child_ids": ["c1"],
                "parent_ids": ["p1"],
                "row_indices": [0],
            }
        ],
        figure_ids=["f1"],
        formula_ids=[],
    )
    observed = _observed_inventory(
        tables=[
            {
                "table_id": "t1",
                "source_block_ids": ["b1"],
                "child_ids": ["c1"],
                "parent_ids": ["p1"],
                "row_indices": [0],
            }
        ],
        figure_ids=["f1"],
        orphans=["orphan-table-chunk"],
        cross_version=["cross-version-child"],
    )

    mismatches = compare_typed_inventory(manifest, observed)

    assert any("orphan structured chunks" in message for message in mismatches)
    assert any("cross-version chunk contamination" in message for message in mismatches)


def test_derive_db_inventory_reads_structured_identity_from_persisted_spans() -> None:
    manifest = _manifest_inventory(
        tables=[
            {
                "table_id": "t1",
                "row_count": 1,
                "source_block_ids": ["b-table"],
                "child_ids": ["c1"],
                "parent_ids": ["p1"],
                "row_indices": [0],
            }
        ],
        figure_ids=["fig-1"],
        formula_ids=[],
    )
    rows = [
        DocumentChunk(
            id="p1",
            document_id="d1",
            parse_version="v1",
            chunk_role="parent",
            block_type="table",
            ordinal=0,
            text="parent",
            source_block_ids=["b-table"],
            source_spans=[{"metadata": {"table_id": "t1"}}],
        ),
        DocumentChunk(
            id="c1",
            document_id="d1",
            parse_version="v1",
            chunk_role="child",
            block_type="table",
            ordinal=1,
            text="child",
            source_block_ids=["b-table"],
            source_spans=[{"metadata": {"table_id": "t1"}, "row_index": 0}],
        ),
        DocumentChunk(
            id="f-child",
            document_id="d1",
            parse_version="v1",
            chunk_role="child",
            block_type="figure",
            ordinal=2,
            text="figure",
            source_spans=[{"metadata": {"figure_id": "fig-1"}}],
        ),
    ]

    observed = derive_db_typed_inventory(manifest, rows)

    assert observed["tables"]["t1"]["parent_ids"] == ["p1"]
    assert observed["tables"]["t1"]["child_ids"] == ["c1"]
    assert observed["tables"]["t1"]["source_block_ids"] == ["b-table"]
    assert observed["tables"]["t1"]["row_indices"] == [0]
    assert observed["figure_ids"] == ["fig-1"]
    assert observed["orphan_structured_chunks"] == []
    assert observed["cross_version_chunk_ids"] == []


def test_derive_db_inventory_detects_cross_version_and_orphans_untyped_chunks() -> None:
    manifest = _manifest_inventory(
        tables=[{"table_id": "t1", "row_count": 0, "source_block_ids": ["b1"]}]
    )
    rows = [
        DocumentChunk(
            id="untyped",
            document_id="d1",
            parse_version="v1",
            chunk_role="child",
            block_type="table",
            ordinal=0,
            text="untyped",
            source_block_ids=["unmapped-block"],
            source_spans=[],
        ),
        DocumentChunk(
            id="cross",
            document_id="d1",
            parse_version="other-version",
            chunk_role="child",
            block_type="narrative",
            ordinal=1,
            text="cross",
            source_spans=[],
        ),
    ]

    observed = derive_db_typed_inventory(manifest, rows)

    # 无结构归属的结构化分块是孤儿：保留 chunk id，激活 gate 据此拒绝。
    assert observed["orphan_structured_chunks"] == ["untyped"]
    assert observed["cross_version_chunk_ids"] == ["cross"]


def test_child_inventory_from_payload_is_deterministic_and_rejects_duplicates() -> None:
    payload = [
        {
            "chunk": {
                "local_id": "p1",
                "chunk_role": "parent",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [0, 1]},
            }
        },
        {
            "chunk": {
                "local_id": "c1",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [0]},
            }
        },
        {
            "chunk": {
                "local_id": "c2",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [1]},
            }
        },
        {
            "chunk": {
                "local_id": "f1",
                "chunk_role": "child",
                "block_type": "figure",
                "metadata": {"figure_id": "fig-1"},
            }
        },
        {
            "chunk": {
                "local_id": "m1",
                "chunk_role": "child",
                "block_type": "formula",
                "metadata": {"formula_id": "form-1"},
            }
        },
    ]

    inventory = derive_child_inventory_from_payload(payload)

    assert inventory["tables"] == {
        "t1": {
            "child_ids": ["c1", "c2"],
            "parent_ids": ["p1"],
            "child_count": 2,
        }
    }
    assert inventory["figure_ids"] == ["fig-1"]
    assert inventory["formula_ids"] == ["form-1"]
    assert inventory["row_indices"] == {"t1": [0, 1]}

    duplicate = [
        {"chunk": dict(payload[1]["chunk"])},
        {"chunk": dict(payload[1]["chunk"])},
    ]
    with pytest.raises(RuntimeError, match="duplicate chunk local ID"):
        derive_child_inventory_from_payload(duplicate)


def test_child_inventory_orphans_table_chunk_without_table_id() -> None:
    payload = [
        {
            "chunk": {
                "local_id": "c1",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {},
            }
        }
    ]

    inventory = derive_child_inventory_from_payload(payload)

    assert inventory["tables"] == {}
    assert inventory["figure_ids"] == []
    assert inventory["formula_ids"] == []
    assert inventory["orphan_structured_chunks"] == ["c1"]


def _table_chunk(
    chunk_id: str,
    *,
    table_id: str,
    chunk_role: str,
    ordinal: int,
    row_index: int | None = None,
) -> DocumentChunk:
    span: dict[str, object] = {"metadata": {"table_id": table_id}}
    if row_index is not None:
        span["row_index"] = row_index
    return DocumentChunk(
        id=chunk_id,
        document_id="d1",
        parse_version="v1",
        chunk_role=chunk_role,
        block_type="table",
        ordinal=ordinal,
        text=chunk_id,
        source_block_ids=[f"block-{table_id}"],
        source_spans=[span],
    )


def test_activation_inventory_gate_passes_when_manifest_and_db_match(
    db: Session,
    monkeypatch,
) -> None:
    from app.services import pipeline as pipeline_module
    from app.services.canonical_artifacts import CanonicalArtifactStore

    manifest = _manifest_inventory(
        tables=[
            {
                "table_id": "t1",
                "row_count": 1,
                "source_block_ids": ["block-t1"],
                "child_ids": ["c1"],
                "parent_ids": ["p1"],
                "row_indices": [0],
            }
        ],
        figure_ids=["fig-1"],
        formula_ids=["form-1"],
    )
    monkeypatch.setattr(
        CanonicalArtifactStore,
        "load_typed_inventory",
        lambda _self, _document_id, _version_key: manifest,
    )
    db.add_all(
        [
            _table_chunk("p1", table_id="t1", chunk_role="parent", ordinal=0),
            _table_chunk("c1", table_id="t1", chunk_role="child", ordinal=1, row_index=0),
            DocumentChunk(
                id="f-child",
                document_id="d1",
                parse_version="v1",
                chunk_role="child",
                block_type="figure",
                ordinal=2,
                text="figure",
                source_spans=[{"metadata": {"figure_id": "fig-1"}}],
            ),
            DocumentChunk(
                id="m-child",
                document_id="d1",
                parse_version="v1",
                chunk_role="child",
                block_type="formula",
                ordinal=3,
                text="formula",
                source_spans=[{"metadata": {"formula_id": "form-1"}}],
            ),
        ]
    )
    db.commit()
    context = SimpleNamespace(
        document=SimpleNamespace(id="d1"),
        version=SimpleNamespace(version_key="v1"),
        db=db,
    )

    pipeline = IngestionPipeline(db)
    pipeline._verify_typed_inventory_gate(context, "v1")


@pytest.mark.parametrize(
    ("manifest_tables", "db_rows", "expected_fragment"),
    [
        (
            [{"table_id": "t1", "row_count": 1, "source_block_ids": ["block-t1"], "child_ids": ["c1", "c2"], "parent_ids": ["p1"], "row_indices": [0]}],
            [
                {"chunk_id": "p1", "table_id": "t1", "chunk_role": "parent", "ordinal": 0},
                {"chunk_id": "c1", "table_id": "t1", "chunk_role": "child", "ordinal": 1, "row_index": 0},
            ],
            "missing children: c2",
        ),
        (
            [{"table_id": "t1", "row_count": 1, "source_block_ids": ["block-t1"], "child_ids": ["c1"], "parent_ids": ["p2"], "row_indices": [0]}],
            [
                {"chunk_id": "p1", "table_id": "t1", "chunk_role": "parent", "ordinal": 0},
                {"chunk_id": "c1", "table_id": "t1", "chunk_role": "child", "ordinal": 1, "row_index": 0},
            ],
            "parent ownership mismatch",
        ),
        (
            [{"table_id": "t1", "row_count": 1, "source_block_ids": ["block-other"], "child_ids": ["c1"], "parent_ids": ["p1"], "row_indices": [0]}],
            [
                {"chunk_id": "p1", "table_id": "t1", "chunk_role": "parent", "ordinal": 0},
                {"chunk_id": "c1", "table_id": "t1", "chunk_role": "child", "ordinal": 1, "row_index": 0},
            ],
            "source block mismatch",
        ),
        (
            [],
            [
                {"chunk_id": "p1", "table_id": "t1", "chunk_role": "parent", "ordinal": 0},
                {"chunk_id": "c1", "table_id": "t1", "chunk_role": "child", "ordinal": 1, "row_index": 0},
            ],
            "extra tables: t1",
        ),
    ],
)
def test_activation_inventory_gate_blocks_mismatched_db(
    db: Session,
    monkeypatch,
    manifest_tables,
    db_rows,
    expected_fragment: str,
) -> None:
    from app.services.canonical_artifacts import CanonicalArtifactStore

    manifest = _manifest_inventory(tables=manifest_tables)
    monkeypatch.setattr(
        CanonicalArtifactStore,
        "load_typed_inventory",
        lambda _self, _document_id, _version_key: manifest,
    )
    rows = [
        _table_chunk(
            row["chunk_id"],
            table_id=row["table_id"],
            chunk_role=row["chunk_role"],
            ordinal=row["ordinal"],
            row_index=row.get("row_index"),
        )
        for row in db_rows
    ]
    db.add_all(rows)
    db.commit()
    context = SimpleNamespace(
        document=SimpleNamespace(id="d1"),
        version=SimpleNamespace(version_key="v1"),
        db=db,
    )

    with pytest.raises(ActivationError, match=expected_fragment):
        IngestionPipeline(db)._verify_typed_inventory_gate(context, "v1")


def test_store_updates_and_loads_typed_inventory(tmp_path: Path) -> None:
    from app.services.canonical_artifacts import CanonicalArtifactStore

    document = CanonicalDocument(
        title="Untitled source",
        blocks=[
            CanonicalBlock(
                block_id="source-1",
                block_type="narrative",
                text="Source content.",
                reading_order=0,
                parser_source="fixture",
            )
        ],
    )
    store = CanonicalArtifactStore(tmp_path / "parsed")
    store.write_staging("doc-empty", "v1", document)
    store.promote("doc-empty", "v1")

    inventory = store.load_typed_inventory("doc-empty", "v1")
    assert inventory["tables"] == []
    assert inventory["figure_ids"] == []
    assert inventory["formula_ids"] == []

    with pytest.raises(ValueError, match="unknown tables"):
        store.update_typed_inventory(
            "doc-empty",
            "v1",
            {"tables": {"t1": {"child_ids": [], "parent_ids": []}}, "row_indices": {}},
        )

    store.update_typed_inventory("doc-empty", "v1", {"tables": {}, "row_indices": {}})
    assert store.load_typed_inventory("doc-empty", "v1")["tables"] == []


def test_child_only_row_coverage_cannot_hide_missing_child_rows(
    tmp_path: Path,
) -> None:
    from app.services.canonical_artifacts import CanonicalArtifactStore
    from app.services.canonical_models import CanonicalCell, CanonicalTable, SourceSpan

    payload = [
        {
            "chunk": {
                "local_id": "p1",
                "chunk_role": "parent",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [0, 1, 2]},
            }
        },
        {
            "chunk": {
                "local_id": "c1",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [0]},
            }
        },
        {
            "chunk": {
                "local_id": "c2",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [1]},
            }
        },
    ]
    inventory = derive_child_inventory_from_payload(payload)
    # 父块覆盖 [0,1,2] 不能掩盖缺失的 child 行 2。
    assert inventory["row_indices"] == {"t1": [0, 1]}

    document = CanonicalDocument(
        title="Untitled source",
        blocks=[
            CanonicalBlock(
                block_id="b-table",
                block_type="table",
                text="| A |",
                reading_order=0,
                parser_source="fixture",
                table_id="t1",
                source_spans=[SourceSpan(page_index=3, source_block_id="table-source")],
            )
        ],
        tables=[
            CanonicalTable(
                table_id="t1",
                headers=["A"],
                rows=[["r0"], ["r1"], ["r2"]],
                cells=[
                    CanonicalCell(text="A", row_index=0, column_index=0, is_header=True),
                    CanonicalCell(text="r0", row_index=1, column_index=0),
                    CanonicalCell(text="r1", row_index=2, column_index=0),
                    CanonicalCell(text="r2", row_index=3, column_index=0),
                ],
                normalized_markdown="| A |\n| --- |\n| r0 |\n| r1 |\n| r2 |",
                source_spans=[SourceSpan(page_index=3, source_block_id="table-source")],
            )
        ],
    )
    store = CanonicalArtifactStore(tmp_path / "parsed")
    store.write_staging("d1", "v1", document)
    store.promote("d1", "v1")

    with pytest.raises(ValueError, match="does not match row_count"):
        store.update_typed_inventory("d1", "v1", inventory)


def test_derive_db_inventory_row_coverage_is_child_only() -> None:
    manifest = _manifest_inventory(
        tables=[
            {
                "table_id": "t1",
                "row_count": 3,
                "source_block_ids": ["b1"],
                "child_ids": ["c1", "c2"],
                "parent_ids": ["p1"],
                "row_indices": [0, 1],
            }
        ]
    )
    rows = [
        DocumentChunk(
            id="p1",
            document_id="d1",
            parse_version="v1",
            chunk_role="parent",
            block_type="table",
            ordinal=0,
            text="parent",
            source_block_ids=["b1"],
            source_spans=[
                {"metadata": {"table_id": "t1"}, "row_index": 0},
                {"metadata": {"table_id": "t1"}, "row_index": 1},
                {"metadata": {"table_id": "t1"}, "row_index": 2},
            ],
        ),
        DocumentChunk(
            id="c1",
            document_id="d1",
            parse_version="v1",
            chunk_role="child",
            block_type="table",
            ordinal=1,
            text="child",
            source_block_ids=["b1"],
            source_spans=[{"metadata": {"table_id": "t1"}, "row_index": 0}],
        ),
        DocumentChunk(
            id="c2",
            document_id="d1",
            parse_version="v1",
            chunk_role="child",
            block_type="table",
            ordinal=2,
            text="child",
            source_block_ids=["b1"],
            source_spans=[{"metadata": {"table_id": "t1"}, "row_index": 1}],
        ),
    ]

    observed = derive_db_typed_inventory(manifest, rows)

    # 父块的行覆盖不参与 child 行覆盖判定；行 2 缺失被 gate 拒绝。
    assert observed["tables"]["t1"]["row_indices"] == [0, 1]
    mismatches = compare_typed_inventory(manifest, observed)
    assert any("row count mismatch" in message for message in mismatches)


def _text_markdown_table_drafts(
    *,
    table_id: str = "t1",
    parent_id: str = "p1",
    child_rows: dict[str, list[int]],
) -> list[ChunkDraft]:
    """构造 text/Markdown 风格的表格草稿：span 只带表级 table_id。

    每个 child 的 ``metadata.row_indices`` 表达其行覆盖，span 本身不带
    逐单元格 ``row_index``（模拟 Markdown/纯文本解析器，与 HTML/DOCX
    每 span 一个 row_index 的形态不同）。
    """
    parse_version = "v1"
    source_block_ids = [f"block-{table_id}"]
    drafts: list[ChunkDraft] = [
        ChunkDraft(
            local_id=parent_id,
            parse_version=parse_version,
            chunk_role="parent",
            block_type="table",
            text="table parent",
            embedding_text="table parent",
            token_count=4,
            source_block_ids=source_block_ids,
            source_spans=[
                {
                    "page_index": 0,
                    "page_label": "1",
                    "metadata": {"table_id": table_id},
                }
            ],
            section_path=["Results"],
            ordinal=0,
            splitter_name="test",
            splitter_version="v1",
            splitting_model="test-model",
            metadata={"table_id": table_id, "row_indices": [0, 1, 2]},
        )
    ]
    for ordinal, (child_id, row_indices) in enumerate(
        sorted(child_rows.items()), start=1
    ):
        drafts.append(
            ChunkDraft(
                local_id=child_id,
                parse_version=parse_version,
                chunk_role="child",
                block_type="table",
                text=f"rows {row_indices}",
                embedding_text=f"rows {row_indices}",
                token_count=4,
                parent_local_id=parent_id,
                source_block_ids=source_block_ids,
                source_spans=[
                    {
                        "page_index": 0,
                        "page_label": "1",
                        "metadata": {"table_id": table_id},
                    }
                ],
                section_path=["Results"],
                ordinal=ordinal,
                splitter_name="test",
                splitter_version="v1",
                splitting_model="test-model",
                metadata={"table_id": table_id, "row_indices": row_indices},
            )
        )
    return drafts


def _footnote_table_draft(
    *,
    local_id: str = "fn1",
    table_id: str = "t1",
    parent_id: str = "p1",
    row_indices: list[int] | None = None,
) -> ChunkDraft:
    """构造"仅脚注"表格子块草稿（现有脚注元数据契约）。

    脚注子块为引用上下文携带完整 ``row_indices``，metadata 含
    ``footnote_index``/``footnote_part_index``/``footnote_part_count``。
    """
    return ChunkDraft(
        local_id=local_id,
        parse_version="v1",
        chunk_role="child",
        block_type="table",
        text="Footnote: a citation note",
        embedding_text="Footnote: a citation note",
        token_count=4,
        parent_local_id=parent_id,
        source_block_ids=[f"block-{table_id}"],
        source_spans=[
            {
                "page_index": 0,
                "page_label": "1",
                "metadata": {"table_id": table_id},
            }
        ],
        section_path=["Results"],
        ordinal=10,
        splitter_name="test",
        splitter_version="v1",
        splitting_model="test-model",
        metadata={
            "table_id": table_id,
            "row_indices": (
                row_indices if row_indices is not None else [0, 1, 2]
            ),
            "footnote_index": 0,
            "footnote_part_index": 0,
            "footnote_part_count": 1,
        },
    )


def test_text_markdown_table_row_coverage_survives_persistence(
    db: Session,
    monkeypatch,
) -> None:
    from app.services.canonical_artifacts import CanonicalArtifactStore

    drafts = _text_markdown_table_drafts(child_rows={"c1": [0, 1], "c2": [2]})
    context = SimpleNamespace(
        document=db.get(Document, "d1"),
        version=db.get(DocumentParseVersion, "pv1"),
    )
    rows = [
        IngestionPipeline._document_chunk_from_draft(
            context,
            draft,
            embedding=None if draft.chunk_role == "parent" else [0.5, 0.5],
        )
        for draft in drafts
    ]
    # 持久化后的 span 不携带逐行 row_index，但必须保留 metadata.row_indices。
    for draft, row in zip(drafts, rows, strict=True):
        if draft.chunk_role != "child":
            continue
        assert row.source_spans
        for span in row.source_spans:
            assert span.get("row_index") is None
            assert span["metadata"]["row_indices"] == draft.metadata["row_indices"]

    db.add_all(rows)
    db.commit()
    persisted = list(
        db.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == "d1",
                DocumentChunk.parse_version == "v1",
            )
            .order_by(DocumentChunk.ordinal)
        ).all()
    )
    manifest = _manifest_inventory(
        tables=[
            {
                "table_id": "t1",
                "row_count": 3,
                "source_block_ids": ["block-t1"],
                "child_ids": ["c1", "c2"],
                "parent_ids": ["p1"],
                "row_indices": [0, 1, 2],
            }
        ]
    )
    observed = derive_db_typed_inventory(manifest, persisted)
    assert observed["tables"]["t1"]["row_indices"] == [0, 1, 2]
    assert compare_typed_inventory(manifest, observed) == []

    monkeypatch.setattr(
        CanonicalArtifactStore,
        "load_typed_inventory",
        lambda _self, _document_id, _version_key: manifest,
    )
    gate_context = SimpleNamespace(
        document=SimpleNamespace(id="d1"),
        version=SimpleNamespace(version_key="v1"),
        db=db,
    )
    IngestionPipeline(db)._verify_typed_inventory_gate(gate_context, "v1")


def test_text_markdown_table_missing_child_row_fails_closed(
    db: Session,
    monkeypatch,
) -> None:
    from app.services.canonical_artifacts import CanonicalArtifactStore

    drafts = _text_markdown_table_drafts(child_rows={"c1": [0, 1]})
    context = SimpleNamespace(
        document=db.get(Document, "d1"),
        version=db.get(DocumentParseVersion, "pv1"),
    )
    rows = [
        IngestionPipeline._document_chunk_from_draft(
            context,
            draft,
            embedding=None if draft.chunk_role == "parent" else [0.5, 0.5],
        )
        for draft in drafts
    ]
    db.add_all(rows)
    db.commit()
    persisted = list(
        db.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == "d1",
                DocumentChunk.parse_version == "v1",
            )
            .order_by(DocumentChunk.ordinal)
        ).all()
    )
    manifest = _manifest_inventory(
        tables=[
            {
                "table_id": "t1",
                "row_count": 3,
                "source_block_ids": ["block-t1"],
                "child_ids": ["c1", "c2"],
                "parent_ids": ["p1"],
                "row_indices": [0, 1, 2],
            }
        ]
    )
    observed = derive_db_typed_inventory(manifest, persisted)
    assert observed["tables"]["t1"]["row_indices"] == [0, 1]
    mismatches = compare_typed_inventory(manifest, observed)
    assert any("missing children: c2" in message for message in mismatches)
    assert any("row coverage mismatch" in message for message in mismatches)

    monkeypatch.setattr(
        CanonicalArtifactStore,
        "load_typed_inventory",
        lambda _self, _document_id, _version_key: manifest,
    )
    gate_context = SimpleNamespace(
        document=SimpleNamespace(id="d1"),
        version=SimpleNamespace(version_key="v1"),
        db=db,
    )
    with pytest.raises(ActivationError, match="row coverage mismatch"):
        IngestionPipeline(db)._verify_typed_inventory_gate(gate_context, "v1")


def test_payload_inventory_footnote_only_child_does_not_cover_rows() -> None:
    payload = [
        {
            "chunk": {
                "local_id": "c1",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [0]},
            }
        },
        {
            "chunk": {
                "local_id": "c2",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [1]},
            }
        },
        {
            "chunk": {
                "local_id": "fn",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {
                    "table_id": "t1",
                    "row_indices": [0, 1, 2],
                    "footnote_index": 0,
                    "footnote_part_index": 0,
                    "footnote_part_count": 1,
                },
            }
        },
    ]

    inventory = derive_child_inventory_from_payload(payload)

    # 脚注子块携带完整 row_indices，但不能掩盖缺失的数据行 2。
    assert inventory["row_indices"] == {"t1": [0, 1]}
    assert inventory["tables"]["t1"]["child_ids"] == ["c1", "c2", "fn"]


def test_payload_inventory_complete_rows_plus_footnote_passes() -> None:
    payload = [
        {
            "chunk": {
                "local_id": "c1",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [0, 1]},
            }
        },
        {
            "chunk": {
                "local_id": "c2",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [2]},
            }
        },
        {
            "chunk": {
                "local_id": "fn",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {
                    "table_id": "t1",
                    "row_indices": [0, 1, 2],
                    "footnote_index": 0,
                },
            }
        },
    ]

    inventory = derive_child_inventory_from_payload(payload)

    assert inventory["row_indices"] == {"t1": [0, 1, 2]}


def test_payload_inventory_missing_data_row_with_footnote_fails_closed(
    tmp_path: Path,
) -> None:
    from app.services.canonical_artifacts import CanonicalArtifactStore
    from app.services.canonical_models import CanonicalCell, CanonicalTable, SourceSpan

    payload = [
        {
            "chunk": {
                "local_id": "p1",
                "chunk_role": "parent",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [0, 1, 2]},
            }
        },
        {
            "chunk": {
                "local_id": "c1",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [0]},
            }
        },
        {
            "chunk": {
                "local_id": "c2",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {"table_id": "t1", "row_indices": [1]},
            }
        },
        {
            "chunk": {
                "local_id": "fn",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {
                    "table_id": "t1",
                    "row_indices": [0, 1, 2],
                    "footnote_index": 0,
                },
            }
        },
    ]
    inventory = derive_child_inventory_from_payload(payload)
    assert inventory["row_indices"] == {"t1": [0, 1]}

    document = CanonicalDocument(
        title="Untitled source",
        blocks=[
            CanonicalBlock(
                block_id="b-table",
                block_type="table",
                text="| A |",
                reading_order=0,
                parser_source="fixture",
                table_id="t1",
                source_spans=[SourceSpan(page_index=3, source_block_id="table-source")],
            )
        ],
        tables=[
            CanonicalTable(
                table_id="t1",
                headers=["A"],
                rows=[["r0"], ["r1"], ["r2"]],
                cells=[
                    CanonicalCell(text="A", row_index=0, column_index=0, is_header=True),
                    CanonicalCell(text="r0", row_index=1, column_index=0),
                    CanonicalCell(text="r1", row_index=2, column_index=0),
                    CanonicalCell(text="r2", row_index=3, column_index=0),
                ],
                normalized_markdown="| A |\n| --- |\n| r0 |\n| r1 |\n| r2 |",
                source_spans=[SourceSpan(page_index=3, source_block_id="table-source")],
            )
        ],
    )
    store = CanonicalArtifactStore(tmp_path / "parsed")
    store.write_staging("d1", "v1", document)
    store.promote("d1", "v1")

    with pytest.raises(ValueError, match="does not match row_count"):
        store.update_typed_inventory("d1", "v1", inventory)


def test_db_inventory_footnote_only_child_does_not_cover_rows(
    db: Session,
) -> None:
    drafts = _text_markdown_table_drafts(child_rows={"c1": [0], "c2": [1]})
    drafts.append(_footnote_table_draft(local_id="fn1"))
    context = SimpleNamespace(
        document=db.get(Document, "d1"),
        version=db.get(DocumentParseVersion, "pv1"),
    )
    rows = [
        IngestionPipeline._document_chunk_from_draft(
            context,
            draft,
            embedding=None if draft.chunk_role == "parent" else [0.5, 0.5],
        )
        for draft in drafts
    ]
    footnote_row = next(row for row in rows if row.id == "fn1")
    assert footnote_row.source_spans
    for span in footnote_row.source_spans:
        assert span["metadata"]["footnote_index"] == 0
        assert "row_indices" not in span["metadata"]
    db.add_all(rows)
    db.commit()
    persisted = list(
        db.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == "d1",
                DocumentChunk.parse_version == "v1",
            )
            .order_by(DocumentChunk.ordinal)
        ).all()
    )
    manifest = _manifest_inventory(
        tables=[
            {
                "table_id": "t1",
                "row_count": 3,
                "source_block_ids": ["block-t1"],
                "child_ids": ["c1", "c2", "fn1"],
                "parent_ids": ["p1"],
                "row_indices": [0, 1, 2],
            }
        ]
    )
    observed = derive_db_typed_inventory(manifest, persisted)
    # 脚注子块（携带完整 [0,1,2]）不贡献行覆盖：观测只有数据行 0/1。
    assert observed["tables"]["t1"]["row_indices"] == [0, 1]
    mismatches = compare_typed_inventory(manifest, observed)
    assert any("row coverage mismatch" in message for message in mismatches)


def test_db_inventory_complete_rows_plus_footnote_passes(db: Session) -> None:
    drafts = _text_markdown_table_drafts(child_rows={"c1": [0, 1], "c2": [2]})
    drafts.append(_footnote_table_draft(local_id="fn1"))
    context = SimpleNamespace(
        document=db.get(Document, "d1"),
        version=db.get(DocumentParseVersion, "pv1"),
    )
    rows = [
        IngestionPipeline._document_chunk_from_draft(
            context,
            draft,
            embedding=None if draft.chunk_role == "parent" else [0.5, 0.5],
        )
        for draft in drafts
    ]
    db.add_all(rows)
    db.commit()
    persisted = list(
        db.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == "d1",
                DocumentChunk.parse_version == "v1",
            )
            .order_by(DocumentChunk.ordinal)
        ).all()
    )
    manifest = _manifest_inventory(
        tables=[
            {
                "table_id": "t1",
                "row_count": 3,
                "source_block_ids": ["block-t1"],
                "child_ids": ["c1", "c2", "fn1"],
                "parent_ids": ["p1"],
                "row_indices": [0, 1, 2],
            }
        ]
    )
    observed = derive_db_typed_inventory(manifest, persisted)
    assert observed["tables"]["t1"]["row_indices"] == [0, 1, 2]
    assert compare_typed_inventory(manifest, observed) == []


def test_db_inventory_missing_data_row_with_footnote_fails_closed(
    db: Session,
) -> None:
    drafts = _text_markdown_table_drafts(child_rows={"c1": [0, 1]})
    drafts.append(_footnote_table_draft(local_id="fn1"))
    context = SimpleNamespace(
        document=db.get(Document, "d1"),
        version=db.get(DocumentParseVersion, "pv1"),
    )
    rows = [
        IngestionPipeline._document_chunk_from_draft(
            context,
            draft,
            embedding=None if draft.chunk_role == "parent" else [0.5, 0.5],
        )
        for draft in drafts
    ]
    db.add_all(rows)
    db.commit()
    persisted = list(
        db.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == "d1",
                DocumentChunk.parse_version == "v1",
            )
            .order_by(DocumentChunk.ordinal)
        ).all()
    )
    manifest = _manifest_inventory(
        tables=[
            {
                "table_id": "t1",
                "row_count": 3,
                "source_block_ids": ["block-t1"],
                "child_ids": ["c1", "fn1"],
                "parent_ids": ["p1"],
                "row_indices": [0, 1, 2],
            }
        ]
    )
    observed = derive_db_typed_inventory(manifest, persisted)
    # 脚注子块携带完整 row_indices，但不能掩盖缺失的数据行 2。
    assert observed["tables"]["t1"]["row_indices"] == [0, 1]
    mismatches = compare_typed_inventory(manifest, observed)
    assert any("row coverage mismatch" in message for message in mismatches)


def test_row_index_normalization_rejects_booleans() -> None:
    payload = [
        {
            "chunk": {
                "local_id": "c1",
                "chunk_role": "child",
                "block_type": "table",
                "metadata": {
                    "table_id": "t1",
                    "row_indices": [0, True, 1, False, 2],
                },
            }
        }
    ]
    inventory = derive_child_inventory_from_payload(payload)
    assert inventory["row_indices"] == {"t1": [0, 1, 2]}

    rows = [
        DocumentChunk(
            id="c1",
            document_id="d1",
            parse_version="v1",
            chunk_role="child",
            block_type="table",
            ordinal=0,
            text="child",
            source_spans=[
                {"metadata": {"table_id": "t1"}, "row_index": 0},
                {"metadata": {"table_id": "t1"}, "row_index": True},
                {"metadata": {"table_id": "t1"}, "row_index": 1},
                {"metadata": {"table_id": "t1"}, "row_index": False},
            ],
        ),
        DocumentChunk(
            id="c2",
            document_id="d1",
            parse_version="v1",
            chunk_role="child",
            block_type="table",
            ordinal=1,
            text="child",
            source_spans=[
                {
                    "metadata": {
                        "table_id": "t1",
                        "row_indices": [True, 0, False, 1],
                    }
                }
            ],
        ),
    ]
    observed = derive_db_typed_inventory(_manifest_inventory(tables=[]), rows)
    assert observed["tables"]["t1"]["row_indices"] == [0, 1]

    draft = ChunkDraft(
        local_id="c3",
        parse_version="v1",
        chunk_role="child",
        block_type="table",
        text="child",
        embedding_text="child",
        token_count=4,
        source_spans=[{"page_index": 0, "metadata": {"table_id": "t1"}}],
        section_path=["Results"],
        ordinal=0,
        splitter_name="test",
        splitter_version="v1",
        splitting_model="test-model",
        metadata={"table_id": "t1", "row_indices": [0, True, 1, False, 2]},
    )
    persisted = IngestionPipeline._persist_table_row_indices(
        draft, [{"page_index": 0, "metadata": {"table_id": "t1"}}]
    )
    assert persisted[0]["metadata"]["row_indices"] == [0, 1, 2]


def test_persist_table_row_indices_marks_footnote_only_children() -> None:
    footnote = _footnote_table_draft()
    persisted = IngestionPipeline._persist_table_row_indices(
        footnote,
        [{"page_index": 0, "metadata": {"table_id": "t1"}}],
    )
    assert persisted[0]["metadata"]["footnote_index"] == 0
    assert "row_indices" not in persisted[0]["metadata"]


def test_compare_rejects_orphans_even_when_manifest_recorded_the_same() -> None:
    manifest = _manifest_inventory(
        tables=[],
        orphans=["orphan-table-chunk"],
    )
    observed = _observed_inventory(orphans=["orphan-table-chunk"])

    mismatches = compare_typed_inventory(manifest, observed)

    assert any("orphan structured chunks" in message for message in mismatches)


def test_activation_inventory_gate_fails_closed_when_bundle_missing(
    db: Session,
    monkeypatch,
) -> None:
    from app.services.canonical_artifacts import CanonicalArtifactStore

    def missing_bundle(*_args, **_kwargs):
        raise FileNotFoundError("canonical bundle does not exist: /parsed/d1/v1")

    monkeypatch.setattr(CanonicalArtifactStore, "load_typed_inventory", missing_bundle)
    context = SimpleNamespace(
        document=SimpleNamespace(id="d1"),
        version=SimpleNamespace(version_key="v1"),
        db=db,
    )

    with pytest.raises(ActivationError, match="rebuild is required"):
        IngestionPipeline(db)._verify_typed_inventory_gate(context, "v1")


def test_activation_inventory_gate_fails_closed_on_missing_typed_inventory(
    db: Session,
    monkeypatch,
) -> None:
    from app.services.canonical_artifacts import CanonicalArtifactStore

    def missing_inventory(*_args, **_kwargs):
        raise ValueError("canonical manifest has no typed_inventory; rebuild required")

    monkeypatch.setattr(CanonicalArtifactStore, "load_typed_inventory", missing_inventory)
    context = SimpleNamespace(
        document=SimpleNamespace(id="d1"),
        version=SimpleNamespace(version_key="v1"),
        db=db,
    )

    with pytest.raises(
        ActivationError, match="Activation typed inventory validation failed"
    ):
        IngestionPipeline(db)._verify_typed_inventory_gate(context, "v1")


def test_legacy_typed_inventory_compatibility_and_missing_fail_closed(
    tmp_path: Path,
) -> None:
    from app.services.canonical_artifacts import CanonicalArtifactStore

    document = CanonicalDocument(
        title="Untitled source",
        blocks=[
            CanonicalBlock(
                block_id="source-1",
                block_type="narrative",
                text="Source content.",
                reading_order=0,
                parser_source="fixture",
            )
        ],
    )
    store = CanonicalArtifactStore(tmp_path / "parsed")
    store.write_staging("doc-empty", "v1", document)
    store.promote("doc-empty", "v1")
    manifest_path = tmp_path / "parsed" / "doc-empty" / "v1" / "manifest.json"

    # 新 schema：typed_inventory 只位于 document 下，顶层不存在。
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert "typed_inventory" not in manifest
    assert "typed_inventory" in manifest["document"]
    assert store.load_typed_inventory("doc-empty", "v1")["tables"] == []

    # legacy 顶层形式：只读/迁移兼容，load_typed_inventory 正常化返回。
    manifest["typed_inventory"] = manifest["document"].pop("typed_inventory")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert store.load("doc-empty", "v1").document_id == "doc-empty"
    assert store.load_typed_inventory("doc-empty", "v1")["tables"] == []

    # legacy 顶层 bundle 的 in-place 更新失败关闭，要求迁移。
    with pytest.raises(ValueError, match="legacy"):
        store.update_typed_inventory(
            "doc-empty", "v1", {"tables": {}, "row_indices": {}}
        )

    # 完全缺失 typed_inventory：静态读取可用，但 load_typed_inventory 失败关闭。
    manifest = json.loads(manifest_path.read_text("utf-8"))
    del manifest["typed_inventory"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert store.load("doc-empty", "v1").document_id == "doc-empty"
    with pytest.raises(ValueError, match="rebuilt/migrated"):
        store.load_typed_inventory("doc-empty", "v1")

    # 顶层与 document 层同时存在：歧义，失败关闭。
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["typed_inventory"] = {
        "document_id": "doc-empty",
        "version": "v1",
        "tables": [],
        "figure_ids": [],
        "formula_ids": [],
        "orphan_structured_chunks": [],
    }
    manifest["document"]["typed_inventory"] = {
        "document_id": "doc-empty",
        "version": "v1",
        "tables": [],
        "figure_ids": [],
        "formula_ids": [],
        "orphan_structured_chunks": [],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="both top-level and document-level"):
        store.load_typed_inventory("doc-empty", "v1")
