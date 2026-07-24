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
from sqlalchemy import create_engine
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
from app.services.pipeline import IngestionPipeline
from app.services.ingestion_stages import (
    INGESTION_STAGES,
    STAGE_QUEUES,
    IngestionStageRunner,
    StageAlreadyClaimed,
    StageCheckpointTooLarge,
    CapabilityUnavailable,
)
from app.services import queue as queue_module
from app.services.contextualization import ContextualizedChunk
from app.services.canonical_models import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalQualityIssue,
    CanonicalQualityReport,
)
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


def test_process_document_enqueues_first_failed_or_incomplete_stage(
    db: Session, monkeypatch
) -> None:
    version = db.get(DocumentParseVersion, "pv1")
    version.version_key = "canonical-v1-abc"
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

    assert calls == [("d1", "canonical-v1-abc", "canonicalize")]


def test_process_document_keeps_completed_version_ready_without_enqueue(
    db: Session, monkeypatch
) -> None:
    version = db.get(DocumentParseVersion, "pv1")
    version.version_key = "canonical-v1-abc"
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
    version.version_key = "canonical-v1-abc"
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

    assert enqueued == [("d1", "canonical-v1-abc", "parse")]
    assert db.get(Document, "d1").status == "ready"
    assert db.get(Document, "d1").active_parse_version == "legacy"

    stage_handlers = handlers([])
    stage_handlers["activate"] = pipeline.ingestion_stage_handlers()["activate"]
    runner = IngestionStageRunner(db, handlers=stage_handlers)
    runner.run_until_blocked("d1", "canonical-v1-abc")
    with pytest.raises(CapabilityUnavailable, match="Task 10"):
        runner.run_stage(
            "d1", "canonical-v1-abc", "activate", enqueue_next=False
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
    version.version_key = "canonical-v1-abc"
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
        ("d1", "canonical-v1-abc", "parse"),
        ("d1", "canonical-v1-abc", "parse"),
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
            )
        ],
        metadata={"expected_page_count": 1, "text_layer_pages": ["Evidence"]},
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


def test_embed_stage_vectors_only_contextualized_children_and_keeps_parents(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    from app.services import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module.settings, "canonical_artifacts_dir", tmp_path / "artifacts"
    )
    pipeline = IngestionPipeline(db)
    directory = tmp_path / "artifacts" / "d1" / "v1.pipeline"
    directory.mkdir(parents=True)
    payload = [
        {
            "local_id": "parent-1",
            "chunk_role": "parent",
            "text": "Parent evidence",
            "embedding_text": "Parent evidence",
        },
        {
            "local_id": "child-1",
            "chunk_role": "child",
            "text": "Child evidence",
            "contextual_prefix": "Child context.",
            "contextualization_model": "context-model",
            "embedding_text": "Child context.\n\nChild evidence",
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
            return [[0.25, 0.75]]

    pipeline.ollama = FakeOllama()

    output = pipeline._run_embed_stage(context)

    assert calls == [["Child context.\n\nChild evidence"]]
    assert output["chunk_count"] == 2
    assert output["child_count"] == 1
    assert output["embedded_count"] == 1
    records = json.loads(Path(output["artifact_path"]).read_text("utf-8"))
    assert records == [
        {"chunk": payload[0], "embedding": None},
        {"chunk": payload[1], "embedding": [0.25, 0.75]},
    ]


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
        source_spans=[],
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
        source_spans=[],
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
    run = db.get(PipelineRun, "run1")
    assert run.status == RunStatus.running.value
    assert run.provider_report["existing"] is True
    assert run.provider_report["progress"]["stage"] == "parse"

    runner.run_stage("d1", "v1", "repair", enqueue_next=False)
    repair_output = db.get(DocumentParseVersion, "pv1").stage_state["repair"][
        "output"
    ]
    assert repair_output["artifact_path"].endswith("manifest.json")
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
    assert rows == []
    assert db.get(DocumentChunk, "legacy-chunk") is not None
    index_output = db.get(DocumentParseVersion, "pv1").stage_state["index"]["output"]
    assert Path(index_output["artifact_path"]).is_file()

    with pytest.raises(CapabilityUnavailable, match="Task 10"):
        runner.run_stage("d1", "v1", "activate", enqueue_next=False)
    db.expire_all()
    assert db.get(Document, "d1").status == "ready"
    assert db.get(Document, "d1").active_parse_version == "legacy"
    assert db.get(DocumentParseVersion, "pv1").status == "activation_failed"
    run = db.get(PipelineRun, "run1")
    assert run.status == RunStatus.failed.value
    assert run.provider_report["progress"]["stage"] == "activate_failed"
