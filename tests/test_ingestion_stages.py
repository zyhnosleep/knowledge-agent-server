from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

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
    StageHandlerUnavailable,
)
from app.services import queue as queue_module


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


def test_dispatcher_uses_named_queue_and_deterministic_job_id(monkeypatch) -> None:
    enqueued: list[tuple[str, tuple, dict]] = []

    class FakeQueue:
        def __init__(self, name: str, connection) -> None:
            self.name = name

        def enqueue(self, func, *args, **kwargs):
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
                "job_timeout": queue_module.settings.queue_job_timeout,
                "job_id": "ingestion:d1:v1:embed",
            },
        )
    ]


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


def test_production_staged_parse_writes_artifact_without_legacy_side_effects(
    db: Session, tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "paper.md"
    source.write_text("# Evidence\n\nA durable source paragraph.", encoding="utf-8")
    document = db.get(Document, "d1")
    version = db.get(DocumentParseVersion, "pv1")
    document.raw_path = str(source)
    version.artifact_dir = str(tmp_path / "artifacts" / "d1" / "v1")
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
    runner = IngestionStageRunner(db, handlers=pipeline.ingestion_stage_handlers())

    runner.run_stage("d1", "v1", "parse", enqueue_next=False)

    db.expire_all()
    stored = db.get(Document, "d1")
    stored_version = db.get(DocumentParseVersion, "pv1")
    output = stored_version.stage_state["parse"]["output"]
    assert stored.status == "quality_checking"
    assert db.query(DocumentChunk).filter_by(document_id="d1").count() == 0
    assert output["artifact_path"].endswith(".canonical.json")
    assert Path(output["artifact_path"]).is_file()
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
    with pytest.raises(StageHandlerUnavailable, match="semantic_split"):
        runner.run_stage("d1", "v1", "semantic_split", enqueue_next=False)

    db.expire_all()
    assert db.get(Document, "d1").status == "parse_failed"
    assert db.get(DocumentParseVersion, "pv1").status == "parse_failed"
    assert db.query(DocumentChunk).filter_by(document_id="d1").count() == 0
    run = db.get(PipelineRun, "run1")
    assert run.status == RunStatus.failed.value
    assert run.provider_report["progress"]["stage"] == "semantic_split_failed"
