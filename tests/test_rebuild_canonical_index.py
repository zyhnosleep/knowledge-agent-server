from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import Document, DocumentChunk, DocumentParseVersion, Project
from scripts import rebuild_canonical_index as rebuild_module


def _test_ingestion_config() -> dict[str, object]:
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
            "content_sha256": "a" * 64,
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


def _canonical_config_hash(snapshot: dict[str, object]) -> str:
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _test_ingestion_config_hash() -> str:
    return _canonical_config_hash(_test_ingestion_config())


def _test_version_key(document: Document) -> str:
    return f"canonical-v4-{document.sha256[:12]}-{_test_ingestion_config_hash()[:12]}"


@pytest.fixture
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as session:
        yield session


@pytest.fixture(autouse=True)
def stub_model_release(monkeypatch) -> None:
    monkeypatch.setattr(rebuild_module, "_release_loaded_models", lambda: [])
    monkeypatch.setattr(
        rebuild_module,
        "_strict_child_token_count",
        lambda text: len(text.split()),
    )
    monkeypatch.setattr(
        rebuild_module, "build_ingestion_config_snapshot", _test_ingestion_config
    )
    monkeypatch.setattr(
        rebuild_module,
        "canonical_ingestion_config_hash",
        lambda _snapshot: _test_ingestion_config_hash(),
    )


@pytest.fixture
def documents(db: Session, tmp_path: Path) -> list[Document]:
    project = Project(id="p1", slug="research", name="Research")
    rows = []
    for index in range(2):
        source = tmp_path / f"paper-{index}.pdf"
        source.write_bytes(b"%PDF fixture")
        rows.append(
            Document(
                id=f"d{index + 1}",
                project_id=project.id,
                title=f"Paper {index + 1}",
                file_name=source.name,
                sha256=f"sha{index + 1:061d}",
                raw_path=str(source),
            )
        )
    db.add_all([project, *rows])
    db.commit()
    return rows


def test_rebuild_dry_run_does_not_mutate_documents(
    db: Session, documents: list[Document], tmp_path: Path
) -> None:
    report = rebuild_module.rebuild(
        db,
        artifact_root=tmp_path / "parsed",
        dry_run=True,
    )

    assert report.total == len(documents)
    assert report.failed_documents == 0
    assert all(row.source_exists for row in report.documents)
    assert all(row.planned_version.startswith("canonical-v4-") for row in report.documents)
    assert list(db.scalars(select(DocumentParseVersion))) == []
    assert all(document.active_parse_version is None for document in documents)
    assert not (tmp_path / "parsed").exists()


def test_rebuild_report_exposes_tokenizer_and_config_identity(
    db: Session, documents: list[Document], tmp_path: Path, monkeypatch
) -> None:
    snapshot = _test_ingestion_config()
    config_hash = _test_ingestion_config_hash()
    monkeypatch.setattr(
        rebuild_module,
        "build_ingestion_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    monkeypatch.setattr(
        rebuild_module,
        "canonical_ingestion_config_hash",
        lambda _snapshot: config_hash,
        raising=False,
    )

    report = rebuild_module.rebuild(
        db,
        artifact_root=tmp_path / "parsed",
        dry_run=True,
    )

    assert report.tokenizer_available is True
    assert report.ingestion_config == snapshot
    assert report.ingestion_config_sha256 == config_hash
    assert all(
        row.planned_version.endswith(f"-{config_hash[:12]}")
        for row in report.documents
    )


def _complete_activation_metrics() -> dict[str, float | None]:
    return {
        "parse_completeness": 1.0,
        **{name: 1.0 for name in rebuild_module.STRICT_REBUILD_METRICS},
        "pgvector_completeness": 1.0,
    }


def test_activate_rebuild_batch_rejects_acceptance_for_another_version_map(
    db: Session,
    monkeypatch,
) -> None:
    called = False

    class FailIfConstructed:
        def __init__(self, *_args, **_kwargs):
            nonlocal called
            called = True

    monkeypatch.setattr(rebuild_module, "IngestionStageRunner", FailIfConstructed)

    with pytest.raises(ValueError, match="version map"):
        rebuild_module.activate_rebuild_batch(
            db,
            parse_version_map={"d1": "new-v"},
            acceptance_report={
                "strict_pass": True,
                "parse_version_map": {"d1": "other-v"},
            },
        )

    assert called is False


def test_activate_rebuild_batch_rechecks_integrity_before_delegating(
    db: Session,
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}
    shadow = {"d1": "new-v"}

    def fake_metrics(_db, **kwargs):
        observed["metrics_kwargs"] = kwargs
        return _complete_activation_metrics()

    class FakePipeline:
        def __init__(self, actual_db):
            observed["pipeline_db"] = actual_db

        def ingestion_stage_handlers(self):
            return {"activate": object()}

        def validate_ingestion_identity(self, *_args):
            return None

    class FakeRunner:
        def __init__(self, actual_db, **kwargs):
            observed["runner_db"] = actual_db
            observed["runner_kwargs"] = kwargs

        def activate_batch(self, version_map):
            observed["activated_map"] = version_map
            return ["activated"]

    monkeypatch.setattr(rebuild_module, "collect_integrity_metrics", fake_metrics)
    monkeypatch.setattr(rebuild_module, "IngestionPipeline", FakePipeline)
    monkeypatch.setattr(rebuild_module, "IngestionStageRunner", FakeRunner)
    monkeypatch.setattr(
        rebuild_module.CanonicalArtifactStore,
        "load_typed_inventory",
        lambda _self, _document_id, _version_key: {
            "document_id": "d1",
            "version": "new-v",
            "tables": [],
            "figure_ids": [],
            "formula_ids": [],
        },
    )

    result = rebuild_module.activate_rebuild_batch(
        db,
        parse_version_map=shadow,
        acceptance_report={
            "strict_pass": True,
            "parse_version_map": shadow,
        },
    )

    assert result == ["activated"]
    assert observed["activated_map"] == shadow
    metrics_kwargs = observed["metrics_kwargs"]
    assert metrics_kwargs["parse_version_map"] == shadow
    assert metrics_kwargs["document_ids"] == {"d1"}
    assert metrics_kwargs["expected_ingestion_config"] == _test_ingestion_config()
    assert metrics_kwargs["expected_ingestion_config_sha256"] == _test_ingestion_config_hash()


def test_activate_rebuild_batch_stops_when_rechecked_integrity_regresses(
    db: Session,
    monkeypatch,
) -> None:
    metrics = _complete_activation_metrics()
    metrics["source_span_validity"] = 0.5
    monkeypatch.setattr(
        rebuild_module,
        "collect_integrity_metrics",
        lambda *_args, **_kwargs: metrics,
    )
    monkeypatch.setattr(
        rebuild_module,
        "IngestionStageRunner",
        lambda *_args, **_kwargs: pytest.fail("activation runner must not start"),
    )

    with pytest.raises(RuntimeError, match="source_span_validity"):
        rebuild_module.activate_rebuild_batch(
            db,
            parse_version_map={"d1": "new-v"},
            acceptance_report={
                "strict_pass": True,
                "parse_version_map": {"d1": "new-v"},
            },
        )


def test_activate_rebuild_batch_rejects_typed_inventory_mismatch_before_runner(
    db: Session,
    monkeypatch,
) -> None:
    shadow = {"d1": "new-v"}
    monkeypatch.setattr(
        rebuild_module,
        "collect_integrity_metrics",
        lambda *_args, **_kwargs: _complete_activation_metrics(),
    )
    monkeypatch.setattr(
        rebuild_module.CanonicalArtifactStore,
        "load_typed_inventory",
        lambda _self, _document_id, _version_key: {
            "document_id": "d1",
            "version": "new-v",
            "tables": [
                {
                    "table_id": "t1",
                    "row_count": 1,
                    "source_block_ids": ["block-t1"],
                    "child_ids": ["c1"],
                    "child_count": 1,
                    "parent_ids": ["p1"],
                    "row_indices": [0],
                }
            ],
            "figure_ids": [],
            "formula_ids": [],
        },
    )
    monkeypatch.setattr(
        rebuild_module,
        "IngestionPipeline",
        lambda _db: pytest.fail("pipeline must not start on inventory mismatch"),
    )
    monkeypatch.setattr(
        rebuild_module,
        "IngestionStageRunner",
        lambda *_args, **_kwargs: pytest.fail("activation runner must not start"),
    )

    with pytest.raises(RuntimeError, match="typed inventory"):
        rebuild_module.activate_rebuild_batch(
            db,
            parse_version_map=shadow,
            acceptance_report={
                "strict_pass": True,
                "parse_version_map": shadow,
            },
        )


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("canonical bundle does not exist; rebuild is required"),
        ValueError("canonical manifest has no typed_inventory; rebuild required"),
    ],
)
def test_activate_rebuild_batch_fails_closed_when_inventory_unavailable(
    db: Session,
    monkeypatch,
    exc: Exception,
) -> None:
    shadow = {"d1": "new-v"}
    monkeypatch.setattr(
        rebuild_module,
        "collect_integrity_metrics",
        lambda *_args, **_kwargs: _complete_activation_metrics(),
    )

    def unavailable(_self, _document_id, _version_key):
        raise exc

    monkeypatch.setattr(
        rebuild_module.CanonicalArtifactStore,
        "load_typed_inventory",
        unavailable,
    )
    monkeypatch.setattr(
        rebuild_module,
        "IngestionPipeline",
        lambda _db: pytest.fail("pipeline must not start on unavailable inventory"),
    )
    monkeypatch.setattr(
        rebuild_module,
        "IngestionStageRunner",
        lambda *_args, **_kwargs: pytest.fail("activation runner must not start"),
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        rebuild_module.activate_rebuild_batch(
            db,
            parse_version_map=shadow,
            acceptance_report={
                "strict_pass": True,
                "parse_version_map": shadow,
            },
        )


def test_integrity_rejects_wrong_source_version_identity(
    db: Session,
    documents: list[Document],
) -> None:
    document = documents[0]
    wrong_version = "canonical-v4-wrong-source-" + _test_ingestion_config_hash()[:12]
    db.add(
        DocumentParseVersion(
            document_id=document.id,
            version_key=wrong_version,
            artifact_dir=f"parsed/{document.id}/{wrong_version}",
            status="ready_to_activate",
            manifest_json={
                "ingestion_config": _test_ingestion_config(),
                "ingestion_config_sha256": _test_ingestion_config_hash(),
            },
        )
    )
    db.commit()

    metrics = rebuild_module.collect_integrity_metrics(
        db,
        document_ids={document.id},
        expected_ingestion_config=_test_ingestion_config(),
        expected_ingestion_config_sha256=_test_ingestion_config_hash(),
        parse_version_map={document.id: wrong_version},
    )

    assert metrics["source_version_identity_completeness"] == 0.0
    assert "source_version_identity_completeness" in rebuild_module.STRICT_REBUILD_METRICS


def test_integrity_retokenizes_children_and_rejects_actual_over_limit_text(
    db: Session,
    documents: list[Document],
    monkeypatch,
) -> None:
    document = documents[0]
    version_key = _test_version_key(document)
    db.add_all(
        [
            DocumentParseVersion(
                document_id=document.id,
                version_key=version_key,
                artifact_dir=f"parsed/{document.id}/{version_key}",
                status="ready_to_activate",
                manifest_json={
                    "ingestion_config": _test_ingestion_config(),
                    "ingestion_config_sha256": _test_ingestion_config_hash(),
                },
            ),
            DocumentChunk(
                id="over-limit-child",
                document_id=document.id,
                parse_version=version_key,
                chunk_role="child",
                block_type="narrative",
                ordinal=0,
                text="stored count is stale",
                embedding_text="stored count is stale",
                token_count=4,
                source_spans=[{"page_index": 0, "page_label": "1"}],
                embedding=[1.0, 0.0],
            ),
        ]
    )
    db.commit()
    monkeypatch.setattr(
        rebuild_module,
        "_strict_child_token_count",
        lambda _text: 601,
        raising=False,
    )

    metrics = rebuild_module.collect_integrity_metrics(
        db,
        document_ids={document.id},
        expected_ingestion_config=_test_ingestion_config(),
        expected_ingestion_config_sha256=_test_ingestion_config_hash(),
        parse_version_map={document.id: version_key},
    )

    assert metrics["child_token_limit_completeness"] == 0.0
    assert "child_token_limit_completeness" in rebuild_module.STRICT_REBUILD_METRICS


def test_rebuild_tokenizer_preflight_is_not_ready_and_never_starts_pipeline(
    db: Session, documents: list[Document], monkeypatch
) -> None:
    from app.services.ingestion_identity import TokenizerUnavailableError

    def unavailable():
        raise TokenizerUnavailableError("pinned tokenizer missing from local cache")

    class ForbiddenPipeline:
        def __init__(self, _session) -> None:
            raise AssertionError("pipeline must not start without tokenizer identity")

    monkeypatch.setattr(rebuild_module, "build_ingestion_config_snapshot", unavailable)
    monkeypatch.setattr(rebuild_module, "IngestionPipeline", ForbiddenPipeline)

    report = rebuild_module.rebuild(db)

    assert report.ready_for_acceptance is False
    assert report.tokenizer_available is False
    assert report.ingestion_config is None
    assert report.ingestion_config_sha256 is None
    assert report.failed_document_ids == [document.id for document in documents]
    assert {row.failure_stage for row in report.documents} == {"tokenizer_preflight"}
    assert all("local cache" in str(row.error) for row in report.documents)


@pytest.mark.parametrize(
    "mismatch",
    ["snapshot", "stored_hash", "version_key"],
)
def test_rebuild_config_identity_mismatches_are_hard_gates(
    db: Session,
    documents: list[Document],
    monkeypatch,
    mismatch: str,
) -> None:
    expected = _test_ingestion_config()
    expected_hash = _canonical_config_hash(expected)
    stored = json.loads(json.dumps(expected))
    stored_hash = expected_hash
    if mismatch == "snapshot":
        stored["tokenizer"]["content_sha256"] = "b" * 64
        stored_hash = _canonical_config_hash(stored)
    elif mismatch == "stored_hash":
        stored_hash = "b" * 64
    suffix = "0" * 12 if mismatch == "version_key" else stored_hash[:12]
    document = documents[0]
    version_key = f"canonical-v4-{document.sha256[:12]}-{suffix}"
    document.active_parse_version = version_key
    db.add(
        DocumentParseVersion(
            document_id=document.id,
            version_key=version_key,
            artifact_dir=f"parsed/{document.id}",
            status="active",
            manifest_json={
                "ingestion_config": stored,
                "ingestion_config_sha256": stored_hash,
            },
        )
    )
    db.commit()
    monkeypatch.setattr(
        rebuild_module,
        "canonical_ingestion_config_hash",
        _canonical_config_hash,
    )

    metrics = rebuild_module.collect_integrity_metrics(
        db,
        document_ids={document.id},
        expected_ingestion_config=expected,
        expected_ingestion_config_sha256=expected_hash,
    )

    assert metrics["config_identity_completeness"] == 0.0
    assert "config_identity_completeness" in rebuild_module.STRICT_REBUILD_METRICS


@pytest.mark.parametrize(
    ("metric_name", "checkpoint_value"),
    [
        ("source_fidelity_completeness", None),
        ("source_fidelity_completeness", 0.5),
        ("structured_limit_completeness", None),
        ("structured_limit_completeness", 0.5),
    ],
)
def test_rebuild_fidelity_checkpoint_metrics_are_strict_gates(
    db: Session,
    documents: list[Document],
    metric_name: str,
    checkpoint_value: float | None,
) -> None:
    document = documents[0]
    version_key = _test_version_key(document)
    document.active_parse_version = version_key
    output = {
        "source_fidelity_completeness": 1.0,
        "structured_limit_completeness": 1.0,
    }
    if checkpoint_value is None:
        output.pop(metric_name)
    else:
        output[metric_name] = checkpoint_value
    db.add(
        DocumentParseVersion(
            document_id=document.id,
            version_key=version_key,
            artifact_dir=f"parsed/{document.id}",
            status="active",
            stage_state={
                "semantic_split": {"status": "completed", "output": output}
            },
        )
    )
    db.commit()

    metrics = rebuild_module.collect_integrity_metrics(db, document_ids={document.id})

    assert metrics[metric_name] == 0.0
    assert metric_name in rebuild_module.STRICT_REBUILD_METRICS
    otherwise_complete = {
        name: 1.0 for name in rebuild_module.STRICT_REBUILD_METRICS
    }
    otherwise_complete[metric_name] = metrics[metric_name]
    assert all(
        otherwise_complete[name] == 1.0
        for name in rebuild_module.STRICT_REBUILD_METRICS
    ) is False


def test_rebuild_reports_complete_fidelity_checkpoints(
    db: Session,
    documents: list[Document],
) -> None:
    document = documents[0]
    version_key = _test_version_key(document)
    document.active_parse_version = version_key
    db.add(
        DocumentParseVersion(
            document_id=document.id,
            version_key=version_key,
            artifact_dir=f"parsed/{document.id}",
            status="active",
            stage_state={
                "semantic_split": {
                    "status": "completed",
                    "output": {
                        "source_fidelity_completeness": 1.0,
                        "structured_limit_completeness": 1.0,
                    }
                }
            },
        )
    )
    db.commit()

    metrics = rebuild_module.collect_integrity_metrics(db, document_ids={document.id})

    assert metrics["source_fidelity_completeness"] == 1.0
    assert metrics["structured_limit_completeness"] == 1.0
    assert "source_fidelity_completeness" in rebuild_module.RebuildReport.__dataclass_fields__
    assert "structured_limit_completeness" in rebuild_module.RebuildReport.__dataclass_fields__


@pytest.mark.parametrize("checkpoint_status", [None, "running", "failed"])
def test_rebuild_rejects_stale_fidelity_output_from_incomplete_checkpoint(
    db: Session,
    documents: list[Document],
    checkpoint_status: str | None,
) -> None:
    document = documents[0]
    version_key = _test_version_key(document)
    document.active_parse_version = version_key
    semantic_split = {
        "output": {
            "source_fidelity_completeness": 1.0,
            "structured_limit_completeness": 1.0,
        }
    }
    if checkpoint_status is not None:
        semantic_split["status"] = checkpoint_status
    db.add(
        DocumentParseVersion(
            document_id=document.id,
            version_key=version_key,
            artifact_dir=f"parsed/{document.id}",
            status="active",
            stage_state={"semantic_split": semantic_split},
        )
    )
    db.commit()

    metrics = rebuild_module.collect_integrity_metrics(db, document_ids={document.id})

    assert metrics["source_fidelity_completeness"] == 0.0
    assert metrics["structured_limit_completeness"] == 0.0


def test_exact_ratio_never_rounds_an_incomplete_corpus_to_complete() -> None:
    assert rebuild_module._exact_ratio(1_999_999, 2_000_000) < 1.0


def test_rebuild_blocks_acceptance_when_one_document_fails(
    db: Session, documents: list[Document], monkeypatch
) -> None:
    calls: list[str] = []
    old_versions: dict[str, str] = {}
    for document in documents:
        old_key = f"old-{document.id}"
        old_versions[document.id] = old_key
        document.active_parse_version = old_key
        db.add(
            DocumentParseVersion(
                document_id=document.id,
                version_key=old_key,
                artifact_dir=f"parsed/{document.id}/{old_key}",
                status="active",
            )
        )
    db.commit()

    class FakePipeline:
        def __init__(self, session: Session) -> None:
            self.db = session

        def _get_or_create_parse_version(self, document: Document):
            version = DocumentParseVersion(
                document_id=document.id,
                version_key=f"canonical-v1-{document.sha256[:12]}",
                artifact_dir=f"parsed/{document.id}",
            )
            self.db.add(version)
            self.db.flush()
            return version

        def ingestion_stage_handlers(self):
            return {}

    class FakeRunner:
        def __init__(self, session: Session, **_kwargs) -> None:
            self.db = session

        def run_until_blocked(self, document_id: str, version_key: str, **_kwargs):
            calls.append(document_id)
            if document_id == documents[-1].id:
                raise RuntimeError("parse failed")
            version = self.db.scalar(
                select(DocumentParseVersion).where(
                    DocumentParseVersion.document_id == document_id,
                    DocumentParseVersion.version_key == version_key,
                )
            )
            version.status = "ready_to_activate"
            self.db.commit()
            return version

    monkeypatch.setattr(rebuild_module, "IngestionPipeline", FakePipeline)
    monkeypatch.setattr(rebuild_module, "IngestionStageRunner", FakeRunner)

    report = rebuild_module.rebuild(db)

    assert calls == [document.id for document in documents]
    assert report.ready_for_acceptance is False
    assert report.failed_document_ids == [documents[-1].id]
    assert report.failed_documents == 1
    assert report.documents[-1].failure_stage == "rebuild"
    assert report.parse_version_map == {
        report.documents[0].document_id: report.documents[0].planned_version
    }
    assert {
        document.id: db.get(Document, document.id).active_parse_version
        for document in documents
    } == old_versions


def test_rebuild_releases_loaded_models_after_each_document(
    db: Session, documents: list[Document], monkeypatch
) -> None:
    releases: list[str] = []

    class FakePipeline:
        def __init__(self, session: Session) -> None:
            self.db = session

        def _get_or_create_parse_version(self, document: Document):
            version = DocumentParseVersion(
                document_id=document.id,
                version_key=f"canonical-v4-{document.sha256[:12]}",
                artifact_dir=f"parsed/{document.id}",
            )
            self.db.add(version)
            self.db.flush()
            return version

        def ingestion_stage_handlers(self):
            return {}

    class FakeRunner:
        def __init__(self, session: Session, **_kwargs) -> None:
            self.db = session

        def run_until_blocked(self, document_id: str, version_key: str, **_kwargs):
            if document_id == documents[-1].id:
                raise RuntimeError("repair failed")
            document = self.db.get(Document, document_id)
            document.active_parse_version = version_key
            self.db.commit()

    monkeypatch.setattr(rebuild_module, "IngestionPipeline", FakePipeline)
    monkeypatch.setattr(rebuild_module, "IngestionStageRunner", FakeRunner)
    monkeypatch.setattr(
        rebuild_module,
        "_release_loaded_models",
        lambda: releases.append("released"),
    )

    report = rebuild_module.rebuild(db)

    assert report.failed_documents == 1
    assert releases == ["released", "released"]


def test_resume_uses_completed_stage_checkpoints(
    db: Session, documents: list[Document], monkeypatch
) -> None:
    document = documents[0]
    version_key = _test_version_key(document)
    db.add(
        DocumentParseVersion(
            document_id=document.id,
            version_key=version_key,
            artifact_dir=f"parsed/{document.id}",
            status="quality_checking",
            stage_state={"parse": {"status": "completed", "attempts": 1}},
        )
    )
    db.commit()
    observed: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, session: Session) -> None:
            self.db = session

        def _get_or_create_parse_version(self, selected: Document):
            return self.db.scalar(
                select(DocumentParseVersion).where(
                    DocumentParseVersion.document_id == selected.id,
                    DocumentParseVersion.version_key == version_key,
                )
            )

        def ingestion_stage_handlers(self):
            return {}

    class FakeRunner:
        def __init__(self, _session: Session, **_kwargs) -> None:
            pass

        def run_until_blocked(self, document_id: str, selected_version: str, **kwargs):
            observed.update(
                document_id=document_id,
                version_key=selected_version,
                include_activation=kwargs.get("include_activation"),
            )
            raise RuntimeError("stop after observing resume")

    monkeypatch.setattr(rebuild_module, "IngestionPipeline", FakePipeline)
    monkeypatch.setattr(rebuild_module, "IngestionStageRunner", FakeRunner)

    rebuild_module.rebuild(db, resume=True, document_id=document.id)

    assert observed == {
        "document_id": document.id,
        "version_key": version_key,
        "include_activation": False,
    }


def test_rebuild_wires_pipeline_identity_validation_into_stage_runner(
    db: Session, documents: list[Document], monkeypatch
) -> None:
    document = documents[0]
    validator = object()
    captured: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, _session: Session) -> None:
            self.validate_ingestion_identity = validator

        def _get_or_create_parse_version(self, selected: Document):
            return SimpleNamespace(version_key=_test_version_key(selected))

        def ingestion_stage_handlers(self):
            return {}

    class FakeRunner:
        def __init__(self, _session: Session, **kwargs) -> None:
            captured.update(kwargs)

        def run_until_blocked(self, *_args, **_kwargs):
            raise RuntimeError("stop after runner wiring")

    monkeypatch.setattr(rebuild_module, "IngestionPipeline", FakePipeline)
    monkeypatch.setattr(rebuild_module, "IngestionStageRunner", FakeRunner)

    rebuild_module.rebuild(db, document_id=document.id)

    assert captured["pre_stage_validator"] is validator


def test_rebuild_requires_resume_for_existing_checkpoints(
    db: Session, documents: list[Document]
) -> None:
    document = documents[0]
    version_key = _test_version_key(document)
    db.add(
        DocumentParseVersion(
            document_id=document.id,
            version_key=version_key,
            artifact_dir=f"parsed/{document.id}",
            status="quality_checking",
            stage_state={"parse": {"status": "completed", "attempts": 1}},
        )
    )
    db.commit()

    report = rebuild_module.rebuild(db, document_id=document.id, resume=False)

    assert report.failed_documents == 1
    assert "--resume" in str(report.documents[0].error)


def test_postgres_without_pgvector_cannot_report_complete_index(
    db: Session, documents: list[Document], monkeypatch
) -> None:
    document = documents[0]
    document.active_parse_version = "canonical-v1-test"
    db.add(
        DocumentChunk(
            id="child",
            document_id=document.id,
            parse_version="canonical-v1-test",
            chunk_role="child",
            ordinal=0,
            text="source",
            embedding=[0.1],
        )
    )
    db.commit()

    class UnavailableStore:
        def available(self) -> bool:
            return False

    monkeypatch.setattr(rebuild_module, "get_vector_store", lambda _db: UnavailableStore())
    monkeypatch.setattr(rebuild_module.get_settings(), "ollama_embedding_dimensions", 1)
    monkeypatch.setattr(db.get_bind().dialect, "name", "postgresql")

    metrics = rebuild_module.collect_integrity_metrics(db)

    assert metrics["embedded_children"] == 1
    assert metrics["indexed_children"] == 0
    assert metrics["pgvector_rows"] == 0
    assert metrics["pgvector_completeness"] == 0.0


def test_integrity_metrics_separate_structured_context_and_plain_embeddings(
    db: Session, documents: list[Document], monkeypatch
) -> None:
    document = documents[0]
    document.active_parse_version = "mixed-v"
    db.add_all(
        [
            DocumentChunk(
                id="plain-child",
                document_id=document.id,
                parse_version="mixed-v",
                chunk_role="child",
                block_type="narrative",
                ordinal=0,
                text="Raw narrative",
                embedding_text="Raw narrative",
                source_spans=[{"page_index": 0, "page_label": "1"}],
                embedding=[1.0, 0.0],
            ),
            DocumentChunk(
                id="table-child",
                document_id=document.id,
                parse_version="mixed-v",
                chunk_role="child",
                block_type="table",
                ordinal=1,
                text="| A | B |",
                embedding_text="| A | B |",
                source_spans=[{"page_index": 0, "page_label": "1"}],
                embedding=[0.0, 1.0],
            ),
            DocumentChunk(
                id="figure-child",
                document_id=document.id,
                parse_version="mixed-v",
                chunk_role="child",
                block_type="figure",
                ordinal=2,
                text="Figure 1",
                embedding_text="Figure 1",
                source_spans=[{"page_index": 0, "page_label": "1"}],
                embedding=[0.5, 0.5],
            ),
        ]
    )
    db.commit()
    monkeypatch.setattr(rebuild_module.get_settings(), "ollama_embedding_dimensions", 2)

    metrics = rebuild_module.collect_integrity_metrics(db)

    assert metrics["contextualization_eligible_children"] == 0
    assert metrics["contextualized_children"] == 0
    assert metrics["plain_embedding_children"] == 3
    assert metrics["contextual_prefix_completeness"] == 1.0
    assert metrics["plain_embedding_completeness"] == 1.0


def test_no_structured_children_have_complete_context_policy_metric(
    db: Session, documents: list[Document], monkeypatch
) -> None:
    document = documents[0]
    document.active_parse_version = "plain-v"
    db.add(
        DocumentChunk(
            id="plain-only",
            document_id=document.id,
            parse_version="plain-v",
            chunk_role="child",
            block_type="appendix",
            ordinal=0,
            text="Appendix evidence",
            embedding_text="Appendix evidence",
            source_spans=[{"page_index": 0, "page_label": "1"}],
            embedding=[1.0, 0.0],
        )
    )
    db.commit()
    monkeypatch.setattr(rebuild_module.get_settings(), "ollama_embedding_dimensions", 2)

    metrics = rebuild_module.collect_integrity_metrics(db)

    assert metrics["contextualization_eligible_children"] == 0
    assert metrics["contextual_prefix_completeness"] == 1.0
    assert metrics["plain_embedding_completeness"] == 1.0


def test_cleanup_requires_ready_report_and_both_explicit_guards(
    db: Session, documents: list[Document], tmp_path: Path
) -> None:
    ready = {"ready_for_acceptance": True, "failed_documents": 0}

    with pytest.raises(ValueError, match="both deletion guards"):
        rebuild_module.cleanup_old_data(
            db,
            artifact_root=tmp_path,
            rebuild_report=ready,
            delete_old_after_acceptance=True,
            confirm_delete_old_data=False,
        )
    with pytest.raises(ValueError, match="not ready"):
        rebuild_module.cleanup_old_data(
            db,
            artifact_root=tmp_path,
            rebuild_report={"ready_for_acceptance": False},
            delete_old_after_acceptance=True,
            confirm_delete_old_data=True,
        )


def test_cleanup_removes_inactive_data_and_preserves_active_and_sources(
    db: Session, documents: list[Document], tmp_path: Path
) -> None:
    document = documents[0]
    artifact_root = tmp_path / "parsed"
    mineru_root = tmp_path / "mineru"
    active_dir = artifact_root / document.id / "canonical-v1-active"
    old_dir = artifact_root / document.id / "canonical-v0-old"
    staged_dir = artifact_root / document.id / "canonical-v2-staged"
    active_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    staged_dir.mkdir(parents=True)
    (active_dir / "canonical.md").write_text("active", encoding="utf-8")
    (old_dir / "canonical.md").write_text("old", encoding="utf-8")
    (staged_dir / "canonical.md").write_text("staged", encoding="utf-8")
    (mineru_root / "paper-run").mkdir(parents=True)
    (mineru_root / "paper-run" / "output.md").write_text("temp", encoding="utf-8")
    document.active_parse_version = "canonical-v1-active"
    db.add_all(
        [
            DocumentParseVersion(
                document_id=document.id,
                version_key="canonical-v1-active",
                artifact_dir=str(active_dir),
                status="active",
            ),
            DocumentParseVersion(
                document_id=document.id,
                version_key="canonical-v0-old",
                artifact_dir=str(old_dir),
                status="superseded",
            ),
            DocumentParseVersion(
                document_id=document.id,
                version_key="canonical-v2-staged",
                artifact_dir=str(staged_dir),
                status="ready_to_activate",
            ),
            DocumentChunk(
                id="active-child",
                document_id=document.id,
                parse_version="canonical-v1-active",
                chunk_role="child",
                ordinal=0,
                text="active",
            ),
            DocumentChunk(
                id="legacy-child",
                document_id=document.id,
                parse_version="legacy",
                chunk_role="child",
                ordinal=0,
                text="legacy",
            ),
            DocumentChunk(
                id="old-child",
                document_id=document.id,
                parse_version="canonical-v0-old",
                chunk_role="child",
                ordinal=0,
                text="old",
            ),
            DocumentChunk(
                id="staged-child",
                document_id=document.id,
                parse_version="canonical-v2-staged",
                chunk_role="child",
                ordinal=0,
                text="staged",
            ),
        ]
    )
    db.commit()

    result = rebuild_module.cleanup_old_data(
        db,
        artifact_root=artifact_root,
        mineru_output_dir=mineru_root,
        rebuild_report={
            "ready_for_acceptance": True,
            "failed_documents": 0,
            "parse_version_map": {document.id: "canonical-v1-active"},
            "activation": {
                "status": "completed",
                "activated_document_ids": [document.id],
            },
        },
        delete_old_after_acceptance=True,
        confirm_delete_old_data=True,
    )

    assert result.deleted_chunks == 2
    assert db.get(DocumentChunk, "active-child") is not None
    assert db.get(DocumentChunk, "legacy-child") is None
    assert db.get(DocumentChunk, "old-child") is None
    assert db.get(DocumentChunk, "staged-child") is not None
    assert active_dir.is_dir()
    assert not old_dir.exists()
    assert staged_dir.is_dir()
    assert Path(document.raw_path).is_file()
    assert mineru_root.is_dir()
    assert list(mineru_root.iterdir()) == []


def test_cleanup_rejects_ready_to_activate_staged_batch(
    db: Session,
    documents: list[Document],
    tmp_path: Path,
) -> None:
    document = documents[0]
    document.active_parse_version = "old-v"
    db.add_all(
        [
            DocumentParseVersion(
                document_id=document.id,
                version_key="old-v",
                artifact_dir=str(tmp_path / "old-v"),
                status="active",
            ),
            DocumentParseVersion(
                document_id=document.id,
                version_key="staged-v",
                artifact_dir=str(tmp_path / "staged-v"),
                status="ready_to_activate",
            ),
        ]
    )
    db.commit()

    with pytest.raises(ValueError, match="activated batch"):
        rebuild_module.cleanup_old_data(
            db,
            artifact_root=tmp_path,
            rebuild_report={
                "ready_for_acceptance": True,
                "failed_documents": 0,
                "parse_version_map": {document.id: "staged-v"},
            },
            delete_old_after_acceptance=True,
            confirm_delete_old_data=True,
        )

    assert db.get(Document, document.id).active_parse_version == "old-v"
    assert db.scalar(
        select(DocumentParseVersion).where(
            DocumentParseVersion.document_id == document.id,
            DocumentParseVersion.version_key == "staged-v",
        )
    ) is not None


def test_cleanup_scopes_legacy_deletion_to_the_activated_batch(
    db: Session,
    documents: list[Document],
    tmp_path: Path,
) -> None:
    activated, untouched = documents
    activated.active_parse_version = "canonical-active"
    untouched.active_parse_version = "legacy"
    db.add_all(
        [
            DocumentParseVersion(
                document_id=activated.id,
                version_key="canonical-active",
                artifact_dir=str(tmp_path / activated.id / "canonical-active"),
                status="active",
            ),
            DocumentChunk(
                id="activated-legacy-child",
                document_id=activated.id,
                parse_version="legacy",
                chunk_role="child",
                ordinal=0,
                text="old activated legacy",
            ),
            DocumentChunk(
                id="untouched-legacy-child",
                document_id=untouched.id,
                parse_version="legacy",
                chunk_role="child",
                ordinal=0,
                text="still active legacy",
            ),
        ]
    )
    db.commit()

    rebuild_module.cleanup_old_data(
        db,
        artifact_root=tmp_path,
        rebuild_report={
            "ready_for_acceptance": True,
            "failed_documents": 0,
            "parse_version_map": {activated.id: "canonical-active"},
            "activation": {
                "status": "completed",
                "activated_document_ids": [activated.id],
            },
        },
        delete_old_after_acceptance=True,
        confirm_delete_old_data=True,
    )

    assert db.get(DocumentChunk, "activated-legacy-child") is None
    assert db.get(DocumentChunk, "untouched-legacy-child") is not None


def test_main_returns_nonzero_for_strict_rebuild_failure(
    tmp_path: Path, monkeypatch
) -> None:
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        rebuild_module,
        "run_cli_rebuild",
        lambda **_kwargs: SimpleNamespace(
            ready_for_acceptance=False,
            to_dict=lambda: {
                "ready_for_acceptance": False,
                "failed_documents": 1,
            },
        ),
    )

    exit_code = rebuild_module.main(["--report", str(report_path)])

    assert exit_code == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["failed_documents"] == 1
