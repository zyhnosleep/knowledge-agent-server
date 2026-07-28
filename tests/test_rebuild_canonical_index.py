from __future__ import annotations

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


@pytest.fixture
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as session:
        yield session


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
    assert all(row.planned_version.startswith("canonical-v1-") for row in report.documents)
    assert list(db.scalars(select(DocumentParseVersion))) == []
    assert all(document.active_parse_version is None for document in documents)
    assert not (tmp_path / "parsed").exists()


def test_rebuild_blocks_acceptance_when_one_document_fails(
    db: Session, documents: list[Document], monkeypatch
) -> None:
    calls: list[str] = []

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
            document = self.db.get(Document, document_id)
            document.active_parse_version = version_key
            version = self.db.scalar(
                select(DocumentParseVersion).where(
                    DocumentParseVersion.document_id == document_id,
                    DocumentParseVersion.version_key == version_key,
                )
            )
            version.status = "active"
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


def test_resume_uses_completed_stage_checkpoints(
    db: Session, documents: list[Document], monkeypatch
) -> None:
    document = documents[0]
    version_key = f"canonical-v1-{document.sha256[:12]}"
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
        "include_activation": True,
    }


def test_rebuild_requires_resume_for_existing_checkpoints(
    db: Session, documents: list[Document]
) -> None:
    document = documents[0]
    version_key = f"canonical-v1-{document.sha256[:12]}"
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
    active_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    (active_dir / "canonical.md").write_text("active", encoding="utf-8")
    (old_dir / "canonical.md").write_text("old", encoding="utf-8")
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
        ]
    )
    db.commit()

    result = rebuild_module.cleanup_old_data(
        db,
        artifact_root=artifact_root,
        mineru_output_dir=mineru_root,
        rebuild_report={"ready_for_acceptance": True, "failed_documents": 0},
        delete_old_after_acceptance=True,
        confirm_delete_old_data=True,
    )

    assert result.deleted_chunks == 2
    assert db.get(DocumentChunk, "active-child") is not None
    assert db.get(DocumentChunk, "legacy-child") is None
    assert db.get(DocumentChunk, "old-child") is None
    assert active_dir.is_dir()
    assert not old_dir.exists()
    assert Path(document.raw_path).is_file()
    assert mineru_root.is_dir()
    assert list(mineru_root.iterdir()) == []


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
