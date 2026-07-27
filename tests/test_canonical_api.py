from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes
from app.api.routes import router
from app.db.session import Base, get_db
from app.models.records import (
    Document,
    DocumentChunk,
    DocumentParseVersion,
    PipelineRun,
    Project,
)
from app.services.canonical_artifacts import CanonicalArtifactStore
from app.services.canonical_models import CanonicalBlock, CanonicalDocument, SourceSpan


def _session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def _client(db: Session) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[get_db] = override_db
    return TestClient(app, raise_server_exceptions=False)


def _canonical_document(
    document_id: str,
    version: str,
    *,
    media_type: str = "application/pdf",
) -> CanonicalDocument:
    return CanonicalDocument(
        document_id=document_id,
        parse_version=version,
        source_path="source/paper.pdf",
        source_media_type=media_type,
        parser_source="mineru",
        parser_metadata={"backend_version": "2.1"},
        title="Paper",
        blocks=[
            CanonicalBlock(
                block_id="source-1",
                block_type="narrative",
                text="Source content.",
                reading_order=0,
                parser_source="mineru",
                source_spans=[SourceSpan(page_index=2, page_label="3")],
            )
        ],
        warnings=["figure analysis unavailable"],
        status="ready",
    )


def _directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"directory links are unavailable: {exc}")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"directory links are unavailable: {result.stderr}")


def _seed_active_parse(
    db: Session,
    root: Path,
    *,
    document_id: str = "doc-1",
    version: str = "canonical-v1",
    file_name: str = "paper.pdf",
    media_type: str = "application/pdf",
) -> tuple[Document, DocumentParseVersion, Path]:
    project = db.get(Project, "project-1")
    if project is None:
        project = Project(id="project-1", slug="demo", name="Demo")
        db.add(project)
    source = root.parent / file_name
    source.write_bytes(b"source")
    document = Document(
        id=document_id,
        project_id=project.id,
        title="Paper",
        file_name=file_name,
        sha256="a" * 64,
        raw_path=str(source),
        status="ready",
        active_parse_version=version,
    )
    db.add(document)
    store = CanonicalArtifactStore(root)
    store.write_staging(
        document_id,
        version,
        _canonical_document(document_id, version, media_type=media_type),
    )
    bundle = store.promote(document_id, version)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    parse_version = DocumentParseVersion(
        id=f"parse-{document_id}",
        document_id=document_id,
        version_key=version,
        status="active",
        artifact_dir=str(bundle),
        parser_name="mineru",
        parser_version="2.1",
        manifest_json={
            "input_fingerprint": manifest["input_fingerprint"],
            "canonical_markdown_sha256": manifest["canonical_markdown_sha256"],
        },
        quality_json={"status": "accepted_with_warnings", "accepted": True, "score": 0.97},
        stage_state={
            "repair": {
                "status": "completed",
                "output": {"repair_requests": [{"page_index": 2}]},
            }
        },
    )
    db.add(parse_version)
    db.add(
        PipelineRun(
            id=f"run-{document_id}",
            document_id=document_id,
            project_id=project.id,
            provider_report={"progress": {"stage": "completed", "percent": 100}},
        )
    )
    db.commit()
    return document, parse_version, bundle


def test_current_parse_status_markdown_and_download_expose_only_active_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, active, bundle = _seed_active_parse(db, root)
    old = DocumentParseVersion(
        id="parse-old",
        document_id=document.id,
        version_key="canonical-old",
        status="superseded",
        artifact_dir=str(tmp_path / "outside"),
        parser_name="legacy-secret",
        manifest_json={"contextual_prefix": "must-not-leak"},
    )
    db.add(old)
    db.commit()
    client = _client(db)

    response = client.get(f"/api/documents/{document.id}/parse")
    assert response.status_code == 200
    assert response.json() == {
        "document_id": document.id,
        "version": active.version_key,
        "parser": "mineru",
        "parser_version": "2.1",
        "progress": {"stage": "completed", "percent": 100},
        "quality": {"status": "accepted_with_warnings", "accepted": True, "score": 0.97},
        "repair_pages": [2],
        "warning_count": 1,
        "download_available": True,
    }
    assert "canonical-old" not in response.text
    assert "contextual_prefix" not in response.text

    markdown = client.get(f"/api/documents/{document.id}/parse/markdown")
    assert markdown.status_code == 200
    assert markdown.json()["version"] == active.version_key
    assert "# Paper" in markdown.json()["markdown"]
    assert markdown.headers["content-type"].startswith("application/json")

    download = client.get(f"/api/documents/{document.id}/parse/download")
    assert download.status_code == 200
    assert download.content == (bundle / "canonical.md").read_bytes()
    assert download.headers["content-disposition"].startswith("attachment;")
    assert "canonical.md" in download.headers["content-disposition"]


def test_parse_status_does_not_leak_newer_inactive_run_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, _, _ = _seed_active_parse(db, root)
    active_run = db.get(PipelineRun, f"run-{document.id}")
    active_run.created_at = datetime.utcnow()
    db.add(
        PipelineRun(
            id="run-inactive-new",
            document_id=document.id,
            project_id=document.project_id,
            created_at=active_run.created_at + timedelta(seconds=1),
            provider_report={
                "parse_version": "canonical-v2-inactive",
                "progress": {"stage": "contextualize", "percent": 63},
            },
        )
    )
    db.commit()

    response = _client(db).get(f"/api/documents/{document.id}/parse")

    assert response.status_code == 200
    assert response.json()["progress"] == {"stage": "completed", "percent": 100}


@pytest.mark.parametrize(
    ("file_name", "media_type", "span", "expected_fragment"),
    [
        ("paper.pdf", "application/pdf", {"page_index": 2, "page_label": "3"}, "#page=3"),
        ("paper.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", {"paragraph_id": "p-7"}, "#paragraph=p-7"),
        ("paper.html", "text/html", {"element_id": "results"}, "#element=results"),
        ("paper.txt", "text/plain", {"line_start": 10, "line_end": 12}, "#line=10"),
    ],
)
def test_location_returns_active_source_spans_and_format_specific_source_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
    media_type: str,
    span: dict,
    expected_fragment: str,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    monkeypatch.setattr(routes.settings, "raw_dir", tmp_path)
    db = _session()
    document, active, _ = _seed_active_parse(
        db,
        root,
        file_name=file_name,
        media_type=media_type,
    )
    chunk = DocumentChunk(
        id="chunk-active",
        document_id=document.id,
        parse_version=active.version_key,
        ordinal=1,
        text="Evidence",
        source_spans=[span],
    )
    db.add(chunk)
    db.commit()

    response = _client(db).get(
        f"/api/documents/{document.id}/citations/{chunk.id}/location"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_spans"] == [span]
    assert payload["source_url"].startswith(
        f"/api/documents/{quote(document.id)}/file?project_slug=demo"
    )
    assert payload["source_url"].endswith(expected_fragment)
    assert payload["parse_version"] == active.version_key
    assert "contextual_prefix" not in response.text
    assert "embedding" not in response.text


def test_location_rejects_old_other_document_and_reference_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, active, _ = _seed_active_parse(db, root)
    other, other_active, _ = _seed_active_parse(
        db,
        root,
        document_id="doc-2",
        version="canonical-v2",
    )
    db.add_all(
        [
            DocumentChunk(id="old", document_id=document.id, parse_version="canonical-old", ordinal=1, text="Old"),
            DocumentChunk(id="other", document_id=other.id, parse_version=other_active.version_key, ordinal=1, text="Other"),
            DocumentChunk(id="reference", document_id=document.id, parse_version=active.version_key, ordinal=2, text="Reference", block_type="reference"),
        ]
    )
    db.commit()
    client = _client(db)

    for chunk_id in ("old", "other", "reference", "missing"):
        assert client.get(
            f"/api/documents/{document.id}/citations/{chunk_id}/location"
        ).status_code == 404


def test_location_redacts_derived_text_and_server_paths_from_span_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    monkeypatch.setattr(routes.settings, "raw_dir", tmp_path)
    db = _session()
    document, active, _ = _seed_active_parse(db, root)
    db.add(
        DocumentChunk(
            id="chunk-sensitive",
            document_id=document.id,
            parse_version=active.version_key,
            ordinal=1,
            text="Evidence",
            contextual_prefix="private prefix",
            embedding_text="private prefix\n\nEvidence",
            source_spans=[
                {
                    "page_index": 2,
                    "metadata": {
                        "table_id": "table-1",
                        "contextual_prefix": "private prefix",
                        "embedding": [0.1, 0.2],
                        "artifact_path": str(root / document.id / active.version_key),
                    },
                }
            ],
        )
    )
    db.commit()

    response = _client(db).get(
        f"/api/documents/{document.id}/citations/chunk-sensitive/location"
    )

    assert response.status_code == 200
    assert response.json()["source_spans"][0]["metadata"] == {"table_id": "table-1"}
    assert "private prefix" not in response.text
    assert str(root) not in response.text


def test_parse_artifacts_reject_artifact_directory_outside_store_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, active, _ = _seed_active_parse(db, root)
    active.artifact_dir = str(tmp_path / "outside" / "canonical-v1")
    db.commit()
    client = _client(db)

    status = client.get(f"/api/documents/{document.id}/parse")
    assert status.status_code == 200
    assert status.json()["download_available"] is False
    assert client.get(f"/api/documents/{document.id}/parse/markdown").status_code == 404
    assert client.get(f"/api/documents/{document.id}/parse/download").status_code == 404


def test_parse_artifacts_reject_checkpoint_fingerprint_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, active, _ = _seed_active_parse(db, root)
    active.manifest_json = {"input_fingerprint": "0" * 64}
    db.commit()

    response = _client(db).get(f"/api/documents/{document.id}/parse/markdown")

    assert response.status_code == 404


def test_parse_artifacts_reject_linked_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, active, bundle = _seed_active_parse(db, root)
    outside = tmp_path / "outside-bundle"
    bundle.rename(outside)
    _directory_link(bundle, outside)
    active.artifact_dir = str(bundle)
    db.commit()

    response = _client(db).get(f"/api/documents/{document.id}/parse/markdown")

    assert response.status_code == 404
