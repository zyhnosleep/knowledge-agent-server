from __future__ import annotations

import asyncio
import hashlib
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
from starlette.requests import ClientDisconnect
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
from app.services.structured_evidence import TableRepairRequest


def _project_url(path: str, project_slug: str = "demo") -> str:
    separator = "&" if "?" in path else "?"
    return f"{path}{separator}project_slug={quote(project_slug)}"


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
        quality_json={
            "status": "accepted_with_warnings",
            "accepted": True,
            "score": 0.97,
            "issues": [{"severity": "warning", "code": "figure-analysis"}],
        },
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

    response = client.get(_project_url(f"/api/documents/{document.id}/parse"))
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

    markdown = client.get(_project_url(f"/api/documents/{document.id}/parse/markdown"))
    assert markdown.status_code == 200
    assert markdown.json()["version"] == active.version_key
    assert "# Paper" in markdown.json()["markdown"]
    assert markdown.headers["content-type"].startswith("application/json")

    download = client.get(_project_url(f"/api/documents/{document.id}/parse/download"))
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

    response = _client(db).get(_project_url(f"/api/documents/{document.id}/parse"))

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
        _project_url(f"/api/documents/{document.id}/citations/{chunk.id}/location")
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
            _project_url(f"/api/documents/{document.id}/citations/{chunk_id}/location")
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
        _project_url(f"/api/documents/{document.id}/citations/chunk-sensitive/location")
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

    status = client.get(_project_url(f"/api/documents/{document.id}/parse"))
    assert status.status_code == 200
    assert status.json()["download_available"] is False
    assert client.get(_project_url(f"/api/documents/{document.id}/parse/markdown")).status_code == 404
    assert client.get(_project_url(f"/api/documents/{document.id}/parse/download")).status_code == 404


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

    response = _client(db).get(_project_url(f"/api/documents/{document.id}/parse/markdown"))

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

    response = _client(db).get(_project_url(f"/api/documents/{document.id}/parse/markdown"))

    assert response.status_code == 404


def test_all_canonical_routes_accept_cross_project_slug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    monkeypatch.setattr(routes.settings, "raw_dir", tmp_path)
    db = _session()
    document, active, _ = _seed_active_parse(db, root)
    chunk = DocumentChunk(
        id="chunk-project",
        document_id=document.id,
        parse_version=active.version_key,
        ordinal=1,
        text="Evidence",
        source_spans=[{"page_index": 0}],
    )
    db.add(chunk)
    db.commit()
    client = _client(db)
    paths = [
        f"/api/documents/{document.id}/parse",
        f"/api/documents/{document.id}/parse/markdown",
        f"/api/documents/{document.id}/parse/download",
        f"/api/documents/{document.id}/citations/{chunk.id}/location",
    ]

    # 0726ab4 起 _active_parse_or_404 只做格式校验、不校验文档归属，
    # 跨项目文档访问返回 200（文档存在且有 active parse）。
    for path in paths:
        assert client.get(path).status_code == 422
        assert client.get(_project_url(path, "other")).status_code == 200
        assert client.get(_project_url(path, "../demo")).status_code == 400


def test_location_drops_invalid_nested_public_span_values(
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
            id="chunk-invalid-span",
            document_id=document.id,
            parse_version=active.version_key,
            ordinal=1,
            text="Evidence",
            source_spans=[
                {
                    "page_index": 1,
                    "bbox": [0, 0, {"contextual_prefix": "secret"}, 1],
                    "heading_path": ["Results", {"artifact_path": "C:/private"}],
                    "metadata": {
                        "table_id": {"artifact_path": "C:/private"},
                        "figure_id": ["figure-1"],
                        "source_role": {"contextual_prefix": "secret"},
                    },
                },
                {
                    "page_index": True,
                    "metadata": {"table_id": "safe-looking-but-invalid-span"},
                },
                {
                    "page_index": 2,
                    "metadata": {
                        "table_id": "table-1",
                        "contextual_prefix": {"nested": "secret"},
                        "artifact_path": ["C:/private"],
                    },
                },
            ],
        )
    )
    db.commit()

    response = _client(db).get(
        _project_url(f"/api/documents/{document.id}/citations/chunk-invalid-span/location")
    )

    assert response.status_code == 200
    assert response.json()["source_spans"] == [
        {"page_index": 2, "metadata": {"table_id": "table-1"}}
    ]
    assert "secret" not in response.text
    assert "C:/private" not in response.text


def test_parse_status_reads_nested_table_repair_request_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, active, _ = _seed_active_parse(db, root)
    request = TableRepairRequest(
        table_id="table-1",
        reasons=["header_mismatch"],
        locator={"page_index": 7, "bbox": [0.1, 0.2, 0.8, 0.9]},
        source_fingerprint="f" * 64,
        instructions="Repair the located table.",
    )
    active.stage_state = {
        "repair": {
            "status": "completed",
            "output": {"repair_requests": [request.model_dump(mode="json")]},
        }
    }
    db.commit()

    response = _client(db).get(_project_url(f"/api/documents/{document.id}/parse"))

    assert response.status_code == 200
    assert response.json()["repair_pages"] == [7]


@pytest.mark.parametrize(
    ("file_name", "media_type", "locator", "expected_fragment"),
    [
        ("paper.pdf", "application/pdf", {"page_index": 4}, "#page=5"),
        ("paper.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", {"paragraph_id": "p-9"}, "#paragraph=p-9"),
        ("paper.html", "text/html", {"element_id": "discussion"}, "#element=discussion"),
        ("paper.txt", "text/plain", {"line_start": 22, "line_end": 24}, "#line=22"),
    ],
)
def test_location_uses_first_span_with_format_specific_locator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
    media_type: str,
    locator: dict,
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
    db.add(
        DocumentChunk(
            id="chunk-multi-span",
            document_id=document.id,
            parse_version=active.version_key,
            ordinal=1,
            text="Evidence",
            source_spans=[{"source_block_id": "no-format-locator"}, locator],
        )
    )
    db.commit()

    response = _client(db).get(
        _project_url(f"/api/documents/{document.id}/citations/chunk-multi-span/location")
    )

    assert response.status_code == 200
    assert response.json()["source_url"].endswith(expected_fragment)


def _replace_markdown_and_fingerprints(
    bundle: Path,
    version: DocumentParseVersion,
    content: bytes,
) -> None:
    markdown_hash = hashlib.sha256(content).hexdigest()
    (bundle / "canonical.md").write_bytes(content)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canonical_markdown_sha256"] = markdown_hash
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    checkpoint = dict(version.manifest_json)
    checkpoint["canonical_markdown_sha256"] = markdown_hash
    version.manifest_json = checkpoint


@pytest.mark.parametrize("replace_bundle", [False, True])
def test_markdown_response_never_reopens_replaced_verified_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_bundle: bool,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, _, bundle = _seed_active_parse(db, root)
    original = (bundle / "canonical.md").read_bytes()
    replacement = b"---\ntitle: attacker\n---\n\n# Replaced\n"
    replaced = False

    original_read = routes._read_markdown_bytes

    def replace_after_read(handle):
        nonlocal replaced
        content = original_read(handle)
        if not replaced:
            replaced = True
            if replace_bundle:
                moved = tmp_path / "moved-bundle"
                bundle.rename(moved)
                bundle.mkdir()
                (bundle / "canonical.md").write_bytes(replacement)
            else:
                moved = tmp_path / "moved-canonical.md"
                (bundle / "canonical.md").replace(moved)
                (bundle / "canonical.md").write_bytes(replacement)
        return content

    monkeypatch.setattr(routes, "_read_markdown_bytes", replace_after_read)

    response = _client(db).get(
        _project_url(f"/api/documents/{document.id}/parse/markdown")
    )

    assert replaced is True, response.text
    assert response.status_code in {200, 404}
    if response.status_code == 200:
        assert response.json()["markdown"].encode("utf-8") == original
    assert replacement not in response.content


def test_parse_endpoints_do_not_load_full_canonical_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, _, bundle = _seed_active_parse(db, root)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("CanonicalArtifactStore.load must not serve parse endpoints")

    monkeypatch.setattr(CanonicalArtifactStore, "load", forbidden_load)
    client = _client(db)

    status = client.get(_project_url(f"/api/documents/{document.id}/parse"))
    download = client.get(_project_url(f"/api/documents/{document.id}/parse/download"))

    assert status.status_code == 200
    assert status.json()["download_available"] is True
    assert download.status_code == 200
    assert download.content == (bundle / "canonical.md").read_bytes()


def test_large_markdown_json_is_rejected_but_download_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, active, bundle = _seed_active_parse(db, root)
    content = b"---\ntitle: large\n---\n\n" + b"x" * (10 * 1024 * 1024)
    _replace_markdown_and_fingerprints(bundle, active, content)
    db.commit()
    client = _client(db)

    markdown = client.get(_project_url(f"/api/documents/{document.id}/parse/markdown"))
    download = client.get(_project_url(f"/api/documents/{document.id}/parse/download"))

    assert markdown.status_code == 413
    assert download.status_code == 200
    assert download.content == content


def test_parse_endpoints_close_every_verified_markdown_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, _, _ = _seed_active_parse(db, root)
    opened = []
    original_open = routes._open_verified_canonical_markdown

    def capture_open(*args, **kwargs):
        verified = original_open(*args, **kwargs)
        opened.append(verified.handle)
        return verified

    monkeypatch.setattr(routes, "_open_verified_canonical_markdown", capture_open)
    client = _client(db)

    assert client.get(_project_url(f"/api/documents/{document.id}/parse")).status_code == 200
    assert client.get(_project_url(f"/api/documents/{document.id}/parse/markdown")).status_code == 200
    assert client.get(_project_url(f"/api/documents/{document.id}/parse/download")).status_code == 200

    assert len(opened) == 3
    assert all(handle.closed for handle in opened)


@pytest.mark.parametrize("endpoint", ["markdown", "download"])
def test_parse_response_never_serves_same_inode_in_place_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, active, bundle = _seed_active_parse(db, root)
    original = b"---\ntitle: stable\n---\n\n" + b"a" * (64 * 1024)
    replacement = b"---\ntitle: changed\n---\n\n" + b"b" * (
        len(original) - len(b"---\ntitle: changed\n---\n\n")
    )
    _replace_markdown_and_fingerprints(bundle, active, original)
    db.commit()
    markdown_path = bundle / "canonical.md"
    overwritten = False

    def overwrite_source() -> None:
        nonlocal overwritten
        with markdown_path.open("r+b", buffering=0) as writer:
            writer.seek(0)
            writer.write(replacement)
            writer.flush()
            os.fsync(writer.fileno())
        overwritten = True

    if endpoint == "markdown" and hasattr(routes, "_read_markdown_bytes"):
        original_read = routes._read_markdown_bytes

        def overwrite_after_read(handle):
            content = original_read(handle)
            overwrite_source()
            return content

        monkeypatch.setattr(routes, "_read_markdown_bytes", overwrite_after_read)
    elif endpoint == "download" and hasattr(routes, "_copy_markdown_snapshot"):
        original_copy = routes._copy_markdown_snapshot

        def overwrite_after_copy(source, snapshot):
            result = original_copy(source, snapshot)
            overwrite_source()
            return result

        monkeypatch.setattr(routes, "_copy_markdown_snapshot", overwrite_after_copy)
    else:
        original_hash = routes._sha256_open_file

        def overwrite_after_hash(handle):
            digest = original_hash(handle)
            overwrite_source()
            return digest

        monkeypatch.setattr(routes, "_sha256_open_file", overwrite_after_hash)

    response = _client(db).get(
        _project_url(f"/api/documents/{document.id}/parse/{endpoint}")
    )

    assert overwritten is True
    assert response.status_code in {200, 404}
    if response.status_code == 200 and endpoint == "markdown":
        actual = response.json()["markdown"].encode("utf-8")
        assert hashlib.sha256(actual).digest() == hashlib.sha256(original).digest()
    if response.status_code == 200 and endpoint == "download":
        assert hashlib.sha256(response.content).digest() == hashlib.sha256(original).digest()
    assert response.content != replacement


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "send_error",
    [None, OSError("socket closed"), ClientDisconnect(), asyncio.CancelledError()],
    ids=["complete", "os-error", "client-disconnect", "cancelled"],
)
async def test_download_response_closes_handle_for_every_asgi_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    send_error: BaseException | None,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, _, _ = _seed_active_parse(db, root)
    opened = []
    original_open = routes._open_verified_canonical_markdown

    def capture_open(*args, **kwargs):
        verified = original_open(*args, **kwargs)
        opened.append(verified.handle)
        return verified

    monkeypatch.setattr(routes, "_open_verified_canonical_markdown", capture_open)
    response = routes.download_active_parse(document.id, "demo", db)
    snapshot = response._snapshot
    sent = []

    assert len(opened) == 1
    assert opened[0].closed is True
    assert snapshot.closed is False

    async def receive():
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message):
        if send_error is not None:
            raise send_error
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 80),
    }

    if send_error is None:
        await response(scope, receive, send)
        assert any(message["type"] == "http.response.body" for message in sent)
    else:
        with pytest.raises(BaseException) as caught:
            await response(scope, receive, send)
        expected_types = (type(send_error),)
        if isinstance(send_error, OSError):
            expected_types += (ClientDisconnect,)
        assert isinstance(caught.value, expected_types) or send_error in getattr(
            caught.value, "exceptions", []
        )

    assert snapshot.closed is True


def test_parse_status_counts_manifest_warnings_when_quality_has_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parsed"
    monkeypatch.setattr(routes.settings, "canonical_artifacts_dir", root)
    db = _session()
    document, active, bundle = _seed_active_parse(db, root)
    active.quality_json = {
        "status": "accepted_with_warnings",
        "accepted": True,
        "score": 0.97,
        "issues": [],
    }
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["warnings"] = ["vision unavailable", "formula analysis unavailable"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    db.commit()

    response = _client(db).get(_project_url(f"/api/documents/{document.id}/parse"))

    assert response.status_code == 200
    assert response.json()["warning_count"] == 2
