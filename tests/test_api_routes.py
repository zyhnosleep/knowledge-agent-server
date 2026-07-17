from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes
from app.api.routes import router
from app.db.session import Base, get_db
from app.models.records import (
    AgentTraceRun,
    AgentTraceStep,
    Claim,
    ConversationSession,
    ConversationTurn,
    Document,
    DocumentChunk,
    PipelineRun,
    Project,
    QuestionAnswer,
    ReviewItem,
    SessionAttachment,
    SessionAttachmentChunk,
)
from app.services.filesystem import InvalidStoragePathError, UploadTooLargeError
from app.services.parser import ParsedDocument
from app.services.pipeline import IngestionPipeline


def make_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def make_client(db: Session, *, raise_server_exceptions: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[get_db] = override_db
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def test_health_distinguishes_api_and_model_readiness(monkeypatch) -> None:
    class FakeReadiness:
        def check(self):
            return {
                "status": "degraded",
                "models": {
                    "fast": {"status": "ready", "model": "qwen3:14b", "context_length": 16384},
                    "deep": {"status": "not_loaded", "model": "qwen3.6:27b", "context_length": 32768},
                    "embedding": {"status": "ready", "model": "qwen3-embedding:8b", "dimensions": 4096},
                },
            }

    class FakeRuntime:
        def snapshot(self):
            return {
                "fast": {"capacity": 1, "active": 0, "queued": 0},
                "deep": {"capacity": 1, "active": 1, "queued": 2},
            }

    monkeypatch.setattr(routes, "get_model_readiness", lambda: FakeReadiness())
    monkeypatch.setattr(routes, "get_model_runtime", lambda: FakeRuntime())

    payload = make_client(make_session()).get("/api/health").json()

    assert payload["api_status"] == "ok"
    assert payload["status"] == "degraded"
    assert payload["models"]["deep"]["status"] == "not_loaded"
    assert payload["queues"]["deep"]["queued"] == 2


def test_list_endpoints_apply_limit_and_offset() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add_all([Project(id="p0", slug="demo-0", name="Demo 0"), project, Project(id="p2", slug="demo-2", name="Demo 2")])
    for index in range(3):
        document = Document(
            id=f"d{index}",
            project_id=project.id,
            title=f"Doc {index}",
            file_name=f"doc-{index}.md",
            sha256=f"sha{index}",
            raw_path=f"raw/doc-{index}.md",
        )
        db.add(document)
        db.add(PipelineRun(id=f"r{index}", project_id=project.id, document_id=document.id, provider_report={}))
        db.add(ReviewItem(id=f"v{index}", project_id=project.id, document_id=document.id, title=f"Review {index}", detail="detail", payload={}))
    db.commit()
    client = make_client(db)

    assert len(client.get("/api/projects?limit=1&offset=1").json()) == 1
    assert len(client.get("/api/documents?limit=1&offset=1").json()) == 1
    assert len(client.get("/api/runs?limit=1&offset=1").json()) == 1
    assert len(client.get("/api/reviews?limit=1&offset=1").json()) == 1
    assert client.get("/api/documents?limit=201").status_code == 422
    assert client.get("/api/reviews?offset=-1").status_code == 422



def test_project_slug_validation_returns_client_error() -> None:
    db = make_session()
    client = make_client(db, raise_server_exceptions=False)

    response = client.post("/api/projects", json={"slug": "../evil", "name": "Bad"})
    assert response.status_code == 400

    response = client.get("/api/documents?project_slug=../evil")
    assert response.status_code == 400


def test_ingest_upload_maps_storage_errors(monkeypatch) -> None:
    db = make_session()
    client = make_client(db, raise_server_exceptions=False)

    async def too_large(project_slug: str, upload) -> None:
        raise UploadTooLargeError("Upload exceeds MAX_UPLOAD_BYTES.")

    monkeypatch.setattr(routes, "save_upload", too_large)
    response = client.post("/api/ingest/upload", files={"file": ("paper.pdf", b"123456", "application/pdf")})
    assert response.status_code == 413

    async def invalid_path(project_slug: str, upload) -> None:
        raise InvalidStoragePathError("Invalid project slug or file name.")

    monkeypatch.setattr(routes, "save_upload", invalid_path)
    response = client.post("/api/ingest/upload", files={"file": ("paper.pdf", b"content", "application/pdf")})
    assert response.status_code == 400


def test_pipeline_dashboard_returns_topic_cards_and_business_runs() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="CHARMM 力场研究")
    ready_doc = Document(
        id="d1",
        project_id=project.id,
        title="CHARMM36m Force Field Review",
        file_name="CHARMM36m_Force_Field_Review.pdf",
        sha256="sha1",
        raw_path="raw/charmm.pdf",
        status="ready",
    )
    processing_doc = Document(
        id="d2",
        project_id=project.id,
        title="protein_ligand_binding_parameters.md",
        file_name="protein_ligand_binding_parameters.md",
        sha256="sha2",
        raw_path="raw/protein.md",
        status="processing",
    )
    db.add_all(
        [
            project,
            ready_doc,
            processing_doc,
            PipelineRun(
                id="r1",
                project_id=project.id,
                document_id=ready_doc.id,
                status="completed",
                provider_report={"progress": {"percent": 100, "stage": "completed", "message": "Done."}},
            ),
            PipelineRun(
                id="r2",
                project_id=project.id,
                document_id=processing_doc.id,
                status="running",
                provider_report={"progress": {"percent": 45, "stage": "chunking", "message": "正在解析切片"}},
            ),
        ]
    )
    db.commit()
    client = make_client(db)

    response = client.get("/api/pipeline/dashboard?project_slug=demo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service_status"] == "ok"
    assert payload["topics"][0]["title"] == "CHARMM 力场研究"
    assert payload["topics"][0]["document_count"] == 2
    assert payload["topics"][0]["completed_count"] == 1
    assert payload["topics"][0]["processing_count"] == 1
    assert payload["topics"][0]["progress_percent"] == 72
    assert payload["runs"][0]["document_title"] == "protein_ligand_binding_parameters.md"
    assert payload["runs"][0]["status"] == "running"
    assert payload["runs"][0]["progress"]["stage"] == "chunking"
    assert payload["runs"][1]["document_title"] == "CHARMM36m Force Field Review"


def test_document_source_route_returns_sorted_chunks_for_source_drawer() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id=project.id,
        title="CHARMM36m Force Field Review",
        file_name="CHARMM36m_Force_Field_Review.pdf",
        sha256="sha1",
        raw_path="raw/charmm.pdf",
        raw_text="Full source text fallback.",
        status="ready",
    )
    db.add_all(
        [
            project,
            document,
            DocumentChunk(id="c2", document_id=document.id, ordinal=2, page_label="3", text="Third chunk."),
            DocumentChunk(id="c1", document_id=document.id, ordinal=1, page_label="2", text="First visible chunk."),
        ]
    )
    db.commit()
    client = make_client(db)

    response = client.get("/api/documents/d1/source")

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_title"] == "CHARMM36m Force Field Review"
    assert payload["chunks"][0]["chunk_id"] == "c1"
    assert payload["chunks"][0]["label"] == "RAG Chunk #01"
    assert payload["chunks"][0]["page_label"] == "2"
    assert payload["chunks"][0]["text"] == "First visible chunk."


def test_ingest_upload_creates_project_and_returns_identity(monkeypatch) -> None:
    db = make_session()
    client = make_client(db, raise_server_exceptions=False)

    uploaded_path = Path("/tmp/test-project/a1b2c3d4-paper.pdf")

    async def fake_save_upload(project_slug: str, upload) -> Path:
        assert project_slug == "my-topic"
        return uploaded_path

    class FakePipeline:
        def __init__(self, db_session: Session) -> None:
            self.db = db_session

        def register_document(self, project_slug: str, project_name: str, file_path: Path):
            assert project_slug == "my-topic"
            assert project_name == "My Topic"
            assert file_path == uploaded_path
            project = Project(id="p-new", slug="my-topic", name="My Topic")
            document = Document(
                id="d-new",
                project_id=project.id,
                title="The Paper",
                file_name="paper.pdf",
                sha256="sha",
                raw_path=str(file_path),
                status="pending",
            )
            run = PipelineRun(
                id="r-new",
                project_id=project.id,
                document_id=document.id,
                status="completed",
            )
            return project, document, run

    monkeypatch.setattr(routes, "save_upload", fake_save_upload)
    monkeypatch.setattr(routes, "IngestionPipeline", FakePipeline)

    response = client.post(
        "/api/ingest/upload?project_slug=my-topic&project_name=My%20Topic",
        files={"file": ("paper.pdf", b"content", "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "d-new"
    assert data["run_id"] == "r-new"
    assert data["project_id"] == "p-new"
    assert data["project_slug"] == "my-topic"
    assert data["document_title"] == "The Paper"
    assert data["status"] == "completed"


def test_pipeline_dashboard_selected_topic_returns_all_documents() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    docs = []
    for index in range(5):
        doc = Document(
            id=f"d{index}",
            project_id=project.id,
            title=f"Doc {index}",
            file_name=f"doc-{index}.pdf",
            sha256=f"sha{index}",
            raw_path=f"raw/doc-{index}.pdf",
            status="ready" if index % 2 == 0 else "processing",
        )
        docs.append(doc)
        db.add(doc)
        db.add(
            PipelineRun(
                id=f"r{index}",
                project_id=project.id,
                document_id=doc.id,
                status="completed" if index % 2 == 0 else "running",
            )
        )
    db.commit()
    client = make_client(db)

    response = client.get("/api/pipeline/dashboard?project_slug=demo")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["topics"]) == 1
    assert payload["topics"][0]["slug"] == "demo"
    assert payload["topics"][0]["document_count"] == 5
    assert len(payload["runs"]) == 5
    for item in payload["runs"]:
        assert item["project_slug"] == "demo"
        assert item["project_title"] == "Demo"
        assert "status" in item
        assert "status_label" in item
        assert "progress" in item
        assert item["action_available"] == (item["status"] == "completed")


def test_pipeline_dashboard_global_returns_one_row_per_document() -> None:
    db = make_session()
    projects = [
        Project(id="p1", slug="topic-a", name="Topic A"),
        Project(id="p2", slug="topic-b", name="Topic B"),
    ]
    db.add_all(projects)
    for project in projects:
        for index in range(3):
            doc = Document(
                id=f"{project.id}-d{index}",
                project_id=project.id,
                title=f"{project.name} Doc {index}",
                file_name=f"{project.slug}-{index}.pdf",
                sha256=f"{project.id}-sha{index}",
                raw_path=f"raw/{project.slug}-{index}.pdf",
                status="ready",
            )
            db.add(doc)
            db.add(
                PipelineRun(
                    id=f"{project.id}-r{index}",
                    project_id=project.id,
                    document_id=doc.id,
                    status="completed",
                )
            )
    db.commit()
    client = make_client(db)

    response = client.get("/api/pipeline/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["topics"]) == 2
    assert len(payload["runs"]) == 6
    document_ids = {item["document_id"] for item in payload["runs"]}
    assert len(document_ids) == 6



def test_document_title_prefers_metadata_title_over_hash_filename() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    pipeline = IngestionPipeline(db)

    # Human-readable metadata title wins over a hash-like filename.
    hash_document = Document(
        id="d1",
        project_id=project.id,
        title="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",
        file_name="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0.pdf",
        sha256="sha1",
        raw_path="raw/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0.pdf",
        status="pending",
    )
    db.add(hash_document)
    db.commit()
    parsed = ParsedDocument(
        title="",
        text="Some text.",
        chunks=[],
        metadata={"title": "Real Parsed Title"},
    )
    assert pipeline._resolve_document_title(parsed, hash_document) == "Real Parsed Title"

    # When metadata.title itself is internal-looking, the human-readable filename/title is used.
    human_document = Document(
        id="d2",
        project_id=project.id,
        title="Human Readable Paper",
        file_name="human_readable_paper.pdf",
        sha256="sha2",
        raw_path="raw/human_readable_paper.pdf",
        status="pending",
    )
    db.add(human_document)
    db.commit()
    parsed_internal_meta = ParsedDocument(
        title="",
        text="Some text.",
        chunks=[],
        metadata={"title": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0"},
    )
    assert pipeline._resolve_document_title(parsed_internal_meta, human_document) == "Human Readable Paper"

    # When both metadata.title and filename/title are internal-looking, fall back to a generic title.
    parsed_no_human = ParsedDocument(
        title="",
        text="Some text.",
        chunks=[],
        metadata={"title": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0"},
    )
    fallback = pipeline._resolve_document_title(parsed_no_human, hash_document)
    assert "Untitled" in fallback


def test_backfill_document_titles_is_idempotent_and_preserves_slugs() -> None:
    """The title backfill script updates titles without changing source slugs."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id=project.id,
        title="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",
        file_name="Readable Paper Name.pdf",
        sha256="sha",
        raw_path="raw/Readable Paper Name.pdf",
        metadata_json={"source_slug": "sources/stable-slug", "source_title": "Old Title"},
        status="pending",
    )
    db.add_all([project, document])
    db.commit()

    import scripts.backfill_document_titles as backfill_mod

    original_session_local = backfill_mod.SessionLocal
    backfill_mod.SessionLocal = lambda: db
    try:
        counts = backfill_mod.backfill_titles(dry_run=False)
        assert counts["total"] == 1
        assert counts["changed"] == 1

        refreshed = db.get(Document, "d1")
        assert refreshed is not None
        assert refreshed.title == "Readable Paper Name"
        assert refreshed.metadata_json.get("source_slug") == "sources/stable-slug"

        # Second run should be a no-op.
        counts2 = backfill_mod.backfill_titles(dry_run=False)
        assert counts2["changed"] == 0
    finally:
        backfill_mod.SessionLocal = original_session_local


def test_list_runs_supports_project_slug_filter_and_enriched_fields() -> None:
    db = make_session()
    topic_a = Project(id="p1", slug="topic-a", name="Topic A")
    topic_b = Project(id="p2", slug="topic-b", name="Topic B")
    db.add_all([topic_a, topic_b])
    for project in (topic_a, topic_b):
        for index in range(2):
            document = Document(
                id=f"{project.id}-d{index}",
                project_id=project.id,
                title=f"{project.name} Doc {index}",
                file_name=f"{project.slug}-{index}.pdf",
                sha256=f"{project.id}-sha{index}",
                raw_path=f"raw/{project.slug}-{index}.pdf",
                status="ready",
            )
            db.add(document)
            db.add(
                PipelineRun(
                    id=f"{project.id}-r{index}",
                    project_id=project.id,
                    document_id=document.id,
                    status="completed",
                    run_type="ingest",
                    notes="done",
                    provider_report={"progress": {"percent": 100, "stage": "completed", "message": "Done."}},
                )
            )
    db.commit()
    client = make_client(db)

    response = client.get("/api/runs?project_slug=topic-a")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    for item in data:
        assert item["project_slug"] == "topic-a"
        assert item["project_title"] == "Topic A"
        assert "document_title" in item
        assert "file_name" in item
        assert "status_label" in item
        assert "progress" in item
        assert item["progress"]["percent"] == 100
        assert "provider_report" in item
        assert item["action_available"] is True
        assert "created_at" in item
        assert "updated_at" in item
        assert item["run_type"] == "ingest"

    # Unknown project slug returns an empty list.
    assert client.get("/api/runs?project_slug=unknown").json() == []

    # Limit/offset validation and pagination still apply.
    assert len(client.get("/api/runs?limit=1&offset=0").json()) == 1
    assert len(client.get("/api/runs?limit=1&offset=1").json()) == 1
    assert client.get("/api/runs?limit=201").status_code == 422


def test_document_source_includes_project_identity() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo Project")
    document = Document(
        id="d1",
        project_id=project.id,
        title="Source Doc",
        file_name="source.pdf",
        sha256="sha1",
        raw_path="raw/source.pdf",
        status="ready",
    )
    db.add_all([project, document])
    db.commit()
    client = make_client(db)

    response = client.get("/api/documents/d1/source")
    assert response.status_code == 200
    payload = response.json()
    assert payload["project_slug"] == "demo"
    assert payload["project_title"] == "Demo Project"


def test_document_file_route_serves_inline_pdf_and_source_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(routes.settings, "raw_dir", tmp_path)
    raw_root = tmp_path / "demo"
    raw_root.mkdir()
    pdf_path = raw_root / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id=project.id,
        title="Paper",
        file_name="paper.pdf",
        sha256="sha1",
        raw_path=str(pdf_path),
        status="ready",
    )
    db.add_all([project, document])
    db.commit()
    client = make_client(db)

    response = client.get("/api/documents/d1/file?project_slug=demo")
    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 test"
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.headers["content-disposition"].startswith("inline;")

    source = client.get("/api/documents/d1/source").json()
    assert source["source_file_available"] is True
    assert source["source_file_is_pdf"] is True
    assert source["source_file_mime"] == "application/pdf"
    assert source["source_file_url"] == "/api/documents/d1/file?project_slug=demo"
    assert client.get("/api/documents/d1/file").status_code == 422


def test_document_file_route_rejects_missing_mismatch_and_out_of_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(routes.settings, "raw_dir", tmp_path / "raw")
    routes.settings.raw_dir.mkdir()
    outside_path = tmp_path / "outside.pdf"
    outside_path.write_bytes(b"outside")

    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    other = Project(id="p2", slug="other", name="Other")
    missing = Document(
        id="missing",
        project_id=project.id,
        title="Missing",
        file_name="missing.pdf",
        sha256="sha1",
        raw_path=str(routes.settings.raw_dir / "missing.pdf"),
    )
    escaped = Document(
        id="escaped",
        project_id=project.id,
        title="Escaped",
        file_name="escaped.pdf",
        sha256="sha2",
        raw_path=str(outside_path),
    )
    db.add_all([project, other, missing, escaped])
    db.commit()
    client = make_client(db)

    assert client.get("/api/documents/missing/file?project_slug=demo").status_code == 404
    assert client.get("/api/documents/escaped/file?project_slug=demo").status_code == 404
    assert client.get("/api/documents/missing/file?project_slug=other").status_code == 404
    source = client.get("/api/documents/escaped/source").json()
    assert source["source_file_available"] is False
    assert source["source_file_url"] is None



def test_delete_document_requires_project_slug() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id=project.id,
        title="Paper",
        file_name="paper.pdf",
        sha256="sha1",
        raw_path="raw/paper.pdf",
    )
    db.add_all([project, document])
    db.commit()
    client = make_client(db)

    response = client.delete("/api/documents/d1")

    assert response.status_code == 422
    assert db.get(Document, "d1") is not None


def test_delete_project_requires_confirmation_and_cleans_project_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(routes.settings, "raw_dir", tmp_path)
    demo_root = tmp_path / "demo"
    other_root = tmp_path / "other"
    demo_root.mkdir()
    other_root.mkdir()
    demo_pdf = demo_root / "demo.pdf"
    other_pdf = other_root / "other.pdf"
    attachment_file = demo_root / "__sessions__" / "s1" / "attachment.pdf"
    attachment_file.parent.mkdir(parents=True)
    demo_pdf.write_bytes(b"demo")
    other_pdf.write_bytes(b"other")
    attachment_file.write_bytes(b"attachment")

    db = make_session()
    demo = Project(id="p1", slug="demo", name="Demo")
    other = Project(id="p2", slug="other", name="Other")
    demo_doc = Document(
        id="d1",
        project_id=demo.id,
        title="Demo Doc",
        file_name="demo.pdf",
        sha256="sha1",
        raw_path=str(demo_pdf),
    )
    other_doc = Document(
        id="d2",
        project_id=other.id,
        title="Other Doc",
        file_name="other.pdf",
        sha256="sha2",
        raw_path=str(other_pdf),
    )
    db.add_all(
        [
            demo,
            other,
            demo_doc,
            other_doc,
            ConversationSession(
                id="s1",
                project_slug="demo",
                expires_at=datetime.utcnow() + timedelta(days=1),
            ),
            ConversationSession(
                id="s2",
                project_slug="other",
                expires_at=datetime.utcnow() + timedelta(days=1),
            ),
            ConversationTurn(id="t1", session_id="s1", turn_index=0, role="user", content="hello"),
            SessionAttachment(
                id="a1",
                session_id="s1",
                project_id=demo.id,
                file_name="attachment.pdf",
                storage_path=str(attachment_file),
                sha256="sha3",
                byte_size=10,
            ),
            SessionAttachmentChunk(id="ac1", attachment_id="a1", ordinal=0, text="attachment chunk"),
            AgentTraceRun(
                id="trace1",
                request_id="req1",
                session_id="s1",
                project_slug="demo",
                query="q",
                constraints={},
                final_answer="a",
                citations=[],
                warnings=[],
                status="completed",
            ),
            AgentTraceStep(id="step1", run_id="trace1", step_id=1, step_type="route", summary="s"),
        ]
    )
    db.commit()
    client = make_client(db)

    assert client.delete("/api/projects/demo?confirm_slug=wrong").status_code == 400
    assert db.get(Project, "p1") is not None

    response = client.delete("/api/projects/demo?confirm_slug=demo")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert db.get(Project, "p1") is None
    assert db.get(Document, "d1") is None
    assert db.get(ConversationSession, "s1") is None
    assert db.get(ConversationTurn, "t1") is None
    assert db.get(SessionAttachment, "a1") is None
    assert db.get(SessionAttachmentChunk, "ac1") is None
    assert db.get(AgentTraceRun, "trace1") is None
    assert db.get(AgentTraceStep, "step1") is None
    assert not demo_pdf.exists()
    assert not attachment_file.exists()

    assert db.get(Project, "p2") is not None
    assert db.get(Document, "d2") is not None
    assert db.get(ConversationSession, "s2") is not None
    assert other_pdf.exists()


def test_delete_document_removes_document_scoped_sessions(tmp_path, monkeypatch) -> None:
    """Deleting a document removes its document-scoped sessions and side data."""
    monkeypatch.setattr(routes.settings, "raw_dir", tmp_path)
    raw_root = tmp_path / "demo"
    raw_root.mkdir()
    pdf_path = raw_root / "paper.pdf"
    pdf_path.write_bytes(b"%PDF")

    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id=project.id,
        title="Paper",
        file_name="paper.pdf",
        sha256="sha1",
        raw_path=str(pdf_path),
        status="ready",
    )
    session = ConversationSession(
        id="doc-scoped-sess",
        project_slug="demo",
        document_id="d1",
        expires_at=datetime.utcnow() + timedelta(days=1),
    )
    turn = ConversationTurn(
        id="t1",
        session_id="doc-scoped-sess",
        turn_index=0,
        role="user",
        content="hello",
    )
    attachment = SessionAttachment(
        id="a1",
        session_id="doc-scoped-sess",
        project_id=project.id,
        file_name="note.pdf",
        storage_path=str(raw_root / "note.pdf"),
        sha256="sha2",
        byte_size=4,
    )
    db.add_all([project, document, session, turn, attachment])
    db.commit()

    client = make_client(db)
    response = client.delete("/api/documents/d1?project_slug=demo")
    assert response.status_code == 200
    assert response.json()["sessions_deleted"] == 1
    assert db.get(ConversationSession, "doc-scoped-sess") is None
    assert db.get(ConversationTurn, "t1") is None
    assert db.get(SessionAttachment, "a1") is None
    assert db.get(Document, "d1") is None
