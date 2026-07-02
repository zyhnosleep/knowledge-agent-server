from __future__ import annotations

from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes
from app.api.routes import router
from app.db.session import Base, get_db
from app.models.records import Document, DocumentChunk, PipelineRun, Project, ReviewItem
from app.services.filesystem import InvalidStoragePathError, UploadTooLargeError


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


def test_wiki_lint_route_passes_pagination(monkeypatch) -> None:
    db = make_session()
    client = make_client(db, raise_server_exceptions=False)

    def fake_lint_project_wiki(db_session: Session, project_slug: str, *, limit: int, offset: int) -> dict:
        assert db_session is db
        return {
            "project_slug": project_slug,
            "page_count": 0,
            "issue_count": 5,
            "limit": limit,
            "offset": offset,
            "returned_issue_count": 1,
            "issues": [{"kind": "demo"}],
        }

    monkeypatch.setattr(routes, "lint_project_wiki", fake_lint_project_wiki)

    response = client.get("/api/wiki/lint?project_slug=demo&limit=1&offset=2")

    assert response.status_code == 200
    assert response.json()["limit"] == 1
    assert response.json()["offset"] == 2
    assert response.json()["returned_issue_count"] == 1
    assert response.json()["issue_count"] == 5


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
