from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes
from app.api.routes import router
from app.db.session import Base, get_db
from app.models.records import Document, DocumentChunk, PageKind, PipelineRun, Project, ReviewItem, WikiPage
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


def test_document_source_route_returns_markdown_body() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id=project.id,
        title="Source Doc",
        file_name="source.pdf",
        sha256="sha1",
        raw_path="raw/source.pdf",
        raw_text="Raw text fallback.",
        status="ready",
    )
    wiki = WikiPage(
        id="w1",
        project_id=project.id,
        slug="source-doc",
        title="Source Doc",
        kind=PageKind.source_summary.value,
        markdown_path="wiki/source-doc.md",
        markdown_content="# Source Doc\n\nWiki markdown body.",
        source_document_ids=[document.id],
    )
    db.add_all([project, document, wiki])
    db.commit()
    client = make_client(db)

    response = client.get("/api/documents/d1/source")

    assert response.status_code == 200
    payload = response.json()
    assert payload["markdown"] == wiki.markdown_content

    # When no wiki page covers the document, raw_text is used.
    wiki.source_document_ids = []
    db.commit()
    response = client.get("/api/documents/d1/source")
    assert response.json()["markdown"] == document.raw_text


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
