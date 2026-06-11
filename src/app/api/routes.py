from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.models.records import Document, PipelineRun, Project, ReviewItem
from app.schemas.common import (
    DocumentRead,
    HealthResponse,
    IngestResponse,
    ProjectCreate,
    ProjectRead,
    QueryRequest,
    QueryResponse,
    ReviewItemRead,
)
from app.services.filesystem import save_upload
from app.services.pipeline import IngestionPipeline
from app.services.queue import JobDispatcher
from app.services.repositories import get_or_create_project
from app.services.search import QueryService

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name)


@router.get("/projects", response_model=list[ProjectRead])
def list_projects(db: Session = Depends(get_db)) -> list[ProjectRead]:
    projects = db.scalars(select(Project).order_by(Project.created_at.desc())).all()
    return [ProjectRead(id=item.id, slug=item.slug, name=item.name, description=item.description) for item in projects]


@router.post("/projects", response_model=ProjectRead)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> ProjectRead:
    project = get_or_create_project(db, payload.slug, payload.name)
    if payload.description and project.description != payload.description:
        project.description = payload.description
        db.commit()
        db.refresh(project)
    return ProjectRead(id=project.id, slug=project.slug, name=project.name, description=project.description)


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    project_slug: str = Query(default=settings.default_project_slug),
    project_name: str = Query(default=settings.default_project_name),
    db: Session = Depends(get_db),
) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    saved_path = await save_upload(project_slug, file)
    pipeline = IngestionPipeline(db)
    _, document, run = pipeline.register_document(project_slug, project_name, saved_path)
    if run.status != "completed":
        result = JobDispatcher().enqueue_or_run("app.workers.jobs.run_document_ingestion", document.id)
        if isinstance(result, str):
            db.refresh(run)
    return IngestResponse(document_id=document.id, run_id=run.id, status=run.status)


@router.get("/documents", response_model=list[DocumentRead])
def list_documents(project_slug: str | None = None, db: Session = Depends(get_db)) -> list[DocumentRead]:
    statement = select(Document).order_by(Document.created_at.desc())
    if project_slug:
        project = db.scalar(select(Project).where(Project.slug == project_slug))
        if project is None:
            return []
        statement = statement.where(Document.project_id == project.id)
    documents = db.scalars(statement).all()
    return [
        DocumentRead(
            id=item.id,
            title=item.title,
            file_name=item.file_name,
            status=item.status,
            sha256=item.sha256,
            metadata_json=item.metadata_json,
        )
        for item in documents
    ]


@router.get("/documents/{document_id}", response_model=DocumentRead)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentRead:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentRead(
        id=document.id,
        title=document.title,
        file_name=document.file_name,
        status=document.status,
        sha256=document.sha256,
        metadata_json=document.metadata_json,
    )


@router.get("/runs", response_model=list[dict])
def list_runs(db: Session = Depends(get_db)) -> list[dict]:
    runs = db.scalars(select(PipelineRun).order_by(PipelineRun.created_at.desc())).all()
    return [
        {
            "id": run.id,
            "document_id": run.document_id,
            "status": run.status,
            "run_type": run.run_type,
            "notes": run.notes,
            "provider_report": run.provider_report,
            "created_at": run.created_at.isoformat(),
        }
        for run in runs
    ]


@router.get("/reviews", response_model=list[ReviewItemRead])
def list_reviews(project_slug: str | None = None, db: Session = Depends(get_db)) -> list[ReviewItemRead]:
    statement = select(ReviewItem).order_by(ReviewItem.created_at.desc())
    if project_slug:
        project = db.scalar(select(Project).where(Project.slug == project_slug))
        if project is None:
            return []
        statement = statement.where(ReviewItem.project_id == project.id)
    items = db.scalars(statement).all()
    return [
        ReviewItemRead(
            id=item.id,
            title=item.title,
            detail=item.detail,
            severity=item.severity,
            status=item.status,
            payload=item.payload,
        )
        for item in items
    ]


@router.post("/query", response_model=QueryResponse)
def answer_query(payload: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    try:
        return QueryService(db).answer(payload.project_slug, payload.question, payload.save_answer)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
