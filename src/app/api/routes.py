from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.models.records import (
    AgentTraceRun,
    AgentTraceStep,
    Claim,
    ConversationSession,
    Document,
    DocumentChunk,
    PipelineRun,
    Project,
    QuestionAnswer,
    ReviewItem,
)
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
from app.services.filesystem import InvalidStoragePathError, UploadTooLargeError, _ensure_within, safe_project_slug, save_upload
from app.services.conversation_memory import ConversationMemory
from app.services.pipeline import IngestionPipeline
from app.services.queue import JobDispatcher
from app.services.repositories import get_or_create_project
from app.services.search import QueryService
from app.services.vector_store import SQLiteVecStore

router = APIRouter()
settings = get_settings()


def _validated_project_slug(project_slug: str) -> str:
    try:
        return safe_project_slug(project_slug)
    except InvalidStoragePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _document_raw_path(document: Document) -> Path:
    raw_base = settings.raw_dir.expanduser().resolve()
    stored_path = Path(document.raw_path)
    candidate = stored_path if stored_path.is_absolute() else raw_base / stored_path
    return _ensure_within(candidate, raw_base)


def _document_file_metadata(document: Document) -> dict:
    try:
        path = _document_raw_path(document)
    except InvalidStoragePathError:
        return _empty_document_file_metadata(document)
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    available = path.exists() and path.is_file()
    source_file_url = None
    if available:
        project_slug = document.project.slug if document.project is not None else None
        source_file_url = f"/api/documents/{quote(document.id)}/file"
        if project_slug:
            source_file_url += f"?project_slug={quote(project_slug)}"
    return {
        "source_file_available": available,
        "source_file_url": source_file_url,
        "source_file_mime": media_type,
        "source_file_name": document.file_name or path.name,
        "source_file_is_pdf": path.suffix.lower() == ".pdf" or media_type == "application/pdf",
    }


def _empty_document_file_metadata(document: Document) -> dict:
    return {
        "source_file_available": False,
        "source_file_url": None,
        "source_file_mime": None,
        "source_file_name": document.file_name,
        "source_file_is_pdf": False,
    }


def _document_raw_path_or_404(document: Document) -> Path:
    try:
        path = _document_raw_path(document)
    except InvalidStoragePathError as exc:
        raise HTTPException(status_code=404, detail="File not found.") from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return path


def _delete_document_file(db: Session, document: Document) -> bool:
    try:
        path = _document_raw_path(document)
    except InvalidStoragePathError:
        return False
    if not path.exists() or not path.is_file():
        return False
    other_document = db.scalar(
        select(Document)
        .where(Document.id != document.id, Document.raw_path == document.raw_path)
        .limit(1)
    )
    if other_document is not None:
        return False
    path.unlink(missing_ok=True)
    return True


def _delete_document_resources(db: Session, document: Document) -> dict:
    document_id = document.id
    project_id = document.project_id
    SQLiteVecStore(db).delete_document(document_id)
    file_deleted = _delete_document_file(db, document)
    trace_runs_deleted = _delete_trace_runs_citing_document(db, document_id)

    query_answers_deleted = 0
    query_answers = db.scalars(select(QuestionAnswer).where(QuestionAnswer.project_id == project_id)).all()
    for answer in query_answers:
        citations = answer.citations if isinstance(answer.citations, list) else []
        if any(isinstance(citation, dict) and citation.get("document_id") == document_id for citation in citations):
            db.delete(answer)
            query_answers_deleted += 1

    reviews_deleted = db.execute(delete(ReviewItem).where(ReviewItem.document_id == document_id)).rowcount or 0
    claims_deleted = db.execute(delete(Claim).where(Claim.document_id == document_id)).rowcount or 0
    runs_deleted = db.execute(delete(PipelineRun).where(PipelineRun.document_id == document_id)).rowcount or 0
    sessions_deleted = ConversationMemory(db).delete_sessions_for_document(document_id)
    db.delete(document)
    db.flush()
    return {
        "file_deleted": file_deleted,
        "reviews_deleted": int(reviews_deleted),
        "claims_deleted": int(claims_deleted),
        "runs_deleted": int(runs_deleted),
        "query_answers_deleted": query_answers_deleted,
        "trace_runs_deleted": trace_runs_deleted,
        "sessions_deleted": sessions_deleted,
    }


def _delete_trace_runs_for_project(db: Session, project_slug: str) -> int:
    trace_ids = [
        trace_id
        for trace_id in db.scalars(select(AgentTraceRun.id).where(AgentTraceRun.project_slug == project_slug)).all()
    ]
    if not trace_ids:
        return 0
    db.execute(delete(AgentTraceStep).where(AgentTraceStep.run_id.in_(trace_ids)))
    db.execute(delete(AgentTraceRun).where(AgentTraceRun.id.in_(trace_ids)))
    return len(trace_ids)


def _delete_trace_runs_citing_document(db: Session, document_id: str) -> int:
    trace_ids: list[str] = []
    traces = db.scalars(select(AgentTraceRun)).all()
    for trace in traces:
        citations = trace.citations if isinstance(trace.citations, list) else []
        if any(isinstance(citation, dict) and citation.get("document_id") == document_id for citation in citations):
            trace_ids.append(trace.id)
    if not trace_ids:
        return 0
    db.execute(delete(AgentTraceStep).where(AgentTraceStep.run_id.in_(trace_ids)))
    db.execute(delete(AgentTraceRun).where(AgentTraceRun.id.in_(trace_ids)))
    return len(trace_ids)


def _clamped_percent(value: object, fallback: int = 0) -> int:
    try:
        percent = int(value)
    except (TypeError, ValueError):
        percent = fallback
    return max(0, min(percent, 100))


def _progress_from_run(run: PipelineRun | None, document: Document | None = None) -> dict:
    if run is not None:
        report = run.provider_report or {}
        progress = report.get("progress") if isinstance(report, dict) else None
        progress = progress if isinstance(progress, dict) else {}
        return {
            "percent": _clamped_percent(progress.get("percent"), _status_default_percent(run.status)),
            "stage": str(progress.get("stage") or run.status),
            "message": str(progress.get("message") or run.notes or ""),
        }
    if document is None:
        return {"percent": 0, "stage": "unknown", "message": ""}
    return {
        "percent": _status_default_percent(document.status),
        "stage": document.status,
        "message": "",
    }


def _status_default_percent(status: str) -> int:
    if status in {"ready", "completed"}:
        return 100
    if status in {"processing", "running"}:
        return 50
    if status in {"pending", "queued"}:
        return 5
    if status == "failed":
        return 100
    return 0


def _business_status(document: Document | None, run: PipelineRun | None = None) -> str:
    raw_status = run.status if run is not None else (document.status if document is not None else "unknown")
    if raw_status in {"ready", "completed"}:
        return "completed"
    if raw_status in {"pending", "processing", "queued", "running"}:
        return "processing" if raw_status == "processing" else raw_status
    if raw_status == "failed":
        return "failed"
    return raw_status


def _status_label(status: str) -> str:
    return {
        "queued": "Queued",
        "running": "Processing",
        "processing": "Processing",
        "completed": "Completed",
        "failed": "Failed",
    }.get(status, status.title())


def _build_pipeline_topic(
    project: Project,
    documents: list[Document],
    latest_runs_by_document: dict[str, PipelineRun],
) -> dict:
    document_count = len(documents)
    completed_count = 0
    processing_count = 0
    failed_count = 0
    percents: list[int] = []

    for document in documents:
        run = latest_runs_by_document.get(document.id)
        status = _business_status(document, run)
        progress = _progress_from_run(run, document)
        percents.append(progress["percent"])
        if status == "completed":
            completed_count += 1
        elif status == "failed":
            failed_count += 1
        elif status in {"pending", "processing", "queued", "running"}:
            processing_count += 1

    if failed_count:
        status = "failed"
        status_label = f"{failed_count} documents failed"
    elif processing_count:
        status = "processing"
        status_label = f"{processing_count} documents processing"
    elif completed_count:
        status = "ready"
        status_label = "Ready"
    else:
        status = "empty"
        status_label = "No documents"

    progress_percent = int(sum(percents) / len(percents)) if percents else 0
    parsing_rate = int(completed_count * 100 / document_count) if document_count else 0
    return {
        "id": project.id,
        "slug": project.slug,
        "title": project.name,
        "document_count": document_count,
        "completed_count": completed_count,
        "processing_count": processing_count,
        "failed_count": failed_count,
        "progress_percent": progress_percent,
        "parsing_rate": parsing_rate,
        "status": status,
        "status_label": status_label,
    }


def _build_pipeline_run_item(run: PipelineRun | None, document: Document | None, project: Project | None = None) -> dict:
    status = _business_status(document, run)
    progress = _progress_from_run(run, document)
    created_at = run.created_at.isoformat() if run is not None else (document.updated_at.isoformat() if document is not None else "")
    updated_at = run.updated_at.isoformat() if run is not None else created_at
    return {
        "id": run.id if run is not None else None,
        "document_id": document.id if document is not None else (run.document_id if run is not None else None),
        "document_title": document.title if document is not None else "Unlinked document",
        "file_name": document.file_name if document is not None else None,
        "project_slug": project.slug if project is not None else (document.project.slug if document is not None and document.project is not None else None),
        "project_title": project.name if project is not None else (document.project.name if document is not None and document.project is not None else None),
        "status": status,
        "status_label": _status_label(status),
        "run_type": run.run_type if run is not None else None,
        "notes": run.notes if run is not None else None,
        "provider_report": run.provider_report if run is not None else None,
        "progress": progress,
        "action_available": bool(document and status == "completed"),
        "created_at": created_at,
        "updated_at": updated_at,
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name)


@router.get("/projects", response_model=list[ProjectRead])
def list_projects(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[ProjectRead]:
    projects = db.scalars(select(Project).order_by(Project.created_at.desc()).limit(limit).offset(offset)).all()
    return [ProjectRead(id=item.id, slug=item.slug, name=item.name, description=item.description) for item in projects]


@router.post("/projects", response_model=ProjectRead)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> ProjectRead:
    project = get_or_create_project(db, _validated_project_slug(payload.slug), payload.name)
    if payload.description and project.description != payload.description:
        project.description = payload.description
        db.commit()
        db.refresh(project)
    return ProjectRead(id=project.id, slug=project.slug, name=project.name, description=project.description)


@router.delete("/projects/{project_slug}", response_model=dict)
def delete_project(
    project_slug: str,
    confirm_slug: str = Query(..., description="Must exactly match the project slug."),
    db: Session = Depends(get_db),
) -> dict:
    safe_slug = _validated_project_slug(project_slug)
    if confirm_slug != safe_slug:
        raise HTTPException(status_code=400, detail="confirm_slug must match project_slug.")
    project = db.scalar(select(Project).where(Project.slug == safe_slug))
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    documents = db.scalars(select(Document).where(Document.project_id == project.id)).all()
    document_results = [_delete_document_resources(db, document) for document in documents]
    session_ids = db.scalars(
        select(ConversationSession.id).where(ConversationSession.project_slug == project.slug)
    ).all()
    memory = ConversationMemory(db)
    deleted_turns = 0
    for session_id in session_ids:
        deleted_turns += memory.delete_session(session_id)

    trace_runs_deleted = _delete_trace_runs_for_project(db, project.slug)
    review_items_deleted = db.execute(delete(ReviewItem).where(ReviewItem.project_id == project.id)).rowcount or 0
    claims_deleted = db.execute(delete(Claim).where(Claim.project_id == project.id)).rowcount or 0
    runs_deleted = db.execute(delete(PipelineRun).where(PipelineRun.project_id == project.id)).rowcount or 0
    answers_deleted = db.execute(delete(QuestionAnswer).where(QuestionAnswer.project_id == project.id)).rowcount or 0
    db.delete(project)
    db.commit()
    return {
        "deleted": True,
        "project_slug": safe_slug,
        "documents_deleted": len(documents),
        "files_deleted": sum(1 for result in document_results if result["file_deleted"]),
        "sessions_deleted": len(session_ids),
        "turns_deleted": deleted_turns,
        "trace_runs_deleted": trace_runs_deleted,
        "review_items_deleted": int(review_items_deleted),
        "claims_deleted": int(claims_deleted),
        "runs_deleted": int(runs_deleted),
        "answers_deleted": int(answers_deleted),
    }


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    project_slug: str = Query(default=settings.default_project_slug),
    project_name: str = Query(default=settings.default_project_name),
    db: Session = Depends(get_db),
) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    try:
        safe_slug = _validated_project_slug(project_slug)
        saved_path = await save_upload(safe_slug, file)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidStoragePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pipeline = IngestionPipeline(db)
    project, document, run = pipeline.register_document(safe_slug, project_name, saved_path)
    if run.status != "completed":
        result = JobDispatcher().enqueue_or_run("app.workers.jobs.run_document_ingestion", document.id)
        if isinstance(result, str):
            db.refresh(run)
    return IngestResponse(
        document_id=document.id,
        project_id=project.id,
        project_slug=project.slug,
        run_id=run.id,
        status=run.status,
        document_title=document.title,
    )


@router.get("/documents", response_model=list[DocumentRead])
def list_documents(
    project_slug: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[DocumentRead]:
    statement = select(Document).order_by(Document.created_at.desc())
    if project_slug:
        project = db.scalar(select(Project).where(Project.slug == _validated_project_slug(project_slug)))
        if project is None:
            return []
        statement = statement.where(Document.project_id == project.id)
    documents = db.scalars(statement.limit(limit).offset(offset)).all()
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


@router.get("/pipeline/dashboard", response_model=dict)
def get_pipeline_dashboard(
    project_slug: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict:
    project_statement = select(Project).order_by(Project.created_at.desc())
    if project_slug:
        project_statement = project_statement.where(Project.slug == _validated_project_slug(project_slug))
    projects = db.scalars(project_statement.limit(limit)).all()
    project_ids = [project.id for project in projects]
    project_map = {project.id: project for project in projects}

    documents: list[Document] = []
    if project_ids:
        documents = db.scalars(
            select(Document)
            .where(Document.project_id.in_(project_ids))
            .order_by(Document.updated_at.desc(), Document.created_at.desc(), Document.id.desc())
            .limit(limit)
        ).all()

    document_ids = [document.id for document in documents]
    documents_by_project: dict[str, list[Document]] = {}
    for document in documents:
        documents_by_project.setdefault(document.project_id, []).append(document)

    # One run item per document: pick the latest run for each document.
    latest_runs_by_document: dict[str, PipelineRun] = {}
    if document_ids:
        runs = db.scalars(
            select(PipelineRun)
            .where(PipelineRun.document_id.in_(document_ids))
            .order_by(PipelineRun.updated_at.desc(), PipelineRun.created_at.desc(), PipelineRun.id.desc())
        ).all()
        for run in runs:
            if run.document_id and run.document_id not in latest_runs_by_document:
                latest_runs_by_document[run.document_id] = run

    topics = [
        _build_pipeline_topic(project, documents_by_project.get(project.id, []), latest_runs_by_document)
        for project in projects
    ]
    run_items = [
        _build_pipeline_run_item(
            latest_runs_by_document.get(document.id),
            document,
            project_map.get(document.project_id),
        )
        for document in documents
    ]
    completed_count = sum(1 for item in run_items if item["status"] == "completed")
    processing_count = sum(1 for item in run_items if item["status"] in {"queued", "running"})
    failed_count = sum(1 for item in run_items if item["status"] == "failed")

    return {
        "service_status": "ok",
        "topics": topics,
        "runs": run_items,
        "totals": {
            "topic_count": len(topics),
            "document_count": len(documents),
            "run_count": len(run_items),
            "completed_count": completed_count,
            "processing_count": processing_count,
            "failed_count": failed_count,
        },
    }


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


@router.get("/documents/{document_id}/file", response_class=FileResponse)
def get_document_file(
    document_id: str,
    project_slug: str = Query(...),
    db: Session = Depends(get_db),
) -> FileResponse:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    safe_slug = _validated_project_slug(project_slug)
    if document.project is None or document.project.slug != safe_slug:
        raise HTTPException(status_code=404, detail="Document not found.")
    path = _document_raw_path_or_404(document)
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(
        path,
        media_type=media_type,
        filename=document.file_name or path.name,
        content_disposition_type="inline",
    )


@router.delete("/documents/{document_id}", response_model=dict)
def delete_document(
    document_id: str,
    project_slug: str = Query(...),
    db: Session = Depends(get_db),
) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    safe_slug = _validated_project_slug(project_slug)
    if document.project is None or document.project.slug != safe_slug:
        raise HTTPException(status_code=404, detail="Document not found.")
    result = _delete_document_resources(db, document)
    db.commit()
    return {
        "deleted": True,
        "document_id": document_id,
        **result,
    }


def _build_document_markdown(document: Document, chunks: list[DocumentChunk], db: Session) -> str:
    """Return the best complete Markdown/source body available for a document."""
    if document.raw_text:
        return document.raw_text
    return "\n\n".join(chunk.text for chunk in chunks)


@router.get("/documents/{document_id}/source", response_model=dict)
def get_document_source(document_id: str, db: Session = Depends(get_db)) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.ordinal.asc())
    ).all()
    chunk_items = [
        {
            "chunk_id": chunk.id,
            "label": f"RAG Chunk #{index:02d}",
            "ordinal": chunk.ordinal,
            "page_label": chunk.page_label,
            "heading": chunk.heading,
            "text": chunk.text,
            "token_estimate": chunk.token_estimate,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
    fallback_preview = (document.raw_text or "").strip()
    if not chunk_items and fallback_preview:
        chunk_items.append(
            {
                "chunk_id": None,
                "label": "RAG Chunk #01",
                "ordinal": 1,
                "page_label": None,
                "heading": None,
                "text": fallback_preview[:2400],
                "token_estimate": max(1, len(fallback_preview) // 4),
            }
        )

    return {
        "document_id": document.id,
        "document_title": document.title,
        "file_name": document.file_name,
        "project_slug": document.project.slug if document.project is not None else None,
        "project_title": document.project.name if document.project is not None else None,
        "status": document.status,
        "raw_preview": fallback_preview[:2400],
        "markdown": _build_document_markdown(document, chunks, db),
        "chunks": chunk_items,
        **_document_file_metadata(document),
    }


@router.get("/documents/{document_id}/quality", response_model=dict)
def get_document_quality(document_id: str, db: Session = Depends(get_db)) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    chunk_count = db.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == document_id)) or 0
    return {"document_id": document.id, "status": document.status, "chunk_count": int(chunk_count)}


@router.get("/runs", response_model=list[dict])
def list_runs(
    project_slug: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[dict]:
    statement = select(PipelineRun).order_by(PipelineRun.created_at.desc())
    project = None
    if project_slug:
        project = db.scalar(select(Project).where(Project.slug == _validated_project_slug(project_slug)))
        if project is None:
            return []
        statement = statement.where(PipelineRun.project_id == project.id)
    runs = db.scalars(statement.limit(limit).offset(offset)).all()

    document_ids = {run.document_id for run in runs if run.document_id}
    project_ids = {run.project_id for run in runs if run.project_id}
    if project is not None:
        project_ids.add(project.id)

    documents: dict[str, Document] = {}
    projects: dict[str, Project] = {}
    if document_ids:
        documents = {doc.id: doc for doc in db.scalars(select(Document).where(Document.id.in_(document_ids))).all()}
    if project_ids:
        projects = {proj.id: proj for proj in db.scalars(select(Project).where(Project.id.in_(project_ids))).all()}

    return [
        _build_pipeline_run_item(
            run,
            documents.get(run.document_id),
            projects.get(run.project_id),
        )
        for run in runs
    ]


@router.get("/reviews", response_model=list[ReviewItemRead])
def list_reviews(
    project_slug: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[ReviewItemRead]:
    statement = select(ReviewItem).order_by(ReviewItem.created_at.desc())
    if project_slug:
        project = db.scalar(select(Project).where(Project.slug == _validated_project_slug(project_slug)))
        if project is None:
            return []
        statement = statement.where(ReviewItem.project_id == project.id)
    items = db.scalars(statement.limit(limit).offset(offset)).all()
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
        return QueryService(db).answer(
            payload.project_slug,
            payload.question,
            payload.save_answer,
            document_id=payload.document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
