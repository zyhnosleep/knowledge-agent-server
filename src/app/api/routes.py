from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError
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
    DocumentParseVersion,
    PipelineRun,
    Project,
    QuestionAnswer,
    ReviewItem,
)
from app.schemas.common import (
    CanonicalMarkdownRead,
    CanonicalParseRead,
    CitationLocationRead,
    DocumentRead,
    HealthResponse,
    IngestResponse,
    ProjectCreate,
    ProjectRead,
    PublicSourceSpan,
    QueryRequest,
    QueryResponse,
    ReviewItemRead,
)
from app.services.filesystem import InvalidStoragePathError, UploadTooLargeError, _ensure_within, safe_project_slug, save_upload
from app.services.canonical_artifacts import CanonicalArtifactStore
from app.services.conversation_memory import ConversationMemory
from app.services.pipeline import IngestionPipeline
from app.services.queue import JobDispatcher
from app.services.repositories import get_or_create_project
from app.services.search import QueryService
from app.services.vector_store import get_vector_store
from app.services.model_readiness import get_model_readiness
from app.services.model_runtime import get_model_runtime

router = APIRouter()
settings = get_settings()
_MAX_CANONICAL_MANIFEST_BYTES = 1024 * 1024
_MAX_CANONICAL_MARKDOWN_JSON_BYTES = 10 * 1024 * 1024
_FILE_CHUNK_BYTES = 1024 * 1024


@dataclass
class _VerifiedMarkdown:
    handle: BinaryIO
    size: int


def _validated_project_slug(project_slug: str) -> str:
    try:
        return safe_project_slug(project_slug)
    except InvalidStoragePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _document_raw_path(document: Document) -> Path:
    raw_base = settings.raw_dir.expanduser().resolve()
    stored_path = Path(document.raw_path)
    candidate = stored_path if stored_path.is_absolute() else raw_base / stored_path
    if stored_path.is_absolute() and not candidate.exists():
        project_slug = document.project.slug if document.project is not None else ""
        if stored_path.parent.name == project_slug:
            migrated_candidate = raw_base / project_slug / stored_path.name
            if migrated_candidate.exists():
                candidate = migrated_candidate
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


def _active_parse_or_404(
    db: Session,
    document_id: str,
    project_slug: str,
) -> tuple[Document, DocumentParseVersion]:
    safe_slug = _validated_project_slug(project_slug)
    document = db.get(Document, document_id)
    if (
        document is None
        or document.project is None
        or document.project.slug != safe_slug
        or not document.active_parse_version
    ):
        raise HTTPException(status_code=404, detail="Active parse not found.")
    version = db.scalar(
        select(DocumentParseVersion).where(
            DocumentParseVersion.document_id == document.id,
            DocumentParseVersion.version_key == document.active_parse_version,
        )
    )
    if version is None:
        raise HTTPException(status_code=404, detail="Active parse not found.")
    return document, version


def _open_regular_file(path: Path, *, maximum_bytes: int | None = None) -> BinaryIO:
    if CanonicalArtifactStore._is_link_or_reparse_point(path):  # noqa: SLF001
        raise ValueError(f"verified file cannot be a link: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"verified file is not regular: {path.name}")
        if maximum_bytes is not None and file_stat.st_size > maximum_bytes:
            raise ValueError(f"verified file exceeds size limit: {path.name}")
        path_stat = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or (path_stat.st_dev, path_stat.st_ino)
            != (file_stat.st_dev, file_stat.st_ino)
        ):
            raise ValueError(f"verified file identity changed: {path.name}")
        return os.fdopen(descriptor, "rb", closefd=True)
    except Exception:
        os.close(descriptor)
        raise


def _sha256_open_file(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(_FILE_CHUNK_BYTES), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _same_open_file(path: Path, handle: BinaryIO) -> bool:
    try:
        path_stat = os.stat(path, follow_symlinks=False)
        file_stat = os.fstat(handle.fileno())
    except OSError:
        return False
    return (
        stat.S_ISREG(path_stat.st_mode)
        and (path_stat.st_dev, path_stat.st_ino)
        == (file_stat.st_dev, file_stat.st_ino)
        and not CanonicalArtifactStore._is_link_or_reparse_point(path)  # noqa: SLF001
    )


def _active_canonical_paths(
    document: Document,
    version: DocumentParseVersion,
) -> tuple[Path, Path]:
    store = CanonicalArtifactStore(settings.canonical_artifacts_dir)
    root = settings.canonical_artifacts_dir.expanduser().resolve()
    store._validate_component(document.id)  # noqa: SLF001
    store._validate_component(version.version_key)  # noqa: SLF001
    document_root = store._prepare_document_root(  # noqa: SLF001
        document.id,
        create=False,
    ).resolve()
    bundle = document_root / version.version_key
    if store._is_link_or_reparse_point(bundle):  # noqa: SLF001
        raise ValueError("canonical bundle cannot be a link")
    if not bundle.is_dir() or bundle.resolve().parent != document_root:
        raise ValueError("canonical bundle escapes its document root")

    artifact_reference = Path(version.artifact_dir)
    if not artifact_reference.is_absolute():
        artifact_reference = root / artifact_reference
    if store._is_link_or_reparse_point(artifact_reference):  # noqa: SLF001
        raise ValueError("parse-version artifact directory cannot be a link")
    resolved_reference = artifact_reference.resolve()
    if (
        not resolved_reference.is_dir()
        or resolved_reference.parent != document_root
        or resolved_reference.name
        not in {version.version_key, f"{version.version_key}.pipeline"}
    ):
        raise ValueError("parse-version artifact directory escapes its document root")
    return bundle / "manifest.json", bundle / "canonical.md"


def _open_verified_canonical_markdown(
    document: Document,
    version: DocumentParseVersion,
) -> _VerifiedMarkdown:
    markdown_handle: BinaryIO | None = None
    try:
        manifest_path, markdown_path = _active_canonical_paths(document, version)
        with _open_regular_file(
            manifest_path,
            maximum_bytes=_MAX_CANONICAL_MANIFEST_BYTES,
        ) as manifest_handle:
            manifest = json.loads(manifest_handle.read().decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("canonical manifest is not an object")
        if (
            manifest.get("document_id") != document.id
            or manifest.get("version") != version.version_key
        ):
            raise ValueError("canonical manifest identity mismatch")

        checkpoint = version.manifest_json if isinstance(version.manifest_json, dict) else {}
        expected_input = checkpoint.get("input_fingerprint")
        expected_markdown = checkpoint.get("canonical_markdown_sha256")
        if (
            not isinstance(expected_input, str)
            or not isinstance(expected_markdown, str)
            or manifest.get("input_fingerprint") != expected_input
            or manifest.get("canonical_markdown_sha256") != expected_markdown
        ):
            raise ValueError("canonical checkpoint fingerprint mismatch")

        markdown_handle = _open_regular_file(markdown_path)
        size = os.fstat(markdown_handle.fileno()).st_size
        actual_markdown = _sha256_open_file(markdown_handle)
        if actual_markdown != expected_markdown:
            raise ValueError("canonical markdown fingerprint mismatch")
        if not _same_open_file(markdown_path, markdown_handle):
            raise ValueError("canonical markdown changed during verification")
        return _VerifiedMarkdown(handle=markdown_handle, size=size)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        if markdown_handle is not None:
            markdown_handle.close()
        raise HTTPException(status_code=404, detail="Canonical artifact not found.") from exc


def _stream_verified_markdown(verified: _VerifiedMarkdown) -> Iterator[bytes]:
    try:
        for chunk in iter(lambda: verified.handle.read(_FILE_CHUNK_BYTES), b""):
            yield chunk
    finally:
        verified.handle.close()


def _parse_progress(version: DocumentParseVersion) -> dict:
    if version.status == "active":
        return {"stage": "completed", "percent": 100}
    stages = (
        "parse",
        "repair",
        "canonicalize",
        "semantic_split",
        "contextualize",
        "embed",
        "index",
        "activate",
    )
    state = version.stage_state if isinstance(version.stage_state, dict) else {}
    completed_count = 0
    for stage in stages:
        checkpoint = state.get(stage) if isinstance(state.get(stage), dict) else {}
        checkpoint_status = checkpoint.get("status")
        if checkpoint_status == "completed":
            completed_count += 1
            continue
        if checkpoint_status in {"running", "failed"}:
            suffix = "_failed" if checkpoint_status == "failed" else ""
            return {
                "stage": stage + suffix,
                "percent": round(100 * completed_count / len(stages)),
            }
        break
    return {
        "stage": version.status or "unknown",
        "percent": round(100 * completed_count / len(stages)),
    }


def _repair_pages(version: DocumentParseVersion) -> list[int]:
    state = version.stage_state if isinstance(version.stage_state, dict) else {}
    repair = state.get("repair") if isinstance(state.get("repair"), dict) else {}
    output = repair.get("output") if isinstance(repair.get("output"), dict) else {}
    requests = output.get("repair_requests")
    if not isinstance(requests, list):
        return []
    pages: set[int] = set()
    for request in requests:
        if not isinstance(request, dict):
            continue
        page_index = request.get("page_index")
        if isinstance(page_index, int) and not isinstance(page_index, bool) and page_index >= 0:
            pages.add(page_index)
        locator = request.get("locator")
        if isinstance(locator, dict):
            locator_page = locator.get("page_index")
            if (
                isinstance(locator_page, int)
                and not isinstance(locator_page, bool)
                and locator_page >= 0
            ):
                pages.add(locator_page)
        page_indices = request.get("page_indices")
        if isinstance(page_indices, list):
            pages.update(
                page
                for page in page_indices
                if isinstance(page, int) and not isinstance(page, bool) and page >= 0
            )
    return sorted(pages)


def _source_type(document: Document) -> str:
    suffix = Path(document.file_name or "").suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix in {".html", ".htm"}:
        return "html"
    return "text"


def _source_fragment(source_type: str, spans: list[dict]) -> str:
    for span in spans:
        if not isinstance(span, dict):
            continue
        if source_type == "pdf":
            page_index = span.get("page_index")
            if (
                isinstance(page_index, int)
                and not isinstance(page_index, bool)
                and page_index >= 0
            ):
                return f"#page={page_index + 1}"
        elif source_type == "docx":
            paragraph_id = span.get("paragraph_id")
            if isinstance(paragraph_id, str) and paragraph_id:
                return f"#paragraph={quote(paragraph_id, safe='')}"
        elif source_type == "html":
            for key, label in (
                ("element_id", "element"),
                ("css_selector", "selector"),
                ("xpath", "xpath"),
            ):
                value = span.get(key)
                if isinstance(value, str) and value:
                    return f"#{label}={quote(value, safe='')}"
        else:
            line_start = span.get("line_start")
            if (
                isinstance(line_start, int)
                and not isinstance(line_start, bool)
                and line_start >= 0
            ):
                return f"#line={line_start}"
    return ""


def _public_source_spans(spans: list[dict]) -> list[PublicSourceSpan]:
    public: list[PublicSourceSpan] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        try:
            public.append(PublicSourceSpan.model_validate(span))
        except ValidationError:
            continue
    return public


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
    get_vector_store(db).delete_document(document_id)
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
    readiness = get_model_readiness().check()
    return HealthResponse(
        status=readiness["status"],
        app_name=settings.app_name,
        api_status="ok",
        models=readiness["models"],
        queues=get_model_runtime().snapshot(),
    )


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


@router.get("/documents/{document_id}/parse", response_model=CanonicalParseRead)
def get_active_parse(
    document_id: str,
    project_slug: str = Query(...),
    db: Session = Depends(get_db),
) -> CanonicalParseRead:
    document, version = _active_parse_or_404(db, document_id, project_slug)
    download_available = False
    try:
        verified = _open_verified_canonical_markdown(document, version)
        verified.handle.close()
        download_available = True
    except HTTPException:
        pass
    quality = version.quality_json if isinstance(version.quality_json, dict) else {}
    warning_count = sum(
        1
        for issue in quality.get("issues", [])
        if isinstance(issue, dict) and issue.get("severity") == "warning"
    )
    return CanonicalParseRead(
        document_id=document.id,
        version=version.version_key,
        parser=version.parser_name,
        parser_version=version.parser_version,
        progress=_parse_progress(version),
        quality={
            "status": quality.get("status"),
            "accepted": quality.get("accepted"),
            "score": quality.get("score"),
        },
        repair_pages=_repair_pages(version),
        warning_count=warning_count,
        download_available=download_available,
    )


@router.get(
    "/documents/{document_id}/parse/markdown",
    response_model=CanonicalMarkdownRead,
)
def get_active_parse_markdown(
    document_id: str,
    project_slug: str = Query(...),
    db: Session = Depends(get_db),
) -> CanonicalMarkdownRead:
    document, version = _active_parse_or_404(db, document_id, project_slug)
    verified = _open_verified_canonical_markdown(document, version)
    if verified.size > _MAX_CANONICAL_MARKDOWN_JSON_BYTES:
        verified.handle.close()
        raise HTTPException(status_code=413, detail="Canonical Markdown is too large for JSON view.")
    try:
        markdown = verified.handle.read().decode("utf-8")
    except UnicodeError as exc:
        raise HTTPException(status_code=404, detail="Canonical artifact not found.") from exc
    finally:
        verified.handle.close()
    return CanonicalMarkdownRead(
        document_id=document.id,
        version=version.version_key,
        markdown=markdown,
    )


@router.get("/documents/{document_id}/parse/download", response_class=StreamingResponse)
def download_active_parse(
    document_id: str,
    project_slug: str = Query(...),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    document, version = _active_parse_or_404(db, document_id, project_slug)
    verified = _open_verified_canonical_markdown(document, version)
    return StreamingResponse(
        _stream_verified_markdown(verified),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="canonical.md"',
            "Content-Length": str(verified.size),
        },
    )


@router.get(
    "/documents/{document_id}/citations/{chunk_id}/location",
    response_model=CitationLocationRead,
    response_model_exclude_defaults=True,
    response_model_exclude_none=True,
)
def get_citation_location(
    document_id: str,
    chunk_id: str,
    project_slug: str = Query(...),
    db: Session = Depends(get_db),
) -> CitationLocationRead:
    document, version = _active_parse_or_404(db, document_id, project_slug)
    chunk = db.scalar(
        select(DocumentChunk).where(
            DocumentChunk.id == chunk_id,
            DocumentChunk.document_id == document.id,
            DocumentChunk.parse_version == version.version_key,
            DocumentChunk.block_type != "reference",
        )
    )
    if chunk is None:
        raise HTTPException(status_code=404, detail="Citation location not found.")
    file_metadata = _document_file_metadata(document)
    source_url = file_metadata.get("source_file_url")
    if not isinstance(source_url, str):
        raise HTTPException(status_code=404, detail="Source file not found.")
    spans = chunk.source_spans if isinstance(chunk.source_spans, list) else []
    public_spans = _public_source_spans(spans)
    public_span_dicts = [
        span.model_dump(mode="json", exclude_none=True) for span in public_spans
    ]
    source_type = _source_type(document)
    return CitationLocationRead(
        document_id=document.id,
        chunk_id=chunk.id,
        parse_version=version.version_key,
        source_type=source_type,
        source_url=source_url + _source_fragment(source_type, public_span_dicts),
        source_spans=public_spans,
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
