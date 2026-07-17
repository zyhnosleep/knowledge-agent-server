from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.api.dependencies import require_business_api_user
from app.db.session import get_db
from app.models.records import ConversationSession, Document, SessionAttachment, SessionAttachmentChunk, User
from app.schemas.agent import (
    AgentConstraints,
    AgentQueryRequest,
    AgentQueryResponse,
    AgentSessionRead,
    AgentTurnRead,
    AttachmentChunkRead,
    AttachmentRead,
    AttachmentUploadResponse,
)
from app.services.agent_executor import AgentExecutor
from app.services.agent_synthesizer import AgentSynthesizer
from app.services.agent_trace_store import AgentTraceStore
from app.services.conversation_memory import ConversationMemory
from app.services.filesystem import InvalidStoragePathError, UploadTooLargeError
from app.services.rag_adapter import RAGAdapter
from app.services.repositories import get_or_create_project
from app.services.session_attachments import (
    delete_session_attachment,
    get_session_attachment,
    list_session_attachments,
    save_session_attachment,
)
from app.services.tool_registry import ToolRegistry

agent_router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


def _build_executor(
    db: Session,
    owner_user_id: str | None = None,
    *,
    event_sink: Callable[[str, dict], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> AgentExecutor:
    """Create an AgentExecutor with all standard dependencies."""
    rag = RAGAdapter()
    synthesizer = AgentSynthesizer()
    tools = ToolRegistry()
    tools._register_builtins(rag, synthesizer=synthesizer)
    memory = ConversationMemory(db, owner_user_id=owner_user_id)
    trace_store = AgentTraceStore(db, owner_user_id=owner_user_id)
    return AgentExecutor(
        rag=rag, tools=tools, memory=memory, db=db,
        trace_store=trace_store, synthesizer=synthesizer,
        event_sink=event_sink, cancel_event=cancel_event,
    )


def _apply_server_constraint_defaults(payload: AgentQueryRequest) -> AgentQueryRequest:
    """Use server-level Agent defaults when clients omit constraints.

    Pydantic fills omitted constraint fields with schema defaults before the
    route sees the payload. Treat those schema defaults as "not specified" so
    .env tuning such as AGENT_TIMEOUT_SECONDS actually affects UI requests.
    """
    constraints = payload.constraints
    schema_defaults = AgentConstraints()
    updates: dict[str, int | bool] = {}

    if constraints.allow_external_network == schema_defaults.allow_external_network:
        updates["allow_external_network"] = settings.agent_allow_external_network
    if constraints.max_steps == schema_defaults.max_steps:
        updates["max_steps"] = _positive_int(settings.agent_max_steps, schema_defaults.max_steps)
    if constraints.max_tool_calls == schema_defaults.max_tool_calls:
        updates["max_tool_calls"] = _positive_int(
            settings.agent_max_tool_calls,
            schema_defaults.max_tool_calls,
        )
    if constraints.budget_tokens == schema_defaults.budget_tokens:
        updates["budget_tokens"] = _positive_int(
            settings.agent_budget_tokens,
            schema_defaults.budget_tokens,
        )
    if constraints.timeout_seconds == schema_defaults.timeout_seconds:
        updates["timeout_seconds"] = _positive_int(
            settings.agent_timeout_seconds,
            schema_defaults.timeout_seconds,
        )

    if not updates:
        return payload
    return payload.model_copy(update={"constraints": constraints.model_copy(update=updates)})


def _positive_int(value: int, fallback: int) -> int:
    return value if isinstance(value, int) and value > 0 else fallback


def _assert_session_project(session_id: str | None, project_slug: str, db: Session) -> None:
    """Raise if an existing session belongs to a different project."""
    if not session_id:
        return
    session = db.get(ConversationSession, session_id)
    if session is not None and session.project_slug != project_slug:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session {session_id} belongs to project {session.project_slug}; "
                f"cannot be used for project {project_slug}."
            ),
        )


def _assert_session_scope(
    session_id: str | None,
    project_slug: str,
    document_id: str | None,
    db: Session,
    owner_user_id: str | None = None,
) -> None:
    """Raise if an existing session is being reused with an incompatible scope."""
    if not session_id:
        return
    session = db.get(ConversationSession, session_id)
    if owner_user_id is not None and session is not None and session.owner_user_id != owner_user_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    _assert_session_project(session_id, project_slug, db)
    if session is None:
        return
    if session.document_id != document_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session {session_id} is scoped to document {session.document_id}; "
                f"cannot be used for document {document_id}."
            ),
        )


def _validate_document_in_project(
    document_id: str | None, project_slug: str, db: Session
) -> None:
    """Raise 404 if the document does not exist or belongs to another project."""
    if document_id is None:
        return
    document = db.get(Document, document_id)
    if document is None or document.project is None or document.project.slug != project_slug:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found in project '{project_slug}'.",
        )


@agent_router.post("/query", response_model=AgentQueryResponse)
def agent_query(
    payload: AgentQueryRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(require_business_api_user),
) -> AgentQueryResponse:
    """Execute a RAG-backed Agent query.

    Returns 503 when ``AGENT_ENABLED=false``.
    """
    if not settings.agent_enabled:
        raise HTTPException(
            status_code=503,
            detail="Agent service is not enabled. Set AGENT_ENABLED=true.",
        )

    _validate_document_in_project(payload.document_id, payload.project_slug, db)
    _assert_session_scope(
        payload.session_id,
        payload.project_slug,
        payload.document_id,
        db,
        current_user.id if current_user else None,
    )

    executor = _build_executor(db, current_user.id) if current_user else _build_executor(db)
    response = executor.execute(_apply_server_constraint_defaults(payload))
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to persist conversation.")
    return response


def _run_executor_in_thread(
    session_factory: sessionmaker,
    payload: AgentQueryRequest,
    owner_user_id: str | None,
    event_sink: Callable[[str, dict], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[AgentQueryResponse, str | None]:
    """Run the blocking AgentExecutor in a worker thread with its own session.

    Returns ``(response, persist_error)``.  When *persist_error* is not
    ``None`` the caller should emit a persistence-error SSE event but the
    *response* itself is still valid (steps, final answer, etc.).
    """
    db = session_factory()
    try:
        executor = (
            _build_executor(
                db,
                owner_user_id,
                event_sink=event_sink,
                cancel_event=cancel_event,
            )
            if owner_user_id is not None
            else _build_executor(
                db,
                event_sink=event_sink,
                cancel_event=cancel_event,
            )
        )
        response = executor.execute(payload)
        try:
            db.commit()
        except Exception:
            db.rollback()
            return (response, "Failed to persist conversation/trace.")
        return (response, None)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@agent_router.post("/query/stream")
async def agent_query_stream(
    payload: AgentQueryRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(require_business_api_user),
):
    """Execute a RAG-backed Agent query with SSE streaming.

    Returns ``text/event-stream``.  Events: start, heartbeat, step,
    warning, final, error, done.

    A worker thread publishes live executor/model events through an
    asyncio queue while the async generator handles heartbeats and
    disconnect cancellation.
    """
    if not settings.agent_enabled:
        raise HTTPException(
            status_code=503,
            detail="Agent service is not enabled. Set AGENT_ENABLED=true.",
        )

    _validate_document_in_project(payload.document_id, payload.project_slug, db)
    _assert_session_scope(
        payload.session_id,
        payload.project_slug,
        payload.document_id,
        db,
        current_user.id if current_user else None,
    )

    # Derive a thread-safe session factory from the injected session's
    # bind so tests that override the DB engine still work.
    bind = db.get_bind()
    SessionFactory = sessionmaker(bind=bind, autoflush=False, autocommit=False, future=True)

    async def event_generator():
        cancellation = threading.Event()
        event_queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def publish(event_name: str, data: dict) -> None:
            loop.call_soon_threadsafe(
                event_queue.put_nowait, (event_name, data)
            )

        async def run_worker():
            try:
                return await asyncio.to_thread(
                    _run_executor_in_thread,
                    SessionFactory,
                    _apply_server_constraint_defaults(payload),
                    current_user.id if current_user else None,
                    publish,
                    cancellation,
                )
            finally:
                await event_queue.put(("__worker_done__", {}))

        try:
            yield _sse_event("start", {"status": "processing"})
            yield _sse_heartbeat()

            worker = asyncio.create_task(run_worker())
            disconnected = False
            while True:
                if await request.is_disconnected():
                    disconnected = True
                    cancellation.set()
                    break
                try:
                    event_name, data = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=settings.agent_stream_heartbeat_seconds,
                    )
                except asyncio.TimeoutError:
                    yield _sse_heartbeat()
                    continue
                if event_name == "__worker_done__":
                    break
                yield _sse_event(event_name, data)

            if disconnected:
                cancellation.set()
                try:
                    await asyncio.wait_for(worker, timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                return

            response, persist_error = await worker

            # Emit warnings
            for warning in response.warnings:
                yield _sse_event("warning", {"message": warning})

            # Emit final event with full response shape plus enriched
            # top-level summary fields for frontend convenience.
            final_data = _build_final_event(response)
            yield _sse_event("final", final_data)

            if persist_error:
                yield _sse_event(
                    "error",
                    {
                        "message": persist_error,
                        "error_type": "persistence_error",
                    },
                )
                yield _sse_event("done", {})
                return

            yield _sse_event("done", {})

        except Exception as exc:
            logger.exception("Agent stream failed")
            yield _sse_event(
                "error",
                {"message": str(exc), "error_type": type(exc).__name__},
            )
            yield _sse_event("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@agent_router.get("/traces")
def list_agent_traces(
    session_id: str | None = Query(None, description="Session ID to filter by"),
    project_slug: str | None = Query(None, description="Project slug to filter by"),
    status: str | None = Query(None, description="Trace status to filter by"),
    provider: str | None = Query(None, description="Provider to filter by"),
    route: str | None = Query(None, description="Route to filter by"),
    limit: int = Query(10, ge=1, le=100, description="Max traces to return"),
    offset: int = Query(0, ge=0, description="Number of traces to skip"),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(require_business_api_user),
):
    """List Agent traces with optional filters, newest first.

    When only ``session_id`` is provided, behaviour is backward-compatible
    with older callers.  At least one filter should be specified to avoid
    scanning the full table.
    """
    store = AgentTraceStore(db, owner_user_id=current_user.id if current_user else None)
    traces = store.list_traces(
        session_id=session_id,
        project_slug=project_slug,
        status=status,
        provider=provider,
        route=route,
        limit=limit,
        offset=offset,
    )
    return {"traces": traces, "total": len(traces)}


@agent_router.get("/traces/{trace_id}")
def get_agent_trace(
    trace_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(require_business_api_user),
):
    """Get a single Agent trace by ID, including ordered steps."""
    store = AgentTraceStore(db, owner_user_id=current_user.id if current_user else None)
    trace = store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace


@agent_router.get("/sessions", response_model=list[AgentSessionRead])
def list_agent_sessions(
    project_slug: str = Query(..., description="Project slug to filter by"),
    document_id: str | None = Query(None, description="Document ID to filter by"),
    limit: int = Query(50, ge=1, le=200, description="Max sessions to return"),
    offset: int = Query(0, ge=0, description="Number of sessions to skip"),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(require_business_api_user),
):
    """List Agent conversation sessions, newest first.

    Sessions are read-only here; creation and updates happen through the
    Agent query endpoints.
    """
    _validate_document_in_project(document_id, project_slug, db)
    stmt = select(ConversationSession).order_by(ConversationSession.updated_at.desc())
    stmt = stmt.where(ConversationSession.project_slug == project_slug)
    if current_user is not None:
        stmt = stmt.where(ConversationSession.owner_user_id == current_user.id)
    if document_id is not None:
        stmt = stmt.where(ConversationSession.document_id == document_id)
    else:
        stmt = stmt.where(ConversationSession.document_id.is_(None))
    rows = db.scalars(stmt.offset(offset).limit(limit)).all()
    memory = ConversationMemory(db)
    sessions: list[AgentSessionRead] = []
    for row in rows:
        sessions.append(_build_agent_session_read(row, memory))
    return sessions


@agent_router.get("/sessions/{session_id}/turns", response_model=list[AgentTurnRead])
def get_agent_session_turns(
    session_id: str,
    project_slug: str = Query(..., description="Project slug to scope access"),
    document_id: str | None = Query(None, description="Document ID to scope access"),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(require_business_api_user),
):
    """Return ordered turns for a single Agent conversation session.

    The session must belong to the requested project and exact document
    scope; otherwise a 404 is returned to prevent cross-topic turn
    restoration.
    """
    _validate_document_in_project(document_id, project_slug, db)
    session = db.get(ConversationSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if current_user is not None and session.owner_user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.project_slug != project_slug or session.document_id != document_id:
        raise HTTPException(status_code=404, detail="Session not found.")

    memory = ConversationMemory(db)
    turns = memory.get_history(session_id)
    return [
        AgentTurnRead(
            turn_index=turn.turn_index,
            role=turn.role,
            content=turn.content,
            tool_name=turn.tool_name,
            tool_args=turn.tool_args,
            tool_result=turn.tool_result,
            step_type=turn.step_type,
            citations=turn.citations if isinstance(turn.citations, list) else [],
            created_at=(turn.created_at.isoformat() if turn.created_at else ""),
        )
        for turn in turns
    ]


@agent_router.delete("/sessions/{session_id}")
def delete_agent_session(
    session_id: str,
    project_slug: str = Query(..., description="Project slug to scope access"),
    document_id: str | None = Query(None, description="Document ID to scope access"),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(require_business_api_user),
) -> dict:
    """Hard-delete one Agent conversation session and its associated data."""
    session = db.get(ConversationSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if current_user is not None and session.owner_user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.project_slug != project_slug:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.document_id != document_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    deleted_project_slug = session.project_slug
    deleted_document_id = session.document_id
    deleted_turns = ConversationMemory(db).delete_session(session_id)
    db.commit()
    return {
        "deleted": True,
        "session_id": session_id,
        "project_slug": deleted_project_slug,
        "document_id": deleted_document_id,
        "turns_deleted": deleted_turns,
    }


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_heartbeat() -> str:
    """Emit a stable heartbeat event with a predictable payload shape."""
    return _sse_event(
        "heartbeat",
        {"timestamp": datetime.utcnow().isoformat()},
    )


def _build_agent_session_read(
    row: ConversationSession, memory: ConversationMemory
) -> AgentSessionRead:
    """Build an AgentSessionRead with scope and title information."""
    first_user_turn = memory.first_user_turn(row.id)
    document_title = None
    if row.document_id is not None:
        # Avoid importing Document at module level to keep startup light.
        from app.models.records import Document

        document = row.document_id and memory._db.get(Document, row.document_id)
        document_title = document.title if document is not None else None
    return AgentSessionRead(
        id=row.id,
        project_slug=row.project_slug,
        scope_type="document" if row.document_id is not None else "project",
        document_id=row.document_id,
        document_title=document_title,
        answer_mode=row.answer_mode,
        preview=(first_user_turn.content if first_user_turn else row.id),
        turn_count=memory.turn_count(row.id),
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
        expires_at=row.expires_at.isoformat(),
    )


def _build_final_event(response: AgentQueryResponse) -> dict:
    """Build the enriched final event payload.

    Preserves the full existing response shape (backward compatible) and
    adds stable top-level summary fields so frontends can consume key
    metadata without digging into nested structures.
    """
    final_data = response.model_dump()
    tool_names = sorted({s.tool_name for s in response.steps if s.tool_name})
    step_summary = [
        {
            "step_id": s.step_id,
            "step_type": s.step_type,
            "summary": s.summary,
        }
        for s in response.steps
    ]
    final_data.update({
        "trace_id": response.trace_id,
        "provider": response.answer_provider,
        "model": response.answer_model,
        "tool_names": tool_names,
        "step_summary": step_summary,
    })
    return final_data


# ------------------------------------------------------------------
# Session-scoped temporary attachments
# ------------------------------------------------------------------


def _ensure_session_for_attachments(
    db: Session,
    session_id: str,
    project_slug: str,
    document_id: str | None = None,
    owner_user_id: str | None = None,
) -> ConversationSession:
    """Return existing session or create/touch one for the given project_slug.

    Raises HTTPException when an existing session belongs to a different scope.
    """
    _validate_document_in_project(document_id, project_slug, db)
    existing = db.get(ConversationSession, session_id)
    if existing is not None and owner_user_id is not None and existing.owner_user_id != owner_user_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    if (
        existing is not None
        and (existing.project_slug != project_slug or existing.document_id != document_id)
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session {session_id} is scoped to project {existing.project_slug} "
                f"and document {existing.document_id}; cannot be used for project "
                f"{project_slug} and document {document_id}."
            ),
        )
    memory = ConversationMemory(db, owner_user_id=owner_user_id)
    memory.touch_session(
        session_id,
        project_slug=project_slug,
        ttl_days=settings.agent_conversation_ttl_days,
        document_id=document_id,
    )
    db.flush()
    return db.get(ConversationSession, session_id)


def _assert_session_project_match(
    db: Session,
    session_id: str,
    project_slug: str,
    document_id: str | None = None,
    owner_user_id: str | None = None,
) -> None:
    """Raise 404 if the session does not exist or belongs to a different scope."""
    session = db.get(ConversationSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if owner_user_id is not None and session.owner_user_id != owner_user_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.project_slug != project_slug:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.document_id != document_id:
        raise HTTPException(status_code=404, detail="Session not found.")


def _attachment_to_read(attachment: SessionAttachment) -> AttachmentRead:
    return AttachmentRead(
        id=attachment.id,
        session_id=attachment.session_id,
        project_slug=attachment.session.project_slug,
        file_name=attachment.file_name,
        title=attachment.title,
        sha256=attachment.sha256,
        byte_size=attachment.byte_size,
        status=attachment.status,
        chunk_count=len(attachment.chunks),
        created_at=(attachment.created_at.isoformat() if attachment.created_at else ""),
        updated_at=(attachment.updated_at.isoformat() if attachment.updated_at else ""),
    )


def _chunk_to_read(chunk: SessionAttachmentChunk) -> AttachmentChunkRead:
    return AttachmentChunkRead(
        id=chunk.id,
        ordinal=chunk.ordinal,
        heading=chunk.heading,
        page_label=chunk.page_label,
        text=chunk.text,
        token_estimate=chunk.token_estimate,
    )


@agent_router.post("/sessions/{session_id}/attachments", response_model=AttachmentUploadResponse)
async def upload_session_attachment(
    session_id: str,
    project_slug: str = Query(..., description="Project slug that owns the session"),
    document_id: str | None = Query(None, description="Document ID that scopes the session"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(require_business_api_user),
) -> AttachmentUploadResponse:
    """Upload a temporary attachment scoped to an Agent conversation session.

    Creates/touches the session for the provided project_slug if missing so the
    frontend can attach files before the first message is sent. The uploaded
    file is parsed into chunks and stored in session attachment tables; no
    Project Document row is created.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    project = get_or_create_project(db, project_slug, project_slug)
    _ensure_session_for_attachments(
        db,
        session_id,
        project_slug,
        document_id,
        current_user.id if current_user else None,
    )

    try:
        attachment, chunks = await save_session_attachment(
            db,
            project_id=project.id,
            project_slug=project_slug,
            session_id=session_id,
            upload=file,
        )
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidStoragePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to save session attachment")
        raise HTTPException(status_code=500, detail=f"Failed to process attachment: {exc}") from exc

    db.commit()
    db.refresh(attachment)
    return AttachmentUploadResponse(
        attachment=_attachment_to_read(attachment),
        chunks=[_chunk_to_read(c) for c in chunks],
    )


@agent_router.get("/sessions/{session_id}/attachments", response_model=list[AttachmentRead])
def list_session_attachments_route(
    session_id: str,
    project_slug: str = Query(..., description="Project slug that owns the session"),
    document_id: str | None = Query(None, description="Document ID that scopes the session"),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(require_business_api_user),
) -> list[AttachmentRead]:
    """List temporary attachment summaries for the session/project."""
    _assert_session_project_match(
        db,
        session_id,
        project_slug,
        document_id,
        current_user.id if current_user else None,
    )
    attachments = list_session_attachments(db, session_id, project_slug=project_slug)
    return [_attachment_to_read(a) for a in attachments]


@agent_router.delete("/sessions/{session_id}/attachments/{attachment_id}")
def delete_session_attachment_route(
    session_id: str,
    attachment_id: str,
    project_slug: str = Query(..., description="Project slug that owns the session"),
    document_id: str | None = Query(None, description="Document ID that scopes the session"),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(require_business_api_user),
) -> dict:
    """Delete a session attachment and its chunks, plus best-effort stored file."""
    _assert_session_project_match(
        db,
        session_id,
        project_slug,
        document_id,
        current_user.id if current_user else None,
    )
    attachment = get_session_attachment(db, attachment_id)
    if attachment is None or attachment.session_id != session_id:
        raise HTTPException(status_code=404, detail="Attachment not found.")

    delete_session_attachment(db, attachment)
    db.commit()
    return {"deleted": True, "attachment_id": attachment_id}
