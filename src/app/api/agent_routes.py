from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.schemas.agent import AgentConstraints, AgentQueryRequest, AgentQueryResponse
from app.services.agent_executor import AgentExecutor
from app.services.agent_synthesizer import AgentSynthesizer
from app.services.agent_trace_store import AgentTraceStore
from app.services.conversation_memory import ConversationMemory
from app.services.rag_adapter import RAGAdapter
from app.services.tool_registry import ToolRegistry

agent_router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


def _build_executor(db: Session) -> AgentExecutor:
    """Create an AgentExecutor with all standard dependencies."""
    rag = RAGAdapter()
    tools = ToolRegistry()
    tools._register_builtins(rag)
    memory = ConversationMemory(db)
    trace_store = AgentTraceStore(db)
    synthesizer = AgentSynthesizer()
    return AgentExecutor(
        rag=rag, tools=tools, memory=memory, db=db,
        trace_store=trace_store, synthesizer=synthesizer,
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


@agent_router.post("/query", response_model=AgentQueryResponse)
def agent_query(
    payload: AgentQueryRequest, db: Session = Depends(get_db)
) -> AgentQueryResponse:
    """Execute a RAG-backed Agent query.

    Returns 503 when ``AGENT_ENABLED=false``.
    """
    if not settings.agent_enabled:
        raise HTTPException(
            status_code=503,
            detail="Agent service is not enabled. Set AGENT_ENABLED=true.",
        )

    executor = _build_executor(db)
    response = executor.execute(_apply_server_constraint_defaults(payload))
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to persist conversation.")
    return response


@agent_router.post("/query/stream")
async def agent_query_stream(
    payload: AgentQueryRequest, request: Request, db: Session = Depends(get_db)
):
    """Execute a RAG-backed Agent query with SSE streaming.

    Returns ``text/event-stream``.  Events: start, heartbeat, step,
    warning, final, error, done.
    """
    if not settings.agent_enabled:
        raise HTTPException(
            status_code=503,
            detail="Agent service is not enabled. Set AGENT_ENABLED=true.",
        )

    async def event_generator():
        try:
            executor = _build_executor(db)
            yield _sse_event("start", {"status": "processing"})

            # Emit a heartbeat before the synchronous executor call so
            # frontends can confirm the stream is alive even when the
            # executor blocks for a while.
            yield _sse_heartbeat()

            # Run executor synchronously (the executor is synchronous;
            # we stream results step by step after it completes — this
            # is step-level streaming, not token-by-token).
            response = executor.execute(_apply_server_constraint_defaults(payload))

            # Emit each step as it's available
            for step in response.steps:
                yield _sse_event("step", step.model_dump())

            # Emit warnings
            for warning in response.warnings:
                yield _sse_event("warning", {"message": warning})

            # Emit final event with full response shape plus enriched
            # top-level summary fields for frontend convenience.
            final_data = _build_final_event(response)
            yield _sse_event("final", final_data)

            try:
                db.commit()
            except Exception:
                db.rollback()
                yield _sse_event(
                    "error",
                    {
                        "message": "Failed to persist conversation/trace.",
                        "error_type": "persistence_error",
                    },
                )
                yield _sse_event("done", {})
                return

            yield _sse_event("done", {})

        except Exception as exc:
            logger.exception("Agent stream failed")
            try:
                db.rollback()
            except Exception:
                pass
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
):
    """List Agent traces with optional filters, newest first.

    When only ``session_id`` is provided, behaviour is backward-compatible
    with older callers.  At least one filter should be specified to avoid
    scanning the full table.
    """
    store = AgentTraceStore(db)
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
):
    """Get a single Agent trace by ID, including ordered steps."""
    store = AgentTraceStore(db)
    trace = store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_heartbeat() -> str:
    """Emit a stable heartbeat event with a predictable payload shape."""
    return _sse_event(
        "heartbeat",
        {"timestamp": datetime.utcnow().isoformat()},
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
