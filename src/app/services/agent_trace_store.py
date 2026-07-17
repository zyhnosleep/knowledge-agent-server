from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models.records import AgentTraceRun, AgentTraceStep
from app.schemas.agent import AgentStep, AgentUsage

logger = logging.getLogger(__name__)


class AgentTraceStore:
    """Persist and retrieve Agent run traces.

    Traces are stored in ``agent_trace_runs`` and ``agent_trace_steps``
    tables. Secrets, API keys, and raw provider responses are never
    persisted.
    """

    def __init__(self, db: Session, owner_user_id: str | None = None) -> None:
        self._db = db
        self._owner_user_id = owner_user_id

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def persist_run(
        self,
        *,
        request_id: str,
        session_id: str,
        project_slug: str,
        query: str,
        constraints: dict[str, Any],
        route: str | None,
        steps: list[AgentStep],
        usage: AgentUsage,
        final_answer: str,
        citations: list[dict[str, Any]],
        warnings: list[str],
        status: str,
        latency_ms: int,
        provider: str = "local",
        model: str = "local-fallback",
    ) -> str:
        """Persist a complete Agent run and its steps. Returns the trace_id."""
        trace_id = str(uuid4())
        now = datetime.utcnow()
        latest_created_at = self._db.scalar(select(func.max(AgentTraceRun.created_at)))
        if latest_created_at is not None and now <= latest_created_at:
            now = latest_created_at + timedelta(microseconds=1)

        run = AgentTraceRun(
            id=trace_id,
            owner_user_id=self._owner_user_id,
            request_id=request_id,
            session_id=session_id,
            project_slug=project_slug,
            query=query,
            constraints=constraints,
            route=route,
            final_answer=final_answer,
            citations=citations,
            warnings=warnings,
            status=status,
            latency_ms=latency_ms,
            provider=provider,
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            tool_calls=usage.tool_calls,
            step_count=usage.steps,
            created_at=now,
        )
        self._db.add(run)

        for step in steps:
            step_row = AgentTraceStep(
                run_id=trace_id,
                step_id=step.step_id,
                step_type=step.step_type,
                summary=step.summary,
                latency_ms=step.latency_ms,
                tool_name=step.tool_name,
                tool_ok=step.tool_ok,
                metadata_json=step.metadata,
            )
            self._db.add(step_row)

        self._db.flush()
        return trace_id

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Return a full trace dict with ordered steps, or None."""
        stmt = select(AgentTraceRun).where(AgentTraceRun.id == trace_id)
        if self._owner_user_id is not None:
            stmt = stmt.where(AgentTraceRun.owner_user_id == self._owner_user_id)
        run = self._db.scalar(stmt)
        if run is None:
            return None
        return _serialize_run(run)

    def list_traces(
        self,
        *,
        session_id: str | None = None,
        project_slug: str | None = None,
        status: str | None = None,
        provider: str | None = None,
        route: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List traces with optional filters, newest first.

        All filter parameters are optional.  When only ``session_id`` is
        provided, behaviour is backward-compatible with callers that
        expected the old signature.

        ``limit`` is clamped to [1, 100]. ``offset`` must be >= 0.
        """
        limit = max(1, min(100, limit))
        offset = max(0, offset)

        stmt = select(AgentTraceRun)
        if self._owner_user_id is not None:
            stmt = stmt.where(AgentTraceRun.owner_user_id == self._owner_user_id)
        if session_id is not None:
            stmt = stmt.where(AgentTraceRun.session_id == session_id)
        if project_slug is not None:
            stmt = stmt.where(AgentTraceRun.project_slug == project_slug)
        if status is not None:
            stmt = stmt.where(AgentTraceRun.status == status)
        if provider is not None:
            stmt = stmt.where(AgentTraceRun.provider == provider)
        if route is not None:
            stmt = stmt.where(AgentTraceRun.route == route)

        stmt = (
            stmt.order_by(AgentTraceRun.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        runs = self._db.scalars(stmt).all()
        return [_serialize_run(r) for r in runs]

    def purge_expired(self, retention_days: int) -> int:
        """Delete traces older than *retention_days*. Returns count deleted."""
        from datetime import timedelta

        cutoff_dt = datetime.utcnow() - timedelta(days=retention_days)

        # Count before delete
        count_result = self._db.execute(
            text("SELECT COUNT(*) FROM agent_trace_runs WHERE created_at < :cutoff"),
            {"cutoff": cutoff_dt},
        )
        count = count_result.scalar() or 0

        if count > 0:
            # Delete steps first, then runs (SQLite cascade only works with PRAGMA foreign_keys=ON)
            self._db.execute(
                text(
                    "DELETE FROM agent_trace_steps WHERE run_id IN "
                    "(SELECT id FROM agent_trace_runs WHERE created_at < :cutoff)"
                ),
                {"cutoff": cutoff_dt},
            )
            self._db.execute(
                text("DELETE FROM agent_trace_runs WHERE created_at < :cutoff"),
                {"cutoff": cutoff_dt},
            )
            self._db.flush()
        return int(count)


def _serialize_run(run: AgentTraceRun) -> dict[str, Any]:
    """Serialize a trace run to a dict safe for API responses.

    Sensitive metadata keys and values (API keys, authorization headers,
    bearer tokens, raw provider responses) are redacted before the dict
    is returned.
    """
    steps = sorted(run.steps, key=lambda s: s.step_id)
    return {
        "trace_id": run.id,
        "request_id": run.request_id,
        "session_id": run.session_id,
        "project_slug": run.project_slug,
        "query": run.query,
        "constraints": _sanitize_dict(run.constraints),
        "route": run.route,
        "final_answer": run.final_answer,
        "citations": run.citations,
        "warnings": run.warnings,
        "status": run.status,
        "latency_ms": run.latency_ms,
        "provider": run.provider,
        "model": run.model,
        "usage": {
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "tool_calls": run.tool_calls,
            "steps": run.step_count,
        },
        "steps": [
            {
                "step_id": s.step_id,
                "step_type": s.step_type,
                "summary": s.summary,
                "latency_ms": s.latency_ms,
                "tool_name": s.tool_name,
                "tool_ok": s.tool_ok,
                "metadata": _sanitize_dict(s.metadata_json),
            }
            for s in steps
        ],
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


# ------------------------------------------------------------------
# sanitization helpers
# ------------------------------------------------------------------

_SENSITIVE_KEY_PATTERNS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "token",
    "secret",
    "password",
    "credential",
    "provider_response",
    "raw_response",
)

_SAFE_METRIC_KEYS = {
    "first_token_ms",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_second",
}


def _sanitize_dict(data: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of *data* with sensitive keys and values redacted."""
    if not data:
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()
        if key_lower in _SAFE_METRIC_KEYS:
            sanitized[key] = value
        elif any(pattern in key_lower for pattern in _SENSITIVE_KEY_PATTERNS):
            sanitized[key] = "[redacted]"
        elif isinstance(value, str) and _looks_like_secret(value):
            sanitized[key] = "[redacted]"
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_dict(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def _looks_like_secret(value: str) -> bool:
    """Heuristic to detect secret-like string values."""
    if len(value) < 8:
        return False
    lower = value.lower()
    for prefix in ("sk-", "bearer ", "basic ", "api-key ", "apikey "):
        if lower.startswith(prefix):
            return True
    # Pattern like "key=value" where value looks like a token
    if "=" in value:
        rhs = value.split("=", 1)[1].strip()
        if len(rhs) >= 16 and not any(c in rhs for c in (" ", "\n", "\t")):
            return True
    return False
