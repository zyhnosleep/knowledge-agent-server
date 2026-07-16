from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models.records import ConversationSession, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    """In-memory representation of a conversation turn."""

    turn_index: int
    role: str
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None
    step_type: str | None = None
    citations: list[dict] = field(default_factory=list)
    created_at: datetime | None = None


class ConversationMemory:
    """Per-session conversation memory backed by the ConversationTurn table.

    Stores turn history so the Agent Executor can resume conversations
    and enforce turn budgets.
    """

    def __init__(self, db: Session, owner_user_id: str | None = None) -> None:
        self._db = db
        self._owner_user_id = owner_user_id

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        tool_result: str | None = None,
        step_type: str | None = None,
        citations: list[dict] | None = None,
    ) -> ConversationTurn:
        """Append a turn and return the persisted row."""
        next_index = self._next_turn_index(session_id)
        turn = ConversationTurn(
            session_id=session_id,
            turn_index=next_index,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            step_type=step_type,
            citations=list(citations or []),
        )
        self._db.add(turn)
        self._db.flush()
        return turn

    def get_history(self, session_id: str, *, last_n: int | None = None) -> list[TurnRecord]:
        """Return turns for *session_id* ordered by turn_index ascending.

        When *last_n* is provided only the most recent N turns are
        returned.
        """
        statement = (
            select(ConversationTurn)
            .where(ConversationTurn.session_id == session_id)
            .order_by(ConversationTurn.turn_index.asc())
        )
        rows = self._db.scalars(statement).all()
        if last_n is not None and last_n > 0:
            rows = rows[-last_n:]
        return [
            TurnRecord(
                turn_index=row.turn_index,
                role=row.role,
                content=row.content,
                tool_name=row.tool_name,
                tool_args=row.tool_args,
                tool_result=row.tool_result,
                step_type=row.step_type,
                citations=row.citations if isinstance(row.citations, list) else [],
                created_at=row.created_at,
            )
            for row in rows
        ]

    def turn_count(self, session_id: str) -> int:
        """Return the number of turns for *session_id*."""
        return int(
            self._db.scalar(
                text(
                    "SELECT COUNT(*) FROM conversation_turns WHERE session_id = :sid"
                ),
                {"sid": session_id},
            )
            or 0
        )

    def first_user_turn(self, session_id: str) -> TurnRecord | None:
        """Return the first user turn for *session_id*, if any."""
        statement = (
            select(ConversationTurn)
            .where(
                ConversationTurn.session_id == session_id,
                ConversationTurn.role == "user",
            )
            .order_by(ConversationTurn.turn_index.asc())
            .limit(1)
        )
        row = self._db.scalar(statement)
        if row is None:
            return None
        return TurnRecord(
            turn_index=row.turn_index,
            role=row.role,
            content=row.content,
            tool_name=row.tool_name,
            tool_args=row.tool_args,
            tool_result=row.tool_result,
            step_type=row.step_type,
            citations=row.citations if isinstance(row.citations, list) else [],
            created_at=row.created_at,
        )

    def compact_history(self, session_id: str, max_turns: int) -> int:
        """Delete oldest turns so at most *max_turns* remain.

        Returns the number of deleted rows.
        """
        if max_turns < 0:
            max_turns = 0
        current = self.turn_count(session_id)
        if current <= max_turns:
            return 0
        to_delete = current - max_turns
        self._db.execute(
            text(
                "DELETE FROM conversation_turns WHERE id IN ("
                "  SELECT id FROM conversation_turns"
                "  WHERE session_id = :sid"
                "  ORDER BY turn_index ASC"
                "  LIMIT :limit"
                ")"
            ),
            {"sid": session_id, "limit": to_delete},
        )
        self._db.flush()
        return to_delete

    def delete_session(self, session_id: str) -> int:
        """Remove one session and its stored side data. Returns deleted turn count."""
        from app.services.session_attachments import delete_attachments_for_session

        count = self.turn_count(session_id)
        delete_attachments_for_session(self._db, session_id)
        self._db.execute(
            text(
                "DELETE FROM agent_trace_steps WHERE run_id IN ("
                "  SELECT id FROM agent_trace_runs WHERE session_id = :sid"
                ")"
            ),
            {"sid": session_id},
        )
        self._db.execute(
            text("DELETE FROM agent_trace_runs WHERE session_id = :sid"),
            {"sid": session_id},
        )
        self._db.execute(
            text("DELETE FROM conversation_turns WHERE session_id = :sid"),
            {"sid": session_id},
        )
        self._db.execute(
            text("DELETE FROM conversation_sessions WHERE id = :sid"),
            {"sid": session_id},
        )
        self._db.flush()
        return count

    def delete_sessions_for_document(self, document_id: str) -> int:
        """Remove all sessions scoped to *document_id* and return session count."""
        rows = self._db.execute(
            text("SELECT id FROM conversation_sessions WHERE document_id = :did"),
            {"did": document_id},
        ).all()
        count = 0
        for (session_id,) in rows:
            self.delete_session(session_id)
            count += 1
        self._db.flush()
        return count

    def touch_session(
        self, session_id: str, *, project_slug: str, ttl_days: int, document_id: str | None = None
    ) -> None:
        """Create or update a conversation session with an expiry time.

        Sets ``expires_at = now + ttl_days``.  An existing session is never
        silently rebound to a different project or document scope.
        """
        existing = self._db.get(ConversationSession, session_id)
        if existing is not None:
            if self._owner_user_id is not None and existing.owner_user_id != self._owner_user_id:
                raise ValueError(f"Session {session_id} belongs to another user.")
            if existing.project_slug != project_slug:
                raise ValueError(
                    f"Session {session_id} belongs to project {existing.project_slug}; "
                    f"cannot rebind to project {project_slug}."
                )
            if existing.document_id != document_id:
                raise ValueError(
                    f"Session {session_id} is scoped to document {existing.document_id}; "
                    f"cannot rebind to document {document_id}."
                )
        expires_at = datetime.utcnow() + timedelta(days=ttl_days)
        if existing:
            existing.expires_at = expires_at
            existing.updated_at = datetime.utcnow()
        else:
            session = ConversationSession(
                id=session_id,
                owner_user_id=self._owner_user_id,
                project_slug=project_slug,
                document_id=document_id,
                expires_at=expires_at,
            )
            self._db.add(session)
        self._db.flush()

    def purge_expired_sessions(self) -> int:
        """Delete expired sessions and their conversation turns and attachments.

        Returns the number of sessions deleted.
        """
        now = datetime.utcnow()
        # Find expired session ids
        result = self._db.execute(
            text("SELECT id FROM conversation_sessions WHERE expires_at < :now"),
            {"now": now},
        )
        expired_ids = [row[0] for row in result.fetchall()]
        if not expired_ids:
            return 0

        # Delete turns, attachments, and sessions one-by-one for SQLite compatibility.
        # SQLAlchemy text() does not expand tuples for IN clauses on
        # SQLite, so a batch DELETE WHERE id IN :ids is not portable.
        count = 0
        for sid in expired_ids:
            self.delete_session(sid)
            count += 1
        self._db.flush()
        return count

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _next_turn_index(self, session_id: str) -> int:
        return self.turn_count(session_id)
