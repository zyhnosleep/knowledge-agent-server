from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.records import SessionAttachment, SessionAttachmentChunk
from app.schemas.agent import EvidenceItem, EvidencePack
from app.services.filesystem import (
    InvalidStoragePathError,
    UploadTooLargeError,
    _safe_upload_filename,
    compute_sha256,
    safe_project_slug,
)
from app.services.parser import ParsedDocument, parse_document

logger = logging.getLogger(__name__)
settings = get_settings()


def session_attachment_dir(project_slug: str, session_id: str) -> Path:
    """Return a safe session-scoped directory for temporary attachment files."""
    safe_slug = safe_project_slug(project_slug)
    # Validate session_id looks like an identifier, not a path traversal string.
    sid = (session_id or "").strip()
    if not sid or "/" in sid or "\\" in sid or sid in {".", ".."}:
        raise InvalidStoragePathError("Invalid session id.")
    base = settings.raw_dir.expanduser().resolve()
    root = base / safe_slug / "__sessions__" / sid
    root.mkdir(parents=True, exist_ok=True)
    # Ensure the resolved path still lives under raw_dir/safe_slug.
    resolved = root.expanduser().resolve()
    if not str(resolved).startswith(str(base / safe_slug)):
        raise InvalidStoragePathError("Session path escapes storage root.")
    return resolved


async def save_session_attachment(
    db: Session,
    project_id: str,
    project_slug: str,
    session_id: str,
    upload: UploadFile,
) -> tuple[SessionAttachment, list[SessionAttachmentChunk]]:
    """Persist an uploaded file as a session-scoped temporary attachment.

    The file is saved under ``raw_dir/<project_slug>/__sessions__/<session_id>/``,
    parsed with ``parse_document()``, and stored as attachment + chunk rows.
    No Project Document row is created.
    """
    target_dir = session_attachment_dir(project_slug, session_id)
    safe_filename = _safe_upload_filename(upload.filename)
    target_name = f"{uuid4().hex}-{safe_filename}"
    target_path = target_dir / target_name
    partial_path = target_dir / f".{target_name}.part"

    total_bytes = 0
    try:
        with partial_path.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    raise UploadTooLargeError(
                        f"Upload exceeds MAX_UPLOAD_BYTES ({settings.max_upload_bytes})."
                    )
                handle.write(chunk)
        partial_path.replace(target_path)
    except Exception:
        if partial_path.exists():
            partial_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    sha256 = compute_sha256(target_path)
    parsed = parse_document(target_path)

    title = parsed.title if parsed.title and parsed.title.strip() else safe_filename
    attachment = SessionAttachment(
        session_id=session_id,
        project_id=project_id,
        file_name=safe_filename,
        title=title,
        storage_path=str(target_path),
        sha256=sha256,
        byte_size=total_bytes,
        status="ready",
    )
    db.add(attachment)
    db.flush()  # get attachment.id

    chunks: list[SessionAttachmentChunk] = []
    for idx, parsed_chunk in enumerate(parsed.chunks):
        chunk = SessionAttachmentChunk(
            attachment_id=attachment.id,
            ordinal=idx,
            heading=parsed_chunk.heading,
            page_label=parsed_chunk.page_label,
            text=parsed_chunk.text,
            token_estimate=_estimate_tokens(parsed_chunk.text),
        )
        db.add(chunk)
        chunks.append(chunk)
    db.flush()

    return attachment, chunks


def list_session_attachments(
    db: Session, session_id: str, project_slug: str | None = None
) -> list[SessionAttachment]:
    """Return attachments for *session_id*, optionally filtered by project_slug."""
    statement = select(SessionAttachment).where(SessionAttachment.session_id == session_id)
    if project_slug is not None:
        # Join ConversationSession to filter by project_slug.
        from app.models.records import ConversationSession

        statement = statement.where(
            SessionAttachment.session_id == ConversationSession.id,
            ConversationSession.project_slug == project_slug,
        )
    statement = statement.order_by(SessionAttachment.created_at.desc())
    return list(db.scalars(statement).all())


def get_session_attachment(db: Session, attachment_id: str) -> SessionAttachment | None:
    """Return a single attachment by id."""
    return db.get(SessionAttachment, attachment_id)


def delete_session_attachment(db: Session, attachment: SessionAttachment) -> None:
    """Delete an attachment, its chunks, and best-effort its stored file."""
    try:
        path = Path(attachment.storage_path)
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)
        # Best-effort: remove parent session dir if empty.
        parent = path.parent
        if parent.exists() and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except Exception:
        logger.exception("Failed to remove attachment file for %s", attachment.id)
    db.delete(attachment)
    db.flush()


def retrieve_session_attachment_evidence(
    db: Session,
    project_slug: str,
    session_id: str,
    query: str,
    limit: int = 5,
) -> EvidencePack:
    """Retrieve temporary attachment chunks scoped to the session/project.

    Ranking is deterministic: token overlap between query terms and chunk text,
    with a fallback to the first substantive chunks for broad/empty queries.
    Attachments from other sessions are never included.
    """
    from app.models.records import ConversationSession

    statement = (
        select(SessionAttachmentChunk, SessionAttachment)
        .join(SessionAttachment, SessionAttachmentChunk.attachment_id == SessionAttachment.id)
        .join(ConversationSession, SessionAttachment.session_id == ConversationSession.id)
        .where(
            SessionAttachment.session_id == session_id,
            ConversationSession.project_slug == project_slug,
        )
        .order_by(SessionAttachmentChunk.ordinal.asc())
    )
    rows = db.execute(statement).all()
    if not rows:
        return EvidencePack(status="empty", items=[])

    query_terms = _tokenize(query)
    scored: list[tuple[int, SessionAttachmentChunk, SessionAttachment]] = []
    for chunk, attachment in rows:
        chunk_terms = _tokenize(chunk.text)
        if query_terms:
            overlap = sum((Counter(query_terms) & Counter(chunk_terms)).values())
        else:
            overlap = 0
        scored.append((overlap, chunk, attachment))

    # Sort by overlap descending, then by original ordinal ascending.
    scored.sort(key=lambda x: (-x[0], x[1].ordinal))

    # Fallback: if no overlap, take first substantive chunks.
    if query_terms and all(score == 0 for score, _, _ in scored):
        substantive = [row for row in scored if len(row[1].text.strip()) > 20]
        selected = substantive[:limit] if substantive else scored[:limit]
    else:
        selected = scored[:limit]

    items: list[EvidenceItem] = []
    for index, (score, chunk, attachment) in enumerate(selected, start=1):
        items.append(
            EvidenceItem(
                index=index,
                document_id=None,
                chunk_id=chunk.id,
                page_slug=None,
                page_title=attachment.title or attachment.file_name,
                page_kind="session_attachment",
                page_label=chunk.page_label,
                score=float(score),
                excerpt=chunk.text[:1000],
                evidence_kind="session_attachment",
                source_stage="session_attachment",
                support_hint="direct" if score > 0 else "contextual",
            )
        )
    return EvidencePack(status="ok" if items else "empty", items=items)


def delete_attachments_for_session(db: Session, session_id: str) -> int:
    """Delete all attachments (and their files) for a session. Returns count."""
    attachments = list_session_attachments(db, session_id)
    for attachment in attachments:
        delete_session_attachment(db, attachment)
    return len(attachments)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens from text."""
    return re.findall(r"[a-z0-9一-鿿]+", text.lower())
