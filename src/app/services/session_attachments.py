"""
session_attachments.py —— 会话级临时附件（上传/解析/检索/删除）模块
=================================================================

职责：
- 管理"会话级临时附件"的完整生命周期：上传落盘、解析、入库、
  列表查询、检索证据、删除清理。
- 与项目正式文档（``Project Document``）的区别：会话附件不创建正式的
  ``Document`` 行，只以 ``SessionAttachment`` + ``SessionAttachmentChunk``
  形式存在，供当前会话（对话）中的问答即时引用。

存储布局：
- 文件保存在 ``raw_dir/<project_slug>/__sessions__/<session_id>/`` 下，
  路径经过安全校验（防路径穿越）。
- 解析复用 ``app.services.parser.parse_document``，得到的分块写入
  ``SessionAttachmentChunk`` 表，并估算 token 数。

检索逻辑：
- ``retrieve_session_attachment_evidence`` 用"查询词与分块文本的 token
  重叠计数"做确定性排序；无重叠时退回选取前若干有实质内容的分块。
- 检索被严格限定在 (project_slug, session_id) 范围内，绝不混入其他会话
  的附件。
"""

from __future__ import annotations

import asyncio
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
    _ensure_within,
    _safe_upload_filename,
    compute_sha256,
    safe_project_slug,
)
from app.services.parser import ParsedChunk, ParsedDocument, parse_document

# 模块级日志器与配置单例
logger = logging.getLogger(__name__)
settings = get_settings()


def _parse_attachment_light(path: Path, display_name: str) -> ParsedDocument:
    """轻量解析会话附件（PDF 走 pypdf 文本层，秒级完成）。

    附件只是临时对话参考材料，不需要正式文档的深度解析链路
    （MinerU / Document Intelligence 视觉模型逐页分析）——那条链路
    逐页渲染 PNG 并同步调用视觉模型，分钟级阻塞。文本层提取足够
    支撑附件检索；非 PDF 格式回退到 ``parse_document``。
    """
    from app.services import parser as parser_module

    if path.suffix.lower() == ".pdf":
        page_texts, _page_count, warnings = parser_module._extract_pdf_text_layer_best_effort(
            path
        )
        chunks = [
            ParsedChunk(ordinal=idx, text=text)
            for idx, text in enumerate(page_texts)
            if text and text.strip()
        ]
        return ParsedDocument(
            title=Path(display_name).stem,
            text="\n\n".join(chunk.text for chunk in chunks),
            chunks=chunks,
            metadata={"parser_source": "pypdf_text_layer", "warnings": warnings},
        )
    return parse_document(path)


def session_attachment_dir(project_slug: str, session_id: str) -> Path:
    """Return a safe session-scoped directory for temporary attachment files.

    返回会话附件专用的安全目录（必要时创建），路径为
    ``raw_dir/<project_slug>/__sessions__/<session_id>/``。

    安全校验：
    - 项目 slug 通过 ``safe_project_slug`` 白名单校验。
    - ``session_id`` 不能为空、不能含路径分隔符、不能为 ``.``/``..``。
    - 解析出的真实路径必须以 ``raw_dir/<safe_slug>`` 为前缀，否则视为
      路径逃逸并抛 ``InvalidStoragePathError``。
    """
    safe_slug = safe_project_slug(project_slug)
    # Validate session_id looks like an identifier, not a path traversal string.
    # 校验 session_id 是标识符而非路径穿越字符串
    sid = (session_id or "").strip()
    if not sid or "/" in sid or "\\" in sid or sid in {".", ".."}:
        raise InvalidStoragePathError("Invalid session id.")
    base = settings.raw_dir.expanduser().resolve()
    root = base / safe_slug / "__sessions__" / sid
    root.mkdir(parents=True, exist_ok=True)
    # Ensure the resolved path still lives under raw_dir/safe_slug.
    # 解析符号链接后再次确认仍位于 raw_dir/<safe_slug> 之下，防逃逸
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

    把上传文件保存为会话级临时附件。

    The file is saved under ``raw_dir/<project_slug>/__sessions__/<session_id>/``,
    parsed with ``parse_document()``, and stored as attachment + chunk rows.
    No Project Document row is created.
    文件保存在 ``raw_dir/<project_slug>/__sessions__/<session_id>/`` 下，
    用 ``parse_document()`` 解析后，以"附件 + 分块"两表行持久化。
    不创建任何项目正式文档（Document）记录。

    流程：
    1. 计算会话附件目录；清洗文件名并生成 UUID 前缀文件名。
    2. 分块（1 MiB）写入临时文件 ``.<name>.part``，累计字节数，
       超限抛 ``UploadTooLargeError``；成功后再原子重命名。
    3. 计算 SHA-256 内容哈希，调用 ``parse_document`` 解析文档。
    4. 写入 ``SessionAttachment`` 行（标题取解析标题，否则用文件名），
       ``flush`` 获取主键。
    5. 遍历解析出的分块，逐块写入 ``SessionAttachmentChunk``（含序号、
       标题、页码、文本与 token 估算）。

    返回：``(attachment, chunks)`` 元组。
    """
    # 先计算目标目录（内含安全校验）
    target_dir = session_attachment_dir(project_slug, session_id)
    safe_filename = _safe_upload_filename(upload.filename)
    target_name = f"{uuid4().hex}-{safe_filename}"
    target_path = target_dir / target_name
    partial_path = target_dir / f".{target_name}.part"

    total_bytes = 0
    try:
        # 分块流式写临时文件，统计大小并检查上限
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
        partial_path.replace(target_path)  # 原子重命名，确保文件完整
    except Exception:
        # 失败时清理临时文件
        if partial_path.exists():
            partial_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    # 计算内容哈希并解析文档（解析产物供入库与检索使用）。
    # 解析是同步重活（PDF 文本层提取），放线程池避免阻塞事件循环——
    # 单 worker 下同步解析会把整站请求全部堵死。
    sha256 = compute_sha256(target_path)
    parsed = await asyncio.to_thread(_parse_attachment_light, target_path, safe_filename)

    # 标题优先取解析出的标题，否则退回安全文件名
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
    # flush 以取得附件主键，供分块外键引用

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
    """Return attachments for *session_id*, optionally filtered by project_slug.

    返回某个会话的附件列表；可选地按 ``project_slug`` 过滤。

    实现：先按 ``session_id`` 过滤；若给出 ``project_slug``，则再联表
    ``ConversationSession`` 限定会话属于该项目。结果按创建时间倒序。
    """
    statement = select(SessionAttachment).where(SessionAttachment.session_id == session_id)
    if project_slug is not None:
        # Join ConversationSession to filter by project_slug.
        # 联表 ConversationSession，按项目 slug 过滤
        from app.models.records import ConversationSession

        statement = statement.where(
            SessionAttachment.session_id == ConversationSession.id,
            ConversationSession.project_slug == project_slug,
        )
    statement = statement.order_by(SessionAttachment.created_at.desc())
    return list(db.scalars(statement).all())


def get_session_attachment(db: Session, attachment_id: str) -> SessionAttachment | None:
    """Return a single attachment by id.

    按主键返回单个附件记录；不存在返回 None。
    """
    return db.get(SessionAttachment, attachment_id)


def delete_session_attachment(db: Session, attachment: SessionAttachment) -> None:
    """Delete an attachment, its chunks, and best-effort its stored file.

    删除附件：尽力删除其存储文件（含空目录清理），并删除数据库记录。

    文件删除为"尽力而为"：
    - 校验存储路径位于 raw 根目录内（防路径穿越），越界则跳过并告警。
    - 文件存在且为普通文件才删除。
    - 若文件父目录因此变空，则顺带删除该目录（best-effort）。
    - 删除过程中的异常被捕获并记录日志，不阻断数据库删除。

    最后 ``db.delete`` + ``db.flush`` 删除附件行；分块行因外键级联
    一并删除。
    """
    try:
        raw_base = settings.raw_dir.expanduser().resolve()
        stored_path = Path(attachment.storage_path)
        # 存储路径可能是相对路径，先拼接到 raw 根下再校验
        candidate = stored_path if stored_path.is_absolute() else raw_base / stored_path
        path = _ensure_within(candidate, raw_base)
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)
        # Best-effort: remove parent session dir if empty.
        # 尽力而为：若父目录已空则移除之
        parent = path.parent
        if parent.exists() and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except InvalidStoragePathError:
        logger.warning("Skipping attachment file outside storage root for %s", attachment.id)
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

    在 (project_slug, session_id) 范围内检索会话附件分块，生成证据包。

    Ranking is deterministic: token overlap between query terms and chunk text,
    with a fallback to the first substantive chunks for broad/empty queries.
    Attachments from other sessions are never included.
    排序是确定性的：按查询词与分块文本的 token 重叠计数排序；
    查询宽泛/为空时退回选取前若干有实质内容的分块。
    其他会话的附件绝不会被混入。

    流程：
    1. 联表查询该会话（且属于该项目）的所有附件分块，按 ordinal 升序。
    2. 无任何分块 → 返回 ``EvidencePack(status="empty")``。
    3. 对每个分块计算与查询词的 token 重叠数（交集计数之和）。
    4. 按重叠数降序、ordinal 升序排序。
    5. 有查询词但全为 0 重叠时，退回选取有实质内容（>20 字符）的分块。
    6. 组装前 ``limit`` 条为 ``EvidenceItem`` 列表并返回。
    """
    # 联表：分块 -> 附件 -> 会话，限定会话与项目范围
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

    # 对每个 (分块, 附件) 计算与查询词的 token 重叠分数
    query_terms = _tokenize(query)
    scored: list[tuple[int, SessionAttachmentChunk, SessionAttachment]] = []
    for chunk, attachment in rows:
        chunk_terms = _tokenize(chunk.text)
        if query_terms:
            # 交集计数之和：统计同时出现在查询与分块中的词频
            overlap = sum((Counter(query_terms) & Counter(chunk_terms)).values())
        else:
            overlap = 0
        scored.append((overlap, chunk, attachment))

    # Sort by overlap descending, then by original ordinal ascending.
    # 排序：重叠数降序优先，其次保持原始序号升序（确定性）
    scored.sort(key=lambda x: (-x[0], x[1].ordinal))

    # Fallback: if no overlap, take first substantive chunks.
    # 退回逻辑：有查询词但全部零重叠时，优先取有实质内容的分块
    if query_terms and all(score == 0 for score, _, _ in scored):
        substantive = [row for row in scored if len(row[1].text.strip()) > 20]
        selected = substantive[:limit] if substantive else scored[:limit]
    else:
        selected = scored[:limit]

    # 组装证据项
    items: list[EvidenceItem] = []
    for index, (score, chunk, attachment) in enumerate(selected, start=1):
        items.append(
            EvidenceItem(
                index=index,
                document_id=None,  # 会话附件不关联正式文档
                chunk_id=chunk.id,
                attachment_id=attachment.id,
                page_slug=None,
                page_title=attachment.title or attachment.file_name,
                page_kind="session_attachment",
                page_label=chunk.page_label,
                score=float(score),
                excerpt=chunk.text[:1000],  # 截取前 1000 字符作为摘录
                evidence_kind="session_attachment",
                source_stage="session_attachment",
                support_hint="direct" if score > 0 else "contextual",
            )
        )
    return EvidencePack(status="ok" if items else "empty", items=items)


def delete_attachments_for_session(db: Session, session_id: str) -> int:
    """Delete all attachments (and their files) for a session. Returns count.

    删除某会话的全部附件（含磁盘文件），返回删除的附件数量。
    """
    attachments = list_session_attachments(db, session_id)
    for attachment in attachments:
        delete_session_attachment(db, attachment)
    return len(attachments)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token.

    粗略的 token 数估算：按约每 4 个字符约等于 1 个 token 计算；
    非空文本最少记为 1。用于记录分块规模，辅助上下文预算。
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens from text.

    从文本中提取小写字母数字（含 CJK 一-鿿 区段）的 token 列表，
    用于查询词与分块文本的重叠计数。
    """
    return re.findall(r"[a-z0-9一-鿿]+", text.lower())
