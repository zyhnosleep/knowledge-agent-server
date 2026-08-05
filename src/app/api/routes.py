"""核心 REST API 路由模块。

本模块是应用的主 API 面：提供健康检查、项目管理、文档摄取与检索、
规范解析（canonical parse）产物读取与下载、流水线看板、引用定位、
问答、评审（review）等全部业务端点。路由挂载在 ``/api`` 前缀下
（具体前缀由应用启动配置决定）。

API 端点概览：
    系统状态：
        GET /health                     —— 健康检查（模型就绪度与队列快照）。
    项目管理：
        GET    /projects                —— 列出项目（分页）。
        POST   /projects                —— 创建项目（按 slug 幂等）。
        DELETE /projects/{project_slug} —— 删除项目及其全部关联数据（需 confirm）。
    文档摄取与检索：
        POST   /ingest/upload               —— 上传文件触发文档摄取流水线。
        GET    /documents                   —— 列出文档（可按项目过滤）。
        GET    /documents/{id}              —— 获取单个文档。
        DELETE /documents/{id}              —— 删除文档及全部关联资源。
        GET    /documents/{id}/source       —— 文档原始文本/分块视图。
        GET    /documents/{id}/quality      —— 文档质量统计（分块数）。
        GET    /documents/{id}/file         —— 下载/内联查看原始文件。
        GET    /documents/{id}/citations/{chunk_id}/location —— 引用定位。
    规范解析（canonical parse）：
        GET /documents/{id}/parse            —— 活动解析版本状态/质量/进度。
        GET /documents/{id}/parse/markdown   —— 活动解析的规范 Markdown（JSON）。
        GET /documents/{id}/parse/download   —— 流式下载规范 Markdown 快照。
    流水线与运行：
        GET /pipeline/dashboard  —— 项目/文档/运行聚合看板。
        GET /runs                —— 列出流水线运行（可按项目过滤）。
    问答与评审：
        POST /reviews            —— 列出评审项（可按项目过滤）。
        POST /query              —— 执行 RAG 问答。

安全设计要点：
    - 所有"原始文件/规范产物"读取都经过符号链接、真实路径与大小校验，
      防止目录穿越（path traversal）与软链接逃逸（见
      ``_open_regular_file`` / ``_active_canonical_paths``）。
    - 项目删除要求前端显式传入与 slug 一致的 ``confirm_slug`` 参数，
      防止误删。
    - 删除文档/项目时会级联清理向量、问答、评审、主张、运行记录、
      对话会话与轨迹等关联数据。
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import stat
import tempfile
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

# 主路由路由器实例，由应用启动代码挂载。
router = APIRouter()
settings = get_settings()
# 规范产物（manifest / markdown）的读取大小上限，防止超大文件耗尽内存。
_MAX_CANONICAL_MANIFEST_BYTES = 1024 * 1024
_MAX_CANONICAL_MARKDOWN_JSON_BYTES = 10 * 1024 * 1024
# 流式读写文件时的分块大小（1 MiB）。
_FILE_CHUNK_BYTES = 1024 * 1024


@dataclass
class _VerifiedMarkdown:
    """一个通过校验的规范 Markdown 文件句柄及其指纹信息。

    用于在多个函数间传递"已打开且已校验"的规范 Markdown，避免重复打开
    与重复校验。字段含义：
        handle (BinaryIO): 已打开（rb）的 Markdown 文件句柄。
        size (int): 文件字节数。
        expected_sha256 (str): 从解析版本检查点（checkpoint）记录的
            预期 SHA-256，用于后续完整性校验。
        manifest (dict): 解析产物 manifest 的原始 JSON 内容。
    """
    handle: BinaryIO
    size: int
    expected_sha256: str
    manifest: dict


class _SnapshotStreamingResponse(StreamingResponse):
    """流式响应封装：发送完毕后自动关闭底层快照文件句柄。

    用于下载规范 Markdown 快照。由于快照来自 ``SpooledTemporaryFile``，
    在响应流结束后必须关闭，以免句柄泄漏；这里通过重写 ``__call__``
    在 ``finally`` 中兜底关闭。

    参数：
        snapshot (BinaryIO): 要发送的快照文件句柄。
        **kwargs: 透传给 ``StreamingResponse`` 的其它参数
            （media_type、headers 等）。
    """
    def __init__(self, snapshot: BinaryIO, **kwargs) -> None:
        self._snapshot = snapshot
        super().__init__(_stream_open_file(snapshot), **kwargs)

    async def __call__(self, scope, receive, send) -> None:
        send_error: BaseException | None = None

        async def tracked_send(message) -> None:
            nonlocal send_error
            try:
                await send(message)
            except BaseException as exc:
                send_error = exc
                raise

        try:
            await super().__call__(scope, receive, tracked_send)
            if send_error is not None:
                raise send_error
        finally:
            # 无论成功失败，响应结束后都要关闭快照句柄。
            self._snapshot.close()


def _validated_project_slug(project_slug: str) -> str:
    """规范化并校验项目 slug；非法时转为 HTTP 400。

    参数：
        project_slug (str): 待校验的原始 slug。

    返回：
        str: 规范化后的安全 slug。

    异常：
        HTTPException(400): slug 包含非法字符或无法通过安全校验。
    """
    try:
        return safe_project_slug(project_slug)
    except InvalidStoragePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _document_raw_path(document: Document) -> Path:
    """解析文档原始文件的绝对路径（兼容旧的相对路径存储）。

    若文档的 ``raw_path`` 是相对路径，则基于配置的 raw 根目录拼接；
    若是绝对路径且文件不存在，尝试按"项目目录迁移"规则在 raw 根目录下
    重新定位（兼容历史数据的存储布局迁移）。最终结果经过
    ``_ensure_within`` 校验，确保不越出 raw 根目录。

    参数：
        document (Document): 文档记录。

    返回：
        Path: 文档原始文件的绝对路径。

    异常：
        InvalidStoragePathError: 解析结果越出 raw 根目录（目录穿越）。
    """
    raw_base = settings.raw_dir.expanduser().resolve()
    stored_path = Path(document.raw_path)
    candidate = stored_path if stored_path.is_absolute() else raw_base / stored_path
    # 兼容迁移布局：绝对路径不存在时，尝试在 raw/<project>/<filename> 下查找。
    if stored_path.is_absolute() and not candidate.exists():
        project_slug = document.project.slug if document.project is not None else ""
        if stored_path.parent.name == project_slug:
            migrated_candidate = raw_base / project_slug / stored_path.name
            if migrated_candidate.exists():
                candidate = migrated_candidate
    return _ensure_within(candidate, raw_base)


def _document_file_metadata(document: Document) -> dict:
    """构造文档源文件的可对外暴露元信息。

    参数：
        document (Document): 文档记录。

    返回：
        dict: 包含 source_file_available / source_file_url /
            source_file_mime / source_file_name / source_file_is_pdf。
    """
    try:
        path = _document_raw_path(document)
    except InvalidStoragePathError:
        # 路径解析非法时，退化为"文件不可用"的空元信息。
        return _empty_document_file_metadata(document)
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    available = path.exists() and path.is_file()
    source_file_url = None
    # 文件存在时给出可访问的下载 URL（附带项目过滤参数）。
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
    """返回"源文件不可用"的空元信息占位。

    参数：
        document (Document): 文档记录（用于读取 file_name）。

    返回：
        dict: 全为不可用状态的元信息字典。
    """
    return {
        "source_file_available": False,
        "source_file_url": None,
        "source_file_mime": None,
        "source_file_name": document.file_name,
        "source_file_is_pdf": False,
    }


def _document_raw_path_or_404(document: Document) -> Path:
    """解析文档原始文件路径，缺失或非法时返回 404。

    参数：
        document (Document): 文档记录。

    返回：
        Path: 存在且为常规文件的绝对路径。

    异常：
        HTTPException(404): 路径解析失败或文件不存在。
    """
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
    """按项目获取文档及其活动解析版本；缺失时返回 404。

    参数：
        db (Session): 数据库会话。
        document_id (str): 文档 ID。
        project_slug (str): 项目 slug（会被规范化校验）。

    返回：
        tuple[Document, DocumentParseVersion]: (文档, 活动解析版本)。

    异常：
        HTTPException(404): 文档不存在、不属于该项目、没有活动解析版本、
            或活动解析版本记录缺失。
    """
    safe_slug = _validated_project_slug(project_slug)
    document = db.get(Document, document_id)
    # 文档必须存在、属于该项目，且设置了活动解析版本。
    if (
        document is None
        or document.project is None
        or document.project.slug != safe_slug
        or not document.active_parse_version
    ):
        raise HTTPException(status_code=404, detail="Active parse not found.")
    # 按 version_key 精确查找活动解析版本记录。
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
    """以只读方式安全地打开一个"常规文件"，拒绝链接与目录。

    安全要点：
        1. 拒绝符号链接 / 重解析点（reparse point）。
        2. 用 O_NOFOLLOW 打开并校验 fstat 为常规文件（防 TOCTOU）。
        3. 可选的大小上限校验。
        4. 再次用 lstat 比对 (dev, ino)，确认打开前后文件身份未变。

    参数：
        path (Path): 目标文件路径。
        maximum_bytes (int | None): 允许的最大字节数；超过则报错。

    返回：
        BinaryIO: 只读二进制文件句柄。

    异常：
        ValueError: 文件是链接、非常规文件、超限或打开期间身份改变。
    """
    # 先拒绝链接/重解析点，避免打开被劫持的路径。
    if CanonicalArtifactStore._is_link_or_reparse_point(path):  # noqa: SLF001
        raise ValueError(f"verified file cannot be a link: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        # 必须是非常规文件（排除目录、设备等）。
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"verified file is not regular: {path.name}")
        if maximum_bytes is not None and file_stat.st_size > maximum_bytes:
            raise ValueError(f"verified file exceeds size limit: {path.name}")
        # lstat 比对：确认路径此刻指向的文件与已打开的 fd 是同一个
        # （防止打开后被替换成其它文件）。
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
    """计算已打开文件句柄的 SHA-256（从当前位置读取到结尾）。

    参数：
        handle (BinaryIO): 已打开（rb）的文件句柄。

    返回：
        str: 十六进制 SHA-256 摘要。
    """
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(_FILE_CHUNK_BYTES), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _read_markdown_bytes(handle: BinaryIO) -> bytes:
    """读取最多 (上限+1) 字节的 Markdown 内容。

    多读 1 字节用于判断文件是否超过 JSON 视图的大小上限。

    参数：
        handle (BinaryIO): 已打开的 Markdown 文件句柄。

    返回：
        bytes: 读取到的原始字节。
    """
    handle.seek(0)
    return handle.read(_MAX_CANONICAL_MARKDOWN_JSON_BYTES + 1)


def _copy_markdown_snapshot(source: BinaryIO, snapshot: BinaryIO) -> tuple[str, int]:
    """把源文件完整复制到快照文件，同时计算 SHA-256 与字节数。

    参数：
        source (BinaryIO): 源文件句柄（只读）。
        snapshot (BinaryIO): 快照文件句柄（可写）。

    返回：
        tuple[str, int]: (sha256 摘要, 字节数)。
    """
    digest = hashlib.sha256()
    size = 0
    source.seek(0)
    snapshot.seek(0)
    snapshot.truncate(0)
    for chunk in iter(lambda: source.read(_FILE_CHUNK_BYTES), b""):
        digest.update(chunk)
        snapshot.write(chunk)
        size += len(chunk)
    snapshot.seek(0)
    return digest.hexdigest(), size


def _validate_markdown_content(
    verified: _VerifiedMarkdown,
    *,
    actual_sha256: str,
    actual_size: int,
) -> None:
    """校验快照的 SHA-256 与大小是否与已验证信息一致。

    参数：
        verified (_VerifiedMarkdown): 校验基准（预期指纹/大小）。
        actual_sha256 (str): 实际计算出的摘要。
        actual_size (int): 实际读取到的字节数。

    异常：
        ValueError: 大小或摘要与预期不符（说明快照期间文件被改动）。
    """
    if actual_size != verified.size:
        raise ValueError("canonical markdown size changed during snapshot")
    if actual_sha256 != verified.expected_sha256:
        raise ValueError("canonical markdown fingerprint mismatch")


def _snapshot_verified_markdown(verified: _VerifiedMarkdown) -> BinaryIO:
    """把已校验的规范 Markdown 复制到内存/磁盘快照并返回句柄。

    返回的快照用于流式下载，避免后续读取时文件被并发改动造成不一致。
    快照文件在超过内存阈值时自动落盘（SpooledTemporaryFile）。

    参数：
        verified (_VerifiedMarkdown): 已校验的 Markdown 信息。

    返回：
        BinaryIO: 快照文件句柄（游标位于文件头）。

    异常：
        HTTPException(404): 复制或校验过程中发生 OSError / ValueError /
            TypeError（视为规范产物不可用）。
    """
    snapshot = tempfile.SpooledTemporaryFile(
        max_size=_MAX_CANONICAL_MARKDOWN_JSON_BYTES,
        mode="w+b",
    )
    try:
        actual_sha256, actual_size = _copy_markdown_snapshot(
            verified.handle,
            snapshot,
        )
        _validate_markdown_content(
            verified,
            actual_sha256=actual_sha256,
            actual_size=actual_size,
        )
        snapshot.seek(0)
        return snapshot
    except (OSError, ValueError, TypeError) as exc:
        snapshot.close()
        raise HTTPException(status_code=404, detail="Canonical artifact not found.") from exc
    finally:
        # 快照完成后即可关闭源句柄。
        verified.handle.close()


def _stream_open_file(handle: BinaryIO) -> Iterator[bytes]:
    """按固定大小分块读取文件句柄，用于流式响应。

    参数：
        handle (BinaryIO): 已打开的文件句柄。

    返回：
        Iterator[bytes]: 分块字节的迭代器。
    """
    for chunk in iter(lambda: handle.read(_FILE_CHUNK_BYTES), b""):
        yield chunk


def _same_open_file(path: Path, handle: BinaryIO) -> bool:
    """判断路径当前指向的文件是否与已打开句柄是同一文件。

    用于"打开后再校验身份"：若路径已被替换为其它文件（如软链接指向变化），
    则返回 False。

    参数：
        path (Path): 文件路径。
        handle (BinaryIO): 已打开的文件句柄。

    返回：
        bool: 是同一常规文件且路径非链接时为 True。
    """
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
    """解析活动解析版本的 manifest 与 Markdown 产物路径。

    对路径做多层安全校验：文档/版本组件必须通过校验，产物包目录不得是
    链接且不得逃逸文档根目录，解析版本的 artifact 目录同样须位于文档根
    目录内。

    参数：
        document (Document): 文档记录。
        version (DocumentParseVersion): 活动解析版本。

    返回：
        tuple[Path, Path]: (manifest.json 路径, canonical.md 路径)。

    异常：
        ValueError: 任何路径校验不通过（视为产物非法/被篡改）。
    """
    store = CanonicalArtifactStore(settings.canonical_artifacts_dir)
    root = settings.canonical_artifacts_dir.expanduser().resolve()
    # 校验文档 ID 与版本 key 是合法路径组件。
    store._validate_component(document.id)  # noqa: SLF001
    store._validate_component(version.version_key)  # noqa: SLF001
    # 构造文档根目录（不创建），再定位到该版本的产物包目录。
    document_root = store._prepare_document_root(  # noqa: SLF001
        document.id,
        create=False,
    ).resolve()
    bundle = document_root / version.version_key
    # 产物包不能是链接，且解析后必须仍在文档根目录内（防逃逸）。
    if store._is_link_or_reparse_point(bundle):  # noqa: SLF001
        raise ValueError("canonical bundle cannot be a link")
    if not bundle.is_dir() or bundle.resolve().parent != document_root:
        raise ValueError("canonical bundle escapes its document root")

    # artifact_dir 可能是相对路径（基于规范产物根目录）或绝对路径。
    artifact_reference = Path(version.artifact_dir)
    if not artifact_reference.is_absolute():
        artifact_reference = root / artifact_reference
    # artifact 目录同样不得是链接，且必须严格位于文档根目录内，
    # 名字必须是 version_key 或 version_key.pipeline。
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
    """打开活动解析版本的规范 Markdown，并完成身份/指纹校验。

    校验链：
        1. manifest 是合法 JSON 对象。
        2. manifest 中的 document_id / version 与记录一致。
        3. 解析版本检查点（checkpoint）记录的 input_fingerprint 与
           canonical_markdown_sha256 与 manifest 一致（防止产物被替换）。
        4. Markdown 文件本身是常规文件，打开前后身份未变。

    参数：
        document (Document): 文档记录。
        version (DocumentParseVersion): 活动解析版本。

    返回：
        _VerifiedMarkdown: 已验证的 Markdown 信息（句柄 + 指纹 + manifest）。

    异常：
        HTTPException(404): 任一步骤校验失败或文件缺失/损坏。
    """
    markdown_handle: BinaryIO | None = None
    try:
        manifest_path, markdown_path = _active_canonical_paths(document, version)
        # 读取 manifest（限制大小），校验为 JSON 对象。
        with _open_regular_file(
            manifest_path,
            maximum_bytes=_MAX_CANONICAL_MANIFEST_BYTES,
        ) as manifest_handle:
            manifest = json.loads(manifest_handle.read().decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("canonical manifest is not an object")
        # manifest 必须自证其归属：document_id 与 version 与记录一致。
        if (
            manifest.get("document_id") != document.id
            or manifest.get("version") != version.version_key
        ):
            raise ValueError("canonical manifest identity mismatch")

        # 用解析版本检查点里的指纹约束 manifest，防止产物被篡改/替换。
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

        # 打开 Markdown 文件并核对身份（打开后路径未被替换）。
        markdown_handle = _open_regular_file(markdown_path)
        size = os.fstat(markdown_handle.fileno()).st_size
        if not _same_open_file(markdown_path, markdown_handle):
            raise ValueError("canonical markdown changed during verification")
        return _VerifiedMarkdown(
            handle=markdown_handle,
            size=size,
            expected_sha256=expected_markdown,
            manifest=manifest,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        # 任何校验/读取失败都统一视为"规范产物不可用"。
        if markdown_handle is not None:
            markdown_handle.close()
        raise HTTPException(status_code=404, detail="Canonical artifact not found.") from exc


def _parse_progress(version: DocumentParseVersion) -> dict:
    """根据解析版本的 stage_state 推算前端展示的解析进度。

    进度按固定阶段顺序（parse → repair → canonicalize → semantic_split
    → contextualize → embed → index → activate）逐段累计已完成比例。

    参数：
        version (DocumentParseVersion): 解析版本记录。

    返回:
        dict: ``{"stage": str, "percent": int}``。
    """
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
        # 当前阶段正在运行或失败：返回所处阶段并标注失败。
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
    """从解析版本的修复阶段输出中提取涉及的页面索引列表。

    兼容三种记录形态：request.page_index、request.locator.page_index、
    request.page_indices（列表）。结果去重并升序排序。

    参数：
        version (DocumentParseVersion): 解析版本记录。

    返回:
        list[int]: 去重排序后的页面索引列表。
    """
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
        # 记录形态 1：request.page_index。
        page_index = request.get("page_index")
        if isinstance(page_index, int) and not isinstance(page_index, bool) and page_index >= 0:
            pages.add(page_index)
        # 记录形态 2：request.locator.page_index。
        locator = request.get("locator")
        if isinstance(locator, dict):
            locator_page = locator.get("page_index")
            if (
                isinstance(locator_page, int)
                and not isinstance(locator_page, bool)
                and locator_page >= 0
            ):
                pages.add(locator_page)
        # 记录形态 3：request.page_indices（数组）。
        page_indices = request.get("page_indices")
        if isinstance(page_indices, list):
            pages.update(
                page
                for page in page_indices
                if isinstance(page, int) and not isinstance(page, bool) and page >= 0
            )
    return sorted(pages)


def _source_type(document: Document) -> str:
    """根据文档文件名后缀判断源文件类型。

    参数：
        document (Document): 文档记录。

    返回:
        str: "pdf" / "docx" / "html" / "text" 之一。
    """
    suffix = Path(document.file_name or "").suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix in {".html", ".htm"}:
        return "html"
    return "text"


def _source_fragment(source_type: str, spans: list[dict]) -> str:
    """从源跨度（span）中提取 URL 片段定位符（锚点）。

    按源类型生成不同格式的锚点：
        - pdf  → ``#page=N``（页面索引从 1 开始）；
        - docx → ``#paragraph=<id>``；
        - html → ``#element=`` / ``#selector=`` / ``#xpath=``；
        - text → ``#line=N``。

    参数：
        source_type (str): 源文件类型（_source_type 的返回值）。
        spans (list[dict]): 源跨度列表。

    返回:
        str: 定位锚点字符串；未命中时返回空串。
    """
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
            # 按优先级依次尝试 element_id / css_selector / xpath。
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
    """把原始源跨度字典过滤/转换为对外暴露的 PublicSourceSpan。

    无法通过 Pydantic 校验的畸形跨度会被跳过，避免把脏数据泄漏给前端。

    参数：
        spans (list[dict]): 原始源跨度列表。

    返回:
        list[PublicSourceSpan]: 合法跨度对象列表。
    """
    public: list[PublicSourceSpan] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        try:
            public.append(PublicSourceSpan.model_validate(span))
        except ValidationError:
            # 跳过畸形跨度，保证对外输出格式统一。
            continue
    return public


def _delete_document_file(db: Session, document: Document) -> bool:
    """删除文档对应的原始文件（若安全）。

    若该 raw_path 同时被其它文档引用，则跳过删除（共享文件保护）。

    参数：
        db (Session): 数据库会话。
        document (Document): 文档记录。

    返回:
        bool: 是否实际删除了文件。
    """
    try:
        path = _document_raw_path(document)
    except InvalidStoragePathError:
        return False
    if not path.exists() or not path.is_file():
        return False
    # 若有其它文档复用同一 raw_path，则不删除文件（避免误删共享数据）。
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
    """级联删除一个文档及其全部关联资源（供删除文档/项目复用）。

    删除范围：向量、原始文件、引用了该文档的 Agent 轨迹、相关问答记录、
    评审项、主张、流水线运行、对话会话，最后删除文档本身。

    参数：
        db (Session): 数据库会话。
        document (Document): 待删除的文档记录。

    返回:
        dict: 各类资源的删除数量统计。
    """
    document_id = document.id
    project_id = document.project_id
    # 删除向量存储中的文档嵌入。
    get_vector_store(db).delete_document(document_id)
    file_deleted = _delete_document_file(db, document)
    # 删除引用了该文档的 Agent 轨迹。
    trace_runs_deleted = _delete_trace_runs_citing_document(db, document_id)

    # 删除"回答中引用了该文档"的问答记录。
    query_answers_deleted = 0
    query_answers = db.scalars(select(QuestionAnswer).where(QuestionAnswer.project_id == project_id)).all()
    for answer in query_answers:
        citations = answer.citations if isinstance(answer.citations, list) else []
        if any(isinstance(citation, dict) and citation.get("document_id") == document_id for citation in citations):
            db.delete(answer)
            query_answers_deleted += 1

    # 批量删除评审项、主张、流水线运行记录。
    reviews_deleted = db.execute(delete(ReviewItem).where(ReviewItem.document_id == document_id)).rowcount or 0
    claims_deleted = db.execute(delete(Claim).where(Claim.document_id == document_id)).rowcount or 0
    runs_deleted = db.execute(delete(PipelineRun).where(PipelineRun.document_id == document_id)).rowcount or 0
    # 删除该文档的全部对话会话。
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
    """删除属于某项目的全部 Agent 轨迹（含其步骤）。

    参数：
        db (Session): 数据库会话。
        project_slug (str): 项目 slug。

    返回:
        int: 删除的轨迹数量。
    """
    trace_ids = [
        trace_id
        for trace_id in db.scalars(select(AgentTraceRun.id).where(AgentTraceRun.project_slug == project_slug)).all()
    ]
    if not trace_ids:
        return 0
    # 先删步骤，再删轨迹主记录（外键顺序）。
    db.execute(delete(AgentTraceStep).where(AgentTraceStep.run_id.in_(trace_ids)))
    db.execute(delete(AgentTraceRun).where(AgentTraceRun.id.in_(trace_ids)))
    return len(trace_ids)


def _delete_trace_runs_citing_document(db: Session, document_id: str) -> int:
    """删除所有"引用了指定文档"的 Agent 轨迹。

    通过扫描轨迹的 citations 字段判断是否引用该文档。

    参数：
        db (Session): 数据库会话。
        document_id (str): 文档 ID。

    返回:
        int: 删除的轨迹数量。
    """
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
    """把任意值安全转换为 0~100 之间的整数百分比。

    参数：
        value (object): 原始值（可能为非数字/空值）。
        fallback (int): 转换失败时的默认值。

    返回:
        int: 归一化到 [0, 100] 的百分比。
    """
    try:
        percent = int(value)
    except (TypeError, ValueError):
        percent = fallback
    return max(0, min(percent, 100))


def _progress_from_run(run: PipelineRun | None, document: Document | None = None) -> dict:
    """从流水线运行记录（或文档状态）构造进度信息。

    优先取运行记录里 provider_report.progress；缺失时退回文档状态。

    参数：
        run (PipelineRun | None): 运行记录（可为 None）。
        document (Document | None): 文档记录（可为 None）。

    返回:
        dict: ``{"percent": int, "stage": str, "message": str}``。
    """
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
    """根据状态返回默认的进度百分比。

    参数：
        status (str): 文档/运行状态。

    返回:
        int: 该状态对应的默认百分比。
    """
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
    """把内部状态归一化为业务侧状态字符串。

    ready/completed → completed；processing 保持 processing；
    queued/running 透传；failed 透传。

    参数：
        document (Document | None): 文档记录。
        run (PipelineRun | None): 运行记录（优先于文档状态）。

    返回:
        str: 归一化后的业务状态。
    """
    raw_status = run.status if run is not None else (document.status if document is not None else "unknown")
    if raw_status in {"ready", "completed"}:
        return "completed"
    if raw_status in {"pending", "processing", "queued", "running"}:
        return "processing" if raw_status == "processing" else raw_status
    if raw_status == "failed":
        return "failed"
    return raw_status


def _status_label(status: str) -> str:
    """把内部状态转换为人类可读的英文标签。

    参数：
        status (str): 内部状态。

    返回:
        str: 展示用标签；未知状态则按标题化处理。
    """
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
    """构造流水线看板中的"项目主题（topic）"聚合信息。

    统计项目下文档总数、完成/处理中/失败数，汇总平均进度与解析率。

    参数：
        project (Project): 项目记录。
        documents (list[Document]): 该项目下的文档列表。
        latest_runs_by_document (dict[str, PipelineRun]): 每个文档最新的
            运行记录（key 为 document_id）。

    返回:
        dict: 主题聚合信息（含 status / status_label / 各类计数）。
    """
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

    # 汇总状态：失败优先，其次处理中，再其次完成，否则视为空。
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

    # 平均进度（可能为空 → 0）；解析率 = 完成数 / 文档总数。
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
    """构造流水线看板中的"单条运行项"信息。

    运行、文档、项目都可能为 None（对应"孤儿运行"或"无运行文档"），
    各字段都做了空值兜底。

    参数：
        run (PipelineRun | None): 运行记录。
        document (Document | None): 文档记录。
        project (Project | None): 项目记录。

    返回:
        dict: 运行项信息（状态、进度、时间、provider 报告等）。
    """
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
        # 只有文档存在且已完成后才允许"再次操作"（如重新摄取）。
        "action_available": bool(document and status == "completed"),
        "created_at": created_at,
        "updated_at": updated_at,
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查端点：返回模型就绪度与队列快照。

    返回：
        HealthResponse: 包含整体状态、应用名、API 状态、各模型就绪度
            以及队列（模型运行时）快照。
    """
    # 检查各模型的就绪状态。
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
    """列出全部项目（按创建时间倒序，分页）。

    参数：
        limit (int): 最多返回条数（1~200）。
        offset (int): 分页偏移。
        db (Session): 数据库会话。

    返回：
        list[ProjectRead]: 项目摘要列表。
    """
    projects = db.scalars(select(Project).order_by(Project.created_at.desc()).limit(limit).offset(offset)).all()
    return [ProjectRead(id=item.id, slug=item.slug, name=item.name, description=item.description) for item in projects]


@router.post("/projects", response_model=ProjectRead)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> ProjectRead:
    """创建项目（按 slug 幂等），并可选地更新描述。

    参数：
        payload (ProjectCreate): 包含 slug 与 name（description 可选）。
        db (Session): 数据库会话。

    返回：
        ProjectRead: 创建/返回的项目信息。

    异常：
        HTTPException(400): slug 非法。
    """
    # slug 规范化后按 slug 幂等获取或创建项目。
    project = get_or_create_project(db, _validated_project_slug(payload.slug), payload.name)
    # 若提供了 description 且与现有值不同，则更新并提交。
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
    """删除项目及其全部关联数据（需要显式确认 slug）。

    级联清理：项目下所有文档及其资源、全部对话会话、Agent 轨迹、
    评审项、主张、运行记录、问答记录，最后删除项目本身。

    参数：
        project_slug (str): 项目 slug。
        confirm_slug (str): 确认参数，必须与项目 slug 完全一致。
        db (Session): 数据库会话。

    返回：
        dict: 删除结果统计。

    异常：
        HTTPException(400): confirm_slug 与项目 slug 不一致。
        HTTPException(404): 项目不存在。
    """
    safe_slug = _validated_project_slug(project_slug)
    # 防误删：前端必须显式传入与 slug 一致的确认参数。
    if confirm_slug != safe_slug:
        raise HTTPException(status_code=400, detail="confirm_slug must match project_slug.")
    project = db.scalar(select(Project).where(Project.slug == safe_slug))
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    # 逐个删除项目下文档及其关联资源。
    documents = db.scalars(select(Document).where(Document.project_id == project.id)).all()
    document_results = [_delete_document_resources(db, document) for document in documents]
    # 删除项目下的全部对话会话及轮次。
    session_ids = db.scalars(
        select(ConversationSession.id).where(ConversationSession.project_slug == project.slug)
    ).all()
    memory = ConversationMemory(db)
    deleted_turns = 0
    for session_id in session_ids:
        deleted_turns += memory.delete_session(session_id)

    # 批量删除项目下的轨迹、评审、主张、运行与问答记录。
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
    """上传文件并触发文档摄取流水线。

    流程：
        1. 保存上传文件到 raw 存储（含大小/路径安全校验）。
        2. 通过 IngestionPipeline 注册文档（创建项目/文档/运行记录）。
        3. 若运行未立即完成，把摄取任务入队异步执行。

    参数：
        file (UploadFile): 上传的文件。
        project_slug (str): 目标项目 slug（默认取配置）。
        project_name (str): 目标项目名（创建项目时使用）。
        db (Session): 数据库会话。

    返回：
        IngestResponse: 含 document_id / project_id / run_id / status 等。

    异常：
        HTTPException(400): 缺少文件名或 slug 非法。
        HTTPException(413): 文件超过大小限制。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    # 保存上传文件到 raw 目录；按失败类型映射 HTTP 状态码。
    try:
        safe_slug = _validated_project_slug(project_slug)
        saved_path = await save_upload(safe_slug, file)
    except UploadTooLargeError as exc:
        # 文件过大 → 413。
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidStoragePathError as exc:
        # slug/路径非法 → 400。
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # 注册文档与运行记录（若项目不存在会自动创建）。
    pipeline = IngestionPipeline(db)
    project, document, run = pipeline.register_document(safe_slug, project_name, saved_path)
    # 若本次运行未在注册阶段完成，则把任务入队（或同步执行）。
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
    """列出文档（可按项目过滤，按创建时间倒序）。

    参数：
        project_slug (str | None): 可选，按项目过滤。
        limit (int): 最多返回条数（1~200）。
        offset (int): 分页偏移。
        db (Session): 数据库会话。

    返回：
        list[DocumentRead]: 文档摘要列表；项目不存在时返回空列表。
    """
    statement = select(Document).order_by(Document.created_at.desc())
    if project_slug:
        # 按项目过滤；项目不存在时直接返回空列表。
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
    """流水线看板聚合接口：项目主题 + 文档运行项 + 汇总统计。

    一次请求返回三层信息：
        - topics：每个项目的聚合状态（完成/处理中/失败/进度/解析率）。
        - runs：每个文档的最新运行项明细。
        - totals：看板级汇总计数。

    参数：
        project_slug (str | None): 可选，只看某个项目。
        limit (int): 最多返回项目/文档条数（1~200）。
        db (Session): 数据库会话。

    返回：
        dict: ``{"service_status", "topics", "runs", "totals"}``。
    """
    # 查询项目列表（可按 slug 过滤）。
    project_statement = select(Project).order_by(Project.created_at.desc())
    if project_slug:
        project_statement = project_statement.where(Project.slug == _validated_project_slug(project_slug))
    projects = db.scalars(project_statement.limit(limit)).all()
    project_ids = [project.id for project in projects]
    project_map = {project.id: project for project in projects}

    # 查询这些项目下的文档（按更新时间倒序）。
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

    # 每个文档只取最新的一条运行记录。
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

    # 组装主题与运行项。
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
    # 汇总统计各类状态的数量。
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
    """获取单个文档的摘要信息。

    参数：
        document_id (str): 文档 ID。
        db (Session): 数据库会话。

    返回：
        DocumentRead: 文档摘要。

    异常：
        HTTPException(404): 文档不存在。
    """
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
    """获取文档活动解析版本的状态、质量、进度与修复页信息。

    会尝试实际打开规范产物以判断"是否可下载"，失败时（产物缺失/损坏）
    仅把 download_available 置为 False，不影响其余信息返回。

    参数：
        document_id (str): 文档 ID。
        project_slug (str): 项目 slug（必填）。
        db (Session): 数据库会话。

    返回：
        CanonicalParseRead: 解析状态摘要（含 progress / quality /
            repair_pages / warning_count / download_available）。

    异常：
        HTTPException(404): 文档/项目/活动解析版本不匹配。
    """
    document, version = _active_parse_or_404(db, document_id, project_slug)
    download_available = False
    verified_manifest: dict | None = None
    # 尝试打开并校验规范产物：成功则标记可下载并保留 manifest。
    try:
        verified = _open_verified_canonical_markdown(document, version)
        snapshot = _snapshot_verified_markdown(verified)
        snapshot.close()
        verified_manifest = verified.manifest
        download_available = True
    except HTTPException:
        # 产物不可用不阻断本端点，仅标记下载不可用。
        pass
    quality = version.quality_json if isinstance(version.quality_json, dict) else {}
    # 警告数优先取 manifest 里的 warnings；否则回退统计质量 issues。
    manifest_warnings = (
        verified_manifest.get("warnings") if verified_manifest is not None else None
    )
    if isinstance(manifest_warnings, list):
        warning_count = len(manifest_warnings)
    else:
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
    """以 JSON 形式返回活动解析版本的规范 Markdown 内容。

    该端点用于在浏览器中直接查看 Markdown 文本；若内容超过 JSON 视图
    大小上限（10 MiB）则返回 413，建议改用下载端点。

    参数：
        document_id (str): 文档 ID。
        project_slug (str): 项目 slug（必填）。
        db (Session): 数据库会话。

    返回：
        CanonicalMarkdownRead: 包含文档、版本与 Markdown 文本。

    异常：
        HTTPException(404): 活动解析/规范产物不可用。
        HTTPException(413): 内容超过 JSON 视图大小上限。
    """
    document, version = _active_parse_or_404(db, document_id, project_slug)
    verified = _open_verified_canonical_markdown(document, version)
    # 超过 JSON 视图上限则拒绝，避免一次性装载超大文本。
    if verified.size > _MAX_CANONICAL_MARKDOWN_JSON_BYTES:
        verified.handle.close()
        raise HTTPException(status_code=413, detail="Canonical Markdown is too large for JSON view.")
    try:
        content = _read_markdown_bytes(verified.handle)
        # 读取结果超过上限（可能读到了上限+1 字节）同样拒绝。
        if len(content) > _MAX_CANONICAL_MARKDOWN_JSON_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Canonical Markdown is too large for JSON view.",
            )
        # 读取后复核指纹，确保返回内容未被并发改动。
        _validate_markdown_content(
            verified,
            actual_sha256=hashlib.sha256(content).hexdigest(),
            actual_size=len(content),
        )
        markdown = content.decode("utf-8")
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
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
    """流式下载活动解析版本的规范 Markdown 快照文件。

    先把规范 Markdown 复制到内存/磁盘快照（防并发改动），再以流式响应
    发送；发送完成后由 _SnapshotStreamingResponse 自动关闭快照句柄。

    参数：
        document_id (str): 文档 ID。
        project_slug (str): 项目 slug（必填）。
        db (Session): 数据库会话。

    返回：
        StreamingResponse: ``text/markdown`` 的附件下载流。

    异常：
        HTTPException(404): 活动解析/规范产物不可用。
    """
    document, version = _active_parse_or_404(db, document_id, project_slug)
    verified = _open_verified_canonical_markdown(document, version)
    snapshot = _snapshot_verified_markdown(verified)
    return _SnapshotStreamingResponse(
        snapshot,
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
    """获取某个引用块（chunk）在源文件中的定位信息。

    返回源文件 URL（含按源类型构造的锚点片段）以及规范化后的源跨度列表。

    参数：
        document_id (str): 文档 ID。
        chunk_id (str): 引用块（DocumentChunk）ID。
        project_slug (str): 项目 slug（必填）。
        db (Session): 数据库会话。

    返回：
        CitationLocationRead: 引用定位信息。

    异常：
        HTTPException(404): 文档/解析版本不匹配、块不存在或非引用块、
            源文件不可用。
    """
    document, version = _active_parse_or_404(db, document_id, project_slug)
    # 查找属于该文档/解析版本、且非 reference 类型的文本块。
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
    # 规范化源跨度，只暴露允许的字段。
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
        # 源文件 URL + 按源类型定位的锚点片段。
        source_url=source_url + _source_fragment(source_type, public_span_dicts),
        source_spans=public_spans,
    )


@router.get("/documents/{document_id}/file", response_class=FileResponse)
def get_document_file(
    document_id: str,
    project_slug: str = Query(...),
    db: Session = Depends(get_db),
) -> FileResponse:
    """内联返回文档的原始文件。

    校验文档存在且属于指定项目，再以 FileResponse 内联展示。

    参数：
        document_id (str): 文档 ID。
        project_slug (str): 项目 slug（必填）。
        db (Session): 数据库会话。

    返回：
        FileResponse: 原始文件的流式响应（inline 展示）。

    异常：
        HTTPException(404): 文档不存在、不属于该项目、或文件缺失。
    """
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    safe_slug = _validated_project_slug(project_slug)
    # 校验文档归属项目，防止跨项目读取文件。
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
    """删除单个文档及其全部关联资源。

    参数：
        document_id (str): 文档 ID。
        project_slug (str): 项目 slug（必填，校验归属）。
        db (Session): 数据库会话。

    返回：
        dict: ``{"deleted": True, "document_id": ..., **资源统计}``。

    异常：
        HTTPException(404): 文档不存在或不属于该项目。
    """
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    safe_slug = _validated_project_slug(project_slug)
    if document.project is None or document.project.slug != safe_slug:
        raise HTTPException(status_code=404, detail="Document not found.")
    # 级联删除文档的所有关联资源。
    result = _delete_document_resources(db, document)
    db.commit()
    return {
        "deleted": True,
        "document_id": document_id,
        **result,
    }


def _build_document_markdown(document: Document, chunks: list[DocumentChunk], db: Session) -> str:
    """返回文档可用的最完整 Markdown/源文本正文。

    优先使用文档的 ``raw_text``（原始提取文本），否则把全部块文本按
    双换行拼接。

    参数：
        document (Document): 文档记录。
        chunks (list[DocumentChunk]): 文档的分块列表。
        db (Session): 数据库会话。

    返回:
        str: Markdown/源文本正文。
    """
    if document.raw_text:
        return document.raw_text
    return "\n\n".join(chunk.text for chunk in chunks)


@router.get("/documents/{document_id}/source", response_model=dict)
def get_document_source(document_id: str, db: Session = Depends(get_db)) -> dict:
    """返回文档的源文本视图：原始预览、Markdown 正文与分块列表。

    当文档没有分块时，用 raw_text 的前 2400 字符构造一个占位分块，
    保证前端始终能展示内容。

    参数：
        document_id (str): 文档 ID。
        db (Session): 数据库会话。

    返回：
        dict: 文档源信息（含 raw_preview / markdown / chunks /
            source_file_* 元信息）。

    异常：
        HTTPException(404): 文档不存在。
    """
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    # 按序号升序加载文档全部分块。
    chunks = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.ordinal.asc())
    ).all()
    # 把每个分块构造成前端展示项（带顺序编号的标签）。
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
    # 没有分块但有原始文本时，构造一个预览占位分块。
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
    """返回文档的质量统计（当前仅分块数量）。

    参数：
        document_id (str): 文档 ID。
        db (Session): 数据库会话。

    返回：
        dict: ``{"document_id", "status", "chunk_count"}``。

    异常：
        HTTPException(404): 文档不存在。
    """
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    # 统计文档的分块总数。
    chunk_count = db.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == document_id)) or 0
    return {"document_id": document.id, "status": document.status, "chunk_count": int(chunk_count)}


@router.get("/runs", response_model=list[dict])
def list_runs(
    project_slug: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[dict]:
    """列出流水线运行记录（可按项目过滤，按创建时间倒序）。

    参数：
        project_slug (str | None): 可选，按项目过滤。
        limit (int): 最多返回条数（1~200）。
        offset (int): 分页偏移。
        db (Session): 数据库会话。

    返回：
        list[dict]: 运行项列表（含文档/项目名称信息）。

    异常：
        HTTPException(400): slug 非法。
    """
    statement = select(PipelineRun).order_by(PipelineRun.created_at.desc())
    project = None
    if project_slug:
        # 按项目过滤；项目不存在时返回空列表。
        project = db.scalar(select(Project).where(Project.slug == _validated_project_slug(project_slug)))
        if project is None:
            return []
        statement = statement.where(PipelineRun.project_id == project.id)
    runs = db.scalars(statement.limit(limit).offset(offset)).all()

    # 批量预取运行涉及的文档与项目，避免 N+1 查询。
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
    """列出评审项（review items，可按项目过滤，按创建时间倒序）。

    参数：
        project_slug (str | None): 可选，按项目过滤。
        limit (int): 最多返回条数（1~200）。
        offset (int): 分页偏移。
        db (Session): 数据库会话。

    返回：
        list[ReviewItemRead]: 评审项列表；项目不存在时返回空列表。
    """
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
    """执行一次 RAG 问答查询并返回答案。

    参数：
        payload (QueryRequest): 查询请求（项目、问题、是否保存答案、
            可选文档作用域）。
        db (Session): 数据库会话。

    返回：
        QueryResponse: 问答结果。

    异常：
        HTTPException(404): 查询相关资源不存在（由 ValueError 转换）。
    """
    try:
        return QueryService(db).answer(
            payload.project_slug,
            payload.question,
            payload.save_answer,
            document_id=payload.document_id,
        )
    except ValueError as exc:
        # 查询服务以 ValueError 表达"资源未找到"类业务错误，
        # 统一转换为 HTTP 404。
        raise HTTPException(status_code=404, detail=str(exc)) from exc
