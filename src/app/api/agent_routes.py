"""Agent（RAG 智能助手）API 路由模块。

本模块提供基于 RAG 的 Agent 对话查询、会话管理以及会话级临时附件
（session attachment）的 HTTP 端点。Agent 执行核心逻辑封装在
services.agent_executor.AgentExecutor 中，本模块只负责：请求校验、
约束（constraints）默认值回填、SSE 流式输出桥接、以及结果持久化。

API 端点：
    对话与查询：
        POST /agent/query                     —— 同步执行一次 Agent 查询。
        POST /agent/query/stream              —— SSE 流式执行 Agent 查询。
    Agent 追踪（trace）：
        GET  /agent/traces                    —— 按条件列出 Agent 执行轨迹。
        GET  /agent/traces/{trace_id}         —— 获取单条轨迹及其步骤。
    会话（session）：
        GET  /agent/sessions                  —— 列出 Agent 对话会话。
        GET  /agent/sessions/{id}/turns       —— 获取某个会话的对话轮次。
        DELETE /agent/sessions/{id}           —— 硬删除某个会话及其关联数据。
    会话级临时附件：
        POST   /agent/sessions/{id}/attachments                     —— 上传附件。
        GET    /agent/sessions/{id}/attachments                     —— 列出附件。
        DELETE /agent/sessions/{id}/attachments/{attachment_id}     —— 删除附件。

涉及的核心服务：
    - RAGAdapter          —— 向量检索（召回复合检索）。
    - AgentSynthesizer    —— 基于检索结果合成最终答案。
    - ToolRegistry        —— Agent 可用的工具注册表。
    - ConversationMemory  —— 会话记忆（对话轮次的读写）。
    - AgentTraceStore     —— Agent 执行轨迹的持久化。
    - SessionAttachment   —— 会话级临时附件及其分块（chunk）的存储。

设计要点：
    - 认证：通过 ``require_business_api_user`` 依赖注入当前用户
      （可为 None，表示匿名模式）；匿名时 session 不做用户归属校验。
    - 权限/作用域：所有会话操作都会校验 session 的 project_slug /
      document_id 归属，防止跨项目、跨主题的会话被复用。
    - 约束默认值：客户端未显式指定的 Agent 约束（如超时、步数）会回填
      服务端 .env 配置（如 AGENT_TIMEOUT_SECONDS），保证配置真正生效。
    - SSE 流式：查询在工作线程中执行，事件（start/heartbeat/step/
      warning/final/error/done）通过 asyncio 队列桥接到响应流。
"""

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

# Agent 路由专用路由器实例，由应用启动代码挂载。
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
    """组装带全部标准依赖的 AgentExecutor 实例。

    将 RAG 检索、答案合成、工具注册、会话记忆与轨迹存储等组件装配成
    一个可执行的 AgentExecutor，避免在多个路由中重复组装。

    参数：
        db (Session): 数据库会话（用于记忆与轨迹持久化）。
        owner_user_id (str | None): 会话归属用户 ID；为 None 表示匿名。
        event_sink (Callable[[str, dict], None] | None): 可选的事件回调，
            用于把执行器产生的事件推送到外部（如 SSE 队列）。
        cancel_event (threading.Event | None): 可选的取消事件，置位后
            执行器应尽早终止。

    返回：
        AgentExecutor: 装配完成的执行器实例。
    """
    # 构造基础组件：检索适配器、答案合成器、工具注册表。
    rag = RAGAdapter()
    synthesizer = AgentSynthesizer()
    tools = ToolRegistry()
    # 注册内置工具（检索、合成等），供 Agent 在推理过程中调用。
    tools._register_builtins(rag, synthesizer=synthesizer)
    # 会话记忆与轨迹存储都按用户维度隔离。
    memory = ConversationMemory(db, owner_user_id=owner_user_id)
    trace_store = AgentTraceStore(db, owner_user_id=owner_user_id)
    return AgentExecutor(
        rag=rag, tools=tools, memory=memory, db=db,
        trace_store=trace_store, synthesizer=synthesizer,
        event_sink=event_sink, cancel_event=cancel_event,
    )


def _apply_server_constraint_defaults(payload: AgentQueryRequest) -> AgentQueryRequest:
    """当客户端省略约束时，回填服务端 .env 级别的 Agent 默认值。

    背景：Pydantic 在路由收到请求前，就会用 schema 默认值填充客户端
    省略的约束字段。因此无法通过"字段缺失"来判断客户端是否显式指定了
    该值。这里的策略是把 schema 默认值视为"未指定"，凡与该默认值相等的
    字段，就用服务端配置（如 AGENT_TIMEOUT_SECONDS）覆盖，从而让 .env
    调优对 UI 请求真正生效。

    参数：
        payload (AgentQueryRequest): 客户端提交的 Agent 查询请求。

    返回：
        AgentQueryRequest: 约束回填后的请求对象（若无需回填则原样返回）。
    """
    constraints = payload.constraints
    schema_defaults = AgentConstraints()
    updates: dict[str, int | bool] = {}

    # 仅当约束值与 schema 默认值完全一致时才认为"未指定"，回填配置。
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

    # 没有任何需要回填的字段时，直接返回原对象，避免无谓的复制。
    if not updates:
        return payload
    # 通过 model_copy 生成带回填约束的新请求对象（保持不可变语义）。
    return payload.model_copy(update={"constraints": constraints.model_copy(update=updates)})


def _positive_int(value: int, fallback: int) -> int:
    """返回正整数 ``value``；若非正整数则退回 ``fallback``。

    用于把可能为 0/负数的配置值归一化，避免约束出现非法取值。

    参数：
        value (int): 待校验的配置值。
        fallback (int): 校验失败时的默认值。

    返回：
        int: 若 ``value`` 是大于 0 的整数则返回它，否则返回 ``fallback``。
    """
    return value if isinstance(value, int) and value > 0 else fallback


def _assert_session_project(session_id: str | None, project_slug: str, db: Session) -> None:
    """当已有会话属于其它项目时抛出异常（校验项目归属）。

    参数：
        session_id (str | None): 会话 ID；为 None 时直接通过（新建场景）。
        project_slug (str): 请求所在项目。
        db (Session): 数据库会话。

    异常：
        HTTPException(409): 会话已存在且属于其它项目，禁止跨项目复用。
    """
    if not session_id:
        return
    session = db.get(ConversationSession, session_id)
    if session is not None and session.project_slug != project_slug:
        # 会话属于其它项目：返回 409 冲突并说明归属。
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
    """当已有会话被以不兼容的作用域复用（或不属于当前用户）时抛出异常。

    依次校验：用户归属（404）→ 项目归属（409）→ 文档作用域（409），
    防止不同主题、不同项目或他人的会话被错误复用。

    参数：
        session_id (str | None): 会话 ID；为 None 表示新建，直接通过。
        project_slug (str): 请求所在项目。
        document_id (str | None): 请求的文档作用域。
        db (Session): 数据库会话。
        owner_user_id (str | None): 当前用户 ID；为 None（匿名）时跳过
            用户归属校验。

    异常：
        HTTPException(404): 会话不存在或属于其他用户。
        HTTPException(409): 会话属于其它项目或其它文档作用域。
    """
    if not session_id:
        return
    session = db.get(ConversationSession, session_id)
    # 登录用户不能读取/复用他人会话，一律按"未找到"处理，避免信息泄露。
    if owner_user_id is not None and session is not None and session.owner_user_id != owner_user_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    _assert_session_project(session_id, project_slug, db)
    if session is None:
        return
    # 会话的作用域（document_id）必须与请求一致，防止跨主题恢复对话。
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
    """校验文档存在且属于指定项目；否则返回 404。

    参数：
        document_id (str | None): 文档 ID；为 None 时直接通过。
        project_slug (str): 期望所属的项目。
        db (Session): 数据库会话。

    异常：
        HTTPException(404): 文档不存在，或文档属于其它项目。
    """
    if document_id is None:
        return
    document = db.get(Document, document_id)
    # 文档缺失，或其所属项目不匹配时按"未找到"处理。
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
    """同步执行一次 RAG 支撑的 Agent 查询并返回完整结果。

    流程：
        1. 若 ``AGENT_ENABLED=false`` 返回 503。
        2. 校验文档/会话的作用域归属。
        3. 回填服务端约束默认值，在工作线程内执行查询。
        4. 提交会话记忆与轨迹，失败则回滚并返回 500。

    参数：
        payload (AgentQueryRequest): Agent 查询请求体。
        db (Session): 数据库会话（用于会话/记忆/轨迹持久化）。
        current_user (User | None): 当前登录用户，可能为 None（匿名）。

    返回：
        AgentQueryResponse: Agent 查询的完整结果（步骤、最终答案、引用等）。

    异常：
        HTTPException(503): Agent 服务未启用。
        HTTPException(404): 文档或会话作用域校验失败。
        HTTPException(500): 执行结果无法持久化。
    """
    if not settings.agent_enabled:
        raise HTTPException(
            status_code=503,
            detail="Agent service is not enabled. Set AGENT_ENABLED=true.",
        )

    # 校验请求涉及的文档与会话归属，防止跨项目/跨主题操作。
    _validate_document_in_project(payload.document_id, payload.project_slug, db)
    _assert_session_scope(
        payload.session_id,
        payload.project_slug,
        payload.document_id,
        db,
        current_user.id if current_user else None,
    )

    # 组装执行器并同步执行；回填服务端约束默认值。
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
    """在独立工作线程中运行阻塞式 AgentExecutor（使用独立数据库会话）。

    由于 AgentExecutor 执行是阻塞的，且可能持有自己的数据库连接，这里
    通过注入的 ``session_factory`` 创建专属于该线程的会话，避免与请求
    线程共享 Session 导致并发问题。

    参数：
        session_factory (sessionmaker): 线程安全的会话工厂。
        payload (AgentQueryRequest): Agent 查询请求。
        owner_user_id (str | None): 会话归属用户 ID。
        event_sink (Callable | None): 事件回调，透传给执行器。
        cancel_event (threading.Event | None): 取消事件，透传给执行器。

    返回：
        tuple[AgentQueryResponse, str | None]: ``(response, persist_error)``。
            当 ``persist_error`` 不为 None 时，调用方应发出持久化错误事件，
            但 ``response`` 本身（步骤、最终答案等）仍是有效的。

    异常：
        Exception: 执行过程中的异常会回滚后重新抛出（由外层转为 SSE 错误）。
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
            # 持久化失败不影响本次查询结果，但需要向调用方报告。
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
    """以 SSE 流式执行一次 RAG 支撑的 Agent 查询。

    返回 ``text/event-stream`` 流。事件类型包括：start（开始）、
    heartbeat（心跳，防超时断连）、step（执行步骤）、warning（警告）、
    final（最终结果）、error（错误）、done（结束）。

    并发模型：
        - 一个工作线程执行阻塞的 AgentExecutor，通过 asyncio 队列
          （``loop.call_soon_threadsafe`` 投递）把实时事件推给异步生成器。
        - 异步生成器在等待事件的同时处理心跳与客户端断连取消。

    参数：
        payload (AgentQueryRequest): Agent 查询请求。
        request (Request): 当前请求，用于检测客户端是否断开。
        db (Session): 数据库会话（仅用于构建线程安全的会话工厂）。
        current_user (User | None): 当前登录用户，可能为 None。

    返回：
        StreamingResponse: ``text/event-stream`` 的 SSE 响应。

    异常：
        HTTPException(503): Agent 服务未启用。
        HTTPException(404): 文档或会话作用域校验失败。
    """
    if not settings.agent_enabled:
        raise HTTPException(
            status_code=503,
            detail="Agent service is not enabled. Set AGENT_ENABLED=true.",
        )

    # 校验请求涉及的文档与会话归属。
    _validate_document_in_project(payload.document_id, payload.project_slug, db)
    _assert_session_scope(
        payload.session_id,
        payload.project_slug,
        payload.document_id,
        db,
        current_user.id if current_user else None,
    )

    # 从注入会话的 bind 派生出线程安全的会话工厂，这样测试里覆盖了
    # DB 引擎时，工作线程仍能使用被覆盖的引擎。
    bind = db.get_bind()
    SessionFactory = sessionmaker(bind=bind, autoflush=False, autocommit=False, future=True)

    # 事件生成器：驱动工作线程并把事件序列化为 SSE 消息。
    async def event_generator():
        cancellation = threading.Event()
        event_queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # 线程安全的发布函数：工作线程调用，通过 event loop 投递到队列。
        def publish(event_name: str, data: dict) -> None:
            loop.call_soon_threadsafe(
                event_queue.put_nowait, (event_name, data)
            )

        # 把阻塞执行放入线程池，完成后发送哨兵事件标记结束。
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
            # 主循环：轮询事件队列，同时监听客户端断连。
            while True:
                if await request.is_disconnected():
                    # 客户端已断开：置位取消事件，让工作线程尽早停止。
                    disconnected = True
                    cancellation.set()
                    break
                try:
                    event_name, data = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=settings.agent_stream_heartbeat_seconds,
                    )
                except asyncio.TimeoutError:
                    # 超时说明暂时无事件，发心跳保活连接后继续等待。
                    yield _sse_heartbeat()
                    continue
                if event_name == "__worker_done__":
                    break
                yield _sse_event(event_name, data)

            if disconnected:
                # 已断连：再次确认取消，并给工作线程最多 2 秒收尾。
                cancellation.set()
                try:
                    await asyncio.wait_for(worker, timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                return

            response, persist_error = await worker

            # 发出警告事件（若有）。
            for warning in response.warnings:
                yield _sse_event("warning", {"message": warning})

            # 发出最终事件：保留完整响应结构，并补充顶层摘要字段
            # （trace_id、provider、model、工具列表、步骤摘要），
            # 方便前端直接读取关键元数据而无需深入嵌套结构。
            final_data = _build_final_event(response)
            yield _sse_event("final", final_data)

            if persist_error:
                # 持久化失败：发错误事件后仍正常结束流。
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
            # 任何未预期异常：记录日志并转为 SSE 错误事件，确保流能结束。
            logger.exception("Agent stream failed")
            yield _sse_event(
                "error",
                {"message": str(exc), "error_type": type(exc).__name__},
            )
            yield _sse_event("done", {})

    # 返回 SSE 流式响应，禁止代理缓冲以保证事件实时下发。
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
    """按可选过滤条件列出 Agent 执行轨迹（traces），最新优先。

    当只提供 ``session_id`` 时，行为与旧调用方兼容。建议至少指定一个
    过滤条件，避免全表扫描。

    参数：
        session_id/project_slug/status/provider/route (str | None):
            可选的过滤维度，分别按会话、项目、状态、模型提供方、路由筛选。
        limit (int): 最多返回的条数（1~100）。
        offset (int): 跳过的条数（用于分页）。
        db (Session): 数据库会话。
        current_user (User | None): 当前登录用户。

    返回：
        dict: ``{"traces": [...], "total": len(traces)}``。
    """
    # 按当前用户维度构造轨迹存储（匿名时不过滤用户）。
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
    """按 ID 获取单条 Agent 轨迹，包含按顺序排列的步骤。

    参数：
        trace_id (str): 轨迹 ID。
        db (Session): 数据库会话。
        current_user (User | None): 当前登录用户。

    返回：
        dict: 轨迹详情（含 steps 列表）。

    异常：
        HTTPException(404): 轨迹不存在或不属于当前用户。
    """
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
    """列出 Agent 对话会话，最新优先。

    这里的会话是只读视图；会话的创建与更新通过 Agent 查询端点发生。

    参数：
        project_slug (str): 必填，按项目过滤。
        document_id (str | None): 可选，按文档作用域过滤（None 表示
            仅返回项目级会话）。
        limit (int): 最多返回条数（1~200）。
        offset (int): 分页偏移。
        db (Session): 数据库会话。
        current_user (User | None): 当前登录用户（登录时仅返回自己的会话）。

    返回：
        list[AgentSessionRead]: 会话摘要列表。
    """
    _validate_document_in_project(document_id, project_slug, db)
    stmt = select(ConversationSession).order_by(ConversationSession.updated_at.desc())
    stmt = stmt.where(ConversationSession.project_slug == project_slug)
    # 登录用户只能看到自己的会话；匿名用户看到该项目的全部会话。
    if current_user is not None:
        stmt = stmt.where(ConversationSession.owner_user_id == current_user.id)
    # 按文档作用域过滤；未指定 document_id 时只返回项目级会话。
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
    """返回单个 Agent 对话会话的有序轮次（turns）。

    会话必须属于请求中的项目与文档作用域，否则返回 404，以阻止跨主题
    恢复对话轮次（防止把 A 主题的历史对话带到 B 主题）。

    参数：
        session_id (str): 会话 ID。
        project_slug (str): 会话应归属的项目。
        document_id (str | None): 会话应归属的文档作用域。
        db (Session): 数据库会话。
        current_user (User | None): 当前登录用户。

    返回：
        list[AgentTurnRead]: 按顺序排列的对话轮次列表。

    异常：
        HTTPException(404): 会话不存在、不属于当前用户、或作用域不匹配。
    """
    _validate_document_in_project(document_id, project_slug, db)
    session = db.get(ConversationSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    # 用户归属校验：他人会话一律按"未找到"处理。
    if current_user is not None and session.owner_user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")
    # 项目与文档作用域必须完全一致，防止跨主题恢复对话。
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
    """硬删除一个 Agent 对话会话及其关联数据。

    删除会话本身及全部对话轮次；会执行项目/文档作用域与用户归属校验。

    参数：
        session_id (str): 会话 ID。
        project_slug (str): 会话应归属的项目。
        document_id (str | None): 会话应归属的文档作用域。
        db (Session): 数据库会话。
        current_user (User | None): 当前登录用户。

    返回：
        dict: 删除结果摘要，包含被删除的会话/文档/轮次数等。

    异常：
        HTTPException(404): 会话不存在、不属于当前用户、或作用域不匹配。
    """
    session = db.get(ConversationSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    # 用户归属与作用域校验，防止删除他人/其它项目的会话。
    if current_user is not None and session.owner_user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.project_slug != project_slug:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.document_id != document_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    # 先记录被删会话的项目/文档，用于构造返回信息。
    deleted_project_slug = session.project_slug
    deleted_document_id = session.document_id
    # 删除会话及其全部轮次，返回删除的轮次数，然后提交事务。
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
    """格式化一条 Server-Sent Events（SSE）消息。

    参数：
        event (str): 事件名称（如 start / step / final）。
        data (dict): 事件负载，将被序列化为 JSON。

    返回：
        str: 符合 SSE 协议的文本，含事件名与数据块，双空行结尾。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_heartbeat() -> str:
    """生成一条稳定的心跳事件，负载结构可预测。

    用于保持长连接活跃，防止代理/浏览器因长时间无数据而断开。

    返回：
        str: SSE 心跳消息。
    """
    return _sse_event(
        "heartbeat",
        {"timestamp": datetime.utcnow().isoformat()},
    )


def _build_agent_session_read(
    row: ConversationSession, memory: ConversationMemory
) -> AgentSessionRead:
    """把会话记录构造成带作用域与标题信息的 AgentSessionRead。

    参数：
        row (ConversationSession): 数据库会话记录。
        memory (ConversationMemory): 会话记忆服务，用于读取首条用户
            消息与轮次数。

    返回：
        AgentSessionRead: 会话摘要对象（含 scope_type、document_title、
            preview、turn_count 等）。
    """
    # 取首条用户消息作为会话预览；没有则退回使用会话 ID。
    first_user_turn = memory.first_user_turn(row.id)
    document_title = None
    if row.document_id is not None:
        # 延迟导入 Document，避免在模块加载期引入重依赖（保持启动轻量）。
        from app.models.records import Document

        document = row.document_id and memory._db.get(Document, row.document_id)
        document_title = document.title if document is not None else None
    return AgentSessionRead(
        id=row.id,
        project_slug=row.project_slug,
        # 依据是否有文档作用域，判定会话属于"文档级"还是"项目级"。
        scope_type="document" if row.document_id is not None else "project",
        document_id=row.document_id,
        document_title=document_title,
        preview=(first_user_turn.content if first_user_turn else row.id),
        turn_count=memory.turn_count(row.id),
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
        expires_at=row.expires_at.isoformat(),
    )


def _build_final_event(response: AgentQueryResponse) -> dict:
    """构造带富化顶层字段的最终 SSE 事件负载。

    保留现有响应的完整结构（向后兼容），同时新增稳定的顶层摘要字段
    （trace_id、provider、model、工具列表、步骤摘要），方便前端直接
    读取关键元数据，而无需深入嵌套结构。

    参数：
        response (AgentQueryResponse): Agent 查询响应。

    返回:
        dict: 富化后的最终事件负载。
    """
    final_data = response.model_dump()
    # 汇总响应中用到的去重工具名列表（按字典序排序）。
    tool_names = sorted({s.tool_name for s in response.steps if s.tool_name})
    # 提取每一步的轻量摘要（step_id / step_type / summary）。
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
# 会话级临时附件：附件挂在会话上，不生成 Project Document 记录。
# ------------------------------------------------------------------


def _ensure_session_for_attachments(
    db: Session,
    session_id: str,
    project_slug: str,
    document_id: str | None = None,
    owner_user_id: str | None = None,
) -> ConversationSession:
    """返回已有会话；若不存在则为给定项目创建/续期一个会话。

    供"上传附件"使用：前端在发送第一条消息前就可以先上传文件，因此
    这里允许会话尚不存在时自动创建（touch）一个。

    参数：
        db (Session): 数据库会话。
        session_id (str): 会话 ID。
        project_slug (str): 项目。
        document_id (str | None): 文档作用域。
        owner_user_id (str | None): 当前用户 ID（匿名时为 None）。

    返回：
        ConversationSession: 存在（或刚创建）的会话对象。

    异常：
        HTTPException(404): 会话存在但属于其他用户。
        HTTPException(409): 会话已存在但作用域（项目/文档）不匹配。
    """
    _validate_document_in_project(document_id, project_slug, db)
    existing = db.get(ConversationSession, session_id)
    # 用户归属校验：他人会话按"未找到"处理。
    if existing is not None and owner_user_id is not None and existing.owner_user_id != owner_user_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    # 已有会话的作用域（项目/文档）必须与请求一致，否则冲突。
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
    # touch 会话：不存在则创建，存在则更新 TTL/作用域。
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
    """校验会话存在且属于指定作用域（项目/文档/用户），否则抛 404。

    参数：
        db (Session): 数据库会话。
        session_id (str): 会话 ID。
        project_slug (str): 期望的项目。
        document_id (str | None): 期望的文档作用域。
        owner_user_id (str | None): 期望的用户（匿名时跳过）。

    异常：
        HTTPException(404): 会话不存在、不属于当前用户、或作用域不匹配。
    """
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
    """把附件记录转换为对外响应的 AttachmentRead 摘要。

    参数：
        attachment (SessionAttachment): 附件数据库记录。

    返回：
        AttachmentRead: 附件摘要对象（含分块数量、SHA256、时间等）。
    """
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
    """把附件分块记录转换为对外响应的 AttachmentChunkRead。

    参数：
        chunk (SessionAttachmentChunk): 附件分块数据库记录。

    返回：
        AttachmentChunkRead: 分块摘要对象。
    """
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
    """向 Agent 对话会话上传一个临时附件。

    若会话尚不存在，会按 project_slug 创建/续期一个会话，以便前端在
    发送第一条消息前就能先上传文件。上传的文件会被解析成分块存入会话
    附件表，不会生成 Project Document 记录。

    参数：
        session_id (str): 会话 ID。
        project_slug (str): 拥有该会话的项目。
        document_id (str | None): 会话的文档作用域。
        file (UploadFile): 上传的文件。
        db (Session): 数据库会话。
        current_user (User | None): 当前登录用户。

    返回：
        AttachmentUploadResponse: 附件摘要及其分块列表。

    异常：
        HTTPException(400): 缺少文件名，或存储路径非法。
        HTTPException(404): 会话属于其他用户。
        HTTPException(409): 会话作用域不匹配。
        HTTPException(413): 上传文件超过大小限制。
        HTTPException(500): 附件处理失败。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    # 确保项目存在（不存在则自动创建），并确保会话可用于挂载附件。
    project = get_or_create_project(db, project_slug, project_slug)
    _ensure_session_for_attachments(
        db,
        session_id,
        project_slug,
        document_id,
        current_user.id if current_user else None,
    )

    # 保存附件：解析文件为分块并入库；针对不同失败原因映射不同 HTTP 状态码。
    try:
        attachment, chunks = await save_session_attachment(
            db,
            project_id=project.id,
            project_slug=project_slug,
            session_id=session_id,
            upload=file,
        )
    except UploadTooLargeError as exc:
        # 文件过大 → 413（Payload Too Large）。
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidStoragePathError as exc:
        # 存储路径非法 → 400。
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # 其它未知错误 → 500，记录日志便于排查。
        logger.exception("Failed to save session attachment")
        raise HTTPException(status_code=500, detail=f"Failed to process attachment: {exc}") from exc

    # 提交事务并刷新，随后返回附件摘要与分块列表。
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
    """列出会话/项目下的临时附件摘要。

    参数：
        session_id (str): 会话 ID。
        project_slug (str): 会话所属项目。
        document_id (str | None): 会话的文档作用域。
        db (Session): 数据库会话。
        current_user (User | None): 当前登录用户。

    返回：
        list[AttachmentRead]: 附件摘要列表。

    异常：
        HTTPException(404): 会话不存在或作用域/归属不匹配。
    """
    # 先校验会话归属与作用域，再查询附件。
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
    """删除一个会话附件及其分块，并尽力删除已存储的文件。

    参数：
        session_id (str): 会话 ID。
        attachment_id (str): 附件 ID。
        project_slug (str): 会话所属项目。
        document_id (str | None): 会话的文档作用域。
        db (Session): 数据库会话。
        current_user (User | None): 当前登录用户。

    返回：
        dict: ``{"deleted": True, "attachment_id": ...}``。

    异常：
        HTTPException(404): 会话不存在/作用域不匹配，或附件不存在。
    """
    # 先校验会话归属与作用域。
    _assert_session_project_match(
        db,
        session_id,
        project_slug,
        document_id,
        current_user.id if current_user else None,
    )
    attachment = get_session_attachment(db, attachment_id)
    # 附件必须存在且属于当前会话，否则按"未找到"处理。
    if attachment is None or attachment.session_id != session_id:
        raise HTTPException(status_code=404, detail="Attachment not found.")

    delete_session_attachment(db, attachment)
    db.commit()
    return {"deleted": True, "attachment_id": attachment_id}
