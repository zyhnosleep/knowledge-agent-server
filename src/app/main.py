"""应用入口：创建 FastAPI 应用、注册路由、配置中间件与启动钩子。

职责：
1. 组装各业务路由（/api、/api/agent、/api 的 quality、/api/auth）。
2. 注册全局中间件（维护模式拦截非 GET 写请求）。
3. 启动时初始化数据库、默认项目，并清理过期会话与 trace。
4. 挂载静态资源（/assets）与前端控制台页面（/）。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.agent_routes import agent_router
from app.api.auth_routes import auth_router
from app.api.dependencies import require_business_api_user
from app.api.quality_routes import quality_router
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import SessionLocal, init_db
from app.services.agent_trace_store import AgentTraceStore
from app.services.conversation_memory import ConversationMemory
from app.services.repositories import get_or_create_project

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期钩子：应用启动时执行初始化，关闭时清理。

    启动流程：
    1. 配置日志；
    2. 初始化数据库（建表、SQLite 兼容列/索引）；
    3. 确保默认项目存在；
    4. 清理过期的会话记录与旧的 Agent trace。
    """
    configure_logging()
    init_db()
    db = SessionLocal()
    try:
        get_or_create_project(db, settings.default_project_slug, settings.default_project_name)
    finally:
        db.close()

    # ---- 启动清理：清除过期会话与 trace ----
    _startup_purge()

    yield


def _startup_purge() -> None:
    """启动时清理过期会话与旧 trace。

    清理失败只记日志，不阻止应用启动。
    """
    db = SessionLocal()
    try:
        # 清理超过 TTL 的对话会话
        conv = ConversationMemory(db)
        deleted_sessions = conv.purge_expired_sessions()
        db.commit()
        if deleted_sessions:
            logger.info("Startup purge: removed %d expired conversation sessions", deleted_sessions)

        # 清理超过保留期的 Agent trace
        trace = AgentTraceStore(db)
        deleted_traces = trace.purge_expired(settings.agent_trace_retention_days)
        db.commit()
        if deleted_traces:
            logger.info("Startup purge: removed %d expired agent traces", deleted_traces)
    except Exception:
        logger.exception("Startup purge failed; continuing")
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


def _maintenance_blocks(method: str, path: str) -> bool:
    """判断请求是否需要被维护模式拦截。

    只拦截 /api 下的写请求（非 GET/HEAD/OPTIONS），
    放行静态资源和 /api/auth（认证不受维护影响）。
    """
    if method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return False
    if not path.startswith("/api") or path.startswith("/api/auth"):
        return False
    return True


@app.middleware("http")
async def enforce_maintenance_mode(request: Request, call_next):
    """维护模式中间件：开启时对写 API 请求返回 503。"""
    if settings.maintenance_mode_enabled and _maintenance_blocks(
        request.method, request.url.path
    ):
        return JSONResponse(
            status_code=503,
            content={"detail": "Service is in maintenance mode."},
            headers={"Retry-After": "60"},
        )
    return await call_next(request)


# 业务 API 依赖：除了 /api/auth 之外的所有接口都要求登录用户。
business_api_dependencies = [Depends(require_business_api_user)]
# 常规业务路由（文档、项目、检索等）。
app.include_router(router, prefix="/api", dependencies=business_api_dependencies)
# Agent 查询路由（AgentExecutor）。
app.include_router(
    agent_router,
    prefix="/api/agent",
    dependencies=business_api_dependencies,
)
# 质量报告路由。
app.include_router(
    quality_router,
    prefix="/api",
    dependencies=business_api_dependencies,
)
# 认证路由：无需登录。
app.include_router(auth_router, prefix="/api/auth")

static_dir = Path(__file__).parent / "static"
# 静态资源（前端 JS/CSS）。
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")


@app.get("/", include_in_schema=False)
def serve_console() -> FileResponse:
    """根路径返回前端控制台页面。"""
    return FileResponse(static_dir / "index.html")
