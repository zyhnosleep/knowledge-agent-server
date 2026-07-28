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
    configure_logging()
    init_db()
    db = SessionLocal()
    try:
        get_or_create_project(db, settings.default_project_slug, settings.default_project_name)
    finally:
        db.close()

    # ---- startup cleanup: purge expired sessions and traces ----
    _startup_purge()

    yield


def _startup_purge() -> None:
    """Purge expired conversation sessions and old agent traces on startup.

    Failures are logged and do not prevent the app from starting.
    """
    db = SessionLocal()
    try:
        # Conversation TTL purge
        conv = ConversationMemory(db)
        deleted_sessions = conv.purge_expired_sessions()
        db.commit()
        if deleted_sessions:
            logger.info("Startup purge: removed %d expired conversation sessions", deleted_sessions)

        # Trace retention purge
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
    if method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return False
    if not path.startswith("/api") or path.startswith("/api/auth"):
        return False
    return True


@app.middleware("http")
async def enforce_maintenance_mode(request: Request, call_next):
    if settings.maintenance_mode_enabled and _maintenance_blocks(
        request.method, request.url.path
    ):
        return JSONResponse(
            status_code=503,
            content={"detail": "Service is in maintenance mode."},
            headers={"Retry-After": "60"},
        )
    return await call_next(request)


business_api_dependencies = [Depends(require_business_api_user)]
app.include_router(router, prefix="/api", dependencies=business_api_dependencies)
app.include_router(
    agent_router,
    prefix="/api/agent",
    dependencies=business_api_dependencies,
)
app.include_router(
    quality_router,
    prefix="/api",
    dependencies=business_api_dependencies,
)
app.include_router(auth_router, prefix="/api/auth")

static_dir = Path(__file__).parent / "static"
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")


@app.get("/", include_in_schema=False)
def serve_console() -> FileResponse:
    return FileResponse(static_dir / "index.html")
