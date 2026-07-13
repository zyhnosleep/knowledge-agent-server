from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.agent_routes import agent_router
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
app.include_router(router, prefix="/api")
app.include_router(agent_router, prefix="/api/agent")
app.include_router(quality_router, prefix="/api")

static_dir = Path(__file__).parent / "static"
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")


@app.get("/", include_in_schema=False)
def serve_console() -> FileResponse:
    return FileResponse(static_dir / "index.html")
