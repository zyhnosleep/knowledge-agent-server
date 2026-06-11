from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import SessionLocal, init_db
from app.services.repositories import get_or_create_project

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    db = SessionLocal()
    try:
        get_or_create_project(db, settings.default_project_slug, settings.default_project_name)
    finally:
        db.close()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router, prefix="/api")

static_dir = Path(__file__).parent / "static"
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")
app.mount("/wiki", StaticFiles(directory=settings.wiki_dir), name="wiki")


@app.get("/", include_in_schema=False)
def serve_console() -> FileResponse:
    return FileResponse(static_dir / "index.html")
