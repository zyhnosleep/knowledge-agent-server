"""Read-only quality dashboard API routes.

Exposes loop run manifest summaries without starting any long-running work.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.config import get_settings
from app.schemas.quality import QualityDashboardResponse
from app.services.quality_reports import QualityReportsService

quality_router = APIRouter()
settings = get_settings()

DEFAULT_LIMIT = 5
MAX_LIMIT = 20


@quality_router.get("/quality/dashboard", response_model=QualityDashboardResponse)
def quality_dashboard(
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> QualityDashboardResponse:
    """Return read-only quality dashboard summaries for recent loop runs.

    *limit* must be between 1 and 20 (inclusive). Returns HTTP 200 even
    when no reports exist under ``QUALITY_REPORTS_DIR``. This route never
    starts loop scripts or long-running work.
    """
    reports_dir = settings.quality_reports_dir
    service = QualityReportsService(reports_dir=reports_dir)
    result = service.collect_runs(limit=limit)

    runs = result["runs"]
    valid_total = result["valid_total"]
    malformed = result["malformed_count"]

    message = f"Showing {len(runs)} of {valid_total} valid runs."
    if malformed > 0:
        message += f" {malformed} malformed manifest(s) skipped."

    return QualityDashboardResponse(
        reports_dir=str(reports_dir.resolve()),
        runs=runs,
        skipped_reports=malformed,
        message=message,
    )
