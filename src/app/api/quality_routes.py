"""只读质量看板 API 路由模块。

本模块暴露质量报告看板（quality dashboard）相关的只读端点，仅用于汇总
展示循环运行（loop run）的清单（manifest）摘要信息，不会启动任何耗时的
循环脚本或后台任务。

API 端点：
    GET /quality/dashboard —— 返回最近若干次循环运行的质量摘要。

涉及的核心服务：
    - QualityReportsService：从 ``QUALITY_REPORTS_DIR`` 目录读取并汇总
      各次循环运行的清单信息。
    - settings.quality_reports_dir：报告清单的存储根目录配置。

安全/行为说明：
    - 该路由永远不触发长时运行任务（纯只读）。
    - 即使报告目录下没有任何报告，也返回 HTTP 200 而非报错。
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.config import get_settings
from app.schemas.quality import QualityDashboardResponse
from app.services.quality_reports import QualityReportsService

# 创建独立的路由器实例，后续由应用启动代码挂载到 /api 前缀之下。
quality_router = APIRouter()
settings = get_settings()

# 看板默认/上限的返回条数：默认展示最近 5 次，最多允许 20 次。
DEFAULT_LIMIT = 5
MAX_LIMIT = 20


@quality_router.get("/quality/dashboard", response_model=QualityDashboardResponse)
def quality_dashboard(
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> QualityDashboardResponse:
    """返回最近循环运行的只读质量看板摘要。

    参数：
        limit (int): 最多返回的循环运行条数，介于 1 ~ 20（含）之间，
            超出范围时 FastAPI 会返回 422 校验错误。

    返回：
        QualityDashboardResponse: 包含报告目录、运行摘要列表、
            被跳过的损坏清单数量以及人类可读的提示消息。

    异常：
        不抛异常：即使 ``QUALITY_REPORTS_DIR`` 下没有任何报告，也返回
        HTTP 200（此时 runs 为空列表）。本路由永远不会启动循环脚本或
        其他长时间运行的工作。
    """
    # 读取质量报告清单的存储根目录。
    reports_dir = settings.quality_reports_dir
    # 构造服务并聚合各次循环运行的清单摘要。
    service = QualityReportsService(reports_dir=reports_dir)
    result = service.collect_runs(limit=limit)

    runs = result["runs"]
    valid_total = result["valid_total"]
    malformed = result["malformed_count"]

    # 拼装给前端展示的提示消息：有效运行数量，以及被跳过损坏清单的数量。
    message = f"Showing {len(runs)} of {valid_total} valid runs."
    if malformed > 0:
        message += f" {malformed} malformed manifest(s) skipped."

    return QualityDashboardResponse(
        reports_dir=str(reports_dir.resolve()),
        runs=runs,
        skipped_reports=malformed,
        message=message,
    )
