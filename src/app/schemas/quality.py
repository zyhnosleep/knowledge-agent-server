"""质量看板（只读）相关的 Pydantic Schema 定义。

本模块为 ``/api/quality/dashboard`` 质量看板接口定义响应模型，用于汇总从“同一目录下
回环（loop）运行”产物中提取出的质量与性能指标：

- ``AgentMetrics``：从查询 / Agent 产物派生的评估指标（运行缺少相应产物时字段可空）。
- ``FailedCaseSummary`` / ``CommandStatus`` / ``QueryGateSummary`` /
  ``MineruSmokeSummary`` / ``ServiceIngestSummary``：各类单项摘要。
- ``RunSummary``：单次回环运行的整体摘要（不暴露内嵌文件路径）。
- ``QualityDashboardResponse``：看板接口的顶层响应。

Pydantic schemas for the read-only quality dashboard.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentMetrics(BaseModel):
    """从同目录回环产物派生的 Agent / 证据评估指标。

    查询类 / Agent 类产物缺失的运行中，相应字段可为空（None）。

    Agent/Evidence evaluation metrics derived from same-directory loop
    artifacts. Nullable for runs without query/agent artifacts.
    """

    query_total: int | None = None  # 查询总数（可空）
    query_selected: int | None = None  # 被选中 / 纳入评估的查询数（可空）
    query_completed: int | None = None  # 完成执行的查询数（可空）
    query_passed: int | None = None  # 通过的查询数（可空）
    query_failed: int | None = None  # 失败的查询数（可空）
    failure_reason_counts: dict[str, int] = Field(default_factory=dict)  # 按失败原因统计的计数
    likely_stage_counts: dict[str, int] = Field(default_factory=dict)  # 按疑似失败阶段统计的计数
    failed_likely_stage_counts: dict[str, int] = Field(default_factory=dict)  # 失败用例按疑似阶段统计的计数
    unattributed_case_ids: list[str] = Field(default_factory=list)  # 未能归因到阶段的用例 ID 列表
    retrieval_coverage: dict | None = None  # 检索覆盖率指标（可空）
    table_evidence_cases: list[dict] = Field(default_factory=list)  # 表格证据用例列表
    agent_tool_counts: dict[str, int] = Field(default_factory=dict)  # Agent 按工具统计的调用计数
    agent_provider_counts: dict[str, int] = Field(default_factory=dict)  # Agent 按提供商统计的调用计数


class FailedCaseSummary(BaseModel):
    """单个失败基准用例及其归因详情。

    A single failed benchmark case with attribution detail.
    """

    id: str  # 用例 ID
    status: str  # 用例状态
    likely_stage: str  # 疑似失败阶段
    failure_reasons: list[str] = Field(default_factory=list)  # 失败原因列表
    missing_answer_terms: list[str] = Field(default_factory=list)  # 回答中缺失的关键词列表
    missing_citation_terms: list[str] = Field(default_factory=list)  # 引用中缺失的关键词列表
    source_hint_matched: bool | None = None  # 是否匹配到来源提示（可空）


class CommandStatus(BaseModel):
    """单个回环命令结果的紧凑表示。

    Compact representation of a single loop command result.
    """

    name: str  # 命令名称
    exit_code: int | None  # 退出码（可空）
    duration_seconds: float | None  # 耗时（秒，可空）


class QueryGateSummary(BaseModel):
    """查询评估质量门禁的结果。

    The query eval quality gate result.
    """

    enabled: bool  # 门禁是否启用
    passed: bool | None  # 是否通过（可空）
    checks: dict[str, bool] = Field(default_factory=dict)  # 各检查项的结果


class MineruSmokeSummary(BaseModel):
    """MinerU 冒烟测试的紧凑摘要。

    Compact MinerU smoke summary.
    """

    status: str | None = None  # 状态（可空）
    parser_mode: str | None = None  # 解析器模式（可空）
    chunks: int | None = None  # 产出的分块数（可空）
    error: str | None = None  # 错误信息（可空）


class ServiceIngestSummary(BaseModel):
    """服务摄取冒烟测试的紧凑摘要。

    Compact service ingest smoke summary.
    """

    status: str | None = None  # 状态（可空）
    error: str | None = None  # 错误信息（可空）


class RunSummary(BaseModel):
    """单次回环运行的摘要（不暴露内嵌路径）。

    A single loop run summary.  Does not expose embedded paths.
    """

    run_id: str  # 运行 ID
    relative_path: str  # 产物目录的相对路径
    modified_time: str  # 最近修改时间（字符串）
    started_at: str | None = None  # 开始时间（可空）
    finished_at: str | None = None  # 结束时间（可空）
    profile: str | None = None  # 运行 profile 名称（可空）
    overall_status: str | None = None  # 整体状态（可空）
    base_url: str | None = None  # 基准 URL（可空）
    benchmark: str | None = None  # 基准集名称（可空）
    command_statuses: list[CommandStatus] = Field(default_factory=list)  # 各命令的执行结果
    query_summary: dict | None = None  # 查询汇总（可空）
    query_gate: QueryGateSummary | None = None  # 查询质量门禁结果（可空）
    mineru_smoke_summary: MineruSmokeSummary | None = None  # MinerU 冒烟摘要（可空）
    service_ingest_summary: ServiceIngestSummary | None = None  # 服务摄取冒烟摘要（可空）
    failed_cases: list[FailedCaseSummary] = Field(default_factory=list)  # 失败用例列表
    agent_metrics: AgentMetrics | None = None  # Agent 指标（可空）


class QualityDashboardResponse(BaseModel):
    """``GET /api/quality/dashboard`` 的顶层响应体。

    Top-level response for GET /api/quality/dashboard.
    """

    reports_dir: str  # 报告根目录
    runs: list[RunSummary]  # 各次运行的摘要
    skipped_reports: int  # 被跳过的报告数量
    message: str  # 提示消息
