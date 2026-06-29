"""Pydantic schemas for the read-only quality dashboard."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentMetrics(BaseModel):
    """Agent/Evidence evaluation metrics derived from same-directory loop
    artifacts. Nullable for runs without query/agent artifacts."""

    query_total: int | None = None
    query_selected: int | None = None
    query_completed: int | None = None
    query_passed: int | None = None
    query_failed: int | None = None
    failure_reason_counts: dict[str, int] = Field(default_factory=dict)
    likely_stage_counts: dict[str, int] = Field(default_factory=dict)
    failed_likely_stage_counts: dict[str, int] = Field(default_factory=dict)
    unattributed_case_ids: list[str] = Field(default_factory=list)
    retrieval_coverage: dict | None = None
    table_evidence_cases: list[dict] = Field(default_factory=list)
    agent_tool_counts: dict[str, int] = Field(default_factory=dict)
    agent_provider_counts: dict[str, int] = Field(default_factory=dict)


class FailedCaseSummary(BaseModel):
    """A single failed benchmark case with attribution detail."""

    id: str
    status: str
    likely_stage: str
    failure_reasons: list[str] = Field(default_factory=list)
    missing_answer_terms: list[str] = Field(default_factory=list)
    missing_citation_terms: list[str] = Field(default_factory=list)
    source_hint_matched: bool | None = None


class CommandStatus(BaseModel):
    """Compact representation of a single loop command result."""

    name: str
    exit_code: int | None
    duration_seconds: float | None


class QueryGateSummary(BaseModel):
    """The query eval quality gate result."""

    enabled: bool
    passed: bool | None
    checks: dict[str, bool] = Field(default_factory=dict)


class MineruSmokeSummary(BaseModel):
    """Compact MinerU smoke summary."""

    status: str | None = None
    parser_mode: str | None = None
    chunks: int | None = None
    error: str | None = None


class ServiceIngestSummary(BaseModel):
    """Compact service ingest smoke summary."""

    status: str | None = None
    error: str | None = None


class RunSummary(BaseModel):
    """A single loop run summary.  Does not expose embedded paths."""

    run_id: str
    relative_path: str
    modified_time: str
    started_at: str | None = None
    finished_at: str | None = None
    profile: str | None = None
    overall_status: str | None = None
    base_url: str | None = None
    benchmark: str | None = None
    command_statuses: list[CommandStatus] = Field(default_factory=list)
    query_summary: dict | None = None
    query_gate: QueryGateSummary | None = None
    mineru_smoke_summary: MineruSmokeSummary | None = None
    service_ingest_summary: ServiceIngestSummary | None = None
    failed_cases: list[FailedCaseSummary] = Field(default_factory=list)
    agent_metrics: AgentMetrics | None = None


class QualityDashboardResponse(BaseModel):
    """Top-level response for GET /api/quality/dashboard."""

    reports_dir: str
    runs: list[RunSummary]
    skipped_reports: int
    message: str
