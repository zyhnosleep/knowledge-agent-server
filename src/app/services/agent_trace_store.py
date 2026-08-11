"""Agent 执行 trace 的持久化与检索。

本模块提供 :class:`AgentTraceStore`，负责把一次完整的 Agent 执行过程
（运行记录 run + 逐步记录 step）写入数据库，并支持按多种条件检索。

存储模型：
- ``agent_trace_runs`` 表保存每次执行的总体信息（查询、路由、耗时、
  用量、最终答案、引用、警告、状态等）。
- ``agent_trace_steps`` 表按 step_id 保存执行过程中的逐步明细
  （工具调用、每步耗时、元数据等）。

安全约束：密钥、API Key 以及原始 provider 响应**永不落库**——对外序列化
时会统一脱敏（见 :func:`_sanitize_dict` 与 :func:`_looks_like_secret`）。

删除策略：提供 ``purge_expired`` 按保留天数清理过期 trace；删除时遵循
SQLite 外键级联的约束，先删 step 再删 run。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models.records import AgentTraceRun, AgentTraceStep
from app.schemas.agent import AgentStep, AgentUsage

logger = logging.getLogger(__name__)


class AgentTraceStore:
    """持久化并检索 Agent 执行 trace。

    Persist and retrieve Agent run traces.

    Traces are stored in ``agent_trace_runs`` and ``agent_trace_steps``
    tables. Secrets, API keys, and raw provider responses are never
    persisted.

    对外检索结果一律经过脱敏处理后再返回，防止敏感信息泄露。
    """

    def __init__(self, db: Session, owner_user_id: str | None = None) -> None:
        """初始化存储访问器。

        :param db: 活动的 SQLAlchemy 会话（事务由调用方管理）。
        :param owner_user_id: 数据归属用户 ID。非 None 时，检索 / 列表
            查询会强制加上该用户过滤，实现多租户数据隔离；写入时也会
            把该值记录为 owner_user_id。
        """
        self._db = db
        self._owner_user_id = owner_user_id

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def persist_run(
        self,
        *,
        request_id: str,
        session_id: str,
        project_slug: str,
        query: str,
        constraints: dict[str, Any],
        route: str | None,
        steps: list[AgentStep],
        usage: AgentUsage,
        final_answer: str,
        citations: list[dict[str, Any]],
        warnings: list[str],
        status: str,
        latency_ms: int,
        provider: str = "local",
        model: str = "local-fallback",
    ) -> str:
        """持久化一次完整的 Agent 执行及其逐步记录，返回生成的 trace_id。

        Persist a complete Agent run and its steps. Returns the trace_id.

        :param request_id: 外部请求关联 ID（用于跨服务追踪）。
        :param session_id: 所属会话 ID。
        :param project_slug: 所属项目标识。
        :param query: 用户原始查询。
        :param constraints: 查询约束字典（序列化前会脱敏）。
        :param route: 本次执行采用的路由类型。
        :param steps: 执行过程的逐步记录（每步含工具名、耗时、元数据等）。
        :param usage: token 用量统计（prompt/completion/tool_calls/steps）。
        :param final_answer: 最终生成的答案文本。
        :param citations: 引用列表。
        :param warnings: 执行过程中产生的警告列表。
        :param status: 执行状态（如 success / failed）。
        :param latency_ms: 本次执行的总耗时（毫秒）。
        :param provider: 使用的推理提供方，默认 "local"。
        :param model: 使用的模型名，默认 "local-fallback"。
        :return: 生成的 ``trace_id`` 字符串。
        """
        trace_id = str(uuid4())
        now = datetime.utcnow()
        # 取当前表中最大的 created_at，保证 created_at 严格单调递增。
        # 原因：列表查询按 created_at 倒序分页，若同毫秒/同秒出现相同时间戳，
        # 排序结果不稳定；这里把时间强制推进到 latest + 1 微秒，避免并列时间戳。
        latest_created_at = self._db.scalar(select(func.max(AgentTraceRun.created_at)))
        if latest_created_at is not None and now <= latest_created_at:
            now = latest_created_at + timedelta(microseconds=1)

        run = AgentTraceRun(
            id=trace_id,
            owner_user_id=self._owner_user_id,
            request_id=request_id,
            session_id=session_id,
            project_slug=project_slug,
            query=query,
            constraints=constraints,
            route=route,
            final_answer=final_answer,
            citations=citations,
            warnings=warnings,
            status=status,
            latency_ms=latency_ms,
            provider=provider,
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            tool_calls=usage.tool_calls,
            step_count=usage.steps,
            created_at=now,
        )
        self._db.add(run)

        # 将每个执行步骤作为独立行写入 agent_trace_steps 表，
        # 通过 run_id 外键关联到上面创建的 run 记录。
        for step in steps:
            step_row = AgentTraceStep(
                run_id=trace_id,
                step_id=step.step_id,
                step_type=step.step_type,
                summary=step.summary,
                latency_ms=step.latency_ms,
                tool_name=step.tool_name,
                tool_ok=step.tool_ok,
                metadata_json=step.metadata,
            )
            self._db.add(step_row)

        # flush 仅把 SQL 推送到数据库以生成主键等，不提交事务；
        # 由调用方在合适时机统一 commit / rollback。
        self._db.flush()
        return trace_id

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """按 trace_id 返回包含有序步骤的完整 trace 字典；不存在时返回 None。

        Return a full trace dict with ordered steps, or None.

        :param trace_id: 要查询的 trace ID。
        :return: 脱敏后的 trace 字典，或 ``None``（未找到或无权访问）。
        """
        stmt = select(AgentTraceRun).where(AgentTraceRun.id == trace_id)
        # 若绑定了 owner_user_id，则强制校验归属，防止跨用户越权读取。
        if self._owner_user_id is not None:
            stmt = stmt.where(AgentTraceRun.owner_user_id == self._owner_user_id)
        run = self._db.scalar(stmt)
        if run is None:
            return None
        return _serialize_run(run)

    def list_traces(
        self,
        *,
        session_id: str | None = None,
        project_slug: str | None = None,
        status: str | None = None,
        provider: str | None = None,
        route: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """按可选过滤条件列出 trace，最新在前。

        List traces with optional filters, newest first.

        All filter parameters are optional.  When only ``session_id`` is
        provided, behaviour is backward-compatible with callers that
        expected the old signature.

        ``limit`` is clamped to [1, 100]. ``offset`` must be >= 0.

        :param session_id: 按会话过滤。
        :param project_slug: 按项目过滤。
        :param status: 按执行状态过滤。
        :param provider: 按推理提供方过滤。
        :param route: 按路由类型过滤。
        :param limit: 返回条数，被钳制在 [1, 100]。
        :param offset: 分页偏移量，必须 >= 0。
        :return: 脱敏后的 trace 字典列表。
        """
        # 将 limit / offset 限制到安全区间，避免异常输入。
        limit = max(1, min(100, limit))
        offset = max(0, offset)

        stmt = select(AgentTraceRun)
        # 多租户隔离：绑定了 owner 时无条件追加归属过滤。
        if self._owner_user_id is not None:
            stmt = stmt.where(AgentTraceRun.owner_user_id == self._owner_user_id)
        # 逐个叠加可选的等值过滤条件。
        if session_id is not None:
            stmt = stmt.where(AgentTraceRun.session_id == session_id)
        if project_slug is not None:
            stmt = stmt.where(AgentTraceRun.project_slug == project_slug)
        if status is not None:
            stmt = stmt.where(AgentTraceRun.status == status)
        if provider is not None:
            stmt = stmt.where(AgentTraceRun.provider == provider)
        if route is not None:
            stmt = stmt.where(AgentTraceRun.route == route)

        # 按创建时间倒序 + 分页；结果逐条脱敏后返回。
        stmt = (
            stmt.order_by(AgentTraceRun.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        runs = self._db.scalars(stmt).all()
        return [_serialize_run(r) for r in runs]

    def purge_expired(self, retention_days: int) -> int:
        """删除早于 *retention_days* 的过期 trace，返回删除条数。

        Delete traces older than *retention_days*. Returns count deleted.

        保留策略：只清理 ``created_at`` 早于截止时间的记录，越新的 trace
        越晚被清理。删除分两步——先删 steps 再删 runs，以保证外键一致性
        （见下方行内注释）。
        """
        from datetime import timedelta

        # 计算保留截止时间：当前 UTC 时间往前推 retention_days 天。
        cutoff_dt = datetime.utcnow() - timedelta(days=retention_days)

        # 先统计待删除的 run 数量，便于调用方记录清理规模。
        count_result = self._db.execute(
            text("SELECT COUNT(*) FROM agent_trace_runs WHERE created_at < :cutoff"),
            {"cutoff": cutoff_dt},
        )
        count = count_result.scalar() or 0

        if count > 0:
            # 必须先删 steps 再删 runs：SQLite 的外键级联删除仅在
            # PRAGMA foreign_keys=ON 时生效，不能依赖级联，必须显式
            # 先删除引用过期 run 的 step 记录，避免外键残留。
            self._db.execute(
                text(
                    "DELETE FROM agent_trace_steps WHERE run_id IN "
                    "(SELECT id FROM agent_trace_runs WHERE created_at < :cutoff)"
                ),
                {"cutoff": cutoff_dt},
            )
            self._db.execute(
                text("DELETE FROM agent_trace_runs WHERE created_at < :cutoff"),
                {"cutoff": cutoff_dt},
            )
            self._db.flush()
        return int(count)


def _serialize_run(run: AgentTraceRun) -> dict[str, Any]:
    """把一条 trace run 序列化为可安全返回给 API 的字典。

    Serialize a trace run to a dict safe for API responses.

    Sensitive metadata keys and values (API keys, authorization headers,
    bearer tokens, raw provider responses) are redacted before the dict
    is returned.

    返回字典中的 constraints 与各步骤的 metadata 都会经过脱敏处理；
    steps 按 step_id 升序排序以保证输出顺序稳定。
    """
    steps = sorted(run.steps, key=lambda s: s.step_id)
    return {
        "trace_id": run.id,
        "request_id": run.request_id,
        "session_id": run.session_id,
        "project_slug": run.project_slug,
        "query": run.query,
        "constraints": _sanitize_dict(run.constraints),
        "route": run.route,
        "final_answer": run.final_answer,
        "citations": run.citations,
        "warnings": run.warnings,
        "status": run.status,
        "latency_ms": run.latency_ms,
        "provider": run.provider,
        "model": run.model,
        "usage": {
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "tool_calls": run.tool_calls,
            "steps": run.step_count,
        },
        "steps": [
            {
                "step_id": s.step_id,
                "step_type": s.step_type,
                "summary": s.summary,
                "latency_ms": s.latency_ms,
                "tool_name": s.tool_name,
                "tool_ok": s.tool_ok,
                "metadata": _sanitize_dict(s.metadata_json),
            }
            for s in steps
        ],
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


# ------------------------------------------------------------------
# sanitization helpers
# ------------------------------------------------------------------

# 判定"敏感字段名"的子串模式：命中这些关键词的字典键，其值将被脱敏。
_SENSITIVE_KEY_PATTERNS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "token",
    "secret",
    "password",
    "credential",
    "provider_response",
    "raw_response",
)

# 白名单：这些性能/用量指标键属于安全数据，即使命中敏感模式也原样保留。
_SAFE_METRIC_KEYS = {
    "first_token_ms",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_second",
}


def _sanitize_dict(data: dict[str, Any] | None) -> dict[str, Any]:
    """返回 *data* 的副本，其中的敏感键与敏感值均被脱敏为 "[redacted]"。

    Return a copy of *data* with sensitive keys and values redacted.

    脱敏规则（按优先级）：
    1. 键名命中安全指标白名单（_SAFE_METRIC_KEYS）→ 原样保留；
    2. 键名含敏感模式（_SENSITIVE_KEY_PATTERNS）→ 值置为 "[redacted]"；
    3. 字符串值本身疑似密钥（_looks_like_secret）→ 值置为 "[redacted]"；
    4. 嵌套的 dict / list 递归脱敏；
    5. 其余数据原样保留。
    """
    if not data:
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()
        if key_lower in _SAFE_METRIC_KEYS:
            # 安全指标键：如 token 用量、延迟等，直接放行。
            sanitized[key] = value
        elif any(pattern in key_lower for pattern in _SENSITIVE_KEY_PATTERNS):
            # 键名本身暗示敏感（如 api_key、authorization）→ 脱敏。
            sanitized[key] = "[redacted]"
        elif isinstance(value, str) and _looks_like_secret(value):
            # 值看起来像密钥（sk- 前缀、key=长token 等）→ 脱敏。
            sanitized[key] = "[redacted]"
        elif isinstance(value, dict):
            # 嵌套字典递归处理。
            sanitized[key] = _sanitize_dict(value)
        elif isinstance(value, list):
            # 列表中的字典逐项递归脱敏，非字典元素保持不变。
            sanitized[key] = [
                _sanitize_dict(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def _looks_like_secret(value: str) -> bool:
    """启发式判断字符串值是否疑似密钥/凭证。

    Heuristic to detect secret-like string values.

    命中任一条件即判定为密钥：
    1. 已知前缀：sk- / bearer  / basic  / api-key  / apikey ；
    2. "key=value" 形式，且等号右侧为 >=16 字符、不含空白的连续 token。
    """
    if len(value) < 8:
        return False
    lower = value.lower()
    for prefix in ("sk-", "bearer ", "basic ", "api-key ", "apikey "):
        if lower.startswith(prefix):
            return True
    # 形如 "key=value" 且值段像 token 的模式：
    # 等号右侧需 >=16 个字符且不含空格/换行/制表符，视为疑似密钥。
    if "=" in value:
        rhs = value.split("=", 1)[1].strip()
        if len(rhs) >= 16 and not any(c in rhs for c in (" ", "\n", "\t")):
            return True
    return False
