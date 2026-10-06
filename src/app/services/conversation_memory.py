"""多轮会话记忆：基于数据库表为每个会话持久化对话历史。

本模块提供 :class:`ConversationMemory` 与轻量数据结构 :class:`TurnRecord`，
负责：

- 追加/读取单个会话（session_id）的对话轮次（turn），支持 Agent 执行器
  在多轮对话间恢复上下文；
- 维护会话的 TTL 过期时间（``expires_at = now + ttl_days``），并清理
  过期会话及其轮次、附件、trace；
- 提供轮次压缩（compact_history）、会话删除、按文档删除会话等辅助能力。

存储模型：
- ``conversation_turns`` 表：每行一轮对话（角色、内容、可选的工具调用信息）。
- ``conversation_sessions`` 表：会话元数据（归属用户、项目、文档范围、过期时间）。

多租户隔离：构造时可绑定 ``owner_user_id``，写入时记录归属、读取时校验归属，
防止跨用户访问。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models.records import ConversationSession, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    """一轮对话的内存表示（数据库行的脱敏/只读视图）。

    In-memory representation of a conversation turn.

    - ``turn_index``: 该轮在会话内的序号（从 0 开始，升序）。
    - ``role``: 发言方角色（如 "user" / "assistant"）。
    - ``content``: 文本内容。
    - ``tool_name`` / ``tool_args`` / ``tool_result``: 若该轮涉及工具调用，
      记录工具名、参数与执行结果。
    - ``step_type``: 该轮对应的 Agent 执行步骤类型。
    - ``citations``: 该轮引用的结构化条目。
    - ``created_at``: 创建时间（可能为空）。
    """

    turn_index: int
    role: str
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None
    step_type: str | None = None
    citations: list[dict] = field(default_factory=list)
    created_at: datetime | None = None


class ConversationMemory:
    """基于 ConversationTurn 表的按会话对话记忆。

    Per-session conversation memory backed by the ConversationTurn table.

    Stores turn history so the Agent Executor can resume conversations
    and enforce turn budgets.

    主要能力：追加轮次、读取历史（可限制最近 N 轮）、统计轮数、
    取首个用户轮次、压缩历史、删除会话、刷新会话过期时间、
    批量清理过期会话。
    """

    def __init__(self, db: Session, owner_user_id: str | None = None) -> None:
        """初始化会话记忆访问器。

        :param db: 活动的 SQLAlchemy 会话（事务由调用方管理）。
        :param owner_user_id: 数据归属用户 ID；写入时记录归属、读取时
            用于归属校验（多租户数据隔离）。为 None 时不启用归属校验。
        """
        self._db = db
        self._owner_user_id = owner_user_id

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        tool_result: str | None = None,
        step_type: str | None = None,
        citations: list[dict] | None = None,
    ) -> ConversationTurn:
        """向会话追加一轮对话，并返回已持久化的行。

        Append a turn and return the persisted row.

        :param session_id: 目标会话 ID。
        :param role: 发言方角色（如 "user" / "assistant"）。
        :param content: 轮次文本内容。
        :param tool_name: 可选的工具调用名。
        :param tool_args: 可选的工具调用参数。
        :param tool_result: 可选的工具执行结果。
        :param step_type: 可选的执行步骤类型。
        :param citations: 可选的引用条目列表（会被拷贝一份，避免外部共享修改）。
        :return: 已 flush 的 :class:`ConversationTurn` 行对象。
        """
        # 下一轮序号 = 当前会话已有轮数（从 0 递增），保证顺序稳定。
        next_index = self._next_turn_index(session_id)
        turn = ConversationTurn(
            session_id=session_id,
            turn_index=next_index,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            step_type=step_type,
            citations=list(citations or []),
        )
        self._db.add(turn)
        self._db.flush()
        return turn

    def get_history(self, session_id: str, *, last_n: int | None = None) -> list[TurnRecord]:
        """返回 *session_id* 的轮次历史，按 turn_index 升序。

        Return turns for *session_id* ordered by turn_index ascending.

        When *last_n* is provided only the most recent N turns are
        returned.

        :param session_id: 目标会话 ID。
        :param last_n: 若指定且 > 0，仅返回最近 N 轮（用于控制送入模型的
            上下文长度）。
        :return: 按时间正序排列的 :class:`TurnRecord` 列表。
        """
        statement = (
            select(ConversationTurn)
            .where(ConversationTurn.session_id == session_id)
            .order_by(ConversationTurn.turn_index.asc())
        )
        rows = self._db.scalars(statement).all()
        # 只保留最近 N 轮：切片取末尾，保持原有升序。
        if last_n is not None and last_n > 0:
            rows = rows[-last_n:]
        return [
            TurnRecord(
                turn_index=row.turn_index,
                role=row.role,
                content=row.content,
                tool_name=row.tool_name,
                tool_args=row.tool_args,
                tool_result=row.tool_result,
                step_type=row.step_type,
                # 兼容旧数据：citations 若不以 list 存储（如 None 或 JSON 串），
                # 一律降级为空列表。
                citations=row.citations if isinstance(row.citations, list) else [],
                created_at=row.created_at,
            )
            for row in rows
        ]

    def get_recent_table_anchors(
        self, session_id: str, *, last_n: int = 2
    ) -> list[dict]:
        """提取最近几轮 agent 答案引用中的表格证据（轻量跨轮锚点）。

        每轮 finalize 时 citations 已随 turn 持久化（agent_executor 以
        model_dump 形式写入）。此处筛选表格类引用（block_type == "table"、
        携带 table_id，或 excerpt 含 markdown 表行），按轮序返回（最新在
        前），供表格指代查询（"第二张表"等）注入候选证据——指代查询无法
        从自身词元检索到表格内容，需要历史引用定位（2026-08-12 回归
        R17/R18/R20/R22 的根因之一）。

        :param session_id: 会话 ID。
        :param last_n: 最多回溯最近几轮 agent（finalize）轮次。
        :return: 表格引用条目列表，每项含 citation 字段与轮次信息。
        """
        statement = (
            select(ConversationTurn)
            .where(
                ConversationTurn.session_id == session_id,
                ConversationTurn.role == "agent",
                ConversationTurn.step_type == "finalize",
            )
            .order_by(ConversationTurn.turn_index.desc())
            .limit(last_n)
        )
        rows = self._db.scalars(statement).all()
        anchors: list[dict] = []
        # rows 已是按 turn_index 降序（最新在前），直接顺序遍历保证
        # anchors 返回顺序也是"最新在前"。
        for row in rows:
            for citation in row.citations or []:
                if not isinstance(citation, dict):
                    continue
                excerpt = str(citation.get("excerpt") or "").strip()
                if not excerpt:
                    continue
                is_table = bool(
                    citation.get("block_type") == "table"
                    or citation.get("table_id")
                    or citation.get("page_kind") == "table"
                    or "|" in excerpt
                )
                if not is_table:
                    continue
                anchors.append(
                    {
                        "excerpt": excerpt,
                        "document_id": citation.get("document_id"),
                        "chunk_id": citation.get("chunk_id"),
                        "parse_version": citation.get("parse_version"),
                        "source_spans": citation.get("source_spans", []),
                        "page_label": citation.get("page_label"),
                        "table_id": citation.get("table_id"),
                        "turn_index": row.turn_index,
                    }
                )
        return anchors

    def turn_count(self, session_id: str) -> int:
        """返回 *session_id* 的轮次数量。

        Return the number of turns for *session_id*.

        通过 COUNT 聚合查询统计，供轮次预算 / 压缩决策使用。
        """
        return int(
            self._db.scalar(
                text(
                    "SELECT COUNT(*) FROM conversation_turns WHERE session_id = :sid"
                ),
                {"sid": session_id},
            )
            or 0
        )

    def first_user_turn(self, session_id: str) -> TurnRecord | None:
        """返回 *session_id* 的首个用户轮次；不存在时返回 None。

        Return the first user turn for *session_id*, if any.

        用于提取会话的初始意图（通常作为上下文摘要的种子）。
        """
        statement = (
            select(ConversationTurn)
            .where(
                ConversationTurn.session_id == session_id,
                ConversationTurn.role == "user",
            )
            .order_by(ConversationTurn.turn_index.asc())
            .limit(1)
        )
        row = self._db.scalar(statement)
        if row is None:
            return None
        return TurnRecord(
            turn_index=row.turn_index,
            role=row.role,
            content=row.content,
            tool_name=row.tool_name,
            tool_args=row.tool_args,
            tool_result=row.tool_result,
            step_type=row.step_type,
            citations=row.citations if isinstance(row.citations, list) else [],
            created_at=row.created_at,
        )

    def compact_history(self, session_id: str, max_turns: int) -> int:
        """删除最旧的轮次，使会话最多保留 *max_turns* 轮。

        Delete oldest turns so at most *max_turns* remain.

        Returns the number of deleted rows.

        :param session_id: 目标会话 ID。
        :param max_turns: 允许保留的最大轮次数；< 0 视为 0。
        :return: 实际删除的行数。
        """
        if max_turns < 0:
            max_turns = 0
        current = self.turn_count(session_id)
        if current <= max_turns:
            # 当前轮数未超上限，无需压缩。
            return 0
        to_delete = current - max_turns
        # 按 turn_index 升序取最旧的 to_delete 行并删除。
        self._db.execute(
            text(
                "DELETE FROM conversation_turns WHERE id IN ("
                "  SELECT id FROM conversation_turns"
                "  WHERE session_id = :sid"
                "  ORDER BY turn_index ASC"
                "  LIMIT :limit"
                ")"
            ),
            {"sid": session_id, "limit": to_delete},
        )
        self._db.flush()
        return to_delete

    def delete_session(self, session_id: str) -> int:
        """删除单个会话及其关联的附属数据，返回删除的轮次数量。

        Remove one session and its stored side data. Returns deleted turn count.

        删除范围（保持外键一致性）：会话附件 → trace steps → trace runs →
        conversation_turns → conversation_sessions。
        """
        # 延迟导入，避免循环依赖（session_attachments 依赖本模块的表结构）。
        from app.services.session_attachments import delete_attachments_for_session

        count = self.turn_count(session_id)
        # 先删除会话绑定的附件（文件等侧数据）。
        delete_attachments_for_session(self._db, session_id)
        # 先删 trace steps 再删 trace runs：SQLite 外键级联不一定开启，
        # 必须显式按引用顺序清理，避免孤儿记录。
        self._db.execute(
            text(
                "DELETE FROM agent_trace_steps WHERE run_id IN ("
                "  SELECT id FROM agent_trace_runs WHERE session_id = :sid"
                ")"
            ),
            {"sid": session_id},
        )
        self._db.execute(
            text("DELETE FROM agent_trace_runs WHERE session_id = :sid"),
            {"sid": session_id},
        )
        # 删除对话轮次与会话本体。
        self._db.execute(
            text("DELETE FROM conversation_turns WHERE session_id = :sid"),
            {"sid": session_id},
        )
        self._db.execute(
            text("DELETE FROM conversation_sessions WHERE id = :sid"),
            {"sid": session_id},
        )
        self._db.flush()
        return count

    def delete_sessions_for_document(self, document_id: str) -> int:
        """删除绑定到 *document_id* 的全部会话，返回删除的会话数量。

        Remove all sessions scoped to *document_id* and return session count.

        文档被删除时，需要级联清理其所有会话。
        """
        rows = self._db.execute(
            text("SELECT id FROM conversation_sessions WHERE document_id = :did"),
            {"did": document_id},
        ).all()
        count = 0
        # 逐个调用 delete_session，复用统一的附属数据清理逻辑。
        for (session_id,) in rows:
            self.delete_session(session_id)
            count += 1
        self._db.flush()
        return count

    def touch_session(
        self,
        session_id: str,
        *,
        project_slug: str,
        ttl_days: int,
        document_id: str | None = None,
    ) -> None:
        """创建或更新会话，并为其设置过期时间（TTL）。

        Create or update a conversation session with an expiry time.

        Sets ``expires_at = now + ttl_days``.  An existing session is never
        silently rebound to a different project or document scope.

        TTL 机制：每次"触碰"会话都会把 ``expires_at`` 刷新为
        ``now + ttl_days``，即会话活跃期随使用自动顺延；长期不活跃的
        会话将自然过期并被 :meth:`purge_expired_sessions` 清理。

        防误绑：已存在的会话不允许被静默改绑到其他项目 / 文档范围，
        也禁止跨用户访问（见下方归属校验）。
        """
        existing = self._db.get(ConversationSession, session_id)
        if existing is not None:
            # 归属校验：绑定了 owner 时，禁止访问他人会话。
            if self._owner_user_id is not None and existing.owner_user_id != self._owner_user_id:
                raise ValueError(f"Session {session_id} belongs to another user.")
            # 项目范围不可变：禁止把会话改绑到其他项目。
            if existing.project_slug != project_slug:
                raise ValueError(
                    f"Session {session_id} belongs to project {existing.project_slug}; "
                    f"cannot rebind to project {project_slug}."
                )
            # 文档范围不可变：禁止把会话改绑到其他文档。
            if existing.document_id != document_id:
                raise ValueError(
                    f"Session {session_id} is scoped to document {existing.document_id}; "
                    f"cannot rebind to document {document_id}."
                )
        # 重新计算过期时间：从当前 UTC 时间顺延 ttl_days 天。
        expires_at = datetime.utcnow() + timedelta(days=ttl_days)
        if existing:
            # 已有会话：只刷新过期时间与更新时间，不改变归属/范围。
            existing.expires_at = expires_at
            existing.updated_at = datetime.utcnow()
        else:
            # 新会话：创建完整记录（含归属用户、项目、可选文档范围）。
            session = ConversationSession(
                id=session_id,
                owner_user_id=self._owner_user_id,
                project_slug=project_slug,
                document_id=document_id,
                expires_at=expires_at,
            )
            self._db.add(session)
        self._db.flush()

    def purge_expired_sessions(self) -> int:
        """删除所有已过期的会话及其对话轮次、附件与 trace。

        Delete expired sessions and their conversation turns and attachments.

        Returns the number of sessions deleted.

        由定时任务周期性调用：避免过期会话无限累积，回收存储空间。
        """
        now = datetime.utcnow()
        # 找出所有 expires_at 早于当前时刻的过期会话 ID。
        result = self._db.execute(
            text("SELECT id FROM conversation_sessions WHERE expires_at < :now"),
            {"now": now},
        )
        expired_ids = [row[0] for row in result.fetchall()]
        if not expired_ids:
            return 0

        # 逐个删除以兼容 SQLite：SQLAlchemy 的 text() 不会为 IN 子句展开
        # 元组参数，因此无法使用可移植的 "DELETE ... WHERE id IN :ids" 批量写法；
        # 逐个调用 delete_session 还能统一复用附件/trace/轮次的清理逻辑。
        count = 0
        for sid in expired_ids:
            self.delete_session(sid)
            count += 1
        self._db.flush()
        return count

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _next_turn_index(self, session_id: str) -> int:
        """计算会话下一轮的序号（= 当前最大轮序 + 1，从 0 递增）。

        使用 MAX(turn_index)+1 而非 turn_count()：compact_history 删除
        最旧轮次后 count 变小，但索引必须保持单调递增——否则新轮次
        复用已删除 index，排序与"上一轮"解析（get_history 升序取 [-2]）
        全部错乱（40 题会话实测 23 条 turn 覆盖 40 轮的乱序 bug）。
        """
        max_index = self._db.scalar(
            text(
                "SELECT COALESCE(MAX(turn_index), -1) FROM conversation_turns "
                "WHERE session_id = :sid"
            ),
            {"sid": session_id},
        )
        return int(max_index) + 1
