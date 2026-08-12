from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base
from app.models.records import ConversationSession, ConversationTurn
from app.services.conversation_memory import ConversationMemory


def make_memory() -> tuple[Session, ConversationMemory]:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()
    return db, ConversationMemory(db)


def test_add_and_get_history() -> None:
    db, memory = make_memory()
    memory.add_turn("s1", role="user", content="hello", step_type="user_query")
    memory.add_turn("s1", role="tool", content="result", tool_name="rag.answer", step_type="tool_call")
    db.commit()

    history = memory.get_history("s1")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "hello"
    assert history[1].role == "tool"
    assert history[1].tool_name == "rag.answer"


def test_touch_session_records_and_enforces_owner() -> None:
    db, _ = make_memory()
    first = ConversationMemory(db, owner_user_id="u1")
    first.touch_session("owned", project_slug="demo", ttl_days=1)

    assert db.get(ConversationSession, "owned").owner_user_id == "u1"
    with pytest.raises(ValueError, match="another user"):
        ConversationMemory(db, owner_user_id="u2").touch_session(
            "owned", project_slug="demo", ttl_days=1
        )


def test_get_history_last_n() -> None:
    db, memory = make_memory()
    for i in range(5):
        memory.add_turn("s1", role="user", content=f"msg {i}")
    db.commit()

    assert len(memory.get_history("s1")) == 5
    assert len(memory.get_history("s1", last_n=2)) == 2
    assert memory.get_history("s1", last_n=2)[0].content == "msg 3"
    assert memory.get_history("s1", last_n=2)[1].content == "msg 4"


def test_turn_count() -> None:
    db, memory = make_memory()
    assert memory.turn_count("s1") == 0
    memory.add_turn("s1", role="user", content="a")
    db.commit()
    assert memory.turn_count("s1") == 1


def test_compact_history() -> None:
    db, memory = make_memory()
    for i in range(10):
        memory.add_turn("s1", role="user", content=f"msg {i}")
    db.commit()
    assert memory.turn_count("s1") == 10

    deleted = memory.compact_history("s1", max_turns=3)
    db.commit()
    assert deleted == 7
    assert memory.turn_count("s1") == 3
    # oldest messages should be gone, newest should remain
    remaining = memory.get_history("s1")
    assert remaining[0].content == "msg 7"
    assert remaining[1].content == "msg 8"
    assert remaining[2].content == "msg 9"


def test_compact_history_no_op() -> None:
    db, memory = make_memory()
    for i in range(3):
        memory.add_turn("s1", role="user", content=f"msg {i}")
    db.commit()
    deleted = memory.compact_history("s1", max_turns=10)
    assert deleted == 0
    assert memory.turn_count("s1") == 3


def test_turn_index_monotonic_after_compact() -> None:
    """压缩删除最旧轮次后，新轮次 turn_index 继续递增、不复用。

    回归 2026-08-11：40 题会话实测仅 23 条 turn 却覆盖 40 轮乱序——
    旧实现 `_next_turn_index` 用 turn_count() 计算，compact 删除行后
    复用已删除 index，排序与"上一轮"解析（[-2]）随之错乱。
    """
    db, memory = make_memory()
    for i in range(10):
        memory.add_turn("s1", role="user", content=f"msg {i}")
    db.commit()
    memory.compact_history("s1", max_turns=3)
    db.commit()

    memory.add_turn("s1", role="user", content="msg 10")
    db.commit()
    history = memory.get_history("s1")
    # 剩余 [7,8,9] + 新轮：索引必须严格递增，不得复用为 3
    assert [t.turn_index for t in history] == [7, 8, 9, 10]
    assert memory.turn_count("s1") == 4


def test_turn_index_monotonic_over_forty_rounds() -> None:
    """模拟 40 轮会话（每轮 user+agent turn + 每轮 compact）：索引严格递增。

    对齐 40 题 eval 场景：max_turns=200 时 40 轮（80 条 turn）不触发
    压缩删除，全部历史保留且顺序正确；_resolvable_previous_turn 依赖
    的升序 [-2] 解析因此可靠。
    """
    db, memory = make_memory()
    for i in range(40):
        memory.add_turn("s1", role="user", content=f"user {i}", step_type="user_query")
        memory.add_turn("s1", role="agent", content=f"agent {i}", step_type="finalize")
        memory.compact_history("s1", max_turns=200)
        db.commit()

    history = memory.get_history("s1")
    assert len(history) == 80  # 40 轮 × 2 条 turn，未触发压缩
    indices = [t.turn_index for t in history]
    assert indices == sorted(indices)
    assert len(set(indices)) == 80  # 无任何复用
    assert history[0].content == "user 0"
    assert history[-1].content == "agent 39"


def test_delete_session() -> None:
    db, memory = make_memory()
    memory.add_turn("s1", role="user", content="a")
    memory.add_turn("s2", role="user", content="b")
    db.commit()

    assert memory.turn_count("s1") == 1
    assert memory.turn_count("s2") == 1

    deleted = memory.delete_session("s1")
    db.commit()
    assert deleted == 1
    assert memory.turn_count("s1") == 0
    assert memory.turn_count("s2") == 1  # other session untouched


def test_sessions_are_isolated() -> None:
    db, memory = make_memory()
    memory.add_turn("s1", role="user", content="a1")
    memory.add_turn("s2", role="user", content="a2")
    memory.add_turn("s1", role="tool", content="b1")
    db.commit()

    h1 = memory.get_history("s1")
    h2 = memory.get_history("s2")
    assert len(h1) == 2
    assert len(h2) == 1
    assert h1[0].content == "a1"
    assert h1[1].content == "b1"
    assert h2[0].content == "a2"


# ----------------------------------------------------------------
# v3 TTL tests — touch_session, purge_expired_sessions
# ----------------------------------------------------------------


def test_touch_session_creates_new_session() -> None:
    """touch_session creates a ConversationSession with expires_at in the future."""
    db, memory = make_memory()
    memory.touch_session("sess-abc", project_slug="demo", ttl_days=30)

    session = db.get(ConversationSession, "sess-abc")
    assert session is not None
    assert session.project_slug == "demo"
    assert not hasattr(session, "answer_mode")
    assert session.expires_at > datetime.utcnow() + timedelta(days=29)  # ~30 days


def test_touch_session_updates_existing_session() -> None:
    """touch_session updates an existing session, extending expires_at."""
    db, memory = make_memory()
    memory.touch_session("sess-abc", project_slug="demo", ttl_days=30)
    original_expires = db.get(ConversationSession, "sess-abc").expires_at

    # Touch again with a different TTL
    memory.touch_session("sess-abc", project_slug="demo", ttl_days=60)
    updated = db.get(ConversationSession, "sess-abc")
    assert updated is not None
    assert updated.expires_at > original_expires
    assert updated.expires_at > datetime.utcnow() + timedelta(days=55)


def test_touch_session_preserves_session_id() -> None:
    """touch_session does not create duplicate sessions."""
    db, memory = make_memory()
    memory.touch_session("sess-abc", project_slug="demo", ttl_days=30)
    memory.touch_session("sess-abc", project_slug="demo", ttl_days=30)

    # There should be exactly one row
    result = db.execute(
        text("SELECT COUNT(*) FROM conversation_sessions WHERE id = 'sess-abc'")
    )
    count = result.scalar() or 0
    assert count == 1


def test_purge_expired_sessions_deletes_expired() -> None:
    """purge_expired_sessions deletes expired sessions and their turns."""
    db, memory = make_memory()

    # Create an expired session by setting expires_at in the past
    expired = ConversationSession(
        id="expired-sess",
        project_slug="demo",
        expires_at=datetime.utcnow() - timedelta(days=1),
    )
    db.add(expired)
    # Add a turn for the expired session
    db.execute(
        text("INSERT INTO conversation_turns (id, session_id, turn_index, role, content, created_at) "
             "VALUES ('t1', 'expired-sess', 0, 'user', 'old message', :now)"),
        {"now": datetime.utcnow()},
    )
    db.commit()

    # Create an active session
    active = ConversationSession(
        id="active-sess",
        project_slug="demo",
        expires_at=datetime.utcnow() + timedelta(days=30),
    )
    db.add(active)
    db.execute(
        text("INSERT INTO conversation_turns (id, session_id, turn_index, role, content, created_at) "
             "VALUES ('t2', 'active-sess', 0, 'user', 'active message', :now)"),
        {"now": datetime.utcnow()},
    )
    db.commit()

    deleted = memory.purge_expired_sessions()
    db.commit()

    assert deleted == 1

    # Expired session should be gone
    assert db.get(ConversationSession, "expired-sess") is None
    # Expired session's turns should be gone
    remaining = db.execute(
        text("SELECT COUNT(*) FROM conversation_turns WHERE session_id = 'expired-sess'")
    )
    assert (remaining.scalar() or 0) == 0

    # Active session should remain
    assert db.get(ConversationSession, "active-sess") is not None
    # Active session's turns should remain
    active_turns = db.execute(
        text("SELECT COUNT(*) FROM conversation_turns WHERE session_id = 'active-sess'")
    )
    assert (active_turns.scalar() or 0) == 1


def test_purge_expired_sessions_handles_no_expired() -> None:
    """purge_expired_sessions returns 0 when no sessions are expired."""
    db, memory = make_memory()
    memory.touch_session("sess-1", project_slug="demo", ttl_days=30)
    db.commit()

    deleted = memory.purge_expired_sessions()
    assert deleted == 0
    assert db.get(ConversationSession, "sess-1") is not None


def test_purge_expired_sessions_batch_delete_works() -> None:
    """purge_expired_sessions handles multiple expired sessions."""
    db, memory = make_memory()

    for i in range(3):
        s = ConversationSession(
            id=f"expired-{i}",
            project_slug="demo",
            expires_at=datetime.utcnow() - timedelta(days=i + 1),
        )
        db.add(s)
    db.commit()

    deleted = memory.purge_expired_sessions()
    db.commit()
    assert deleted == 3
    for i in range(3):
        assert db.get(ConversationSession, f"expired-{i}") is None


def test_sqlite_migration_adds_session_mode_document_id_and_citations_columns() -> None:
    """Old schema is upgraded with mode/scope/citation fields idempotently."""
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Create only the original schema subset.
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE conversation_sessions (
                id TEXT PRIMARY KEY,
                project_slug TEXT NOT NULL,
                expires_at DATETIME NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE conversation_turns (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))

    # Run the migration helpers directly against this engine.
    import app.db.session as session_module
    original_engine = session_module.engine
    original_url = session_module.settings.database_url
    session_module.engine = engine
    session_module.settings.database_url = "sqlite:///:memory:"
    try:
        session_module._ensure_sqlite_columns()
        session_module._ensure_sqlite_indexes()
    finally:
        session_module.engine = original_engine
        session_module.settings.database_url = original_url

    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()
    db.execute(
        text(
            "INSERT INTO conversation_sessions "
            "(id, project_slug, document_id, expires_at) "
            "VALUES ('s1', 'demo', 'd1', :now)"
        ),
        {"now": datetime.utcnow()},
    )
    db.execute(
        text(
            "INSERT INTO conversation_turns (id, session_id, turn_index, role, content, created_at) "
            "VALUES ('t1', 's1', 0, 'user', 'hello', :now)"
        ),
        {"now": datetime.utcnow()},
    )
    db.commit()
    columns = {
        row[1]
        for row in db.execute(text("PRAGMA table_info('conversation_sessions')"))
    }
    assert "answer_mode" not in columns


def test_turn_citations_round_trip() -> None:
    """Structured citations are stored and restored on conversation turns."""
    db, memory = make_memory()
    citation = {"document_id": "d1", "chunk_id": "c1", "excerpt": "excerpt"}
    memory.add_turn(
        "s1",
        role="agent",
        content="answer",
        step_type="finalize",
        citations=[citation],
    )
    db.commit()

    history = memory.get_history("s1")
    assert len(history) == 1
    assert history[0].citations == [citation]


def test_document_scope_rebind_is_rejected() -> None:
    """touch_session rejects changing an existing session's document scope."""
    db, memory = make_memory()
    memory.touch_session("scoped", project_slug="demo", ttl_days=30, document_id="d1")
    db.commit()

    with pytest.raises(ValueError):
        memory.touch_session("scoped", project_slug="demo", ttl_days=30, document_id="d2")


def test_get_recent_table_anchors_extracts_table_citations() -> None:
    """仅 agent/finalize 轮的表格类引用被提取；段落引用被跳过。"""
    db, memory = make_memory()
    memory.add_turn("s1", role="user", content="第一张表是什么", step_type="user_query")
    memory.add_turn(
        "s1",
        role="agent",
        content="answer1",
        step_type="finalize",
        citations=[
            {
                "block_type": "table",
                "excerpt": "| A | B |\n| --- | --- |\n| 1 | 2 |",
                "document_id": "d1",
                "page_label": "8",
            },
            {"block_type": "paragraph", "excerpt": "plain prose", "document_id": "d1"},
        ],
    )
    memory.add_turn(
        "s1",
        role="agent",
        content="answer2",
        step_type="finalize",
        citations=[{"block_type": "paragraph", "excerpt": "more prose", "document_id": "d1"}],
    )
    db.commit()

    anchors = memory.get_recent_table_anchors("s1")
    assert len(anchors) == 1
    assert anchors[0]["excerpt"] == "| A | B |\n| --- | --- |\n| 1 | 2 |"
    assert anchors[0]["document_id"] == "d1"
    assert anchors[0]["page_label"] == "8"
    assert anchors[0]["turn_index"] == 1


def test_get_recent_table_anchors_newest_first_and_limited() -> None:
    """按轮次最新在前返回，last_n 限制回溯深度。"""
    db, memory = make_memory()
    for i in range(3):
        memory.add_turn(
            "s1",
            role="agent",
            content=f"ans{i}",
            step_type="finalize",
            citations=[{"table_id": f"tbl-{i}", "excerpt": f"| row {i} | 1 | 2 |"}],
        )
    db.commit()

    anchors = memory.get_recent_table_anchors("s1", last_n=2)
    assert [a["table_id"] for a in anchors] == ["tbl-2", "tbl-1"]
    assert anchors[0]["turn_index"] == 2


def test_get_recent_table_anchors_skips_non_table_excerpts() -> None:
    """段落引用与空摘录不构成表格锚点（空摘录即使 block_type=table 也跳过）。"""
    db, memory = make_memory()
    memory.add_turn(
        "s1",
        role="agent",
        content="answer",
        step_type="finalize",
        citations=[
            {"block_type": "paragraph", "excerpt": "plain prose", "document_id": "d1"},
            {"block_type": "table", "excerpt": "", "document_id": "d1"},
        ],
    )
    db.commit()

    assert memory.get_recent_table_anchors("s1") == []
