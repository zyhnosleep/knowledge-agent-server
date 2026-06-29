from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base


def make_db() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def test_startup_purge_calls_conversation_and_trace_cleanup() -> None:
    """_startup_purge calls both ConversationMemory and AgentTraceStore cleanup."""
    db = make_db()

    conv_deleted = 0
    trace_deleted = 0

    class FakeConversationMemory:
        def __init__(self, db):
            pass

        def purge_expired_sessions(self):
            nonlocal conv_deleted
            conv_deleted = 3
            return 3

    class FakeAgentTraceStore:
        def __init__(self, db):
            pass

        def purge_expired(self, retention_days):
            nonlocal trace_deleted
            trace_deleted = 2
            return 2

    with patch(
        "app.main.ConversationMemory", FakeConversationMemory
    ), patch(
        "app.main.AgentTraceStore", FakeAgentTraceStore
    ), patch(
        "app.main.SessionLocal", return_value=db
    ):
        from app.main import _startup_purge
        _startup_purge()

    assert conv_deleted == 3, "Conversation purge should have been called"
    assert trace_deleted == 2, "Trace purge should have been called"


def test_startup_purge_failure_is_non_fatal(caplog) -> None:
    """When cleanup raises, it is logged and does not propagate."""
    db = make_db()

    class FailingConversationMemory:
        def __init__(self, db):
            pass

        def purge_expired_sessions(self):
            raise RuntimeError("simulated purge failure")

    with patch(
        "app.main.ConversationMemory", FailingConversationMemory
    ), patch(
        "app.main.SessionLocal", return_value=db
    ), caplog.at_level(logging.ERROR):
        from app.main import _startup_purge
        # Must not raise
        _startup_purge()

    # The failure should have been logged
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) >= 1, "Startup purge failure should be logged"


def test_startup_purge_handles_db_rollback_failure() -> None:
    """When cleanup raises AND rollback also fails, still no propagate."""
    db = make_db()

    class FailingConvMemory:
        def __init__(self, d):
            pass

        def purge_expired_sessions(self):
            raise RuntimeError("simulated purge failure")

    class BrokenDB:
        """Session-like object whose rollback also raises."""
        def rollback(self):
            raise RuntimeError("simulated rollback failure")

        def close(self):
            pass

    broken = BrokenDB()

    with patch(
        "app.main.ConversationMemory", FailingConvMemory
    ), patch(
        "app.main.SessionLocal", return_value=broken
    ):
        from app.main import _startup_purge
        # Must not raise
        _startup_purge()


def test_startup_purge_zero_results_logs_nothing_special() -> None:
    """When no sessions or traces are expired, the function returns cleanly."""
    db = make_db()

    class NoopConversationMemory:
        def __init__(self, db):
            pass

        def purge_expired_sessions(self):
            return 0

    class NoopAgentTraceStore:
        def __init__(self, db):
            pass

        def purge_expired(self, retention_days):
            return 0

    with patch(
        "app.main.ConversationMemory", NoopConversationMemory
    ), patch(
        "app.main.AgentTraceStore", NoopAgentTraceStore
    ), patch(
        "app.main.SessionLocal", return_value=db
    ):
        from app.main import _startup_purge
        _startup_purge()
    # No assertion needed — just verifying no exception


def test_lifespan_includes_startup_purge() -> None:
    """The FastAPI lifespan context manager includes _startup_purge."""
    # Verify _startup_purge is called during lifespan startup by patching
    # it and confirming it was called.
    called = False

    def fake_purge():
        nonlocal called
        called = True

    db = make_db()

    import app.main as main_module

    with patch.object(main_module, "_startup_purge", fake_purge), patch(
        "app.main.SessionLocal", return_value=db
    ), patch("app.main.init_db"), patch("app.main.configure_logging"), patch(
        "app.main.get_or_create_project"
    ):
        # Import the lifespan and run its startup side
        from app.main import lifespan as app_lifespan
        import asyncio

        async def run_lifespan_startup():
            async with app_lifespan(main_module.app):
                pass  # Just trigger startup, then immediately shutdown

        asyncio.run(run_lifespan_startup())

    assert called, "lifespan should call _startup_purge during startup"
