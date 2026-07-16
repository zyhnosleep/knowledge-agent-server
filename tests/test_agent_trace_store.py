from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base
from app.schemas.agent import AgentStep, AgentUsage
from app.services.agent_trace_store import AgentTraceStore


def make_db() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def _make_steps() -> list[AgentStep]:
    return [
        AgentStep(step_id=0, step_type="route", summary="route step"),
        AgentStep(step_id=1, step_type="tool_call", summary="rag.answer", tool_name="rag.answer", tool_ok=True),
        AgentStep(step_id=2, step_type="finalize", summary="done"),
    ]


def _make_usage() -> AgentUsage:
    return AgentUsage(prompt_tokens=100, completion_tokens=50, tool_calls=2, steps=3)


def test_persist_run_and_steps() -> None:
    """AgentTraceStore.persist_run persists a run and its steps."""
    db = make_db()
    store = AgentTraceStore(db)
    trace_id = store.persist_run(
        request_id="req-1",
        session_id="sess-1",
        project_slug="demo",
        query="test query",
        constraints={"max_steps": 8},
        route="simple_rag",
        steps=_make_steps(),
        usage=_make_usage(),
        final_answer="test answer",
        citations=[],
        warnings=[],
        status="completed",
        latency_ms=1000,
        provider="local",
        model="local-fallback",
    )
    db.commit()
    assert trace_id is not None
    # Verify persisted
    run = store.get_trace(trace_id)
    assert run is not None
    assert run["status"] == "completed"
    assert run["final_answer"] == "test answer"
    assert len(run["steps"]) == 3


def test_trace_store_records_owner_and_hides_other_users_traces() -> None:
    db = make_db()
    first = AgentTraceStore(db, owner_user_id="u1")
    trace_id = first.persist_run(
        request_id="req-owner",
        session_id="sess-owner",
        project_slug="demo",
        query="private",
        constraints={},
        route="simple_rag",
        steps=[],
        usage=AgentUsage(),
        final_answer="private answer",
        citations=[],
        warnings=[],
        status="completed",
        latency_ms=1,
    )
    db.commit()

    second = AgentTraceStore(db, owner_user_id="u2")
    assert first.get_trace(trace_id) is not None
    assert second.get_trace(trace_id) is None
    assert second.list_traces(project_slug="demo") == []


def test_get_trace_by_unknown_id_returns_none() -> None:
    """get_trace returns None for an unknown trace id."""
    db = make_db()
    store = AgentTraceStore(db)
    assert store.get_trace("nonexistent-id") is None


def test_list_traces_by_session() -> None:
    """list_traces returns traces filtered by session_id."""
    db = make_db()
    store = AgentTraceStore(db)
    store.persist_run(
        request_id="req-1", session_id="sess-a", project_slug="demo", query="q1",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a1", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    store.persist_run(
        request_id="req-2", session_id="sess-b", project_slug="demo", query="q2",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a2", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    db.commit()

    traces_a = store.list_traces(session_id="sess-a", limit=10, offset=0)
    traces_b = store.list_traces(session_id="sess-b", limit=10, offset=0)
    assert len(traces_a) == 1
    assert traces_a[0]["final_answer"] == "a1"
    assert len(traces_b) == 1
    assert traces_b[0]["final_answer"] == "a2"


def test_list_traces_newest_first(monkeypatch) -> None:
    """list_traces returns newest traces first."""
    import app.services.agent_trace_store as trace_store_module

    class FrozenDateTime:
        @classmethod
        def utcnow(cls) -> datetime:
            return datetime(2026, 7, 13, 12, 0, 0)

    monkeypatch.setattr(trace_store_module, "datetime", FrozenDateTime)
    db = make_db()
    store = AgentTraceStore(db)
    trace_id_1 = store.persist_run(
        request_id="req-1", session_id="sess-a", project_slug="demo", query="q1",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a1", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    trace_id_2 = store.persist_run(
        request_id="req-2", session_id="sess-a", project_slug="demo", query="q2",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a2", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    db.commit()
    traces = store.list_traces(session_id="sess-a", limit=10, offset=0)
    assert len(traces) == 2
    # Newest first: trace_id_2 before trace_id_1
    assert traces[0]["trace_id"] == trace_id_2
    assert traces[1]["trace_id"] == trace_id_1


def test_list_traces_with_pagination() -> None:
    """list_traces respects limit and offset."""
    db = make_db()
    store = AgentTraceStore(db)
    for i in range(5):
        store.persist_run(
            request_id=f"req-{i}", session_id="sess-a", project_slug="demo", query=f"q{i}",
            constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
            final_answer=f"a{i}", citations=[], warnings=[], status="completed",
            latency_ms=100, provider="local", model="local-fallback",
        )
    db.commit()

    page1 = store.list_traces(session_id="sess-a", limit=2, offset=0)
    assert len(page1) == 2
    page2 = store.list_traces(session_id="sess-a", limit=2, offset=2)
    assert len(page2) == 2
    page3 = store.list_traces(session_id="sess-a", limit=2, offset=4)
    assert len(page3) == 1


def test_list_traces_limit_boundaries() -> None:
    """list_traces clamps limit to [1, 100]."""
    db = make_db()
    store = AgentTraceStore(db)
    # limit=0 should be clamped to 1
    result = store.list_traces(session_id="sess-a", limit=0, offset=0)
    assert len(result) == 0
    assert isinstance(result, list)


def test_trace_does_not_persist_secrets() -> None:
    """Persisted trace data does not include API keys or full raw provider responses."""
    db = make_db()
    store = AgentTraceStore(db)
    trace_id = store.persist_run(
        request_id="req-1", session_id="sess-1", project_slug="demo", query="q",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="answer", citations=[{"document_id": "d1"}], warnings=[],
        status="completed", latency_ms=100, provider="local", model="local-fallback",
    )
    db.commit()
    run = store.get_trace(trace_id)
    assert run is not None
    # No API keys in the serialized run data
    import json
    run_str = json.dumps(run, default=str)
    assert "api_key" not in run_str.lower()
    assert "authorization" not in run_str.lower()
    assert "bearer" not in run_str.lower()


def test_trace_detail_includes_ordered_steps() -> None:
    """get_trace returns steps ordered by step_id."""
    db = make_db()
    store = AgentTraceStore(db)
    trace_id = store.persist_run(
        request_id="req-1", session_id="sess-1", project_slug="demo", query="q",
        constraints={}, route="simple_rag",
        steps=[
            AgentStep(step_id=0, step_type="route", summary="first"),
            AgentStep(step_id=2, step_type="tool_call", summary="third", tool_name="rag"),
            AgentStep(step_id=1, step_type="tool_call", summary="second", tool_name="verify"),
        ],
        usage=_make_usage(), final_answer="a", citations=[], warnings=[],
        status="completed", latency_ms=100, provider="local", model="local-fallback",
    )
    db.commit()
    run = store.get_trace(trace_id)
    assert run is not None
    step_ids = [s["step_id"] for s in run["steps"]]
    assert step_ids == [0, 1, 2]


# ----------------------------------------------------------------
# Phase 5 — optional filter and sanitization tests
# ----------------------------------------------------------------


def test_list_traces_with_project_slug_filter() -> None:
    """list_traces filters by project_slug."""
    db = make_db()
    store = AgentTraceStore(db)
    store.persist_run(
        request_id="req-1", session_id="s1", project_slug="demo", query="q",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a1", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    store.persist_run(
        request_id="req-2", session_id="s1", project_slug="other", query="q",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a2", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    db.commit()

    all_traces = store.list_traces(session_id="s1")
    assert len(all_traces) == 2

    demo_traces = store.list_traces(session_id="s1", project_slug="demo")
    assert len(demo_traces) == 1
    assert demo_traces[0]["project_slug"] == "demo"


def test_list_traces_with_status_filter() -> None:
    """list_traces filters by status."""
    db = make_db()
    store = AgentTraceStore(db)
    store.persist_run(
        request_id="req-1", session_id="s1", project_slug="demo", query="q",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    store.persist_run(
        request_id="req-2", session_id="s1", project_slug="demo", query="q",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="", citations=[], warnings=["err"], status="error",
        latency_ms=100, provider="local", model="local-fallback",
    )
    db.commit()

    completed = store.list_traces(status="completed")
    assert len(completed) == 1
    assert completed[0]["status"] == "completed"

    errors = store.list_traces(status="error")
    assert len(errors) == 1
    assert errors[0]["status"] == "error"


def test_list_traces_with_provider_filter() -> None:
    """list_traces filters by provider."""
    db = make_db()
    store = AgentTraceStore(db)
    store.persist_run(
        request_id="req-1", session_id="s1", project_slug="demo", query="q",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    store.persist_run(
        request_id="req-2", session_id="s1", project_slug="demo", query="q",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="external_api", model="deepseek-v4-pro",
    )
    db.commit()

    local = store.list_traces(provider="local")
    assert len(local) == 1
    assert local[0]["provider"] == "local"

    external = store.list_traces(provider="external_api")
    assert len(external) == 1
    assert external[0]["provider"] == "external_api"


def test_list_traces_with_route_filter() -> None:
    """list_traces filters by route."""
    db = make_db()
    store = AgentTraceStore(db)
    store.persist_run(
        request_id="req-1", session_id="s1", project_slug="demo", query="q",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    store.persist_run(
        request_id="req-2", session_id="s1", project_slug="demo", query="q",
        constraints={}, route="complex_multi_hop", steps=_make_steps(), usage=_make_usage(),
        final_answer="a", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    db.commit()

    simple = store.list_traces(route="simple_rag")
    assert len(simple) == 1
    assert simple[0]["route"] == "simple_rag"


def test_list_traces_no_filters_returns_all() -> None:
    """list_traces with no filters returns all traces up to limit."""
    db = make_db()
    store = AgentTraceStore(db)
    for i in range(3):
        store.persist_run(
            request_id=f"req-{i}", session_id=f"s{i}", project_slug="demo", query=f"q{i}",
            constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
            final_answer=f"a{i}", citations=[], warnings=[], status="completed",
            latency_ms=100, provider="local", model="local-fallback",
        )
    db.commit()

    results = store.list_traces(limit=10)
    assert len(results) == 3


def test_list_traces_combined_filters() -> None:
    """list_traces supports combining multiple filters."""
    db = make_db()
    store = AgentTraceStore(db)
    store.persist_run(
        request_id="req-1", session_id="s1", project_slug="demo", query="q",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    store.persist_run(
        request_id="req-2", session_id="s1", project_slug="demo", query="q",
        constraints={}, route="evidence_required", steps=_make_steps(), usage=_make_usage(),
        final_answer="", citations=[], warnings=["err"], status="error",
        latency_ms=100, provider="external_api", model="deepseek-v4-pro",
    )
    store.persist_run(
        request_id="req-3", session_id="s2", project_slug="other", query="q",
        constraints={}, route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="external_api", model="deepseek-v4-pro",
    )
    db.commit()

    # Combine session_id + status
    r1 = store.list_traces(session_id="s1", status="error")
    assert len(r1) == 1
    assert r1[0]["status"] == "error"

    # Combine provider + route
    r2 = store.list_traces(provider="external_api", route="evidence_required")
    assert len(r2) == 1
    assert r2[0]["provider"] == "external_api"

    # Multiple filters with no match
    r3 = store.list_traces(
        session_id="s1", status="completed", provider="external_api"
    )
    assert len(r3) == 0


# ----------------------------------------------------------------
# Phase 5 — secret sanitization tests
# ----------------------------------------------------------------


def test_serialize_redacts_api_key_in_constraints() -> None:
    """Constraints dict with api_key is redacted in serialized output."""
    db = make_db()
    store = AgentTraceStore(db)
    trace_id = store.persist_run(
        request_id="req-1", session_id="s1", project_slug="demo", query="q",
        constraints={"max_steps": 8, "api_key": "sk-secret-value"},
        route="simple_rag", steps=_make_steps(), usage=_make_usage(),
        final_answer="a", citations=[], warnings=[], status="completed",
        latency_ms=100, provider="local", model="local-fallback",
    )
    db.commit()
    run = store.get_trace(trace_id)
    assert run is not None
    constraints = run.get("constraints", {})
    # api_key must be redacted
    assert constraints.get("api_key") == "[redacted]"
    # Non-sensitive keys preserved
    assert constraints.get("max_steps") == 8


def test_serialize_redacts_authorization_in_step_metadata() -> None:
    """Step metadata with 'authorization' key is redacted."""
    from app.services.agent_trace_store import _sanitize_dict
    result = _sanitize_dict({
        "tool": "rag.answer",
        "authorization": "Bearer abc123",
        "nested": {"secret_token": "xyz"},
    })
    assert result["tool"] == "rag.answer"
    assert result["authorization"] == "[redacted]"
    assert result["nested"]["secret_token"] == "[redacted]"


def test_serialize_redacts_secret_looking_values() -> None:
    """String values that look like secrets are redacted."""
    from app.services.agent_trace_store import _sanitize_dict
    result = _sanitize_dict({
        "header": "Bearer sk-proj-1234567890abcdef",
        "prefix": "sk-ant-api03-xxxxxxxxxxxx",
        "normal": "hello world",
    })
    assert result["header"] == "[redacted]"
    assert result["prefix"] == "[redacted]"
    assert result["normal"] == "hello world"


def test_serialize_nested_dicts_are_recursively_sanitized() -> None:
    """Deeply nested dicts are recursively sanitized."""
    from app.services.agent_trace_store import _sanitize_dict
    result = _sanitize_dict({
        "level1": {
            "level2": {
                "api_key": "nested-secret",
                "data": "ok",
            }
        }
    })
    assert result["level1"]["level2"]["api_key"] == "[redacted]"
    assert result["level1"]["level2"]["data"] == "ok"


def test_serialize_list_values_are_sanitized() -> None:
    """Lists of dicts are sanitized element by element."""
    from app.services.agent_trace_store import _sanitize_dict
    result = _sanitize_dict({
        "items": [
            {"name": "a", "token": "secret1"},
            {"name": "b", "token": "secret2"},
        ]
    })
    assert result["items"][0]["token"] == "[redacted]"
    assert result["items"][1]["token"] == "[redacted]"
    assert result["items"][0]["name"] == "a"


def test_serialize_empty_and_none_input() -> None:
    """_sanitize_dict handles None and empty dicts."""
    from app.services.agent_trace_store import _sanitize_dict
    assert _sanitize_dict(None) == {}
    assert _sanitize_dict({}) == {}


def test_trace_serialization_redacts_secrets_in_steps() -> None:
    """Full trace serialization redacts secrets from step metadata."""
    db = make_db()
    store = AgentTraceStore(db)
    trace_id = store.persist_run(
        request_id="req-1", session_id="s1", project_slug="demo", query="q",
        constraints={"api_key": "should-be-redacted"},
        route="simple_rag",
        steps=[
            AgentStep(
                step_id=0, step_type="route", summary="route",
                metadata={"authorization": "Bearer xyz123"},
            ),
        ],
        usage=_make_usage(), final_answer="a", citations=[], warnings=[],
        status="completed", latency_ms=100, provider="local", model="local-fallback",
    )
    db.commit()
    run = store.get_trace(trace_id)
    assert run is not None
    assert run["constraints"].get("api_key") == "[redacted]"
    step_meta = run["steps"][0]["metadata"]
    assert step_meta.get("authorization") == "[redacted]"
