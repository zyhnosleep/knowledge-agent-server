from __future__ import annotations

from collections.abc import Iterator

from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.agent_routes import agent_router
from app.core.config import get_settings
from app.db.session import Base, get_db
from app.models.records import ConversationSession, ConversationTurn, Project


def make_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def make_client(db: Session, *, agent_enabled: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(agent_router, prefix="/api/agent")

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[get_db] = override_db

    # Keep route tests deterministic even when the developer/server .env has
    # a real external API key configured.
    settings = get_settings()
    settings.agent_enabled = agent_enabled
    settings.agent_synthesis_provider = "local"
    settings.external_api_enabled = False
    settings.external_api_key = None

    return TestClient(app, raise_server_exceptions=False)


def test_agent_disabled_returns_503(monkeypatch) -> None:
    """When AGENT_ENABLED=false the route returns 503."""
    db = make_session()
    client = make_client(db, agent_enabled=True)

    # Force settings to report agent_enabled=False
    import app.api.agent_routes as agent_mod
    monkeypatch.setattr(agent_mod, "settings", _DisabledSettings())

    response = client.post(
        "/api/agent/query",
        json={"project_slug": "demo", "query": "hello?"},
    )
    assert response.status_code == 503
    assert "not enabled" in response.json()["detail"].lower()


def test_agent_enabled_with_valid_payload_returns_200(monkeypatch) -> None:
    """When enabled, the route returns 200 with AgentQueryResponse shape."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.schemas.common import QueryResponse, Citation

    class StubRAG:
        def answer(self, db, project_slug, question):
            return QueryResponse(
                answer_markdown="route test answer",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.88,
                        excerpt="rt excerpt",
                    )
                ],
                verification_status="local-only",
            )

    client = make_client(db)
    # Make RAGAdapter produce a real response
    monkeypatch.setattr(
        "app.services.rag_adapter.RAGAdapter.answer",
        lambda self, db, project_slug, question: StubRAG().answer(db, project_slug, question),
    )

    response = client.post(
        "/api/agent/query",
        json={"project_slug": "demo", "query": "hello?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["final_answer"] == "route test answer"
    assert len(data["citations"]) == 1
    # v2a: route + rag.answer + answer.verify + finalize
    assert len(data["steps"]) >= 3
    assert data["usage"]["tool_calls"] >= 2


def test_agent_route_applies_server_constraint_defaults(monkeypatch) -> None:
    """UI requests without constraints use AGENT_* server defaults."""
    import app.api.agent_routes as agent_mod
    from app.schemas.agent import AgentQueryResponse, AgentUsage

    db = make_session()
    captured = {}

    class RouteSettings:
        agent_enabled = True
        agent_allow_external_network = True
        agent_max_steps = 11
        agent_max_tool_calls = 7
        agent_budget_tokens = 30000
        agent_timeout_seconds = 120

    class FakeExecutor:
        def execute(self, payload):
            captured["constraints"] = payload.constraints
            return AgentQueryResponse(
                request_id="req",
                session_id="sess",
                status="completed",
                final_answer="ok",
                citations=[],
                steps=[],
                usage=AgentUsage(),
                warnings=[],
            )

    monkeypatch.setattr(agent_mod, "settings", RouteSettings())
    monkeypatch.setattr(agent_mod, "_build_executor", lambda db: FakeExecutor())

    client = make_client(db)
    response = client.post(
        "/api/agent/query",
        json={"project_slug": "demo", "query": "hello?"},
    )
    assert response.status_code == 200
    constraints = captured["constraints"]
    assert constraints.allow_external_network is True
    assert constraints.max_steps == 11
    assert constraints.max_tool_calls == 7
    assert constraints.budget_tokens == 30000
    assert constraints.timeout_seconds == 120


def test_agent_invalid_payload_returns_422() -> None:
    """Missing required fields returns 422."""
    db = make_session()
    client = make_client(db, agent_enabled=True)

    response = client.post("/api/agent/query", json={})
    assert response.status_code == 422


def test_agent_response_has_required_shape(monkeypatch) -> None:
    """Response contains all required fields."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.schemas.common import QueryResponse

    class StubRAG:
        def answer(self, db, project_slug, question):
            return QueryResponse(
                answer_markdown="shape test",
                citations=[],
                verification_status="local-only",
            )

    monkeypatch.setattr(
        "app.services.rag_adapter.RAGAdapter.answer",
        lambda self, db, project_slug, question: StubRAG().answer(db, project_slug, question),
    )

    client = make_client(db)
    response = client.post(
        "/api/agent/query",
        json={"project_slug": "demo", "query": "test"},
    )
    assert response.status_code == 200
    data = response.json()
    for field in ("request_id", "session_id", "status", "final_answer", "citations", "steps", "usage"):
        assert field in data, f"Missing field: {field}"
    # usage sub-fields
    for field in ("prompt_tokens", "completion_tokens", "tool_calls", "steps"):
        assert field in data["usage"], f"Missing usage field: {field}"


def test_agent_turns_persist_across_sessions(monkeypatch, tmp_path) -> None:
    """Turns survive across separate DB sessions (simulates real HTTP requests)."""
    from app.db.session import Base as AppBase, get_db as app_get_db

    # Use a file-based DB so two sessions can share the same data
    db_path = str(tmp_path / "agent_persist.db")
    engine1 = create_engine(
        f"sqlite:///{db_path}",
        future=True,
        connect_args={"check_same_thread": False},
    )
    AppBase.metadata.create_all(engine1)
    db1 = sessionmaker(bind=engine1, autoflush=False, autocommit=False, future=True)()

    # Seed project
    project = Project(id="p1", slug="demo", name="Demo")
    db1.add(project)
    db1.commit()

    # --- Request 1: create an agent session ---
    app1 = FastAPI()
    app1.include_router(agent_router, prefix="/api/agent")

    def override_db1() -> Iterator[Session]:
        yield db1

    app1.dependency_overrides[app_get_db] = override_db1
    monkeypatch.setattr(
        "app.services.rag_adapter.RAGAdapter.answer",
        lambda self, db, project_slug, question: _stub_answer("turn 1 answer"),
    )
    client1 = TestClient(app1, raise_server_exceptions=False)
    r1 = client1.post(
        "/api/agent/query",
        json={"project_slug": "demo", "query": "first?", "session_id": "persist-test"},
    )
    assert r1.status_code == 200
    sid = r1.json()["session_id"]
    assert sid == "persist-test"
    db1.close()

    # --- Request 2: new session reads the persisted turns ---
    engine2 = create_engine(
        f"sqlite:///{db_path}",
        future=True,
        connect_args={"check_same_thread": False},
    )
    db2 = sessionmaker(bind=engine2, autoflush=False, autocommit=False, future=True)()

    from app.services.conversation_memory import ConversationMemory
    memory2 = ConversationMemory(db2)
    history = memory2.get_history("persist-test")
    assert len(history) >= 2, f"Expected at least 2 turns, got {len(history)}"
    roles = {t.role for t in history}
    assert "user" in roles
    assert "tool" in roles or "agent" in roles
    db2.close()


# ----------------------------------------------------------------
# v2a tests — route, warnings, needs_clarification
# ----------------------------------------------------------------


def test_agent_response_includes_route_and_warnings(monkeypatch) -> None:
    """Response includes route decision and warnings list."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.schemas.common import QueryResponse

    class StubRAG:
        def answer(self, db, project_slug, question):
            return QueryResponse(
                answer_markdown="v2a test",
                citations=[],
                verification_status="local-only",
            )

    monkeypatch.setattr(
        "app.services.rag_adapter.RAGAdapter.answer",
        lambda self, db, project_slug, question: StubRAG().answer(db, project_slug, question),
    )

    client = make_client(db)
    response = client.post(
        "/api/agent/query",
        json={"project_slug": "demo", "query": "hello?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "route" in data, "Missing 'route' field in response"
    assert "warnings" in data, "Missing 'warnings' field in response"
    assert data["route"] is None or isinstance(data["route"], dict)
    assert isinstance(data["warnings"], list)


def test_needs_clarification_route_in_response(monkeypatch) -> None:
    """Empty query returns needs_clarification route with warning, no error."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    client = make_client(db)
    response = client.post(
        "/api/agent/query",
        json={"project_slug": "demo", "query": "   "},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["route"] is not None
    assert data["route"]["route"] == "needs_clarification"
    assert len(data["warnings"]) > 0
    assert data["status"] == "completed"


# ----------------------------------------------------------------
# v2a rework-002 — invalid constraint validation (422)
# ----------------------------------------------------------------


def test_agent_invalid_constraints_max_steps_zero_returns_422() -> None:
    """max_steps=0 in constraints returns HTTP 422."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    client = make_client(db)
    response = client.post(
        "/api/agent/query",
        json={
            "project_slug": "demo",
            "query": "hello?",
            "constraints": {"max_steps": 0},
        },
    )
    assert response.status_code == 422


def test_agent_invalid_constraints_max_tool_calls_zero_returns_422() -> None:
    """max_tool_calls=0 in constraints returns HTTP 422."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    client = make_client(db)
    response = client.post(
        "/api/agent/query",
        json={
            "project_slug": "demo",
            "query": "hello?",
            "constraints": {"max_tool_calls": 0},
        },
    )
    assert response.status_code == 422


def test_agent_invalid_constraints_timeout_seconds_zero_returns_422() -> None:
    """timeout_seconds=0 in constraints returns HTTP 422."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    client = make_client(db)
    response = client.post(
        "/api/agent/query",
        json={
            "project_slug": "demo",
            "query": "hello?",
            "constraints": {"timeout_seconds": 0},
        },
    )
    assert response.status_code == 422


def test_agent_invalid_constraints_negative_values_returns_422() -> None:
    """Negative max_steps/-1 in constraints returns HTTP 422."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    client = make_client(db)
    response = client.post(
        "/api/agent/query",
        json={
            "project_slug": "demo",
            "query": "hello?",
            "constraints": {"max_steps": -1},
        },
    )
    assert response.status_code == 422


def _stub_answer(text: str):
    from app.schemas.common import QueryResponse
    return QueryResponse(
        answer_markdown=text,
        citations=[],
        verification_status="local-only",
    )


class _DisabledSettings:
    agent_enabled = False


# ----------------------------------------------------------------
# Phase 5 — trace list filter tests
# ----------------------------------------------------------------


def test_list_traces_with_session_id_only_is_backward_compatible() -> None:
    """GET /api/agent/traces?session_id=X still works (backward compat)."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.services.agent_trace_store import AgentTraceStore
    store = AgentTraceStore(db)
    store.persist_run(
        request_id="req-1",
        session_id="filter-sess",
        project_slug="demo",
        query="q1",
        constraints={},
        route="simple_rag",
        steps=[],
        usage=__import__("app.schemas.agent", fromlist=["AgentUsage"]).AgentUsage(),
        final_answer="a1",
        citations=[],
        warnings=[],
        status="completed",
        latency_ms=100,
        provider="local",
        model="local-fallback",
    )
    db.commit()

    client = make_client(db)
    response = client.get("/api/agent/traces?session_id=filter-sess")
    assert response.status_code == 200
    data = response.json()
    assert "traces" in data
    assert len(data["traces"]) >= 1
    assert data["traces"][0]["session_id"] == "filter-sess"


def test_list_traces_with_optional_filters() -> None:
    """GET /api/agent/traces supports project_slug, status, provider, route filters."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.services.agent_trace_store import AgentTraceStore
    from app.schemas.agent import AgentUsage

    store = AgentTraceStore(db)
    store.persist_run(
        request_id="req-a", session_id="s1", project_slug="demo", query="q",
        constraints={}, route="simple_rag",
        steps=[], usage=AgentUsage(), final_answer="a", citations=[], warnings=[],
        status="completed", latency_ms=100, provider="local", model="local-fallback",
    )
    store.persist_run(
        request_id="req-b", session_id="s1", project_slug="other", query="q",
        constraints={}, route="evidence_required",
        steps=[], usage=AgentUsage(), final_answer="b", citations=[], warnings=[],
        status="error", latency_ms=200, provider="external_api", model="deepseek-v4-pro",
    )
    db.commit()

    client = make_client(db)

    # Filter by status
    r1 = client.get("/api/agent/traces?status=error")
    assert r1.status_code == 200
    assert len(r1.json()["traces"]) == 1
    assert r1.json()["traces"][0]["status"] == "error"

    # Filter by provider
    r2 = client.get("/api/agent/traces?provider=external_api")
    assert r2.status_code == 200
    assert len(r2.json()["traces"]) == 1
    assert r2.json()["traces"][0]["provider"] == "external_api"

    # Filter by route
    r3 = client.get("/api/agent/traces?route=evidence_required")
    assert r3.status_code == 200
    assert len(r3.json()["traces"]) == 1
    assert r3.json()["traces"][0]["route"] == "evidence_required"

    # Filter by project_slug
    r4 = client.get("/api/agent/traces?project_slug=other")
    assert r4.status_code == 200
    assert len(r4.json()["traces"]) == 1
    assert r4.json()["traces"][0]["project_slug"] == "other"

    # Combined filter
    r5 = client.get("/api/agent/traces?status=completed&provider=local")
    assert r5.status_code == 200
    assert len(r5.json()["traces"]) == 1


def test_list_traces_no_session_id_returns_all_matching() -> None:
    """GET /api/agent/traces without session_id returns traces matching other filters."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.services.agent_trace_store import AgentTraceStore
    from app.schemas.agent import AgentUsage

    store = AgentTraceStore(db)
    store.persist_run(
        request_id="req-1", session_id="s1", project_slug="demo", query="q",
        constraints={}, route="simple_rag",
        steps=[], usage=AgentUsage(), final_answer="a", citations=[], warnings=[],
        status="completed", latency_ms=100, provider="local", model="local-fallback",
    )
    store.persist_run(
        request_id="req-2", session_id="s2", project_slug="demo", query="q",
        constraints={}, route="simple_rag",
        steps=[], usage=AgentUsage(), final_answer="b", citations=[], warnings=[],
        status="completed", latency_ms=200, provider="local", model="local-fallback",
    )
    db.commit()

    client = make_client(db)
    # With project_slug filter, no session_id needed
    r = client.get("/api/agent/traces?project_slug=demo")
    assert r.status_code == 200
    assert len(r.json()["traces"]) == 2


def test_list_agent_sessions_by_project() -> None:
    """GET /api/agent/sessions lists sessions filtered by project_slug."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.add_all(
        [
            ConversationSession(
                id="sess-a",
                project_slug="demo",
                expires_at=datetime.utcnow() + timedelta(days=30),
            ),
            ConversationSession(
                id="sess-b",
                project_slug="demo",
                expires_at=datetime.utcnow() + timedelta(days=30),
            ),
            ConversationSession(
                id="sess-c",
                project_slug="other",
                expires_at=datetime.utcnow() + timedelta(days=30),
            ),
        ]
    )
    db.commit()

    client = make_client(db)
    r = client.get("/api/agent/sessions?project_slug=demo")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    assert {s["id"] for s in data} == {"sess-a", "sess-b"}
    for session in data:
        assert session["project_slug"] == "demo"
        assert "turn_count" in session
        assert "created_at" in session
        assert "updated_at" in session
        assert "expires_at" in session


def test_get_agent_session_turns_ordered() -> None:
    """GET /api/agent/sessions/{id}/turns returns ordered turns."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.add(
        ConversationSession(
            id="sess-turns",
            project_slug="demo",
            expires_at=datetime.utcnow() + timedelta(days=30),
        )
    )
    db.add_all(
        [
            ConversationTurn(
                id="t1",
                session_id="sess-turns",
                turn_index=0,
                role="user",
                content="hello",
                created_at=datetime.utcnow(),
            ),
            ConversationTurn(
                id="t2",
                session_id="sess-turns",
                turn_index=1,
                role="agent",
                content="hi there",
                created_at=datetime.utcnow(),
            ),
            ConversationTurn(
                id="t3",
                session_id="sess-turns",
                turn_index=2,
                role="tool",
                content="result",
                tool_name="rag.answer",
                step_type="tool_call",
                created_at=datetime.utcnow(),
            ),
        ]
    )
    db.commit()

    client = make_client(db)
    r = client.get("/api/agent/sessions/sess-turns/turns")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 3
    assert [t["turn_index"] for t in data] == [0, 1, 2]
    assert [t["role"] for t in data] == ["user", "agent", "tool"]
    assert data[2]["tool_name"] == "rag.answer"
    assert data[2]["step_type"] == "tool_call"


def test_get_agent_session_turns_unknown_session_returns_404() -> None:
    """GET /api/agent/sessions/{id}/turns returns 404 for an empty unknown session."""
    db = make_session()
    client = make_client(db)
    r = client.get("/api/agent/sessions/no-such-session/turns")
    assert r.status_code == 404
