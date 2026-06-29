from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.agent_routes import agent_router
from app.db.session import Base, get_db
from app.models.records import Project


def make_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def make_client(db: Session) -> TestClient:
    app = FastAPI()
    app.include_router(agent_router, prefix="/api/agent")

    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[get_db] = override_db
    return TestClient(app, raise_server_exceptions=False)


def _parse_sse_events(response_text: str) -> list[dict]:
    """Parse text/event-stream into a list of event dicts."""
    events = []
    current_event = None
    for line in response_text.split("\n"):
        line = line.strip()
        if not line:
            if current_event:
                events.append(current_event)
                current_event = None
            continue
        if line.startswith("event:"):
            current_event = current_event or {}
            current_event["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            current_event = current_event or {}
            try:
                current_event["data"] = json.loads(data_str)
            except json.JSONDecodeError:
                current_event["data"] = data_str
    if current_event:
        events.append(current_event)
    return events


def test_stream_returns_200_with_valid_payload(monkeypatch) -> None:
    """POST /api/agent/query/stream returns 200 with text/event-stream."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.schemas.common import QueryResponse

    class StubRAG:
        def answer(self, db, project_slug, question):
            return QueryResponse(
                answer_markdown="stream answer",
                citations=[],
                verification_status="local-only",
            )

    monkeypatch.setattr(
        "app.services.rag_adapter.RAGAdapter.answer",
        lambda self, db, project_slug, question: StubRAG().answer(db, project_slug, question),
    )
    # Disable synthesis for test speed
    monkeypatch.setattr(
        "app.services.agent_synthesizer.AgentSynthesizer.synthesize",
        lambda self, **kw: {
            "answer_markdown": kw["rag_answer"],
            "cited_indexes": [],
            "warnings": [],
            "confidence": 1.0,
            "provider": "local",
            "model": "local-fallback",
        },
    )

    client = make_client(db)
    response = client.post(
        "/api/agent/query/stream",
        json={"project_slug": "demo", "query": "hello?"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


def test_stream_has_expected_event_names(monkeypatch) -> None:
    """Stream events include start, step, final, and done."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.schemas.common import QueryResponse

    class StubRAG:
        def answer(self, db, project_slug, question):
            return QueryResponse(
                answer_markdown="stream answer",
                citations=[],
                verification_status="local-only",
            )

    monkeypatch.setattr(
        "app.services.rag_adapter.RAGAdapter.answer",
        lambda self, db, project_slug, question: StubRAG().answer(db, project_slug, question),
    )
    monkeypatch.setattr(
        "app.services.agent_synthesizer.AgentSynthesizer.synthesize",
        lambda self, **kw: {
            "answer_markdown": kw["rag_answer"],
            "cited_indexes": [],
            "warnings": [],
            "confidence": 1.0,
            "provider": "local",
            "model": "local-fallback",
        },
    )

    client = make_client(db)
    response = client.post(
        "/api/agent/query/stream",
        json={"project_slug": "demo", "query": "hello?"},
    )
    events = _parse_sse_events(response.text)
    event_names = [e["event"] for e in events if "event" in e]
    assert "start" in event_names, f"Missing 'start' event in {event_names}"
    assert "step" in event_names, f"Missing 'step' event in {event_names}"
    assert "final" in event_names, f"Missing 'final' event in {event_names}"
    assert "done" in event_names, f"Missing 'done' event in {event_names}"


def test_stream_error_path_emits_error_then_done(monkeypatch) -> None:
    """When AgentExecutor raises, the stream emits error then done."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    # Make executor fail
    monkeypatch.setattr(
        "app.services.agent_executor.AgentExecutor.execute",
        lambda self, request: (_ for _ in ()).throw(RuntimeError("simulated failure")),
    )

    client = make_client(db)
    response = client.post(
        "/api/agent/query/stream",
        json={"project_slug": "demo", "query": "hello?"},
    )
    events = _parse_sse_events(response.text)
    event_names = [e["event"] for e in events if "event" in e]
    assert "error" in event_names, f"Missing 'error' event in {event_names}"
    assert "done" in event_names, f"Missing 'done' event in {event_names}"
    # error should come before done
    error_idx = event_names.index("error")
    done_idx = event_names.index("done")
    assert error_idx < done_idx, "error event must come before done event"


def test_stream_final_event_has_trace_id(monkeypatch) -> None:
    """The final event includes trace_id in its data payload."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.schemas.common import QueryResponse

    class StubRAG:
        def answer(self, db, project_slug, question):
            return QueryResponse(
                answer_markdown="stream answer",
                citations=[],
                verification_status="local-only",
            )

    monkeypatch.setattr(
        "app.services.rag_adapter.RAGAdapter.answer",
        lambda self, db, project_slug, question: StubRAG().answer(db, project_slug, question),
    )
    monkeypatch.setattr(
        "app.services.agent_synthesizer.AgentSynthesizer.synthesize",
        lambda self, **kw: {
            "answer_markdown": kw["rag_answer"],
            "cited_indexes": [],
            "warnings": [],
            "confidence": 1.0,
            "provider": "local",
            "model": "local-fallback",
        },
    )

    client = make_client(db)
    response = client.post(
        "/api/agent/query/stream",
        json={"project_slug": "demo", "query": "hello?"},
    )
    events = _parse_sse_events(response.text)
    final_events = [e for e in events if e.get("event") == "final"]
    assert len(final_events) >= 1, "Expected at least one 'final' event"
    final_data = final_events[0].get("data", {})
    assert "trace_id" in final_data, f"Missing 'trace_id' in final event: {final_data}"


def test_stream_needs_clarification_works(monkeypatch) -> None:
    """Stream also works for needs_clarification route."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    client = make_client(db)
    response = client.post(
        "/api/agent/query/stream",
        json={"project_slug": "demo", "query": "   "},
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    event_names = [e["event"] for e in events if "event" in e]
    assert "start" in event_names
    assert "done" in event_names


# ----------------------------------------------------------------
# Phase 5 enterprise hardening tests
# ----------------------------------------------------------------


def test_stream_has_heartbeat_event(monkeypatch) -> None:
    """The SSE stream includes a heartbeat event with a timestamp payload."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.schemas.common import QueryResponse

    class StubRAG:
        def answer(self, db, project_slug, question):
            return QueryResponse(
                answer_markdown="hb answer",
                citations=[],
                verification_status="local-only",
            )

    monkeypatch.setattr(
        "app.services.rag_adapter.RAGAdapter.answer",
        lambda self, db, project_slug, question: StubRAG().answer(db, project_slug, question),
    )
    monkeypatch.setattr(
        "app.services.agent_synthesizer.AgentSynthesizer.synthesize",
        lambda self, **kw: {
            "answer_markdown": kw["rag_answer"],
            "cited_indexes": [],
            "warnings": [],
            "confidence": 1.0,
            "provider": "local",
            "model": "local-fallback",
        },
    )

    client = make_client(db)
    response = client.post(
        "/api/agent/query/stream",
        json={"project_slug": "demo", "query": "hello?"},
    )
    events = _parse_sse_events(response.text)
    event_names = [e["event"] for e in events if "event" in e]
    assert "heartbeat" in event_names, f"Missing 'heartbeat' event in {event_names}"

    # Heartbeat should come after start
    hb_events = [e for e in events if e.get("event") == "heartbeat"]
    assert len(hb_events) >= 1
    hb_data = hb_events[0].get("data", {})
    assert "timestamp" in hb_data, f"Heartbeat missing timestamp: {hb_data}"


def test_stream_error_event_has_message_and_error_type(monkeypatch) -> None:
    """SSE error events consistently include 'message' and 'error_type'."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    # Make executor fail
    monkeypatch.setattr(
        "app.services.agent_executor.AgentExecutor.execute",
        lambda self, request: (_ for _ in ()).throw(ValueError("bad input")),
    )

    client = make_client(db)
    response = client.post(
        "/api/agent/query/stream",
        json={"project_slug": "demo", "query": "hello?"},
    )
    events = _parse_sse_events(response.text)
    error_events = [e for e in events if e.get("event") == "error"]
    assert len(error_events) >= 1, "Expected at least one error event"

    for err in error_events:
        data = err.get("data", {})
        assert "message" in data, f"Error missing 'message': {data}"
        assert "error_type" in data, f"Error missing 'error_type': {data}"


def test_stream_final_event_has_enriched_summary_fields(monkeypatch) -> None:
    """Final event includes trace_id, provider, model, tool_names, step_summary."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    from app.schemas.common import QueryResponse

    class StubRAG:
        def answer(self, db, project_slug, question):
            return QueryResponse(
                answer_markdown="enriched answer",
                citations=[],
                verification_status="local-only",
            )

    monkeypatch.setattr(
        "app.services.rag_adapter.RAGAdapter.answer",
        lambda self, db, project_slug, question: StubRAG().answer(db, project_slug, question),
    )
    monkeypatch.setattr(
        "app.services.agent_synthesizer.AgentSynthesizer.synthesize",
        lambda self, **kw: {
            "answer_markdown": kw["rag_answer"],
            "cited_indexes": [],
            "warnings": [],
            "confidence": 1.0,
            "provider": "local",
            "model": "local-fallback",
        },
    )

    client = make_client(db)
    response = client.post(
        "/api/agent/query/stream",
        json={"project_slug": "demo", "query": "hello?"},
    )
    events = _parse_sse_events(response.text)
    final_events = [e for e in events if e.get("event") == "final"]
    assert len(final_events) >= 1, "Expected at least one 'final' event"
    final_data = final_events[0].get("data", {})

    # Backward compatibility — full response fields still present
    for field in ("request_id", "session_id", "status", "final_answer", "steps", "usage"):
        assert field in final_data, f"Missing legacy field '{field}' in final event"

    # New enriched top-level fields
    assert "trace_id" in final_data, "Missing 'trace_id' in final event"
    assert "provider" in final_data, "Missing 'provider' in final event"
    assert "model" in final_data, "Missing 'model' in final event"
    assert "tool_names" in final_data, "Missing 'tool_names' in final event"
    assert isinstance(final_data["tool_names"], list)
    assert "step_summary" in final_data, "Missing 'step_summary' in final event"
    assert isinstance(final_data["step_summary"], list)

    # step_summary items have expected shape
    for item in final_data["step_summary"]:
        assert "step_id" in item
        assert "step_type" in item
        assert "summary" in item
