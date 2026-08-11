from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator

import httpx
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
        def answer(self, db, project_slug, question, document_id=None):
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
        def answer(self, db, project_slug, question, document_id=None):
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


def test_stream_emits_live_route_tokens_and_citations_in_order(monkeypatch) -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()

    from app.schemas.common import Citation, QueryResponse

    monkeypatch.setattr(
        "app.services.rag_adapter.RAGAdapter.answer",
        lambda self, db, project_slug, question, document_id=None: QueryResponse(
            answer_markdown="draft",
            citations=[
                Citation(
                    document_id="d1",
                    chunk_id="c1",
                    score=0.9,
                    excerpt="evidence",
                )
            ],
            verification_status="local-only",
        ),
    )

    def fake_stream(self, *, event_sink, cancel_event=None, **kwargs):
        event_sink("token", {"delta": "流", "model": kwargs["target"].model})
        event_sink("token", {"delta": "式", "model": kwargs["target"].model})
        event_sink("citation", {"index": 0, **kwargs["citations"][0]})
        return {
            "answer_markdown": "流式 [0]",
            "cited_indexes": [0],
            "warnings": [],
            "confidence": 1.0,
            "provider": "local",
            "model": kwargs["target"].model,
        }

    monkeypatch.setattr(
        "app.services.agent_synthesizer.AgentSynthesizer.synthesize_stream",
        fake_stream,
        raising=False,
    )

    client = make_client(db)
    # 9.1 之后单轮/对比路由跳过 synthesize（无流式 token）。要触发
    # synthesize_stream，需要跨轮引用场景：先发一轮请求建立会话历史，
    # 再用含指代词的 query 走同一 session。
    client.post(
        "/api/agent/query",
        json={
            "project_slug": "demo",
            "query": "charmm36m 的 accuracy 是多少",
            "session_id": "stream-sess",
        },
    )
    events = _parse_sse_events(
        client.post(
            "/api/agent/query/stream",
            json={
                "project_slug": "demo",
                "query": "那个的 accuracy 呢？",
                "session_id": "stream-sess",
            },
        ).text
    )
    names = [event["event"] for event in events]

    assert names.count("token") == 2
    assert names.index("route") < names.index("token")
    assert names.index("token") < names.index("citation")
    assert names.index("citation") < names.index("final") < names.index("done")


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
        def answer(self, db, project_slug, question, document_id=None):
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
        def answer(self, db, project_slug, question, document_id=None):
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
        def answer(self, db, project_slug, question, document_id=None):
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


# ---------------------------------------------------------------------------
# Concurrency regression test
# ---------------------------------------------------------------------------


def test_stream_concurrent_requests_do_not_serialize(monkeypatch) -> None:
    """Two concurrent stream requests overlap instead of serializing.

    Before the ``asyncio.to_thread`` fix the synchronous executor call
    blocked the event loop, forcing concurrent requests to queue up behind
    each other.  This test monkeypatches ``AgentExecutor.execute`` to sleep
    for roughly one second and then fires two requests concurrently.  When
    the executor runs off the event loop the total wall time should be
    close to one second (< 1.8 s); when requests serialize it will be
    roughly two seconds.
    """
    # -- in-memory SQLite shared across threads --------------------------------
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    db = SessionFactory()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    # -- monkeypatch AgentExecutor.execute to block ~1 s -----------------------
    from app.schemas.agent import (
        AgentQueryResponse,
        AgentRouteDecision,
        AgentStep,
        AgentUsage,
    )

    def _slow_execute(self, request):
        time.sleep(1.0)
        return AgentQueryResponse(
            request_id="req-x",
            session_id="ses-x",
            status="completed",
            final_answer="concurrent ok",
            steps=[
                AgentStep(step_id=1, step_type="route", summary="routed"),
            ],
            usage=AgentUsage(),
            route=AgentRouteDecision(route="simple_rag"),
            warnings=[],
            trace_id="trace-x",
            answer_provider="local",
            answer_model="local-test",
        )

    monkeypatch.setattr(
        "app.services.agent_executor.AgentExecutor.execute",
        _slow_execute,
    )

    # -- build ASGI app --------------------------------------------------------
    app = FastAPI()
    app.include_router(agent_router, prefix="/api/agent")

    def _override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[get_db] = _override_db

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    # -- fire two concurrent requests via asyncio.run --------------------------
    async def _run_concurrent():
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            t0 = time.perf_counter()
            r1, r2 = await asyncio.gather(
                client.post(
                    "/api/agent/query/stream",
                    json={"project_slug": "demo", "query": "hello 1"},
                ),
                client.post(
                    "/api/agent/query/stream",
                    json={"project_slug": "demo", "query": "hello 2"},
                ),
            )
            elapsed = time.perf_counter() - t0
        return r1, r2, elapsed

    r1, r2, elapsed = asyncio.run(_run_concurrent())

    assert r1.status_code == 200, f"Request 1 returned {r1.status_code}"
    assert r2.status_code == 200, f"Request 2 returned {r2.status_code}"
    assert elapsed < 1.8, (
        f"Expected concurrent wall time < 1.8 s but took {elapsed:.2f} s "
        f"- requests appear to be serialized on the event loop"
    )


# ------------------------------------------------------------------
# 9.3.1/任务9：stream 路径 narrow_context 收窄断言
# ------------------------------------------------------------------


def _fake_settings_local():
    from app.core.config import Settings

    s = Settings()
    s.agent_synthesis_provider = "local"
    s.external_api_enabled = False
    s.external_api_key = None
    s.external_api_model = "gpt-4o-mini"
    s.external_api_base_url = "https://api.openai.com/v1"
    s.external_api_timeout = 90
    return s


class _FakeStreamingOllama:
    """流式 local client：首轮流式缺锚点触发 coverage retry（generate_chat）。"""

    def __init__(self, *, stream_text: str, retry_text: str) -> None:
        self.stream_text = stream_text
        self.retry_text = retry_text
        self.stream_calls: list[dict] = []
        self.retry_calls: list[dict] = []

    def stream_chat(self, **kwargs):
        self.stream_calls.append(kwargs)
        midpoint = max(1, len(self.stream_text) // 2)
        yield {"content": self.stream_text[:midpoint], "model": kwargs["model"], "done": False}
        yield {"content": self.stream_text[midpoint:], "model": kwargs["model"], "done": True}

    def generate_chat(self, **kwargs):
        self.retry_calls.append(kwargs)
        return {
            "content": self.retry_text,
            "model": kwargs["model"],
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }

    def generate_structured(self, *args, **kwargs):
        raise AssertionError("Local synthesis must not request structured JSON")


def test_stream_narrow_context_excludes_items_and_inventory(monkeypatch) -> None:
    """9.3.1/任务9：stream 路径 narrow_context=True 时，首轮流式 prompt
    与 coverage retry 第二次调用均不含 evidence items 摘录与完整
    inventory，结构化 table_facts 保留。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    from app.services.agent_model_router import InferenceTarget
    from app.services.agent_synthesizer import AgentSynthesizer

    ollama = _FakeStreamingOllama(
        stream_text="Standard analysis was performed.",
        retry_text="NMR spectroscopy at 7.5 kcal/mol was used.",
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    result = syn.synthesize_stream(
        query="What technique?",
        route="evidence_required",
        rag_answer="Some analysis.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy at 7.5 kcal/mol."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "full narrative evidence paragraph excluded",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
            "inventory": [
                {
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "table_id": "table-7",
                    "row_count": 2,
                    "source_block_ids": ["sb-1"],
                    "child_ids": ["inv-child-1"],
                    "child_count": 1,
                    "parent_ids": ["p-1"],
                    "row_indices": [0, 1],
                }
            ],
            "table_facts": [
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "value",
                    "value": "21.0",
                    "row_index": 0,
                }
            ],
        },
        target=InferenceTarget(
            profile="generation",
            base_url="http://generation:11434",
            model="qwen3.5:9b",
            context_length=32768,
            reason="test generation",
        ),
        event_sink=lambda name, data: None,
        narrow_context=True,
    )
    # coverage retry 被触发（首轮流式缺锚点），第二次调用同样收窄
    assert ollama.stream_calls and ollama.retry_calls
    stream_prompt = (
        ollama.stream_calls[0]["messages"][0]["content"]
        + "\n"
        + ollama.stream_calls[0]["messages"][1]["content"]
    )
    retry_prompt = (
        ollama.retry_calls[0]["messages"][0]["content"]
        + "\n"
        + ollama.retry_calls[0]["messages"][1]["content"]
    )
    for prompt in (stream_prompt, retry_prompt):
        assert "full narrative evidence paragraph excluded" not in prompt
        assert "inv-child-1" not in prompt
        # 结构化 table_facts 在收窄时仍保留（只去 items/inventory）
        assert "21.0" in prompt
    assert result["provider"] == "local"
