"""JSON/SSE share execution, source pixels, constraints and single finalization."""
import base64
import asyncio
import hashlib
import json
import threading

import pymupdf
import pytest
from sqlalchemy import select

from app.api.agent_routes import _apply_server_constraint_defaults
from app.models.records import AgentTraceRun, ConversationTurn, Document, DocumentChunk, DocumentParseVersion, Project
from app.schemas.adaptive_agent import AdaptiveDecision
from app.schemas.agent import AgentConstraints, AgentQueryRequest
from app.services.agent_decider import AgentDecider
from app.services.conversation_memory import ConversationMemory
from app.services.rag_adapter import RAGAdapter
from app.services.search import QueryService
from test_agent_executor import build_executor_with_rag, make_db
from test_agent_streaming import make_client, _parse_sse_events


COMPLEX = "First find the color in Figure 1, then determine the method."
REAL_DECIDE = AgentDecider.decide


@pytest.fixture
def integration(tmp_path, monkeypatch):
    from app.services import agent_executor as module
    from app.services import search
    from app.core.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "agent_adaptive_enabled", True)
    monkeypatch.setattr(settings, "embedding_revision", "fixture-weights")
    monkeypatch.setattr(settings, "embedding_processor_hash", "fixture-processor")
    monkeypatch.setattr(settings, "canonical_artifacts_dir", tmp_path)
    db = make_db()
    db.add(Project(id="p1", slug="pilot", name="Pilot"))
    db.add(Document(id="d1", project_id="p1", title="Paper", file_name="paper.pdf", raw_path="paper.pdf",
        sha256="test", status="ready", active_parse_version="v5"))
    db.flush()
    bundle = tmp_path / "d1/v5"
    (bundle / "assets").mkdir(parents=True)
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=40, height=30)
        page.get_pixmap().save(bundle / "assets/figure.png")
    digest = hashlib.sha256((bundle / "assets/figure.png").read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps({"document_id": "d1", "version": "v5",
        "assets": [{"path": "assets/figure.png", "sha256": digest}]}))
    (bundle / "figures.json").write_text(json.dumps([{"figure_id": "figure1", "asset_path": "assets/figure.png"}]))
    db.add(DocumentParseVersion(document_id="d1", version_key="v5", artifact_dir=str(bundle), status="active",
        manifest_json={"ingestion_config": {"embedding": {"provider": settings.active_embedding_provider,
            "model": settings.active_embedding_model, "dimensions": 2}}}))
    db.add(DocumentChunk(id="fig", document_id="d1", parse_version="v5", chunk_role="child", block_type="figure",
        ordinal=0, text="Figure 1: legend. ![figure](assets/figure.png)", embedding=[1, 0]))
    db.add(DocumentChunk(id="method", document_id="d1", parse_version="v5", chunk_role="child", block_type="paragraph",
        ordinal=1, text="The method uses attention alignment.", embedding=[1, 0]))
    db.commit()
    monkeypatch.setattr(search._RETRIEVAL_TOKEN_PROVIDER, "estimate_tokens", lambda text: max(1, len(text.split())))
    monkeypatch.setattr("app.services.ai.OllamaClient.embed", lambda self, texts: [[1, 0] for _ in texts])
    posts, observations = [], []
    def post(self, payload):
        posts.append(payload)
        return {"message": {"content": '{"answer_markdown":"Purple. [0]","citations":[0]}'}}
    monkeypatch.setattr("app.services.ai.OllamaClient._post_chat", post)
    def decide(self, observation, target):
        observations.append(observation)
        return AdaptiveDecision(action="answer")
    monkeypatch.setattr(AgentDecider, "decide", decide)
    # Keep the original static synthesis boundary deterministic, without replacing RAG.
    monkeypatch.setattr("app.services.agent_synthesizer.AgentSynthesizer.synthesize", lambda self, **kw:
        {"answer_markdown": kw["rag_answer"], "cited_indexes": [0], "warnings": [],
         "confidence": 1, "provider": "local", "model": "offline"})
    yield db, bundle, posts, observations
    db.close()


def request(query=COMPLEX, **kwargs):
    return AgentQueryRequest(project_slug="pilot", query=query, document_id="d1",
        constraints=AgentConstraints(max_steps=12, max_tool_calls=6, budget_tokens=60000, timeout_seconds=120), **kwargs)


def test_explicit_schema_defaults_are_preserved(monkeypatch):
    import app.api.agent_routes as routes
    monkeypatch.setattr(routes.settings, "agent_max_steps", 20)
    monkeypatch.setattr(routes.settings, "agent_timeout_seconds", 120)
    monkeypatch.setattr(routes.settings, "agent_allow_external_network", True)
    explicit = AgentQueryRequest(project_slug="p", query="q", constraints={"max_steps": 8, "timeout_seconds": 45,
                                                                          "allow_external_network": False})
    resolved = _apply_server_constraint_defaults(explicit)
    assert resolved.constraints.max_steps == 8 and resolved.constraints.timeout_seconds == 45
    assert resolved.constraints.allow_external_network is False
    omitted = _apply_server_constraint_defaults(AgentQueryRequest(project_slug="p", query="q", constraints={}))
    assert omitted.constraints.max_steps == 20 and omitted.constraints.timeout_seconds == 120
    assert explicit.constraints.timeout_seconds == 45  # Input remains unchanged.


@pytest.mark.parametrize("query,enabled", [("What is the method?", True), (COMPLEX, False),
                                           ("Compare the two methods", True)])
def test_non_adaptive_routes_never_plan(integration, monkeypatch, query, enabled):
    import app.services.agent_executor as module
    db, _, _, observations = integration
    monkeypatch.setattr(module.settings, "agent_adaptive_enabled", enabled)
    response = build_executor_with_rag(db, RAGAdapter()).execute(request(query))
    assert observations == []
    assert response.metadata["execution_mode"] == "static"


def test_complex_adaptive_uses_pixels_and_writes_one_final_turn_trace(integration):
    db, bundle, posts, observations = integration
    response = build_executor_with_rag(db, RAGAdapter()).execute(request(session_id="single"))
    assert response.status == "completed" and len(observations) == 1
    assert response.metadata["execution_mode"] == "adaptive"
    assert response.metadata["stop_reason"] == "answer"
    assert response.metadata["verification_state"] == "passed"
    assert response.metadata["verification_scope"] == "structural"
    assert response.metadata["retrieval_backend"] != "not_queried"
    sent = posts[0]["messages"][1]["images"]
    assert base64.b64decode(sent[0]) == (bundle / "assets/figure.png").read_bytes()
    assert response.citations[0].chunk_id == "fig" and response.citations[0].parse_version == "v5"
    assert len(db.scalars(select(ConversationTurn).where(ConversationTurn.session_id == "single",
                                                       ConversationTurn.role == "agent")).all()) == 1
    assert len(db.scalars(select(AgentTraceRun).where(AgentTraceRun.session_id == "single")).all()) == 1
    assert len(response.steps) == response.usage.steps <= 12
    assert response.usage.tool_calls == 3  # retrieve, answer, verify; planner is a model call.
    assert [s.step_id for s in response.steps] == list(range(len(response.steps)))


def test_adaptive_answer_enters_shared_generation_lease(integration, monkeypatch):
    from app.services.model_runtime import ModelRuntime
    db, _, posts, _ = integration
    executor = build_executor_with_rag(db, RAGAdapter())
    runtime = ModelRuntime({'generation':1})
    executor._model_runtime = runtime
    active_during_posts = []
    def post(self, payload):
        active_during_posts.append(runtime.snapshot()['generation']['active'])
        return {'message':{'content':'{"answer_markdown":"Purple. [0]","citations":[0]}'}}
    monkeypatch.setattr('app.services.ai.OllamaClient._post_chat', post)
    response = executor.execute(request())
    assert response.status == 'completed'
    assert active_during_posts == [1]
    assert runtime.snapshot()['generation']['active'] == 0


def test_text_and_visual_cross_turn_keep_session_history_without_old_intent(integration):
    db, _, posts, observations = integration
    executor = build_executor_with_rag(db, RAGAdapter())
    executor.execute(request(session_id="history"))
    executor.execute(request("First determine the method, then find its objective.", session_id="history"))
    assert len(observations) == 2
    assert "Figure 1" in observations[1].conversation_summary
    assert "images" not in posts[-1]["messages"][1]
    assert "conversation" in posts[-1]["messages"][1]["content"].lower() or "Purple" in posts[-1]["messages"][1]["content"]


def test_adaptive_supplement_tool_receives_frozen_map(integration, monkeypatch):
    db, _, _, observations = integration
    recorded = []
    prepare = RAGAdapter.prepare_evidence
    def capture(self, db, project, question, **kwargs):
        recorded.append(kwargs.get("parse_version_map"))
        return prepare(self, db, project, question, **kwargs)
    monkeypatch.setattr(RAGAdapter, "prepare_evidence", capture)
    monkeypatch.setattr(AgentDecider, "decide", lambda self, obs, target:
        AdaptiveDecision(action="retrieve", query="attention alignment", document_id="d1"))
    response = build_executor_with_rag(db, RAGAdapter()).execute(request())
    assert len(recorded) >= 2
    assert recorded[1] == {"d1": "v5"}
    assert response.metadata["execution_mode"] == "adaptive"


def test_adaptive_budget_stop_never_looks_completed(integration):
    db, _, _, observations = integration
    low = request().model_copy(update={"constraints": AgentConstraints(max_steps=3)})
    response = build_executor_with_rag(db, RAGAdapter()).execute(low)
    assert response.status != "completed" and observations == []
    assert response.metadata["verification_state"] == "skipped"
    assert response.metadata["stop_reason"] == "step_limit"
    assert len(response.steps) <= 3


def test_sse_and_json_share_adaptive_metadata_and_single_finalization(integration):
    db, _, _, observations = integration
    client = make_client(db)
    sync = client.post("/api/agent/query", json=request(session_id="json").model_dump()).json()
    stream = client.post("/api/agent/query/stream", json=request(session_id="sse").model_dump())
    final = next(e["data"] for e in _parse_sse_events(stream.text) if e["event"] == "final")
    for result in (sync, final):
        assert result["metadata"]["execution_mode"] == "adaptive"
        assert result["metadata"]["stop_reason"] == "answer"
        assert result["metadata"]["verification_executed"] is True
    assert sync["citations"] == final["citations"]
    assert len(observations) == 2
    assert len(db.scalars(select(ConversationTurn).where(ConversationTurn.session_id == "sse",
                                                       ConversationTurn.role == "agent")).all()) == 1


def test_sse_cancellation_reaches_adaptive_planner_budget(integration, monkeypatch):
    from app.services.execution_budget import current_execution_budget
    db, _, _, observations = integration
    executor = build_executor_with_rag(db, RAGAdapter())
    cancelled = threading.Event()
    executor._cancel_event = cancelled
    def decide(self, obs, target):
        cancelled.set()
        current_execution_budget().check_deadline()
    monkeypatch.setattr(AgentDecider, "decide", decide)
    response = executor.execute(request())
    assert response.status == "timeout"
    assert response.metadata["stop_reason"] == "cancelled"
    assert response.metadata["verification_state"] == "skipped"


def test_adaptive_trace_does_not_contain_observation_or_model_error(integration, monkeypatch):
    db, _, _, observations = integration
    monkeypatch.setattr(AgentDecider, "decide", lambda self, obs, target:
        (_ for _ in ()).throw(RuntimeError("private document text /home/secret")))
    response = build_executor_with_rag(db, RAGAdapter()).execute(request())
    assert response.metadata["stop_reason"] == "decision_invalid"
    assert "private document text" not in str([s.model_dump() for s in response.steps])
    assert not any("observation" in s.metadata for s in response.steps)


def test_adaptive_attachments_are_current_session_only(integration):
    from app.models.records import SessionAttachment, SessionAttachmentChunk
    db, _, _, observations = integration
    memory = ConversationMemory(db)
    for sid in ("owned", "foreign"):
        memory.touch_session(sid, project_slug="pilot", document_id="d1", ttl_days=1)
        db.add(SessionAttachment(id=sid, session_id=sid, project_id="p1", file_name="note.txt",
            storage_path="unused", sha256=sid, byte_size=40))
        db.flush()
        db.add(SessionAttachmentChunk(id=sid + "-chunk", attachment_id=sid, ordinal=0,
            text="Figure method additional " + sid + " session evidence"))
    db.commit()
    response = build_executor_with_rag(db, RAGAdapter()).execute(request(session_id="owned"))
    assert response.status == "completed"
    assert {e["attachment_id"] for e in observations[0].evidence if e["attachment_id"]} == {"owned"}


@pytest.mark.asyncio
async def test_generator_close_cancels_running_adaptive_request(integration, monkeypatch):
    from app.api.agent_routes import agent_query_stream
    from app.services.execution_budget import current_execution_budget
    db, _, _, _ = integration
    started, ended, force_release = threading.Event(), threading.Event(), threading.Event()
    captured = []
    def decide(self, obs, target):
        budget = current_execution_budget()
        captured.append(budget.cancel_event)
        started.set()
        try:
            for _ in range(100):
                if budget.cancel_event.wait(.01) or force_release.is_set():
                    break
            budget.check_deadline()
            return AdaptiveDecision(action="abstain")
        finally:
            ended.set()
    monkeypatch.setattr(AgentDecider, "decide", decide)
    class ConnectedRequest:
        async def is_disconnected(self):
            return False
    response = await agent_query_stream(request(), ConnectedRequest(), db=db, current_user=None)
    stream = response.body_iterator
    try:
        await anext(stream)  # start
        await anext(stream)  # heartbeat
        await anext(stream)  # route from worker
        assert await asyncio.to_thread(started.wait, 1)
        await stream.aclose()  # ASGI cancellation/early consumer close, not explicit polling.
        assert captured[0].is_set()
        assert await asyncio.to_thread(ended.wait, 1)
    finally:
        force_release.set()
        await asyncio.to_thread(ended.wait, 2)


def test_adaptive_planner_cancel_releases_real_model_queue(integration, monkeypatch):
    import time
    from app.services.model_runtime import ModelRuntime
    db, _, _, _ = integration
    executor = build_executor_with_rag(db, RAGAdapter())
    runtime = ModelRuntime({"generation": 1})
    executor._model_runtime = runtime
    executor._cancel_event = threading.Event()
    monkeypatch.setattr(AgentDecider, "decide", REAL_DECIDE)
    results = []
    with runtime.acquire("generation"):
        worker = threading.Thread(target=lambda: results.append(executor.execute(request())))
        worker.start()
        deadline = time.monotonic() + 2
        while not runtime.snapshot()["generation"]["queued"] and time.monotonic() < deadline:
            time.sleep(.01)
        assert runtime.snapshot()["generation"]["queued"] == 1
        executor._cancel_event.set()
        worker.join(2)
        assert not worker.is_alive()
    assert runtime.snapshot()["generation"]["active"] == runtime.snapshot()["generation"]["queued"] == 0
    assert results[0].metadata["stop_reason"] == "cancelled"
