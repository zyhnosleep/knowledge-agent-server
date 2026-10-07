from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base
from app.models.records import Project
from app.schemas.agent import AgentQueryRequest, EvidencePack, ToolSpec
from app.schemas.common import Citation, QueryResponse
from app.services.agent_executor import AgentExecutor
from app.services.agent_trace_store import AgentTraceStore
from app.services.conversation_memory import ConversationMemory
from app.services.rag_adapter import RAGAdapter
from app.services.tool_registry import ToolRegistry


def make_db() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def make_executor(db: Session, *, rag_answer_text: str = "test answer") -> AgentExecutor:
    """Build an AgentExecutor with a RAGAdapter that returns a fixed answer."""
    class StubRAG:
        def answer(self, db, project_slug, question, document_id=None):
            from app.schemas.common import QueryResponse, Citation
            citations = []
            if rag_answer_text:
                citations = [
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.92,
                        excerpt="sample excerpt",
                    )
                ]
            return QueryResponse(
                answer_markdown=rag_answer_text,
                citations=citations,
                verification_status="local-only",
            )

    rag = StubRAG()
    tools = ToolRegistry()
    tools._register_builtins(rag)
    tools.register(
        ToolSpec(
            name="answer.synthesize",
            description="deterministic test synthesizer",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "rag_answer": {"type": "string"},
                    "citations": {"type": "array"},
                },
                "required": ["query", "rag_answer"],
            },
            side_effect_level="none",
        ),
        lambda args, ctx: {
            "answer_markdown": args.get("rag_answer", ""),
            "cited_indexes": list(range(len(args.get("citations") or []))),
            "warnings": [],
            "confidence": 1.0,
            "provider": "local",
            "model": "local-fallback",
        },
    )
    memory = ConversationMemory(db)
    trace_store = AgentTraceStore(db)
    return AgentExecutor(rag=rag, tools=tools, memory=memory, db=db, trace_store=trace_store)


def build_executor_with_rag(db: Session, rag) -> AgentExecutor:
    tools = ToolRegistry()
    tools._register_builtins(rag)
    return AgentExecutor(
        rag=rag,
        tools=tools,
        memory=ConversationMemory(db),
        db=db,
        trace_store=AgentTraceStore(db),
        synthesizer=None,
    )


def test_execute_commits_session_turn_before_rag(monkeypatch) -> None:
    db = make_db()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    commit_count = 0
    commit_counts_seen: list[int] = []
    real_commit = db.commit

    def tracked_commit() -> None:
        nonlocal commit_count
        commit_count += 1
        real_commit()

    monkeypatch.setattr(db, "commit", tracked_commit)

    class InspectingRAG:
        def retrieve_evidence(
            self, db, project_slug, question, *, limit=15, document_id=None
        ):
            return EvidencePack(status="empty", items=[])

        def answer(self, db, project_slug, question, document_id=None):
            commit_counts_seen.append(commit_count)
            return QueryResponse(
                answer_markdown="ok",
                citations=[],
                verification_status="local-only",
            )

    executor = build_executor_with_rag(db, InspectingRAG())
    response = executor.execute(
        AgentQueryRequest(project_slug="demo", query="hello", session_id="commit-sess")
    )

    assert response.status == "completed"
    assert commit_counts_seen and commit_counts_seen[0] >= 1


def test_generation_target_is_recorded_and_passed_to_synthesis() -> None:
    db = make_db()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    executor = make_executor(db, rag_answer_text="test answer")
    captured: dict = {}

    def routed_synthesis(args, ctx):
        captured.update(args["target"])
        return {
            "answer_markdown": "generation answer [0]",
            "cited_indexes": [0],
            "warnings": [],
            "confidence": 1.0,
            "provider": "local",
            "model": args["target"]["model"],
        }

    executor._tools.get("answer.synthesize")["handler"] = routed_synthesis
    response = executor.execute(
        _cross_turn_request(db, session_id="generation-session")
    )

    route_step = next(step for step in response.steps if step.step_type == "route")
    assert "requested_answer_mode" not in route_step.metadata
    assert route_step.metadata["inference_profile"] == "generation"
    assert route_step.metadata["inference_model"] == "qwen3.5:9b"
    assert captured["context_length"] == 32768
    assert response.answer_model == "qwen3.5:9b"


def test_generation_is_wrapped_in_selected_model_lease() -> None:
    db = make_db()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    executor = make_executor(db, rag_answer_text="test answer")
    events: list[str] = []

    class RecordingRuntime:
        @contextmanager
        def acquire(self, profile, **kwargs):
            events.append(f"acquire:{profile}")
            try:
                yield
            finally:
                events.append(f"release:{profile}")

    executor._model_runtime = RecordingRuntime()
    response = executor.execute(
        _cross_turn_request(db, session_id="leased-session")
    )

    assert response.status == "completed"
    assert events == ["acquire:generation", "release:generation"]


def test_concurrent_sessions_release_initial_write_transaction(tmp_path) -> None:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'agent-concurrency.db'}",
        future=True,
        connect_args={"check_same_thread": False, "timeout": 0.2},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    with session_factory() as setup_db:
        setup_db.add(Project(id="p1", slug="demo", name="Demo"))
        setup_db.commit()

    first_in_rag = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    errors: list[BaseException] = []
    responses = {}

    class BlockingRAG:
        def retrieve_evidence(
            self, db, project_slug, question, *, limit=15, document_id=None
        ):
            return EvidencePack(status="empty", items=[])

        def answer(self, db, project_slug, question, document_id=None):
            if question == "first":
                first_in_rag.set()
                if not release_first.wait(timeout=5.0):
                    raise TimeoutError("test did not release first request")
            return QueryResponse(
                answer_markdown=f"answer for {question}",
                citations=[],
                verification_status="local-only",
            )

    rag = BlockingRAG()

    def run_query(name: str, session_id: str, finished=None) -> None:
        db = session_factory()
        try:
            executor = build_executor_with_rag(db, rag)
            responses[name] = executor.execute(
                AgentQueryRequest(
                    project_slug="demo", query=name, session_id=session_id
                )
            )
            db.commit()
        except BaseException as exc:
            errors.append(exc)
            db.rollback()
        finally:
            db.close()
            if finished is not None:
                finished.set()

    first_thread = threading.Thread(
        target=run_query, args=("first", "concurrent-first"), daemon=True
    )
    second_thread = threading.Thread(
        target=run_query,
        args=("second", "concurrent-second", second_finished),
        daemon=True,
    )

    first_thread.start()
    assert first_in_rag.wait(timeout=2.0)
    second_thread.start()
    assert second_finished.wait(timeout=2.0)
    release_first.set()
    first_thread.join(timeout=2.0)
    second_thread.join(timeout=2.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    assert responses["first"].status == "completed"
    assert responses["second"].status == "completed"


def test_execute_returns_completed_status() -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    assert response.status == "completed"
    assert response.final_answer == "test answer"
    assert len(response.citations) == 1
    assert response.citations[0].document_id == "d1"
    # v3 fields
    assert response.trace_id is not None
    assert response.answer_provider is not None
    assert response.answer_model is not None


def test_execute_persists_final_answer_citations_in_history() -> None:
    """Restored conversation turns keep final-answer citation metadata."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    response = executor.execute(
        AgentQueryRequest(project_slug="demo", query="hello?", session_id="cite-sess")
    )

    history = ConversationMemory(db).get_history(response.session_id)
    final_turns = [
        turn
        for turn in history
        if turn.role == "agent" and turn.step_type == "finalize"
    ]
    assert final_turns
    assert final_turns[-1].citations
    assert final_turns[-1].citations[0]["document_id"] == "d1"


def test_execute_has_route_and_tool_and_finalize_steps() -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    # v2a: route (0), rag.answer (1), answer.verify (2), finalize (3)
    assert len(response.steps) >= 3
    assert response.steps[0].step_type == "route"
    assert response.steps[1].step_type == "tool_call"
    assert response.steps[1].tool_name == "rag.answer"
    assert response.steps[1].tool_ok is True
    assert response.steps[-1].step_type == "finalize"


def test_execute_generates_session_id() -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    assert response.session_id.startswith("sess_")


def test_execute_respects_provided_session_id() -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request = AgentQueryRequest(
        project_slug="demo", query="hello?", session_id="my-session-123"
    )
    response = executor.execute(request)
    assert response.session_id == "my-session-123"


def test_execute_generates_request_id() -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    assert len(response.request_id) > 0


def test_execute_records_usage() -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    # v2a: route + rag.answer + verify + finalize
    assert response.usage.steps >= 3
    assert response.usage.tool_calls >= 2  # rag.answer + answer.verify
    # This deterministic fixture sends no model requests: don't invent usage.
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 0
    assert response.usage.model_requests == 0
    assert response.usage.usage_source == 'none'


def test_execute_persists_turns_in_memory() -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)
    db.commit()

    memory = ConversationMemory(db)
    history = memory.get_history(response.session_id)
    # user + tool + agent turns
    assert len(history) >= 3
    roles = {t.role for t in history}
    assert "user" in roles
    assert "tool" in roles
    assert "agent" in roles


def test_execute_resumes_existing_session() -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request1 = AgentQueryRequest(project_slug="demo", query="first?", session_id="s1")
    r1 = executor.execute(request1)
    db.commit()

    request2 = AgentQueryRequest(project_slug="demo", query="second?", session_id="s1")
    r2 = executor.execute(request2)
    db.commit()

    assert r1.session_id == r2.session_id == "s1"
    memory = ConversationMemory(db)
    history = memory.get_history("s1")
    # 2 user turns + 2 tool turns + 2 agent turns
    assert len(history) >= 5


def test_follow_up_question_uses_previous_user_question_for_retrieval() -> None:
    db = make_db()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    received_questions: list[str] = []

    class RecordingRAG:
        def retrieve_evidence(
            self, db, project_slug, question, *, limit=15, document_id=None
        ):
            received_questions.append(question)
            return EvidencePack(status="empty", items=[])

        def answer(self, db, project_slug, question, document_id=None):
            received_questions.append(question)
            return QueryResponse(
                answer_markdown="ok",
                citations=[],
                verification_status="local-only",
            )

    executor = build_executor_with_rag(db, RecordingRAG())
    executor.execute(
        AgentQueryRequest(
            project_slug="demo",
            document_id="doc-1",
            query="请说明文章采用的研究方法",
            session_id="follow-up-session",
        )
    )
    received_questions.clear()

    executor.execute(
        AgentQueryRequest(
            project_slug="demo",
            document_id="doc-1",
            query="请详细讲解一下",
            session_id="follow-up-session",
        )
    )

    assert received_questions
    assert all("请说明文章采用的研究方法" in question for question in received_questions)
    assert all("请详细讲解一下" in question for question in received_questions)


def test_execute_empty_answer() -> None:
    """Executor still succeeds when RAG returns empty answer."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="")
    request = AgentQueryRequest(project_slug="demo", query="empty?")
    response = executor.execute(request)

    assert response.status == "error"
    assert response.metadata['verification_state'] == 'skipped'
    assert response.final_answer == ""
    assert response.citations == []
    # v2a: steps[1] is the rag.answer tool call (steps[0] is route)
    rag_step = next(s for s in response.steps if s.tool_name == "rag.answer")
    assert rag_step.tool_ok is True


def test_execute_error_path_returns_error_status(monkeypatch) -> None:
    """When call_tool raises an unexpected exception, status is 'error'."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    # Make call_tool raise a non-ToolError exception (simulates e.g. a
    # database failure that escapes the normal error path)
    monkeypatch.setattr(
        executor._tools,
        "call_tool",
        lambda name, args=None, ctx=None: (_ for _ in ()).throw(
            RuntimeError("simulated infrastructure failure")
        ),
    )
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    assert response.status == "error"
    assert "simulated infrastructure failure" in response.steps[-1].summary


def test_execute_timeout_path_returns_timeout_status(monkeypatch) -> None:
    """When execution exceeds timeout_seconds, status is 'timeout'."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    import time as time_mod
    from itertools import chain, repeat

    # Simulate time passing: early calls return 0, then all later calls
    # return far future. Keep the tail infinite so adding trace steps
    # cannot exhaust the iterator.
    calls = chain(repeat(0.0, 8), repeat(99999.0))
    monkeypatch.setattr(time_mod, "monotonic", lambda: next(calls))

    executor = make_executor(db)
    request = AgentQueryRequest(
        project_slug="demo",
        query="hello?",
        constraints={"timeout_seconds": 5},
    )
    response = executor.execute(request)
    assert response.status == "timeout"


# ----------------------------------------------------------------
# v2a tests — route, verify, retry, warnings
# ----------------------------------------------------------------


def test_agentstep_has_metadata_field() -> None:
    """AgentStep can carry metadata dict."""
    from app.schemas.agent import AgentStep

    step = AgentStep(
        step_id=0,
        step_type="route",
        summary="test",
        metadata={"route": "simple_rag", "reason": "default"},
    )
    assert step.metadata == {"route": "simple_rag", "reason": "default"}


def test_response_includes_route_and_warnings() -> None:
    """AgentQueryResponse carries route decision and warnings list."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    assert response.route is not None
    assert response.route.route == "simple_rag"
    assert isinstance(response.warnings, list)


def test_needs_clarification_route_skips_rag() -> None:
    """Empty/whitespace query → needs_clarification, no RAG call, warning returned."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="should not be called")
    request = AgentQueryRequest(project_slug="demo", query="   ")
    response = executor.execute(request)

    assert response.route is not None
    assert response.route.route == "needs_clarification"
    # No RAG answer produced — answer should be empty
    assert response.final_answer == ""
    # Should have a useful warning
    assert any("clarif" in w.lower() or "empty" in w.lower() for w in response.warnings)
    # Only route + finalize steps, no tool call step
    assert response.usage.tool_calls == 0


def test_verify_step_is_included_in_trace() -> None:
    """Normal execution includes an answer.verify step between rag.answer and finalize."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    assert response.status == "completed"
    tool_names = [s.tool_name for s in response.steps if s.tool_name]
    assert "rag.answer" in tool_names
    assert "answer.verify" in tool_names


def test_route_step_is_first() -> None:
    """The first step is always a route step."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    assert response.steps[0].step_type == "route"
    assert response.steps[0].metadata.get("route") is not None


def test_retry_when_verify_recommends_within_limits() -> None:
    """When verify recommends retry and route allows it, perform one more rag.answer."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    # Use a query that routes to evidence_required (has retries=1)
    executor = make_executor(db, rag_answer_text="")  # empty answer triggers retry
    request = AgentQueryRequest(project_slug="demo", query="cite this source")
    response = executor.execute(request)

    # Should have called rag.answer at least twice (original + retry)
    rag_steps = [s for s in response.steps if s.tool_name == "rag.answer"]
    # When answer is empty, verify recommends retry, and evidence_required has max_retries=1
    assert len(rag_steps) >= 2, f"Expected at least 2 rag.answer calls, got {len(rag_steps)}"
    assert response.usage.tool_calls >= 2  # rag.answer * 2 + answer.verify


def test_empty_answer_retries_even_when_route_max_retries_zero() -> None:
    """空答案强制重试（T3）：simple_rag（max_retries=0）下空答案也必须重试一次。"""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    # "general question" 路由 simple_rag（max_retries=0）：修复前空答案不重试
    executor = make_executor(db, rag_answer_text="")  # empty answer triggers retry
    request = AgentQueryRequest(project_slug="demo", query="general question")
    response = executor.execute(request)

    rag_steps = [s for s in response.steps if s.tool_name == "rag.answer"]
    assert len(rag_steps) >= 2, (
        f"simple_rag (max_retries=0) 的空答案也应强制重试一次，"
        f"got {len(rag_steps)} rag.answer calls"
    )


def test_empty_answer_retries_at_most_once() -> None:
    """空答案强制重试只发生一次：两次都空则交付空答案，不无限重试（T3）。"""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="")
    request = AgentQueryRequest(project_slug="demo", query="general question")
    response = executor.execute(request)

    rag_steps = [s for s in response.steps if s.tool_name == "rag.answer"]
    assert len(rag_steps) == 2, (
        f"空答案重试只应发生一次（原始 + 1 次重试），got {len(rag_steps)} calls"
    )


def test_non_empty_answer_does_not_extra_retry_on_simple_rag() -> None:
    """非空答案不因 T3 增加重试：simple_rag 正常答案仍只调用一次 rag.answer。"""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="test answer")
    request = AgentQueryRequest(project_slug="demo", query="general question")
    response = executor.execute(request)

    rag_steps = [s for s in response.steps if s.tool_name == "rag.answer"]
    assert len(rag_steps) == 1, (
        f"非空答案不应触发额外重试，got {len(rag_steps)} rag.answer calls"
    )


def test_empty_answer_after_retry_marks_degraded_delivery() -> None:
    """空答案重试后仍空 → 显式降级标记（warning + finalize metadata，T2）。

    回归背景（2026-08-12）：R40 空答案重试一次后仍空，直接交付空字符串，
    消费方无法区分"无答案"与正常内容。降级标记让上游可拦截、人工可核对。
    """
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="")
    request = AgentQueryRequest(project_slug="demo", query="general question")
    response = executor.execute(request)

    assert response.status == "error"
    assert response.metadata['verification_state'] == 'skipped'
    assert any(
        "empty after retry" in w.lower() and "degraded" in w.lower()
        for w in response.warnings
    ), f"expected degraded-empty warning, got warnings={response.warnings}"

    finalize_steps = [s for s in response.steps if s.step_type == "finalize"]
    assert finalize_steps, "finalize step must exist"
    assert finalize_steps[-1].metadata.get("degraded_empty_answer") is True, (
        f"finalize metadata must flag degraded empty answer, "
        f"got {finalize_steps[-1].metadata}"
    )


def test_ok_answer_not_marked_degraded() -> None:
    """正常答案不得打降级标记。"""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="test answer")
    request = AgentQueryRequest(project_slug="demo", query="general question")
    response = executor.execute(request)

    finalize_steps = [s for s in response.steps if s.step_type == "finalize"]
    assert finalize_steps[-1].metadata.get("degraded_empty_answer") is False
    assert not any(
        "degraded" in w.lower() for w in response.warnings
    ), response.warnings


def test_max_tool_calls_limit_respected() -> None:
    """Agent never exceeds constraints.max_tool_calls."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="")
    # Tight constraint: only 1 tool call allowed
    request = AgentQueryRequest(
        project_slug="demo",
        query="cite this source",
        constraints={"max_tool_calls": 1, "max_steps": 10},
    )
    response = executor.execute(request)

    assert response.usage.tool_calls <= 1
    # Should still produce a valid response
    assert response.status == 'error'
    assert response.metadata['verification_executed'] is False
    assert response.final_answer is not None


def test_max_steps_limit_respected() -> None:
    """Agent never exceeds constraints.max_steps."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="")
    # Very tight step limit
    request = AgentQueryRequest(
        project_slug="demo",
        query="cite this source",
        constraints={"max_steps": 2, "max_tool_calls": 10},
    )
    response = executor.execute(request)

    # Strict enforcement: must not exceed max_steps
    assert len(response.steps) <= 2, (
        f"Expected at most 2 steps, got {len(response.steps)}"
    )
    # Should have status max_steps or a warning
    assert response.status in ("completed", "max_steps")
    if response.status == "max_steps" or any(
        "max_steps" in w.lower() or "step" in w.lower() for w in response.warnings
    ):
        pass  # acceptable handling
    else:
        # Also acceptable: completes with fewer steps than planned
        pass
    assert response.final_answer is not None


def test_max_steps_2_strict_limit() -> None:
    """With max_steps=2, exactly 2 steps are returned — no more."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="some answer")
    request = AgentQueryRequest(
        project_slug="demo",
        query="hello?",
        constraints={"max_steps": 2, "max_tool_calls": 10},
    )
    response = executor.execute(request)

    assert len(response.steps) <= 2, (
        f"max_steps=2 but got {len(response.steps)} steps"
    )


def test_max_steps_1_strict_limit() -> None:
    """With max_steps=1, at most 1 step is returned."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="some answer")
    request = AgentQueryRequest(
        project_slug="demo",
        query="hello?",
        constraints={"max_steps": 1, "max_tool_calls": 10},
    )
    response = executor.execute(request)

    assert len(response.steps) <= 1, (
        f"max_steps=1 but got {len(response.steps)} steps"
    )


def test_max_steps_exhausted_status() -> None:
    """When max_steps is exhausted before normal completion, status is max_steps."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="some answer")
    request = AgentQueryRequest(
        project_slug="demo",
        query="hello?",
        constraints={"max_steps": 1, "max_tool_calls": 10},
    )
    response = executor.execute(request)

    # With max_steps=1 (normal needs ~4), status should be max_steps
    # or there should be a clear max_steps warning
    has_max_steps_signal = (
        response.status == "max_steps"
        or any("max_steps" in w.lower() for w in response.warnings)
    )
    assert has_max_steps_signal, (
        f"Expected max_steps status or warning with max_steps=1, "
        f"got status={response.status}, warnings={response.warnings}"
    )


def test_usage_steps_equals_actual_steps() -> None:
    """usage.steps always matches len(response.steps)."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    assert response.usage.steps == len(response.steps), (
        f"usage.steps={response.usage.steps} != len(steps)={len(response.steps)}"
    )


def test_needs_clarification_respects_max_steps() -> None:
    """needs_clarification still works even with tight max_steps=1."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="should not be called")
    request = AgentQueryRequest(
        project_slug="demo",
        query="   ",
        constraints={"max_steps": 1, "max_tool_calls": 10},
    )
    response = executor.execute(request)

    assert response.route is not None
    assert response.route.route == "needs_clarification"
    assert len(response.steps) <= 1, (
        f"needs_clarification with max_steps=1 should return at most 1 step, "
        f"got {len(response.steps)}"
    )
    assert response.usage.tool_calls == 0


# ----------------------------------------------------------------
# v2a rework-002 — AgentConstraints validation
# ----------------------------------------------------------------


def test_constraints_rejects_max_steps_zero() -> None:
    """AgentConstraints(max_steps=0) raises ValidationError."""
    from pydantic import ValidationError
    from app.schemas.agent import AgentConstraints
    with pytest.raises(ValidationError):
        AgentConstraints(max_steps=0)


def test_constraints_rejects_max_steps_negative() -> None:
    """AgentConstraints(max_steps=-1) raises ValidationError."""
    from pydantic import ValidationError
    from app.schemas.agent import AgentConstraints
    with pytest.raises(ValidationError):
        AgentConstraints(max_steps=-1)


def test_constraints_rejects_max_tool_calls_zero() -> None:
    """AgentConstraints(max_tool_calls=0) raises ValidationError."""
    from pydantic import ValidationError
    from app.schemas.agent import AgentConstraints
    with pytest.raises(ValidationError):
        AgentConstraints(max_tool_calls=0)


def test_constraints_rejects_max_tool_calls_negative() -> None:
    """AgentConstraints(max_tool_calls=-1) raises ValidationError."""
    from pydantic import ValidationError
    from app.schemas.agent import AgentConstraints
    with pytest.raises(ValidationError):
        AgentConstraints(max_tool_calls=-1)


def test_constraints_rejects_timeout_seconds_zero() -> None:
    """AgentConstraints(timeout_seconds=0) raises ValidationError."""
    from pydantic import ValidationError
    from app.schemas.agent import AgentConstraints
    with pytest.raises(ValidationError):
        AgentConstraints(timeout_seconds=0)


def test_constraints_rejects_budget_tokens_zero() -> None:
    """AgentConstraints(budget_tokens=0) raises ValidationError."""
    from pydantic import ValidationError
    from app.schemas.agent import AgentConstraints
    with pytest.raises(ValidationError):
        AgentConstraints(budget_tokens=0)


def test_constraints_defaults_unchanged() -> None:
    """Default values remain: max_steps=8, max_tool_calls=5, budget_tokens=20000, timeout_seconds=45."""
    from app.schemas.agent import AgentConstraints
    c = AgentConstraints()
    assert c.max_steps == 8
    assert c.max_tool_calls == 5
    assert c.budget_tokens == 20000
    assert c.timeout_seconds == 45


def test_constraints_accepts_valid_minimal_values() -> None:
    """Positive values (1) are accepted."""
    from app.schemas.agent import AgentConstraints
    c = AgentConstraints(max_steps=1, max_tool_calls=1, budget_tokens=1, timeout_seconds=1)
    assert c.max_steps == 1
    assert c.max_tool_calls == 1
    assert c.budget_tokens == 1
    assert c.timeout_seconds == 1


# ------------------------------------------------------------------
# Phase 4: Controlled Complex Agent — plan step tests
# ------------------------------------------------------------------


def make_complex_query_executor(
    db: Session,
    *,
    rag_answer_text: str = "test answer for complex query",
) -> AgentExecutor:
    """Build an executor where RAGAdapter answers complex queries."""
    class StubRAGComplex:
        def answer(self, db, project_slug, question, document_id=None):
            from app.schemas.common import QueryResponse, Citation
            citations = []
            if rag_answer_text:
                citations = [
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.92,
                        excerpt="complex excerpt",
                    )
                ]
            return QueryResponse(
                answer_markdown=rag_answer_text,
                citations=citations,
                verification_status="local-only",
            )

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            from app.schemas.agent import EvidenceItem, EvidencePack
            return EvidencePack(
                status="ok",
                items=[
                    EvidenceItem(
                        index=0, document_id="d1", score=0.95,
                        excerpt="complex evidence",
                        evidence_kind="source_chunk",
                        source_stage="source_chunk",
                        support_hint="direct",
                    )
                ],
            )

    rag = StubRAGComplex()
    tools = ToolRegistry()
    tools._register_builtins(rag)
    tools.register(
        ToolSpec(
            name="answer.synthesize",
            description="deterministic test synthesizer",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "rag_answer": {"type": "string"},
                    "citations": {"type": "array"},
                },
                "required": ["query", "rag_answer"],
            },
            side_effect_level="none",
        ),
        lambda args, ctx: {
            "answer_markdown": args.get("rag_answer", ""),
            "cited_indexes": list(range(len(args.get("citations") or []))),
            "warnings": [],
            "confidence": 1.0,
            "provider": "local",
            "model": "local-fallback",
        },
    )
    memory = ConversationMemory(db)
    trace_store = AgentTraceStore(db)
    return AgentExecutor(
        rag=rag, tools=tools, memory=memory, db=db, trace_store=trace_store
    )


def test_complex_query_includes_plan_step() -> None:
    """Complex multi-hop queries get a plan step after route and before retrieve."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_complex_query_executor(db)
    request = AgentQueryRequest(
        project_slug="demo",
        query="first find the structure, then determine the binding affinity",
    )
    response = executor.execute(request)

    assert response.status == "completed"
    # Find the plan step
    plan_steps = [s for s in response.steps if s.step_type == "plan"]
    assert len(plan_steps) == 1, (
        f"Expected exactly 1 plan step, got {len(plan_steps)}: "
        f"{[s.step_type for s in response.steps]}"
    )
    # Plan step should come after route (step 0) and before retrieve/rag.answer
    plan = plan_steps[0]
    assert plan.step_id >= 1
    # Plan step has structured metadata
    assert plan.metadata.get("plan_type") == "controlled_complex"
    assert "rag.retrieve_evidence" in plan.metadata.get("allowed_tools", [])
    assert "rag.answer" in plan.metadata.get("allowed_tools", [])
    assert "answer.synthesize" in plan.metadata.get("allowed_tools", [])
    assert "answer.verify" in plan.metadata.get("allowed_tools", [])
    assert "shell" in plan.metadata.get("forbidden_tools", [])
    assert "sql" in plan.metadata.get("forbidden_tools", [])
    assert "network" in plan.metadata.get("forbidden_tools", [])
    assert "subtasks" in plan.metadata

    # Plan step appears before retrieve and rag.answer in step sequence
    plan_idx = next(i for i, s in enumerate(response.steps) if s.step_type == "plan")
    retrieve_idx = next(
        (i for i, s in enumerate(response.steps) if s.step_type == "retrieve"),
        None,
    )
    rag_idx = next(
        (i for i, s in enumerate(response.steps) if s.tool_name == "rag.answer"),
        None,
    )
    assert plan_idx < retrieve_idx, (
        f"Plan step ({plan_idx}) must be before retrieve ({retrieve_idx})"
    )
    assert plan_idx < rag_idx, (
        f"Plan step ({plan_idx}) must be before rag.answer ({rag_idx})"
    )


def test_simple_query_does_not_include_plan_step() -> None:
    """Simple queries route to simple_rag and do NOT get a plan step."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_complex_query_executor(db)
    request = AgentQueryRequest(
        project_slug="demo",
        query="what is the binding affinity of protein X",
    )
    response = executor.execute(request)

    assert response.status == "completed"
    plan_steps = [s for s in response.steps if s.step_type == "plan"]
    assert len(plan_steps) == 0, (
        f"Simple query should NOT get a plan step, "
        f"but found: {[(s.step_type, s.summary[:50]) for s in plan_steps]}"
    )
    # Verify it still completes normally through the RAG pipeline
    assert response.final_answer is not None
    assert response.route is not None
    assert response.route.route == "simple_rag"


def test_plan_step_metadata_has_whitelist_and_forbidden_tools() -> None:
    """Plan step exposes only safe read-only whitelist tools and explicitly
    excludes shell/SQL/write/network."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_complex_query_executor(db)
    request = AgentQueryRequest(
        project_slug="demo",
        query="step by step analyze the molecular dynamics",
    )
    response = executor.execute(request)

    plan_steps = [s for s in response.steps if s.step_type == "plan"]
    assert len(plan_steps) == 1

    meta = plan_steps[0].metadata

    # Whitelist: only safe read-only tools
    allowed = meta.get("allowed_tools", [])
    for tool in allowed:
        assert tool in {
            "rag.retrieve_evidence",
            "rag.answer",
            "answer.synthesize",
            "answer.verify",
        }, f"allowed_tools contains unexpected tool: {tool}"

    # No dangerous tools in allowed list
    for dangerous in ("shell", "sql", "write", "network"):
        assert dangerous not in allowed, (
            f"Dangerous tool '{dangerous}' must not be in allowed_tools: {allowed}"
        )

    # Forbidden: explicitly excludes dangerous tool categories
    forbidden = meta.get("forbidden_tools", [])
    for dangerous in ("shell", "sql", "write", "network"):
        assert dangerous in forbidden, (
            f"Dangerous tool '{dangerous}' must be in forbidden_tools: {forbidden}"
        )

    # Max steps and tool calls are present
    assert meta.get("max_steps", 0) > 0
    assert meta.get("max_tool_calls", 0) > 0

    # Subtasks are present
    assert len(meta.get("subtasks", [])) > 0


def test_complex_plan_step_with_tight_max_steps() -> None:
    """With very tight max_steps=2, plan step may be included but
    execution is truncated before all normal steps complete."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_complex_query_executor(db)
    request = AgentQueryRequest(
        project_slug="demo",
        query="step by step analyze the data",
        constraints={"max_steps": 2, "max_tool_calls": 10},
    )
    response = executor.execute(request)

    # Must not exceed max_steps
    assert len(response.steps) <= 2, (
        f"max_steps=2 but got {len(response.steps)} steps"
    )
    # Plan step should be present (it's step 1 after route)
    plan_steps = [s for s in response.steps if s.step_type == "plan"]
    assert len(plan_steps) == 1
    # Should signal max_steps in status or warnings
    has_signal = (
        response.status == "max_steps"
        or any("max_steps" in w.lower() for w in response.warnings)
        or any("truncat" in w.lower() for w in response.warnings)
    )
    assert has_signal, (
        f"Expected max_steps signal, got status={response.status}, "
        f"warnings={response.warnings}"
    )


def test_complex_plan_step_with_max_steps_3() -> None:
    """With max_steps=3, route+plan+1 more step is all we get."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_complex_query_executor(db)
    request = AgentQueryRequest(
        project_slug="demo",
        query="step by step analyze the data",
        constraints={"max_steps": 3, "max_tool_calls": 10},
    )
    response = executor.execute(request)

    assert len(response.steps) <= 3, (
        f"max_steps=3 but got {len(response.steps)} steps"
    )
    # Should have route and plan at minimum
    step_types = [s.step_type for s in response.steps]
    assert "route" in step_types
    assert "plan" in step_types

    assert response.status in ("completed", "max_steps")
    assert response.final_answer is not None


def test_complex_plan_with_tight_max_tool_calls() -> None:
    """With max_tool_calls=1, only the retrieve_evidence tool call fits."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_complex_query_executor(db)
    request = AgentQueryRequest(
        project_slug="demo",
        query="step by step analyze the data",
        constraints={"max_tool_calls": 1, "max_steps": 10},
    )
    response = executor.execute(request)

    assert response.usage.tool_calls <= 1, (
        f"max_tool_calls=1 but got {response.usage.tool_calls}"
    )
    # Plan step should still be present (it's not a tool call)
    plan_steps = [s for s in response.steps if s.step_type == "plan"]
    assert len(plan_steps) == 1
    assert response.status == 'error'
    assert response.metadata['verification_state'] == 'skipped'


# ------------------------------------------------------------------
# evidence pack retrieve step tests
# ------------------------------------------------------------------


def make_executor_with_retrieve(
    db: Session,
    *,
    rag_answer_text: str = "test answer",
    evidence_items: list | None = None,
) -> AgentExecutor:
    """Build an AgentExecutor whose RAGAdapter also supports retrieve_evidence."""

    if evidence_items is None:
        evidence_items = [
            {
                "index": 0,
                "document_id": "d1",
                "chunk_id": "c1",
                "score": 0.92,
                "excerpt": "sample excerpt",
                "evidence_kind": "source_chunk",
                "source_stage": "source_chunk",
                "support_hint": "direct",
            }
        ]

    class StubRAGWithRetrieve:
        def answer(self, db, project_slug, question, document_id=None):
            from app.schemas.common import QueryResponse, Citation

            citations = []
            if rag_answer_text:
                citations = [
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.92,
                        excerpt="sample excerpt",
                    )
                ]
            return QueryResponse(
                answer_markdown=rag_answer_text,
                citations=citations,
                verification_status="local-only",
            )

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            from app.schemas.agent import EvidenceItem, EvidencePack

            items = [
                EvidenceItem(**item) if isinstance(item, dict) else item
                for item in evidence_items
            ]
            return EvidencePack(status="ok" if items else "empty", items=items)

    rag = StubRAGWithRetrieve()
    tools = ToolRegistry()
    tools._register_builtins(rag)
    memory = ConversationMemory(db)
    trace_store = AgentTraceStore(db)
    return AgentExecutor(
        rag=rag, tools=tools, memory=memory, db=db, trace_store=trace_store
    )


def make_executor_with_scoped_retrieve(
    db: Session, *, rag_answer_text: str = "test answer"
) -> AgentExecutor:
    """retrieve_evidence 按 document_id 区分结果（T1 回退场景）。

    项目级（document_id=None）→ 空（模拟弱指代查询全项目路由 miss）；
    带 document_id → 该文档命中证据。
    """

    class ScopedStubRAG:
        def answer(self, db, project_slug, question, document_id=None):
            from app.schemas.common import QueryResponse, Citation

            citations = []
            if rag_answer_text:
                citations = [
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.92,
                        excerpt="sample excerpt",
                    )
                ]
            return QueryResponse(
                answer_markdown=rag_answer_text,
                citations=citations,
                verification_status="local-only",
            )

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            from app.schemas.agent import EvidenceItem, EvidencePack

            if document_id is None:
                return EvidencePack(status="empty", items=[])
            return EvidencePack(
                status="ok",
                items=[
                    EvidenceItem(
                        index=0,
                        document_id=document_id,
                        chunk_id="c1",
                        score=0.92,
                        excerpt="evidence from history document",
                        evidence_kind="source_chunk",
                        source_stage="source_chunk",
                        support_hint="direct",
                    )
                ],
            )

    rag = ScopedStubRAG()
    tools = ToolRegistry()
    tools._register_builtins(rag)
    memory = ConversationMemory(db)
    trace_store = AgentTraceStore(db)
    return AgentExecutor(
        rag=rag, tools=tools, memory=memory, db=db, trace_store=trace_store
    )


def _seed_retrieve_turn(
    db: Session, session_id: str, document_ids: str | list[str]
) -> None:
    """预置一轮成功检索的历史（retrieve turn 带命中文档 citations）。"""
    ids = [document_ids] if isinstance(document_ids, str) else document_ids
    memory = ConversationMemory(db)
    memory.add_turn(
        session_id,
        role="tool",
        content="Retrieved 3 evidence items",
        tool_name="rag.retrieve_evidence",
        tool_args={"project_slug": "demo", "question": "previous question"},
        tool_result="3",
        step_type="retrieve",
        citations=[{"document_id": doc_id} for doc_id in ids],
    )
    db.commit()


def test_retrieve_empty_falls_back_to_history_document() -> None:
    """T1：项目级会话检索 0 items 时回退到历史最近成功命中文档重检索。"""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor_with_scoped_retrieve(db)
    _seed_retrieve_turn(db, "sess_t1", "d1")

    request = AgentQueryRequest(
        project_slug="demo", query="general question", session_id="sess_t1"
    )
    response = executor.execute(request)

    retrieve_steps = [s for s in response.steps if s.step_type == "retrieve"]
    assert len(retrieve_steps) >= 2, (
        f"检索空结果应回退历史文档重检索，got {len(retrieve_steps)} retrieve steps"
    )
    # 第二次（回退）检索携带历史命中文档 d1
    assert retrieve_steps[-1].metadata.get("document_id") == "d1"


def test_retrieve_empty_without_history_no_fallback() -> None:
    """T1：无历史成功检索时不回退（只检索一次）。"""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor_with_scoped_retrieve(db)
    request = AgentQueryRequest(
        project_slug="demo", query="general question", session_id="sess_t2"
    )
    response = executor.execute(request)

    retrieve_steps = [s for s in response.steps if s.step_type == "retrieve"]
    assert len(retrieve_steps) == 1, (
        f"无历史命中文档不应回退，got {len(retrieve_steps)} retrieve steps"
    )


def test_locked_session_does_not_fallback() -> None:
    """T1：锁定文档会话（document_id 非空）不触发历史回退。"""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor_with_scoped_retrieve(db)
    _seed_retrieve_turn(db, "sess_t3", "d1")

    request = AgentQueryRequest(
        project_slug="demo",
        query="general question",
        session_id="sess_t3",
        document_id="d9",
    )
    response = executor.execute(request)

    retrieve_steps = [s for s in response.steps if s.step_type == "retrieve"]
    assert len(retrieve_steps) == 1, (
        f"锁定会话应直接检索不回退，got {len(retrieve_steps)} retrieve steps"
    )
    assert retrieve_steps[0].metadata.get("document_id") == "d9"


def test_retrieve_persists_hit_document_ids() -> None:
    """T1：检索命中时 retrieve turn 的 citations 持久化命中文档 id。"""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor_with_retrieve(db)  # 默认 retrieve 命中 d1
    request = AgentQueryRequest(
        project_slug="demo", query="hello?", session_id="sess_t4"
    )
    response = executor.execute(request)

    assert response.status == "completed"
    history = ConversationMemory(db).get_history("sess_t4")
    retrieve_turns = [t for t in history if t.step_type == "retrieve"]
    assert retrieve_turns, "会话历史应包含 retrieve turn"
    doc_ids = {
        str(c.get("document_id") or "")
        for t in retrieve_turns
        for c in t.citations
    }
    assert "d1" in doc_ids, f"retrieve turn 应持久化命中文档 d1, got {doc_ids}"


def test_majority_hit_document_id_majority_vote() -> None:
    """T2：回退锚定取历史多数（按轮次投票，防单轮大量 citations 碾压）。"""
    db = make_db()
    executor = make_executor_with_scoped_retrieve(db)
    _seed_retrieve_turn(db, "s_maj", "d1")  # 主题轮 ×2
    _seed_retrieve_turn(db, "s_maj", "d1")
    _seed_retrieve_turn(db, "s_maj", ["d2"] * 4)  # 跨主题轮：4 citations 只算 1 票
    assert executor._majority_hit_document_id("s_maj") == "d1"


def test_majority_hit_document_id_recent_loses_to_majority() -> None:
    """T2：最近命中被多数否决——跨主题轮（如 R33/R34 命中 OPLS5）不覆盖主主题。"""
    db = make_db()
    executor = make_executor_with_scoped_retrieve(db)
    _seed_retrieve_turn(db, "s_rec", "d1")
    _seed_retrieve_turn(db, "s_rec", "d1")
    _seed_retrieve_turn(db, "s_rec", "d1")
    _seed_retrieve_turn(db, "s_rec", "d2")  # 最近一轮命中 d2
    assert executor._majority_hit_document_id("s_rec") == "d1"


def test_majority_hit_document_id_tie_breaks_by_recency() -> None:
    """T2：平票时取最近命中文档（保持指代就近性）。"""
    db = make_db()
    executor = make_executor_with_scoped_retrieve(db)
    _seed_retrieve_turn(db, "s_tie", "d1")
    _seed_retrieve_turn(db, "s_tie", "d2")
    _seed_retrieve_turn(db, "s_tie", "d1")
    _seed_retrieve_turn(db, "s_tie", "d2")  # d2 最近且平票
    assert executor._majority_hit_document_id("s_tie") == "d2"


def test_majority_hit_document_id_empty_history_returns_none() -> None:
    """T2：无历史检索命中时返回 None（不触发回退）。"""
    db = make_db()
    executor = make_executor_with_scoped_retrieve(db)
    assert executor._majority_hit_document_id("s_none") is None


def test_fallback_prefers_majority_document_over_recent_hit() -> None:
    """T2（集成）：项目级 0 items 回退锚定多数主题文档，而非最近命中。"""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor_with_scoped_retrieve(db)
    _seed_retrieve_turn(db, "s_t5", "d1")  # 主题轮 ×3
    _seed_retrieve_turn(db, "s_t5", "d1")
    _seed_retrieve_turn(db, "s_t5", "d1")
    _seed_retrieve_turn(db, "s_t5", "d2")  # 跨主题轮（最近命中 d2）

    request = AgentQueryRequest(
        project_slug="demo", query="general question", session_id="s_t5"
    )
    response = executor.execute(request)

    retrieve_steps = [s for s in response.steps if s.step_type == "retrieve"]
    assert len(retrieve_steps) >= 2, (
        f"检索空结果应回退历史文档重检索，got {len(retrieve_steps)} retrieve steps"
    )
    assert retrieve_steps[-1].metadata.get("document_id") == "d1"


def test_execute_includes_retrieve_step() -> None:
    """Normal execution includes a retrieve step between route and rag.answer."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor_with_retrieve(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    assert response.status == "completed"
    # Find the retrieve step
    retrieve_steps = [
        s for s in response.steps if s.step_type == "retrieve"
    ]
    assert len(retrieve_steps) >= 1, (
        f"Expected at least 1 retrieve step, got steps={[s.step_type for s in response.steps]}"
    )
    retrieve = retrieve_steps[0]
    assert retrieve.tool_name == "rag.retrieve_evidence"
    assert retrieve.tool_ok is True


def test_retrieve_step_has_evidence_metadata() -> None:
    """The retrieve step metadata includes evidence counts and kinds."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    evidence_items = [
        {
            "index": 0,
            "document_id": "d1",
            "chunk_id": "c1",
            "score": 0.95,
            "excerpt": "table data",
            "evidence_kind": "table",
            "source_stage": "document_table",
            "support_hint": "direct",
        },
        {
            "index": 1,
            "document_id": "d2",
            "score": 0.80,
            "excerpt": "text evidence",
            "evidence_kind": "source_chunk",
            "source_stage": "source_chunk",
            "support_hint": "contextual",
        },
    ]
    executor = make_executor_with_retrieve(db, evidence_items=evidence_items)
    request = _cross_turn_request(db)
    response = executor.execute(request)

    retrieve_steps = [
        s for s in response.steps if s.step_type == "retrieve"
    ]
    assert len(retrieve_steps) >= 1
    meta = retrieve_steps[0].metadata
    assert meta.get("evidence_count") == 2
    assert meta.get("table_evidence_count") == 1
    assert "evidence_kinds" in meta
    assert "table" in meta["evidence_kinds"]
    assert "source_chunk" in meta["evidence_kinds"]
    assert "source_stages" in meta
    assert "document_table" in meta["source_stages"]
    assert "support_hints" in meta
    assert "direct" in meta["support_hints"]


def test_needs_clarification_skips_retrieve() -> None:
    """needs_clarification route still has no retrieve or RAG tool call."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor_with_retrieve(db, rag_answer_text="should not be called")
    request = AgentQueryRequest(project_slug="demo", query="   ")
    response = executor.execute(request)

    assert response.route.route == "needs_clarification"
    retrieve_steps = [
        s for s in response.steps if s.step_type == "retrieve"
    ]
    assert len(retrieve_steps) == 0
    assert response.usage.tool_calls == 0


def test_retrieve_step_respects_max_tool_calls() -> None:
    """The retrieve step counts toward tool_calls limit."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor_with_retrieve(db, rag_answer_text="")
    request = AgentQueryRequest(
        project_slug="demo",
        query="hello?",
        constraints={"max_tool_calls": 1, "max_steps": 10},
    )
    response = executor.execute(request)

    # With max_tool_calls=1, only the retrieve step should execute
    assert response.usage.tool_calls <= 1
    assert response.status == 'error'
    assert response.metadata['verification_state'] == 'skipped'


def test_retrieve_step_before_rag_answer() -> None:
    """The retrieve step occurs before rag.answer in the step sequence."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor_with_retrieve(db)
    request = AgentQueryRequest(project_slug="demo", query="hello?")
    response = executor.execute(request)

    step_types = [s.step_type for s in response.steps]
    # Order: route -> retrieve -> tool_call (rag.answer) -> synthesis -> tool_call (verify) -> finalize
    retrieve_idx = step_types.index("retrieve") if "retrieve" in step_types else -1
    rag_idx = next(
        (i for i, s in enumerate(response.steps) if s.tool_name == "rag.answer"),
        -1,
    )
    assert retrieve_idx >= 0, f"No retrieve step found in {step_types}"
    assert rag_idx >= 0, f"No rag.answer step found"
    assert retrieve_idx < rag_idx, (
        f"Expected retrieve ({retrieve_idx}) before rag.answer ({rag_idx})"
    )


# ------------------------------------------------------------------
# evidence-aware synthesis tests (Phase 3)
# ------------------------------------------------------------------


def test_synthesize_step_has_evidence_pack_metadata() -> None:
    """When evidence pack is available, synthesis step metadata includes
    evidence_pack_items, table_evidence_count, and source_stages."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    evidence_items = [
        {
            "index": 0,
            "document_id": "d1",
            "chunk_id": "c1",
            "score": 0.95,
            "excerpt": "table data",
            "evidence_kind": "table",
            "source_stage": "document_table",
            "support_hint": "direct",
        },
        {
            "index": 1,
            "document_id": "d2",
            "score": 0.80,
            "excerpt": "text evidence",
            "evidence_kind": "source_chunk",
            "source_stage": "source_chunk",
            "support_hint": "contextual",
        },
    ]
    executor = make_executor_with_retrieve(db, evidence_items=evidence_items)
    request = _cross_turn_request(db)
    response = executor.execute(request)

    assert response.status == "completed"
    # Find the synthesis step
    synth_steps = [
        s for s in response.steps if s.step_type == "synthesis"
    ]
    assert len(synth_steps) >= 1, (
        f"Expected at least 1 synthesis step, got steps={[s.step_type for s in response.steps]}"
    )
    synth_meta = synth_steps[0].metadata
    assert synth_meta.get("evidence_pack_items") == 2, (
        f"Expected 2 evidence_pack_items, got {synth_meta}"
    )
    assert synth_meta.get("table_evidence_count") == 1, (
        f"Expected 1 table_evidence_count, got {synth_meta}"
    )
    assert "source_stages" in synth_meta
    assert "document_table" in synth_meta["source_stages"]
    assert "source_chunk" in synth_meta["source_stages"]


def test_synthesize_step_without_evidence_pack_no_metadata_leak() -> None:
    """When no retrieve step runs (no retrieve_evidence on RAG), synthesis step
    does not include evidence_pack metadata fields."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    # Use make_executor (without retrieve_evidence support)
    executor = make_executor(db)
    request = _cross_turn_request(db)
    response = executor.execute(request)

    assert response.status == "completed"
    synth_steps = [
        s for s in response.steps if s.step_type == "synthesis"
    ]
    assert len(synth_steps) >= 1
    synth_meta = synth_steps[0].metadata
    # Should still have provider/model/confidence but no evidence pack fields
    assert "provider" in synth_meta
    assert "evidence_pack_items" not in synth_meta, (
        "Should not have evidence_pack_items when no retrieve step ran"
    )


def test_executor_passes_evidence_pack_to_synthesize_tool(monkeypatch) -> None:
    """Verify that when evidence pack is available, the answer.synthesize tool
    call includes evidence_pack in its arguments."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    captured_tool_args = {}

    class StubRAGWithRetrieve:
        def answer(self, db, project_slug, question, document_id=None):
            from app.schemas.common import QueryResponse, Citation
            return QueryResponse(
                answer_markdown="test answer",
                citations=[Citation(document_id="d1", chunk_id="c1", score=0.92, excerpt="sample")],
                verification_status="local-only",
            )

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            from app.schemas.agent import EvidenceItem, EvidencePack
            return EvidencePack(
                status="ok",
                items=[
                    EvidenceItem(
                        index=0, document_id="d1", score=0.95, excerpt="test",
                        evidence_kind="table", source_stage="document_table",
                        support_hint="direct",
                    )
                ],
            )

    rag = StubRAGWithRetrieve()
    tools = ToolRegistry()
    tools._register_builtins(rag)

    # Override call_tool to capture synthesize args
    original_call_tool = tools.call_tool

    def capturing_call_tool(name, args=None, *, ctx=None):
        if name == "answer.synthesize" and args:
            captured_tool_args.update(args)
        return original_call_tool(name, args=args, ctx=ctx)

    monkeypatch.setattr(tools, "call_tool", capturing_call_tool)

    memory = ConversationMemory(db)
    trace_store = AgentTraceStore(db)
    executor = AgentExecutor(
        rag=rag, tools=tools, memory=memory, db=db, trace_store=trace_store
    )
    request = _cross_turn_request(db)
    response = executor.execute(request)

    assert response.status == "completed"
    assert "evidence_pack" in captured_tool_args, (
        f"Expected evidence_pack in synthesize tool args, got keys: {list(captured_tool_args.keys())}"
    )
    ep = captured_tool_args["evidence_pack"]
    assert ep["status"] == "ok"
    assert len(ep["items"]) == 1
    assert ep["items"][0]["evidence_kind"] == "table"


# ------------------------------------------------------------------
# insufficient-evidence surface tests
# ------------------------------------------------------------------


def test_executor_surfaces_insufficient_evidence_in_warnings() -> None:
    """When RAG returns an insufficient-evidence answer, the executor must
    include a warning so the caller knows the answer is not grounded."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(
        db, rag_answer_text="## Insufficient Evidence\n\nThe retrieved source documents do not contain information about the specific scientific terms in your question (charmm36m)."
    )
    request = AgentQueryRequest(project_slug="demo", query="What is the difference between CHARMM36m and CHARMM36?")
    response = executor.execute(request)

    assert response.status == "completed"
    # Must include a warning about insufficient evidence
    has_insufficient_warning = any(
        "insufficient" in w.lower() or "insufficient evidence" in w.lower()
        for w in response.warnings
    )
    assert has_insufficient_warning, (
        f"Expected insufficient-evidence warning, got warnings={response.warnings}"
    )


def test_executor_normal_answer_no_false_insufficient_warning() -> None:
    """Normal grounded answers must NOT produce a false insufficient-evidence warning."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="CHARMM36m improves backbone CMAP over CHARMM36.")
    request = AgentQueryRequest(project_slug="demo", query="What is CHARMM36m?")
    response = executor.execute(request)

    assert response.status == "completed"
    # Must NOT include an insufficient-evidence warning
    has_insufficient_warning = any(
        "insufficient evidence" in w.lower()
        for w in response.warnings
    )
    assert not has_insufficient_warning, (
        f"Regular answer should NOT trigger insufficient-evidence warning, "
        f"got warnings={response.warnings}"
    )


def test_retarget_citation_markers_after_synthesis_filtering() -> None:
    """Inline markers must follow the compacted final citation list."""
    answer = "Table 2 supports this value [2], while Table 4 supports that value [0]."

    retargeted = AgentExecutor._retarget_citation_markers(answer, [0, 2])

    assert "[1]" in retargeted
    assert "[0]" in retargeted
    assert "[2]" not in retargeted


def test_table_reference_query_injects_anchors_even_when_retrieval_empty() -> None:
    """检索 0 items 时，表格指代查询仍必须注入会话锚点（T1b code review 修正）。

    回归场景：存量文档尚无 DI tables（迁移前）→ 放宽过滤仍拿不到任何
    候选表格；此时会话锚点是唯一可用的表格证据。旧实现被
    ``evidence_pack.get("items")`` 非空条件挡住，miss 时锚点全部丢失。
    """
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    # 会话历史：第 1 轮提问 + 第 1 轮 finalize 引用过表格。
    # 第 1 轮 user turn 使本轮成为第 2 轮 → synthesize 路径（锚点注入
    # 的效果在 synthesize step 的 evidence metadata 中可见）。
    memory = ConversationMemory(db)
    memory.add_turn(
        "s_anchor",
        role="user",
        content="第一张表验证了什么？",
        step_type="user_query",
    )
    memory.add_turn(
        "s_anchor",
        role="agent",
        content="第一张表的数值是 C22/CMAP 1.80 0.74 3.72 2.09。",
        step_type="finalize",
        citations=[
            {
                "block_type": "table",
                "excerpt": "| C22/CMAP | 1.80 | 0.74 | 3.72 | 2.09 |",
                "document_id": "d1",
                "page_label": "31",
            }
        ],
    )
    db.commit()

    captured_tool_args: dict = {}

    class StubRAGWithRetrieve:
        def answer(self, db, project_slug, question, document_id=None):
            from app.schemas.common import QueryResponse, Citation

            return QueryResponse(
                answer_markdown="answer",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.92,
                        excerpt="sample excerpt",
                    )
                ],
                verification_status="local-only",
            )

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            from app.schemas.agent import EvidencePack

            return EvidencePack(status="empty", items=[])

    rag = StubRAGWithRetrieve()
    tools = ToolRegistry()
    tools._register_builtins(rag)

    original_call_tool = tools.call_tool

    def capturing_call_tool(name, args=None, *, ctx=None):
        if name == "answer.synthesize" and args:
            captured_tool_args.update(args)
        return original_call_tool(name, args=args, ctx=ctx)

    tools.call_tool = capturing_call_tool  # type: ignore[method-assign]

    trace_store = AgentTraceStore(db)
    executor = AgentExecutor(
        rag=rag, tools=tools, memory=ConversationMemory(db), db=db, trace_store=trace_store
    )
    request = AgentQueryRequest(
        project_slug="demo",
        query="现在展开第二张，但不要重复第一张的内容。",
        session_id="s_anchor",
    )
    response = executor.execute(request)

    assert response.status == "completed"
    assert "evidence_pack" in captured_tool_args, (
        f"synthesize tool args keys: {list(captured_tool_args.keys())}"
    )
    ep = captured_tool_args["evidence_pack"]
    items = ep.get("items") or []
    assert any(
        item.get("source_stage") == "session_table_anchor"
        and "C22/CMAP" in item.get("excerpt", "")
        for item in items
    ), f"锚点必须注入空检索结果，got items={items}"


def test_merge_table_anchor_evidence_injects_anchors_and_dedupes() -> None:
    """会话锚点注入为补充证据；与已有证据重复的摘录不重复注入。"""
    evidence_pack = {
        "items": [
            {
                "index": 0,
                "document_id": "doc-1",
                "evidence_kind": "table",
                "excerpt": "| C36 | 1.12 |",
                "score": 0.95,
            }
        ]
    }
    anchors = [
        {
            "excerpt": "| C22/CMAP | 1.80 | 0.74 | 3.72 | 2.09 |",
            "document_id": "doc-1",
            "page_label": "31",
            "table_id": "table-3",
            "turn_index": 1,
        },
        # 与既有证据相同 → 跳过
        {"excerpt": "| C36 | 1.12 |", "document_id": "doc-1", "turn_index": 0},
        # 空摘录 → 跳过
        {"excerpt": "", "document_id": "doc-1", "turn_index": 0},
    ]

    augmented = AgentExecutor._merge_table_anchor_evidence(evidence_pack, anchors)

    items = augmented["items"]
    assert len(items) == 2
    injected = items[1]
    assert injected["index"] == 1
    assert injected["evidence_kind"] == "table"
    assert injected["source_stage"] == "session_table_anchor"
    assert injected["support_hint"] == "contextual"
    assert injected["score"] == 0.9
    assert injected["excerpt"] == "| C22/CMAP | 1.80 | 0.74 | 3.72 | 2.09 |"
    assert injected["document_id"] == "doc-1"
    assert injected["page_label"] == "31"
    assert injected["table_id"] == "table-3"


def test_merge_table_anchor_evidence_no_anchors_keeps_pack_unchanged() -> None:
    """无锚点时空跑：原证据包原样返回（字段共享但 items 一致）。"""
    evidence_pack = {
        "items": [
            {
                "index": 0,
                "document_id": "doc-1",
                "evidence_kind": "paragraph",
                "excerpt": "prose",
                "score": 0.8,
            }
        ]
    }
    augmented = AgentExecutor._merge_table_anchor_evidence(evidence_pack, [])
    assert augmented["items"] == evidence_pack["items"]


def test_table_evidence_items_are_added_to_synthesis_citations() -> None:
    """Agent synthesis must see every retrieved table Child, not only RAG's top five."""
    base = [
        Citation(
            document_id="doc-1",
            chunk_id="child-1",
            score=0.9,
            excerpt="Table 2. HFE | OPLS4 | OPLS5 | 0.76 | 0.46",
        )
    ]
    evidence_pack = {
        "items": [
            {
                "document_id": "doc-1",
                "chunk_id": "child-1",
                "evidence_kind": "table",
                "table_id": "table-2",
                "excerpt": "Table 2. HFE | OPLS4 | OPLS5 | 0.76 | 0.46",
                "score": 0.9,
                "page_label": "10",
                "parse_version": "canonical-v4",
            },
            {
                "document_id": "doc-1",
                "chunk_id": "child-5",
                "evidence_kind": "table",
                "table_id": "table-5",
                "excerpt": "Table 5. GLU | OPLS4 | OPLS5 | 0.70 | 0.61",
                "score": 0.8,
                "page_label": "12",
                "parse_version": "canonical-v4",
            },
        ]
    }

    augmented = AgentExecutor._merge_table_evidence_citations(base, evidence_pack)

    assert [citation.chunk_id for citation in augmented] == ["child-1", "child-5"]
    assert augmented[1].table_id == "table-5"
    assert augmented[1].parse_version == "canonical-v4"


def test_table_fact_source_citations_are_added_even_when_child_is_not_top_context() -> None:
    """A fact's original Child must be available for exact citation binding."""
    extra = [
        Citation(
            document_id="doc-1",
            chunk_id="child-total",
            score=0.0,
            excerpt="Table 7 | TotalWeightedAverage | OPLS4 Edgewise 1.18 | OPLS5 Edgewise 1.12",
            parse_version="canonical-v4",
            block_type="table",
            table_id="table-7",
        )
    ]

    augmented = AgentExecutor._merge_table_evidence_citations(
        [],
        {"items": [], "table_facts": []},
        extra_citations=extra,
    )

    assert [citation.chunk_id for citation in augmented] == ["child-total"]
    assert "1.18" in augmented[0].excerpt


def test_is_cross_turn_query_reverse_default_from_second_turn() -> None:
    """9.2 跨轮判定（2026-08-11 反向默认）：第 2 轮起一律跨轮。

    不做指代词词表匹配：短到"那 C36 呢"、长到"审计整个对话"、无
    指代词的完整问题，只要历史中有上一轮，都判跨轮；第 1 轮保留
    draft 直通（False）。
    """
    db = make_db()
    executor = make_executor(db)
    session_id = "sess-cross-turn"

    def second_turn_call(query: str) -> bool:
        """模拟 executor 时序：当前 query 先写入历史（agent_executor:158），
        再判跨轮；[-2] 因此是真正的上一轮。"""
        executor._memory.add_turn(
            session_id, role="user", content=query, step_type="user_query"
        )
        return executor._is_cross_turn_query(query, session_id)

    # 无 session_id：一律 False
    assert executor._is_cross_turn_query("那 C36 呢", None) is False

    # 第 1 轮（历史中只有当前 turn，无上一轮可解析）：False
    executor._memory.add_turn(
        session_id, role="user", content="charmm36m 的 accuracy 是多少", step_type="user_query"
    )
    assert executor._is_cross_turn_query("那 C36 呢", session_id) is False

    # 第 2 轮起：无论 query 内容如何，一律 True
    assert second_turn_call("那 C36 呢") is True
    assert second_turn_call("对比之前那篇") is True
    # 无指代词的完整问题也跨轮（反向默认）
    assert second_turn_call("charmm36m 的 accuracy 是多少") is True
    assert second_turn_call("对比 OPLS4 和 OPLS5") is True
    # 长查询（>30 字符）同样跨轮：长度门槛只作用于检索上下文化
    assert second_turn_call("刚才说的证据分别来自哪些体系？请按体系分组。") is True

    # 重复提问（verbatim 重试同一问题）：仍跨轮——"一律跨轮"无例外
    # （旧实现经 _resolvable_previous_turn 的防重复守卫会漏判为单轮）
    executor._memory.add_turn(
        session_id, role="user", content="对比 OPLS4 和 OPLS5", step_type="user_query"
    )
    assert executor._is_cross_turn_query("对比 OPLS4 和 OPLS5", session_id) is True

    # 长间隙：中间夹 >12 条 agent turn 后，上一轮 user turn 掉出
    # last_n=12 窗口，仍判跨轮（跨轮判定统计全部 user turn，不设窗口）
    for i in range(15):
        executor._memory.add_turn(
            session_id, role="agent", content=f"intermediate {i}", step_type="finalize"
        )
    assert executor._is_cross_turn_query("新的问题", session_id) is True


def test_contextualize_retrieval_query_length_gate() -> None:
    """9.2 上下文化（2026-08-11 去词表）：≤30 字符一律包装，>30 字符不包装。

    词表漏检（"第二点/展开"等不在旧 9 词内）导致短引用查询原始检索
    miss；新逻辑短查询一律包装为"上一轮问题+当前追问"。长查询视为
    自带上下文，不做包装避免污染独立新问题。
    """
    db = make_db()
    executor = make_executor(db)
    session_id = "sess-ctx"

    executor._memory.add_turn(
        session_id,
        role="user",
        content="CHARMM36 主链修改的验证方法是什么",
        step_type="user_query",
    )
    executor._memory.add_turn(
        session_id, role="user", content="那第二点呢", step_type="user_query"
    )

    # ≤30 字符：包装（含指代词）
    ctx = executor._contextualize_retrieval_query(session_id, "那第二点呢")
    assert ctx.startswith("上一轮问题：")
    assert "CHARMM36 主链修改的验证方法是什么" in ctx
    assert "当前追问：那第二点呢" in ctx

    # ≤30 字符且无指代词：同样包装（去词表，词表漏检案例）
    ctx2 = executor._contextualize_retrieval_query(session_id, "展开第一张表")
    assert ctx2.startswith("上一轮问题：")
    assert "当前追问：展开第一张表" in ctx2

    # >30 字符：视为自带上下文，不包装
    long_q = "请把你刚才列出的所有主链修改精确术语完整列出并给出每个的验证方法"
    ctx3 = executor._contextualize_retrieval_query(session_id, long_q)
    assert ctx3 == long_q

    # 无历史（新会话）：不包装
    ctx4 = executor._contextualize_retrieval_query("fresh-session", "第二点呢")
    assert ctx4 == "第二点呢"


def test_long_table_followup_recovers_previous_retrieval_targets() -> None:
    db = make_db()
    executor = make_executor(db)
    previous = 'Alpha 表1的 MethodA 与 MethodB 在 TaskA 和 TaskB 的成绩是多少？'
    followup = '先确定刚才两列成绩对应的方法，再计算各自的提升，并比较哪个任务提升更大。'
    executor._memory.add_turn('table-followup', role='user', content=previous, step_type='user_query')
    executor._memory.add_turn('table-followup', role='user', content=followup, step_type='user_query')
    result = executor._contextualize_retrieval_query('table-followup', followup)
    assert previous in result
    assert followup in result
    assert executor._is_table_reference_query(followup)
    assert executor._contextualize_retrieval_query('fresh', followup) == followup


def _counting_synthesize(executor) -> list[int]:
    """Wrap executor._run_synthesize with a call counter."""
    called: list[int] = []
    original = executor._run_synthesize

    def counting(*args, **kwargs):
        called.append(1)
        return original(*args, **kwargs)

    executor._run_synthesize = counting
    return called


def _cross_turn_request(
    db: Session,
    *,
    query: str = "那个的 accuracy 呢？",
    session_id: str = "synth-sess",
) -> AgentQueryRequest:
    """构造跨轮引用请求：先加一轮历史，使 synthesize 保留调用。

    9.1 之后单轮/对比路由跳过 synthesize，只有跨轮引用与
    complex_multi_hop 保留。需要验证 synthesize 行为的测试用此 helper。
    """
    memory = ConversationMemory(db)
    memory.touch_session(session_id, project_slug="demo", ttl_days=30, document_id=None)
    memory.add_turn(
        session_id,
        role="user",
        content="charmm36m 的 accuracy 是多少",
        step_type="user_query",
    )
    return AgentQueryRequest(
        project_slug="demo", query=query, session_id=session_id
    )


def _make_project(db) -> None:
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()


def _assert_direct_pass_through(response, *, rag_answer: str = "test answer") -> None:
    """Task 1：断言普通首轮直通结果（无 synthesis step、rag-direct 标记）。"""
    assert response.final_answer == rag_answer
    assert response.answer_model == "rag-direct"
    assert response.answer_provider == "local"
    assert not any(step.step_type == "synthesis" for step in response.steps)
    finalize_step = next(s for s in response.steps if s.step_type == "finalize")
    assert finalize_step.metadata.get("synthesis_skipped") is True


def test_route_matrix_skips_synthesis_for_single_turn_queries() -> None:
    """9.1：单轮路由（simple_rag / evidence_required / table_or_metric /
    multi_source_compare）跳过 synthesize，直通 rag.answer。"""
    db = make_db()
    _make_project(db)
    executor = make_executor(db)
    called = _counting_synthesize(executor)

    # simple_rag：普通知识库问题
    response = executor.execute(
        AgentQueryRequest(project_slug="demo", query="charmm36m 的 accuracy 是多少？")
    )
    assert called == [], "simple_rag 路由不应调用 synthesize"
    _assert_direct_pass_through(response)

    # evidence_required：证据类问题
    executor_er = make_executor(db)
    called_er = _counting_synthesize(executor_er)
    response_er = executor_er.execute(
        AgentQueryRequest(project_slug="demo", query="charmm36m 的 accuracy 引用了哪些来源？")
    )
    assert called_er == [], "evidence_required 路由不应调用 synthesize"
    _assert_direct_pass_through(response_er)

    # table_or_metric：指标问题（stub 答案带数值，否则 verifier 按 9.1.4
    # 判定"缺少表格/数值证据"→ retry → 正确阻止直通）
    executor2 = make_executor(db, rag_answer_text="test answer 1.18")
    called2 = _counting_synthesize(executor2)
    response2 = executor2.execute(
        AgentQueryRequest(project_slug="demo", query="OPLS5 的表格中 binding RMSE 是多少？")
    )
    assert called2 == [], "table_or_metric 路由不应调用 synthesize"
    _assert_direct_pass_through(response2, rag_answer="test answer 1.18")

    # multi_source_compare：对比问题（单轮）
    executor3 = make_executor(db)
    called3 = _counting_synthesize(executor3)
    response3 = executor3.execute(
        AgentQueryRequest(project_slug="demo", query="对比 OPLS4 和 OPLS5 的 accuracy")
    )
    assert called3 == [], "multi_source_compare 单轮不应调用 synthesize"
    _assert_direct_pass_through(response3)


def test_route_matrix_calls_synthesis_for_complex_multi_hop() -> None:
    """9.1：complex_multi_hop 路由保留 synthesize。"""
    db = make_db()
    _make_project(db)
    executor = make_executor(db)
    called = _counting_synthesize(executor)

    executor.execute(
        AgentQueryRequest(
            project_slug="demo",
            query="首先找到 OPLS5 论文，然后计算它的 binding RMSE 相比 OPLS4 的改善",
        )
    )
    assert called, "complex_multi_hop 路由应调用 synthesize"


def test_route_matrix_does_not_pass_through_insufficient_evidence() -> None:
    """9.1.4：evidence-insufficient 答案不透传（保留 synthesize/降级路径）。"""
    db = make_db()
    _make_project(db)
    executor = make_executor(
        db,
        rag_answer_text=(
            "## Insufficient Evidence\n\nThe retrieved source documents do not "
            "contain information about the specific scientific terms in your "
            "question (charmm36m)."
        ),
    )
    called = _counting_synthesize(executor)
    response = executor.execute(
        AgentQueryRequest(project_slug="demo", query="charmm36m 的 accuracy 是多少？")
    )
    # 证据不足时不跳过 synthesize（不把未经验证的草稿标记为 rag-direct）
    assert called, "evidence-insufficient 答案不应跳过 synthesize"
    assert response.answer_model != "rag-direct"


def test_route_matrix_does_not_pass_through_contradicted_rag(monkeypatch) -> None:
    """9.1.4：RAG verification_status 为 contradicted 时不透传（保留降级路径）。"""
    db = make_db()
    _make_project(db)

    class ContradictedRAG:
        def answer(self, db, project_slug, question, document_id=None):
            return QueryResponse(
                answer_markdown="The value is 21.0%.",
                citations=[Citation(document_id="d1", chunk_id="c1", score=0.9, excerpt="e")],
                verification_status="contradicted",
            )

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            return EvidencePack(status="ok", items=[])

    executor = build_executor_with_rag(db, ContradictedRAG())
    called = _counting_synthesize(executor)
    response = executor.execute(
        AgentQueryRequest(project_slug="demo", query="charmm36m 的 accuracy 是多少？")
    )
    # 矛盾验证结果不透传
    assert called, "contradicted 结果不应跳过 synthesize"
    assert response.answer_model != "rag-direct"


def test_route_matrix_does_not_pass_through_empty_rag_answer(monkeypatch) -> None:
    """9.1.4：RAG answer 为空时不透传（保留 synthesize/降级路径）。"""
    db = make_db()
    _make_project(db)

    class EmptyRAG:
        def answer(self, db, project_slug, question, document_id=None):
            return QueryResponse(
                answer_markdown="",
                citations=[Citation(document_id="d1", chunk_id="c1", score=0.9, excerpt="e")],
                verification_status="local-only",
            )

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            return EvidencePack(status="ok", items=[])

    executor = build_executor_with_rag(db, EmptyRAG())
    called = _counting_synthesize(executor)
    response = executor.execute(
        AgentQueryRequest(project_slug="demo", query="charmm36m 的 accuracy 是多少？")
    )
    assert called, "空答案不应跳过 synthesize"
    assert response.answer_model != "rag-direct"


def test_executor_passes_narrow_context_only_for_cross_turn(monkeypatch) -> None:
    """9.3.1：narrow_context 只在跨轮引用场景传给 synthesize。"""
    db = make_db()
    _make_project(db)
    executor = make_executor(db)
    captured: list[dict] = []

    original_call_tool = executor._tools.call_tool

    def capturing_call_tool(name, args=None, *, ctx=None):
        if name == "answer.synthesize" and args:
            captured.append({"narrow_context": args.get("narrow_context")})
        return original_call_tool(name, args=args, ctx=ctx)

    monkeypatch.setattr(executor._tools, "call_tool", capturing_call_tool)

    # 跨轮引用：narrow_context=True
    executor.execute(_cross_turn_request(db))
    assert captured, "跨轮场景应调用 synthesize"
    assert captured[0]["narrow_context"] is True

    # 单轮 complex_multi_hop：narrow_context=False（非跨轮）
    captured.clear()
    executor.execute(
        AgentQueryRequest(
            project_slug="demo",
            query="首先找到 OPLS5 论文，然后计算它的 binding RMSE 相比 OPLS4 的改善",
        )
    )
    assert captured, "complex 场景应调用 synthesize"
    assert captured[0]["narrow_context"] is False


def test_coverage_partial_blocks_direct_pass_through() -> None:
    """9.1.6：table_or_metric + coverage partial → 直通被阻止并记录原因。"""
    from app.schemas.agent import (
        EvidenceItem,
        TableCoverage,
        TableFactEvidence,
    )

    db = make_db()
    _make_project(db)

    class StubRAG:
        def answer(self, db, project_slug, question, document_id=None):
            from app.schemas.common import Citation, QueryResponse

            return QueryResponse(
                answer_markdown="test answer 1.18",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.92,
                        excerpt="sample excerpt",
                    )
                ],
                verification_status="local-only",
            )

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            return EvidencePack(
                status="ok",
                items=[],
                table_facts=[
                    TableFactEvidence(
                        table_id="table-7",
                        document_id="d1",
                        parse_version="v1",
                        row_label="OPLS5",
                        column="RMSE",
                        value="1.18",
                        row_index=2,
                    )
                ],
                inventory=[
                    TableCoverage(
                        document_id="d1",
                        parse_version="v1",
                        table_id="table-7",
                        row_count=3,
                        row_indices=[1, 2, 3],
                    )
                ],
                coverage_status="partial",
                coverage_missing_tables=["table-7"],
            )

    rag = StubRAG()
    tools = ToolRegistry()
    tools._register_builtins(rag)
    tools.register(
        ToolSpec(
            name="answer.synthesize",
            description="deterministic test synthesizer",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            side_effect_level="none",
        ),
        lambda args, ctx: {"answer_markdown": args.get("rag_answer", "synth")},
    )
    executor = AgentExecutor(
        rag=rag,
        tools=tools,
        memory=ConversationMemory(db),
        db=db,
        trace_store=AgentTraceStore(db),
    )
    called = _counting_synthesize(executor)
    response = executor.execute(
        AgentQueryRequest(project_slug="demo", query="OPLS5 的表格中 binding RMSE 是多少？")
    )
    assert called, "coverage partial 时 table_or_metric 不应直通，应调用 synthesize"
    assert response.answer_model != "rag-direct"
    finalize = next(s for s in response.steps if s.step_type == "finalize")
    assert finalize.metadata.get("direct_block_reason") == "coverage_partial"
    assert finalize.metadata.get("coverage_status") == "partial"


def test_coverage_unknown_does_not_block_direct_pass_through() -> None:
    """9.1.6：旧 EvidencePack 无 coverage 字段（unknown）→ 不阻塞直通。"""
    db = make_db()
    _make_project(db)
    executor = make_executor_with_retrieve(db, rag_answer_text="test answer 1.18")
    called = _counting_synthesize(executor)
    response = executor.execute(
        AgentQueryRequest(project_slug="demo", query="OPLS5 的表格中 binding RMSE 是多少？")
    )
    assert called == [], "coverage unknown 不应阻塞 table_or_metric 直通"
    _assert_direct_pass_through(response, rag_answer="test answer 1.18")


def test_coverage_partial_with_empty_table_facts_blocks_direct() -> None:
    """Task 9：table_or_metric + coverage partial + 空 table_facts 也阻止直通。

    RAG 明确声明 coverage_status="partial" 即代表表格覆盖不完整，即使
    table_facts 为空也必须阻止 rag-direct，并记录 coverage_partial 原因。
    """
    db = make_db()
    _make_project(db)

    class PartialNoFactsRAG:
        def answer(self, db, project_slug, question, document_id=None):
            return QueryResponse(
                answer_markdown="test answer 1.18",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.92,
                        excerpt="sample excerpt",
                    )
                ],
                verification_status="local-only",
            )

        def retrieve_evidence(
            self, db, project_slug, question, limit=15, document_id=None
        ):
            return EvidencePack(
                status="ok",
                items=[],
                table_facts=[],
                inventory=[],
                coverage_status="partial",
                coverage_missing_tables=["table-7"],
            )

    executor = build_executor_with_rag(db, PartialNoFactsRAG())
    executor._tools.get("answer.synthesize")["handler"] = lambda args, ctx: {
        "answer_markdown": args.get("rag_answer", ""),
        "cited_indexes": list(range(len(args.get("citations") or []))),
        "warnings": [],
        "confidence": 1.0,
        "provider": "local",
        "model": "local-fallback",
    }
    called = _counting_synthesize(executor)
    response = executor.execute(
        AgentQueryRequest(
            project_slug="demo", query="OPLS5 的表格中 binding RMSE 是多少？"
        )
    )
    assert called, "coverage partial（即使 table_facts 为空）时 table_or_metric 不应直通"
    assert response.answer_model != "rag-direct"
    finalize = next(s for s in response.steps if s.step_type == "finalize")
    assert finalize.metadata.get("direct_block_reason") == "coverage_partial"
    assert finalize.metadata.get("coverage_status") == "partial"


def test_coverage_partial_does_not_block_non_table_route() -> None:
    """Task 9：非表格路由即使收到 coverage partial 也不被 coverage 阻塞。"""
    db = make_db()
    _make_project(db)

    class PartialCoverageRAG:
        def answer(self, db, project_slug, question, document_id=None):
            return QueryResponse(
                answer_markdown="test answer",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.92,
                        excerpt="sample excerpt",
                    )
                ],
                verification_status="local-only",
            )

        def retrieve_evidence(
            self, db, project_slug, question, limit=15, document_id=None
        ):
            return EvidencePack(
                status="ok",
                items=[],
                table_facts=[],
                coverage_status="partial",
                coverage_missing_tables=["table-7"],
            )

    executor = build_executor_with_rag(db, PartialCoverageRAG())
    called = _counting_synthesize(executor)
    response = executor.execute(
        AgentQueryRequest(project_slug="demo", query="charmm36m 的 accuracy 是多少？")
    )
    assert called == [], "simple_rag 路由不应因 coverage partial 被阻塞"
    _assert_direct_pass_through(response)


def test_final_verify_retry_reverifies_retry_answer() -> None:
    """Task 9：final verify 触发 RAG retry 后，必须对 retry 的最终答案重验。

    synthesize 路径（跨轮引用 + evidence_required，max_retries=1）下，final
    verify 建议 retry → RAG retry 返回新答案与新 citations → 必须再次执行
    verify；trace 中最后一条 answer.verify 与 finalize metadata 的
    rag_verification_status 必须对应最终答案，不能复用第一次 verify 的结果。
    """
    db = make_db()
    _make_project(db)

    rag_calls: list[str] = []

    class RetryRAG:
        def answer(self, db, project_slug, question, document_id=None):
            rag_calls.append(question)
            if len(rag_calls) == 1:
                return QueryResponse(
                    answer_markdown="first draft without table value",
                    citations=[
                        Citation(
                            document_id="d1",
                            chunk_id="c1",
                            score=0.9,
                            excerpt="draft evidence",
                        )
                    ],
                    verification_status="local-only",
                )
            return QueryResponse(
                answer_markdown="retry answer with table value 1.18",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="c9",
                        score=0.95,
                        excerpt="retry evidence",
                    )
                ],
                verification_status="contradicted",
            )

    executor = build_executor_with_rag(db, RetryRAG())
    # 覆盖为确定性 synthesize，避免真实模型调用
    executor._tools.get("answer.synthesize")["handler"] = lambda args, ctx: {
        "answer_markdown": args.get("rag_answer", ""),
        "cited_indexes": list(range(len(args.get("citations") or []))),
        "warnings": [],
        "confidence": 1.0,
        "provider": "local",
        "model": "local-fallback",
    }
    verify_calls: list[dict] = []

    def recording_verify(args, ctx):
        verify_calls.append(
            {
                "answer_markdown": args.get("answer_markdown", ""),
                "citations": args.get("citations") or [],
            }
        )
        first = len(verify_calls) == 1
        return {
            "ok": True,
            "warnings": ["draft verify recommends retry"] if first else [],
            "retry_recommended": first,
            "reason": "retry" if first else "ok",
        }

    executor._tools.get("answer.verify")["handler"] = recording_verify

    response = executor.execute(
        _cross_turn_request(
            db,
            query="那个的 accuracy 引用了哪些来源？",
            session_id="retry-reverify-session",
        )
    )

    # retry 发生：rag.answer 共调用两次
    rag_steps = [s for s in response.steps if s.tool_name == "rag.answer"]
    assert len(rag_steps) == 2, f"Expected 2 rag.answer calls, got {len(rag_steps)}"
    # 最终答案采用 retry 结果
    assert response.final_answer == "retry answer with table value 1.18"
    # verify 共执行两次，且第二次收到的必须是 retry 的答案与 citations
    assert len(verify_calls) == 2, f"Expected 2 verify calls, got {len(verify_calls)}"
    last_verify = verify_calls[-1]
    assert last_verify["answer_markdown"] == "retry answer with table value 1.18"
    assert len(last_verify["citations"]) == 1
    assert last_verify["citations"][0]["chunk_id"] == "c9"
    # trace 中的最后一条 answer.verify 步骤对应重验结果（retry_recommended=False）
    verify_steps = [s for s in response.steps if s.tool_name == "answer.verify"]
    assert len(verify_steps) == 2, f"Expected 2 verify steps, got {len(verify_steps)}"
    assert verify_steps[-1].metadata.get("retry_recommended") is False
    # finalize metadata：rag_verification_status 对应 retry 返回的状态
    finalize = next(s for s in response.steps if s.step_type == "finalize")
    assert finalize.metadata.get("rag_verification_status") == "contradicted"
    # synthesize 路径不标记 rag-direct
    assert response.answer_model != "rag-direct"


def test_final_verify_retry_second_verify_still_recommends_retry() -> None:
    """Task 9：retry 后第二次 verify 仍建议 retry 时，trace 不得宣称已验证。

    重验仍建议 retry 时不再发起第二次重试（retry 有界），但最后一条
    answer.verify 步骤保留 retry_recommended=True，finalize 不得标记
    rag-direct，也不得把最终答案当作已验证通过。
    """
    db = make_db()
    _make_project(db)

    class AlwaysRetryRAG:
        def answer(self, db, project_slug, question, document_id=None):
            return QueryResponse(
                answer_markdown="answer attempt with table value 1.18",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.9,
                        excerpt="evidence",
                    )
                ],
                verification_status="local-only",
            )

    executor = build_executor_with_rag(db, AlwaysRetryRAG())
    # 覆盖为确定性 synthesize，避免真实模型调用
    executor._tools.get("answer.synthesize")["handler"] = lambda args, ctx: {
        "answer_markdown": args.get("rag_answer", ""),
        "cited_indexes": list(range(len(args.get("citations") or []))),
        "warnings": [],
        "confidence": 1.0,
        "provider": "local",
        "model": "local-fallback",
    }
    executor._tools.get("answer.verify")["handler"] = lambda args, ctx: {
        "ok": True,
        "warnings": ["verify keeps recommending retry"],
        "retry_recommended": True,
        "reason": "retry",
    }

    response = executor.execute(
        _cross_turn_request(
            db,
            query="那个的 accuracy 引用了哪些来源？",
            session_id="retry-still-retry-session",
        )
    )

    # 只发生一次 retry，不进入重试循环
    rag_steps = [s for s in response.steps if s.tool_name == "rag.answer"]
    assert len(rag_steps) == 2, f"Expected exactly 2 rag.answer calls, got {len(rag_steps)}"
    # 最后一条 verify 步骤仍保留 retry_recommended=True：trace 不得宣称通过
    verify_steps = [s for s in response.steps if s.tool_name == "answer.verify"]
    assert len(verify_steps) == 2, f"Expected 2 verify steps, got {len(verify_steps)}"
    assert verify_steps[-1].metadata.get("retry_recommended") is True
    # 未通过直通门禁：不得标记 rag-direct
    assert response.answer_model != "rag-direct"
    assert response.final_answer == "answer attempt with table value 1.18"


def test_draft_verify_retry_syncs_status_from_retry_answer() -> None:
    """Task 9：draft verify retry 采用 retry 答案后，直通门禁与 trace 只依据最终结果。

    第一次 RAG 答案触发 draft verify retry（retry_recommended=True），RAG
    retry 返回不同答案且 verification_status=contradicted：必须对 retry 的
    最终答案重新执行 verify，直通门禁不得沿用第一次答案的 local-only 状态
    错误放行，finalize trace 的 rag_verification_status 必须与最终答案一致，
    synthesize/fallback 不能被错误跳过。
    """
    db = make_db()
    _make_project(db)

    rag_calls: list[str] = []

    class DraftRetryRAG:
        def answer(self, db, project_slug, question, document_id=None):
            rag_calls.append(question)
            if len(rag_calls) == 1:
                return QueryResponse(
                    answer_markdown="first draft answer without table value",
                    citations=[
                        Citation(
                            document_id="d1",
                            chunk_id="c1",
                            score=0.9,
                            excerpt="draft evidence",
                        )
                    ],
                    verification_status="local-only",
                )
            return QueryResponse(
                answer_markdown="retry answer is contradicted",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="c9",
                        score=0.95,
                        excerpt="retry evidence",
                    )
                ],
                verification_status="contradicted",
            )

        def retrieve_evidence(
            self, db, project_slug, question, limit=15, document_id=None
        ):
            return EvidencePack(status="ok", items=[])

    executor = build_executor_with_rag(db, DraftRetryRAG())
    called = _counting_synthesize(executor)
    verify_calls: list[dict] = []

    def recording_verify(args, ctx):
        verify_calls.append(
            {
                "answer_markdown": args.get("answer_markdown", ""),
                "citations": args.get("citations") or [],
            }
        )
        first = len(verify_calls) == 1
        return {
            "ok": True,
            "warnings": ["draft verify recommends retry"] if first else [],
            "retry_recommended": first,
            "reason": "retry" if first else "ok",
        }

    executor._tools.get("answer.verify")["handler"] = recording_verify

    response = executor.execute(
        AgentQueryRequest(
            project_slug="demo", query="charmm36m 的 accuracy 引用了哪些来源？"
        )
    )

    # retry 发生：rag.answer 共调用两次
    rag_steps = [s for s in response.steps if s.tool_name == "rag.answer"]
    assert len(rag_steps) == 2, f"Expected 2 rag.answer calls, got {len(rag_steps)}"
    # 最终答案采用 retry 结果
    assert response.final_answer == "retry answer is contradicted"
    # verify 至少执行两次，且第二次验证必须针对 retry 的答案与 citations
    assert len(verify_calls) >= 2, f"Expected >= 2 verify calls, got {len(verify_calls)}"
    second_verify = verify_calls[1]
    assert second_verify["answer_markdown"] == "retry answer is contradicted"
    assert len(second_verify["citations"]) == 1
    assert second_verify["citations"][0]["chunk_id"] == "c9"
    # retry 答案 contradicted：不得标记 rag-direct，synthesize 未被错误跳过
    assert response.answer_model != "rag-direct"
    assert called, "contradicted retry 答案不应跳过 synthesize"
    # finalize trace 的 rag_verification_status 与最终答案一致（来自 retry），
    # 不保留第一次答案的 local-only 旧状态
    finalize = next(s for s in response.steps if s.step_type == "finalize")
    assert finalize.metadata.get("rag_verification_status") == "contradicted"
    assert finalize.metadata.get("direct_block_reason") == "rag_contradicted"
    assert finalize.metadata.get("synthesis_skipped") is False
