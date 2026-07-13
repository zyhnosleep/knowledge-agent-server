from __future__ import annotations

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base
from app.models.records import Project
from app.schemas.agent import AgentQueryRequest, ToolSpec
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
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


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


def test_execute_empty_answer() -> None:
    """Executor still succeeds when RAG returns empty answer."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    executor = make_executor(db, rag_answer_text="")
    request = AgentQueryRequest(project_slug="demo", query="empty?")
    response = executor.execute(request)

    assert response.status == "completed"
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
    assert response.status in ("completed", "max_steps")
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
    assert response.status in ("completed", "max_steps")


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
    request = AgentQueryRequest(project_slug="demo", query="hello?")
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
    assert response.status in ("completed", "max_steps")


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
    request = AgentQueryRequest(project_slug="demo", query="hello?")
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
    request = AgentQueryRequest(project_slug="demo", query="hello?")
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
    request = AgentQueryRequest(project_slug="demo", query="hello?")
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
