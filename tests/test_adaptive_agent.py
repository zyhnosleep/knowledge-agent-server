"""Real state machine / canonical merge; only decision and IO are deterministic."""
from dataclasses import FrozenInstanceError, replace
import threading

import pytest

from app.models.records import DocumentChunk
from app.schemas.adaptive_agent import AdaptiveDecision
from app.schemas.agent import AgentConstraints
from app.schemas.common import QueryResponse
from app.services.agent_decider import DecisionInvalid, DecisionUnavailable
from app.services.agent_model_router import InferenceTarget
from app.services.execution_budget import ExecutionBudget, BudgetExceeded, execution_budget_scope
from app.services.runtime_contract import EmbeddingIdentity
from app.services.adaptive_agent import AdaptiveAgent, RequestScope
from test_adaptive_evidence import snapshot
from test_visual_routing import evidence


TARGET = InferenceTarget("generation", "http://127.0.0.1:18080", "base", 32768, "test")
IDENTITY = EmbeddingIdentity("ollama", "vl", "weights", "processor", 2)


class Decider:
    def __init__(self, choose):
        self.choose, self.observations = choose, []

    def decide(self, observation, target):
        self.observations.append(observation)
        return self.choose(observation)


@pytest.fixture
def scenario(evidence):
    db, service, root = evidence
    db.add(DocumentChunk(id="table3", document_id="d1", parse_version="v5", chunk_role="child",
        block_type="table", ordinal=1, source_spans=[{"table_id": "table-3"}],
        text="Table 3\n| Model | Accuracy |\n|---|---|\n| A | 94% |", embedding=[1, 0]))
    db.commit()
    figures = service._search_document_figure_contexts("Figure 1 legend", "p1", ["d1"])
    all_contexts = figures + service._search_source_chunks("Table 3 accuracy", "p1", ["d1"], question_vector=[1, 0])
    assert figures
    sparse = snapshot(service, figures, versions={"d1": "v5"})
    complete = snapshot(service, all_contexts, versions={"d1": "v5"})
    scope = RequestScope("pilot", "session", "d1", {"d1": "v5"}, IDENTITY)
    sparse, complete = replace(sparse, document_id="d1"), replace(complete, document_id="d1")
    return service, sparse, complete, scope


def execute(scenario, choose, *, initial=None, retrieve=None, answer=None, constraints=None,
            max_decisions=3, max_supplements=2, cancel_event=None, clock=None):
    service, sparse, complete, scope = scenario
    decider, steps, retrievals = Decider(choose), [], []
    def default_retrieve(query, document, limit):
        retrievals.append((query, document, limit))
        return complete
    def default_answer(prepared):
        return QueryResponse(answer_markdown="94% [0]", citations=[prepared.contexts[0].citation],
                             verification_status="local-only")
    kwargs = {"cancel_event": cancel_event}
    if clock:
        kwargs["clock"] = clock
    budget = ExecutionBudget(constraints or AgentConstraints(max_steps=12, max_tool_calls=6), **kwargs)
    agent = AdaptiveAgent(decider, retrieve or default_retrieve, answer or default_answer,
                          steps.append, merge=service.merge_prepared_snapshots)
    with execution_budget_scope(budget):
        outcome = agent.run(question="Compare Figure 2 and Table 3", conversation_summary="Previous method question",
            scope=scope, initial=initial or sparse, target=TARGET, budget=budget,
            max_decisions=max_decisions, max_supplements=max_supplements)
    return outcome, decider, steps, retrievals, budget


def test_observation_changes_actual_query_and_then_answers(scenario):
    def choose(obs):
        if not any(e["table_id"] == "table-3" for e in obs.evidence):
            return AdaptiveDecision(action="retrieve", query="论文 A 表 3 最优方法指标", document_id="d1", limit=7)
        return AdaptiveDecision(action="answer")
    sparse, decider, steps, retrievals, _ = execute(scenario, choose)
    complete, _, _, full_retrievals, _ = execute(scenario, choose, initial=scenario[2])
    assert retrievals == [("论文 A 表 3 最优方法指标", "d1", 7)]
    assert full_retrievals == []
    assert sparse.supplement_retrievals == 1 and complete.supplement_retrievals == 0
    assert sparse.decisions == 2 and complete.decisions == 1
    assert sparse.answer is not None and sparse.stop_reason == "answer"
    assert len(decider.observations[1].evidence) > len(decider.observations[0].evidence)
    assert steps[0].metadata["decision_index"] == 1
    assert any(s.metadata.get("new_evidence") for s in steps)


def test_request_scope_copies_and_freezes_versions():
    original = {"d1": "v5"}
    scope = RequestScope("pilot", "session", None, original, IDENTITY)
    original["d1"] = "v6"
    assert scope.parse_version_map["d1"] == "v5"
    with pytest.raises(TypeError):
        scope.parse_version_map["d2"] = "v5"
    with pytest.raises(FrozenInstanceError):
        scope.project_slug = "other"


def test_no_progress_stops_without_another_decision(scenario):
    outcome, decider, _, _, _ = execute(scenario, lambda obs: AdaptiveDecision(action="retrieve", query="legend"),
                                      retrieve=lambda *args: scenario[1])
    assert outcome.stop_reason == "no_progress" and outcome.answer is None
    assert len(decider.observations) == 1


@pytest.mark.parametrize("focus", ["foreign", "d2"])
def test_focus_cannot_expand_request_scope(scenario, focus):
    def forbidden(*args):
        pytest.fail("out-of-scope retrieval ran")
    outcome, _, _, _, _ = execute(scenario, lambda obs: AdaptiveDecision(action="retrieve", query="x", document_id=focus),
                                  retrieve=forbidden)
    assert outcome.stop_reason == "scope_violation"


def test_scope_mismatch_is_rejected_before_planner(scenario):
    def forbidden(obs):
        pytest.fail("planner received another project's evidence")
    outcome, _, _, _, _ = execute(scenario, forbidden, initial=replace(scenario[1], project_slug="other"))
    assert outcome.stop_reason == "scope_violation"


def test_forged_supplement_does_not_count_as_progress(scenario):
    forged = replace(scenario[2], parse_version_map={"d1": "v6"})
    outcome, _, _, _, _ = execute(scenario, lambda obs: AdaptiveDecision(action="retrieve", query="x"),
                                  retrieve=lambda *args: forged)
    assert outcome.stop_reason == "scope_violation"
    assert outcome.prepared.parse_version_map == {"d1": "v5"}


def test_finish_requires_a_candidate(scenario):
    outcome, _, _, _, _ = execute(scenario, lambda obs: AdaptiveDecision(action="finish"))
    assert outcome.stop_reason == "decision_invalid" and outcome.answer is None


def test_empty_evidence_abstains_without_answer_io(scenario):
    empty = replace(scenario[1], contexts=[], pack=scenario[1].pack.model_copy(update={"items": [], "status": "empty"}))
    def forbidden(*args):
        pytest.fail("empty-evidence answer ran")
    outcome, _, _, _, _ = execute(scenario, lambda obs: AdaptiveDecision(action="abstain"), initial=empty, answer=forbidden)
    assert outcome.stop_reason == "abstain" and outcome.answer is None


def test_finish_uses_exact_existing_candidate_without_new_generation(scenario):
    initial = replace(scenario[2], pack=scenario[2].pack.model_copy(update={"coverage_status": "partial"}))
    candidate = QueryResponse(answer_markdown="Known value 94% [0]", citations=[initial.contexts[0].citation],
                              verification_status="local-only")
    def choose(obs):
        return AdaptiveDecision(action="finish" if obs.candidate_answer else "answer")
    calls = []
    def answer(prepared):
        calls.append(prepared)
        return candidate
    outcome, decider, _, _, _ = execute(scenario, choose, initial=initial, answer=answer)
    assert outcome.answer is candidate and outcome.stop_reason == "finish"
    assert len(calls) == 1 and len(decider.observations) == 2


def test_answer_cannot_return_foreign_citations(scenario):
    foreign = scenario[1].contexts[0].citation.model_copy(update={"document_id": "other"})
    outcome, _, _, _, _ = execute(scenario, lambda obs: AdaptiveDecision(action="answer"),
        answer=lambda p: QueryResponse(answer_markdown="x", citations=[foreign], verification_status="local-only"))
    assert outcome.stop_reason == "scope_violation" and outcome.answer is None


@pytest.mark.parametrize("error,reason", [(RuntimeError("secret file"), "tool_error"),
    (BudgetExceeded("deadline"), "deadline")])
def test_tool_error_is_not_evidence_absence(scenario, error, reason):
    def retrieve(*args):
        raise error
    outcome, _, steps, _, _ = execute(scenario, lambda obs: AdaptiveDecision(action="retrieve", query="x"), retrieve=retrieve)
    assert outcome.stop_reason == reason
    assert "secret file" not in str([s.model_dump() for s in steps])


@pytest.mark.parametrize("error,reason", [(DecisionInvalid(), "decision_invalid"),
                                        (DecisionUnavailable(), "decision_unavailable")])
def test_planner_failures_have_distinct_sanitized_stops(scenario, error, reason):
    def choose(obs):
        raise error
    outcome, _, _, _, _ = execute(scenario, choose)
    assert outcome.stop_reason == reason


@pytest.mark.parametrize("constraints,reason", [(AgentConstraints(max_steps=4), "step_limit"),
    (AgentConstraints(max_tool_calls=1), "tool_limit"), (AgentConstraints(budget_tokens=1), "token_limit")])
def test_low_budget_reserves_tail_before_planner(scenario, constraints, reason):
    def forbidden(obs):
        pytest.fail("planner consumed the finalization allowance")
    outcome, decider, steps, _, _ = execute(scenario, forbidden, constraints=constraints)
    assert outcome.stop_reason == reason and not decider.observations
    assert len(steps) + 2 <= constraints.max_steps


def test_default_eight_steps_preserves_verify_and_finalize(scenario):
    outcome, _, steps, _, _ = execute(scenario, lambda obs: AdaptiveDecision(action="retrieve", query="Table 3"),
                                      constraints=AgentConstraints())
    assert outcome.stop_reason == "step_limit"
    assert len(steps) + 2 + 2 <= 8  # initial route/retrieve and tail verify/finalize


def test_cancelled_does_not_call_planner(scenario):
    cancelled = threading.Event()
    cancelled.set()
    outcome, decider, _, _, _ = execute(scenario, lambda obs: pytest.fail("planner ran"), cancel_event=cancelled)
    assert outcome.stop_reason == "cancelled" and decider.observations == []


def test_deadline_expiring_during_retrieval_stops_before_answer(scenario):
    now = [0.0]
    def retrieve(*args):
        now[0] = 46.0
        return scenario[2]
    outcome, decider, _, _, _ = execute(scenario, lambda obs: AdaptiveDecision(action="retrieve", query="Table 3"),
        retrieve=retrieve, clock=lambda: now[0])
    assert outcome.stop_reason == "deadline" and len(decider.observations) == 1


def test_supplement_limit_does_not_run_a_third_tool(scenario):
    outcome, decider, _, retrievals, _ = execute(scenario, lambda obs: AdaptiveDecision(action="retrieve", query="Table 3"),
                                               max_supplements=0)
    assert outcome.stop_reason == "supplement_limit" and retrievals == []


def test_decision_limit_never_reports_an_answer_not_generated(scenario):
    outcome, decider, _, retrievals, _ = execute(scenario, lambda obs: AdaptiveDecision(action="retrieve", query="Table 3"),
                                               max_decisions=1)
    assert outcome.stop_reason == "decision_limit" and outcome.answer is None
    assert len(decider.observations) == 1


def test_second_retrieval_parameters_come_from_new_observation(scenario):
    service, sparse, complete, scope = scenario
    service.db.add(DocumentChunk(id="mechanism", document_id="d1", parse_version="v5", chunk_role="child",
        block_type="paragraph", ordinal=2, text="A uses attention for alignment.", embedding=[1, 0]))
    service.db.commit()
    mechanism = service._search_source_chunks("attention alignment", "p1", ["d1"], question_vector=[1, 0])
    final = snapshot(service, complete.contexts + mechanism, versions={"d1": "v5"}, document_id="d1")
    queries = []
    def choose(obs):
        if any(e["chunk_id"] == "mechanism" for e in obs.evidence):
            return AdaptiveDecision(action="answer")
        table = next((e for e in obs.evidence if e["table_id"] == "table-3"), None)
        if table:
            assert "94%" in table["excerpt"]
            return AdaptiveDecision(action="retrieve", query="Model A 94% attention alignment")
        return AdaptiveDecision(action="retrieve", query="Table 3 accuracy")
    def retrieve(query, document, limit):
        queries.append(query)
        return complete if len(queries) == 1 else final
    outcome, _, _, _, _ = execute(scenario, choose, retrieve=retrieve)
    assert queries == ["Table 3 accuracy", "Model A 94% attention alignment"]
    assert outcome.decisions == 3 and outcome.supplement_retrievals == 2
    assert outcome.stop_reason == "answer"


def test_repeated_query_after_progress_stops_before_duplicate_io(scenario):
    outcome, _, _, retrievals, _ = execute(scenario, lambda obs: AdaptiveDecision(action="retrieve", query="Table 3"))
    assert outcome.stop_reason == "no_progress" and len(retrievals) == 1


def test_unknown_decider_error_does_not_leak(scenario):
    def choose(obs):
        raise RuntimeError("private observation secret")
    outcome, _, steps, _, _ = execute(scenario, choose)
    assert outcome.stop_reason == "decision_invalid"
    assert "private observation secret" not in str(steps)
