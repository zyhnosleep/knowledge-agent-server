"""Bounded observation-driven read-only actions, never arbitrary tool dispatch."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from app.schemas.adaptive_agent import AdaptiveDecision
from app.schemas.agent import AgentStep
from app.schemas.common import QueryResponse
from app.services.agent_decider import AgentDecider, DecisionUnavailable, build_observation
from app.services.agent_model_router import InferenceTarget
from app.services.execution_budget import BudgetExceeded, ExecutionBudget, current_execution_budget
from app.services.runtime_contract import EmbeddingIdentity
from app.services.search import PreparedEvidence


@dataclass(frozen=True)
class RequestScope:
    project_slug: str
    session_id: str
    document_id: str | None
    parse_version_map: Mapping[str, str]
    embedding_identity: EmbeddingIdentity

    def __post_init__(self):
        versions = dict(self.parse_version_map)
        if (not self.project_slug or not self.session_id
                or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in versions.items())
                or (self.document_id is not None and self.document_id not in versions)):
            raise ValueError("Invalid frozen request scope")
        object.__setattr__(self, "parse_version_map", MappingProxyType(versions))


@dataclass
class AdaptiveOutcome:
    prepared: PreparedEvidence
    answer: QueryResponse | None
    stop_reason: str
    decisions: int
    supplement_retrievals: int


def _identity(item):
    return (item.document_id, item.parse_version or "legacy", item.chunk_id,
            item.attachment_id, item.table_id, item.figure_id, item.asset_id,
            item.excerpt if not item.chunk_id else "")


def _scope_matches(scope: RequestScope, prepared: PreparedEvidence, focus: str | None) -> bool:
    if (prepared.project_slug != scope.project_slug or dict(scope.parse_version_map) != prepared.parse_version_map
            or prepared.document_id != focus):
        return False
    for item in prepared.pack.items:
        if item.attachment_id:
            continue  # Initial attachments are executor-authorized; supplements cannot add them.
        if (item.document_id not in scope.parse_version_map
                or (item.parse_version or "legacy") != scope.parse_version_map[item.document_id]
                or (focus is not None and item.document_id != focus)):
            return False
    return True


class AdaptiveAgent:
    def __init__(self, decider: AgentDecider,
                 retrieve: Callable[[str, str | None, int], PreparedEvidence],
                 answer: Callable[[PreparedEvidence], QueryResponse],
                 emit_step: Callable[[AgentStep], None], *,
                 merge: Callable[[PreparedEvidence, PreparedEvidence], PreparedEvidence]):
        self.decider, self.retrieve, self.answer = decider, retrieve, answer
        self.emit_step, self.merge = emit_step, merge

    def run(self, *, question: str, conversation_summary: str, scope: RequestScope,
            initial: PreparedEvidence, target: InferenceTarget, budget: ExecutionBudget,
            max_decisions: int = 3, max_supplements: int = 2,
            step_count: int = 2, tool_calls: int = 1) -> AdaptiveOutcome:
        prepared, candidate = initial, None
        decisions, supplements = 0, 0
        queries: set[tuple[str, str | None]] = set()

        def outcome(reason):
            return AdaptiveOutcome(prepared, candidate, reason, decisions, supplements)

        def emit(kind, *, action=None, ok=True, metadata=None):
            nonlocal step_count, tool_calls
            tool = {"retrieve": "rag.retrieve_evidence", "answer": "rag.answer"}.get(kind)
            data = {"execution_mode": "adaptive", "decision_index": decisions, **(metadata or {})}
            if action:
                data["action"] = action
            self.emit_step(AgentStep(step_id=step_count, step_type=kind,
                summary=f"Adaptive {kind}" + (f": {action}" if action else ""),
                tool_name=tool, tool_ok=ok if tool else None, metadata=data))
            step_count += 1
            tool_calls += int(tool is not None)

        def allowance(*, operation_steps: int, operation_tools: int, need_answer=True):
            budget.check_deadline()
            tail_steps, tail_tools = (3, 2) if need_answer else (2, 1)
            if step_count + operation_steps + tail_steps > budget.constraints.max_steps:
                return "step_limit"
            if tool_calls + operation_tools + tail_tools > budget.constraints.max_tool_calls:
                return "tool_limit"
            if budget.usage_metadata()["remaining_tokens"] <= 512:
                return "token_limit"
            return None

        if not _scope_matches(scope, prepared, scope.document_id):
            return outcome("scope_violation")
        if current_execution_budget() is not budget:
            return outcome("decision_invalid")
        max_decisions, max_supplements = max(0, min(3, max_decisions)), max(0, min(2, max_supplements))
        try:
            while decisions < max_decisions:
                reason = allowance(operation_steps=1, operation_tools=0, need_answer=candidate is None)
                if reason:
                    return outcome(reason)
                observation = build_observation(question=question, conversation_summary=conversation_summary,
                    prepared=prepared, candidate=candidate, budget=budget)
                try:
                    raw = self.decider.decide(observation, target)
                    decision = AdaptiveDecision.model_validate(raw.model_dump(exclude_unset=True))
                except DecisionUnavailable:
                    return outcome("decision_unavailable")
                except BudgetExceeded:
                    raise
                except Exception:
                    return outcome("decision_invalid")
                decisions += 1
                budget.check_deadline()
                emit("decision", action=decision.action, metadata={"reason": decision.reason})
                if decision.action == "abstain":
                    candidate = None
                    return outcome("abstain")
                if decision.action == "finish":
                    return outcome("finish" if candidate is not None else "decision_invalid")
                if decision.action == "retrieve":
                    focus = decision.document_id or scope.document_id
                    if (focus is not None and focus not in scope.parse_version_map
                            or scope.document_id is not None and focus != scope.document_id):
                        return outcome("scope_violation")
                    if supplements >= max_supplements:
                        return outcome("supplement_limit")
                    reason = allowance(operation_steps=1, operation_tools=1)
                    if reason:
                        return outcome(reason)
                    key = (" ".join(decision.query.casefold().split()), focus)
                    if key in queries:
                        return outcome("no_progress")
                    queries.add(key)
                    supplements += 1
                    before = {_identity(item) for item in prepared.pack.items}
                    try:
                        extra = self.retrieve(decision.query, focus, decision.limit)
                        if not _scope_matches(scope, extra, focus):
                            emit("retrieve", ok=False, metadata={"error": "scope_violation"})
                            return outcome("scope_violation")
                        merged = self.merge(prepared, extra)
                        if not _scope_matches(scope, merged, scope.document_id):
                            emit("retrieve", ok=False, metadata={"error": "scope_violation"})
                            return outcome("scope_violation")
                    except BudgetExceeded:
                        raise
                    except Exception:
                        emit("retrieve", ok=False, metadata={"error": "tool_error"})
                        return outcome("tool_error")
                    new = {_identity(item) for item in merged.pack.items} - before
                    emit("retrieve", metadata={"document_id": focus, "limit": decision.limit,
                        "query": decision.query[:240], "retrieval_backend": extra.retrieval_backend,
                        "new_evidence": [list(identity[:7]) for identity in sorted(new, key=str)]})
                    prepared = merged
                    if not new:
                        return outcome("no_progress")
                    candidate = None  # A new snapshot requires a fresh, identity-aligned candidate.
                    continue
                # The only remaining schema action is answer; no model-selected tool name.
                if not prepared.pack.items:
                    return outcome("abstain")
                reason = allowance(operation_steps=1, operation_tools=1, need_answer=False)
                if reason:
                    return outcome(reason)
                try:
                    generated = self.answer(prepared)
                    known = {_identity(item) for item in prepared.pack.items}
                    if any(_identity(citation) not in known for citation in generated.citations):
                        emit("answer", ok=False, metadata={"error": "scope_violation"})
                        return outcome("scope_violation")
                    candidate = generated
                except BudgetExceeded:
                    raise
                except Exception:
                    emit("answer", ok=False, metadata={"error": "tool_error"})
                    return outcome("tool_error")
                emit("answer", metadata={"evidence_snapshot_reused": True,
                    "visual_evidence": prepared.visual_evidence_trace, "answer_chars": len(candidate.answer_markdown)})
                budget.check_deadline()
                if prepared.pack.coverage_status != "partial":
                    return outcome("answer")
            return outcome("decision_limit")
        except BudgetExceeded as exc:
            return outcome(exc.reason)
