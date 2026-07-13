from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.schemas.agent import (
    AgentConstraints,
    AgentQueryRequest,
    AgentQueryResponse,
    AgentRouteDecision,
    AgentStep,
    AgentUsage,
    ComplexPlan,
)
from app.schemas.common import Citation
from app.services.agent_policy import PolicyRouter
from app.services.agent_synthesizer import AgentSynthesizer
from app.services.agent_trace_store import AgentTraceStore
from app.services.conversation_memory import ConversationMemory
from app.services.rag_adapter import RAGAdapter, _INSUFFICIENT_EVIDENCE_RE
from app.services.session_attachments import retrieve_session_attachment_evidence
from app.services.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentExecutor:
    """Orchestrates a single-turn Agent query using read-only RAG.

    The v3 executor follows a deterministic, bounded loop:

    1. **route** — classify the query with ``PolicyRouter``
    2. **plan** — for ``complex_multi_hop`` routes only, emit a
       structured, auditable plan step with whitelist/forbidden tools
    3. **rag.answer** — call RAG via the ToolRegistry
    4. **answer.synthesize** — synthesize answer from RAG evidence
    5. **answer.verify** — quality check the result
    6. **retry** (optional) — at most one additional ``rag.answer``
       call when verification recommends it and the route allows it
    7. **finalize** — compose the ``AgentQueryResponse``

    Step and tool-call limits from ``AgentConstraints`` are always
    respected.  The ``needs_clarification`` route skips RAG entirely.
    """

    def __init__(
        self,
        *,
        rag: RAGAdapter,
        tools: ToolRegistry,
        memory: ConversationMemory,
        db: Session,
        trace_store: AgentTraceStore | None = None,
        synthesizer: AgentSynthesizer | None = None,
    ) -> None:
        self._rag = rag
        self._tools = tools
        self._memory = memory
        self._db = db
        self._trace_store = trace_store
        self._synthesizer = synthesizer

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def execute(self, request: AgentQueryRequest) -> AgentQueryResponse:
        """Run the agent loop and return a structured response.

        ``max_steps`` is a strict trace boundary — no step is appended
        when ``len(steps) >= constraints.max_steps``.
        """
        request_id = str(uuid4())
        session_id = request.session_id or f"sess_{uuid4().hex[:12]}"
        constraints = request.constraints
        steps: list[AgentStep] = []
        usage = AgentUsage()
        warnings: list[str] = []
        t_start = time.monotonic()
        max_steps_hit = False

        # ---- touch conversation session TTL ----
        self._memory.touch_session(
            session_id,
            project_slug=request.project_slug,
            ttl_days=settings.agent_conversation_ttl_days,
            document_id=request.document_id,
        )
        self._memory.purge_expired_sessions()

        # ---- record user turn ----
        self._memory.add_turn(
            session_id, role="user", content=request.query, step_type="user_query"
        )
        self._compact_if_needed(session_id, constraints)
        # Release SQLite's write lock before retrieval and model calls, which
        # may take much longer than the initial conversation bookkeeping.
        self._commit_progress()

        try:
            # ==============================================================
            # Step 0: route
            # ==============================================================
            route_t0 = time.monotonic()
            route = PolicyRouter().route(request.query)
            route_latency = int((time.monotonic() - route_t0) * 1000)

            route_step = AgentStep(
                step_id=0,
                step_type="route",
                summary=f"Route: {route.route} (retries={route.max_retries})",
                latency_ms=route_latency,
                metadata={
                    "route": route.route,
                    "reason": route.reason,
                    "max_retries": route.max_retries,
                },
            )
            steps.append(route_step)
            max_steps_hit = len(steps) >= constraints.max_steps

            # ==============================================================
            # Step 0.5: plan (complex_multi_hop only)
            # ==============================================================
            if route.route == "complex_multi_hop" and not max_steps_hit:
                plan = ComplexPlan(
                    plan_type="controlled_complex",
                    allowed_tools=[
                        "rag.retrieve_evidence",
                        "rag.answer",
                        "answer.synthesize",
                        "answer.verify",
                    ],
                    forbidden_tools=[
                        "shell",
                        "sql",
                        "write",
                        "network",
                        "file_system_write",
                        "arbitrary_code_execution",
                    ],
                    max_steps=constraints.max_steps,
                    max_tool_calls=constraints.max_tool_calls,
                    subtasks=[
                        "retrieve_evidence",
                        "rag_answer",
                        "synthesize",
                        "verify",
                    ],
                )
                plan_step = AgentStep(
                    step_id=len(steps),
                    step_type="plan",
                    summary=(
                        f"Complex plan: type={plan.plan_type}, "
                        f"allowed_tools={plan.allowed_tools}, "
                        f"forbidden_tools={plan.forbidden_tools}"
                    ),
                    latency_ms=0,
                    metadata={
                        "plan_type": plan.plan_type,
                        "allowed_tools": plan.allowed_tools,
                        "forbidden_tools": plan.forbidden_tools,
                        "max_steps": plan.max_steps,
                        "max_tool_calls": plan.max_tool_calls,
                        "subtasks": plan.subtasks,
                    },
                )
                steps.append(plan_step)
                max_steps_hit = len(steps) >= constraints.max_steps

            # ---- needs_clarification: return early ----
            if route.route == "needs_clarification":
                warnings.append(
                    f"Query needs clarification: {route.reason}. "
                    "Please provide a more specific question."
                )
                if not max_steps_hit:
                    step_id = len(steps)
                    finalize_step = AgentStep(
                        step_id=step_id,
                        step_type="finalize",
                        summary="Needs clarification — no RAG call",
                        metadata={"route": route.route},
                    )
                    steps.append(finalize_step)
                else:
                    warnings.append(
                        f"Reached max_steps limit ({constraints.max_steps}) "
                        "before finalize — trace truncated"
                    )
                usage.steps = len(steps)

                self._memory.add_turn(
                    session_id,
                    role="agent",
                    content="[needs_clarification]",
                    step_type="finalize",
                )

                total_elapsed = int((time.monotonic() - t_start) * 1000)
                # ---- persist trace ----
                trace_id = None
                if self._trace_store is not None:
                    try:
                        trace_id = self._trace_store.persist_run(
                            request_id=request_id,
                            session_id=session_id,
                            project_slug=request.project_slug,
                            query=request.query,
                            constraints=constraints.model_dump(),
                            route=route.route,
                            steps=steps,
                            usage=usage,
                            final_answer="",
                            citations=[],
                            warnings=warnings,
                            status="max_steps" if max_steps_hit else "completed",
                            latency_ms=total_elapsed,
                            provider="local",
                            model="local-fallback",
                        )
                    except Exception:
                        logger.exception("Failed to persist needs_clarification trace")

                return AgentQueryResponse(
                    request_id=request_id,
                    session_id=session_id,
                    status="max_steps" if max_steps_hit else "completed",
                    final_answer="",
                    citations=[],
                    steps=steps,
                    usage=usage,
                    route=route,
                    warnings=warnings,
                    trace_id=trace_id,
                    answer_provider="local",
                    answer_model="local-fallback",
                )

            # ==============================================================
            # Normal routes: retrieve → rag.answer → synthesize → verify
            # ==============================================================
            # Stop early if step budget already exhausted
            if max_steps_hit:
                status = self._finalize_truncated(
                    steps, usage, warnings, route, constraints,
                    request_id, session_id, "", [], t_start, request,
                )
                return status

            # ---- rag.retrieve_evidence (v4) ----
            evidence_pack = self._run_retrieve_evidence(
                request.project_slug,
                request.query,
                request.document_id,
                constraints,
                steps,
                usage,
                session_id,
            )

            # ---- session-scoped temporary attachments ----
            session_attachment_pack = self._run_retrieve_session_attachments(
                request.project_slug,
                request.query,
                constraints,
                steps,
                usage,
                session_id,
            )
            evidence_pack = self._merge_evidence_pack(
                evidence_pack, session_attachment_pack, max_items=15
            )

            self._commit_progress()
            answer_text, citations, tool_calls_used = self._run_rag_answer(
                request.project_slug,
                request.query,
                request.document_id,
                constraints,
                steps,
                usage,
                session_id,
            )
            tool_calls = tool_calls_used
            attachment_citations = self._session_attachment_citations(evidence_pack)
            if attachment_citations:
                citations = self._merge_citations(citations, attachment_citations)

            # ---- detect insufficient evidence ----
            _evidence_insufficient = bool(
                answer_text and _INSUFFICIENT_EVIDENCE_RE.search(answer_text)
            )
            if attachment_citations and (_evidence_insufficient or not answer_text.strip()):
                answer_text = self._draft_session_attachment_answer(
                    request.query, attachment_citations
                )
                warnings.append(
                    "Answered from temporary attachments scoped to this session."
                )
            elif _evidence_insufficient:
                warnings.append(
                    "Insufficient evidence — retrieved documents do not contain "
                    "information relevant to the query terms. Consider uploading "
                    "the target documents."
                )

            # ---- answer.synthesize (v3) ----
            synth_provider = "local"
            synth_model = "local-fallback"
            if not max_steps_hit:
                self._commit_progress()
                synth_result = self._run_synthesize(
                    request.query,
                    route.route,
                    answer_text,
                    citations,
                    constraints,
                    steps,
                    usage,
                    session_id,
                    evidence_pack=evidence_pack,
                )
                if synth_result.get("ok"):
                    synth_data = synth_result.get("result", {})
                    answer_text = synth_data.get("answer_markdown", answer_text)
                    synth_provider = synth_data.get("provider", "local")
                    synth_model = synth_data.get("model", "local-fallback")
                    synth_warnings = synth_data.get("warnings", [])
                    if isinstance(synth_warnings, list):
                        warnings.extend(synth_warnings)
                    # Sanitize citations against synthesis cited_indexes
                    cited_indexes = synth_data.get("cited_indexes", [])
                    if isinstance(cited_indexes, list) and cited_indexes:
                        citations = [c for i, c in enumerate(citations) if i in cited_indexes]

            # ---- answer.verify ----
            verify_result = self._run_verify(
                request.query,
                answer_text,
                citations,
                route.route,
                constraints,
                steps,
                usage,
                tool_calls,
            )
            # _run_verify handles usage.tool_calls internally; track for
            # the retry gate using the current usage count.
            tool_calls = usage.tool_calls
            if verify_result["ok"]:
                verify_warnings = verify_result.get("result", {}).get("warnings", [])
                if isinstance(verify_warnings, list):
                    warnings.extend(verify_warnings)

            # ---- retry (at most one additional rag.answer) ----
            retry_recommended = (
                verify_result["ok"]
                and verify_result.get("result", {}).get("retry_recommended", False)
            )
            if (
                retry_recommended
                and route.max_retries > 0
                and tool_calls < constraints.max_tool_calls
                and len(steps) < constraints.max_steps
            ):
                retry_answer, retry_citations, retry_calls = self._run_rag_answer(
                    request.project_slug,
                    request.query,
                    request.document_id,
                    constraints,
                    steps,
                    usage,
                    session_id,
                )
                if retry_calls > 0:
                    tool_calls += retry_calls
                    if retry_answer.strip():
                        answer_text = retry_answer
                        citations = retry_citations
                    warnings.append("Retry performed after verification warning")

            # ---- check max_steps ----
            hit_limit = len(steps) >= constraints.max_steps
            if hit_limit:
                warnings.append(
                    f"Reached max_steps limit ({constraints.max_steps}). "
                    "Trace truncated."
                )

            # ==============================================================
            # Finalize (only if step budget remains)
            # ==============================================================
            if not hit_limit:
                step_id = len(steps)
                finalize_step = AgentStep(
                    step_id=step_id,
                    step_type="finalize",
                    summary=(
                        f"Agent produced final answer with "
                        f"{len(citations)} citations"
                    ),
                    metadata={"route": route.route},
                )
                steps.append(finalize_step)

            # Approximate token usage
            prompt_tokens = _estimate_tokens(request.query)
            completion_tokens = _estimate_tokens(answer_text)
            usage.prompt_tokens = prompt_tokens
            usage.completion_tokens = completion_tokens
            usage.steps = len(steps)

            self._memory.add_turn(
                session_id,
                role="agent",
                content=answer_text[:4000],
                step_type="finalize",
                citations=[c.model_dump() for c in citations],
            )

            total_elapsed = int((time.monotonic() - t_start) * 1000)
            status = "completed"
            if hit_limit:
                status = "max_steps"
            if total_elapsed > constraints.timeout_seconds * 1000:
                status = "timeout"

            # ---- persist trace ----
            trace_id = None
            if self._trace_store is not None:
                try:
                    raw_citations_for_trace = [
                        {
                            "document_id": c.document_id,
                            "chunk_id": c.chunk_id,
                            "page_slug": c.page_slug,
                            "page_title": c.page_title,
                            "excerpt": c.excerpt,
                            "score": c.score,
                        }
                        for c in citations
                    ]
                    trace_id = self._trace_store.persist_run(
                        request_id=request_id,
                        session_id=session_id,
                        project_slug=request.project_slug,
                        query=request.query,
                        constraints=constraints.model_dump(),
                        route=route.route,
                        steps=steps,
                        usage=usage,
                        final_answer=answer_text,
                        citations=raw_citations_for_trace,
                        warnings=warnings,
                        status=status,
                        latency_ms=total_elapsed,
                        provider=synth_provider,
                        model=synth_model,
                    )
                except Exception:
                    logger.exception("Failed to persist trace run")

            return AgentQueryResponse(
                request_id=request_id,
                session_id=session_id,
                status=status,
                final_answer=answer_text,
                citations=citations,
                steps=steps,
                usage=usage,
                route=route,
                warnings=warnings,
                trace_id=trace_id,
                answer_provider=synth_provider,
                answer_model=synth_model,
            )

        except Exception as exc:
            logger.exception("AgentExecutor.execute failed")
            if len(steps) < constraints.max_steps:
                steps.append(
                    AgentStep(
                        step_id=len(steps),
                        step_type="finalize",
                        summary=f"Error: {exc}",
                    )
                )
            usage.steps = len(steps)
            total_elapsed = int((time.monotonic() - t_start) * 1000)

            # ---- persist trace for error ----
            trace_id = None
            if self._trace_store is not None:
                try:
                    trace_id = self._trace_store.persist_run(
                        request_id=request_id,
                        session_id=session_id,
                        project_slug=request.project_slug,
                        query=request.query,
                        constraints=constraints.model_dump(),
                        route=None,
                        steps=steps,
                        usage=usage,
                        final_answer="",
                        citations=[],
                        warnings=warnings,
                        status="error",
                        latency_ms=total_elapsed,
                        provider="local",
                        model="local-fallback",
                    )
                except Exception:
                    logger.exception("Failed to persist error trace")

            return AgentQueryResponse(
                request_id=request_id,
                session_id=session_id,
                status="error",
                final_answer="",
                citations=[],
                steps=steps,
                usage=usage,
                warnings=warnings,
                trace_id=trace_id,
                answer_provider="local",
                answer_model="local-fallback",
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _commit_progress(self) -> None:
        """Persist short bookkeeping writes before a potentially long step."""
        try:
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def _finalize_truncated(
        self,
        steps: list[AgentStep],
        usage: AgentUsage,
        warnings: list[str],
        route: AgentRouteDecision,
        constraints: AgentConstraints,
        request_id: str,
        session_id: str,
        answer_text: str,
        citations: list[Citation],
        t_start: float,
        request: AgentQueryRequest,
    ) -> AgentQueryResponse:
        """Build a response when the step budget was exhausted early."""
        warnings.append(
            f"Reached max_steps limit ({constraints.max_steps}). "
            "Trace truncated — no tool calls were made."
        )
        prompt_tokens = _estimate_tokens(request.query)
        completion_tokens = _estimate_tokens(answer_text)
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.steps = len(steps)

        self._memory.add_turn(
            session_id,
            role="agent",
            content="[truncated - max_steps]",
            step_type="finalize",
        )

        total_elapsed = int((time.monotonic() - t_start) * 1000)
        status = "max_steps"
        if total_elapsed > constraints.timeout_seconds * 1000:
            status = "timeout"

        # ---- persist trace ----
        trace_id = None
        if self._trace_store is not None:
            try:
                trace_id = self._trace_store.persist_run(
                    request_id=request_id,
                    session_id=session_id,
                    project_slug=request.project_slug,
                    query=request.query,
                    constraints=constraints.model_dump(),
                    route=route.route,
                    steps=steps,
                    usage=usage,
                    final_answer="",
                    citations=[],
                    warnings=warnings,
                    status=status,
                    latency_ms=total_elapsed,
                    provider="local",
                    model="local-fallback",
                )
            except Exception:
                logger.exception("Failed to persist truncated trace")

        return AgentQueryResponse(
            request_id=request_id,
            session_id=session_id,
            status=status,
            final_answer="",
            citations=[],
            steps=steps,
            usage=usage,
            route=route,
            warnings=warnings,
            trace_id=trace_id,
            answer_provider="local",
            answer_model="local-fallback",
        )

    def _compact_if_needed(
        self, session_id: str, constraints: AgentConstraints
    ) -> None:
        self._memory.compact_history(
            session_id, max_turns=settings.agent_max_conversation_turns
        )

    def _run_retrieve_evidence(
        self,
        project_slug: str,
        question: str,
        document_id: str | None,
        constraints: AgentConstraints,
        steps: list[AgentStep],
        usage: AgentUsage,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Call rag.retrieve_evidence, record step, return evidence pack dict.

        Returns ``None`` when the tool is not registered or limits are hit.
        The returned dict has keys: ``status``, ``items`` (list of dicts).
        """
        # Skip if tool is not registered (backward compatibility)
        try:
            self._tools.get("rag.retrieve_evidence")
        except Exception:
            return None

        # Enforce limits
        if usage.tool_calls >= constraints.max_tool_calls:
            return None
        if len(steps) >= constraints.max_steps:
            return None

        step_id = len(steps)
        t0 = time.monotonic()
        tool_args: dict[str, Any] = {
            "project_slug": project_slug,
            "question": question,
            "limit": 15,
        }
        if document_id is not None:
            tool_args["document_id"] = document_id
        tool_result = self._tools.call_tool(
            "rag.retrieve_evidence",
            tool_args,
            ctx={"db": self._db},
        )
        latency = int((time.monotonic() - t0) * 1000)
        usage.tool_calls += 1

        if tool_result["ok"]:
            rdata: dict[str, Any] = tool_result.get("result", {})
            pack_status = rdata.get("status", "empty")
            items: list[dict[str, Any]] = rdata.get("items", [])
            evidence_count = len(items)
            table_evidence_count = sum(
                1 for item in items if item.get("evidence_kind") == "table"
            )
            evidence_kinds = sorted(
                {item.get("evidence_kind") for item in items if item.get("evidence_kind")}
            )
            source_stages = sorted(
                {item.get("source_stage", "unknown") for item in items}
            )
            support_hints = sorted(
                {item.get("support_hint", "weak") for item in items}
            )

            step = AgentStep(
                step_id=step_id,
                step_type="retrieve",
                summary=(
                    f"rag.retrieve_evidence returned {evidence_count} items "
                    f"(status={pack_status}), {table_evidence_count} table evidence"
                ),
                latency_ms=latency,
                tool_name="rag.retrieve_evidence",
                tool_ok=True,
                metadata={
                    "evidence_count": evidence_count,
                    "table_evidence_count": table_evidence_count,
                    "evidence_kinds": evidence_kinds,
                    "source_stages": source_stages,
                    "support_hints": support_hints,
                    "status": pack_status,
                    "document_id": document_id,
                },
            )
            steps.append(step)
            self._memory.add_turn(
                session_id,
                role="tool",
                content=f"Retrieved {evidence_count} evidence items",
                tool_name="rag.retrieve_evidence",
                tool_args=tool_args,
                tool_result=str(evidence_count),
                step_type="retrieve",
            )
            return rdata

        # Tool call failed
        step = AgentStep(
            step_id=step_id,
            step_type="retrieve",
            summary=(
                f"rag.retrieve_evidence failed: "
                f"{tool_result.get('error', 'unknown error')}"
            ),
            latency_ms=latency,
            tool_name="rag.retrieve_evidence",
            tool_ok=False,
            metadata={"error": tool_result.get("error", "")},
        )
        steps.append(step)
        return None

    def _run_rag_answer(
        self,
        project_slug: str,
        question: str,
        document_id: str | None,
        constraints: AgentConstraints,
        steps: list[AgentStep],
        usage: AgentUsage,
        session_id: str,
    ) -> tuple[str, list[Citation], int]:
        """Call rag.answer, record step, return (answer_text, citations, tool_calls_used).

        Returns (\"\", [], 0) when ``max_tool_calls`` or ``max_steps``
        would be violated.
        """
        # Enforce limits
        if usage.tool_calls >= constraints.max_tool_calls:
            return "", [], 0
        if len(steps) >= constraints.max_steps:
            return "", [], 0

        step_id = len(steps)
        t0 = time.monotonic()
        tool_args: dict[str, Any] = {
            "project_slug": project_slug,
            "question": question,
        }
        if document_id is not None:
            tool_args["document_id"] = document_id
        tool_result = self._tools.call_tool(
            "rag.answer",
            tool_args,
            ctx={"db": self._db},
        )
        latency = int((time.monotonic() - t0) * 1000)
        usage.tool_calls += 1

        if tool_result["ok"]:
            rag_data: dict[str, Any] = tool_result.get("result", {})
            answer_text = rag_data.get("answer_markdown", "")
            raw_citations = rag_data.get("citations", [])
            citations = [
                Citation(
                    document_id=c.get("document_id"),
                    chunk_id=c.get("chunk_id"),
                    page_slug=c.get("page_slug"),
                    page_title=c.get("page_title"),
                    page_kind=c.get("page_kind"),
                    score=float(c.get("score", 0)),
                    page_label=c.get("page_label"),
                    excerpt=str(c.get("excerpt", "")),
                )
                for c in raw_citations
            ]
            step = AgentStep(
                step_id=step_id,
                step_type="tool_call",
                summary=(
                    f"rag.answer returned {len(citations)} citation(s), "
                    f"{len(answer_text)} chars answer"
                ),
                latency_ms=latency,
                tool_name="rag.answer",
                tool_ok=True,
                metadata={"answer_chars": len(answer_text), "document_id": document_id},
            )
        else:
            answer_text = ""
            citations = []
            step = AgentStep(
                step_id=step_id,
                step_type="tool_call",
                summary=(
                    f"rag.answer failed: {tool_result.get('error', 'unknown error')}"
                ),
                latency_ms=latency,
                tool_name="rag.answer",
                tool_ok=False,
                metadata={"error": tool_result.get("error", "")},
            )

        steps.append(step)
        self._memory.add_turn(
            session_id,
            role="tool",
            content=answer_text[:2000],
            tool_name="rag.answer",
            tool_args=tool_args,
            tool_result=answer_text[:4000],
            step_type="tool_call",
        )

        return answer_text, citations, 1

    def _run_verify(
        self,
        question: str,
        answer_text: str,
        citations: list[Citation],
        route_type: str,
        constraints: AgentConstraints,
        steps: list[AgentStep],
        usage: AgentUsage,
        tool_calls_so_far: int,
    ) -> dict[str, Any]:
        """Call answer.verify, record step, return tool result dict.

        Returns a no-op dict when ``max_tool_calls`` or ``max_steps``
        would be violated.  Only increments ``usage.tool_calls`` when a
        step is actually appended.
        """
        # Enforce limits
        if usage.tool_calls >= constraints.max_tool_calls:
            return {
                "ok": True,
                "name": "answer.verify",
                "result": {
                    "ok": True,
                    "warnings": ["Skipped verification — tool call limit reached"],
                    "retry_recommended": False,
                    "reason": "Tool call limit",
                },
                "latency_ms": 0,
            }
        if len(steps) >= constraints.max_steps:
            return {
                "ok": True,
                "name": "answer.verify",
                "result": {
                    "ok": True,
                    "warnings": ["Skipped verification — step limit reached"],
                    "retry_recommended": False,
                    "reason": "Step limit",
                },
                "latency_ms": 0,
            }

        step_id = len(steps)
        t0 = time.monotonic()
        raw_citations = [
            {
                "document_id": c.document_id,
                "chunk_id": c.chunk_id,
                "page_slug": c.page_slug,
                "page_title": c.page_title,
                "page_kind": c.page_kind,
                "score": c.score,
                "page_label": c.page_label,
                "excerpt": c.excerpt,
            }
            for c in citations
        ]
        verify_result = self._tools.call_tool(
            "answer.verify",
            {
                "question": question,
                "answer_markdown": answer_text,
                "citations": raw_citations,
                "route_type": route_type,
            },
        )
        latency = int((time.monotonic() - t0) * 1000)
        usage.tool_calls += 1

        if verify_result["ok"]:
            vdata = verify_result.get("result", {})
            step = AgentStep(
                step_id=step_id,
                step_type="tool_call",
                summary=(
                    f"answer.verify: ok={vdata.get('ok')}, "
                    f"retry={vdata.get('retry_recommended')}, "
                    f"warnings={len(vdata.get('warnings', []))}"
                ),
                latency_ms=latency,
                tool_name="answer.verify",
                tool_ok=True,
                metadata={
                    "verify_ok": vdata.get("ok"),
                    "retry_recommended": vdata.get("retry_recommended"),
                    "verify_warnings": vdata.get("warnings", []),
                },
            )
        else:
            step = AgentStep(
                step_id=step_id,
                step_type="tool_call",
                summary=f"answer.verify failed: {verify_result.get('error', 'unknown')}",
                latency_ms=latency,
                tool_name="answer.verify",
                tool_ok=False,
                metadata={"error": verify_result.get("error", "")},
            )

        steps.append(step)
        return verify_result

    def _run_synthesize(
        self,
        query: str,
        route: str,
        answer_text: str,
        citations: list[Citation],
        constraints: AgentConstraints,
        steps: list[AgentStep],
        usage: AgentUsage,
        session_id: str,
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call answer.synthesize, record step, return tool result dict.

        Returns a no-op dict when ``max_tool_calls`` or ``max_steps``
        would be violated.  When *evidence_pack* is provided, it is
        passed through to the synthesis tool and compact evidence-aware
        metadata is recorded on the step.
        """
        # Enforce limits
        if usage.tool_calls >= constraints.max_tool_calls:
            return {
                "ok": True,
                "name": "answer.synthesize",
                "result": {
                    "answer_markdown": answer_text,
                    "cited_indexes": list(range(len(citations))),
                    "warnings": ["Skipped synthesis — tool call limit reached"],
                    "confidence": 1.0,
                    "provider": "local",
                    "model": "local-fallback",
                },
            }
        if len(steps) >= constraints.max_steps:
            return {
                "ok": True,
                "name": "answer.synthesize",
                "result": {
                    "answer_markdown": answer_text,
                    "cited_indexes": list(range(len(citations))),
                    "warnings": ["Skipped synthesis — step limit reached"],
                    "confidence": 1.0,
                    "provider": "local",
                    "model": "local-fallback",
                },
            }

        step_id = len(steps)
        t0 = time.monotonic()
        raw_citations = [
            {
                "document_id": c.document_id,
                "chunk_id": c.chunk_id,
                "page_slug": c.page_slug,
                "page_title": c.page_title,
                "page_kind": c.page_kind,
                "score": c.score,
                "page_label": c.page_label,
                "excerpt": c.excerpt,
            }
            for c in citations
        ]

        # Build conversation summary from memory
        conv_summary = self._build_conversation_summary(session_id)

        # Build tool args, including evidence_pack only when present
        tool_args: dict[str, Any] = {
            "query": query,
            "route": route,
            "conversation_summary": conv_summary,
            "rag_answer": answer_text,
            "citations": raw_citations,
        }
        if evidence_pack is not None:
            tool_args["evidence_pack"] = evidence_pack

        synth_result = self._tools.call_tool(
            "answer.synthesize",
            tool_args,
        )
        latency = int((time.monotonic() - t0) * 1000)
        usage.tool_calls += 1

        if synth_result["ok"]:
            sdata = synth_result.get("result", {})
            # Build compact evidence-aware metadata for the synthesis step
            synth_meta: dict[str, Any] = {
                "provider": sdata.get("provider"),
                "model": sdata.get("model"),
                "confidence": sdata.get("confidence", 1.0),
                "cited_indexes": sdata.get("cited_indexes", []),
            }
            if evidence_pack and evidence_pack.get("items"):
                items = evidence_pack["items"]
                synth_meta["evidence_pack_items"] = len(items)
                synth_meta["table_evidence_count"] = sum(
                    1 for item in items if item.get("evidence_kind") == "table"
                )
                synth_meta["source_stages"] = sorted(
                    {item.get("source_stage", "unknown") for item in items}
                )
            # Flag insufficient evidence so the trace is auditable
            if _INSUFFICIENT_EVIDENCE_RE.search(answer_text):
                synth_meta["evidence_insufficient"] = True
            step = AgentStep(
                step_id=step_id,
                step_type="synthesis",
                summary=(
                    f"answer.synthesize: provider={sdata.get('provider')}, "
                    f"confidence={sdata.get('confidence', 1.0)}"
                ),
                latency_ms=latency,
                tool_name="answer.synthesize",
                tool_ok=True,
                metadata=synth_meta,
            )
        else:
            step = AgentStep(
                step_id=step_id,
                step_type="synthesis",
                summary=f"answer.synthesize failed: {synth_result.get('error', 'unknown')}",
                latency_ms=latency,
                tool_name="answer.synthesize",
                tool_ok=False,
                metadata={"error": synth_result.get("error", "")},
            )

        steps.append(step)
        return synth_result

    def _build_conversation_summary(self, session_id: str) -> str:
        """Build a brief text summary of recent conversation turns."""
        try:
            history = self._memory.get_history(session_id, last_n=10)
            if not history:
                return ""
            parts = []
            for turn in history:
                if turn.step_type == "user_query" or turn.role == "user":
                    parts.append(f"User: {turn.content[:200]}")
                elif turn.step_type == "finalize" or turn.role == "agent":
                    parts.append(f"Agent: {turn.content[:200]}")
            return "\n".join(parts[-6:])  # last 6 turns max
        except Exception:
            return ""

    def _run_retrieve_session_attachments(
        self,
        project_slug: str,
        question: str,
        constraints: AgentConstraints,
        steps: list[AgentStep],
        usage: AgentUsage,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve evidence from session-scoped temporary attachments.

        Returns a dict evidence pack or None when limits are hit. Adds a
        visible ``retrieve`` step so the trace shows the attachment lookup.
        """
        if usage.tool_calls >= constraints.max_tool_calls:
            return None
        if len(steps) >= constraints.max_steps:
            return None

        t0 = time.monotonic()
        pack = retrieve_session_attachment_evidence(
            self._db, project_slug, session_id, question, limit=5
        )
        latency = int((time.monotonic() - t0) * 1000)

        item_count = len(pack.items)
        if item_count == 0:
            # No attachments for this session; do not consume a step or tool call.
            return None

        usage.tool_calls += 1
        step_id = len(steps)
        step = AgentStep(
            step_id=step_id,
            step_type="retrieve",
            summary=(
                f"session_attachment.retrieve returned {item_count} item(s) "
                f"for session {session_id}"
            ),
            latency_ms=latency,
            tool_name="session_attachment.retrieve",
            tool_ok=pack.status != "project_not_found",
            metadata={
                "status": pack.status,
                "evidence_count": item_count,
                "source_stages": ["session_attachment"],
                "evidence_kinds": ["session_attachment"],
            },
        )
        steps.append(step)
        self._memory.add_turn(
            session_id,
            role="tool",
            content=f"Retrieved {item_count} session attachment evidence item(s)",
            tool_name="session_attachment.retrieve",
            tool_args={"project_slug": project_slug, "question": question},
            tool_result=str(item_count),
            step_type="retrieve",
        )
        return pack.model_dump()

    def _merge_evidence_pack(
        self,
        base_pack: dict[str, Any] | None,
        session_pack: dict[str, Any] | None,
        max_items: int = 15,
    ) -> dict[str, Any]:
        """Merge session attachment evidence into the base evidence pack.

        Preserves base status semantics while appending attachment items and
        renumbering indexes deterministically.
        """
        base = dict(base_pack) if base_pack else {"status": "empty", "items": []}
        items = list(base.get("items", []))
        if session_pack and session_pack.get("items"):
            items.extend(session_pack["items"])
        # Renumber indexes deterministically.
        for idx, item in enumerate(items, start=1):
            item["index"] = idx
        # Cap total items to avoid overwhelming the synthesizer.
        if len(items) > max_items:
            items = items[:max_items]
        # If either side found evidence, status is ok.
        status = base.get("status", "empty")
        if status in ("empty", None) and session_pack and session_pack.get("status") == "ok":
            status = "ok"
        if status == "project_not_found" and items:
            status = "ok"
        return {"status": status, "items": items}

    @staticmethod
    def _session_attachment_citations(
        evidence_pack: dict[str, Any] | None,
    ) -> list[Citation]:
        """Convert session attachment evidence items into normal citations."""
        if not evidence_pack or not evidence_pack.get("items"):
            return []
        citations: list[Citation] = []
        for item in evidence_pack["items"]:
            if item.get("source_stage") != "session_attachment":
                continue
            excerpt = str(item.get("excerpt") or "").strip()
            if not excerpt:
                continue
            citations.append(
                Citation(
                    document_id=item.get("document_id"),
                    chunk_id=item.get("chunk_id"),
                    attachment_id=item.get("attachment_id"),
                    page_slug=item.get("page_slug"),
                    page_title=item.get("page_title"),
                    page_kind=item.get("page_kind") or "session_attachment",
                    score=float(item.get("score") or 0.0),
                    page_label=item.get("page_label"),
                    excerpt=excerpt,
                )
            )
        return citations

    @staticmethod
    def _merge_citations(
        base_citations: list[Citation],
        attachment_citations: list[Citation],
    ) -> list[Citation]:
        """Append attachment citations without duplicating identical chunks."""
        merged = list(base_citations)
        seen = {
            (
                c.page_kind,
                c.document_id,
                c.chunk_id,
                c.page_title,
                c.excerpt,
            )
            for c in merged
        }
        for citation in attachment_citations:
            key = (
                citation.page_kind,
                citation.document_id,
                citation.chunk_id,
                citation.page_title,
                citation.excerpt,
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(citation)
        return merged

    @staticmethod
    def _draft_session_attachment_answer(
        question: str,
        citations: list[Citation],
        max_items: int = 3,
    ) -> str:
        """Build a deterministic extractive answer from session-only files."""
        if not citations:
            return ""
        lines = [
            "根据当前对话临时文件，我先从原文中定位到这些相关内容：",
            "",
        ]
        for idx, citation in enumerate(citations[:max_items], start=1):
            title = citation.page_title or citation.document_id or "临时文件"
            excerpt = " ".join(citation.excerpt.split())
            if len(excerpt) > 700:
                excerpt = excerpt[:700].rstrip() + "..."
            lines.append(f"{idx}. **{title}**：{excerpt}")
        if len(citations) > max_items:
            lines.append(f"\n还有 {len(citations) - max_items} 段相关原文未展开。")
        lines.append(
            "\n以上只使用当前对话的临时附件，不会引用其他专题或会话里的文档。"
        )
        return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)
