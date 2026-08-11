from __future__ import annotations

"""
Agent Executor：单轮 Agent 查询的编排器。

整个执行流程是确定性的、有界的循环，不是自由 LLM ReAct：
1. route      — 用 PolicyRouter 分类查询意图
2. plan       — 仅 complex_multi_hop 路由生成可审计计划
3. retrieve   — 调用 rag.retrieve_evidence 获取证据包
4. rag.answer — 调用 RAG 生成草稿答案
5. synthesize — 调用 answer.synthesize 综合终稿
6. verify     — 调用 answer.verify 质量检查
7. retry      — 若验证建议重试且路由允许，最多再调一次 rag.answer
8. finalize   — 组装 AgentQueryResponse，持久化 trace

关键约束：
- max_steps：严格限制 trace 步数；
- max_tool_calls：严格限制工具调用次数；
- timeout_seconds：总超时限制；
- needs_clarification 路由直接返回，不走 RAG。

所有工具调用都通过 ToolRegistry，AgentExecutor 本身不直接访问 RAG 内部。
"""

import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.records import DocumentChunk
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
from app.services.agent_model_router import AgentModelRouter, InferenceTarget
from app.services.agent_synthesizer import AgentSynthesizer
from app.services.agent_trace_store import AgentTraceStore
from app.services.conversation_memory import ConversationMemory
from app.services.model_runtime import ModelRuntime, get_model_runtime
from app.services.rag_adapter import RAGAdapter, _INSUFFICIENT_EVIDENCE_RE
from app.services.session_attachments import retrieve_session_attachment_evidence
from app.services.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)
settings = get_settings()


class _EventStepList(list[AgentStep]):
    """带事件转发的 AgentStep 列表。

    每次 append 一个 step 时，自动通过 event_sink 推送 "route" 或 "step" 事件，
    供前端/日志实时展示执行进度。
    """

    def __init__(self, event_sink: Callable[[str, dict[str, Any]], None] | None):
        super().__init__()
        self._event_sink = event_sink

    def append(self, step: AgentStep) -> None:
        super().append(step)
        if self._event_sink is not None:
            event_name = "route" if step.step_type == "route" else "step"
            self._event_sink(event_name, step.model_dump())


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
        model_runtime: ModelRuntime | None = None,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self._rag = rag
        self._tools = tools
        self._memory = memory
        self._db = db
        self._trace_store = trace_store
        self._synthesizer = synthesizer
        self._model_runtime = model_runtime or get_model_runtime()
        self._event_sink = event_sink
        self._cancel_event = cancel_event

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def execute(self, request: AgentQueryRequest) -> AgentQueryResponse:
        """执行单轮 Agent 查询并返回结构化响应。

        流程中的所有步骤都受到 constraints.max_steps 限制；
        当 len(steps) >= max_steps 时，不再追加新 step，直接截断返回。
        """
        request_id = str(uuid4())
        session_id = request.session_id or f"sess_{uuid4().hex[:12]}"
        constraints = request.constraints
        steps: list[AgentStep] = _EventStepList(self._event_sink)
        usage = AgentUsage()
        warnings: list[str] = []
        t_start = time.monotonic()
        max_steps_hit = False

        # ---- 会话维护：更新 TTL、清理过期会话 ----
        self._memory.touch_session(
            session_id,
            project_slug=request.project_slug,
            ttl_days=settings.agent_conversation_ttl_days,
            document_id=request.document_id,
        )
        self._memory.purge_expired_sessions()

        # ---- 记录用户本轮输入 ----
        self._memory.add_turn(
            session_id, role="user", content=request.query, step_type="user_query"
        )
        self._compact_if_needed(session_id, constraints)
        # 在长时间检索和模型调用前提交会话写入，释放 SQLite 写锁。
        self._commit_progress()

        try:
            # ==============================================================
            # Step 0: 路由 — 决定查询类型和后续策略
            # ==============================================================
            route_t0 = time.monotonic()
            route = PolicyRouter().route(request.query)
            inference_target = AgentModelRouter().select(route.route)
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
                    "inference_profile": inference_target.profile,
                    "inference_model": inference_target.model,
                    "inference_context_length": inference_target.context_length,
                    "inference_reason": inference_target.reason,
                },
            )
            steps.append(route_step)
            max_steps_hit = len(steps) >= constraints.max_steps

            # ==============================================================
            # Step 0.5: 计划 — 仅 complex_multi_hop 路由生成受控计划
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

            # ---- needs_clarification：无需 RAG，直接返回澄清提示 ----
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
            # 常规路由：检索 → RAG 草稿 → 综合 → 验证 →（可选）重试
            # ==============================================================
            # 如果 step budget 已耗尽，直接截断返回
            if max_steps_hit:
                status = self._finalize_truncated(
                    steps, usage, warnings, route, constraints,
                    request_id, session_id, "", [], t_start, request,
                )
                return status

            attachment_only_requested = self._is_attachment_only_query(request.query)
            attachment_only = False
            session_attachment_pack = None
            # 预置：attachment_only 分支不定义 `_evidence_insufficient` 与
            # `rag_verification_status`，而 skip_synthesis 会引用它们；预置
            # 避免依赖合取短路顺序。
            _evidence_insufficient = False
            rag_verification_status = "local-only"
            retrieval_query = self._contextualize_retrieval_query(
                session_id, request.query
            )

            if attachment_only_requested:
                session_attachment_pack = self._run_retrieve_session_attachments(
                    request.project_slug,
                    request.query,
                    constraints,
                    steps,
                    usage,
                    session_id,
                )
                attachment_only = bool(
                    session_attachment_pack and session_attachment_pack.get("items")
                )

            if attachment_only:
                evidence_pack = self._merge_evidence_pack(
                    None, session_attachment_pack, max_items=15
                )
                citations = self._session_attachment_citations(evidence_pack)
                answer_text = self._draft_session_attachment_answer(
                    request.query, citations
                )
                tool_calls = usage.tool_calls
                warnings.append(
                    "Answered only from temporary attachments scoped to this session."
                )
            else:
                # ---- 从项目知识库检索证据（v4） ----
                evidence_pack = self._run_retrieve_evidence(
                    request.project_slug,
                    retrieval_query,
                    request.document_id,
                    constraints,
                    steps,
                    usage,
                    session_id,
                )

                # T1：项目级会话（无文档锁定）检索 0 items 时，回退到会话
                # 历史最近成功检索命中的文档重新检索——弱指代查询（"展开
                # 第二张"等）与论文 profile 词元零交集导致全项目路由 miss，
                # 历史命中文档是当前主题最可能的归属。
                effective_document_id = request.document_id
                if (
                    (not evidence_pack or not evidence_pack.get("items"))
                    and request.document_id is None
                ):
                    history_document_id = self._last_hit_document_id(session_id)
                    if history_document_id is not None:
                        fallback_pack = self._run_retrieve_evidence(
                            request.project_slug,
                            retrieval_query,
                            history_document_id,
                            constraints,
                            steps,
                            usage,
                            session_id,
                        )
                        if fallback_pack and fallback_pack.get("items"):
                            evidence_pack = fallback_pack
                            effective_document_id = history_document_id

                # ---- 合并当前会话的临时附件证据 ----
                if session_attachment_pack is None:
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
                # ---- 调用 RAG 生成草稿答案 ----
                (
                    answer_text,
                    citations,
                    tool_calls_used,
                    rag_verification_status,
                ) = self._run_rag_answer(
                    request.project_slug,
                    retrieval_query,
                    effective_document_id,
                    constraints,
                    steps,
                    usage,
                    session_id,
                )
                tool_calls = tool_calls_used
                attachment_citations = self._session_attachment_citations(evidence_pack)
                if attachment_citations:
                    citations = self._merge_citations(citations, attachment_citations)

                # ``rag.answer`` intentionally returns a compact citation list
                # for ordinary answers.  Table retrieval, however, assembles
                # complete same-table evidence in the evidence pack and the
                # synthesis pass must be able to see every table Child (not
                # only the five citations selected by the draft answer).
                citations = self._merge_table_evidence_citations(
                    citations,
                    evidence_pack,
                    extra_citations=self._table_fact_source_citations(evidence_pack),
                )

                # ---- 检测证据不足 ----
                _evidence_insufficient = bool(
                    answer_text and _INSUFFICIENT_EVIDENCE_RE.search(answer_text)
                )
                if attachment_citations and (
                    _evidence_insufficient or not answer_text.strip()
                ):
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

            # ==============================================================
            # Task 2：draft verify 前置 + 直通门禁 + synthesize/final verify
            # ==============================================================
            # 9.1 路由决策矩阵：单轮/对比路由（simple_rag / evidence_required /
            # table_or_metric / multi_source_compare）且非跨轮引用时，先执行
            # draft verify（确定性校验，非第二次 LLM 综合）；通过才直通
            # rag.answer，关闭"LLM 无差别重写压缩精确事实"的破坏路径。
            # 复杂多跳与跨轮引用保留 synthesize（受约束组织器）+ final verify。
            _direct_routes = {
                "simple_rag",
                "evidence_required",
                "table_or_metric",
                "multi_source_compare",
            }
            direct_candidate = (
                not attachment_only
                and not max_steps_hit
                and route.route in _direct_routes
                and not self._is_cross_turn_query(request.query, session_id)
            )
            direct_block_reason: str | None = None
            draft_verification = False
            final_verification = False
            draft_verify_result: dict[str, Any] | None = None

            # ---- 前置 draft verify（仅 direct candidate）----
            if direct_candidate:
                draft_verification = True
                draft_verify_result = self._run_verify(
                    request.query,
                    answer_text,
                    citations,
                    route.route,
                    constraints,
                    steps,
                    usage,
                    tool_calls,
                )
                tool_calls = usage.tool_calls
                draft_retry = bool(
                    draft_verify_result["ok"]
                    and draft_verify_result.get("result", {}).get(
                        "retry_recommended", False
                    )
                )
                # draft verify 建议 retry 且路由允许时，RAG retry 后重验
                if (
                    draft_retry
                    and route.max_retries > 0
                    and tool_calls < constraints.max_tool_calls
                    and len(steps) < constraints.max_steps
                ):
                    (
                        retry_answer,
                        retry_citations,
                        retry_calls,
                        _retry_verification_status,
                    ) = self._run_rag_answer(
                        request.project_slug,
                        retrieval_query,
                        effective_document_id,
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
                            # Task 9：draft retry 采用新答案时，verification status
                            # 与 evidence-insufficient 必须同步为最终答案的状态，
                            # 直通门禁与 finalize trace 不得沿用第一次答案的状态。
                            rag_verification_status = _retry_verification_status
                            _evidence_insufficient = bool(
                                answer_text
                                and _INSUFFICIENT_EVIDENCE_RE.search(answer_text)
                            )
                        warnings.append(
                            "Retry performed after draft verification warning"
                        )
                        # retry 后的结果重新执行 draft verify，不能沿用旧结果
                        draft_verify_result = self._run_verify(
                            request.query,
                            answer_text,
                            citations,
                            route.route,
                            constraints,
                            steps,
                            usage,
                            tool_calls,
                        )
                        tool_calls = usage.tool_calls
                        draft_retry = bool(
                            draft_verify_result["ok"]
                            and draft_verify_result.get("result", {}).get(
                                "retry_recommended", False
                            )
                        )

            # ---- 直通门禁（Task 2：draft verify 通过 + RAG 校验；Task 3：coverage）----
            # 9.1.6 coverage 门禁生效范围：仅 table_or_metric 且 RAG 明确声明
            # coverage_status == "partial" 时阻塞直通。partial 即代表表格覆盖不
            # 完整——即使 table_facts 为空，也必须阻止 rag-direct，不能把未覆盖
            # 的表格答案标记为已直通；unknown（旧 EvidencePack 缺 inventory
            # 字段）与非表格路由一律中立，不得因缺字段或空 facts 让已通过的直通退化。
            coverage_partial = bool(
                route.route == "table_or_metric"
                and evidence_pack is not None
                and evidence_pack.get("coverage_status") == "partial"
            )
            skip_synthesis = bool(
                direct_candidate
                and not _evidence_insufficient
                and bool(answer_text.strip())
                and bool(citations)
                and rag_verification_status != "contradicted"
                and not draft_retry
                and not coverage_partial
            )
            if direct_candidate and not skip_synthesis:
                if _evidence_insufficient:
                    direct_block_reason = "evidence_insufficient"
                elif not answer_text.strip():
                    direct_block_reason = "empty_answer"
                elif not citations:
                    direct_block_reason = "no_citations"
                elif rag_verification_status == "contradicted":
                    direct_block_reason = "rag_contradicted"
                elif draft_retry:
                    direct_block_reason = "draft_verify_retry"
                elif coverage_partial:
                    direct_block_reason = "coverage_partial"

            synth_provider = "local"
            synth_model = "rag-direct" if skip_synthesis else "local-fallback"
            # 9.8：需求侧期望 facts 状态（synthesize 未调用时为 None）
            expected_facts_status: str | None = None
            if not max_steps_hit and not attachment_only and not skip_synthesis:
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
                    target=inference_target,
                )
                if synth_result.get("ok"):
                    synth_data = synth_result.get("result", {})
                    answer_text = synth_data.get("answer_markdown", answer_text)
                    synth_provider = synth_data.get("provider", "local")
                    synth_model = synth_data.get("model", "local-fallback")
                    expected_facts_status = synth_data.get("expected_facts_status")
                    synth_warnings = synth_data.get("warnings", [])
                    if isinstance(synth_warnings, list):
                        warnings.extend(synth_warnings)
                    # 根据综合结果过滤引用：只保留被明确引用的 citation
                    cited_indexes = synth_data.get("cited_indexes", [])
                    if isinstance(cited_indexes, list) and cited_indexes:
                        selected_indexes = sorted(
                            {
                                index
                                for index in cited_indexes
                                if isinstance(index, int) and 0 <= index < len(citations)
                            }
                        )
                        if selected_indexes:
                            answer_text = self._retarget_citation_markers(
                                answer_text,
                                selected_indexes,
                            )
                            selected_set = set(selected_indexes)
                            citations = [
                                citation
                                for index, citation in enumerate(citations)
                                if index in selected_set
                            ]

            # ---- final verify：直通路径 draft verify 即最终校验（不重复消耗）；
            #      synthesize 路径在综合后执行 final verify ----
            if skip_synthesis:
                verify_result = draft_verify_result
            else:
                final_verification = True
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
                tool_calls = usage.tool_calls
                if verify_result["ok"]:
                    verify_warnings = verify_result.get("result", {}).get(
                        "warnings", []
                    )
                    if isinstance(verify_warnings, list):
                        warnings.extend(verify_warnings)

            # ---- 重试：仅 synthesize 路径的 final verify 建议重试时 ----
            retry_recommended = (
                not skip_synthesis
                and verify_result["ok"]
                and verify_result.get("result", {}).get("retry_recommended", False)
            )
            # 空答案兜底（T3）：模型生成空/纯空白答案是不可交付的硬失败，
            # 无视路由 max_retries 配额强制重试一次（二次仍空才交付）。
            answer_is_empty = not (answer_text or "").strip()
            if (
                not attachment_only
                and (
                    (retry_recommended and route.max_retries > 0)
                    or answer_is_empty
                )
                and tool_calls < constraints.max_tool_calls
                and len(steps) < constraints.max_steps
            ):
                (
                    retry_answer,
                    retry_citations,
                    retry_calls,
                    _retry_verification_status,
                ) = self._run_rag_answer(
                    request.project_slug,
                    retrieval_query,
                    effective_document_id,
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
                        # Task 9：最终答案来自 retry，verification status 必须
                        # 同步为 retry 返回的状态，finalize trace 才能与最终答案一致。
                        rag_verification_status = _retry_verification_status
                    warnings.append("Retry performed after verification warning")
                    # Task 9：retry 后必须对最终答案与 citations 重新执行 final
                    # verify；retry 前的第一次 verify 结果不能作为最终校验依据。
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
                    tool_calls = usage.tool_calls
                    if verify_result["ok"]:
                        verify_warnings = verify_result.get("result", {}).get(
                            "warnings", []
                        )
                        if isinstance(verify_warnings, list):
                            warnings.extend(verify_warnings)
                    # 重验后若仍建议 retry，不再发起第二次重试（retry 只执行
                    # 一次），但 retry_recommended 保留重验结果，finalize trace
                    # 不得宣称最终答案已验证通过。
                    retry_recommended = (
                        verify_result["ok"]
                        and verify_result.get("result", {}).get(
                            "retry_recommended", False
                        )
                    )


            # ---- 检查 max_steps 是否耗尽 ----
            hit_limit = len(steps) >= constraints.max_steps
            if hit_limit:
                warnings.append(
                    f"Reached max_steps limit ({constraints.max_steps}). "
                    "Trace truncated."
                )

            # ==============================================================
            # Finalize：组装最终响应
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
                    metadata={
                        "route": route.route,
                        "source_scope": (
                            "session_attachments_only"
                            if attachment_only
                            else "project_and_session"
                        ),
                        "project_rag_skipped": attachment_only,
                        "synthesis_skipped": skip_synthesis,
                        # 9.6.3：trace 完整记录校验与门禁结果
                        "draft_verification": draft_verification,
                        "final_verification": final_verification,
                        "direct_block_reason": direct_block_reason,
                        "rag_verification_status": rag_verification_status,
                        "coverage_status": (
                            evidence_pack.get("coverage_status", "unknown")
                            if evidence_pack
                            else "unknown"
                        ),
                        "expected_facts_status": expected_facts_status,
                    },
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
                    raw_citations_for_trace = [c.model_dump() for c in citations]
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
            # ---- 执行异常：记录错误 trace 并返回 error 状态 ----
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
    # 辅助方法
    # ------------------------------------------------------------------

    def _commit_progress(self) -> None:
        """在可能耗时较长的步骤前提交数据库事务，释放写锁。"""
        try:
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    @staticmethod
    def _is_attachment_only_query(query: str) -> bool:
        """判断用户是否明确要求只使用当前会话的临时附件作答。"""
        normalized = " ".join(query.lower().split())
        has_attachment = any(
            term in normalized
            for term in ("附件", "临时文件", "attachment", "attached file")
        )
        has_exclusive_scope = any(
            term in normalized
            for term in (
                "只根据",
                "仅根据",
                "只使用",
                "仅使用",
                "不要引用项目",
                "only the attachment",
                "only attachment",
                "using only",
            )
        )
        is_comparison = any(
            term in normalized for term in ("比较", "对比", "compare")
        )
        return has_attachment and has_exclusive_scope and not is_comparison

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
        """当 step budget 在早期就耗尽时，返回截断的响应。"""
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

    def _last_hit_document_id(self, session_id: str) -> str | None:
        """返回会话历史最近一次成功检索命中的文档 id（T1 回退锚定）。

        从后往前扫描 retrieve 轮次，取 citations 中含 document_id 的
        最近一轮（当前轮 0 items 时 citations 为空，自然被跳过）。
        """
        history = self._memory.get_history(session_id)
        for turn in reversed(history):
            if turn.step_type != "retrieve":
                continue
            for cit in turn.citations:
                doc = str(cit.get("document_id") or "").strip()
                if doc:
                    return doc
        return None

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
        """调用 rag.retrieve_evidence 工具，记录 step，返回 evidence pack。

        如果工具未注册、或已达到 max_tool_calls / max_steps 限制，返回 None。
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
            # T1：检索命中的文档 id 持久化到 turn 的 citations，供后续轮次
            # 检索空结果时回退锚定（_history_retrieve_document）。
            hit_docs: list[dict] = []
            if items:
                seen: set[str] = set()
                for item in items:
                    doc = str(item.get("document_id") or "").strip()
                    if doc and doc not in seen:
                        seen.add(doc)
                        hit_docs.append({"document_id": doc})
            self._memory.add_turn(
                session_id,
                role="tool",
                content=f"Retrieved {evidence_count} evidence items",
                tool_name="rag.retrieve_evidence",
                tool_args=tool_args,
                tool_result=str(evidence_count),
                step_type="retrieve",
                citations=hit_docs,
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
    ) -> tuple[str, list[Citation], int, str]:
        """调用 rag.answer 工具，记录 step，返回
        (答案文本, 引用列表, 实际工具调用数, verification_status)。

        如果受 max_tool_calls / max_steps 限制，返回 ("", [], 0, "local-only")。
        """
        # Enforce limits
        if usage.tool_calls >= constraints.max_tool_calls:
            return "", [], 0, "local-only"
        if len(steps) >= constraints.max_steps:
            return "", [], 0, "local-only"

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
                    attachment_id=c.get("attachment_id"),
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

        verification_status = str(
            rag_data.get("verification_status", "local-only")
        ) if tool_result["ok"] else "local-only"
        return answer_text, citations, 1, verification_status

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
        """调用 answer.verify 工具，记录 step，返回工具结果。

        如果受限制，返回 no-op 结果（不触发重试）。
        验证工具会返回 retry_recommended，决定是否进入后续重试逻辑。
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
        raw_citations = [c.model_dump() for c in citations]
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
        target: InferenceTarget | None = None,
    ) -> dict[str, Any]:
        """调用 answer.synthesize 工具，记录 step，返回工具结果。

        如果受 max_tool_calls / max_steps 限制，返回 no-op 结果（保持原答案）。
        会把 evidence_pack 和 conversation_summary 传给综合工具。
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
        raw_citations = [c.model_dump() for c in citations]

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
        if target is not None:
            tool_args["target"] = asdict(target)
        # 9.3.1：跨轮引用场景收窄 prompt（只含 summary + rag_answer +
        # citations + table_facts，不含 evidence_pack items 摘录）。
        tool_args["narrow_context"] = self._is_cross_turn_query(query, session_id)

        queue_wait_ms = 0
        if target is not None and target.profile == "generation":
            queue_started = time.monotonic()
            with self._model_runtime.acquire(
                target.profile,
                on_queue=lambda position: self._emit_queue_event(position, target),
                cancel_event=self._cancel_event,
            ):
                queue_wait_ms = int((time.monotonic() - queue_started) * 1000)
                synth_result = self._tools.call_tool(
                    "answer.synthesize",
                    tool_args,
                    ctx={
                        "event_sink": self._event_sink,
                        "cancel_event": self._cancel_event,
                    },
                )
        else:
            synth_result = self._tools.call_tool(
                "answer.synthesize",
                tool_args,
                ctx={
                    "event_sink": self._event_sink,
                    "cancel_event": self._cancel_event,
                },
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
                "queue_ms": queue_wait_ms,
            }
            performance = sdata.get("performance")
            if isinstance(performance, dict):
                synth_meta.update(performance)
            if target is not None:
                synth_meta.update(
                    {
                        "inference_profile": target.profile,
                        "inference_context_length": target.context_length,
                        "inference_reason": target.reason,
                    }
                )
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

    def _emit_queue_event(self, position: int, target: InferenceTarget) -> None:
        if self._event_sink is not None:
            self._event_sink(
                "queue",
                {
                    "position": position,
                    "profile": target.profile,
                    "model": target.model,
                },
            )

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

    def _resolvable_previous_turn(self, session_id: str, normalized: str) -> str | None:
        """返回可解析的上一轮用户提问；无 session/历史/与当前相同 → None。

        仅供 `_contextualize_retrieval_query` 使用（检索上下文化需要
        "上一轮问题"文本；防重复与 last_n 窗口守卫在这里仍然合理）。
        跨轮判定已独立为轮次计数（见 `_is_cross_turn_query`）。
        """
        if not session_id:
            return None
        try:
            user_turns = [
                turn
                for turn in self._memory.get_history(session_id, last_n=12)
                if turn.role == "user" or turn.step_type == "user_query"
            ]
        except Exception:
            return None
        if len(user_turns) < 2:
            return None
        previous_query = user_turns[-2].content.strip()
        if not previous_query or previous_query == normalized:
            return None
        return previous_query

    def _is_cross_turn_query(self, query: str, session_id: str | None) -> bool:
        """判断当前 query 是否为跨轮查询（第 2 轮起一律视为跨轮）。

        反向默认（2026-08-11 grill 收敛）：不做指代词词表匹配——词表
        永远追不完用户表达（"第二点/展开/审计"等实测漏检 8/8 直通失败）。
        会话第 2 轮起的查询一律走综合路径（带历史摘要），第 1 轮保留
        draft 直通。

        注意调用顺序契约：调用方必须先把当前 query 写入历史
        （user_query turn，见 execute 开头），因此历史中 user turn
        数量 ≥ 2 即"第 2 轮起"。独立统计全部 user turn，不使用
        `_resolvable_previous_turn`——后者的防重复（重复提问）与
        last_n 窗口守卫在 marker 语义下合理，但与"一律跨轮"冲突
        （verbatim 重试同一问题 / 长间隙会话会漏判）。

        返回值只决定 synthesize 是否启用；检索侧的记忆扩展仍由
        `_contextualize_retrieval_query` 负责。
        """
        if not session_id:
            return False
        normalized = (query or "").strip()
        if not normalized:
            return False
        try:
            user_turns = [
                turn
                for turn in self._memory.get_history(session_id)
                if turn.role == "user" or turn.step_type == "user_query"
            ]
        except Exception:
            return False
        return len(user_turns) >= 2

    def _contextualize_retrieval_query(self, session_id: str, query: str) -> str:
        """把第 2 轮起的简短查询扩展为自包含的 RAG 查询。

        例如用户先问“这篇文章的方法是什么”，再问“它的准确率呢？”；
        第二个 query 很短，需要把前一个问题拼接进去，否则 RAG 检索
        不到上下文。短查询（≤30 字符）几乎必然是引用上文（"第二点呢"
        "为什么""展开"），一律包装，不再依赖指代词词表（词表漏检导致
        Q5"第二点"类查询原始检索 miss）；长查询视为自带上下文，不做
        包装，避免污染独立新问题。
        """
        normalized = query.strip()
        if len(normalized) > 30:
            return query
        previous_query = self._resolvable_previous_turn(session_id, normalized)
        if previous_query is None:
            return query
        return f"上一轮问题：{previous_query}\n当前追问：{normalized}"

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
        table_facts = list(base.get("table_facts", []))
        if session_pack and session_pack.get("items"):
            items.extend(session_pack["items"])
        if session_pack and session_pack.get("table_facts"):
            table_facts.extend(session_pack["table_facts"])
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
        result = {"status": status, "items": items, "table_facts": table_facts}
        # 9.7.3：coverage 元数据透传——附件 facts 并入后无法按单一 inventory
        # 验证，降级为 unknown（门禁中性）；否则原样保留 base 的 coverage 字段。
        if session_pack and session_pack.get("table_facts"):
            result["coverage_status"] = "unknown"
            result["coverage_missing_tables"] = []
            result["inventory"] = []
        else:
            for key in ("inventory", "coverage_status", "coverage_missing_tables"):
                if key in base:
                    result[key] = base[key]
        return result

    @staticmethod
    def _retarget_citation_markers(
        answer_text: str,
        selected_indexes: list[int],
    ) -> str:
        """Retarget inline citation markers after compacting citations.

        The synthesis model cites indexes from the pre-filter citation list.
        Once uncited entries are removed for the final response, the remaining
        citations receive compact zero-based indexes.  Rewrite only markers
        that refer to selected entries and drop stale markers so the answer
        cannot point at a different source than the response metadata.
        """
        index_map = {
            old_index: new_index
            for new_index, old_index in enumerate(selected_indexes)
        }

        def replace(match: re.Match[str]) -> str:
            old_index = int(match.group(1))
            new_index = index_map.get(old_index)
            return f"[{new_index}]" if new_index is not None else ""

        return re.sub(r"\[(\d+)\]", replace, answer_text)

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
    def _merge_table_evidence_citations(
        base_citations: list[Citation],
        evidence_pack: dict[str, Any] | None,
        *,
        extra_citations: list[Citation] | None = None,
    ) -> list[Citation]:
        """Add complete table evidence as independent synthesis citations.

        ``rag.answer`` may return only its top few citations even when
        ``rag.retrieve_evidence`` assembled all requested table Children.
        Promote those table items into the citation list used by synthesis,
        preserving exact excerpts and source metadata.  An exact duplicate
        replaces the compact citation so the richer table identity is kept.
        Narrative evidence is deliberately untouched.
        """
        merged = list(base_citations)
        positions: dict[tuple[Any, ...], int] = {}

        def key(citation: Citation) -> tuple[Any, ...]:
            if citation.chunk_id:
                return (
                    "chunk",
                    citation.document_id,
                    citation.chunk_id,
                    citation.excerpt,
                )
            return (
                "page",
                citation.document_id,
                citation.excerpt,
                citation.page_title,
                citation.page_kind,
            )

        for position, citation in enumerate(merged):
            positions[key(citation)] = position

        table_items = evidence_pack.get("items", []) if evidence_pack else []
        if not isinstance(table_items, list):
            table_items = []
        for item in table_items:
            if not isinstance(item, dict):
                continue
            if item.get("evidence_kind") != "table" and not item.get("table_id"):
                continue
            excerpt = str(item.get("excerpt") or "").strip()
            if not excerpt:
                continue
            source_spans = item.get("source_spans")
            if not isinstance(source_spans, list):
                source_spans = []
            citation = Citation(
                document_id=item.get("document_id"),
                chunk_id=item.get("chunk_id"),
                attachment_id=item.get("attachment_id"),
                page_slug=item.get("page_slug"),
                page_title=item.get("page_title"),
                page_kind=item.get("page_kind") or "source",
                score=float(item.get("score") or 0.0),
                page_label=item.get("page_label"),
                excerpt=excerpt,
                parse_version=item.get("parse_version"),
                parent_chunk_id=item.get("parent_chunk_id"),
                block_type=item.get("block_type") or "table",
                source_spans=source_spans,
                asset_id=item.get("asset_id"),
                table_id=item.get("table_id"),
                figure_id=item.get("figure_id"),
                formula_id=item.get("formula_id"),
            )
            citation_key = key(citation)
            existing_position = positions.get(citation_key)
            if existing_position is None:
                positions[citation_key] = len(merged)
                merged.append(citation)
            else:
                merged[existing_position] = citation

        for citation in extra_citations or []:
            if not isinstance(citation, Citation) or not citation.excerpt.strip():
                continue
            citation_key = key(citation)
            existing_position = positions.get(citation_key)
            if existing_position is None:
                positions[citation_key] = len(merged)
                merged.append(citation)
            else:
                merged[existing_position] = citation

        return merged

    def _table_fact_source_citations(
        self,
        evidence_pack: dict[str, Any] | None,
    ) -> list[Citation]:
        """Load exact Child excerpts referenced by complete table facts.

        Table facts can be assembled from sibling Children that fall outside
        the bounded top-context list.  Resolve their source IDs directly from
        the same database, but require document/table/parse-version identity
        from the fact before exposing a citation to synthesis.
        """
        if not self._db or not evidence_pack:
            return []
        facts = evidence_pack.get("table_facts")
        items = evidence_pack.get("items")
        if not isinstance(facts, list) or not facts:
            return []
        if not isinstance(items, list):
            items = []

        expected: dict[str, tuple[str, str, str]] = {}
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            document_id = str(fact.get("document_id") or "").strip()
            parse_version = str(fact.get("parse_version") or "").strip()
            table_id = str(fact.get("table_id") or "").strip()
            source_ids = fact.get("source_chunk_ids")
            if not document_id or not parse_version or not table_id:
                continue
            if not isinstance(source_ids, list):
                continue
            for source_id in source_ids:
                source_id = str(source_id or "").strip()
                if source_id:
                    expected[source_id] = (document_id, parse_version, table_id)
        if not expected:
            return []

        templates: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("document_id") or ""),
                str(item.get("parse_version") or ""),
                str(item.get("table_id") or ""),
            )
            if all(key):
                templates.setdefault(key, item)

        citations: list[Citation] = []
        for chunk_id in sorted(expected):
            document_id, parse_version, table_id = expected[chunk_id]
            chunk = self._db.get(DocumentChunk, chunk_id)
            if chunk is None:
                continue
            if (
                chunk.document_id != document_id
                or str(chunk.parse_version or "") != parse_version
                or chunk.chunk_role != "child"
                or chunk.block_type != "table"
            ):
                continue
            template = templates.get((document_id, parse_version, table_id), {})
            source_spans = chunk.source_spans if isinstance(chunk.source_spans, list) else []
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    page_slug=template.get("page_slug"),
                    page_title=template.get("page_title") or getattr(chunk.document, "title", None),
                    page_kind=template.get("page_kind") or "source",
                    score=float(template.get("score") or 0.0),
                    page_label=chunk.page_label or template.get("page_label"),
                    excerpt=str(chunk.text or "").strip(),
                    parse_version=chunk.parse_version,
                    parent_chunk_id=chunk.parent_chunk_id,
                    block_type=chunk.block_type,
                    source_spans=source_spans,
                    table_id=table_id,
                )
            )
        return citations

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
    """粗略估算 token 数：按每 4 个字符 1 个 token 计算。"""
    if not text:
        return 0
    return max(1, len(text) // 4)
