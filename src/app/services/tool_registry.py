"""Agent 工具注册表：注册、校验与调用具名工具。

本模块提供 :class:`ToolRegistry` 及一组工具调用相关的异常类型，是 Agent
执行链路中"工具"的统一入口。核心职责：

- **注册**：把每个工具绑定为一对 ``(ToolSpec, handler)``；ToolSpec 描述
  工具的名称、输入/输出 schema、副作用等级与超时预算。
- **校验**：调用前按工具 ``input_schema``（JSON Schema 子集）校验参数，
  缺必填字段或类型不符时返回结构化错误，而非抛异常。
- **调用**：调用 handler，记录耗时，并把成功/失败统一封装为
  ``{"ok": bool, "name": str, ...}`` 结果字典。
- **内置工具**：默认注册一组只读 RAG / 答案工具（rag.answer、
  rag.retrieve_evidence、answer.synthesize、answer.verify）。

超时说明：``ToolSpec.timeout_ms`` 声明的是每个工具允许的执行预算，
但实际超时强制的执行权在调用方（Agent 执行器），本模块不做超时中断。
"""

from __future__ import annotations

import logging
import inspect
import time
from collections.abc import Callable
from typing import Any

from app.schemas.agent import ToolSpec
from app.services.execution_budget import BudgetExceeded, current_execution_budget

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """工具调用失败的基础异常。

    Base exception for tool-call failures.
    """


class ToolNotFoundError(ToolError):
    """请求的工具名未注册时抛出。

    Raised when a requested tool name is not registered.
    """


class ToolSchemaError(ToolError):
    """工具参数未通过输入 schema 校验时抛出。

    Raised when tool arguments fail input-schema validation.
    """


class ToolTimeoutError(ToolError):
    """工具 handler 执行超过其超时预算时抛出。

    Raised when a tool handler exceeds its timeout.
    """


class ToolExecutionError(ToolError):
    """工具 handler 抛出意外异常时抛出。

    Raised when a tool handler raises an unexpected exception.
    """


class ToolRegistry:
    """具名工具的可插拔注册表，带 schema 校验与耗时统计。

    Pluggable registry of named tools with schema validation.

    Each tool is a *(ToolSpec, handler)* pair.  The registry validates
    arguments against the tool's ``input_schema`` (JSON Schema subset)
    and records timing metadata.  Timeout enforcement is the
    responsibility of the caller (see ``ToolSpec.timeout_ms`` for the
    declared budget per tool).

    - 内部以 ``{name: {"spec": ToolSpec, "handler": callable}}`` 存储。
    - ``call_tool`` 返回的结果字典始终包含 ``ok`` / ``name`` 字段；
      成功附带 ``result`` 与 ``latency_ms``，失败附带 ``error`` / ``error_type``。
    """

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._tools: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def register(self, spec: ToolSpec, handler: Callable[..., Any]) -> None:
        """把 *handler* 注册到 *spec.name* 名下。

        Register *handler* under *spec.name*.

        :param spec: 工具规格描述（名称、schema、超时等）。
        :param handler: 可调用对象，签名通常为 ``handler(args, ctx)``。
        """
        self._tools[spec.name] = {"spec": spec, "handler": handler}

    def get(self, name: str) -> dict[str, Any]:
        """返回 *name* 对应的 ``{"spec": ..., "handler": ...}`` 字典。

        Return the ``{"spec": ..., "handler": ...}`` dict for *name*.

        Raises ``ToolNotFoundError`` when the tool is unknown.

        :param name: 工具名。
        :return: 包含 ``spec`` 与 ``handler`` 两个键的内部记录。
        :raises ToolNotFoundError: 工具未注册时。
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def list_tools(self) -> list[ToolSpec]:
        """返回所有已注册工具的规格列表。

        Return every registered tool spec.

        :return: :class:`ToolSpec` 列表（顺序为注册顺序）。
        """
        return [item["spec"] for item in self._tools.values()]

    def call_tool(
        self, name: str, args: dict[str, Any] | None = None, *, ctx: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """校验 *args*、调用 handler，并返回统一的结果字典。

        Validate *args*, invoke the handler, and return a result dict.

        The result dict always contains at least ``{"ok": bool, "name": str}``.
        On success it also includes ``"result"``; on failure it includes
        ``"error"`` and ``"error_type"``.

        :param name: 工具名。
        :param args: 工具调用参数（可为 None，视为空字典）。
        :param ctx: 调用上下文（如数据库会话、事件流等），可为 None。
        :return: 统一结果字典，始终含 ``ok`` 与 ``name`` 字段。
        """
        args = args or {}
        ctx = ctx or {}
        item = self.get(name)
        spec: ToolSpec = item["spec"]

        # ---- 参数 schema 校验 ----
        # 先按工具的 input_schema 检查参数；校验失败以结构化结果返回，
        # 不抛异常，便于 Agent 直接消费错误信息。
        validation_error = self._validate_args(args, spec.input_schema)
        if validation_error:
            return {
                "ok": False,
                "name": name,
                "error": validation_error,
                "error_type": "ToolSchemaError",
            }

        # ---- 执行 handler 并统计耗时 ----
        handler = item["handler"]
        # 使用单调时钟计时，不受系统时间跳变影响。
        t0 = time.monotonic()
        budget = current_execution_budget()
        if budget:
            budget.check_deadline()
        try:
            result = handler(args, ctx)
        except BudgetExceeded:
            raise
        except ToolError as exc:
            # 已知的工具类异常：保留其类型名，便于 Agent 识别错误类别。
            return {
                "ok": False,
                "name": name,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        except Exception as exc:
            # 未知异常：记警告日志（含工具名与异常类型），统一归类为
            # ToolExecutionError，避免把堆栈细节暴露给调用方。
            logger.warning("Tool '%s' raised %s: %s", name, type(exc).__name__, exc)
            return {
                "ok": False,
                "name": name,
                "error": str(exc),
                "error_type": "ToolExecutionError",
            }
        # 成功路径：换算为毫秒耗时，附加到结果字典。
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.debug("Tool '%s' completed in %d ms", name, elapsed_ms)
        return {"ok": True, "name": name, "result": result, "latency_ms": elapsed_ms}

    # ------------------------------------------------------------------
    # built-in tools
    # ------------------------------------------------------------------

    def _register_builtins(
        self, rag_adapter: Any, *, synthesizer: Any | None = None
    ) -> None:
        """注册只读工具：``rag.answer``、``rag.retrieve_evidence``、
        ``answer.synthesize`` 与 ``answer.verify``。

        Register the read-only ``rag.answer``, ``rag.retrieve_evidence``,
        ``answer.synthesize``, and ``answer.verify`` tools.

        *rag_adapter* must expose ``answer(db, project_slug, question)`` and
        ``retrieve_evidence(db, project_slug, question, limit)`` methods.

        :param rag_adapter: RAG 适配器对象（提供 answer / retrieve_evidence）。
        :param synthesizer: 可选的答案合成器；为空时在 handler 内延迟创建
            AgentSynthesizer 实例。
        """
        from app.schemas.agent import ToolSpec

        # ---- rag.retrieve_evidence：先检索证据、不生成答案 ----
        # 仅当适配器实现了 retrieve_evidence 时才注册，保持向后兼容。
        if hasattr(rag_adapter, "retrieve_evidence"):
            self.register(
                ToolSpec(
                    name="rag.retrieve_evidence",
                    description=(
                        "Retrieve evidence items from the RAG knowledge base"
                        " without drafting an answer. Returns an evidence pack"
                        " with structured metadata. Use this tool to inspect"
                        " what evidence is available before answering."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "project_slug": {
                                "type": "string",
                                "description": "Project slug to query.",
                            },
                            "question": {
                                "type": "string",
                                "description": "Natural-language question.",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max evidence items to return (default 15).",
                            },
                            "document_id": {
                                "type": "string",
                                "description": "Optional document ID to scope retrieval to a single document.",
                            },
                        },
                        "required": ["project_slug", "question"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "items": {"type": "array"},
                        },
                    },
                    side_effect_level="none",
                    timeout_ms=120000,
                ),
                lambda args, ctx: _rag_retrieve_evidence_handler(
                    rag_adapter, ctx, args
                ),
            )

        # ---- rag.answer：检索知识库并返回带引用的答案 ----
        self.register(
            ToolSpec(
                name="rag.answer",
                description=(
                    "Query the RAG knowledge base and return an answer with"
                    " citations.  Use this tool when you need evidence-backed"
                    " answers from ingested documents."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "project_slug": {
                            "type": "string",
                            "description": "Project slug to query.",
                        },
                        "question": {
                            "type": "string",
                            "description": "Natural-language question.",
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Optional document ID to scope the answer to a single document.",
                        },
                    },
                    "required": ["project_slug", "question"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "answer_markdown": {"type": "string"},
                        "citations": {"type": "array"},
                        "verification_status": {"type": "string"},
                    },
                },
                side_effect_level="none",
                timeout_ms=120000,
            ),
            lambda args, ctx: _rag_answer_handler(rag_adapter, ctx, args),
        )

        # ---- answer.synthesize：基于 RAG 证据合成最终答案 ----
        # 该工具可调用外部 API 或本地回退模型；支持在 ctx 提供 event_sink
        # 时走流式合成路径（见 _answer_synthesize_handler）。
        self.register(
            ToolSpec(
                name="answer.synthesize",
                description=(
                    "Synthesize a final answer from RAG evidence using"
                    " external API or local fallback.  Returns structured"
                    " data with answer_markdown, cited_indexes, warnings,"
                    " confidence, provider, and model."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The original query.",
                        },
                        "route": {
                            "type": "string",
                            "description": "The route type from PolicyRouter.",
                        },
                        "conversation_summary": {
                            "type": "string",
                            "description": "Summary of previous conversation turns.",
                        },
                        "rag_answer": {
                            "type": "string",
                            "description": "The answer text from rag.answer.",
                        },
                        "citations": {
                            "type": "array",
                            "description": "List of citation dicts from rag.answer.",
                        },
                        "evidence_pack": {
                            "type": "object",
                            "description": (
                                "Optional evidence pack from rag.retrieve_evidence "
                                "with status and items keys."
                            ),
                        },
                        "target": {
                            "type": "object",
                            "description": "Resolved local inference profile.",
                        },
                        "narrow_context": {
                            "type": "boolean",
                            "description": (
                                "跨轮引用场景为 True 时收窄合成上下文：prompt "
                                "不含 evidence_pack items 摘录与完整 inventory，"
                                "只保留 citations 与结构化 table_facts。"
                            ),
                        },
                    },
                    "required": ["query", "rag_answer"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "answer_markdown": {"type": "string"},
                        "cited_indexes": {"type": "array"},
                        "warnings": {"type": "array"},
                        "confidence": {"type": "number"},
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                    },
                },
                side_effect_level="none",
                timeout_ms=120000,
            ),
            lambda args, ctx: _answer_synthesize_handler(args, synthesizer, ctx),
        )

        # ---- answer.verify：按路由要求校验答案质量 ----
        # 检查空答案、缺失引用、缺失表格/数值证据等，并给出重试建议。
        from app.services.answer_verifier import AnswerVerifier

        _verifier = AnswerVerifier()
        self.register(
            ToolSpec(
                name="answer.verify",
                description=(
                    "Verify answer quality against route requirements."
                    " Checks for empty answers, missing citations, and"
                    " missing table/numeric evidence.  Returns warnings"
                    " and a retry recommendation."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The original question.",
                        },
                        "answer_markdown": {
                            "type": "string",
                            "description": "The answer text in markdown.",
                        },
                        "citations": {
                            "type": "array",
                            "description": "List of citation dicts from rag.answer.",
                        },
                        "route_type": {
                            "type": "string",
                            "description": "One of: simple_rag, evidence_required, table_or_metric, multi_source_compare, needs_clarification.",
                        },
                    },
                    "required": ["question", "answer_markdown"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "warnings": {"type": "array"},
                        "retry_recommended": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                },
                side_effect_level="none",
                timeout_ms=5000,
            ),
            lambda args, ctx: _answer_verify_handler(_verifier, args),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_args(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
        """按 JSON Schema 子集校验 *args*。

        Validate *args* against a JSON Schema subset.

        Returns an error message string or ``None``.

        支持的校验：必填字段存在性 + 基础类型检查
        （string / number / integer / boolean / array / object）。
        非 object 类型的 schema 跳过校验，交由 handler 自行处理。

        :param args: 待校验的参数字典。
        :param schema: 工具的 input_schema（JSON Schema 子集）。
        :return: 校验失败时返回错误消息字符串，否则返回 ``None``。
        """
        if schema.get("type") != "object":
            return None  # 非 object schema：留给 handler 自行校验
        required: list[str] = schema.get("required", [])
        properties: dict[str, Any] = schema.get("properties", {})

        # 先检查必填字段是否齐全。
        for key in required:
            if key not in args:
                return f"Missing required field: {key!r}"

        # 再逐字段检查类型；未在 schema 中声明的字段（prop is None）直接放行。
        for key, value in args.items():
            prop = properties.get(key)
            if prop is None:
                continue
            expected_type = prop.get("type")
            if expected_type == "string" and not isinstance(value, str):
                return f"Field {key!r} must be a string, got {type(value).__name__}"
            if expected_type == "number" and not isinstance(value, (int, float)):
                return f"Field {key!r} must be a number, got {type(value).__name__}"
            # 注意：由于 bool 是 int 的子类，integer 检查会让 True/False 通过；
            # 这是既有行为，此处仅按原逻辑注释说明，不改变判断。
            if expected_type == "integer" and not isinstance(value, int):
                return f"Field {key!r} must be an integer, got {type(value).__name__}"
            if expected_type == "boolean" and not isinstance(value, bool):
                return f"Field {key!r} must be a boolean, got {type(value).__name__}"
            if expected_type == "array" and not isinstance(value, list):
                return f"Field {key!r} must be an array, got {type(value).__name__}"
            if expected_type == "object" and not isinstance(value, dict):
                return f"Field {key!r} must be an object, got {type(value).__name__}"
        return None


# ----------------------------------------------------------------------
# handler implementations
# ----------------------------------------------------------------------


def _supports_prepared_answer(rag_adapter: Any) -> bool:
    """Only use snapshot handoff when the answer endpoint accepts it.

    Inspect before calling; retrying a TypeError after invocation could
    duplicate a model request if the error originated inside the adapter.
    """
    try:
        inspect.signature(rag_adapter.answer).bind(
            None, 'project', 'question', document_id=None,
            prepared_evidence=None, conversation_summary='', visual_intent=None,
        )
    except (TypeError, ValueError):
        return False
    return True


def _rag_answer_handler(
    rag_adapter: Any, ctx: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    """调用 RAGAdapter.answer，返回供工具结果使用的普通字典。

    Call RAGAdapter.answer and return a plain dict for the tool result.
    """
    db = ctx.get("db")
    if db is None:
        raise ToolExecutionError("Tool context is missing 'db' Session")
    internal = {}
    prepared = ctx.get('prepared_evidence') if _supports_prepared_answer(rag_adapter) else None
    if prepared is not None:
        from app.schemas.agent import EvidencePack
        from app.services.search import QueryService
        raw_pack = ctx.get('evidence_pack')
        if raw_pack is not None:
            prepared = QueryService(db).merge_prepared_evidence(
                prepared, EvidencePack.model_validate(raw_pack),
                trusted_attachment_ids=ctx.get('trusted_attachment_ids', frozenset()),
            )
            # All downstream projections (table citations/text synthesis) must
            # use the same scope-validated request-local pack as generation.
            raw_pack.clear()
            raw_pack.update(prepared.pack.model_dump())
        internal = {
            'prepared_evidence': prepared,
            'conversation_summary': ctx.get('conversation_summary', ''),
            'visual_intent': QueryService._resolve_visual_intent(
                ctx.get('answer_query', args['question']), prepared.retrieval_question),
        }
    response = rag_adapter.answer(
        db,
        args["project_slug"],
        ctx.get('answer_query', args["question"]) if prepared is not None else args['question'],
        document_id=args.get("document_id"),
        **internal,
    )
    # 把响应对象转换为普通字典；引用列表逐项 model_dump 以便 JSON 序列化。
    return {
        "answer_markdown": response.answer_markdown,
        "citations": [c.model_dump() for c in response.citations],
        "verification_status": response.verification_status,
        "evidence_snapshot_reused": prepared is not None,
        "visual_evidence": prepared.visual_evidence_trace if prepared is not None else {},
    }


def _rag_retrieve_evidence_handler(
    rag_adapter: Any, ctx: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    """调用 RAGAdapter.retrieve_evidence，返回普通字典。

    Call RAGAdapter.retrieve_evidence and return a plain dict.
    """
    db = ctx.get("db")
    if db is None:
        raise ToolExecutionError("Tool context is missing 'db' Session")
    # limit 参数安全兜底：缺失或类型不正确时回退到默认 15。
    limit = args.get("limit", 15)
    if not isinstance(limit, int):
        limit = 15
    prepare = getattr(rag_adapter, 'prepare_evidence', None)
    holder = ctx.get('prepared_evidence_out')
    if holder is not None and callable(prepare) and _supports_prepared_answer(rag_adapter):
        prepared = prepare(db, args['project_slug'], args['question'], limit=limit,
                           document_id=args.get('document_id'))
        holder['prepared'] = prepared
        pack = prepared.pack
    else:
        pack = rag_adapter.retrieve_evidence(
        db,
        args["project_slug"],
        args["question"],
        limit=limit,
        document_id=args.get("document_id"),
        )
    # 证据包对象转普通字典：status + items（逐项 model_dump）。
    return {
        "status": pack.status,
        "items": [item.model_dump() for item in pack.items],
        # Structured table facts are source-linked and lossless; dropping them
        # here forces Agent synthesis to reconstruct long-table values from a
        # bounded citation excerpt and can hide rows that retrieval already
        # proved.  Keep the wire shape JSON-safe for ToolRegistry callers.
        "table_facts": [fact.model_dump() for fact in pack.table_facts],
        # 9.7：表格覆盖元数据原样透传，Agent 只做门禁，不在工具层重新推断。
        "inventory": [inv.model_dump() for inv in pack.inventory],
        "coverage_status": pack.coverage_status,
        "coverage_missing_tables": list(pack.coverage_missing_tables),
    }


def _answer_verify_handler(
    verifier: Any, args: dict[str, Any]
) -> dict[str, Any]:
    """调用 AnswerVerifier.verify，返回其结果字典。

    Call AnswerVerifier.verify and return the result dict.

    各参数均设默认值，缺失字段不会导致校验工具崩溃。
    """
    return verifier.verify(
        question=args.get("question", ""),
        answer_markdown=args.get("answer_markdown", ""),
        citations=args.get("citations") or [],
        route_type=args.get("route_type", "simple_rag"),
    )


def _answer_synthesize_handler(
    args: dict[str, Any],
    synthesizer: Any | None = None,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """调用 AgentSynthesizer.synthesize（或流式 synthesize_stream）。

    Call AgentSynthesizer.synthesize and return the result dict.

    若上下文提供了 ``event_sink`` 且合成器支持流式输出，则走流式路径；
    否则走一次性 synthesize 路径。
    """
    # 延迟导入，避免模块加载期的循环依赖。
    from app.services.agent_synthesizer import AgentSynthesizer
    from app.services.agent_model_router import InferenceTarget

    synthesizer = synthesizer or AgentSynthesizer()
    # 可选参数 target 允许以 dict 形式传入推理目标，重建为 InferenceTarget；
    # 非 dict 时（缺失/非法）置为 None，交给合成器内部回退逻辑处理。
    raw_target = args.get("target")
    target = InferenceTarget(**raw_target) if isinstance(raw_target, dict) else None
    # 收集合成所需的全部参数，缺失字段一律使用安全的默认值。
    kwargs = dict(
        query=args.get("query", ""),
        route=args.get("route", "simple_rag"),
        conversation_summary=args.get("conversation_summary", ""),
        rag_answer=args.get("rag_answer", ""),
        citations=args.get("citations") or [],
        evidence_pack=args.get("evidence_pack"),
        target=target,
        narrow_context=bool(args.get("narrow_context", False)),
    )
    ctx = ctx or {}
    event_sink = ctx.get("event_sink")
    # 流式优先：有事件汇（event_sink）且合成器实现了 synthesize_stream 时，
    # 通过事件流逐段推送结果，并透传取消事件用于协作式取消。
    if event_sink is not None and hasattr(synthesizer, "synthesize_stream"):
        return synthesizer.synthesize_stream(
            **kwargs,
            event_sink=event_sink,
            cancel_event=ctx.get("cancel_event"),
        )
    return synthesizer.synthesize(**kwargs)
