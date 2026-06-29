from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from app.schemas.agent import ToolSpec

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Base exception for tool-call failures."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool name is not registered."""


class ToolSchemaError(ToolError):
    """Raised when tool arguments fail input-schema validation."""


class ToolTimeoutError(ToolError):
    """Raised when a tool handler exceeds its timeout."""


class ToolExecutionError(ToolError):
    """Raised when a tool handler raises an unexpected exception."""


class ToolRegistry:
    """Pluggable registry of named tools with schema validation.

    Each tool is a *(ToolSpec, handler)* pair.  The registry validates
    arguments against the tool's ``input_schema`` (JSON Schema subset)
    and records timing metadata.  Timeout enforcement is the
    responsibility of the caller (see ``ToolSpec.timeout_ms`` for the
    declared budget per tool).
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def register(self, spec: ToolSpec, handler: Callable[..., Any]) -> None:
        """Register *handler* under *spec.name*."""
        self._tools[spec.name] = {"spec": spec, "handler": handler}

    def get(self, name: str) -> dict[str, Any]:
        """Return the ``{"spec": ..., "handler": ...}`` dict for *name*.

        Raises ``ToolNotFoundError`` when the tool is unknown.
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def list_tools(self) -> list[ToolSpec]:
        """Return every registered tool spec."""
        return [item["spec"] for item in self._tools.values()]

    def call_tool(
        self, name: str, args: dict[str, Any] | None = None, *, ctx: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Validate *args*, invoke the handler, and return a result dict.

        The result dict always contains at least ``{"ok": bool, "name": str}``.
        On success it also includes ``"result"``; on failure it includes
        ``"error"`` and ``"error_type"``.
        """
        args = args or {}
        ctx = ctx or {}
        item = self.get(name)
        spec: ToolSpec = item["spec"]

        # ---- schema validation ----
        validation_error = self._validate_args(args, spec.input_schema)
        if validation_error:
            return {
                "ok": False,
                "name": name,
                "error": validation_error,
                "error_type": "ToolSchemaError",
            }

        # ---- execute with timeout ----
        handler = item["handler"]
        t0 = time.monotonic()
        try:
            result = handler(args, ctx)
        except ToolError as exc:
            return {
                "ok": False,
                "name": name,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        except Exception as exc:
            logger.warning("Tool '%s' raised %s: %s", name, type(exc).__name__, exc)
            return {
                "ok": False,
                "name": name,
                "error": str(exc),
                "error_type": "ToolExecutionError",
            }
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.debug("Tool '%s' completed in %d ms", name, elapsed_ms)
        return {"ok": True, "name": name, "result": result, "latency_ms": elapsed_ms}

    # ------------------------------------------------------------------
    # built-in tools
    # ------------------------------------------------------------------

    def _register_builtins(self, rag_adapter: Any) -> None:
        """Register the read-only ``rag.answer``, ``rag.retrieve_evidence``,
        ``answer.synthesize``, and ``answer.verify`` tools.

        *rag_adapter* must expose ``answer(db, project_slug, question)`` and
        ``retrieve_evidence(db, project_slug, question, limit)`` methods.
        """
        from app.schemas.agent import ToolSpec

        # ---- rag.retrieve_evidence ----
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

        # ---- rag.answer ----
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

        # ---- answer.synthesize ----
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
            lambda args, ctx: _answer_synthesize_handler(args),
        )

        # ---- answer.verify ----
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
        """Validate *args* against a JSON Schema subset.

        Returns an error message string or ``None``.
        """
        if schema.get("type") != "object":
            return None  # non-object schemas are left to the handler
        required: list[str] = schema.get("required", [])
        properties: dict[str, Any] = schema.get("properties", {})

        for key in required:
            if key not in args:
                return f"Missing required field: {key!r}"

        for key, value in args.items():
            prop = properties.get(key)
            if prop is None:
                continue
            expected_type = prop.get("type")
            if expected_type == "string" and not isinstance(value, str):
                return f"Field {key!r} must be a string, got {type(value).__name__}"
            if expected_type == "number" and not isinstance(value, (int, float)):
                return f"Field {key!r} must be a number, got {type(value).__name__}"
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


def _rag_answer_handler(
    rag_adapter: Any, ctx: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    """Call RAGAdapter.answer and return a plain dict for the tool result."""
    db = ctx.get("db")
    if db is None:
        raise ToolExecutionError("Tool context is missing 'db' Session")
    response = rag_adapter.answer(db, args["project_slug"], args["question"])
    return {
        "answer_markdown": response.answer_markdown,
        "citations": [
            {
                "document_id": c.document_id,
                "chunk_id": c.chunk_id,
                "page_slug": c.page_slug,
                "page_title": c.page_title,
                "score": c.score,
                "excerpt": c.excerpt,
                "page_kind": c.page_kind,
                "page_label": c.page_label,
            }
            for c in response.citations
        ],
        "verification_status": response.verification_status,
    }


def _rag_retrieve_evidence_handler(
    rag_adapter: Any, ctx: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    """Call RAGAdapter.retrieve_evidence and return a plain dict."""
    db = ctx.get("db")
    if db is None:
        raise ToolExecutionError("Tool context is missing 'db' Session")
    limit = args.get("limit", 15)
    if not isinstance(limit, int):
        limit = 15
    pack = rag_adapter.retrieve_evidence(
        db, args["project_slug"], args["question"], limit=limit
    )
    return {
        "status": pack.status,
        "items": [item.model_dump() for item in pack.items],
    }


def _answer_verify_handler(
    verifier: Any, args: dict[str, Any]
) -> dict[str, Any]:
    """Call AnswerVerifier.verify and return the result dict."""
    return verifier.verify(
        question=args.get("question", ""),
        answer_markdown=args.get("answer_markdown", ""),
        citations=args.get("citations") or [],
        route_type=args.get("route_type", "simple_rag"),
    )


def _answer_synthesize_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Call AgentSynthesizer.synthesize and return the result dict."""
    from app.services.agent_synthesizer import AgentSynthesizer

    synthesizer = AgentSynthesizer()
    return synthesizer.synthesize(
        query=args.get("query", ""),
        route=args.get("route", "simple_rag"),
        conversation_summary=args.get("conversation_summary", ""),
        rag_answer=args.get("rag_answer", ""),
        citations=args.get("citations") or [],
        evidence_pack=args.get("evidence_pack"),
    )
