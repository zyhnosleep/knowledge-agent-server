from __future__ import annotations

import pytest

from app.schemas.agent import ToolSpec
from app.services.tool_registry import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
    ToolSchemaError,
)


def test_register_and_get() -> None:
    reg = ToolRegistry()
    spec = ToolSpec(name="echo", description="echo", input_schema={})
    reg.register(spec, lambda args, ctx: args)
    item = reg.get("echo")
    assert item["spec"].name == "echo"
    assert callable(item["handler"])


def test_get_nonexistent_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolNotFoundError, match="nonexistent"):
        reg.get("nonexistent")


def test_list_tools() -> None:
    reg = ToolRegistry()
    reg.register(ToolSpec(name="a", description="a", input_schema={}), lambda a, c: None)
    reg.register(ToolSpec(name="b", description="b", input_schema={}), lambda a, c: None)
    tools = reg.list_tools()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"a", "b"}


def test_call_tool_ok() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(name="add", description="add", input_schema={}),
        lambda args, ctx: args["a"] + args["b"],
    )
    result = reg.call_tool("add", {"a": 1, "b": 2})
    assert result["ok"] is True
    assert result["result"] == 3
    assert result["name"] == "add"


def test_call_tool_schema_validation_missing_required() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="greet",
            description="greet",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name"],
            },
        ),
        lambda args, ctx: f"Hello {args['name']}",
    )
    # missing required "name"
    result = reg.call_tool("greet", {"age": 30})
    assert result["ok"] is False
    assert result["error_type"] == "ToolSchemaError"
    assert "name" in result["error"]


def test_call_tool_schema_type_mismatch() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="double",
            description="double",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        ),
        lambda args, ctx: args["value"] * 2,
    )
    # string where integer expected
    result = reg.call_tool("double", {"value": "hello"})
    assert result["ok"] is False
    assert result["error_type"] == "ToolSchemaError"


def test_call_tool_validates_array_type() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="sum",
            description="sum",
            input_schema={
                "type": "object",
                "properties": {"items": {"type": "array"}},
                "required": ["items"],
            },
        ),
        lambda args, ctx: sum(args["items"]),
    )
    # string where array expected
    result = reg.call_tool("sum", {"items": "not-an-array"})
    assert result["ok"] is False
    assert result["error_type"] == "ToolSchemaError"


def test_call_tool_handler_exception() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(name="boom", description="boom", input_schema={}),
        lambda args, ctx: 1 / 0,
    )
    result = reg.call_tool("boom", {})
    assert result["ok"] is False
    assert result["error_type"] == "ToolExecutionError"
    assert "division" in result["error"].lower()


def test_builtin_rag_answer_registered() -> None:
    """_register_builtins adds rag.answer, answer.synthesize, and answer.verify tools."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            from app.schemas.common import QueryResponse
            return QueryResponse(
                answer_markdown="fake answer",
                citations=[],
                verification_status="local-only",
            )

    rag = FakeRAG()
    reg._register_builtins(rag)
    tools = reg.list_tools()
    assert len(tools) == 3  # rag.answer + answer.synthesize + answer.verify
    names = {t.name for t in tools}
    assert "rag.answer" in names


def test_builtin_rag_answer_call() -> None:
    """Calling rag.answer returns structured result."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.db.session import Base

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            from app.schemas.common import QueryResponse, Citation
            return QueryResponse(
                answer_markdown="fake answer text",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.95,
                        excerpt="test excerpt",
                    )
                ],
                verification_status="local-only",
            )

    reg = ToolRegistry()
    rag = FakeRAG()
    reg._register_builtins(rag)

    result = reg.call_tool(
        "rag.answer",
        {"project_slug": "demo", "question": "hello"},
        ctx={"db": db},
    )
    assert result["ok"] is True
    assert result["result"]["answer_markdown"] == "fake answer text"
    assert len(result["result"]["citations"]) == 1
    assert result["result"]["citations"][0]["document_id"] == "d1"


def test_call_tool_missing_db_context() -> None:
    """rag.answer fails when ctx is missing 'db'."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

    reg._register_builtins(FakeRAG())
    result = reg.call_tool(
        "rag.answer",
        {"project_slug": "demo", "question": "hello"},
        ctx={},  # no 'db'
    )
    assert result["ok"] is False
    assert result["error_type"] == "ToolExecutionError"


# ----------------------------------------------------------------
# v2a tests — answer.verify tool, page_kind/page_label
# ----------------------------------------------------------------


def test_builtins_include_answer_verify() -> None:
    """_register_builtins registers both rag.answer and answer.verify."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

    reg._register_builtins(FakeRAG())
    tools = reg.list_tools()
    names = {t.name for t in tools}
    assert "rag.answer" in names
    assert "answer.verify" in names


def test_answer_verify_tool_call_ok() -> None:
    """Calling answer.verify with valid args returns structured result."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

    reg._register_builtins(FakeRAG())
    result = reg.call_tool(
        "answer.verify",
        {
            "question": "What is X?",
            "answer_markdown": "X is a thing.",
            "citations": [{"document_id": "d1"}],
            "route_type": "simple_rag",
        },
    )
    assert result["ok"] is True
    assert "ok" in result["result"]
    assert "warnings" in result["result"]
    assert "retry_recommended" in result["result"]
    assert "reason" in result["result"]


def test_rag_answer_citation_includes_page_kind_and_page_label() -> None:
    """rag.answer citations include page_kind and page_label when available."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.db.session import Base

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            from app.schemas.common import QueryResponse, Citation
            return QueryResponse(
                answer_markdown="test",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.95,
                        excerpt="test excerpt",
                        page_kind="table",
                        page_label="Table 1",
                    )
                ],
                verification_status="local-only",
            )

    reg = ToolRegistry()
    reg._register_builtins(FakeRAG())
    result = reg.call_tool(
        "rag.answer",
        {"project_slug": "demo", "question": "hello"},
        ctx={"db": db},
    )
    assert result["ok"] is True
    cit = result["result"]["citations"][0]
    assert cit.get("page_kind") == "table"
    assert cit.get("page_label") == "Table 1"


def test_answer_verify_is_read_only() -> None:
    """answer.verify has side_effect_level='none'."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

    reg._register_builtins(FakeRAG())
    spec = reg.get("answer.verify")["spec"]
    assert spec.side_effect_level == "none"


# ------------------------------------------------------------------
# rag.retrieve_evidence tool tests
# ------------------------------------------------------------------


def test_builtins_include_retrieve_evidence() -> None:
    """_register_builtins registers rag.retrieve_evidence alongside existing tools."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            from app.schemas.agent import EvidencePack
            return EvidencePack(status="ok", items=[])

    reg._register_builtins(FakeRAG())
    tools = reg.list_tools()
    names = {t.name for t in tools}
    # All four tools must be registered
    assert "rag.answer" in names
    assert "answer.synthesize" in names
    assert "answer.verify" in names
    assert "rag.retrieve_evidence" in names
    assert len(tools) == 4


def test_retrieve_evidence_tool_call_ok() -> None:
    """Calling rag.retrieve_evidence with valid args returns evidence pack."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.db.session import Base

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()

    from app.schemas.agent import EvidenceItem, EvidencePack

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            return EvidencePack(
                status="ok",
                items=[
                    EvidenceItem(
                        index=0,
                        document_id="d1",
                        score=0.95,
                        excerpt="test excerpt",
                        evidence_kind="table",
                        source_stage="document_table",
                        support_hint="direct",
                    )
                ],
            )

    reg = ToolRegistry()
    rag = FakeRAG()
    reg._register_builtins(rag)

    result = reg.call_tool(
        "rag.retrieve_evidence",
        {"project_slug": "demo", "question": "hello"},
        ctx={"db": db},
    )
    assert result["ok"] is True
    r = result["result"]
    assert r["status"] == "ok"
    assert len(r["items"]) == 1
    assert r["items"][0]["document_id"] == "d1"
    assert r["items"][0]["evidence_kind"] == "table"


def test_retrieve_evidence_tool_accepts_optional_limit() -> None:
    """rag.retrieve_evidence accepts optional limit parameter."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.db.session import Base

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()

    call_args = {}

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            call_args["limit"] = limit
            from app.schemas.agent import EvidencePack
            return EvidencePack(status="ok", items=[])

    reg = ToolRegistry()
    rag = FakeRAG()
    reg._register_builtins(rag)

    result = reg.call_tool(
        "rag.retrieve_evidence",
        {"project_slug": "demo", "question": "hello", "limit": 7},
        ctx={"db": db},
    )
    assert result["ok"] is True
    assert call_args["limit"] == 7


def test_retrieve_evidence_tool_is_read_only() -> None:
    """rag.retrieve_evidence has side_effect_level='none'."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            from app.schemas.agent import EvidencePack
            return EvidencePack(status="ok", items=[])

    reg._register_builtins(FakeRAG())
    spec = reg.get("rag.retrieve_evidence")["spec"]
    assert spec.side_effect_level == "none"


def test_retrieve_evidence_tool_requires_project_slug() -> None:
    """rag.retrieve_evidence fails schema validation when project_slug is missing."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            pass

    reg._register_builtins(FakeRAG())
    result = reg.call_tool(
        "rag.retrieve_evidence",
        {"question": "hello"},  # missing project_slug
    )
    assert result["ok"] is False
    assert result["error_type"] == "ToolSchemaError"


def test_retrieve_evidence_tool_missing_db_context() -> None:
    """rag.retrieve_evidence fails when ctx is missing 'db'."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            pass

    reg._register_builtins(FakeRAG())
    result = reg.call_tool(
        "rag.retrieve_evidence",
        {"project_slug": "demo", "question": "hello"},
        ctx={},
    )
    assert result["ok"] is False
    assert result["error_type"] == "ToolExecutionError"


# ------------------------------------------------------------------
# answer.synthesize evidence_pack tests (Phase 3)
# ------------------------------------------------------------------


def test_answer_synthesize_schema_includes_evidence_pack() -> None:
    """answer.synthesize input_schema includes optional evidence_pack field."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

    reg._register_builtins(FakeRAG())
    spec = reg.get("answer.synthesize")["spec"]
    props = spec.input_schema.get("properties", {})
    assert "evidence_pack" in props, (
        f"evidence_pack should be in schema properties, got {list(props.keys())}"
    )
    assert props["evidence_pack"]["type"] == "object"
    # evidence_pack is NOT required
    assert "evidence_pack" not in spec.input_schema.get("required", [])


def test_answer_synthesize_uses_injected_synthesizer_and_target() -> None:
    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

    class FakeSynthesizer:
        def __init__(self) -> None:
            self.target = None

        def synthesize(self, **kwargs):
            self.target = kwargs["target"]
            return {
                "answer_markdown": "routed [0]",
                "cited_indexes": [0],
                "warnings": [],
                "confidence": 1.0,
                "provider": "local",
                "model": self.target.model,
            }

    synthesizer = FakeSynthesizer()
    reg = ToolRegistry()
    reg._register_builtins(FakeRAG(), synthesizer=synthesizer)
    result = reg.call_tool(
        "answer.synthesize",
        {
            "query": "Question",
            "rag_answer": "Draft",
            "citations": [{"excerpt": "evidence"}],
            "target": {
                "profile": "deep",
                "base_url": "http://deep:11434",
                "model": "qwen3.6:27b",
                "context_length": 32768,
                "reason": "manual deep",
            },
        },
    )

    assert result["ok"] is True
    assert synthesizer.target.profile == "deep"
    assert result["result"]["model"] == "qwen3.6:27b"


def test_answer_synthesize_accepts_evidence_pack_in_call() -> None:
    """Calling answer.synthesize with evidence_pack passes schema validation."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

    reg._register_builtins(FakeRAG())
    result = reg.call_tool(
        "answer.synthesize",
        {
            "query": "What is X?",
            "rag_answer": "X is a thing.",
            "evidence_pack": {
                "status": "ok",
                "items": [
                    {
                        "index": 0,
                        "document_id": "d1",
                        "score": 0.95,
                        "excerpt": "evidence text",
                        "evidence_kind": "source_chunk",
                        "source_stage": "source_chunk",
                        "support_hint": "direct",
                    }
                ],
            },
        },
    )
    assert result["ok"] is True
    assert "answer_markdown" in result["result"]


def test_answer_synthesize_without_evidence_pack_still_works() -> None:
    """Calling answer.synthesize without evidence_pack is backward compatible."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

    reg._register_builtins(FakeRAG())
    result = reg.call_tool(
        "answer.synthesize",
        {
            "query": "What is X?",
            "rag_answer": "X is a thing.",
        },
    )
    assert result["ok"] is True
    assert "answer_markdown" in result["result"]


def test_answer_synthesize_evidence_pack_invalid_type_rejected() -> None:
    """Passing a non-object as evidence_pack fails schema validation."""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

    reg._register_builtins(FakeRAG())
    result = reg.call_tool(
        "answer.synthesize",
        {
            "query": "What is X?",
            "rag_answer": "X is a thing.",
            "evidence_pack": "not-an-object",  # invalid type
        },
    )
    assert result["ok"] is False
    assert result["error_type"] == "ToolSchemaError"
