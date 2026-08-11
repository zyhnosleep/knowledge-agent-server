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


def test_fact_unit_parses_trailing_units() -> None:
    """9.3.2：_fact_unit 从 value 尾部解析单位。"""
    from app.services.search import _fact_unit

    assert _fact_unit("21.0%") == "%"
    assert _fact_unit("80.5 kcal/mol") == "kcal/mol"
    assert _fact_unit("2.5 nm") == "nm"
    assert _fact_unit("1.18") == ""
    assert _fact_unit("") == ""


def test_fact_id_is_stable_hash() -> None:
    """9.3.2：_fact_id 对相同内容稳定、不同内容不同。"""
    from app.services.search import _fact_id

    a = _fact_id("table-1", 0, "F1", "91.2")
    b = _fact_id("table-1", 0, "F1", "91.2")
    c = _fact_id("table-1", 1, "F1", "91.2")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_retrieve_evidence_tool_preserves_source_linked_table_facts() -> None:
    """Agent synthesis must receive canonical table facts, not only excerpts."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.db.session import Base
    from app.schemas.agent import EvidenceItem, EvidencePack, TableFactEvidence

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
            pass

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            return EvidencePack(
                status="ok",
                items=[
                    EvidenceItem(
                        index=0,
                        document_id="d1",
                        chunk_id="row-a",
                        table_id="table-1",
                        block_type="table",
                        score=0.95,
                        excerpt="Table 1 | Model-A | 91.2",
                        evidence_kind="table",
                    )
                ],
                table_facts=[
                    TableFactEvidence(
                        table_id="table-1",
                        document_id="d1",
                        row_label="Model-A",
                        column="F1",
                        value="91.2",
                        row_index=1,
                        source_chunk_ids=["row-a"],
                    )
                ],
            )

    reg = ToolRegistry()
    reg._register_builtins(FakeRAG())

    result = reg.call_tool(
        "rag.retrieve_evidence",
        {"project_slug": "demo", "question": "What does Table 1 report?"},
        ctx={"db": db},
    )

    assert result["ok"] is True
    # 9.3.2：TableFactEvidence 新增 fact_id/unit/term 字段（默认空）
    assert result["result"]["table_facts"] == [
        {
            "table_id": "table-1",
            "document_id": "d1",
            "parse_version": None,
            "row_label": "Model-A",
            "column": "F1",
            "value": "91.2",
            "row_index": 1,
            "source_chunk_ids": ["row-a"],
            "fact_id": "",
            "unit": "",
            "term": "",
        }
    ]


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
    # 9.7：旧 EvidencePack 无 coverage 字段 → 透传默认 unknown（兼容）
    assert result["result"]["coverage_status"] == "unknown"
    assert result["result"]["inventory"] == []
    assert result["result"]["coverage_missing_tables"] == []


def test_retrieve_evidence_tool_passes_through_coverage_metadata() -> None:
    """9.7.3：rag.retrieve_evidence 工具原样透传 inventory/coverage 元数据。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.db.session import Base
    from app.schemas.agent import EvidencePack, TableCoverage, TableFactEvidence

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
            pass

        def retrieve_evidence(self, db, project_slug, question, limit=15, document_id=None):
            return EvidencePack(
                status="ok",
                items=[],
                table_facts=[
                    TableFactEvidence(
                        table_id="table-1",
                        document_id="d1",
                        parse_version="v1",
                        row_label="Model-A",
                        column="F1",
                        value="91.2",
                        row_index=1,
                    )
                ],
                inventory=[
                    TableCoverage(
                        document_id="d1",
                        parse_version="v1",
                        table_id="table-1",
                        row_count=2,
                        row_indices=[1, 2],
                    )
                ],
                coverage_status="partial",
                coverage_missing_tables=["table-1"],
            )

    reg = ToolRegistry()
    reg._register_builtins(FakeRAG())

    result = reg.call_tool(
        "rag.retrieve_evidence",
        {"project_slug": "demo", "question": "What does Table 1 report?"},
        ctx={"db": db},
    )

    assert result["ok"] is True
    assert result["result"]["coverage_status"] == "partial"
    assert result["result"]["coverage_missing_tables"] == ["table-1"]
    assert result["result"]["inventory"] == [
        {
            "document_id": "d1",
            "parse_version": "v1",
            "table_id": "table-1",
            "row_count": 2,
            "source_block_ids": [],
            "child_ids": [],
            "child_count": 0,
            "parent_ids": [],
            "row_indices": [1, 2],
        }
    ]


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
                "model": "qwen3.5:9b",
                "context_length": 32768,
                "reason": "manual deep",
            },
        },
    )

    assert result["ok"] is True
    assert synthesizer.target.profile == "deep"
    assert result["result"]["model"] == "qwen3.5:9b"


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


def test_answer_synthesize_schema_includes_narrow_context() -> None:
    """9.3.1/任务9：answer.synthesize input_schema 声明 narrow_context 布尔字段。"""
    reg = ToolRegistry()

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

    reg._register_builtins(FakeRAG())
    spec = reg.get("answer.synthesize")["spec"]
    props = spec.input_schema.get("properties", {})
    assert "narrow_context" in props, (
        f"narrow_context should be in schema properties, got {list(props.keys())}"
    )
    assert props["narrow_context"]["type"] == "boolean"
    # 可选的：默认缺省时走非收窄路径
    assert "narrow_context" not in spec.input_schema.get("required", [])


def test_answer_synthesize_forwards_narrow_context_and_evidence_pack() -> None:
    """9.7/任务9：answer.synthesize 原样透传 narrow_context 与 evidence_pack
    （含 inventory/coverage 字段），不在工具层重新猜 facts。"""
    captured: dict = {}

    class FakeRAG:
        def answer(self, db, project_slug, question, document_id=None):
            pass

    class FakeSynthesizer:
        def synthesize(self, **kwargs):
            captured.update(kwargs)
            return {
                "answer_markdown": "ok [0]",
                "cited_indexes": [0],
                "warnings": [],
                "confidence": 1.0,
                "provider": "local",
                "model": "local",
            }

    evidence_pack = {
        "status": "ok",
        "items": [
            {
                "index": 0,
                "document_id": "d1",
                "excerpt": "evidence text",
                "evidence_kind": "source_chunk",
                "source_stage": "source_chunk",
                "support_hint": "direct",
            }
        ],
        "table_facts": [
            {
                "table_id": "table-7",
                "document_id": "d1",
                "parse_version": "canonical-v4",
                "row_label": "total",
                "column": "value",
                "value": "21.0",
                "row_index": 0,
            }
        ],
        "inventory": [
            {
                "document_id": "d1",
                "parse_version": "canonical-v4",
                "table_id": "table-7",
                "row_count": 2,
                "source_block_ids": ["sb-1"],
                "child_ids": ["inv-child-1"],
                "child_count": 1,
                "parent_ids": ["p-1"],
                "row_indices": [0, 1],
            }
        ],
        "coverage_status": "partial",
        "coverage_missing_tables": ["table-7"],
    }
    reg = ToolRegistry()
    reg._register_builtins(FakeRAG(), synthesizer=FakeSynthesizer())
    result = reg.call_tool(
        "answer.synthesize",
        {
            "query": "What is X?",
            "rag_answer": "X is a thing.",
            "citations": [{"excerpt": "evidence"}],
            "evidence_pack": evidence_pack,
            "narrow_context": True,
        },
    )

    assert result["ok"] is True
    assert captured["narrow_context"] is True
    # evidence_pack 原样透传：工具层不拆解、不重猜 facts/coverage
    assert captured["evidence_pack"] == evidence_pack
