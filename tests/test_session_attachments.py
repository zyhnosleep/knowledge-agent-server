from __future__ import annotations

import io
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.agent_routes import agent_router
from app.core.config import get_settings
from app.db.session import Base, get_db
from app.models.records import ConversationSession, Project, SessionAttachment, SessionAttachmentChunk
from app.schemas.agent import AgentQueryRequest, EvidencePack
from app.schemas.common import Citation, QueryResponse
from app.services.agent_executor import AgentExecutor
from app.services.agent_trace_store import AgentTraceStore
from app.services.conversation_memory import ConversationMemory
from app.services.rag_adapter import RAGAdapter
from app.services.session_attachments import (
    delete_attachments_for_session,
    list_session_attachments,
    retrieve_session_attachment_evidence,
)
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


def make_client(db: Session) -> TestClient:
    app = FastAPI()
    app.include_router(agent_router, prefix="/api/agent")

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    settings = get_settings()
    settings.agent_enabled = True
    settings.agent_synthesis_provider = "local"
    return TestClient(app, raise_server_exceptions=False)


def _make_txt_upload(name: str, content: str) -> tuple[bytes, str]:
    return (content.encode("utf-8"), name)


def _pdf_with_nul_text() -> bytes:
    """构造文本层含 NUL (0x00) 的最小单页 PDF（与 test_parser_errors 同构）。"""
    content = b"BT /F1 12 Tf 72 720 Td (alpha\x00beta) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
    ]
    body = b"%PDF-1.4\n"
    offsets: list[int] = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(body))
        body += b"%d 0 obj\n" % index + obj + b"\nendobj\n"
    xref_offset = len(body)
    xref = b"xref\n0 %d\n" % (len(objects) + 1)
    xref += b"0000000000 65535 f \n"
    for off in offsets:
        xref += b"%010d 00000 n \n" % off
    trailer = b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    return body + xref + trailer + b"startxref\n%d\n%%%%EOF\n" % xref_offset


def test_upload_pdf_attachment_with_nul_bytes_strips_before_store(tmp_path) -> None:
    """含 NUL 文本层的 PDF 附件上传必须成功，chunk 文本不含 NUL。

    线上故障复现：pypdf 文本层含 0x00 → INSERT 报
    ``PostgreSQL text fields cannot contain NUL (0x00) bytes``。
    """
    db = make_db()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()

    settings = get_settings()
    settings.raw_dir = tmp_path / "raw"

    client = make_client(db)
    response = client.post(
        "/api/agent/sessions/sess-nul/attachments?project_slug=demo",
        files={"file": ("nul.pdf", io.BytesIO(_pdf_with_nul_text()), "application/pdf")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["attachment"]["file_name"] == "nul.pdf"
    assert payload["chunks"]
    for chunk in payload["chunks"]:
        assert "\x00" not in chunk["text"]
    assert "alphabeta" in payload["chunks"][0]["text"]


def test_upload_creates_session_and_attachment(tmp_path) -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    settings = get_settings()
    settings.raw_dir = tmp_path / "raw"

    client = make_client(db)
    data, filename = _make_txt_upload("notes.txt", "The quick brown fox jumps over the lazy dog.")
    response = client.post(
        "/api/agent/sessions/sess-upload/attachments?project_slug=demo",
        files={"file": (filename, io.BytesIO(data), "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["attachment"]["file_name"] == "notes.txt"
    assert payload["attachment"]["session_id"] == "sess-upload"
    assert payload["attachment"]["project_slug"] == "demo"
    assert len(payload["chunks"]) > 0

    session = db.get(ConversationSession, "sess-upload")
    assert session is not None
    assert session.project_slug == "demo"

    attachment = db.get(SessionAttachment, payload["attachment"]["id"])
    assert attachment is not None
    assert attachment.session_id == "sess-upload"
    assert attachment.project_id == "p1"


def test_list_attachments_session_scoped(tmp_path) -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    settings = get_settings()
    settings.raw_dir = tmp_path / "raw"

    client = make_client(db)
    data, filename = _make_txt_upload("notes.txt", "hello world")
    client.post(
        "/api/agent/sessions/sess-list/attachments?project_slug=demo",
        files={"file": (filename, io.BytesIO(data), "text/plain")},
    )

    response = client.get("/api/agent/sessions/sess-list/attachments?project_slug=demo")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["file_name"] == "notes.txt"


def test_list_attachments_rejects_cross_project_session(tmp_path) -> None:
    db = make_db()
    project1 = Project(id="p1", slug="demo", name="Demo")
    project2 = Project(id="p2", slug="other", name="Other")
    db.add_all([project1, project2])
    memory = ConversationMemory(db)
    memory.touch_session("cross-sess", project_slug="demo", ttl_days=30)
    db.commit()

    client = make_client(db)
    response = client.get("/api/agent/sessions/cross-sess/attachments?project_slug=other")
    assert response.status_code == 404


def test_delete_attachment_removes_rows_and_file(tmp_path) -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    settings = get_settings()
    settings.raw_dir = tmp_path / "raw"

    client = make_client(db)
    data, filename = _make_txt_upload("notes.txt", "hello world")
    upload_response = client.post(
        "/api/agent/sessions/sess-delete/attachments?project_slug=demo",
        files={"file": (filename, io.BytesIO(data), "text/plain")},
    )
    attachment_id = upload_response.json()["attachment"]["id"]
    assert db.get(SessionAttachment, attachment_id) is not None

    delete_response = client.delete(
        f"/api/agent/sessions/sess-delete/attachments/{attachment_id}?project_slug=demo"
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    db.commit()
    assert db.get(SessionAttachment, attachment_id) is None
    assert list_session_attachments(db, "sess-delete") == []


def test_retrieve_attachment_evidence_is_session_scoped() -> None:
    db = make_db()
    project1 = Project(id="p1", slug="demo", name="Demo")
    project2 = Project(id="p2", slug="other", name="Other")
    db.add_all([project1, project2])
    memory = ConversationMemory(db)
    memory.touch_session("s1", project_slug="demo", ttl_days=30)
    memory.touch_session("s2", project_slug="other", ttl_days=30)

    attachment1 = SessionAttachment(
        id="a1",
        session_id="s1",
        project_id="p1",
        file_name="demo.txt",
        storage_path="/dev/null",
        sha256="x",
        byte_size=10,
    )
    db.add(attachment1)
    db.add(SessionAttachmentChunk(id="c1", attachment_id="a1", ordinal=0, text="session one content"))
    db.commit()

    pack = retrieve_session_attachment_evidence(db, "demo", "s1", "content", limit=5)
    assert pack.status == "ok"
    assert len(pack.items) == 1
    assert pack.items[0].page_title == "demo.txt"
    assert pack.items[0].source_stage == "session_attachment"
    assert pack.items[0].evidence_kind == "session_attachment"
    assert pack.items[0].attachment_id == "a1"
    assert pack.items[0].support_hint == "direct"
    assert pack.items[0].page_kind == "session_attachment"

    # Other session has no attachments
    other = retrieve_session_attachment_evidence(db, "other", "s2", "content", limit=5)
    assert other.status == "empty"


def test_retrieve_attachment_evidence_ranks_by_overlap() -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    memory = ConversationMemory(db)
    memory.touch_session("rank-sess", project_slug="demo", ttl_days=30)

    attachment = SessionAttachment(
        id="a1",
        session_id="rank-sess",
        project_id="p1",
        file_name="rank.txt",
        storage_path="/dev/null",
        sha256="x",
        byte_size=10,
    )
    db.add(attachment)
    db.add(SessionAttachmentChunk(id="c1", attachment_id="a1", ordinal=0, text="alpha beta gamma"))
    db.add(SessionAttachmentChunk(id="c2", attachment_id="a1", ordinal=1, text="delta echo foxtrot"))
    db.commit()

    pack = retrieve_session_attachment_evidence(db, "demo", "rank-sess", "foxtrot", limit=1)
    assert pack.status == "ok"
    assert len(pack.items) == 1
    assert pack.items[0].excerpt == "delta echo foxtrot"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("只根据当前附件回答 token 和 color", True),
        ("请读取这个临时文件，不要引用项目文档", True),
        ("Answer using only the attachment", True),
        ("比较附件与项目文档", False),
        ("介绍项目中的 CHARMM36 文献", False),
    ],
)
def test_attachment_only_query_detection(query: str, expected: bool) -> None:
    assert AgentExecutor._is_attachment_only_query(query) is expected


def test_executor_attachment_only_query_skips_project_rag() -> None:
    db = make_db()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    memory = ConversationMemory(db)
    memory.touch_session("attachment-only-sess", project_slug="demo", ttl_days=30)
    db.add(
        SessionAttachment(
            id="alpha-attachment",
            session_id="attachment-only-sess",
            project_id="p1",
            file_name="alpha.txt",
            storage_path="/dev/null",
            sha256="alpha",
            byte_size=40,
        )
    )
    db.add(
        SessionAttachmentChunk(
            id="alpha-chunk",
            attachment_id="alpha-attachment",
            ordinal=0,
            text="token: helios-314159\ncolor: crimson",
        )
    )
    db.commit()

    class ForbiddenProjectRAG:
        retrieve_calls = 0
        answer_calls = 0

        def retrieve_evidence(
            self, db, project_slug, question, *, limit=15, document_id=None
        ):
            self.retrieve_calls += 1
            return EvidencePack(status="empty", items=[])

        def answer(self, db, project_slug, question, document_id=None):
            self.answer_calls += 1
            return QueryResponse(
                answer_markdown="unrelated project answer",
                citations=[
                    Citation(
                        document_id="project-doc",
                        chunk_id="project-chunk",
                        score=0.9,
                        excerpt="unrelated project evidence",
                    )
                ],
                verification_status="local-only",
            )

    rag = ForbiddenProjectRAG()
    tools = ToolRegistry()
    tools._register_builtins(rag)
    executor = AgentExecutor(
        rag=rag,
        tools=tools,
        memory=ConversationMemory(db),
        db=db,
        trace_store=AgentTraceStore(db),
        synthesizer=None,
    )

    response = executor.execute(
        AgentQueryRequest(
            project_slug="demo",
            query="只根据当前附件回答 token 和 color，不要引用项目文档",
            session_id="attachment-only-sess",
        )
    )

    assert response.status == "completed"
    assert "helios-314159" in response.final_answer
    assert "crimson" in response.final_answer
    assert rag.retrieve_calls == 0
    assert rag.answer_calls == 0
    assert response.citations
    assert all(c.page_kind == "session_attachment" for c in response.citations)
    assert all(c.attachment_id == "alpha-attachment" for c in response.citations)
    assert not any(step.tool_name == "rag.answer" for step in response.steps)
    assert not any(step.tool_name == "rag.retrieve_evidence" for step in response.steps)
    assert not any(step.tool_name == "answer.synthesize" for step in response.steps)
    finalize_step = next(step for step in response.steps if step.step_type == "finalize")
    assert finalize_step.metadata["source_scope"] == "session_attachments_only"
    assert finalize_step.metadata["project_rag_skipped"] is True
    trace = AgentTraceStore(db).get_trace(response.trace_id)
    assert trace is not None
    assert trace["citations"][0]["attachment_id"] == "alpha-attachment"


def test_executor_mixed_attachment_query_keeps_project_rag() -> None:
    db = make_db()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    memory = ConversationMemory(db)
    memory.touch_session("mixed-sess", project_slug="demo", ttl_days=30)
    db.add(
        SessionAttachment(
            id="mixed-attachment",
            session_id="mixed-sess",
            project_id="p1",
            file_name="mixed.txt",
            storage_path="/dev/null",
            sha256="mixed",
            byte_size=30,
        )
    )
    db.add(
        SessionAttachmentChunk(
            id="mixed-chunk",
            attachment_id="mixed-attachment",
            ordinal=0,
            text="attachment comparison evidence",
        )
    )
    db.commit()

    class CountingRAG:
        retrieve_calls = 0
        answer_calls = 0

        def retrieve_evidence(
            self, db, project_slug, question, *, limit=15, document_id=None
        ):
            self.retrieve_calls += 1
            return EvidencePack(status="empty", items=[])

        def answer(self, db, project_slug, question, document_id=None):
            self.answer_calls += 1
            return QueryResponse(
                answer_markdown="project comparison evidence",
                citations=[
                    Citation(
                        document_id="project-doc",
                        chunk_id="project-chunk",
                        score=0.9,
                        excerpt="project comparison evidence",
                    )
                ],
                verification_status="local-only",
            )

    rag = CountingRAG()
    tools = ToolRegistry()
    tools._register_builtins(rag)
    executor = AgentExecutor(
        rag=rag,
        tools=tools,
        memory=ConversationMemory(db),
        db=db,
        trace_store=AgentTraceStore(db),
        synthesizer=None,
    )
    response = executor.execute(
        AgentQueryRequest(
            project_slug="demo",
            query="比较附件与项目文档",
            session_id="mixed-sess",
        )
    )

    assert response.status == "completed"
    assert rag.retrieve_calls == 1
    assert rag.answer_calls == 1
    assert any(c.document_id == "project-doc" for c in response.citations)
    assert any(c.attachment_id == "mixed-attachment" for c in response.citations)


def test_executor_includes_session_attachment_evidence(monkeypatch, tmp_path) -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    memory = ConversationMemory(db)
    memory.touch_session("exec-sess", project_slug="demo", ttl_days=30)
    db.commit()

    # Seed an attachment chunk that matches the query.
    attachment = SessionAttachment(
        id="a1",
        session_id="exec-sess",
        project_id="p1",
        file_name="exec.txt",
        storage_path="/dev/null",
        sha256="x",
        byte_size=10,
    )
    db.add(attachment)
    db.add(SessionAttachmentChunk(id="c1", attachment_id="a1", ordinal=0, text="the quick brown fox"))
    db.commit()

    class StubRAG:
        def answer(self, db, project_slug, question):
            return QueryResponse(
                answer_markdown="stub answer",
                citations=[
                    Citation(
                        document_id="d1",
                        chunk_id="c1",
                        score=0.9,
                        excerpt="stub excerpt",
                    )
                ],
                verification_status="local-only",
            )

    rag = StubRAG()
    tools = ToolRegistry()
    tools._register_builtins(rag)
    executor = AgentExecutor(
        rag=rag, tools=tools, memory=ConversationMemory(db), db=db,
        trace_store=AgentTraceStore(db), synthesizer=None,
    )
    request = AgentQueryRequest(
        project_slug="demo", query="quick fox", session_id="exec-sess"
    )
    response = executor.execute(request)

    assert response.status == "completed"
    step_types = [s.step_type for s in response.steps]
    assert "retrieve" in step_types
    retrieval_step = next(s for s in response.steps if s.tool_name == "session_attachment.retrieve")
    assert retrieval_step.tool_ok is True
    assert retrieval_step.metadata["evidence_count"] == 1


def test_executor_answers_from_attachment_when_rag_is_insufficient() -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    memory = ConversationMemory(db)
    memory.touch_session("attachment-answer-sess", project_slug="demo", ttl_days=30)
    db.commit()

    attachment = SessionAttachment(
        id="a1",
        session_id="attachment-answer-sess",
        project_id="p1",
        file_name="bod.txt",
        title="BOD",
        storage_path="/dev/null",
        sha256="x",
        byte_size=10,
    )
    db.add(attachment)
    db.add(
        SessionAttachmentChunk(
            id="c1",
            attachment_id="a1",
            ordinal=0,
            text=(
                "BOD uses generative adversarial distillation with a "
                "Bradley-Terry discriminator loss to improve black-box "
                "distillation and distribution generalization."
            ),
        )
    )
    db.commit()

    class InsufficientRAG:
        def answer(self, db, project_slug, question):
            return QueryResponse(
                answer_markdown=(
                    "## Insufficient Evidence\n\nThe retrieved source documents "
                    "do not contain information relevant to the query."
                ),
                citations=[],
                verification_status="local-only",
            )

    rag = InsufficientRAG()
    tools = ToolRegistry()
    tools._register_builtins(rag)
    executor = AgentExecutor(
        rag=rag,
        tools=tools,
        memory=ConversationMemory(db),
        db=db,
        trace_store=AgentTraceStore(db),
        synthesizer=None,
    )
    request = AgentQueryRequest(
        project_slug="demo",
        query="BOD improves what?",
        session_id="attachment-answer-sess",
    )
    response = executor.execute(request)

    assert response.status == "completed"
    assert "generative adversarial distillation" in response.final_answer
    assert "当前对话" in response.final_answer
    assert response.citations
    attachment_citations = [
        citation
        for citation in response.citations
        if citation.page_kind == "session_attachment"
    ]
    assert attachment_citations
    assert attachment_citations[0].page_title == "BOD"
    assert attachment_citations[0].attachment_id == "a1"
    assert any("temporary attachments" in w for w in response.warnings)


def test_purge_expired_sessions_deletes_attachments(tmp_path, monkeypatch) -> None:
    import app.services.session_attachments as attachment_mod

    monkeypatch.setattr(attachment_mod.settings, "raw_dir", tmp_path)
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    expired = ConversationSession(
        id="expired-sess",
        project_slug="demo",
        expires_at=datetime.utcnow() - timedelta(days=1),
    )
    db.add(expired)
    attachment = SessionAttachment(
        id="a1",
        session_id="expired-sess",
        project_id="p1",
        file_name="old.txt",
        storage_path=str(tmp_path / "old.txt"),
        sha256="x",
        byte_size=10,
    )
    db.add(attachment)
    db.add(SessionAttachmentChunk(id="c1", attachment_id="a1", ordinal=0, text="old"))
    db.commit()

    # Create the file so we can verify best-effort deletion.
    (tmp_path / "old.txt").write_text("old content")

    memory = ConversationMemory(db)
    deleted = memory.purge_expired_sessions()
    db.commit()

    assert deleted == 1
    assert db.get(ConversationSession, "expired-sess") is None
    assert db.get(SessionAttachment, "a1") is None
    assert db.get(SessionAttachmentChunk, "c1") is None
    assert not (tmp_path / "old.txt").exists()


def test_delete_attachments_for_session_count() -> None:
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    memory = ConversationMemory(db)
    memory.touch_session("bulk-sess", project_slug="demo", ttl_days=30)

    for i in range(3):
        attachment = SessionAttachment(
            id=f"a{i}",
            session_id="bulk-sess",
            project_id="p1",
            file_name=f"f{i}.txt",
            storage_path="/dev/null",
            sha256="x",
            byte_size=10,
        )
        db.add(attachment)
    db.commit()

    count = delete_attachments_for_session(db, "bulk-sess")
    db.commit()
    assert count == 3
    assert list_session_attachments(db, "bulk-sess") == []


def test_attachment_evidence_is_isolated_by_session() -> None:
    """Attachment retrieval only returns chunks from the same session."""
    db = make_db()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.commit()

    memory = ConversationMemory(db)
    memory.touch_session("sess-a", project_slug="demo", ttl_days=30)
    memory.touch_session("sess-b", project_slug="demo", ttl_days=30)
    db.commit()

    attachment_a = SessionAttachment(
        id="a1",
        session_id="sess-a",
        project_id="p1",
        file_name="alpha.txt",
        storage_path="/dev/null",
        sha256="sha-a",
        byte_size=10,
    )
    chunk_a = SessionAttachmentChunk(
        id="c1",
        attachment_id="a1",
        ordinal=0,
        text="Alpha session attachment content.",
    )
    attachment_b = SessionAttachment(
        id="a2",
        session_id="sess-b",
        project_id="p1",
        file_name="beta.txt",
        storage_path="/dev/null",
        sha256="sha-b",
        byte_size=10,
    )
    chunk_b = SessionAttachmentChunk(
        id="c2",
        attachment_id="a2",
        ordinal=0,
        text="Beta session attachment content.",
    )
    db.add_all([attachment_a, chunk_a, attachment_b, chunk_b])
    db.commit()

    pack_a = retrieve_session_attachment_evidence(db, "demo", "sess-a", "content")
    assert pack_a.status == "ok"
    assert all("Alpha" in item.excerpt for item in pack_a.items)
    assert not any("Beta" in item.excerpt for item in pack_a.items)

    # Cross-session query terms do not leak evidence from sess-b into sess-a.
    cross_pack = retrieve_session_attachment_evidence(db, "demo", "sess-a", "beta")
    assert cross_pack.status == "ok"
    assert cross_pack.items
    assert all("Alpha" in item.excerpt for item in cross_pack.items)
    assert not any("Beta" in item.excerpt for item in cross_pack.items)
