"""R1 全库向量补充路由（检索层改进，spec: 2026-08-17-retrieval-layer-improvement-design.md）。

行为契约：
1. 非 lock 检索：`_build_rag_contexts` 总是并入一次全库向量检索（document_ids=[]）结果——
   语义相关的文档即使未过词法路由也能进入证据候选（并集语义）。
2. lock 检索（用户明确锁定某论文）：不触发全库补充，精确语义原样保留。
3. `_search_source_chunks` 传入 `question_vector` 时复用向量，不再调用 embed。
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import Document, DocumentChunk, Project
from app.services.search import PaperMatch, QueryService


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


class VectorFakeOllama:
    """二维确定性嵌入：含 "alpha" 的文本 → [1, 0]，否则 → [0, 1]。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]


class ExplodingEmbedOllama:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed must not be called when question_vector is reused")


def seed_project(db: Session) -> None:
    project = Project(id="p1", slug="demo", name="Demo")
    alpha = Document(
        id="alpha",
        project_id="p1",
        title="Alpha Paper",
        file_name="alpha.md",
        sha256="alpha",
        raw_path="raw/alpha.md",
        raw_text="alpha content",
        status="ready",
    )
    beta = Document(
        id="beta",
        project_id="p1",
        title="Beta Paper",
        file_name="beta.md",
        sha256="beta",
        raw_path="raw/beta.md",
        raw_text="beta content",
        status="ready",
    )
    alpha_chunk = DocumentChunk(
        id="alpha-c", document_id="alpha", ordinal=0,
        text="alpha evidence content", embedding=[1.0, 0.0],
    )
    beta_chunk = DocumentChunk(
        id="beta-c", document_id="beta", ordinal=0,
        text="beta evidence content", embedding=[0.0, 1.0],
    )
    db.add_all([project, alpha, beta, alpha_chunk, beta_chunk])
    db.commit()


def test_full_library_supplement_rescues_semantic_doc_beyond_lexical_route() -> None:
    """词法路由选错（只命中 beta）时，全库向量补充必须把语义相关的 alpha 捞回候选。"""
    db = make_session()
    seed_project(db)
    service = QueryService(db)
    service.ollama = VectorFakeOllama()

    # 模拟词法路由输出：只选中 beta（词法分够但语义无关），且未锁定
    contexts = service._build_rag_contexts(
        "alpha evidence content",
        "p1",
        [PaperMatch(document=db.get(Document, "beta"), score=8.0)],
    )

    assert any(context.citation.document_id == "alpha" for context in contexts)


def test_explicit_document_scope_skips_full_library_supplement(monkeypatch) -> None:
    """显式作用域（document_ids 参数，API 语义 lock-and-never-widen）不得做全库补充。"""
    db = make_session()
    seed_project(db)
    service = QueryService(db)
    service.ollama = VectorFakeOllama()

    calls: list[list[str]] = []
    original = QueryService._search_source_chunks

    def spy(self, question, project_id, document_ids, **kwargs):
        calls.append(document_ids)
        return original(self, question, project_id, document_ids, **kwargs)

    monkeypatch.setattr(QueryService, "_search_source_chunks", spy)

    service._build_rag_contexts(
        "alpha evidence content",
        "p1",
        [PaperMatch(document=db.get(Document, "alpha"), score=8.0, locked=True)],
        document_ids=["alpha"],
    )

    assert [] not in calls  # 全库（document_ids=[]）补充调用从未发生


def test_route_locked_scope_runs_full_library_supplement(monkeypatch) -> None:
    """路由锁定（词法路由 locked 分支，非显式作用域）不得跳过全库补充。

    R3 归因（2026-08-19，.task18-corpus 归因实验，attribution-report-fixed.json）：
    路由锁定是 SciFact 唯一主导伤害源（noroute−routed = +0.1182，68/300 条
    帮倒忙，8 条 nDCG 1.0→0.0）——词法路由锁错论文时补充路被整体跳过，
    正确文档永远不可见。修复：路由锁定不再跳过补充路，由 _finalize_contexts
    分数融合保住锁定文档占优（词法 route bonus），显式作用域仍锁定不 widen。
    """
    db = make_session()
    seed_project(db)
    service = QueryService(db)
    service.ollama = VectorFakeOllama()

    calls: list[list[str]] = []
    original = QueryService._search_source_chunks

    def spy(self, question, project_id, document_ids, **kwargs):
        calls.append(document_ids)
        return original(self, question, project_id, document_ids, **kwargs)

    monkeypatch.setattr(QueryService, "_search_source_chunks", spy)

    service._build_rag_contexts(
        "alpha evidence content",
        "p1",
        [PaperMatch(document=db.get(Document, "alpha"), score=8.0, locked=True)],
    )

    assert [] in calls  # 全库（document_ids=[]）补充调用必须发生


def test_route_locked_wrong_doc_supplement_rescues_correct_doc() -> None:
    """路由锁错论文（锁定 beta、问题问 alpha）时，补充路必须把 alpha 捞回候选。

    复刻归因实验 q94 机制：词法路由锁定文档与问题无关 → 锁定集内检索全是不
    相关低分 chunk → 旧语义跳过补充路 → 正确文档 nDCG 1.0→0.0（8 条）。
    新语义下补充路并入，alpha 10×cosine=10.0 压过 beta 低分 chunk 浮上。
    """
    db = make_session()
    seed_project(db)
    service = QueryService(db)
    service.ollama = VectorFakeOllama()

    contexts = service._build_rag_contexts(
        "alpha evidence content",
        "p1",
        [PaperMatch(document=db.get(Document, "beta"), score=8.0, locked=True)],
    )

    assert any(context.citation.document_id == "alpha" for context in contexts), (
        "路由锁错文档（beta）时补充路必须把语义相关的 alpha 捞回候选"
    )


def test_search_source_chunks_reuses_provided_question_vector() -> None:
    """传入 question_vector 时不再调用 embed（复用嵌入，0 额外 LLM 调用）。"""
    db = make_session()
    seed_project(db)
    service = QueryService(db)
    service.ollama = ExplodingEmbedOllama()

    contexts = service._search_source_chunks(
        "alpha evidence content",
        "p1",
        ["alpha"],
        limit=3,
        question_vector=[1.0, 0.0],
    )

    assert any(context.citation.document_id == "alpha" for context in contexts)


def test_named_comparison_searches_each_paper_with_an_explicit_scope(monkeypatch):
    db = make_session()
    seed_project(db)
    service = QueryService(db)
    service.ollama = VectorFakeOllama()
    calls = []
    original = service._search_source_chunks
    def observe(question, project_id, document_ids, **kwargs):
        calls.append(list(document_ids))
        return original(question, project_id, document_ids, **kwargs)
    monkeypatch.setattr(service, '_search_source_chunks', observe)
    matches = [PaperMatch(document=db.get(Document, doc), score=10, exact_alias=True)
               for doc in ('alpha', 'beta')]
    result = service._build_rag_contexts('Compare alpha and beta evidence content.', 'p1', matches)
    assert ['alpha'] in calls and ['beta'] in calls
    assert [] not in calls
    assert {c.citation.document_id for c in result} == {'alpha', 'beta'}
    assert all(getattr(c, 'comparison_source', False) for c in result)


def test_paper_routing_recognizes_source_defined_title_acronym():
    db = make_session()
    seed_project(db)
    paper = db.get(Document, 'alpha')
    paper.title = 'Evidence-Conditioned Answer Generation for Research Tasks'
    paper.raw_text = 'We present evidence-conditioned answer generation (ECAG). It retrieves scientific evidence.'
    db.commit()
    service = QueryService(db)
    matches = service._route_papers('Compare ECAG and Beta Paper mechanisms.', 'p1')
    assert {m.document.id for m in matches if m.exact_alias} == {'alpha', 'beta'}


def test_title_initials_are_not_aliases_without_source_definition():
    db = make_session()
    seed_project(db)
    paper = db.get(Document, 'alpha')
    paper.title = 'Evidence-Conditioned Answer Generation for Research Tasks'
    paper.raw_text = 'This paper mentions ECAG as an unrelated prior system.'
    db.commit()
    service = QueryService(db)
    matches = service._route_papers('Compare ECAG and Beta Paper mechanisms.', 'p1')
    assert not any(m.document.id == 'alpha' and m.exact_alias for m in matches)


def test_chinese_accuracy_query_prioritizes_result_table_over_prompt_examples():
    db = make_session()
    seed_project(db)
    db.add_all([
        DocumentChunk(id='alpha-result', document_id='alpha', ordinal=1,
            text='Table 1: Alpha prompting results on HotpotQA and Fever.\n'
                 '| Method | HotpotQA (EM) | Fever (Acc) |\n| --- | --- | --- |\n| Alpha | 27.4 | 60.9 |',
            embedding=[0.8, 0.6]),
        DocumentChunk(id='alpha-prompts', document_id='alpha', ordinal=2,
            text='Fever Prompts\n| Claim | Thought | Answer |\n| --- | --- | --- |\n'
                 '| Alpha Fever HotpotQA | Example in 2003 | SUPPORTS |',
            embedding=[1.0, 0.0]),
    ])
    db.commit()
    service = QueryService(db)
    service.ollama = ExplodingEmbedOllama()
    result = service._search_source_chunks('Alpha 在 HotpotQA/Fever 的准确率是多少？',
        'p1', ['alpha'], question_vector=[1.0, 0.0], limit=1)
    assert result[0].citation.chunk_id == 'alpha-result'


class TableVectorFakeOllama:
    """二维确定性嵌入：含 'hfe'（大小写不敏感）的问题 → [1, 0]（贴近 OPLS4 表 chunk）。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "hfe" in text.lower() else [0.0, 1.0] for text in texts]


def seed_metric_project(db: Session) -> None:
    """两个带表格 chunk 的文档：OPLS5（词法路命中）与 OPLS4（向量路更贴）。

    OPLS4 表 chunk 嵌入 [1.0, 0.0]（10×cosine=10.0）比 OPLS5 的
    [0.0, 1.0]（0.0）更贴近问题向量 [1, 0]——模拟内部回归
    opls5_table_metrics 中补充路的 OPLS4 在带提权竞争下总分离于
    OPLS5 的实景。
    """
    project = Project(id="p1", slug="demo", name="Demo")
    opls5 = Document(
        id="opls5",
        project_id="p1",
        title="OPLS5",
        file_name="opls5.md",
        sha256="opls5",
        raw_path="raw/opls5.md",
        raw_text="opls5 content",
        status="ready",
    )
    opls4 = Document(
        id="opls4",
        project_id="p1",
        title="OPLS4",
        file_name="opls4.md",
        sha256="opls4",
        raw_path="raw/opls4.md",
        raw_text="opls4 content",
        status="ready",
    )
    # 表格行文本不含论文名（与生产一致：表格行是列名 + 数值），两个 chunk
    # 文本同构、命中同样的锚点与通用词元，只有嵌入不同——问题在拿两者
    # 对比，OPLS4 的表格嵌入（10×cosine=10.0）比 OPLS5 的（0.0）更贴
    # 问题向量 [1, 0]。排序决胜靠原始分（结构/锚点/证据类型全部平手），
    # 复现内部回归 opls5_table_metrics 的失败机制。
    opls5_chunk = DocumentChunk(
        id="opls5-t", document_id="opls5", ordinal=0,
        text="| Table 2 | HFE | RMSE |\n|---|---|---|\n| Model | 0.76 | 0.8 |",
        embedding=[0.0, 1.0],
    )
    opls4_chunk = DocumentChunk(
        id="opls4-t", document_id="opls4", ordinal=0,
        text="| Table 2 | HFE | RMSE |\n|---|---|---|\n| Model | 0.46 | 0.9 |",
        embedding=[1.0, 0.0],
    )
    db.add_all([project, opls5, opls4, opls5_chunk, opls4_chunk])
    db.commit()


def test_metric_query_table_supplement_does_not_displace_lexical_table() -> None:
    """指标查询：全库补充不得用同等表格提权与词法路表格竞争。

    词法路由已选中 OPLS5；补充检索的 OPLS4 表 chunk 向量分与文本词元
    均更贴近问题（问题本身就在拿两者对比）。若补充路也吃
    TABLE_CONTEXT_SCORE_BOOST（+40），两个表 chunk 在最终排序中平手、
    由原始分决胜，OPLS4 会把 OPLS5 挤出 top-10——内部回归
    opls5_table_metrics 实测回归。补充只做语义兜底，不参与表格提权竞争。
    """
    db = make_session()
    seed_metric_project(db)
    service = QueryService(db)
    service.ollama = TableVectorFakeOllama()

    contexts = service._build_rag_contexts(
        "OPLS5 的表数据中，Table 2 的 HFE 与 RMSE 数值与 OPLS4 相比哪个更接近？",
        "p1",
        [PaperMatch(document=db.get(Document, "opls5"), score=8.0)],
    )

    table_contexts = [
        context for context in contexts if context.evidence_kind == "table"
    ]
    assert table_contexts, "指标查询必须产出表格证据"
    assert table_contexts[0].citation.document_id == "opls5", (
        "词法路命中的 OPLS5 表格必须排在补充路 OPLS4 之前"
    )
