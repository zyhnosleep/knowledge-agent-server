"""公式类问题的检索回归测试。

背景（2026-08-24，Forward KL 查询）：问"Forward kl 的定义是什么，公式是什么"
时，最相关的公式证据（Eq 6 D(p_T∥pS) 定义、Eq 7 JSD_β）从未进入证据包：

1. 候选生成层：`_search_source_chunks(limit=MAX_CONTEXTS=8)` 按融合分数取前 8，
   TeX 公式 chunk 与自然语言查询的向量相似度天然偏低（4.5-5.1 vs 正文 5.3+），
   公式排 12-38 位被截断在 finalize 之前。
2. 排序层：`_finalize_contexts` 的 context_sort_key 里为"公式问题"设计的
   formula_query 结构提权 +4.0 被 `if not table_query: return raw score`
   短路——非表格查询永不生效，公式证据即使进入候选也按原始分数垫底出局。

修复：公式类问题（查询含 formula/equation/公式）候选 limit 放宽到
FORMULA_SOURCE_CONTEXT_LIMIT=24；非表格查询下 formula 证据同样结构提权。
"""

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import Document
from app.schemas.common import Citation
from app.services.search import (
    FORMULA_SOURCE_CONTEXT_LIMIT,
    MAX_CONTEXTS,
    PaperMatch,
    QueryService,
    RetrievedContext,
)


@pytest.fixture(autouse=True)
def deterministic_retrieval_token_counter(monkeypatch) -> None:
    from app.services import search as search_module

    monkeypatch.setattr(
        search_module._RETRIEVAL_TOKEN_PROVIDER,
        "estimate_tokens",
        lambda text: max(1, len(text.split())),
    )


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def make_document(id: str = "opsd") -> Document:
    return Document(
        id=id,
        project_id="p1",
        title="OPSD",
        file_name=f"{id}.pdf",
        sha256=id,
        raw_path=f"raw/{id}.pdf",
        raw_text="On-Policy Self-Distillation.",
        status="ready",
    )


def _formula_context(chunk_id: str, score: float, excerpt: str) -> RetrievedContext:
    return RetrievedContext(
        citation=Citation(
            document_id="opsd",
            chunk_id=chunk_id,
            page_slug="sources/opsd",
            page_title="OPSD",
            page_kind="source_summary",
            page_label="6",
            score=score,
            excerpt=excerpt,
        ),
        prompt_text=excerpt,
        score=score,
        evidence_kind="formula",
    )


def _narrative_context(chunk_id: str, score: float, excerpt: str) -> RetrievedContext:
    return RetrievedContext(
        citation=Citation(
            document_id="opsd",
            chunk_id=chunk_id,
            page_slug="sources/opsd",
            page_title="OPSD",
            page_kind="source_summary",
            page_label="3",
            score=score,
            excerpt=excerpt,
        ),
        prompt_text=excerpt,
        score=score,
        evidence_kind=None,
    )


def test_build_rag_contexts_formula_query_widens_source_candidates(monkeypatch) -> None:
    """公式类问题必须向 _search_source_chunks 请求更宽的候选窗口。

    失败基线：公式查询也传 MAX_CONTEXTS=8，TeX 公式 chunk 排在
    12-38 位被截断，公式证据永远进不了 finalize（Forward KL 回归）。
    """
    service = QueryService(make_session())
    document = make_document()
    requested_limits: list[int] = []

    def fake_source_chunks(
        _question, _project_id, _document_ids, *, limit=MAX_CONTEXTS, **_kwargs
    ):
        requested_limits.append(limit)
        return []

    monkeypatch.setattr(service, "_search_source_chunks", fake_source_chunks)
    # 其余多路检索一律走空（聚焦候选窗口断言）
    for method in (
        "_search_document_intro_contexts",
        "_search_document_limitation_contexts",
        "_search_document_parameterization_contexts",
        "_search_document_scientific_anchor_contexts",
        "_search_claim_evidence_contexts",
        "_supplement_profile_term_contexts",
    ):
        monkeypatch.setattr(service, method, lambda *a, **k: [])
    monkeypatch.setattr(service.ollama, "embed", lambda texts: [[] for _ in texts])

    service._build_rag_contexts(
        "Forward kl的定义是什么，公式是什么",
        "p1",
        [PaperMatch(document=document, score=20, locked=True)],
        document_ids=["opsd"],
    )

    assert requested_limits and requested_limits[0] == FORMULA_SOURCE_CONTEXT_LIMIT


def test_build_rag_contexts_plain_query_keeps_default_candidate_window(monkeypatch) -> None:
    """非公式类问题保持 MAX_CONTEXTS 候选窗口（不因放宽而拖慢普通查询）。"""
    service = QueryService(make_session())
    document = make_document()
    requested_limits: list[int] = []

    def fake_source_chunks(
        _question, _project_id, _document_ids, *, limit=MAX_CONTEXTS, **_kwargs
    ):
        requested_limits.append(limit)
        return []

    monkeypatch.setattr(service, "_search_source_chunks", fake_source_chunks)
    for method in (
        "_search_document_intro_contexts",
        "_search_document_limitation_contexts",
        "_search_document_parameterization_contexts",
        "_search_document_scientific_anchor_contexts",
        "_search_claim_evidence_contexts",
        "_supplement_profile_term_contexts",
    ):
        monkeypatch.setattr(service, method, lambda *a, **k: [])
    monkeypatch.setattr(service.ollama, "embed", lambda texts: [[] for _ in texts])

    service._build_rag_contexts(
        "OPSD 方法的核心贡献是什么？",
        "p1",
        [PaperMatch(document=document, score=20, locked=True)],
        document_ids=["opsd"],
    )

    assert requested_limits and requested_limits[0] == MAX_CONTEXTS


def test_finalize_contexts_promotes_formula_for_formula_query() -> None:
    """非表格查询显式问公式时，formula 证据结构提权排最前。

    失败基线：非表格查询 sort key 直接返回原始分数，低分公式（TeX 向量
    相似度天然偏低）被高分正文挤出（Forward KL 回归：Eq 6 分 4.52 vs
    Table 3 分 5.46）。
    """
    service = QueryService(make_session())
    contexts = [
        _narrative_context("table3", 5.46, "Table 3. Comparison of divergence objectives on AIME25."),
        _formula_context(
            "eq6",
            4.52,
            r"$$D\big(p_T\|p_S\big)\big(\hat{y}\mid x\big) \triangleq \frac{1}{|\hat{y}|}\sum D(...)$$",
        ),
        _narrative_context("kl-text", 5.38, "we analyze the forward KL divergence KL(pT ∥pS) across"),
    ]

    finalized = service._finalize_contexts(
        contexts, question="Forward kl的定义是什么，公式是什么"
    )

    assert finalized[0].citation.chunk_id == "eq6"
    assert finalized[0].evidence_kind == "formula"


def test_finalize_contexts_keeps_score_order_for_plain_query() -> None:
    """普通查询（不含公式词）不提权：仍按原始分数排序，防止公式霸榜。"""
    service = QueryService(make_session())
    contexts = [
        _narrative_context("table3", 5.46, "Table 3. Comparison of divergence objectives on AIME25."),
        _formula_context(
            "eq6",
            4.52,
            r"$$D\big(p_T\|p_S\big)\big(\hat{y}\mid x\big) \triangleq \frac{1}{|\hat{y}|}\sum D(...)$$",
        ),
        _narrative_context("kl-text", 5.38, "we analyze the forward KL divergence KL(pT ∥pS) across"),
    ]

    finalized = service._finalize_contexts(
        contexts, question="OPSD 的 divergence 目标函数有哪些？"
    )

    assert [c.citation.chunk_id for c in finalized] == ["table3", "kl-text", "eq6"]
