from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import Document, DocumentChunk, Project, WikiPage
from app.schemas.common import QueryResponse
from app.services.ai import QueryAnswerPayload, VerificationPayload
from app.services.search import QueryService, RetrievedContext
from app.schemas.common import Citation


class FakeOllama:
    def __init__(self) -> None:
        self.last_prompt = ""
        self.payload = QueryAnswerPayload(answer_markdown="Follow-up is recommended in two weeks.", citations=[0, 1], risk_level="normal")

    def generate_structured(self, schema, *, system_prompt: str, user_prompt: str, model: str | None = None):
        self.last_prompt = user_prompt
        return self.payload

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class FakeVerifier:
    def verify_claims(self, summary, claims):
        return VerificationPayload(verdict="local-only", notes="", flagged_claim_indexes=[])


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def test_query_service_uses_wiki_page_context_and_returns_page_citation() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="Medical Case", file_name="case.md", sha256="abc", raw_path="raw/case.md", status="ready")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/medical-case",
        title="8b4b1a24d0f74a7cab2d54a0a5ebdb4d-Medical Case Summary",
        kind="source_summary",
        markdown_path="wiki/demo/sources/medical-case.md",
        markdown_content="# Medical Case Summary\n\nThe doctor recommended a follow-up in two weeks after discharge.",
        source_document_ids=["d1"],
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="Patient demographics and medication details appear here before any follow-up timing is mentioned later.",
        page_label="1",
        embedding=None,
    )
    db.add_all([project, document, wiki_page, chunk])
    db.commit()

    service = QueryService(db)
    fake_ollama = FakeOllama()
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()

    response = service.answer("demo", "When is the follow-up?", save_answer=False)

    assert isinstance(response, QueryResponse)
    assert response.answer_markdown == "Follow-up is recommended in two weeks."
    assert response.citations[0].page_slug == "sources/medical-case"
    assert response.citations[0].page_title == "Medical Case Summary"
    assert "follow-up in two weeks" in fake_ollama.last_prompt.lower()


def test_query_service_saves_query_page_when_requested() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/medical-case",
        title="Medical Case Summary",
        kind="source_summary",
        markdown_path="wiki/demo/sources/medical-case.md",
        markdown_content="# Medical Case Summary\n\nFollow-up is recommended in two weeks.",
        source_document_ids=[],
    )
    db.add_all([project, wiki_page])
    db.commit()

    service = QueryService(db)
    service.ollama = FakeOllama()
    service.verifier = FakeVerifier()

    service.answer("demo", "When is the follow-up?", save_answer=True)

    query_pages = db.query(WikiPage).filter(WikiPage.kind == "query_answer").all()
    assert len(query_pages) == 1
    assert query_pages[0].slug.startswith("queries/")
    assert "## Answer" in query_pages[0].markdown_content


def test_query_service_prefers_wiki_only_for_strong_chinese_match() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="社区慢病随访记录", file_name="case.md", sha256="abc", raw_path="raw/case.md", status="ready")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/社区慢病随访记录",
        title="社区慢病随访记录",
        kind="source_summary",
        markdown_path="wiki/demo/sources/case.md",
        markdown_content="# 社区慢病随访记录\n\n医生建议患者在3个月后进行复查，并继续当前治疗方案。",
        source_document_ids=["d1"],
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="患者基本信息、诊断和详细用药在此，后文才提到复查时间。",
        page_label="1",
        embedding=None,
    )
    db.add_all([project, document, wiki_page, chunk])
    db.commit()

    service = QueryService(db)
    fake_ollama = FakeOllama()
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()

    response = service.answer("demo", "医生建议患者多久后进行复查？", save_answer=False)

    assert response.citations
    assert response.citations[0].page_slug == "sources/社区慢病随访记录"
    assert response.citations[0].document_id is None
    assert "患者基本信息、诊断和详细用药" not in fake_ollama.last_prompt


def test_query_service_filters_irrelevant_wiki_pages_and_empty_entities() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    relevant_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/community-followup",
        title="社区慢病随访记录",
        kind="source_summary",
        markdown_path="wiki/demo/sources/community-followup.md",
        markdown_content="# 社区慢病随访记录\n\n医生建议患者在3个月后进行复查，并继续当前治疗方案。",
        source_document_ids=["d1"],
        metadata_json={"verified_claim_count": 3, "key_terms": ["复查", "高血压"]},
    )
    irrelevant_page = WikiPage(
        id="w2",
        project_id="p1",
        slug="sources/sample",
        title="Sample",
        kind="source_summary",
        markdown_path="wiki/demo/sources/sample.md",
        markdown_content="# Sample\n\nMethod A improves control stability. Method B is slower.",
        source_document_ids=["d2"],
        metadata_json={"verified_claim_count": 2, "key_terms": ["method", "control"]},
    )
    empty_entity = WikiPage(
        id="w3",
        project_id="p1",
        slug="entities/demographics",
        title="Demographics",
        kind="entity",
        markdown_path="wiki/demo/entities/demographics.md",
        markdown_content="# Demographics\n\n## Summary\nNo summary available.\n\n## Claims\n- No claims yet.",
        source_document_ids=[],
        metadata_json={"verified_claim_count": 0, "key_terms": ["demographics"]},
    )
    db.add_all([project, relevant_page, irrelevant_page, empty_entity])
    db.commit()

    service = QueryService(db)
    fake_ollama = FakeOllama()
    fake_ollama.payload = QueryAnswerPayload(answer_markdown="建议 3 个月后复查。", citations=[], risk_level="normal")
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()

    response = service.answer("demo", "医生建议患者多久后进行复查？", save_answer=False)

    assert len(response.citations) == 1
    assert response.citations[0].page_slug == "sources/community-followup"
    assert "Method A improves control stability" not in fake_ollama.last_prompt
    assert "No summary available." not in fake_ollama.last_prompt


def test_query_service_infers_citation_indexes_from_answer_markdown() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    first_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/one",
        title="One",
        kind="source_summary",
        markdown_path="wiki/demo/sources/one.md",
        markdown_content="# One\n\n第一条无关。",
        source_document_ids=["d1"],
        metadata_json={"verified_claim_count": 1, "key_terms": ["第一条"]},
    )
    second_page = WikiPage(
        id="w2",
        project_id="p1",
        slug="sources/two",
        title="Two",
        kind="source_summary",
        markdown_path="wiki/demo/sources/two.md",
        markdown_content="# Two\n\n医生建议患者在3个月后复查。",
        source_document_ids=["d2"],
        metadata_json={"verified_claim_count": 2, "key_terms": ["复查", "3个月"]},
    )
    db.add_all([project, first_page, second_page])
    db.commit()

    service = QueryService(db)
    fake_ollama = FakeOllama()
    fake_ollama.payload = QueryAnswerPayload(answer_markdown="根据资料，建议在 3 个月后复查。[1]", citations=[], risk_level="normal")
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()

    response = service.answer("demo", "多久后复查？", save_answer=False)

    assert len(response.citations) == 1
    assert response.citations[0].page_slug == "sources/two"


def test_query_service_promotes_raw_chunk_citations_to_wiki_pages_when_source_evidence_not_requested() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="社区慢病随访记录", file_name="case.md", sha256="abc", raw_path="raw/case.md", status="ready")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/community-followup",
        title="社区慢病随访记录",
        kind="source_summary",
        markdown_path="wiki/demo/sources/community-followup.md",
        markdown_content="# 社区慢病随访记录\n\n医生建议患者在3个月后进行复查，并继续当前治疗方案。",
        source_document_ids=["d1"],
        metadata_json={"verified_claim_count": 3, "key_terms": ["复查", "高血压"]},
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="患者诊断为高血压。医生建议患者在3个月后进行复查。",
        page_label="1",
        embedding=None,
    )
    db.add_all([project, document, wiki_page, chunk])
    db.commit()

    service = QueryService(db)
    fake_ollama = FakeOllama()
    fake_ollama.payload = QueryAnswerPayload(answer_markdown="建议在 3 个月后复查。", citations=[1], risk_level="normal")
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()
    service._should_use_wiki_only = lambda question, page_matches: False

    response = service.answer("demo", "医生建议患者多久后进行复查？", save_answer=False)

    assert len(response.citations) == 1
    assert response.citations[0].page_slug == "sources/community-followup"
    assert response.citations[0].chunk_id is None


# ---- New tests for PDF query validation fixes ----


def test_window_text_prioritizes_figure_references() -> None:
    """_window_text should center on Figure/Table mentions, not just any query term."""
    text = (
        "Introduction paragraph with lots of irrelevant content. "
        "More filler text here that mentions the word 'figure' in passing. "
        "Figure 1 illustrates the overall architecture of the SAC-KG pipeline. "
        "The input consists of text, instruction, and examples segments. "
        "More content continues after the figure description."
    )
    query_terms = {"figure", "1", "illustrates"}
    result = QueryService._window_text(text, query_terms, max_chars=200, question="Figure 1 展示了什么？")
    # Should contain the Figure 1 description, not just the first "figure" mention.
    assert "Figure 1" in result
    assert "SAC-KG pipeline" in result


def test_window_text_prioritizes_dataset_names() -> None:
    """_window_text should center on dataset names like OIE2016, NYT."""
    text = (
        "Background section with general information about knowledge graphs. "
        "Many approaches have been proposed. " * 5
        + "On OIE2016, our model achieves an F1 score of 74.7 and AUC of 73.2. "
        + "On NYT, the F1 score is 88.8 and AUC is 87.3. "
        + "More discussion follows." * 5
    )
    query_terms = {"oie2016", "f1", "score"}
    result = QueryService._window_text(text, query_terms, max_chars=300, question="SAC-KG 在 OIE2016 上的指标是什么？")
    # Should contain the metrics around OIE2016, not the background.
    assert "OIE2016" in result
    assert "74.7" in result


def test_window_text_falls_back_to_query_terms_when_no_figure_or_dataset() -> None:
    """When no Figure/Table/dataset anchors exist, fall back to generic term positions."""
    text = "A" * 500 + "TARGET_TERM_HERE" + "B" * 500
    query_terms = {"target_term_here"}
    result = QueryService._window_text(text, query_terms, max_chars=200)
    assert "TARGET_TERM_HERE" in result


def test_window_text_does_not_prioritize_unasked_figures() -> None:
    text = (
        "Figure 1 appears early but is unrelated to the question. "
        + "A" * 450
        + "The ablation study shows that removing the pruner causes performance degradation. "
        + "B" * 450
    )
    query_terms = {"ablation", "pruner", "performance"}
    result = QueryService._window_text(text, query_terms, max_chars=240, question="ablation studies 得出了什么结论？")

    assert "ablation study" in result
    assert "removing the pruner" in result


def test_is_figure_query_detects_figure_questions() -> None:
    assert QueryService._is_figure_query("Figure 1 展示了什么流程？")
    assert QueryService._is_figure_query("What does Fig. 3 show?")
    assert QueryService._is_figure_query("请描述图2的内容")
    assert not QueryService._is_figure_query("What is the main contribution?")


def test_is_table_query_detects_table_questions() -> None:
    assert QueryService._is_table_query("Table 2 的结果是什么？")
    assert QueryService._is_table_query("What are the results in table 5?")
    assert not QueryService._is_table_query("What is the abstract about?")


def test_is_metric_query_detects_metric_questions() -> None:
    assert QueryService._is_metric_query("SAC-KG 在 OIE2016 上的指标是什么？")
    assert QueryService._is_metric_query("What is the F1 score?")
    assert QueryService._is_metric_query("NYT AUC performance")
    assert not QueryService._is_metric_query("Who wrote this paper?")


def test_extract_figure_blocks_from_wiki_markdown() -> None:
    markdown = """
## Summary
Some text.

## Figure Notes
- Page 1: Figure 1 illustrates the SAC-KG architecture with three input components.
- Page 3: Figure 2 shows the pruner decision flow.
- Page 5: Figure 3 compares F1 across datasets.

## Related Pages
- None
"""
    blocks = QueryService._extract_figure_blocks(markdown)
    assert len(blocks) >= 2
    assert any("Figure 1" in b for b in blocks)
    assert any("SAC-KG architecture" in b for b in blocks)


def test_extract_table_blocks_from_wiki_markdown() -> None:
    markdown = """
## Tables
### Page 5
| Model | F1 | AUC |
|-------|----|-----|
| SAC-KG | 74.7 | 73.2 |

### Page 7
| Dataset | Precision | Recall |
|----------|-----------|--------|
| NYT | 88.8 | 87.3 |

## Formulas
- Page 3: Some formula here.
"""
    blocks = QueryService._extract_table_blocks(markdown)
    assert len(blocks) >= 2
    combined = "\n".join(blocks)
    assert "74.7" in combined or "SAC-KG" in combined


def test_select_citations_dedups_same_page_slug() -> None:
    """Same page_slug should appear at most 2 times with distinct excerpts."""
    from app.schemas.common import Citation
    from app.services.search import RetrievedContext

    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/test-paper",
        title="Test Paper",
        kind="source_summary",
        markdown_path="wiki/demo/sources/test-paper.md",
        markdown_content="# Test Paper\n\nContent A\n\nContent B\n\nContent C",
        source_document_ids=["d1"],
    )
    db.add_all([project, wiki_page])
    db.commit()

    service = QueryService(db)
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/test-paper", page_title="Test Paper",
                page_kind="source_summary", score=5.0, page_label="1",
                excerpt="Content A excerpt",
            ),
            prompt_text="Content A excerpt",
            score=5.0,
        ),
        RetrievedContext(
            citation=Citation(
                page_slug="sources/test-paper", page_title="Test Paper",
                page_kind="source_summary", score=4.0, page_label="1",
                excerpt="Content A excerpt",  # same excerpt → should be deduped
            ),
            prompt_text="Content A excerpt",
            score=4.0,
        ),
        RetrievedContext(
            citation=Citation(
                page_slug="sources/test-paper", page_title="Test Paper",
                page_kind="source_summary", score=3.0, page_label="2",
                excerpt="Content B different excerpt",
            ),
            prompt_text="Content B different excerpt",
            score=3.0,
        ),
        RetrievedContext(
            citation=Citation(
                page_slug="sources/test-paper", page_title="Test Paper",
                page_kind="source_summary", score=2.0, page_label="3",
                excerpt="Content C yet another excerpt",
            ),
            prompt_text="Content C yet another excerpt",
            score=2.0,
        ),
        RetrievedContext(
            citation=Citation(
                page_slug="sources/other-page", page_title="Other Page",
                page_kind="source_summary", score=1.0, page_label="1",
                excerpt="Different page content",
            ),
            prompt_text="Different page content",
            score=1.0,
        ),
    ]
    citations = service._select_citations(contexts, [0, 1, 2, 3, 4])
    # Should dedup same excerpt, keep at most 2 per page_slug.
    test_paper_citations = [c for c in citations if c.page_slug == "sources/test-paper"]
    assert len(test_paper_citations) <= 2
    # Should still include other page.
    assert any(c.page_slug == "sources/other-page" for c in citations)


def test_build_answer_constraints_figure_query_with_context() -> None:
    """When context has figure info, constraints should forbid saying 'not included'."""
    db = make_session()
    service = QueryService(db)
    ctx = type("ctx", (), {"prompt_text": "Figure 1 shows the architecture with three components."})()
    constraints = service._build_answer_constraints("Figure 1 展示了什么？", [ctx])
    assert "Do NOT say" in constraints
    assert "Figure" in constraints


def test_build_answer_constraints_dataset_query() -> None:
    """Dataset questions should get classification guardrails."""
    db = make_session()
    service = QueryService(db)
    constraints = service._build_answer_constraints("使用了哪些数据集？", [])
    assert "Benchmark datasets" in constraints or "benchmark" in constraints.lower()
    assert "case study" in constraints.lower() or "Case study" in constraints


def test_build_answer_constraints_ablation_query() -> None:
    """Ablation questions should require specific conclusions."""
    db = make_session()
    service = QueryService(db)
    constraints = service._build_answer_constraints("ablation studies 得出了什么结论？", [])
    assert "ablation" in constraints.lower()


def test_rank_blocks_prioritizes_dataset_metric_table() -> None:
    db = make_session()
    service = QueryService(db)
    blocks = [
        "### Page 4\nTable 1: Domain KG evaluation\nOpenIE 6 Precision 42.05 Recall 1.94",
        "### Page 8\nTable 5: Benchmark results\nOIE2016 F1 74.7 AUC 73.2\nNYT F1 88.8 AUC 87.3",
    ]

    ranked = service._rank_blocks("SAC-KG 在 OIE2016 或 NYT 数据集上的指标是什么？", blocks)

    assert ranked[0][0].startswith("### Page 8")
    assert "74.7" in ranked[0][0]


def test_build_contexts_adds_component_facets() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/sac-kg",
        title="SAC-KG",
        kind="source_summary",
        markdown_path="wiki/demo/sources/sac-kg.md",
        markdown_content=(
            "# SAC-KG\n\n"
            "SAC-KG is a framework.\n\n"
            "Generator extracts relations and tail entities.\n\n"
            "Verifier corrects generation errors.\n\n"
            "Pruner decides whether tail entities should grow."
        ),
        source_document_ids=[],
        metadata_json={"verified_claim_count": 3, "key_terms": ["SAC-KG", "Generator", "Verifier", "Pruner"]},
    )
    db.add_all([project, wiki_page])
    db.commit()
    service = QueryService(db)
    matches = service._search_wiki_pages("SAC-KG 的 Generator、Verifier、Pruner 分别做什么？", "p1")

    contexts = service._build_contexts("SAC-KG 的 Generator、Verifier、Pruner 分别做什么？", "p1", matches)
    prompt = "\n".join(context.prompt_text for context in contexts)

    assert "Generator extracts" in prompt
    assert "Verifier corrects" in prompt
    assert "Pruner decides" in prompt


def test_unsupported_answer_numbers_detects_numbers_missing_from_evidence() -> None:
    db = make_session()
    service = QueryService(db)
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/sac-kg",
                page_title="SAC-KG",
                page_kind="source_summary",
                score=10.0,
                excerpt="OIE2016 F1 74.7 AUC 73.2",
            ),
            prompt_text="OIE2016 F1 74.7 AUC 73.2",
            score=10.0,
        )
    ]

    unsupported = service._unsupported_answer_numbers("OIE2016 F1=74.7, NYT F1=88.8", contexts, [0])

    assert "88.8" in unsupported
    assert "74.7" not in unsupported
