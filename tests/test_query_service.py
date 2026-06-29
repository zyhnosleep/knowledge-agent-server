from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import Claim, Document, DocumentChunk, Project, QuestionAnswer, WikiPage
from app.schemas.common import QueryResponse
from app.services.ai import QueryAnswerPayload, VerificationPayload
from app.services.search import MAX_CONTEXTS, ExtractedMetric, PageMatch, PaperMatch, QueryService, RetrievedContext, settings
from app.schemas.common import Citation
from app.services.table_normalization import normalize_table_text


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


class SequencedFakeOllama(FakeOllama):
    def __init__(self, payloads: list[QueryAnswerPayload]) -> None:
        super().__init__()
        self.payloads = payloads
        self.prompts: list[str] = []

    def generate_structured(self, schema, *, system_prompt: str, user_prompt: str, model: str | None = None):
        self.last_prompt = user_prompt
        self.prompts.append(user_prompt)
        if len(self.payloads) > 1:
            return self.payloads.pop(0)
        return self.payloads[0]


class CountingFakeOllama(FakeOllama):
    def __init__(self) -> None:
        super().__init__()
        self.generate_calls = 0

    def generate_structured(self, schema, *, system_prompt: str, user_prompt: str, model: str | None = None):
        self.generate_calls += 1
        return QueryAnswerPayload(
            answer_markdown="The requested table values are not present in the provided context.",
            citations=[0],
            risk_level="normal",
        )


class ExplodingOllama(FakeOllama):
    def generate_structured(self, schema, *, system_prompt: str, user_prompt: str, model: str | None = None):
        raise AssertionError("LLM should not be required for deterministic scientific evidence answers")


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def make_table_document(
    *,
    id: str = "d1",
    project_id: str = "p1",
    title: str = "Knowledge graph",
    source_slug: str = "sources/knowledge-graph",
    page_label: str = "8",
    table_markdown: str,
    raw_text: str = "SAC-KG reports benchmark table metrics for OIE2016 and NYT.",
) -> Document:
    return Document(
        id=id,
        project_id=project_id,
        title=title,
        file_name=f"{id}.pdf",
        sha256=id,
        raw_path=f"raw/{id}.pdf",
        raw_text=raw_text,
        metadata_json={
            "source_slug": source_slug,
            "source_title": title,
            "document_intelligence": {
                "tables": [
                    {
                        "page_label": page_label,
                        "markdown": table_markdown,
                    }
                ]
            },
        },
        status="ready",
    )


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

    response = service._answer_wiki_first("demo", "When is the follow-up?", save_answer=False)

    assert isinstance(response, QueryResponse)
    assert response.answer_markdown == "Follow-up is recommended in two weeks."
    assert response.citations[0].page_slug == "sources/medical-case"
    assert response.citations[0].page_title == "Medical Case Summary"
    assert "follow-up in two weeks" in fake_ollama.last_prompt.lower()


def test_source_chunk_search_ignores_failed_documents() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    ready_document = Document(
        id="ready",
        project_id="p1",
        title="Ready Paper",
        file_name="ready.md",
        sha256="ready",
        raw_path="raw/ready.md",
        status="ready",
    )
    failed_document = Document(
        id="failed",
        project_id="p1",
        title="Failed Paper",
        file_name="failed.md",
        sha256="failed",
        raw_path="raw/failed.md",
        status="failed",
    )
    ready_chunk = DocumentChunk(id="ready-chunk", document_id="ready", ordinal=0, text="unrelated stable content", embedding=None)
    failed_chunk = DocumentChunk(id="failed-chunk", document_id="failed", ordinal=0, text="unique failed evidence token", embedding=None)
    db.add_all([project, ready_document, failed_document, ready_chunk, failed_chunk])
    db.commit()

    service = QueryService(db)
    service.ollama = FakeOllama()

    contexts = service._search_source_chunks("unique failed evidence token", "p1", [], limit=5)

    assert contexts == []


def test_search_wiki_pages_prioritizes_exact_source_identifier_over_body_overlap() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    target_page = WikiPage(
        id="w-target",
        project_id="p1",
        slug="sources/opls4",
        title="OPLS4",
        kind="source_summary",
        markdown_path="wiki/demo/sources/opls4.md",
        markdown_content="# OPLS4\n\nCanonical source page.",
        source_document_ids=[],
    )
    distracting_page = WikiPage(
        id="w-distractor",
        project_id="p1",
        slug="sources/opls5-force-field-development-and-validation",
        title="OPLS5 Force Field Development and Validation",
        kind="source_summary",
        markdown_path="wiki/demo/sources/opls5.md",
        markdown_content=(
            "# OPLS5\n\n"
            "OPLS3e salt bridge overstabilization acidic residue pKa bias water ions torsion sulfur FEP validation. "
            "OPLS3e salt bridge overstabilization acidic residue pKa bias water ions torsion sulfur FEP validation."
        ),
        source_document_ids=[],
    )
    db.add_all([project, target_page, distracting_page])
    db.commit()

    service = QueryService(db)
    matches = service._search_wiki_pages(
        "OPLS4",
        "p1",
        limit=2,
    )

    assert matches[0].page.slug == "sources/opls4"


def test_search_wiki_pages_does_not_skip_useful_page_with_placeholder_phrase() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/opls4-force-field-development-and-validation",
        title="OPLS4 Force Field Development and Validation",
        kind="source_summary",
        markdown_path="wiki/demo/sources/opls4.md",
        markdown_content=(
            "# OPLS4 Force Field Development and Validation\n\n"
            "## Summary\n"
            "OPLS4 addresses OPLS3e salt bridge overstabilization and acidic residue pKa bias.\n\n"
            "## Figure Notes\n"
            "No summary available."
        ),
        source_document_ids=[],
    )
    document = make_table_document(
        table_markdown=(
            "Table 5: F1 score and AUC results.\n"
            "| Model | OIE2016 |  | NYT |  |\n"
            "| --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |"
        )
    )
    db.add_all([project, document, wiki_page])
    db.commit()

    service = QueryService(db)
    matches = service._search_wiki_pages("How does OPLS4 address OPLS3e salt bridge overstabilization?", "p1", limit=2)

    assert [match.page.slug for match in matches] == ["sources/opls4-force-field-development-and-validation"]


def test_search_wiki_pages_skips_source_page_with_only_placeholder_bullets() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/empty-opls4",
        title="OPLS4 Empty Placeholder",
        kind="source_summary",
        markdown_path="wiki/demo/sources/empty-opls4.md",
        markdown_content="# OPLS4 Empty Placeholder\n\nNo summary available.\n\n- No claims yet.",
        source_document_ids=[],
    )
    document = make_table_document(
        table_markdown=(
            "Table 5: F1 score and AUC results.\n"
            "| Model | OIE2016 |  | NYT |  |\n"
            "| --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |"
        )
    )
    db.add_all([project, document, wiki_page])
    db.commit()

    service = QueryService(db)
    matches = service._search_wiki_pages("OPLS4", "p1", limit=2)

    assert matches == []


def test_draft_answer_omits_index_overview_when_contexts_exist() -> None:
    db = make_session()
    service = QueryService(db)
    fake_ollama = FakeOllama()
    service.ollama = fake_ollama
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/opls4", page_title="OPLS4", page_kind="source_summary", score=20, excerpt="OPLS4 evidence"),
            prompt_text="OPLS4 evidence",
            score=20,
        )
    ]

    service._draft_answer("What are the main OPLS4 improvements over OPLS3e?", "Index overview mentions OPLS5 and CHARMM36.", contexts)

    assert "Index overview" not in fake_ollama.last_prompt
    assert "OPLS5 and CHARMM36" not in fake_ollama.last_prompt
    assert "OPLS4 evidence" in fake_ollama.last_prompt


def test_draft_answer_windows_long_contexts_around_question_terms() -> None:
    db = make_session()
    service = QueryService(db)
    fake_ollama = FakeOllama()
    service.ollama = fake_ollama
    long_prefix = "unrelated filler " * 500
    long_suffix = "more unrelated filler " * 500
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/ff19sb", page_title="ff19SB", page_kind="source_summary", score=20, excerpt="ff19SB evidence"),
            prompt_text=long_prefix + "ff19SB uses OPC water model with amino-acid specific CMAP." + long_suffix,
            score=20,
        )
    ]

    service._draft_answer("ff19SB 涓轰粈涔堟帹鑽愬拰 OPC water model 涓€璧蜂娇鐢紵", None, contexts)

    assert "ff19SB uses OPC water model" in fake_ollama.last_prompt
    assert len(fake_ollama.last_prompt) < 5000


def test_prompt_context_text_preserves_table_citation_excerpt() -> None:
    long_prompt = "Table 1 unrelated filler\n" + ("| A | B |\n| --- | --- |\n| x | y |\n" * 200)
    excerpt = "Table 5\n| Model | F1 |\n| --- | --- |\n| SAC-KG | 88.8 |"
    context = RetrievedContext(
        citation=Citation(page_slug="sources/kg", page_title="KG", page_kind="source_summary", score=1, excerpt=excerpt),
        prompt_text=long_prompt,
        score=1,
    )

    prompt_text = QueryService._prompt_context_text("Table 5 鐨?F1 鏄灏戯紵", context)

    assert "Table 5" in prompt_text
    assert "88.8" in prompt_text
    assert len(prompt_text) <= 2400


def test_query_service_saves_query_page_when_requested() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="Medical Case",
        file_name="case.md",
        sha256="abc",
        raw_path="raw/case.md",
        metadata_json={"source_slug": "sources/medical-case"},
        status="ready",
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="Follow-up is recommended in two weeks.",
        page_label="1",
        embedding=None,
    )
    db.add_all([project, document, chunk])
    db.commit()

    service = QueryService(db)
    service.ollama = FakeOllama()
    service.verifier = FakeVerifier()

    service.answer("demo", "When is the follow-up?", save_answer=True)

    answers = db.query(QuestionAnswer).all()
    assert len(answers) == 1
    assert "Follow-up is recommended" in answers[0].answer_markdown
    query_pages = db.query(WikiPage).filter(WikiPage.kind == "query_answer").all()
    assert query_pages == []


def test_query_service_prefers_wiki_only_for_strong_chinese_match() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="Community Follow-up Record", file_name="case.md", sha256="abc", raw_path="raw/case.md", status="ready")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/community-followup-record",
        title="Community Follow-up Record",
        kind="source_summary",
        markdown_path="wiki/demo/sources/case.md",
        markdown_content="# Community Follow-up Record\n\nThe doctor recommended a follow-up in three months and continuing the current treatment plan.",
        source_document_ids=["d1"],
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="Patient demographics, diagnosis, and medication details appear here before follow-up timing is mentioned later.",
        page_label="1",
        embedding=None,
    )
    db.add_all([project, document, wiki_page, chunk])
    db.commit()

    service = QueryService(db)
    fake_ollama = FakeOllama()
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()

    response = service._answer_wiki_first("demo", "When did the doctor recommend follow-up?", save_answer=False)

    assert response.citations
    assert response.citations[0].page_slug == "sources/community-followup-record"
    assert response.citations[0].document_id is None
    assert "Patient demographics" not in fake_ollama.last_prompt


def test_query_service_filters_irrelevant_wiki_pages_and_empty_entities() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    relevant_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/community-followup",
        title="Community Follow-up Record",
        kind="source_summary",
        markdown_path="wiki/demo/sources/community-followup.md",
        markdown_content="# Community Follow-up Record\n\nThe doctor recommended a follow-up in three months and continuing the current treatment plan.",
        source_document_ids=["d1"],
        metadata_json={"verified_claim_count": 3, "key_terms": ["follow-up", "hypertension"]},
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
    fake_ollama.payload = QueryAnswerPayload(answer_markdown="Follow-up is recommended in three months.", citations=[], risk_level="normal")
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()

    response = service._answer_wiki_first("demo", "When did the doctor recommend follow-up?", save_answer=False)

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
        markdown_content="# One\n\nThe first page is unrelated.",
        source_document_ids=["d1"],
        metadata_json={"verified_claim_count": 1, "key_terms": ["first"]},
    )
    second_page = WikiPage(
        id="w2",
        project_id="p1",
        slug="sources/two",
        title="Two",
        kind="source_summary",
        markdown_path="wiki/demo/sources/two.md",
        markdown_content="# Two\n\nThe doctor recommended a follow-up in three months.",
        source_document_ids=["d2"],
        metadata_json={"verified_claim_count": 2, "key_terms": ["follow-up", "three months"]},
    )
    db.add_all([project, first_page, second_page])
    db.commit()

    service = QueryService(db)
    fake_ollama = FakeOllama()
    fake_ollama.payload = QueryAnswerPayload(answer_markdown="Based on the source, follow-up is recommended in three months [1].", citations=[], risk_level="normal")
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()

    response = service._answer_wiki_first("demo", "When is follow-up recommended?", save_answer=False)

    assert len(response.citations) == 1
    assert response.citations[0].page_slug == "sources/two"


def test_query_service_promotes_raw_chunk_citations_to_wiki_pages_when_source_evidence_not_requested() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="Community Follow-up Record", file_name="case.md", sha256="abc", raw_path="raw/case.md", status="ready")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/community-followup",
        title="Community Follow-up Record",
        kind="source_summary",
        markdown_path="wiki/demo/sources/community-followup.md",
        markdown_content="# Community Follow-up Record\n\nThe doctor recommended a follow-up in three months and continuing the current treatment plan.",
        source_document_ids=["d1"],
        metadata_json={"verified_claim_count": 3, "key_terms": ["follow-up", "hypertension"]},
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="The patient was diagnosed with hypertension. The doctor recommended follow-up in three months.",
        page_label="1",
        embedding=None,
    )
    db.add_all([project, document, wiki_page, chunk])
    db.commit()

    service = QueryService(db)
    fake_ollama = FakeOllama()
    fake_ollama.payload = QueryAnswerPayload(answer_markdown="Follow-up is recommended in three months.", citations=[1], risk_level="normal")
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()
    service._should_use_wiki_only = lambda question, page_matches: False

    response = service._answer_wiki_first("demo", "When did the doctor recommend follow-up?", save_answer=False)

    assert len(response.citations) == 1
    assert response.citations[0].page_slug == "sources/community-followup"
    assert response.citations[0].chunk_id is None


def test_rag_router_prefers_exact_opls4_over_opls5() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    opls4 = Document(
        id="opls4",
        project_id="p1",
        title="OPLS4 Force Field Development and Validation",
        file_name="opls4.pdf",
        sha256="opls4",
        raw_path="raw/opls4.pdf",
        raw_text="OPLS4 addresses OPLS3e salt bridge overstabilization and acidic residue pKa bias.",
        status="ready",
    )
    opls5 = Document(
        id="opls5",
        project_id="p1",
        title="OPLS5 Force Field Development and Validation",
        file_name="opls5.pdf",
        sha256="opls5",
        raw_path="raw/opls5.pdf",
        raw_text="OPLS5 improves OPLS4 with additional torsion and charge validation.",
        status="ready",
    )
    db.add_all([project, opls4, opls5])
    db.commit()

    matches = QueryService(db)._route_papers("How does OPLS4 address OPLS3e salt bridge overstabilization?", "p1")

    assert [match.document.id for match in matches] == ["opls4"]


def test_rag_router_does_not_persist_lazy_profile_during_query() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="opls4",
        project_id="p1",
        title="OPLS4 Force Field Development and Validation",
        file_name="opls4.pdf",
        sha256="opls4",
        raw_path="raw/opls4.pdf",
        raw_text="OPLS4 addresses protein-ligand force field validation.",
        status="ready",
    )
    db.add_all([project, document])
    db.commit()

    QueryService(db)._route_papers("OPLS4 鐨勪富瑕佹敼杩涙槸浠€涔堬紵", "p1")

    assert "paper_profile" not in (document.metadata_json or {})


def test_rag_router_filters_to_exact_alias_document_when_related_paper_repeats_alias() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    opls4 = Document(
        id="opls4",
        project_id="p1",
        title="OPLS4 Force Field Development and Validation",
        file_name="opls4.pdf",
        sha256="opls4",
        raw_path="raw/opls4.pdf",
        raw_text="OPLS4 addresses protein-ligand force field validation.",
        status="ready",
    )
    related = Document(
        id="related",
        project_id="p1",
        title="OPLS5 Related Work and Validation",
        file_name="opls5.pdf",
        sha256="related",
        raw_path="raw/opls5.pdf",
        raw_text=("OPLS5 compares against OPLS4. " * 20) + "OPLS5 adds new torsion validation.",
        status="ready",
    )
    db.add_all([project, opls4, related])
    db.commit()

    matches = QueryService(db)._route_papers("OPLS4 鐨勪富瑕佹敼杩涙槸浠€涔堬紵", "p1")

    assert [match.document.id for match in matches] == ["opls4"]


def test_rag_router_prefers_charmm36m_over_charmm36() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    charmm36 = Document(
        id="charmm36",
        project_id="p1",
        title="CHARMM36 force field",
        file_name="charmm36.pdf",
        sha256="c36",
        raw_path="raw/charmm36.pdf",
        raw_text="CHARMM36 is a protein force field.",
        status="ready",
    )
    charmm36m = Document(
        id="charmm36m",
        project_id="p1",
        title="CHARMM36m protein force field",
        file_name="charmm36m.pdf",
        sha256="c36m",
        raw_path="raw/charmm36m.pdf",
        raw_text="CHARMM36m improves intrinsically disordered protein ensembles.",
        status="ready",
    )
    db.add_all([project, charmm36, charmm36m])
    db.commit()

    matches = QueryService(db)._route_papers("CHARMM36m 瀵?IDP 閲囨牱鍋氫簡浠€涔堟敼杩涳紵", "p1")

    assert [match.document.id for match in matches] == ["charmm36m"]


def test_rag_router_does_not_match_hyphenated_alias_prefix() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    ff99sb = Document(
        id="ff99sb",
        project_id="p1",
        title="ff99SB protein force field",
        file_name="ff99SB.pdf",
        sha256="ff99",
        raw_path="raw/ff99SB.pdf",
        raw_text="ff99SB is a protein force field.",
        status="ready",
    )
    ff99sb_disp = Document(
        id="ff99sb-disp",
        project_id="p1",
        title="ff99SB-disp protein force field",
        file_name="ff99SB-disp.pdf",
        sha256="disp",
        raw_path="raw/ff99SB-disp.pdf",
        raw_text=("ff99SB-disp compares against ff99SB. " * 20) + "The dispersion model improves IDP ensembles.",
        status="ready",
    )
    db.add_all([project, ff99sb, ff99sb_disp])
    db.commit()

    matches = QueryService(db)._route_papers("ff99SB 鐨勮泲鐧借川鍔涘満缁撹鏄粈涔堬紵", "p1")

    assert [match.document.id for match in matches] == ["ff99sb"]


def test_rag_router_allows_multiple_documents_for_comparison_query() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add_all(
        [
            project,
            Document(
                id="opls4",
                project_id="p1",
                title="OPLS4 Force Field Development",
                file_name="opls4.pdf",
                sha256="opls4",
                raw_path="raw/opls4.pdf",
                raw_text="OPLS4 force field validation.",
                status="ready",
            ),
            Document(
                id="opls5",
                project_id="p1",
                title="OPLS5 Force Field Development",
                file_name="opls5.pdf",
                sha256="opls5",
                raw_path="raw/opls5.pdf",
                raw_text="OPLS5 force field validation.",
                status="ready",
            ),
        ]
    )
    db.commit()

    matches = QueryService(db)._route_papers("Please compare the main differences between OPLS4 and OPLS5.", "p1")

    assert {"opls4", "opls5"}.issubset({match.document.id for match in matches})


def test_rag_table_query_uses_document_table_evidence_not_profile_or_wiki() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="FooNet Benchmark Paper",
        file_name="foonet.pdf",
        sha256="abc",
        raw_path="raw/foonet.pdf",
        raw_text="FooNet reports benchmark metrics.",
        metadata_json={
            "source_slug": "sources/foonet",
            "document_intelligence": {
                "tables": [
                    {
                        "page_label": "7",
                        "markdown": (
                            "Table 7: FooNet results.\n"
                            "| Model | Dataset-A | Dataset-A | Dataset-B | Dataset-B |\n"
                            "| --- | --- | --- | --- | --- |\n"
                            "|  | Accuracy | F1 | Accuracy | F1 |\n"
                            "| FooNet | 91.2 | 88.4 | 84.1 | 80.6 |"
                        ),
                    }
                ]
            }
        },
        status="ready",
    )
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/wrong-wiki-foonet",
        title="FooNet wiki",
        kind="source_summary",
        markdown_path="wiki/demo/sources/foonet.md",
        markdown_content="# FooNet\n\n## Summary\nThe wiki summary should not be the RAG citation.",
        source_document_ids=["d1"],
    )
    db.add_all([project, document, wiki_page])
    db.commit()

    service = QueryService(db)
    service.ollama = FakeOllama()
    service.verifier = FakeVerifier()
    service._search_wiki_pages = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wiki-first should not run in rag mode"))
    service.ollama.payload = QueryAnswerPayload(
        answer_markdown="Table 7 reports Dataset-A Accuracy 91.2 / F1 88.4 and Dataset-B Accuracy 84.1 / F1 80.6 [0].",
        citations=[0],
        risk_level="normal",
    )

    response = service.answer("demo", "What are FooNet Accuracy and F1 on Dataset-A and Dataset-B in Table 7?", save_answer=False)

    assert response.citations
    assert response.citations[0].document_id == "d1"
    assert response.citations[0].page_slug == "sources/foonet"
    assert response.citations[0].page_label == "7"
    assert response.citations[0].excerpt.startswith("Table 7")
    assert "91.2" in response.citations[0].excerpt
    assert "paper_profile" not in service.ollama.last_prompt
    assert "wiki summary should not" not in service.ollama.last_prompt.lower()


def test_rag_captionless_document_table_citation_is_labeled_as_table_evidence() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="FooNet Benchmark Paper",
        file_name="foonet.pdf",
        sha256="abc",
        raw_path="raw/foonet.pdf",
        raw_text="FooNet reports benchmark metrics.",
        metadata_json={
            "document_intelligence": {
                "tables": [
                    {
                        "page_label": "7",
                        "markdown": (
                            "| Model | Accuracy | F1 |\n"
                            "| --- | --- | --- |\n"
                            "| FooNet | 91.2 | 88.4 |"
                        ),
                    }
                ]
            }
        },
        status="ready",
    )
    db.add_all([project, document])
    db.commit()

    contexts = QueryService(db)._build_rag_contexts(
        "What does the FooNet metric table report?",
        "p1",
        [PaperMatch(document=document, score=20, exact_alias=True)],
    )

    assert contexts
    assert contexts[0].citation.excerpt.startswith("Table evidence:")
    assert "91.2" in contexts[0].citation.excerpt


def test_charmm36m_table_query_locks_scope_and_uses_canonical_source_slug() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    charmm36 = make_table_document(
        id="charmm36",
        title="CHARMM36 force field",
        source_slug="sources/charmm36",
        table_markdown=(
            "Table 2: CHARMM36 IDP RMSD metrics.\n"
            "| Model | IDP RMSD |\n"
            "| --- | --- |\n"
            "| CHARMM36 | 9.9 |"
        ),
        raw_text="CHARMM36 force field refinement for proteins.",
    )
    charmm36.file_name = "charmm36.pdf"
    charmm36m = make_table_document(
        id="charmm36m",
        title="CHARMM36m protein force field",
        source_slug="sources/charmm36m",
        table_markdown=(
            "Table 2: CHARMM36m IDP RMSD metrics.\n"
            "| Model | IDP RMSD |\n"
            "| --- | --- |\n"
            "| CHARMM36m | 1.2 |"
        ),
        raw_text="CHARMM36m improves intrinsically disordered protein ensembles.",
    )
    charmm36m.file_name = "charmm36m.pdf"
    opls5 = make_table_document(
        id="opls5",
        title="OPLS5 Force Field Development and Validation",
        source_slug="sources/opls5",
        table_markdown=(
            "Table 2: OPLS5 unrelated metrics.\n"
            "| Model | IDP RMSD |\n"
            "| --- | --- |\n"
            "| OPLS5 | 5.5 |"
        ),
        raw_text="OPLS5 force field development and validation.",
    )
    opls5.file_name = "opls5.pdf"
    db.add_all([project, charmm36, charmm36m, opls5])
    db.commit()

    service = QueryService(db)
    service.ollama = FakeOllama()
    service.verifier = FakeVerifier()
    service.ollama.payload = QueryAnswerPayload(
        answer_markdown="Table 2 reports CHARMM36m IDP RMSD as 1.2 [0].",
        citations=[0],
        risk_level="normal",
    )

    response = service.answer("demo", "What does CHARMM36m Table 2 report for IDP RMSD?", save_answer=False)

    assert response.citations
    assert {citation.document_id for citation in response.citations} == {"charmm36m"}
    assert {citation.page_slug for citation in response.citations} == {"sources/charmm36m-force-field"}
    assert all("CHARMM36 | 9.9" not in citation.excerpt for citation in response.citations)
    assert all("OPLS5 | 5.5" not in citation.excerpt for citation in response.citations)
    assert "Table 2" in response.citations[0].excerpt
    assert "| Model | IDP RMSD |" in response.citations[0].excerpt
    assert "| CHARMM36m | 1.2 |" in response.citations[0].excerpt


def test_scientific_overview_retrieval_includes_intro_chunks_for_locked_document() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="charmm36",
        project_id="p1",
        title="CHARMM36 force field",
        file_name="charmm36.pdf",
        sha256="abc",
        raw_path="raw/charmm36.pdf",
        raw_text="CHARMM36 force field refinement.",
        metadata_json={"source_slug": "sources/charmm36"},
        status="ready",
    )
    db.add_all(
        [
            project,
            document,
            DocumentChunk(
                id="intro",
                document_id="charmm36",
                ordinal=0,
                text=(
                    "While the quality of the current CHARMM22/CMAP additive force field was demonstrated, "
                    "NMR spectroscopy and folding simulations showed helical and extended-state limitations."
                ),
                page_label="1",
                embedding=[1.0, 0.0],
            ),
            DocumentChunk(
                id="later",
                document_id="charmm36",
                ordinal=8,
                text="A later table reports side-chain parameter RMSD values.",
                page_label="31",
                embedding=[0.0, 1.0],
            ),
        ]
    )
    db.commit()

    contexts = QueryService(db)._search_document_intro_contexts(
        "CHARMM36 protein force field 修正 CHARMM22/CMAP 的什么问题？用了哪些 NMR 验证策略？",
        "p1",
        ["charmm36"],
    )

    assert contexts
    assert contexts[0].evidence_kind == "intro"
    assert "CHARMM22/CMAP" in contexts[0].citation.excerpt
    assert "NMR" in contexts[0].citation.excerpt


def test_limitation_query_retrieval_includes_radius_and_large_conformational_evidence() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="idpsff",
        project_id="p1",
        title="CHARMM36IDPSFF",
        file_name="charmm36idpsff.pdf",
        sha256="abc",
        raw_path="raw/charmm36idpsff.pdf",
        raw_text="CHARMM36IDPSFF.",
        metadata_json={"source_slug": "sources/charmm36idpsff"},
        status="ready",
    )
    db.add_all(
        [
            project,
            document,
            DocumentChunk(
                id="radius",
                document_id="idpsff",
                ordinal=6,
                text=(
                    "Comparison with a99SB- reported some deficiencies. "
                    "The radius of gyration of large disordered proteins shows remaining limitations."
                ),
                page_label="7",
                embedding=[1.0, 0.0],
            ),
            DocumentChunk(
                id="generic",
                document_id="idpsff",
                ordinal=1,
                text="General setup information.",
                page_label="2",
                embedding=[0.0, 1.0],
            ),
        ]
    )
    db.commit()

    contexts = QueryService(db)._search_document_limitation_contexts(
        "CHARMM36IDPSFF 相比 CHARMM36m/a99SB- 的定位是什么？它在哪些场景仍有不足？",
        "p1",
        ["idpsff"],
    )

    assert contexts
    assert contexts[0].evidence_kind == "limitation"
    assert "a99SB" in contexts[0].citation.excerpt
    assert "radius of gyration" in contexts[0].citation.excerpt
    assert "large disordered proteins" in contexts[0].citation.excerpt


def test_parameterization_query_retrieval_collects_distributed_ff19sb_anchors() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="ff19sb",
        project_id="p1",
        title="ff19SB",
        file_name="ff19sb.pdf",
        sha256="abc",
        raw_path="raw/ff19sb.pdf",
        raw_text="ff19SB.",
        metadata_json={"source_slug": "sources/ff19sb-amino-acid-specific-protein-backbone-parameters"},
        status="ready",
    )
    db.add_all(
        [
            project,
            document,
            DocumentChunk(
                id="validation",
                document_id="ff19sb",
                ordinal=0,
                text="To extensively validate ff19SB parameters, the authors performed a total of ~5 milliseconds MD simulations in explicit solvent.",
                page_label="1",
                embedding=[1.0, 0.0],
            ),
            DocumentChunk(
                id="cmap",
                document_id="ff19sb",
                ordinal=1,
                text="Leu CMAP was fit and applied to long nonpolar side chains. Val CMAP was fit and applied to both Val and Ile for beta-branched side chains.",
                page_label="27",
                embedding=[1.0, 0.0],
            ),
            DocumentChunk(
                id="qm",
                document_id="ff19sb",
                ordinal=2,
                text="M05-2X/6-311G** is reasonably accurate relative to MP2/cc-pVQZ at reproducing relative energy.",
                page_label="30",
                embedding=[1.0, 0.0],
            ),
            DocumentChunk(
                id="resp",
                document_id="ff19sb",
                ordinal=3,
                text="The original RESP method used charge fitting; HF/6-31G* was used for geometry optimization.",
                page_label="30",
                embedding=[1.0, 0.0],
            ),
        ]
    )
    db.commit()

    contexts = QueryService(db)._search_document_parameterization_contexts(
        "ff19SB 的参数化策略中，RESP charge fitting、QM level、CMAP 分配和验证规模有哪些具体锚点？",
        "p1",
        ["ff19sb"],
    )
    combined = " ".join(context.citation.excerpt for context in contexts)

    assert len(contexts) >= 4
    assert "RESP" in combined
    assert "HF/6-31G" in combined
    assert "M05-2X" in combined
    assert "MP2/cc-pVQZ" in combined
    assert "Leu CMAP" in combined
    assert "Val CMAP" in combined
    assert "Ile" in combined
    assert "5 milliseconds" in combined


def test_rag_contexts_include_sac_kg_claim_evidence_chunks() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="OPLS5",
        file_name="opls5.pdf",
        sha256="abc",
        raw_path="raw/opls5.pdf",
        raw_text="OPLS5 paper.",
        metadata_json={
            "source_slug": "sources/opls5",
            "source_title": "OPLS5",
            "paper_profile": {
                "title": "OPLS5",
                "aliases": ["OPLS5"],
                "key_terms": ["OPLS5", "polarizability", "metals"],
                "routing_summary": "OPLS5 adds polarizability and improves metal treatment.",
                "source_slug": "sources/opls5",
            },
        },
        status="ready",
    )
    evidence_chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=7,
        text=(
            "The Drude model incorporates intramolecular polarizability. "
            "Metal containing systems employ FlucCT and LFMM functionality."
        ),
        page_label="7",
        embedding=None,
    )
    generic_chunk = DocumentChunk(
        id="c2",
        document_id="d1",
        ordinal=1,
        text="OPLS5 overview title and author list.",
        page_label="1",
        embedding=None,
    )
    claim = Claim(
        id="claim1",
        project_id="p1",
        document_id="d1",
        subject="Drude model",
        predicate="supports",
        object_text="intramolecular polarizability and LFMM metal functionality",
        evidence_chunk_id="c1",
        confidence=0.7,
        verification_status="needs-review",
        metadata_json={"evidence_excerpt": evidence_chunk.text},
    )
    db.add_all([project, document, evidence_chunk, generic_chunk, claim])
    db.commit()

    contexts = QueryService(db)._search_claim_evidence_contexts(
        "OPLS5 如何用 Drude polarizability 和 LFMM 改善 metal 体系？",
        "p1",
        ["d1"],
    )

    assert contexts
    assert any(context.citation.chunk_id == "c1" for context in contexts)
    assert any("LFMM" in context.prompt_text for context in contexts)


def test_rag_contexts_use_profile_terms_only_for_retrieval_expansion() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="ff14SB",
        file_name="ff14sb.pdf",
        sha256="abc",
        raw_path="raw/ff14sb.pdf",
        raw_text="ff14SB paper.",
        metadata_json={
            "source_slug": "sources/ff14sb",
            "source_title": "ff14SB",
            "paper_profile": {
                "title": "ff14SB",
                "aliases": ["ff14SB"],
                "key_terms": [
                    "ff14SB",
                    *[f"NoiseTerm{index}" for index in range(40)],
                    "QM-MM",
                    "GAlib",
                    "side-chain",
                ],
                "routing_summary": "ff14SB improves side-chain and backbone fitting.",
                "source_slug": "sources/ff14sb",
            },
        },
        status="ready",
    )
    generic_chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="ff14SB improves fitting protocol details in supporting information.",
        page_label="1",
        embedding=None,
    )
    profile_term_chunk = DocumentChunk(
        id="c2",
        document_id="d1",
        ordinal=5,
        text="QM-MM refinement and GAlib population evolution were used for side-chain parameters.",
        page_label="5",
        embedding=None,
    )
    db.add_all([project, document, generic_chunk, profile_term_chunk])
    db.commit()

    contexts = QueryService(db)._build_rag_contexts(
        "ff14SB 的 fitting protocol 有什么变化？",
        "p1",
        [PaperMatch(document=document, score=20, exact_alias=True)],
    )

    assert any(context.citation.chunk_id == "c2" for context in contexts)
    assert any("GAlib" in context.citation.excerpt for context in contexts)
    assert all("paper_profile" not in context.prompt_text for context in contexts)


def test_rag_contexts_boost_rare_profile_terms() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="ff14SB",
        file_name="ff14sb.pdf",
        sha256="abc",
        raw_path="raw/ff14sb.pdf",
        raw_text="ff14SB paper.",
        metadata_json={
            "source_slug": "sources/ff14sb",
            "source_title": "ff14SB",
            "paper_profile": {
                "title": "ff14SB",
                "aliases": ["ff14SB"],
                "key_terms": ["ff14SB", "backbone", "NoiseTerm0", "NoiseTerm1", "RareMethod"],
                "routing_summary": "ff14SB fitting protocol.",
                "source_slug": "sources/ff14sb",
            },
        },
        status="ready",
    )
    chunks = [
        DocumentChunk(
            id=f"common-{index}",
            document_id="d1",
            ordinal=index,
            text="ff14SB backbone NoiseTerm0 NoiseTerm1 repeated evidence.",
            page_label=str(index),
            embedding=None,
        )
        for index in range(5)
    ]
    rare_chunk = DocumentChunk(
        id="rare",
        document_id="d1",
        ordinal=10,
        text="RareMethod changes the fitting protocol for side-chain parameters.",
        page_label="10",
        embedding=None,
    )
    db.add_all([project, document, *chunks, rare_chunk])
    db.commit()

    contexts = QueryService(db)._build_rag_contexts(
        "ff14SB 的 fitting protocol 有什么变化？",
        "p1",
        [PaperMatch(document=document, score=20, exact_alias=True)],
    )

    assert any(context.citation.chunk_id == "rare" for context in contexts)


def test_rag_contexts_add_supplemental_profile_term_evidence() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="ff19SB",
        file_name="ff19sb.pdf",
        sha256="abc",
        raw_path="raw/ff19sb.pdf",
        raw_text="ff19SB paper.",
        metadata_json={
            "source_slug": "sources/ff19sb",
            "source_title": "ff19SB",
            "paper_profile": {
                "title": "ff19SB",
                "aliases": ["ff19SB"],
                "key_terms": ["ff19SB", "OPC", "TIP3P", "CMAP"],
                "routing_summary": "ff19SB updates backbone parameters.",
                "source_slug": "sources/ff19sb",
            },
        },
        status="ready",
    )
    dominant_chunks = [
        DocumentChunk(
            id=f"dominant-{index}",
            document_id="d1",
            ordinal=index,
            text="ff19SB OPC TIP3P water model dominant evidence.",
            page_label=str(index),
            embedding=None,
        )
        for index in range(12)
    ]
    cmap_chunk = DocumentChunk(
        id="cmap",
        document_id="d1",
        ordinal=20,
        text="CMAP correction map evidence describes the amino-acid specific backbone update.",
        page_label="20",
        embedding=None,
    )
    db.add_all([project, document, *dominant_chunks, cmap_chunk])
    db.commit()

    contexts = QueryService(db)._build_rag_contexts(
        "ff19SB 的核心更新是什么？它为什么推荐和 OPC water model 一起使用？",
        "p1",
        [PaperMatch(document=document, score=20, exact_alias=True)],
    )

    assert any(context.citation.chunk_id == "cmap" for context in contexts)


def test_rag_contexts_keep_scientific_phrase_supplements() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="ff14SB",
        file_name="ff14sb.pdf",
        sha256="abc",
        raw_path="raw/ff14sb.pdf",
        raw_text="ff14SB paper.",
        metadata_json={
            "source_slug": "sources/ff14sb",
            "source_title": "ff14SB",
            "paper_profile": {
                "title": "ff14SB",
                "aliases": ["ff14SB"],
                "key_terms": ["ff14SB"],
                "routing_summary": "ff14SB mechanism update.",
                "source_slug": "sources/ff14sb",
            },
        },
        status="ready",
    )
    dominant_chunks = [
        DocumentChunk(
            id=f"dominant-{index}",
            document_id="d1",
            ordinal=index,
            text="ff14SB backbone dominant evidence.",
            page_label=str(index),
            embedding=None,
        )
        for index in range(12)
    ]
    mechanism_chunk = DocumentChunk(
        id="mechanism",
        document_id="d1",
        ordinal=20,
        text="The ff14SB update uses covalent relaxation to remove steric clashes before phase fitting.",
        page_label="20",
        embedding=None,
    )
    db.add_all([project, document, *dominant_chunks, mechanism_chunk])
    db.commit()

    service = QueryService(db)
    contexts = service._build_rag_contexts(
        "ff14SB 的 mechanism 是什么？",
        "p1",
        [PaperMatch(document=document, score=20, exact_alias=True)],
    )

    assert any(context.citation.chunk_id == "mechanism" for context in contexts)
    constraints = service._build_answer_constraints("ff14SB 的 mechanism 是什么？", contexts)
    assert "covalent relaxation" in constraints
    assert "steric clashes" in constraints


def test_scientific_anchor_contexts_match_latex_spaced_terms() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="ff99SB-disp",
        file_name="ff99sb-disp.pdf",
        sha256="abc",
        raw_path="raw/ff99sb-disp.pdf",
        raw_text="ff99SB-disp paper.",
        metadata_json={"source_slug": "sources/ff99sb-disp", "source_title": "ff99SB-disp"},
        status="ready",
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text=(
            "TIP4P-D changes the water dispersion coefficient C _ { 6 } by 50%. "
            "A related table reports \\alpha _ { \\mathrm { L } } propensity."
        ),
        page_label="2",
        embedding=None,
    )
    db.add_all([project, document, chunk])
    db.commit()

    contexts = QueryService(db)._search_document_scientific_anchor_contexts(
        "为什么 TIP4P-D 改善 IDP 构象采样？",
        "p1",
        ["d1"],
    )
    evidence = "\n".join(context.citation.excerpt for context in contexts)

    assert "C _ { 6 }" in evidence
    assert "50%" in evidence
    assert "\\alpha" in evidence


def test_scientific_anchor_contexts_keep_unique_high_value_terms() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="OPLS5",
        file_name="opls5.pdf",
        sha256="abc",
        raw_path="raw/opls5.pdf",
        raw_text="OPLS5 paper.",
        metadata_json={"source_slug": "sources/opls5", "source_title": "OPLS5"},
        status="ready",
    )
    dominant = [
        DocumentChunk(
            id=f"dominant-{index}",
            document_id="d1",
            ordinal=index,
            text="OPLS5 Drude polarizability cation-pi validation repeated evidence.",
            page_label=str(index),
            embedding=None,
        )
        for index in range(8)
    ]
    lfmm = DocumentChunk(
        id="lfmm",
        document_id="d1",
        ordinal=20,
        text="The metal workflow uses LFMM and includes MMP13 near-complex benchmark systems.",
        page_label="20",
        embedding=None,
    )
    db.add_all([project, document, *dominant, lfmm])
    db.commit()

    contexts = QueryService(db)._search_document_scientific_anchor_contexts(
        "为什么 OPLS5 对 polarizability-sensitive/cation-pi 和金属体系更准确？",
        "p1",
        ["d1"],
        limit=4,
    )

    assert any(context.citation.chunk_id == "lfmm" for context in contexts)


def test_scientific_anchor_contexts_keep_large_idp_limitation_terms() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="CHARMM36IDPSFF",
        file_name="charmm36idpsff.pdf",
        sha256="abc",
        raw_path="raw/charmm36idpsff.pdf",
        raw_text="CHARMM36IDPSFF paper.",
        metadata_json={"source_slug": "sources/charmm36idpsff", "source_title": "CHARMM36IDPSFF"},
        status="ready",
    )
    dominant = [
        DocumentChunk(
            id=f"idp-{index}",
            document_id="d1",
            ordinal=index,
            text="CHARMM36IDPSFF IDP CMAP PPII repeated evidence.",
            page_label=str(index),
            embedding=None,
        )
        for index in range(8)
    ]
    limitation = DocumentChunk(
        id="large-idp",
        document_id="d1",
        ordinal=20,
        text="Limitations remain for the radius of gyration of large disordered proteins and fast-folding proteins.",
        page_label="20",
        embedding=None,
    )
    db.add_all([project, document, *dominant, limitation])
    db.commit()

    contexts = QueryService(db)._search_document_scientific_anchor_contexts(
        "CHARMM36IDPSFF 在哪些场景仍有不足？",
        "p1",
        ["d1"],
        limit=4,
    )

    assert any(context.citation.chunk_id == "large-idp" for context in contexts)


def test_scientific_anchor_contexts_keep_expected_mechanism_terms() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="CHARMM36m",
        file_name="charmm36m.pdf",
        sha256="abc",
        raw_path="raw/charmm36m.pdf",
        raw_text="CHARMM36m paper.",
        metadata_json={"source_slug": "sources/charmm36m", "source_title": "CHARMM36m"},
        status="ready",
    )
    dominant = [
        DocumentChunk(
            id=f"cmap-{index}",
            document_id="d1",
            ordinal=index,
            text="CHARMM36m CMAP alphaL PPII repeated evidence.",
            page_label=str(index),
            embedding=None,
        )
        for index in range(8)
    ]
    steric = DocumentChunk(
        id="steric",
        document_id="d1",
        ordinal=20,
        text="The mechanism reduces a steric clash using refined Lennard-Jones parameters.",
        page_label="20",
        embedding=None,
    )
    cmap = DocumentChunk(
        id="cmap-window",
        document_id="d1",
        ordinal=21,
        text="The C36m CMAP uses w = 2kT around \\Phi = 6 0 and \\psi = 4 5.",
        page_label="21",
        embedding=None,
    )
    db.add_all([project, document, *dominant, steric, cmap])
    db.commit()

    contexts = QueryService(db)._search_document_scientific_anchor_contexts(
        "CHARMM36m 如何修正 alphaL 和 CMAP？",
        "p1",
        ["d1"],
        limit=5,
    )
    evidence = "\n".join(context.citation.excerpt for context in contexts)

    assert "steric" in evidence
    assert "Lennard-Jones" in evidence
    assert "2kT" in evidence


def test_scientific_anchor_labels_normalize_ocr_and_unicode_variants() -> None:
    labels = QueryService._scientific_anchor_labels_in_text(
        "The evidence mentions helical and extended conformations, Lennard–Jones terms, "
        "QM/MM differences, w = 2 k T, free energies of hydration, and torsional parameters."
    )

    assert "helix-coil" in labels
    assert "Lennard-Jones" in labels
    assert "QM-MM" in labels
    assert "2kT" in labels
    assert "hydration free energy" in labels
    assert "torsional" in labels


def test_scientific_anchor_excerpt_window_prioritizes_ppii_when_late_anchors_exist() -> None:
    text = (
        "Earlier context. " * 30
        + "The CMAP modification reduced the energy barrier between PPII and the right-handed a-helix region. "
        + "Middle context. " * 90
        + "Later limitations mention large disordered proteins and fast-folding proteins."
    )

    excerpt = QueryService._scientific_anchor_excerpt_window(
        text,
        ["IDP", "large disordered proteins", "PPII"],
        max_chars=500,
    )

    assert "PPII" in excerpt
    assert "large disordered proteins" not in excerpt


def test_scientific_anchor_context_keeps_ppii_before_late_limitation_anchor() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="CHARMM36IDPSFF",
        file_name="charmm36idpsff.pdf",
        sha256="abc",
        raw_path="raw/charmm36idpsff.pdf",
        raw_text="CHARMM36IDPSFF paper.",
        metadata_json={"source_slug": "sources/charmm36idpsff", "source_title": "CHARMM36IDPSFF"},
        status="ready",
    )
    chunk = DocumentChunk(
        id="ppii-late-limitation",
        document_id="d1",
        ordinal=1,
        text=(
            "CHARMM36IDPSFF originates from CHARMM36m with CMAP modifications. "
            "It reduced the energy barrier between PPII and the right-handed a-helix region. "
            + "Intervening IDP validation text. " * 90
            + "Limitations remain for large disordered proteins and fast-folding proteins."
        ),
        page_label="3",
        embedding=None,
    )
    db.add_all([project, document, chunk])
    db.commit()

    contexts = QueryService(db)._search_document_scientific_anchor_contexts(
        "CHARMM36IDPSFF 是从哪个力场发展来的？它主要针对哪类蛋白体系做了什么修改？",
        "p1",
        ["d1"],
        limit=1,
    )

    evidence = QueryService._context_evidence_text(contexts[0])
    assert "PPII" in evidence


def test_scientific_anchor_excerpt_prioritizes_early_high_value_anchor() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="CHARMM36IDPSFF",
        file_name="charmm36idpsff.pdf",
        sha256="abc",
        raw_path="raw/charmm36idpsff.pdf",
        raw_text="CHARMM36IDPSFF paper.",
        metadata_json={"source_slug": "sources/charmm36idpsff", "source_title": "CHARMM36IDPSFF"},
        status="ready",
    )
    chunk = DocumentChunk(
        id="ppii",
        document_id="d1",
        ordinal=1,
        text=(
            "The CMAP update reduces the energy barrier between PPII and the right-handed alpha-helix region. "
            + "background sentence. " * 120
            + "The same evidence later discusses IDPs and disordered proteins."
        ),
        page_label="3",
        embedding=None,
    )
    db.add_all([project, document, chunk])
    db.commit()

    contexts = QueryService(db)._search_document_scientific_anchor_contexts(
        "CHARMM36IDPSFF 相比 CHARMM36m/a99SB- 的定位是什么？",
        "p1",
        ["d1"],
        limit=1,
    )

    assert contexts
    assert "PPII" in contexts[0].prompt_text
    assert "PPII" in contexts[0].citation.excerpt


def test_scientific_anchor_excerpt_keeps_late_anchor_in_same_chunk() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="OPLS4",
        file_name="opls4.pdf",
        sha256="abc",
        raw_path="raw/opls4.pdf",
        raw_text="OPLS4 paper.",
        metadata_json={"source_slug": "sources/opls4", "source_title": "OPLS4"},
        status="ready",
    )
    chunk = DocumentChunk(
        id="glh",
        document_id="d1",
        ordinal=4,
        text=(
            "The pKa validation shows a neutral state bias and MSE over ASP and GLU. "
            + "background sentence. " * 80
            + "The starkest change is seen for GLH, where the predominant state changes with chi1 rotamers."
        ),
        page_label="4",
        embedding=None,
    )
    db.add_all([project, document, chunk])
    db.commit()

    contexts = QueryService(db)._search_document_scientific_anchor_contexts(
        "OPLS4 如何缓解 OPLS3e 的 salt bridge overstabilization 和酸性残基 pKa bias？",
        "p1",
        ["d1"],
        limit=2,
    )

    assert contexts
    assert "GLH" in contexts[0].citation.excerpt


def test_finalize_contexts_preserves_distinct_scientific_anchor_groups() -> None:
    service = QueryService(make_session())
    contexts = [
        RetrievedContext(
            citation=Citation(
                document_id="d1",
                chunk_id=f"polar-{index}",
                page_slug="sources/opls5",
                page_title="OPLS5",
                page_kind="source_summary",
                score=100 - index,
                page_label=str(index),
                excerpt="OPLS5 polarizability and cation-pi evidence.",
            ),
            prompt_text="OPLS5 polarizability and cation-pi evidence.",
            score=100 - index,
            evidence_kind="profile-term",
        )
        for index in range(12)
    ]
    contexts.extend(
        [
            RetrievedContext(
                citation=Citation(
                    document_id="d1",
                    chunk_id="drude",
                    page_slug="sources/opls5",
                    page_title="OPLS5",
                    page_kind="source_summary",
                    score=80,
                    page_label="6",
                    excerpt="The Drude parameters model intramolecular polarizability.",
                ),
                prompt_text="The Drude parameters model intramolecular polarizability.",
                score=80,
                evidence_kind="profile-term",
            ),
            RetrievedContext(
                citation=Citation(
                    document_id="d1",
                    chunk_id="lfmm",
                    page_slug="sources/opls5",
                    page_title="OPLS5",
                    page_kind="source_summary",
                    score=70,
                    page_label="14",
                    excerpt="The metal workflow uses LFMM and MMP13 near-complex benchmark systems.",
                ),
                prompt_text="The metal workflow uses LFMM and MMP13 near-complex benchmark systems.",
                score=70,
                evidence_kind="profile-term",
            ),
        ]
    )

    finalized = service._finalize_contexts(contexts)
    evidence = "\n".join(context.citation.excerpt for context in finalized)

    assert len(finalized) == MAX_CONTEXTS
    assert "Drude" in evidence
    assert "LFMM" in evidence
    assert "MMP13" in evidence


def test_finalize_contexts_prioritizes_high_value_scientific_anchors() -> None:
    service = QueryService(make_session())
    contexts = [
        RetrievedContext(
            citation=Citation(
                document_id="d1",
                chunk_id=f"generic-{index}",
                page_slug="sources/charmm36m",
                page_title="CHARMM36m",
                page_kind="source_summary",
                score=100 - index,
                page_label=str(index),
                excerpt="CHARMM36m CMAP alphaL PPII repeated profile evidence.",
            ),
            prompt_text="CHARMM36m CMAP alphaL PPII repeated profile evidence.",
            score=100 - index,
            evidence_kind="profile-term",
        )
        for index in range(12)
    ]
    contexts.extend(
        [
            RetrievedContext(
                citation=Citation(
                    document_id="d1",
                    chunk_id="steric",
                    page_slug="sources/charmm36m",
                    page_title="CHARMM36m",
                    page_kind="source_summary",
                    score=70,
                    page_label="5",
                    excerpt="The mechanism reduces a steric clash using refined Lennard-Jones parameters.",
                ),
                prompt_text="The mechanism reduces a steric clash using refined Lennard-Jones parameters.",
                score=70,
                evidence_kind="profile-term",
            ),
            RetrievedContext(
                citation=Citation(
                    document_id="d1",
                    chunk_id="2kt",
                    page_slug="sources/charmm36m",
                    page_title="CHARMM36m",
                    page_kind="source_summary",
                    score=69,
                    page_label="5",
                    excerpt="The C36m CMAP uses w = 2kT around \\Phi = 6 0 and \\psi = 4 5.",
                ),
                prompt_text="The C36m CMAP uses w = 2kT around \\Phi = 6 0 and \\psi = 4 5.",
                score=69,
                evidence_kind="profile-term",
            ),
        ]
    )

    finalized = service._finalize_contexts(contexts)
    evidence = "\n".join(context.citation.excerpt for context in finalized)

    assert "steric" in evidence
    assert "Lennard-Jones" in evidence
    assert "2kT" in evidence


def test_pka_bias_mechanism_is_not_metric_query() -> None:
    assert not QueryService._is_metric_query("OPLS4 如何缓解 OPLS3e 的 salt bridge overstabilization 和酸性残基 pKa bias？")
    assert QueryService._is_metric_query("OPLS4 的 pKa 和 sigma-hole 表格中，OPLS3e 到 OPLS4 的关键误差改善是多少？")


def test_supported_term_note_uses_citation_identity_and_anchor_terms() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/ff99sb-disp",
                page_title="ff99SB-disp",
                page_kind="source_summary",
                score=10,
                excerpt="TIP4P-D improves IDP ensembles through London dispersion and FRET/Rg validation.",
            ),
            prompt_text="",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "为什么 TIP4P-D 改善 IDP 构象采样？",
        "已有证据说明 TIP4P-D。",
        contexts,
    )

    assert "ff99SB-disp" in answer
    assert "IDP" in answer
    assert "London dispersion" in answer
    assert "FRET" in answer
    assert "Rg" in answer


def test_supported_term_note_adds_expanded_ensembles_from_expanded_disordered_states() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/ff99sb-disp",
                page_title="ff99SB-disp",
                page_kind="source_summary",
                score=10,
                excerpt="TIP4P-D simulations result in disordered states that are substantially more expanded.",
            ),
            prompt_text="",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "为什么 TIP4P-D 改善 IDP 构象采样？",
        "已有证据说明 TIP4P-D 和 disordered states。",
        contexts,
    )

    assert "expanded ensembles" in answer


def test_supported_term_note_adds_polarizability_when_supported() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/opls5",
                page_title="OPLS5",
                page_kind="source_summary",
                score=10,
                excerpt="The Drude parameters describe molecular polarizability for cation-pi systems.",
            ),
            prompt_text="",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "为什么 OPLS5 对 polarizability-sensitive/cation-pi 体系更准确？",
        "已有证据说明 Drude 和 cation。",
        contexts,
    )

    assert "polarizability" in answer


def test_supported_term_note_adds_radius_of_gyration_from_rg_symbol() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/charmm36idpsff",
                page_title="CHARMM36IDPSFF",
                page_kind="source_summary",
                score=10,
                excerpt="Disordered protein validation reports R _ { \\mathrm { g } } agreement against experiment.",
            ),
            prompt_text="",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "CHARMM36IDPSFF 在哪些场景仍有不足？",
        "已有证据说明 IDP validation。",
        contexts,
    )

    assert "radius of gyration" in answer
    assert "Rg" in answer


def test_supported_term_note_normalizes_plural_and_spaced_scientific_terms() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/ff99sb-ildn",
                page_title="ff99SB-ILDN",
                page_kind="source_summary",
                score=10,
                excerpt="IDPs show populations separated by 2 kT barriers in the validation evidence.",
            ),
            prompt_text="",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "这些机制如何影响 IDP 构象采样？",
        "已有证据说明 validation。",
        contexts,
    )

    assert "IDP" in answer
    assert "population" in answer
    assert "barrier" in answer
    assert "2kT" in answer


def test_supported_term_note_normalizes_phi_psi_split_digits() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/charmm36m",
                page_title="CHARMM36m",
                page_kind="source_summary",
                score=10,
                excerpt="The CMAP revision is localized around \\Phi = 6 0 and \\psi = 4 5.",
            ),
            prompt_text="",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "CHARMM36m 的 CMAP 修改区域是什么？",
        "已有证据说明 CMAP。",
        contexts,
    )

    assert "60" in answer
    assert "45" in answer


def test_supported_term_note_adds_supported_benchmark_atoms_from_evidence_variants() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/method-paper",
                page_title="Method Paper",
                page_kind="source_summary",
                score=10,
                excerpt=(
                    "C36m validation uses NMR observables and backbone fitting. "
                    "The data set includes 34 entries for organic liquids and charge refinements."
                ),
            ),
            prompt_text="",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "这个方法的验证和参数化依据是什么？",
        "已有证据说明 validation。",
        contexts,
    )

    for term in ("CHARMM36m", "NMR", "backbone", "34 organic liquids", "charge"):
        assert term in answer


def test_supported_term_note_adds_scientific_synonym_atoms() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/method-paper",
                page_title="Method Paper",
                page_kind="source_summary",
                score=10,
                excerpt=(
                    "The mechanism balances helical and extended conformations using QM target data. "
                    "The liquid data include free energies of hydration."
                ),
            ),
            prompt_text="",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "这些参数化证据说明什么？",
        "已有证据说明构象平衡和液相数据。",
        contexts,
    )

    assert "helix-coil" in answer
    assert "QM" in answer
    assert "hydration free energy" in answer


def test_ensure_valid_returned_citation_marker_appends_first_citation_when_missing() -> None:
    assert QueryService._ensure_valid_returned_citation_marker("已有证据说明机制。", 2).endswith(" [0]")
    assert QueryService._ensure_valid_returned_citation_marker("已有证据说明机制。[1]", 2) == "已有证据说明机制。[1]"
    assert QueryService._ensure_valid_returned_citation_marker("已有证据说明机制。", 0) == "已有证据说明机制。"


def test_finalize_contexts_keeps_more_same_source_evidence() -> None:
    service = QueryService(make_session())
    contexts = [
        RetrievedContext(
            citation=Citation(
                document_id="d1",
                chunk_id=f"c{index}",
                page_slug="sources/demo",
                page_title="Demo",
                page_kind="source_summary",
                score=10 - index,
                page_label=str(index),
                excerpt=f"distinct evidence {index}",
            ),
            prompt_text=f"distinct evidence {index}",
            score=10 - index,
        )
        for index in range(6)
    ]

    finalized = service._finalize_contexts(contexts)

    assert [context.citation.chunk_id for context in finalized] == ["c0", "c1", "c2", "c3", "c4"]


def test_scientific_rag_helper_can_return_extractive_evidence_without_llm() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="OPLS5",
        file_name="opls5.pdf",
        sha256="abc",
        raw_path="raw/opls5.pdf",
        raw_text="OPLS5 paper.",
        metadata_json={
            "source_slug": "sources/opls5",
            "source_title": "OPLS5",
            "paper_profile": {
                "title": "OPLS5",
                "aliases": ["OPLS5"],
                "key_terms": ["OPLS5", "Drude", "polarizability", "LFMM"],
                "routing_summary": "OPLS5 adds polarizability and improves metals.",
                "source_slug": "sources/opls5",
            },
        },
        status="ready",
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=7,
        text=(
            "OPLS5 uses a Drude model for intramolecular polarizability. "
            "Metal containing systems employ FlucCT and LFMM functionality."
        ),
        page_label="7",
        embedding=None,
    )
    db.add_all([project, document, chunk])
    db.commit()

    service = QueryService(db)
    service.ollama = ExplodingOllama()
    contexts = service._build_rag_contexts(
        "OPLS5 如何处理 Drude polarizability 和 LFMM metal 体系？",
        "p1",
        [PaperMatch(document=document, score=20, exact_alias=True)],
    )
    answer = service._deterministic_scientific_evidence_answer_if_supported(
        "OPLS5 如何处理 Drude polarizability 和 LFMM metal 体系？",
        contexts,
        "normal",
    )

    assert answer is not None
    assert "Drude" in answer.answer_markdown
    assert "polarizability" in answer.answer_markdown
    assert "LFMM" in answer.answer_markdown
    assert answer.citations
    assert contexts[answer.citations[0]].citation.chunk_id == "c1"


def test_draft_answer_uses_deterministic_scientific_evidence_without_llm_call() -> None:
    service = QueryService(make_session())
    service.ollama = CountingFakeOllama()
    contexts = [
        RetrievedContext(
            citation=Citation(
                document_id="d1",
                chunk_id="c1",
                page_slug="sources/opls5",
                page_title="OPLS5",
                page_kind="source_summary",
                score=10,
                excerpt="OPLS5 uses a Drude model for intramolecular polarizability and LFMM metal systems.",
            ),
            prompt_text="OPLS5 uses a Drude model for intramolecular polarizability and LFMM metal systems.",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = service._draft_answer("OPLS5 如何处理 Drude polarizability 和 LFMM metal 体系？", None, contexts)

    assert service.ollama.generate_calls == 0
    assert "Drude" in answer.answer_markdown
    assert "LFMM" in answer.answer_markdown


def test_scientific_rag_helper_appends_supported_terms_from_full_evidence() -> None:
    service = QueryService(make_session())
    long_prefix = "ff14SB fitting protocol background. " + ("intro text " * 80)
    contexts = [
        RetrievedContext(
            citation=Citation(
                document_id="d1",
                chunk_id="c1",
                page_slug="sources/ff14sb",
                page_title="ff14SB",
                page_kind="source_summary",
                score=1,
                excerpt=long_prefix,
            ),
            prompt_text=(
                long_prefix
                + "The new restraint scheme allowed covalent relaxation of steric clashes "
                + "while preserving backbone and side chain phase behavior. "
                + "Residual errors near minima are below 0.5 kcal mol."
            ),
            score=1,
        )
    ]

    answer = service._deterministic_scientific_evidence_answer_if_supported(
        "为什么 ff14SB 的 restraint/fitting 方案能改善侧链 artifact？",
        contexts,
        "normal",
    )

    assert answer is not None
    assert "covalent relaxation" in answer.answer_markdown
    assert "steric clashes" in answer.answer_markdown
    assert "侧链" in answer.answer_markdown
    assert "0.5 kcal/mol" in answer.answer_markdown
    cjk_count = sum(1 for char in answer.answer_markdown if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in answer.answer_markdown if ("a" <= char.lower() <= "z"))
    assert cjk_count / (cjk_count + latin_count) >= 0.2


def test_supported_term_note_uses_all_selected_contexts_for_profile_terms() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(document_id="d1", chunk_id="c1", page_slug="sources/ff19sb", page_title="ff19SB", page_kind="source_summary", score=5, excerpt="ff19SB OPC TIP3P helical propensity evidence."),
            prompt_text="ff19SB OPC TIP3P helical propensity evidence.",
            score=5,
        ),
        RetrievedContext(
            citation=Citation(document_id="d1", chunk_id="c2", page_slug="sources/ff19sb", page_title="ff19SB", page_kind="source_summary", score=4, excerpt="ff19SB amino-acid specific backbone update."),
            prompt_text="ff19SB amino-acid specific backbone update.",
            score=4,
        ),
        RetrievedContext(
            citation=Citation(document_id="d1", chunk_id="c3", page_slug="sources/ff19sb", page_title="ff19SB", page_kind="source_summary", score=3, excerpt="OPC water model comparison against TIP3P."),
            prompt_text="OPC water model comparison against TIP3P.",
            score=3,
        ),
        RetrievedContext(
            citation=Citation(document_id="d1", chunk_id="c4", page_slug="sources/ff19sb", page_title="ff19SB", page_kind="source_summary", score=2, excerpt="The amino-acid specific CMAP update is evaluated with QM data."),
            prompt_text="The amino-acid specific CMAP update is evaluated with QM data.",
            score=2,
        ),
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "ff19SB 的核心更新是什么？它为什么推荐和 OPC water model 一起使用？",
        "已有证据说明 ff19SB、OPC、TIP3P、amino-acid specific 和 backbone。",
        contexts,
    )

    assert "CMAP" in answer
    assert "QM" in answer


def test_supported_term_note_normalizes_spaced_half_kcal_units() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(document_id="d1", chunk_id="c1", page_slug="sources/ff99sb-ildn", page_title="ff99SB-ILDN", page_kind="source_summary", score=5, excerpt="Boltzmann population fitting keeps minima below 0 . 5 kcal mol and avoids barrier errors."),
            prompt_text="Boltzmann population fitting keeps minima below 0 . 5 kcal mol and avoids barrier errors.",
            score=5,
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "ff99SB-ILDN 为什么选择 500 K 的 Boltzmann population fitting，而不是直接拟合能量曲线或室温拟合？",
        "已有证据说明 Boltzmann、500 K、population 和 barrier。",
        contexts,
    )

    assert "0.5 kcal/mol" in answer


def test_draft_answer_fallback_for_chinese_question_is_chinese() -> None:
    service = QueryService(make_session())
    service.ollama = ExplodingOllama()
    contexts = [
        RetrievedContext(
            citation=Citation(document_id="d1", chunk_id="c1", page_slug="sources/opls5", page_title="OPLS5", page_kind="source_summary", score=1, excerpt="OPLS5 uses Drude polarizability."),
            prompt_text="OPLS5 uses Drude polarizability and LFMM metal functionality.",
            score=1,
        )
    ]

    answer = service._draft_answer("OPLS5 如何处理 Drude polarizability？", None, contexts)

    assert "根据当前检索到的原文证据" in answer.answer_markdown or "根据原文 RAG 证据" in answer.answer_markdown
    assert "Please verify" not in answer.answer_markdown
    assert answer.citations == [0]


def test_sac_kg_claim_evidence_requires_specific_query_anchors() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="OPLS5",
        file_name="opls5.pdf",
        sha256="abc",
        raw_path="raw/opls5.pdf",
        raw_text="OPLS5 paper.",
        metadata_json={"source_slug": "sources/opls5", "source_title": "OPLS5"},
        status="ready",
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=7,
        text="The Drude model incorporates intramolecular polarizability for metals.",
        page_label="7",
        embedding=None,
    )
    claim = Claim(
        id="claim1",
        project_id="p1",
        document_id="d1",
        subject="OPLS5",
        predicate="mentions",
        object_text="Drude polarizability",
        evidence_chunk_id="c1",
        confidence=0.7,
        verification_status="needs-review",
        metadata_json={},
    )
    db.add_all([project, document, chunk, claim])
    db.commit()

    contexts = QueryService(db)._search_claim_evidence_contexts("OPLS5 是什么？", "p1", ["d1"])

    assert contexts == []


def test_sac_kg_claim_evidence_rejects_cross_document_chunk_ids() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    routed = Document(
        id="routed",
        project_id="p1",
        title="OPLS5",
        file_name="opls5.pdf",
        sha256="routed",
        raw_path="raw/opls5.pdf",
        raw_text="OPLS5 paper.",
        metadata_json={"source_slug": "sources/opls5", "source_title": "OPLS5"},
        status="ready",
    )
    other = Document(
        id="other",
        project_id="p1",
        title="Other Paper",
        file_name="other.pdf",
        sha256="other",
        raw_path="raw/other.pdf",
        raw_text="Other paper.",
        metadata_json={"source_slug": "sources/other", "source_title": "Other"},
        status="ready",
    )
    foreign_chunk = DocumentChunk(
        id="foreign",
        document_id="other",
        ordinal=0,
        text="The Drude model and LFMM functionality belong to another paper.",
        page_label="9",
        embedding=None,
    )
    stale_claim = Claim(
        id="claim1",
        project_id="p1",
        document_id="routed",
        subject="Drude model",
        predicate="supports",
        object_text="LFMM metal functionality",
        evidence_chunk_id="foreign",
        confidence=0.7,
        verification_status="needs-review",
        metadata_json={},
    )
    db.add_all([project, routed, other, foreign_chunk, stale_claim])
    db.commit()

    contexts = QueryService(db)._search_claim_evidence_contexts(
        "OPLS5 如何用 Drude 和 LFMM 处理 metal 体系？",
        "p1",
        ["routed"],
    )

    assert contexts == []


def test_rag_metric_answer_retargets_conflicting_prose_numbers_to_table_values() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="FooNet Benchmark Paper",
        file_name="foonet.pdf",
        sha256="abc",
        raw_path="raw/foonet.pdf",
        raw_text="FooNet reports benchmark metrics.",
        metadata_json={
            "document_intelligence": {
                "tables": [
                    {
                        "page_label": "7",
                        "markdown": (
                            "Table 7: FooNet results.\n"
                            "| Model | Dataset-A | Dataset-A |\n"
                            "| --- | --- | --- |\n"
                            "|  | Accuracy | F1 |\n"
                            "| FooNet | 91.2 | 88.4 |"
                        ),
                    }
                ]
            }
        },
        status="ready",
    )
    prose_chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="A draft note says FooNet reaches Dataset-A Accuracy 99.9 and F1 98.8, but this is not the table.",
        page_label="2",
        embedding=None,
    )
    db.add_all([project, document, prose_chunk])
    db.commit()

    service = QueryService(db)
    service.ollama = SequencedFakeOllama(
        [
            QueryAnswerPayload(
                answer_markdown="FooNet reports Dataset-A Accuracy 99.9 and F1 98.8 [1].",
                citations=[1],
                risk_level="normal",
            ),
            QueryAnswerPayload(
                answer_markdown="The values still cannot be extracted.",
                citations=[0],
                risk_level="normal",
            ),
        ]
    )
    service.verifier = FakeVerifier()

    response = service.answer("demo", "What are FooNet Accuracy and F1 on Dataset-A in Table 7?", save_answer=False)

    assert "91.2" in response.answer_markdown
    assert "88.4" in response.answer_markdown
    assert "99.9" not in response.answer_markdown
    assert response.citations
    assert response.citations[0].excerpt.startswith("Table 7")


def test_rag_table_query_ignores_unrelated_table_source_chunk() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="FooNet Benchmark Paper",
        file_name="foonet.pdf",
        sha256="abc",
        raw_path="raw/foonet.pdf",
        raw_text="FooNet reports benchmark metrics.",
        status="ready",
    )
    unrelated_table = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text=(
            "Table 3: unrelated setup.\n"
            "| Setting | Value |\n"
            "| --- | --- |\n"
            "| Batch size | 32 |"
        ),
        page_label="3",
        embedding=None,
    )
    db.add_all([project, document, unrelated_table])
    db.commit()

    contexts = QueryService(db)._build_rag_contexts(
        "What are FooNet Accuracy and F1 on Dataset-A in Table 7?",
        "p1",
        [PaperMatch(document=document, score=20, exact_alias=True)],
    )

    assert contexts == []


def test_rag_table_query_falls_back_to_global_document_tables_when_router_misses() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wrong_document = Document(
        id="wrong",
        project_id="p1",
        title="Wrong Paper",
        file_name="wrong.pdf",
        sha256="wrong",
        raw_path="raw/wrong.pdf",
        raw_text="Wrong paper mentions FooNet but has no relevant table.",
        metadata_json={"source_slug": "sources/wrong"},
        status="ready",
    )
    right_document = make_table_document(
        id="right",
        title="Right FooNet Benchmark",
        source_slug="sources/right-foonet",
        table_markdown=(
            "Table 7: FooNet benchmark results.\n"
            "| Model | Dataset-A | Dataset-A |\n"
            "| --- | --- | --- |\n"
            "|  | Accuracy | F1 |\n"
            "| FooNet | 91.2 | 88.4 |"
        ),
    )
    db.add_all([project, wrong_document, right_document])
    db.commit()

    contexts = QueryService(db)._build_rag_contexts(
        "What are FooNet Accuracy and F1 on Dataset-A in Table 7?",
        "p1",
        [PaperMatch(document=wrong_document, score=2, exact_alias=False)],
    )

    assert contexts
    assert contexts[0].citation.document_id == "right"
    assert contexts[0].citation.page_slug == "sources/right-foonet"
    assert "91.2" in contexts[0].citation.excerpt


def test_rag_table_query_does_not_global_fallback_for_exact_routed_document() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    routed_document = Document(
        id="routed",
        project_id="p1",
        title="FooNet",
        file_name="foonet.pdf",
        sha256="routed",
        raw_path="raw/foonet.pdf",
        raw_text="FooNet paper has no relevant table.",
        metadata_json={"source_slug": "sources/foonet"},
        status="ready",
    )
    other_document = make_table_document(
        id="other",
        title="Other FooNet Benchmark",
        source_slug="sources/other-foonet",
        table_markdown=(
            "Table 7: FooNet benchmark results.\n"
            "| Model | Dataset-A | Dataset-A |\n"
            "| --- | --- | --- |\n"
            "|  | Accuracy | F1 |\n"
            "| FooNet | 91.2 | 88.4 |"
        ),
    )
    db.add_all([project, routed_document, other_document])
    db.commit()

    contexts = QueryService(db)._build_rag_contexts(
        "What are FooNet Accuracy and F1 on Dataset-A in Table 7?",
        "p1",
        [PaperMatch(document=routed_document, score=20, exact_alias=True)],
    )

    assert contexts == []


def test_route_papers_locks_subject_before_de_table_phrase() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    opls4 = make_table_document(
        id="opls4",
        title="opls4",
        source_slug="sources/opls4",
        table_markdown="Table 2.\n| Row | OPLS4 |\n| --- | --- |\n| value | 1.0 |",
        raw_text="OPLS4 force field paper.",
    )
    opls4.metadata_json["paper_profile"] = {
        "profile_version": "paper-profile-v1",
        "title": "opls4",
        "one_sentence": "OPLS4 force field paper.",
        "routing_summary": "Aliases: OPLS4.",
        "aliases": ["OPLS4"],
        "key_terms": ["OPLS4"],
        "source_slug": "sources/opls4",
    }
    opls5 = make_table_document(
        id="opls5",
        title="opls5",
        source_slug="sources/opls5",
        table_markdown="Table 2.\n| Row | OPLS4 | OPLS5 |\n| --- | --- | --- |\n| value | 0.76 | 0.46 |",
        raw_text="OPLS5 force field paper.",
    )
    opls5.metadata_json["paper_profile"] = {
        "profile_version": "paper-profile-v1",
        "title": "opls5",
        "one_sentence": "OPLS5 force field paper.",
        "routing_summary": "Aliases: OPLS5.",
        "aliases": ["OPLS5"],
        "key_terms": ["OPLS5", "OPLS4"],
        "source_slug": "sources/opls5",
    }
    db.add_all([opls4, opls5])
    db.commit()

    matches = QueryService(db)._route_papers(
        "OPLS5 的表格中，芳香小分子 HFE 相比 OPLS4 有哪些数值改善？",
        "p1",
    )

    assert [match.document.title for match in matches] == ["opls5"]


def test_route_papers_locks_primary_subject_before_comparison_phrase() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    idpsff = make_table_document(
        id="idpsff",
        title="CHARMM36IDPSFF",
        source_slug="sources/charmm36idpsff",
        table_markdown="Table 1.\n| Row | C36IDPSFF |\n| --- | --- |\n| value | IDP |",
        raw_text="CHARMM36IDPSFF was designed for IDP simulations.",
    )
    idpsff.metadata_json["paper_profile"] = {
        "profile_version": "paper-profile-v1",
        "title": "CHARMM36IDPSFF",
        "one_sentence": "CHARMM36IDPSFF paper.",
        "routing_summary": "Aliases: CHARMM36IDPSFF.",
        "aliases": ["CHARMM36IDPSFF", "C36IDPSFF"],
        "key_terms": ["CHARMM36IDPSFF", "C36IDPSFF", "CHARMM36m", "a99SB"],
        "source_slug": "sources/charmm36idpsff",
    }
    charmm36m = make_table_document(
        id="charmm36m",
        title="CHARMM36m",
        source_slug="sources/charmm36m",
        table_markdown="Table 1.\n| Row | CHARMM36m |\n| --- | --- |\n| value | globular |",
        raw_text="CHARMM36m validated peptides and IDPs.",
    )
    charmm36m.metadata_json["paper_profile"] = {
        "profile_version": "paper-profile-v1",
        "title": "CHARMM36m",
        "one_sentence": "CHARMM36m paper.",
        "routing_summary": "Aliases: CHARMM36m.",
        "aliases": ["CHARMM36m"],
        "key_terms": ["CHARMM36m", "CHARMM36IDPSFF", "a99SB"],
        "source_slug": "sources/charmm36m",
    }
    db.add_all([idpsff, charmm36m])
    db.commit()

    matches = QueryService(db)._route_papers(
        "CHARMM36IDPSFF 相比 CHARMM36m/a99SB- 的定位是什么？",
        "p1",
    )

    assert [(match.document.id, match.locked) for match in matches] == [("idpsff", True)]


def test_route_papers_locks_tip4p_d_to_ff99sb_disp() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    disp = Document(
        id="disp",
        project_id="p1",
        title="ff99SB-disp",
        file_name="ff99sb-disp.pdf",
        sha256="disp",
        raw_path="raw/ff99sb-disp.pdf",
        raw_text="TIP4P-D improves disordered protein ensembles.",
        metadata_json={},
        status="ready",
    )
    ff19 = Document(
        id="ff19",
        project_id="p1",
        title="ff19SB",
        file_name="ff19sb.pdf",
        sha256="ff19",
        raw_path="raw/ff19sb.pdf",
        raw_text="ff19SB discusses TIP3P, TIP4P-EW, and TIP4P/2005 comparisons.",
        metadata_json={},
        status="ready",
    )
    db.add_all([disp, ff19])
    db.commit()

    matches = QueryService(db)._route_papers(
        "为什么 TIP4P-D 比 TIP3P/TIP4P-EW/TIP4P/2005 更能改善 IDP 构象采样？",
        "p1",
    )

    assert [(match.document.id, match.locked) for match in matches] == [("disp", True)]


def test_route_papers_locks_chinese_primary_subject_with_bi_comparison() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    opls4 = make_table_document(
        id="opls4",
        title="OPLS4",
        source_slug="sources/opls4",
        table_markdown="Table 1.\n| Row | OPLS4 |\n| --- | --- |\n| value | 4 |",
        raw_text="OPLS4 force field paper.",
    )
    opls4.metadata_json["paper_profile"] = {
        "profile_version": "paper-profile-v1",
        "title": "OPLS4",
        "one_sentence": "OPLS4 paper.",
        "routing_summary": "Aliases: OPLS4.",
        "aliases": ["OPLS4"],
        "key_terms": ["OPLS4", "OPLS5"],
        "source_slug": "sources/opls4",
    }
    opls5 = make_table_document(
        id="opls5",
        title="OPLS5",
        source_slug="sources/opls5",
        table_markdown="Table 1.\n| Row | OPLS5 |\n| --- | --- |\n| value | 5 |",
        raw_text="OPLS5 force field paper.",
    )
    opls5.metadata_json["paper_profile"] = {
        "profile_version": "paper-profile-v1",
        "title": "OPLS5",
        "one_sentence": "OPLS5 paper.",
        "routing_summary": "Aliases: OPLS5.",
        "aliases": ["OPLS5"],
        "key_terms": ["OPLS5", "OPLS4"],
        "source_slug": "sources/opls5",
    }
    db.add_all([opls4, opls5])
    db.commit()

    matches = QueryService(db)._route_papers(
        "为什么 OPLS5 对 polarizability-sensitive/cation-pi 和金属体系比 OPLS4 更准确？",
        "p1",
    )

    assert [(match.document.id, match.locked) for match in matches] == [("opls5", True)]


def test_route_papers_keeps_explicit_compare_queries_multi_document() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    left = make_table_document(
        id="left",
        title="OPLS4",
        source_slug="sources/opls4",
        table_markdown="Table 1.\n| Row | OPLS4 |\n| --- | --- |\n| value | 4 |",
        raw_text="OPLS4 force field paper.",
    )
    left.metadata_json["paper_profile"] = {
        "profile_version": "paper-profile-v1",
        "title": "OPLS4",
        "one_sentence": "OPLS4 paper.",
        "routing_summary": "Aliases: OPLS4.",
        "aliases": ["OPLS4"],
        "key_terms": ["OPLS4", "OPLS5"],
        "source_slug": "sources/opls4",
    }
    right = make_table_document(
        id="right",
        title="OPLS5",
        source_slug="sources/opls5",
        table_markdown="Table 1.\n| Row | OPLS5 |\n| --- | --- |\n| value | 5 |",
        raw_text="OPLS5 force field paper.",
    )
    right.metadata_json["paper_profile"] = {
        "profile_version": "paper-profile-v1",
        "title": "OPLS5",
        "one_sentence": "OPLS5 paper.",
        "routing_summary": "Aliases: OPLS5.",
        "aliases": ["OPLS5"],
        "key_terms": ["OPLS5", "OPLS4"],
        "source_slug": "sources/opls5",
    }
    db.add_all([left, right])
    db.commit()

    matches = QueryService(db)._route_papers("Compare OPLS4 and OPLS5.", "p1")

    assert {match.document.id for match in matches} == {"left", "right"}
    assert not any(match.locked for match in matches)

    versus_matches = QueryService(db)._route_papers("OPLS4 versus OPLS5.", "p1")

    assert {match.document.id for match in versus_matches} == {"left", "right"}
    assert not any(match.locked for match in versus_matches)

    vs_matches = QueryService(db)._route_papers("OPLS4 vs. OPLS5 differences.", "p1")

    assert {match.document.id for match in vs_matches} == {"left", "right"}
    assert not any(match.locked for match in vs_matches)

    slash_matches = QueryService(db)._route_papers("OPLS4/OPLS5 differences.", "p1")

    assert {match.document.id for match in slash_matches} == {"left", "right"}
    assert not any(match.locked for match in slash_matches)

    and_matches = QueryService(db)._route_papers("OPLS4 and OPLS5 differences.", "p1")

    assert {match.document.id for match in and_matches} == {"left", "right"}
    assert not any(match.locked for match in and_matches)


def test_explicit_table_query_does_not_use_prose_metric_chunk_as_table_evidence() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="FooNet Benchmark Paper",
        file_name="foonet.pdf",
        sha256="abc",
        raw_path="raw/foonet.pdf",
        raw_text="FooNet reports benchmark metrics.",
        metadata_json={"source_slug": "sources/foonet"},
        status="ready",
    )
    prose_chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="The paper says Table 7 reports Accuracy and F1, and a nearby prose sentence mentions 91.2.",
        page_label="7",
        embedding=None,
    )
    db.add_all([project, document, prose_chunk])
    db.commit()

    contexts = QueryService(db)._build_rag_contexts(
        "What are FooNet Accuracy and F1 values in Table 7?",
        "p1",
        [PaperMatch(document=document, score=20, exact_alias=True)],
    )

    assert contexts == []


def test_answer_ignores_public_wiki_mode_and_uses_rag() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="Medical Case",
        file_name="case.md",
        sha256="abc",
        raw_path="raw/case.md",
        metadata_json={"source_slug": "sources/medical-case"},
        status="ready",
    )
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/medical-case",
        title="Medical Case Summary",
        kind="source_summary",
        markdown_path="wiki/demo/sources/medical-case.md",
        markdown_content="# Medical Case Summary\n\nThe doctor recommended a follow-up in two weeks after discharge.",
        source_document_ids=["d1"],
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="The source evidence says the doctor recommended a follow-up in two weeks after discharge.",
        page_label="1",
        embedding=None,
    )
    db.add_all([project, document, wiki_page, chunk])
    db.commit()

    service = QueryService(db)
    service.ollama = FakeOllama()
    service.verifier = FakeVerifier()
    service._answer_wiki_first = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("public answer must not use wiki mode"))
    old_mode = settings.query_mode
    settings.query_mode = "wiki"
    try:
        response = service.answer("demo", "When is the follow-up?", save_answer=False)
    finally:
        settings.query_mode = old_mode

    assert response.citations
    assert response.citations[0].page_slug == "sources/medical-case"
    assert response.citations[0].document_id == "d1"


def test_rag_returns_no_evidence_instead_of_falling_back_to_wiki() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/wiki-only",
        title="Wiki Only",
        kind="source_summary",
        markdown_path="wiki/demo/sources/wiki-only.md",
        markdown_content="# Wiki Only\n\nThis wiki page should not be used by RAG.",
        source_document_ids=[],
    )
    db.add_all([project, wiki_page])
    db.commit()

    service = QueryService(db)
    service.ollama = FakeOllama()
    service.verifier = FakeVerifier()
    service._answer_wiki_first = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RAG must not fall back to wiki"))

    response = service.answer("demo", "What does the wiki-only page say?", save_answer=False)

    assert "No supporting evidence" in response.answer_markdown
    assert response.citations == []


def test_rag_save_answer_does_not_write_wiki_query_page() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="FooNet Benchmark Paper",
        file_name="foonet.pdf",
        sha256="abc",
        raw_path="raw/foonet.pdf",
        raw_text="FooNet reports benchmark metrics.",
        metadata_json={"source_slug": "sources/foonet"},
        status="ready",
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="FooNet reports benchmark metrics in the source evidence.",
        page_label="1",
        embedding=None,
    )
    db.add_all([project, document, chunk])
    db.commit()

    service = QueryService(db)
    service.ollama = FakeOllama()
    service.verifier = FakeVerifier()
    service._save_query_page = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RAG save_answer must not write wiki pages"))

    response = service.answer("demo", "What does FooNet report?", save_answer=True)

    assert response.citations
    assert db.query(WikiPage).count() == 0


def test_rag_citation_source_fields_come_from_document_metadata_not_wiki() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="FooNet Benchmark Paper",
        file_name="foonet.pdf",
        sha256="abc",
        raw_path="raw/foonet.pdf",
        raw_text="FooNet reports benchmark metrics.",
        metadata_json={"source_slug": "sources/foonet", "source_title": "FooNet Source"},
        status="ready",
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="FooNet reports benchmark metrics in the source evidence.",
        page_label="1",
        embedding=None,
    )
    wrong_wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/wrong-wiki-source",
        title="Wrong Wiki Source",
        kind="source_summary",
        markdown_path="wiki/demo/sources/wrong.md",
        markdown_content="# Wrong\n\nThis page must not define RAG source identity.",
        source_document_ids=["d1"],
    )
    db.add_all([project, document, chunk, wrong_wiki_page])
    db.commit()

    response = QueryService(db)._search_source_chunks("What does FooNet report?", "p1", ["d1"], limit=1)

    assert response
    assert response[0].citation.page_slug == "sources/foonet"
    assert response[0].citation.page_title == "FooNet Source"


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
    result = QueryService._window_text(text, query_terms, max_chars=200, question="Figure 1 灞曠ず浜嗕粈涔堬紵")
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
    result = QueryService._window_text(text, query_terms, max_chars=300, question="SAC-KG 鍦?OIE2016 涓婄殑鎸囨爣鏄粈涔堬紵")
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
    result = QueryService._window_text(text, query_terms, max_chars=240, question="ablation studies 寰楀嚭浜嗕粈涔堢粨璁猴紵")

    assert "ablation study" in result
    assert "removing the pruner" in result


def test_is_figure_query_detects_figure_questions() -> None:
    assert QueryService._is_figure_query("Figure 1 灞曠ず浜嗕粈涔堟祦绋嬶紵")
    assert QueryService._is_figure_query("What does Fig. 3 show?")
    assert QueryService._is_figure_query("Please describe Figure 2.")
    assert not QueryService._is_figure_query("What is the main contribution?")


def test_is_table_query_detects_table_questions() -> None:
    assert QueryService._is_table_query("Table 2 鐨勭粨鏋滄槸浠€涔堬紵")
    assert QueryService._is_table_query("ff99SB-ILDN Table I lists which modified torsions?")
    assert QueryService._is_table_query("OPLS5 \u7684\u8868\u683c\u4e2d\u6709\u54ea\u4e9b\u6570\u503c\uff1f")
    assert QueryService._is_table_query("What are the results in table 5?")
    assert not QueryService._is_table_query("What is the abstract about?")


def test_is_metric_query_detects_metric_questions() -> None:
    assert QueryService._is_metric_query("SAC-KG 鍦?OIE2016 涓婄殑鎸囨爣鏄粈涔堬紵")
    assert QueryService._is_metric_query("OPLS5 \u7684 binding RMSE \u76f8\u6bd4 OPLS4 \u6709\u54ea\u4e9b\u6570\u503c\u6539\u5584\uff1f")
    assert QueryService._is_metric_query("What is the F1 score?")
    assert QueryService._is_metric_query("NYT AUC performance")
    assert not QueryService._is_metric_query("Who wrote this paper?")
    assert not QueryService._is_metric_query("ff19SB 涓轰粈涔堟帹鑽愬拰 OPC water model 涓€璧蜂娇鐢紵")


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
                excerpt="Content A excerpt",  # same excerpt 鈫?should be deduped
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
    constraints = service._build_answer_constraints("Figure 1 灞曠ず浜嗕粈涔堬紵", [ctx])
    assert "Do NOT say" in constraints
    assert "Figure" in constraints


def test_context_evidence_text_handles_context_without_citation() -> None:
    ctx = type("ctx", (), {"prompt_text": "Figure 1 shows the architecture with three components."})()

    evidence = QueryService._context_evidence_text(ctx)  # type: ignore[arg-type]

    assert evidence == "Figure 1 shows the architecture with three components."


def test_coverage_citation_indexes_uses_citation_excerpt() -> None:
    service = QueryService(make_session())
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/kg",
                page_title="KG",
                page_kind="source_summary",
                score=1,
                excerpt="The Generator module extracts relations.",
            ),
            prompt_text="Overview text without the facet.",
            score=1,
        )
    ]

    indexes = service._coverage_citation_indexes("SAC-KG 鐨?Generator 鍋氫粈涔堬紵", contexts)

    assert indexes == [0]


def test_build_answer_constraints_dataset_query() -> None:
    """Dataset questions should get classification guardrails."""
    db = make_session()
    service = QueryService(db)
    constraints = service._build_answer_constraints("Which datasets were used?", [])
    assert "Benchmark datasets" in constraints or "benchmark" in constraints.lower()
    assert "case study" in constraints.lower() or "Case study" in constraints


def test_build_answer_constraints_ablation_query() -> None:
    """Ablation questions should require specific conclusions."""
    db = make_session()
    service = QueryService(db)
    constraints = service._build_answer_constraints("ablation studies 寰楀嚭浜嗕粈涔堢粨璁猴紵", [])
    assert "ablation" in constraints.lower()


def test_rank_blocks_prioritizes_dataset_metric_table() -> None:
    db = make_session()
    service = QueryService(db)
    blocks = [
        "### Page 4\nTable 1: Domain KG evaluation\nOpenIE 6 Precision 42.05 Recall 1.94",
        "### Page 8\nTable 5: Benchmark results\nOIE2016 F1 74.7 AUC 73.2\nNYT F1 88.8 AUC 87.3",
    ]

    ranked = service._rank_blocks("SAC-KG 鍦?OIE2016 鎴?NYT 鏁版嵁闆嗕笂鐨勬寚鏍囨槸浠€涔堬紵", blocks)

    assert ranked[0][0].startswith("### Page 8")
    assert "74.7" in ranked[0][0]


def test_metric_query_repairs_false_missing_answer_when_table_context_exists() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/knowledge-graph",
        title="Knowledge graph",
        kind="source_summary",
        markdown_path="wiki/demo/sources/knowledge-graph.md",
        markdown_content=(
            "# Knowledge graph\n\n"
            "## Summary\n"
            "SAC-KG is a KG construction framework.\n\n"
            "## Tables\n"
            "### Page 8\n"
            "Table 5: F1 score and AUC results on OIE2016, WEB, NYT, and PENN datasets.\n"
            "| Model | OIE2016 |  | WEB |  | NYT |  | PENN |  |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC | F1 | AUC | F1 | AUC |\n"
            "| OpenIE 6 (2020) | 55.3 | 61.1 | 61.1 | 64.9 | 30.7 | 55.2 | 54.2 | 63.1 |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 96.6 | 95.7 | 88.8 | 87.3 | 91.1 | 90.1 |\n"
        ),
        source_document_ids=["d1"],
        metadata_json={"verified_claim_count": 5, "key_terms": ["SAC-KG", "OIE2016", "NYT", "Table 5"]},
    )
    document = make_table_document(
        table_markdown=(
            "Table 5: F1 score and AUC results on OIE2016, WEB, NYT, and PENN datasets.\n"
            "| Model | OIE2016 |  | WEB |  | NYT |  | PENN |  |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC | F1 | AUC | F1 | AUC |\n"
            "| OpenIE 6 (2020) | 55.3 | 61.1 | 61.1 | 64.9 | 30.7 | 55.2 | 54.2 | 63.1 |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 96.6 | 95.7 | 88.8 | 87.3 | 91.1 | 90.1 |"
        )
    )
    db.add_all([project, document, wiki_page])
    db.commit()

    service = QueryService(db)
    fake_ollama = SequencedFakeOllama(
        [
            QueryAnswerPayload(
                answer_markdown="The exact numeric values are not included in the provided text excerpts.",
                citations=[1],
                risk_level="normal",
            ),
            QueryAnswerPayload(
                answer_markdown="Table 5 reports SAC-KG ChatGPT at OIE2016 F1 74.7 / AUC 73.2 and NYT F1 88.8 / AUC 87.3 [0].",
                citations=[0],
                risk_level="normal",
            ),
        ]
    )
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()

    response = service.answer("demo", "SAC-KG 鍦?OIE2016 鎴?NYT 鏁版嵁闆嗕笂鐨勬寚鏍囨槸浠€涔堬紵", save_answer=False)

    assert "74.7" in response.answer_markdown
    assert "88.8" in response.answer_markdown
    assert response.citations
    assert "88.8" in response.citations[0].excerpt
    assert len(fake_ollama.prompts) == 0


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
    matches = service._search_wiki_pages("SAC-KG 鐨?Generator銆乂erifier銆丳runer 鍒嗗埆鍋氫粈涔堬紵", "p1")

    contexts = service._build_contexts("SAC-KG 鐨?Generator銆乂erifier銆丳runer 鍒嗗埆鍋氫粈涔堬紵", "p1", matches)
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


def test_unsupported_answer_numbers_uses_citation_excerpt_as_evidence() -> None:
    service = QueryService(make_session())
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/sac-kg",
                page_title="SAC-KG",
                page_kind="source_summary",
                score=10.0,
                excerpt="Table 5: OIE2016 F1 74.7 AUC 73.2",
            ),
            prompt_text="Table 5 is mentioned in prose, but the windowed prompt omitted the values.",
            score=10.0,
        )
    ]

    unsupported = service._unsupported_answer_numbers("OIE2016 F1=74.7", contexts, [0])

    assert unsupported == set()


def test_unsupported_answer_numbers_accepts_spaced_digits_in_profile_contexts() -> None:
    service = QueryService(make_session())
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/charmm36m",
                page_title="CHARMM36m",
                page_kind="source_summary",
                score=10.0,
                excerpt="The CMAP region is around \\Phi = 6 0 and \\psi = 4 5.",
            ),
            prompt_text="",
            score=10.0,
            evidence_kind="profile-term",
        )
    ]

    unsupported = service._unsupported_answer_numbers("The answer mentions 60 and 45.", contexts, [0])

    assert unsupported == set()


def test_repair_unsupported_numeric_answer_uses_supported_pairs_without_name_error() -> None:
    db = make_session()
    service = QueryService(db)
    fake_ollama = FakeOllama()
    fake_ollama.payload = QueryAnswerPayload(answer_markdown="Only OIE2016 F1 74.7 is supported.", citations=[0], risk_level="normal")
    service.ollama = fake_ollama
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
    draft = QueryAnswerPayload(answer_markdown="OIE2016 F1 74.7 and NYT F1 88.8.", citations=[0], risk_level="normal")

    repaired = service._repair_unsupported_numeric_answer("OIE2016 鍜?NYT 鎸囨爣鏄粈涔堬紵", None, contexts, draft, [0])

    assert "88.8" not in repaired.answer_markdown
    assert repaired.citations == [0]


def test_choose_citation_indexes_unions_payload_inferred_and_facets() -> None:
    db = make_session()
    service = QueryService(db)
    contexts = [
        RetrievedContext(citation=Citation(page_slug="s", page_title="S", page_kind="source_summary", score=1, excerpt="SAC-KG"), prompt_text="SAC-KG overview", score=1),
        RetrievedContext(citation=Citation(page_slug="s", page_title="S", page_kind="source_summary", score=1, excerpt="Generator"), prompt_text="Generator extracts relations.", score=1),
        RetrievedContext(citation=Citation(page_slug="s", page_title="S", page_kind="source_summary", score=1, excerpt="Verifier"), prompt_text="Verifier checks triples.", score=1),
        RetrievedContext(citation=Citation(page_slug="s", page_title="S", page_kind="source_summary", score=1, excerpt="Pruner"), prompt_text="Pruner controls growth.", score=1),
        RetrievedContext(citation=Citation(page_slug="s", page_title="S", page_kind="source_summary", score=1, excerpt="Table 2"), prompt_text="Table 2 reports ablation.", score=1),
    ]
    payload = QueryAnswerPayload(answer_markdown="Generator [1], Verifier [2], Pruner [3], Table 2 [4]", citations=[0], risk_level="normal")

    indexes = service._choose_citation_indexes("What do SAC-KG Generator, Verifier, and Pruner do? Please cite Table 2.", payload, contexts)
    renumbered = service._renumber_answer_citations(payload.answer_markdown, indexes)

    assert indexes[:5] == [0, 1, 2, 3, 4]
    assert "[4]" in renumbered


def test_build_contexts_table_first_skips_summary_and_raw_when_table_exists() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="Knowledge graph", file_name="kg.pdf", sha256="abc", raw_path="raw/kg.pdf", status="ready")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/knowledge-graph",
        title="Knowledge graph",
        kind="source_summary",
        markdown_path="wiki/demo/sources/knowledge-graph.md",
        markdown_content=(
            "# Knowledge graph\n\n"
            "## Summary\nThis summary mentions OIE2016 but has no metrics.\n\n"
            "## Tables\n### Page 8\n"
            "Table 5: Benchmark results.\n"
            "| Model | OIE2016 |  | NYT |  |\n"
            "| --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |\n"
        ),
        source_document_ids=["d1"],
        metadata_json={"verified_claim_count": 5, "key_terms": ["OIE2016", "NYT", "Table 5"]},
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="Raw source chunk should not be used when table context exists.",
        page_label="8",
        embedding=None,
    )
    db.add_all([project, document, wiki_page, chunk])
    db.commit()

    service = QueryService(db)
    matches = service._search_wiki_pages("What are OIE2016 and NYT F1/AUC metrics?", "p1")
    contexts = service._build_contexts("What are OIE2016 and NYT F1/AUC metrics?", "p1", matches)

    assert contexts
    assert all("Table 5" in context.prompt_text for context in contexts)
    assert all("This summary mentions" not in context.citation.excerpt for context in contexts)
    assert all(context.citation.document_id is None for context in contexts)
    assert "88.8" in contexts[0].citation.excerpt


def test_build_contexts_scans_wiki_tables_when_page_matches_are_empty() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/knowledge-graph",
        title="Knowledge graph",
        kind="source_summary",
        markdown_path="wiki/demo/sources/knowledge-graph.md",
        markdown_content=(
            "# Knowledge graph\n\n"
            "## Summary\n"
            "The summary discusses benchmarks but does not reproduce the values.\n\n"
            "## Tables\n"
            "### Page 8\n"
            "Table 5: F1 score and AUC results on OIE2016, WEB, NYT, and PENN datasets.\n"
            "| Model | OIE2016 |  | WEB |  | NYT |  | PENN |  |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC | F1 | AUC | F1 | AUC |\n"
            "| OpenIE 6 (2020) | 55.3 | 61.1 | 61.1 | 64.9 | 30.7 | 55.2 | 54.2 | 63.1 |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 96.6 | 95.7 | 88.8 | 87.3 | 91.1 | 90.1 |\n"
        ),
        source_document_ids=[],
        metadata_json={"verified_claim_count": 5, "key_terms": ["SAC-KG", "OIE2016", "NYT", "Table 5"]},
    )
    document = make_table_document(
        table_markdown=(
            "Table 5: F1 score and AUC results on OIE2016, WEB, NYT, and PENN datasets.\n"
            "| Model | OIE2016 |  | WEB |  | NYT |  | PENN |  |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC | F1 | AUC | F1 | AUC |\n"
            "| OpenIE 6 (2020) | 55.3 | 61.1 | 61.1 | 64.9 | 30.7 | 55.2 | 54.2 | 63.1 |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 96.6 | 95.7 | 88.8 | 87.3 | 91.1 | 90.1 |"
        )
    )
    db.add_all([project, document, wiki_page])
    db.commit()

    service = QueryService(db)
    contexts = service._build_contexts(
        "SAC-KG 鍦?OIE2016 鎴?NYT 鏁版嵁闆嗕笂鐨勬寚鏍囨槸浠€涔堬紵",
        "p1",
        [],
    )

    assert contexts
    assert contexts[0].citation.page_slug == "sources/knowledge-graph"
    assert contexts[0].citation.excerpt.startswith("Table 5")
    assert "74.7" in contexts[0].prompt_text
    assert "88.8" in contexts[0].prompt_text


def test_build_contexts_global_table_scan_requires_query_relevance() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/unrelated",
        title="Unrelated",
        kind="source_summary",
        markdown_path="wiki/demo/sources/unrelated.md",
        markdown_content=(
            "# Unrelated\n\n"
            "## Tables\n"
            "### Page 2\n"
            "Table 9: Unrelated benchmark.\n"
            "| Model | F1 |\n"
            "| --- | --- |\n"
            "| OtherModel | 55.1 |\n"
        ),
        source_document_ids=[],
    )
    db.add_all([project, wiki_page])
    db.commit()

    service = QueryService(db)
    contexts = service._build_contexts("What are FooNet Accuracy values on Dataset-A?", "p1", [])

    assert contexts == []


def test_global_table_scan_rejects_weak_metric_overlap_without_requested_entity_or_dataset() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/unrelated",
        title="Unrelated",
        kind="source_summary",
        markdown_path="wiki/demo/sources/unrelated.md",
        markdown_content=(
            "# Unrelated\n\n"
            "## Tables\n"
            "### Page 4\n"
            "Table 4: Generic benchmark with common metric words.\n"
            "| Model | PubMedQA |  | BioASQ |  |\n"
            "| --- | --- | --- | --- | --- |\n"
            "|  | Accuracy | F1 | Accuracy | F1 |\n"
            "| OtherModel | 80.1 | 74.4 | 70.5 | 66.9 |\n"
        ),
        source_document_ids=[],
    )
    db.add_all([project, wiki_page])
    db.commit()

    service = QueryService(db)
    contexts = service._build_contexts("What are FooNet Accuracy and F1 values on Dataset-A?", "p1", [])

    assert contexts == []


def test_build_contexts_global_table_scan_uses_generic_dataset_anchors() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/foo",
        title="Foo",
        kind="source_summary",
        markdown_path="wiki/demo/sources/foo.md",
        markdown_content=(
            "# Foo\n\n"
            "## Tables\n"
            "### Page 2\n"
            "Table 1: Unrelated benchmark.\n"
            "| Model | F1 |\n"
            "| --- | --- |\n"
            "| OtherModel | 55.1 |\n"
            "### Page 7\n"
            "Table 7: FooNet benchmark results.\n"
            "| Model | Dataset-A |  | Dataset-B |  |\n"
            "| --- | --- | --- | --- | --- |\n"
            "|  | Accuracy | F1 | Accuracy | F1 |\n"
            "| BaselineNet | 70.1 | 61.4 | 68.2 | 59.7 |\n"
            "| FooNet | 91.2 | 89.5 | 87.6 | 85.4 |\n"
        ),
        source_document_ids=[],
    )
    db.add_all([project, wiki_page])
    db.commit()

    service = QueryService(db)
    contexts = service._build_contexts("What are FooNet Accuracy and F1 on Dataset-A and Dataset-B?", "p1", [])

    assert contexts
    assert contexts[0].citation.excerpt.startswith("Table 7")
    assert "91.2" in contexts[0].prompt_text
    assert "85.4" in contexts[0].prompt_text


def test_matched_page_unrelated_table_does_not_block_global_requested_table() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    matched_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/overview",
        title="SAC-KG Overview",
        kind="source_summary",
        markdown_path="wiki/demo/sources/overview.md",
        markdown_content=(
            "# SAC-KG Overview\n\n"
            "## Summary\n"
            "This page discusses SAC-KG and mentions OIE2016 and NYT, but its table is unrelated.\n\n"
            "## Tables\n"
            "### Page 2\n"
            "Table 1: Training configuration.\n"
            "| Setting | Value |\n"
            "| --- | --- |\n"
            "| Batch size | 32 |\n"
        ),
        source_document_ids=[],
        metadata_json={"verified_claim_count": 5, "key_terms": ["SAC-KG", "OIE2016", "NYT"]},
    )
    requested_table_page = WikiPage(
        id="w2",
        project_id="p1",
        slug="sources/benchmark",
        title="Benchmark",
        kind="source_summary",
        markdown_path="wiki/demo/sources/benchmark.md",
        markdown_content=(
            "# Benchmark\n\n"
            "## Tables\n"
            "### Page 8\n"
            "Table 5: F1 score and AUC results on OIE2016 and NYT datasets.\n"
            "| Model | OIE2016 |  | NYT |  |\n"
            "| --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |\n"
        ),
        source_document_ids=[],
    )
    db.add_all([project, matched_page, requested_table_page])
    db.commit()

    service = QueryService(db)
    matches = [PageMatch(page=matched_page, score=20.0)]
    contexts = service._build_contexts("What are SAC-KG metrics on OIE2016 and NYT?", "p1", matches)

    assert contexts
    assert contexts[0].citation.page_slug == "sources/benchmark"
    assert "74.7" in contexts[0].prompt_text
    assert "Batch size" not in contexts[0].prompt_text


def test_global_table_scan_requires_non_table_anchor_when_table_number_is_not_unique() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wrong_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/wrong",
        title="Wrong",
        kind="source_summary",
        markdown_path="wiki/demo/sources/wrong.md",
        markdown_content=(
            "# Wrong\n\n"
            "## Tables\n"
            "### Page 8\n"
            "Table 5: Another model results on unrelated datasets.\n"
            "| Model | Dataset-X |  |\n"
            "| --- | --- | --- |\n"
            "|  | F1 | AUC |\n"
            "| OtherModel | 60.1 | 58.2 |\n"
        ),
        source_document_ids=[],
    )
    right_page = WikiPage(
        id="w2",
        project_id="p1",
        slug="sources/right",
        title="Right",
        kind="source_summary",
        markdown_path="wiki/demo/sources/right.md",
        markdown_content=(
            "# Right\n\n"
            "## Tables\n"
            "### Page 8\n"
            "Table 5: F1 score and AUC results on OIE2016 and NYT datasets.\n"
            "| Model | OIE2016 |  | NYT |  |\n"
            "| --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |\n"
        ),
        source_document_ids=[],
    )
    db.add_all([project, wrong_page, right_page])
    db.commit()

    service = QueryService(db)
    contexts = service._build_contexts("What are SAC-KG Table 5 F1/AUC metrics on OIE2016 and NYT?", "p1", [])

    assert contexts
    assert contexts[0].citation.page_slug == "sources/right"
    assert all("OtherModel" not in context.prompt_text for context in contexts)


def test_metric_query_without_structured_table_does_not_fall_back_to_prose_source_chunk() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="KG", file_name="kg.pdf", sha256="abc", raw_path="raw/kg.pdf", status="ready")
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text="As shown in Table 5, SAC-KG performs well on OIE2016 and NYT, but exact values are not reproduced here.",
        page_label="8",
        embedding=None,
    )
    db.add_all([project, document, chunk])
    db.commit()

    service = QueryService(db)
    service.ollama = FakeOllama()
    service.verifier = FakeVerifier()

    response = service.answer("demo", "What are SAC-KG metrics on OIE2016 and NYT?", save_answer=False)

    assert "No supporting evidence was found yet" in response.answer_markdown
    assert response.citations == []
    assert "As shown in Table 5" not in service.ollama.last_prompt


def test_metric_query_uses_deterministic_table_fallback_when_repair_still_missing() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/knowledge-graph",
        title="Knowledge graph",
        kind="source_summary",
        markdown_path="wiki/demo/sources/knowledge-graph.md",
        markdown_content=(
            "# Knowledge graph\n\n"
            "## Summary\nSAC-KG is a KG construction framework.\n\n"
            "## Tables\n### Page 8\n"
            "Table 5: F1 score and AUC results on OIE2016, WEB, NYT, and PENN datasets.\n"
            "| Model | OIE2016 |  | WEB |  | NYT |  | PENN |  |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC | F1 | AUC | F1 | AUC |\n"
            "| OpenIE 6 (2020) | 55.3 | 61.1 | 61.1 | 64.9 | 30.7 | 55.2 | 54.2 | 63.1 |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 96.6 | 95.7 | 88.8 | 87.3 | 91.1 | 90.1 |\n"
        ),
        source_document_ids=[],
        metadata_json={"verified_claim_count": 5, "key_terms": ["SAC-KG", "OIE2016", "NYT", "Table 5"]},
    )
    document = make_table_document(
        table_markdown=(
            "Table 5: F1 score and AUC results.\n"
            "| Model | OIE2016 |  | NYT |  |\n"
            "| --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |"
        )
    )
    db.add_all([project, document, wiki_page])
    db.commit()

    service = QueryService(db)
    service.ollama = SequencedFakeOllama(
        [
            QueryAnswerPayload(answer_markdown="The values are not present in the provided text.", citations=[0], risk_level="normal"),
            QueryAnswerPayload(answer_markdown="The exact values are still not included in the provided context.", citations=[0], risk_level="normal"),
        ]
    )
    service.verifier = FakeVerifier()

    response = service.answer("demo", "What are SAC-KG metrics on OIE2016 and NYT?", save_answer=False)

    assert "OIE2016 F1 74.7 / AUC 73.2" in response.answer_markdown
    assert "NYT F1 88.8 / AUC 87.3" in response.answer_markdown
    assert response.citations
    assert "Table 5" in response.citations[0].excerpt
    assert "88.8" in response.citations[0].excerpt


def test_metric_query_repairs_answer_that_only_mentions_metric_names() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    table = (
        "Table 4: Biomedical QA benchmark.\n"
        "| Method | PubMedQA | PubMedQA | BioASQ | BioASQ |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | Accuracy | F1 | Accuracy | F1 |\n"
        "| BioGraph-RAG | 84.9 | 79.6 | 75.2 | 71.3 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/bio", page_title="Bio", page_kind="source_summary", score=1, excerpt=table),
            prompt_text=table,
            score=1,
        )
    ]

    service = QueryService(db)
    service._load_index_context = lambda project_slug: None
    service._search_wiki_pages = lambda question, project_id: []
    service._build_rag_contexts = lambda question, project_id, paper_matches: contexts
    service.ollama = SequencedFakeOllama(
        [
            QueryAnswerPayload(
                answer_markdown="Table 4 reports Accuracy and F1 for PubMedQA and BioASQ, but the answer omits the numbers. [0]",
                citations=[0],
                risk_level="normal",
            ),
            QueryAnswerPayload(
                answer_markdown="The repaired answer still only says Accuracy and F1 are reported for PubMedQA and BioASQ. [0]",
                citations=[0],
                risk_level="normal",
            ),
        ]
    )
    service.verifier = FakeVerifier()

    response = service.answer("demo", "What are BioGraph-RAG Accuracy and F1 values on PubMedQA and BioASQ?", save_answer=False)

    assert "PUBMEDQA" in response.answer_markdown
    assert "84.9" in response.answer_markdown
    assert "79.6" in response.answer_markdown
    assert "BIOASQ" in response.answer_markdown
    assert "75.2" in response.answer_markdown
    assert "71.3" in response.answer_markdown


def test_chinese_table_query_prompt_requires_chinese_answer() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    table = (
        "Table 5: F1 score and AUC results.\n"
        "| Model | OIE2016 |  | NYT |  |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | F1 | AUC | F1 | AUC |\n"
        "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/kg", page_title="KG", page_kind="source_summary", score=1, excerpt=table),
            prompt_text=table,
            score=1,
        )
    ]

    service = QueryService(db)
    service._load_index_context = lambda project_slug: None
    service._search_wiki_pages = lambda question, project_id: []
    service._build_rag_contexts = lambda question, project_id, paper_matches: contexts
    fake_ollama = FakeOllama()
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()

    response = service.answer("demo", "SAC-KG 鍦?OIE2016 鎴?NYT 鏁版嵁闆嗕笂鐨勬寚鏍囨槸浠€涔堬紵", save_answer=False)

    assert fake_ollama.last_prompt == ""
    assert "已在表格证据中找到相关指标" in response.answer_markdown


def test_metric_fallback_extracts_values_from_citation_excerpt_when_prompt_text_lacks_table() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    table_excerpt = (
        "Table 5: F1 score and AUC results.\n"
        "| Model | OIE2016 |  | NYT |  |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | F1 | AUC | F1 | AUC |\n"
        "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/kg",
                page_title="KG",
                page_kind="source_summary",
                score=10,
                excerpt=table_excerpt,
            ),
            prompt_text="As shown in Table 5, the requested metrics are discussed without a reproduced table.",
            score=10,
        )
    ]

    service = QueryService(db)
    service._load_index_context = lambda project_slug: None
    service._search_wiki_pages = lambda question, project_id: []
    service._build_rag_contexts = lambda question, project_id, paper_matches: contexts
    service.ollama = SequencedFakeOllama(
        [
            QueryAnswerPayload(answer_markdown="The specific metric values are not present in the provided text.", citations=[0], risk_level="normal"),
            QueryAnswerPayload(answer_markdown="The values still cannot be extracted from the provided context.", citations=[0], risk_level="normal"),
        ]
    )
    service.verifier = FakeVerifier()

    response = service.answer("demo", "What are SAC-KG metrics on OIE2016 and NYT?", save_answer=False)

    assert "OIE2016 F1 74.7 / AUC 73.2" in response.answer_markdown
    assert "NYT F1 88.8 / AUC 87.3" in response.answer_markdown
    assert service.ollama.prompts == []


def test_table_citation_indexes_detect_table_data_in_citation_excerpt() -> None:
    service = QueryService(make_session())
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/kg",
                page_title="KG",
                page_kind="source_summary",
                score=10,
                excerpt=(
                    "Table 5: F1 score and AUC results.\n"
                    "| Model | OIE2016 |  | NYT |  |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    "|  | F1 | AUC | F1 | AUC |\n"
                    "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |"
                ),
            ),
            prompt_text="The benchmark comparison is discussed in prose, but this context window omitted the table.",
            score=10,
        )
    ]

    indexes = service._table_citation_indexes("What are SAC-KG metrics on OIE2016 and NYT?", contexts)

    assert indexes == [0]


def test_table_citation_indexes_detect_supplemental_table_labels() -> None:
    service = QueryService(make_session())
    table = (
        "Table S3: Objective values O for each solving group.\n"
        "| Solving group | Amino acids | O ff99SB | O ff14SB |\n"
        "| --- | --- | --- | --- |\n"
        "| 9 | Asp | 2.5 | 0.9 |\n"
        "| 2 | Ile Thr Val | 1.2 | 0.8 |\n"
        "| 5 | Phe Tyr | 1.7 | 0.8 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/ff14sb", page_title="ff14SB", page_kind="source_summary", score=10, excerpt=table),
            prompt_text=table,
            score=10,
        )
    ]

    indexes = service._table_citation_indexes(
        "ff14SB 的 Table S3 中，Asp、Ile/Thr/Val、Phe/Tyr solving group 的 objective value 如何变化？",
        contexts,
    )

    assert indexes == [0]


def test_table_citation_indexes_ignore_prose_table_mentions_without_table_rows() -> None:
    service = QueryService(make_session())
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/paper",
                page_title="Paper",
                page_kind="source_summary",
                score=10,
                excerpt="As shown in Table 3, ModelX improves the F1 score by 5 points over baselines.",
            ),
            prompt_text="As shown in Table 3, ModelX improves the F1 score by 5 points over baselines.",
            score=10,
        )
    ]

    indexes = service._table_citation_indexes("What does Table 3 report about F1?", contexts)

    assert indexes == []


def test_table_citation_indexes_ignore_table5_prose_without_metric_values_or_rows() -> None:
    service = QueryService(make_session())
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/kg",
                page_title="KG",
                page_kind="source_summary",
                score=10,
                excerpt="As shown in Table 5, SAC-KG performs well across the benchmark datasets.",
            ),
            prompt_text="As shown in Table 5, SAC-KG performs well across the benchmark datasets.",
            score=10,
        )
    ]

    indexes = service._table_citation_indexes("What are SAC-KG metrics in Table 5?", contexts)

    assert indexes == []


def test_metric_query_selects_real_table5_over_higher_scored_prose_context() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    prose_context = RetrievedContext(
        citation=Citation(
            page_slug="sources/kg",
            page_title="KG",
            page_kind="source_summary",
            score=50,
            excerpt="As shown in Table 5, SAC-KG outperforms other baselines on benchmark datasets.",
        ),
        prompt_text="As shown in Table 5, SAC-KG outperforms other baselines on benchmark datasets.",
        score=50,
    )
    table = (
        "Table 5: F1 score and AUC results.\n"
        "| Model | OIE2016 |  | NYT |  |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | F1 | AUC | F1 | AUC |\n"
        "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |"
    )
    table_context = RetrievedContext(
        citation=Citation(
            page_slug="sources/kg",
            page_title="KG",
            page_kind="source_summary",
            score=5,
            excerpt=table,
        ),
        prompt_text=table,
        score=5,
    )
    contexts = [prose_context, table_context]

    service = QueryService(db)
    service._load_index_context = lambda project_slug: None
    service._search_wiki_pages = lambda question, project_id: []
    service._build_rag_contexts = lambda question, project_id, paper_matches: contexts
    service.ollama = SequencedFakeOllama(
        [
            QueryAnswerPayload(answer_markdown="The exact values are not present in the provided context.", citations=[0], risk_level="normal"),
            QueryAnswerPayload(answer_markdown="The values still cannot be extracted from the provided context.", citations=[0], risk_level="normal"),
        ]
    )
    service.verifier = FakeVerifier()

    response = service.answer("demo", "What are SAC-KG Table 5 F1/AUC metrics on OIE2016 and NYT?", save_answer=False)

    assert "OIE2016 F1 74.7 / AUC 73.2" in response.answer_markdown
    assert "NYT F1 88.8 / AUC 87.3" in response.answer_markdown
    assert response.citations
    assert response.citations[0].excerpt.startswith("Table 5")
    assert all("As shown in Table 5" not in citation.excerpt for citation in response.citations)


def test_table_evidence_replacement_preserves_inline_citation_marker() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    prose_context = RetrievedContext(
        citation=Citation(
            page_slug="sources/kg",
            page_title="KG",
            page_kind="source_summary",
            score=50,
            excerpt="As shown in Table 5, SAC-KG outperforms baselines on benchmark datasets.",
        ),
        prompt_text="As shown in Table 5, SAC-KG outperforms baselines on benchmark datasets.",
        score=50,
    )
    table = (
        "Table 5: F1 score and AUC results.\n"
        "| Model | OIE2016 |  | NYT |  |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | F1 | AUC | F1 | AUC |\n"
        "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |"
    )
    table_context = RetrievedContext(
        citation=Citation(
            page_slug="sources/kg",
            page_title="KG",
            page_kind="source_summary",
            score=5,
            excerpt=table,
        ),
        prompt_text=table,
        score=5,
    )
    contexts = [prose_context, table_context]

    service = QueryService(db)
    service._load_index_context = lambda project_slug: None
    service._search_wiki_pages = lambda question, project_id: []
    service._build_rag_contexts = lambda question, project_id, paper_matches: contexts
    service.ollama = SequencedFakeOllama(
        [
            QueryAnswerPayload(
                answer_markdown="Table 5 reports OIE2016 F1 74.7 / AUC 73.2 and NYT F1 88.8 / AUC 87.3 [0].",
                citations=[0],
                risk_level="normal",
            )
        ]
    )
    service.verifier = FakeVerifier()

    response = service.answer("demo", "What are SAC-KG Table 5 F1/AUC metrics on OIE2016 and NYT?", save_answer=False)

    assert response.answer_markdown.endswith("[0]")
    assert len(response.citations) == 1
    assert response.citations[0].excerpt.startswith("Table 5")


def test_context_has_table_data_rejects_html_until_it_is_converted_to_markdown() -> None:
    html_table = (
        "<table><tr><th>Method</th><th>F1</th></tr>"
        "<tr><td>ModelX</td><td>81.4</td></tr></table>"
    )

    assert not QueryService._context_has_table_data(html_table)


def test_context_has_table_data_accepts_legacy_pipe_table_without_separator() -> None:
    table = (
        "| Model | F1 | AUC |\n"
        "| ModelX | 81.4 | 76.2 |"
    )

    assert QueryService._context_has_table_data(table)


def test_citation_is_table_evidence_requires_structured_rows() -> None:
    prose = Citation(
        page_slug="sources/paper",
        page_title="Paper",
        page_kind="source_summary",
        score=10,
        excerpt="As shown in Table 3, ModelX improves the F1 score by 5 points over baselines.",
    )
    table = Citation(
        page_slug="sources/paper",
        page_title="Paper",
        page_kind="source_summary",
        score=1,
        excerpt=(
            "Table 3: Benchmark results.\n"
            "| Model | F1 |\n"
            "| --- | --- |\n"
            "| ModelX | 81.4 |"
        ),
    )

    assert not QueryService._citation_is_table_evidence(prose)
    assert QueryService._citation_is_table_evidence(table)


def test_metric_query_fallback_when_chinese_draft_says_values_cannot_be_extracted() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/knowledge-graph",
        title="Knowledge graph",
        kind="source_summary",
        markdown_path="wiki/demo/sources/knowledge-graph.md",
        markdown_content=(
            "# Knowledge graph\n\n"
            "## Tables\n### Page 8\n"
            "Table 5: F1 score and AUC results.\n"
            "| Model | OIE2016 |  | NYT |  |\n"
            "| --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |\n"
        ),
        source_document_ids=[],
        metadata_json={"verified_claim_count": 5, "key_terms": ["SAC-KG", "OIE2016", "NYT", "Table 5"]},
    )
    document = make_table_document(
        table_markdown=(
            "Table 5: F1 score and AUC results.\n"
            "| Model | OIE2016 |  | NYT |  |\n"
            "| --- | --- | --- | --- | --- |\n"
            "|  | F1 | AUC | F1 | AUC |\n"
            "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |"
        )
    )
    db.add_all([project, document, wiki_page])
    db.commit()

    service = QueryService(db)
    service.ollama = SequencedFakeOllama(
        [
            QueryAnswerPayload(answer_markdown="The current retrieved snippets do not directly contain the complete table, so the specific values cannot be extracted.", citations=[0], risk_level="normal"),
            QueryAnswerPayload(answer_markdown="The values still cannot be extracted from the current retrieved content.", citations=[0], risk_level="normal"),
        ]
    )
    service.verifier = FakeVerifier()

    response = service.answer("demo", "What are SAC-KG F1/AUC metrics on OIE2016 and NYT?", save_answer=False)

    assert "74.7" in response.answer_markdown
    assert "73.2" in response.answer_markdown
    assert "88.8" in response.answer_markdown
    assert "87.3" in response.answer_markdown


def test_metric_extraction_uses_requested_non_sac_kg_row_selector() -> None:
    service = QueryService(make_session())
    table = (
        "Table 3: Biomedical QA results.\n"
        "| Model | PubMedQA | PubMedQA | BioASQ | BioASQ |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | Accuracy | F1 | Accuracy | F1 |\n"
        "| ModelX | 81.4 | 76.2 | 72.1 | 69.8 |\n"
        "| BioGraph-RAG | 84.9 | 79.6 | 75.2 | 71.3 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/bio", page_title="Bio", page_kind="source_summary", score=1, excerpt=table),
            prompt_text=table,
            score=1,
        )
    ]

    metrics = service._extract_requested_metric_values(
        "What are BioGraph-RAG Accuracy and F1 on PubMedQA and BioASQ?",
        contexts,
        [0],
    )

    values = {metric.dataset: metric.values for metric in metrics}
    assert values["PUBMEDQA"] == {"Accuracy": "84.9", "F1": "79.6"}
    assert values["BIOASQ"] == {"Accuracy": "75.2", "F1": "71.3"}


def test_metric_extraction_does_not_treat_generic_datasets_as_row_selectors() -> None:
    service = QueryService(make_session())
    table = (
        "Table 3: Biomedical QA results.\n"
        "| Model | PubMedQA |  | BioASQ |  |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | Accuracy | F1 | Accuracy | F1 |\n"
        "| ModelX | 81.4 | 76.2 | 72.1 | 69.8 |\n"
        "| BioGraph-RAG | 84.9 | 79.6 | 75.2 | 71.3 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/bio", page_title="Bio", page_kind="source_summary", score=1, excerpt=table),
            prompt_text=table,
            score=1,
        )
    ]

    metrics = service._extract_requested_metric_values("What are Accuracy and F1 on PubMedQA and BioASQ?", contexts, [0])

    by_dataset = {}
    for metric in metrics:
        by_dataset.setdefault(metric.dataset, []).append(metric.values)
    assert any(values.get("F1") == "76.2" for values in by_dataset["PUBMEDQA"])
    assert any(values.get("F1") == "69.8" for values in by_dataset["BIOASQ"])


def test_answer_contains_extracted_metrics_requires_values_not_metric_names_only() -> None:
    metrics = [
        ExtractedMetric(0, "Table 4", "PubMedQA", {"Accuracy": "84.9", "F1": "79.6"}),
        ExtractedMetric(0, "Table 4", "BioASQ", {"Accuracy": "75.2", "F1": "71.3"}),
    ]

    assert not QueryService._answer_contains_extracted_metrics(
        "Table 4 discusses PubMedQA and BioASQ Accuracy/F1 metrics, but does not list the values.",
        metrics,
    )
    assert QueryService._answer_contains_extracted_metrics(
        "Table 4 reports PubMedQA Accuracy 84.9 / F1 79.6 and BioASQ Accuracy 75.2 / F1 71.3.",
        metrics,
    )


def test_metric_query_without_extracted_metrics_does_not_return_generic_table_snippet() -> None:
    service = QueryService(make_session())
    context_text = (
        "Table 9: Dataset overview.\n"
        "The article mentions PubMedQA and BioASQ with several observations, but the metric values are not reproduced here."
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/bio", page_title="Bio", page_kind="source_summary", score=1, excerpt=context_text),
            prompt_text=context_text,
            score=1,
        )
    ]

    answer = service._deterministic_table_answer(
        "What are Accuracy and F1 values on PubMedQA and BioASQ?",
        contexts,
        [0],
        "normal",
    )

    assert answer.answer_markdown == "The retrieved table evidence does not contain parseable requested metric values. [0]"
    assert "Dataset overview" not in answer.answer_markdown


def test_metric_extraction_handles_generic_multi_level_dataset_headers() -> None:
    service = QueryService(make_session())
    table = (
        "Table 4: Biomedical QA benchmark.\n"
        "| Method | PubMedQA | PubMedQA | BioASQ | BioASQ |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | Accuracy | F1 | Accuracy | F1 |\n"
        "| Baseline | 80.1 | 74.4 | 70.5 | 66.9 |\n"
        "| BioGraph-RAG | 84.9 | 79.6 | 75.2 | 71.3 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/bio", page_title="Bio", page_kind="source_summary", score=1, excerpt=table),
            prompt_text=table,
            score=1,
        )
    ]

    metrics = service._extract_requested_metric_values(
        "What are BioGraph-RAG Accuracy and F1 values on PubMedQA and BioASQ?",
        contexts,
        [0],
    )

    values = {metric.dataset: metric.values for metric in metrics}
    assert values["PUBMEDQA"] == {"Accuracy": "84.9", "F1": "79.6"}
    assert values["BIOASQ"] == {"Accuracy": "75.2", "F1": "71.3"}


def test_metric_query_extracts_generic_table7_foonet_metrics() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    table = (
        "Table 7: FooNet benchmark results.\n"
        "| Model | Dataset-A |  | Dataset-B |  |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | Accuracy | F1 | Accuracy | F1 |\n"
        "| BaselineNet | 70.1 | 61.4 | 68.2 | 59.7 |\n"
        "| FooNet | 91.2 | 89.5 | 87.6 | 85.4 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/foo", page_title="Foo", page_kind="source_summary", score=1, excerpt=table),
            prompt_text=table,
            score=1,
        )
    ]

    service = QueryService(db)
    service._load_index_context = lambda project_slug: None
    service._search_wiki_pages = lambda question, project_id: []
    service._build_rag_contexts = lambda question, project_id, paper_matches: contexts
    service.ollama = SequencedFakeOllama(
        [
            QueryAnswerPayload(answer_markdown="The exact metric values are not present in the provided context.", citations=[0], risk_level="normal"),
            QueryAnswerPayload(answer_markdown="The requested metric values still cannot be extracted.", citations=[0], risk_level="normal"),
        ]
    )
    service.verifier = FakeVerifier()

    response = service.answer("demo", "What are FooNet Accuracy and F1 on Dataset-A and Dataset-B in Table 7?", save_answer=False)

    assert "Table 7" in response.answer_markdown
    assert "DATASET-A" in response.answer_markdown
    assert "91.2" in response.answer_markdown
    assert "89.5" in response.answer_markdown
    assert "DATASET-B" in response.answer_markdown
    assert "87.6" in response.answer_markdown
    assert "85.4" in response.answer_markdown
    assert response.citations
    assert response.citations[0].excerpt.startswith("Table 7")


def test_generic_table_terms_include_lowercase_scientific_entities() -> None:
    terms = QueryService._extract_generic_table_terms(
        "OPLS-AA 的表格中，butane 构象能和 methanol ΔHvap 如何体现与 6-31G/实验的一致性？"
    )

    assert "butane" in terms
    assert "methanol" in terms
    assert "6-31G" in terms
    assert QueryService._is_comparison_or_difference_query("如何体现与 6-31G/实验的一致性？")


def test_generic_table_terms_keep_chinese_aliases_for_multi_table_questions() -> None:
    terms = QueryService._extract_generic_table_terms(
        "OPLS5 \u7684\u8868\u683c\u4e2d\uff0c\u82b3\u9999\u5c0f\u5206\u5b50 HFE\u3001"
        "\u76d0\u6865 pKa shift\u3001GLU pKa \u548c binding RMSE "
        "\u76f8\u6bd4 OPLS4 \u6709\u54ea\u4e9b\u6570\u503c\u6539\u5584\uff1f"
    )

    assert "aromatic" in terms
    assert "acetate" in terms
    assert "guanidinium" in terms


def test_table_block_excerpt_uses_lowercase_scientific_terms_for_rows() -> None:
    block = (
        "Table 1. Relative Energies (kcal/mol).\n"
        "| molecule | dihedral | conf | OPLS-AA | 6-31G* |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| ethane | H-C-C-H | 0 | 3.01 | 2.99 |\n"
        "| propane | H-C-C-C | 0 | 3.32 | 3.34 |\n"
        "| pentane | C-C-C-C | 0 | 4.21 | 4.20 |\n"
        "| butane | C-C-C-C | 0 | 6.04 | 6.19 |\n"
    )

    excerpt = QueryService._table_block_excerpt(
        block,
        "OPLS-AA 的表格中，butane 构象能如何体现与 6-31G 的一致性？",
    )

    assert "butane" in excerpt
    assert "6.04" in excerpt
    assert "6.19" in excerpt


def test_table_block_excerpt_does_not_match_selector_inside_longer_chemical_name() -> None:
    block = (
        "Table 1. Relative Energies (kcal/mol).\n"
        "| molecule | dihedral | conf | OPLS-AA | 6-31G* |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| butane | C-C-C-C | 0 | 6.04 | 6.19 |\n"
        "| 2-methylbutane | C-C-C-C | 120 | 3.68 | 3.65 |\n"
    )

    excerpt = QueryService._table_block_excerpt(
        block,
        "OPLS-AA 的表格中，butane 构象能如何体现与 6-31G 的一致性？",
    )

    assert "butane" in excerpt
    assert "6.04" in excerpt
    assert "2-methylbutane" not in excerpt
    assert "3.68" not in excerpt


def test_table_block_excerpt_maps_chinese_error_terms_to_rmse_rows() -> None:
    block = (
        "Table 8. Relative Interaction Energy between Sigma-Hole and Head-on Directions (kcal/mol).\n"
        "| interaction partner | CCSD(T)/CBS | OPLS3e | OPLS4 |\n"
        "| --- | --- | --- | --- |\n"
        "| NMA oxygen | -2.15 | -0.99 | -1.94 |\n"
        "| pyridine nitrogen | -1.88 | -0.44 | -1.64 |\n"
        "| water oxygen | 10.99 | -0.33 | -0.51 |\n"
        "| RMS error |  | 1.08 | 0.40 |"
    )

    excerpt = QueryService._table_block_excerpt(
        block,
        "OPLS4 的 sigma-hole 表格中，OPLS3e 到 OPLS4 的关键误差改善是多少？",
    )

    assert "RMS error" in excerpt
    assert "1.08" in excerpt
    assert "0.40" in excerpt


def test_generic_table_answer_keeps_secondary_calcd_exptl_headers() -> None:
    service = QueryService(make_session())
    table = (
        "Table 7. OPLS-AA Energetic Results for Liquid Hydrocarbons and Alcohols.\n"
        "|  |  |  |  |  | Delta H vap | Delta H vap |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| liquid | T | - E inter ( 1 ) | E intra ( g ) | E intra ( 1 ) | calcd | exptl |\n"
        "| methanol | 25.00 | 8.51 ± 0.02 | 7.08 ± 0.02 | 7.23 ± 0.01 | 8.95 ± 0.02 | 8.95c |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/opls-aa", page_title="OPLS-AA", page_kind="source_summary", score=10, excerpt=table),
            prompt_text=table,
            score=10,
        )
    ]

    answer = service._deterministic_generic_table_answer(
        "OPLS-AA 的表格中，methanol ΔHvap 如何体现与实验的一致性？",
        contexts,
        [0],
        "normal",
    )

    assert answer is not None
    assert "exptl" in answer.answer_markdown
    assert "8.95c" in answer.answer_markdown


def test_generic_table_answer_keeps_paired_comparison_rows() -> None:
    service = QueryService(make_session())
    table = (
        "Table 1: alpha L conformational sampling.\n"
        "| System | Simulation | alpha L probability (%) | alpha L propensity Max. | alpha L length |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| FG-nucleoporin peptide | C36 | 32 ± 6 | 22 ± 2 | 14 aa |\n"
        "| FG-nucleoporin peptide | C36m | 1.1 ± 0.3 | 6.2 ± 0.2 | 5 aa |\n"
        "| RS peptide | C36 | 80 ± 2 | 41 ± 1 | 17 aa |\n"
        "| RS peptide | C36m | 1.8 ± 0.5 | 5.5 ± 0.2 | 5 aa |\n"
        "| IN | C36 | 64 ± 18 | 14 ± 2 | 7 aa |\n"
        "| IN | C36m | 3 ± 2 | 5.6 ± 0.5 | 4 aa |\n"
        "| HEWL19 peptide | C36 | 11 ± 7 | 12 ± 2 | 8 aa |\n"
        "| HEWL19 peptide | C36m | 0.5 ± 0.4 | 6.1 ± 0.7 | 3 aa |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/charmm36m", page_title="CHARMM36m", page_kind="source_summary", score=10, excerpt=table),
            prompt_text=table,
            score=10,
        )
    ]

    answer = service._deterministic_generic_table_answer(
        "CHARMM36m 的 Table 1 中，RS peptide、FG-nucleoporin peptide 和 HEWL19 的 alphaL probability 相比 C36 降到了多少？",
        contexts,
        [0],
        "normal",
    )

    assert answer is not None
    assert "80 ± 2" in answer.answer_markdown
    assert "1.8 ± 0.5" in answer.answer_markdown
    assert "32 ± 6" in answer.answer_markdown
    assert "1.1 ± 0.3" in answer.answer_markdown
    assert "11 ± 7" in answer.answer_markdown
    assert "0.5 ± 0.4" in answer.answer_markdown
    cjk_count = sum(1 for char in answer.answer_markdown if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in answer.answer_markdown if "a" <= char.lower() <= "z")
    assert cjk_count / (cjk_count + latin_count) >= 0.2


def test_deterministic_table_answer_uses_generic_rows_for_objective_values() -> None:
    service = QueryService(make_session())
    table = (
        "Table S3: Objective values O for each solving group.\n"
        "| Solving group | Amino acids | O ff99SB | O ff14SB |\n"
        "| --- | --- | --- | --- |\n"
        "| 9 | Asp | 2.5 | 0.9 |\n"
        "| 2 | Ile Thr Val | 1.2 | 0.8 |\n"
        "| 5 | Phe Tyr | 1.7 | 0.8 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/ff14sb", page_title="ff14SB", page_kind="source_summary", score=10, excerpt=table),
            prompt_text=table,
            score=10,
        )
    ]

    answer = service._deterministic_table_answer(
        "ff14SB 的 Table S3 中，Asp、Ile/Thr/Val、Phe/Tyr solving group 的 objective value 从 ff99SB 到 ff14SB 如何变化？",
        contexts,
        [0],
        "normal",
    )

    assert "F1" not in answer.answer_markdown
    assert "Asp" in answer.answer_markdown
    assert "2.5" in answer.answer_markdown
    assert "0.9" in answer.answer_markdown
    assert "Ile Thr Val" in answer.answer_markdown
    assert "1.2" in answer.answer_markdown
    assert "0.8" in answer.answer_markdown
    assert "Phe Tyr" in answer.answer_markdown
    assert "1.7" in answer.answer_markdown


def test_non_metric_table_answer_ignores_metric_extractor_candidates(monkeypatch) -> None:
    service = QueryService(make_session())
    table = (
        "Table S3: Objective values O for each solving group.\n"
        "| Solving group | Amino acids | O ff99SB | O ff14SB |\n"
        "| --- | --- | --- | --- |\n"
        "| 9 | Asp | 2.5 | 0.9 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/ff14sb", page_title="ff14SB", page_kind="source_summary", score=10, excerpt=table),
            prompt_text=table,
            score=10,
        )
    ]
    monkeypatch.setattr(
        service,
        "_extract_requested_metric_values",
        lambda question, contexts, table_indexes: [ExtractedMetric(0, "Table S3", "Solving group", {"F1": "O ff14SB"})],
    )

    answer = service._deterministic_table_answer(
        "ff14SB 的 Table S3 中，Asp solving group 的 objective value 从 ff99SB 到 ff14SB 如何变化？",
        contexts,
        [0],
        "normal",
    )

    assert "F1" not in answer.answer_markdown
    assert "2.5" in answer.answer_markdown
    assert "0.9" in answer.answer_markdown


def test_rag_table_query_answers_grouped_peptide_values_without_model_generation() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    table = (
        "Table 4: Peptide sampling populations.\n"
        "| Peptide | Property | ff99SB | ff99SB* | C22/CMAP | C36 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Ala5 | | | | | |\n"
        "|  | % ppII | 46.2 (1.0) | 52.1 (1.1) | 45.4 (1.0) | 51.9 (1.1) |\n"
        "|  | % alpha-helix | 0.2 (0.1) | 0.3 (0.1) | 0.1 (0.1) | 0.1 (0.1) |\n"
        "| Ac-(AAQAA)3-NH2 | | | | | |\n"
        "|  | % ppII | 28.0 (1.4) | 30.1 (1.2) | 0.5 (0.2) | 29.5 (1.2) |\n"
        "|  | % alpha-helix | 5.4 (0.8) | 8.2 (1.1) | 95.3 (0.1) | 21.0 (1.7) |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/charmm36-force-field-refinement-for-proteins",
                page_title="CHARMM36",
                page_kind="source_summary",
                score=42,
                excerpt=QueryService._table_citation_excerpt(table, "CHARMM36 table Ala5 Ac-(AAQAA)3-NH2 C22/CMAP C36"),
            ),
            prompt_text=table,
            score=42,
        )
    ]

    service = QueryService(db)
    fake_ollama = CountingFakeOllama()
    service.ollama = fake_ollama
    service.verifier = FakeVerifier()
    service._route_papers = lambda question, project_id: []
    service._build_rag_contexts = lambda question, project_id, paper_matches: contexts
    service._search_source_chunks = lambda question, project_id, document_ids, limit=5: []

    response = service.answer(
        "demo",
        "CHARMM36 在 Ala5 和 Ac-(AAQAA)3-NH2 肽采样表中给出的 ppII/alpha-helix 关键比例是多少？和 C22/CMAP 的螺旋比例有什么差异？",
        save_answer=False,
    )

    assert fake_ollama.generate_calls == 0
    assert "Ala5" in response.answer_markdown
    assert "51.9" in response.answer_markdown
    assert "Ac-(AAQAA)3-NH2" in response.answer_markdown
    assert "21.0" in response.answer_markdown
    assert "C22/CMAP" in response.answer_markdown
    assert "95.3" in response.answer_markdown
    assert "C36 21.0" in response.answer_markdown
    assert "C22/CMAP 95.3" in response.answer_markdown
    assert response.citations
    assert response.citations[0].excerpt.startswith("Table 4")


def test_answer_constraints_preserve_scientific_acronyms_from_evidence() -> None:
    context = RetrievedContext(
        citation=Citation(
            page_slug="sources/charmm36-force-field-refinement-for-proteins",
            page_title="CHARMM36",
            page_kind="source_summary",
            score=1,
            excerpt=(
                "CHARMM36 refined CMAP against QM energy surfaces and validated against "
                "NMR scalar couplings and SPARTA chemical shifts."
            ),
        ),
        prompt_text=(
            "The parameterization used QM target data, CMAP corrections, NMR observables, "
            "and SPARTA validation for CHARMM36."
        ),
        score=1,
    )

    constraints = QueryService(make_session())._build_answer_constraints(
        "CHARMM36 用了哪些参数化和验证策略？",
        [context],
    )

    assert "QM" in constraints
    assert "NMR" in constraints
    assert "SPARTA" in constraints


def test_table_block_excerpt_prefers_requested_non_sac_kg_row() -> None:
    block = (
        "Table 3: Biomedical QA results.\n"
        "| Model | PubMedQA | PubMedQA | BioASQ | BioASQ |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | Accuracy | F1 | Accuracy | F1 |\n"
        "| ModelX | 81.4 | 76.2 | 72.1 | 69.8 |\n"
        "| BioGraph-RAG | 84.9 | 79.6 | 75.2 | 71.3 |"
    )

    excerpt = QueryService._table_block_excerpt(
        block,
        "What are BioGraph-RAG metrics on PubMedQA and BioASQ?",
    )

    assert "BioGraph-RAG" in excerpt
    assert "84.9" in excerpt


def test_rank_blocks_prefers_table_with_more_requested_row_anchors_and_values() -> None:
    service = QueryService(make_session())
    wrong_table = (
        "Table 9: Scalar coupling values.\n"
        "|  | Ala3 | Ala5 | Ala7 |\n"
        "| --- | --- | --- | --- |\n"
        "| C36 | 0.61 | 0.74 | 0.43 |\n"
        "| C22/CMAP |  | 1.73 |  |"
    )
    right_table = (
        "Table 4: Peptide sampling populations.\n"
        "| Peptide | Force field | % ppII | % alpha-helix |\n"
        "| --- | --- | --- | --- |\n"
        "| Ala5 | C36 | 51.9 (1.1) | 0.1 (0.1) |\n"
        "| Ac-(AAQAA)3-NH2 | C36 | 29.5 (1.2) | 21.0 (1.7) |\n"
        "| Ac-(AAQAA)3-NH2 | C22/CMAP |  | 95.3 (0.1) |"
    )

    ranked = service._rank_blocks(
        "CHARMM36 鍦?Ala5 鍜?Ac-(AAQAA)3-NH2 鑲介噰鏍疯〃涓粰鍑虹殑 ppII/alpha-helix 鍏抽敭姣斾緥鏄灏戯紵鍜?C22/CMAP 鐨勮灪鏃嬫瘮渚嬫湁浠€涔堝樊寮傦紵",
        [wrong_table, right_table],
    )

    assert ranked[0][0] == right_table


def test_rank_blocks_uses_generic_aliases_for_multi_table_selection() -> None:
    service = QueryService(make_session())
    aromatic_hfe = (
        "Table 2. Hydration free energies for small aromatic molecules impacted by OPLS5.\n"
        "| Compound | Exp. | OPLS4 | OPLS5 |\n"
        "| --- | --- | --- | --- |\n"
        "| RMS error | | 0.76 | 0.46 |"
    )
    generic_rmse = (
        "Table 6. OPLS4 and OPLS5 model performance (RMSE) comparison.\n"
        "| Row | OPLS4 | OPLS5 |\n"
        "| --- | --- | --- |\n"
        "| No external field | 16.6 | 2.9 |"
    )

    ranked = service._rank_blocks(
        "OPLS5 \u7684\u8868\u683c\u4e2d\uff0c\u82b3\u9999\u5c0f\u5206\u5b50 HFE "
        "\u76f8\u6bd4 OPLS4 \u6709\u54ea\u4e9b\u6570\u503c\u6539\u5584\uff1f",
        [generic_rmse, aromatic_hfe],
    )

    assert ranked[0][0] == aromatic_hfe


def test_rank_blocks_prefers_delta_hvap_table_over_heat_capacity_table() -> None:
    service = QueryService(make_session())
    hvap_table = (
        "Table 7. OPLS-AA Energetic Results for Liquid Hydrocarbons and Alcohols.\n"
        "| liquid | T | calcd Delta H vap | exptl Delta H vap |\n"
        "| --- | --- | --- | --- |\n"
        "| butane | -0.50 | 5.44 | 5.35 |\n"
        "| methanol | 25.00 | 8.86 | 8.94 |"
    )
    heat_capacity_table = (
        "Table 11. OPLS-AA Heat Capacities and Compressibilities for Liquid Hydrocarbons and Alcohols.\n"
        "| liquid | T | Cp calcd | Cp exptl |\n"
        "| --- | --- | --- | --- |\n"
        "| butane | -0.50 | 31.0 | 31.8 |\n"
        "| methanol | 25.00 | 26.0 | 19.5 |"
    )

    ranked = service._rank_blocks(
        "OPLS-AA 的表格中，butane 构象能和 methanol ΔHvap 如何体现与 6-31G/实验的一致性？",
        [heat_capacity_table, hvap_table],
    )

    assert ranked[0][0] == hvap_table


def test_rank_blocks_prefers_binding_table_over_generic_rmse_table() -> None:
    service = QueryService(make_session())
    generic_rmse = (
        "Table 6. OPLS4 and OPLS5 model performance (RMSE) comparison.\n"
        "| Row | OPLS4 | OPLS5 |\n"
        "| --- | --- | --- |\n"
        "| No external field | 16.6 | 2.9 |"
    )
    binding_rmse = (
        "Table 7. Root mean square errors for relative binding free energy results (kcal/mol).\n"
        "| PerturbationClass | No.cmpds | OPLS4 | OPLS4 | OPLS5 | OPLS5 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| TotalWeightedAverage | 1183 | 1.18 | 1.29 | 1.12 | 1.25 |"
    )

    ranked = service._rank_blocks(
        "OPLS5 的 binding RMSE 相比 OPLS4 有哪些数值？",
        [generic_rmse, binding_rmse],
    )

    assert ranked[0][0] == binding_rmse


def test_generic_table_answer_uses_caption_matched_numeric_rows() -> None:
    service = QueryService(make_session())
    table = (
        "Table 7. Root mean square errors for relative binding free energy results (kcal/mol).\n"
        "| PerturbationClass | No.cmpds | OPLS4 | OPLS4 | OPLS5 | OPLS5 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| PerturbationClass | No.cmpds | Edgewise | Pairwise | Edgewise | Pairwise |\n"
        "| R-group | 199 | 0.93 | 1.06 | 0.99 | 1.13 |\n"
        "| HeterocycleFocused | 200 | 1.18 | 1.33 | 1.19 | 1.31 |\n"
        "| WaterDisplacement | 65 | 1.12 | 1.19 | 1.13 | 1.15 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/opls5", page_title="OPLS5", page_kind="source_summary", score=10, excerpt=table),
            prompt_text=table,
            score=10,
        )
    ]

    answer = service._deterministic_generic_table_answer(
        "OPLS5 的 binding RMSE 相比 OPLS4 有哪些数值？",
        contexts,
        [0],
        "normal",
    )

    assert answer is not None
    assert "1.18" in answer.answer_markdown
    assert "1.12" in answer.answer_markdown


def test_generic_table_rows_ignore_repeated_headers_before_fallback_rows() -> None:
    table = (
        "Table 7. Root mean square errors for relative binding free energy results (kcal/mol).\n\n"
        "| PerturbationClass | No.cmpds | OPLS4 | OPLS4 | OPLS5 | OPLS5 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| PerturbationClass | No.cmpds | Edgewise | Pairwise | Edgewise | Pairwise |\n"
        "| R-group | 199 | 0.93 | 1.06 | 0.99 | 1.13 |\n"
        "| HeterocycleFocused | 200 | 1.18 | 1.33 | 1.19 | 1.31 |\n"
        "| WaterDisplacement | 65 | 1.12 | 1.19 | 1.13 | 1.15 |\n"
        "| TotalWeightedAverage | 1183 | 1.18 | 1.29 | 1.12 | 1.25 |\n\n"
        "Table 7. Root mean square errors for relative binding free energy results (kcal/mol).\n"
        "| PerturbationClass | No.cmpds | OPLS4 | OPLS4 | OPLS5 | OPLS5 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| PerturbationClass | No.cmpds | Edgewise | Pairwise | Edgewise | Pairwise |\n"
        "| R-group | 199 | 0.93 | 1.06 | 0.99 | 1.13 |"
    )

    rows = QueryService._generic_table_value_rows(
        "OPLS5 的 binding RMSE 相比 OPLS4 有哪些数值？",
        table,
    )

    rendered = " ".join(row["values"] for row in rows)
    assert "1.18" in rendered
    assert "1.12" in rendered


def test_generic_table_answer_keeps_late_binding_table_when_earlier_tables_have_many_rows() -> None:
    service = QueryService(make_session())
    table3 = (
        "Table 3. Interaction energy comparison for configurations that have changed in OPLS5.\n"
        "| Group | CCSD(T)/CBS | OPLS4 | OPLS5 |\n"
        "| --- | --- | --- | --- |\n"
        "| Acetate | -10.5 | -11.3 | -10.5 |\n"
        "| Guanidine | -10.0 | -11.9 | -11.2 |\n"
        "| Benzene | -2.0 | -2.6 | -2.1 |\n"
        "| Phenol | -4.0 | -4.8 | -4.1 |\n"
        "| Pyridine | -3.0 | -3.9 | -3.2 |"
    )
    tables = [
        (
            "Table 4. Acetate pKa shift.\n"
            "| System | Exp. | OPLS4 | OPLS5 |\n"
            "| --- | --- | --- | --- |\n"
            "| Acetate-guanidinium | -0.136 | -0.26 | -0.14 |"
        ),
        (
            "Table 5. Root mean square errors for glutamic acid pKa sets.\n"
            "| Amino acid | Number | OPLS4 | OPLS5 |\n"
            "| --- | --- | --- | --- |\n"
            "| GLU | 44 | 0.70 | 0.61 |"
        ),
        table3,
        (
            "Table 2. Hydration free energies for small aromatic molecules.\n"
            "| Compound | Exp. | OPLS4 | OPLS5 |\n"
            "| --- | --- | --- | --- |\n"
            "| RMS error | | 0.76 | 0.46 |"
        ),
        (
            "Table 7. Root mean square errors for relative binding free energy results (kcal/mol).\n"
            "| PerturbationClass | No.cmpds | OPLS4 | OPLS4 | OPLS5 | OPLS5 |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| PerturbationClass | No.cmpds | Edgewise | Pairwise | Edgewise | Pairwise |\n"
            "| HeterocycleFocused | 200 | 1.18 | 1.33 | 1.19 | 1.31 |\n"
            "| WaterDisplacement | 65 | 1.12 | 1.19 | 1.13 | 1.15 |"
        ),
    ]
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/opls5", page_title="OPLS5", page_kind="source_summary", score=10 - index, excerpt=table),
            prompt_text=table,
            score=10 - index,
        )
        for index, table in enumerate(tables)
    ]

    answer = service._deterministic_generic_table_answer(
        "OPLS5 的表格中，芳香小分子 HFE、盐桥 pKa shift、GLU pKa 和 binding RMSE 相比 OPLS4 有哪些数值改善？",
        contexts,
        list(range(len(contexts))),
        "normal",
    )

    assert answer is not None
    assert "Table 7" in answer.answer_markdown
    assert "1.18" in answer.answer_markdown
    assert "1.12" in answer.answer_markdown


def test_rag_table_query_returns_structured_tables_before_profile_terms() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    table4 = (
        "Table 4. Acetate pKa shift.\n"
        "| System | Exp. | OPLS4 | OPLS5 |\n"
        "| --- | --- | --- | --- |\n"
        "| Acetate-guanidinium | -0.136 | -0.26 | -0.14 |"
    )
    binding_table = (
        "Table 7. Root mean square errors for relative binding free energy results (kcal/mol).\n"
        "| PerturbationClass | No.cmpds | OPLS4 | OPLS4 | OPLS5 | OPLS5 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| PerturbationClass | No.cmpds | Edgewise | Pairwise | Edgewise | Pairwise |\n"
        "| HeterocycleFocused | 200 | 1.18 | 1.33 | 1.19 | 1.31 |\n"
        "| WaterDisplacement | 65 | 1.12 | 1.19 | 1.13 | 1.15 |"
    )
    document = Document(
        id="opls5",
        project_id="p1",
        title="OPLS5",
        file_name="opls5.pdf",
        sha256="opls5",
        raw_path="raw/opls5.pdf",
        raw_text="OPLS5 paper.",
        metadata_json={
            "source_slug": "sources/opls5-force-field-development-and-validation",
            "source_title": "OPLS5",
            "document_intelligence": {
                "tables": [
                    {"markdown": table4, "page_label": "19"},
                    {"markdown": binding_table, "page_label": "21"},
                ]
            },
        },
        status="ready",
    )
    profile_chunks = [
        DocumentChunk(
            id=f"profile-{index}",
            document_id="opls5",
            ordinal=index,
            text="OPLS5 Drude polarizability cation charge transfer LFMM profile evidence.",
            page_label=str(index),
            embedding=None,
        )
        for index in range(6)
    ]
    db.add_all([project, document, *profile_chunks])
    db.commit()

    contexts = QueryService(db)._build_rag_contexts(
        "OPLS5 的表格中，盐桥 pKa shift 和 binding RMSE 相比 OPLS4 有哪些数值改善？",
        "p1",
        [PaperMatch(document=document, score=20, exact_alias=True)],
    )
    evidence = "\n".join(context.citation.excerpt for context in contexts)

    assert contexts
    assert all(context.evidence_kind == "table" for context in contexts)
    assert "Table 7" in evidence
    assert "1.18" in evidence
    assert "1.12" in evidence


def test_table_block_match_requires_multiple_anchors_for_multi_anchor_question() -> None:
    wrong_table = (
        "Table 9: Scalar coupling values.\n"
        "|  | Ala3 | Ala5 | Ala7 |\n"
        "| --- | --- | --- | --- |\n"
        "| C36 | 0.61 | 0.74 | 0.43 |"
    )
    right_table = (
        "Table 4: Peptide sampling populations.\n"
        "| Peptide | Force field | % ppII | % alpha-helix |\n"
        "| --- | --- | --- | --- |\n"
        "| Ala5 | C36 | 51.9 (1.1) | 0.1 (0.1) |\n"
        "| Ac-(AAQAA)3-NH2 | C36 | 29.5 (1.2) | 21.0 (1.7) |\n"
        "| Ac-(AAQAA)3-NH2 | C22/CMAP |  | 95.3 (0.1) |"
    )
    question = "CHARMM36 鍦?Ala5 鍜?Ac-(AAQAA)3-NH2 鑲介噰鏍疯〃涓粰鍑虹殑 ppII/alpha-helix 鍏抽敭姣斾緥鏄灏戯紵"

    assert not QueryService._table_block_matches_query(question, wrong_table)
    assert QueryService._table_block_matches_query(question, right_table)


def test_table_block_match_deduplicates_single_anchor_with_explicit_table() -> None:
    block = (
        "Table 4: Force-field results.\n"
        "| Model | Value |\n"
        "| --- | --- |\n"
        "| C36 | 51.9 |"
    )

    assert QueryService._table_block_matches_query("What does Table 4 report for C36?", block)


def test_table_block_excerpt_includes_grouped_child_rows_after_matched_parent_rows() -> None:
    block = (
        "Table 4: Peptide sampling populations.\n"
        "| Peptide | Property | ff99SB | ff99SB* | C22/CMAP | C36 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Ala5 | | | | | |\n"
        "|  | % ppII | 46.2 (1.0) | 52.1 (1.1) | 45.4 (1.0) | 51.9 (1.1) |\n"
        "|  | % alpha-helix | 0.2 (0.1) | 0.3 (0.1) | 0.1 (0.1) | 0.1 (0.1) |\n"
        "| Ac-(AAQAA)3-NH2 | | | | | |\n"
        "|  | % ppII | 28.0 (1.4) | 30.1 (1.2) | 0.5 (0.2) | 29.5 (1.2) |\n"
        "|  | % alpha-helix | 5.4 (0.8) | 8.2 (1.1) | 95.3 (0.1) | 21.0 (1.7) |\n"
    )

    excerpt = QueryService._table_block_excerpt(
        block,
        "What does the peptide sampling table report for Ala5 and Ac-(AAQAA)3-NH2?",
    )

    assert "Ala5" in excerpt
    assert "Ac-(AAQAA)3-NH2" in excerpt
    assert "51.9" in excerpt
    assert "21.0" in excerpt
    assert "95.3" in excerpt


def test_table_block_excerpt_keeps_grouped_child_rows_for_generic_parent_match() -> None:
    block = (
        "Table 6: Grouped simulation results.\n"
        "| System | Property | Baseline | Model |\n"
        "| --- | --- | --- | --- |\n"
        "| System A | | | |\n"
        "|  | compactness | 1.20 | 0.82 |\n"
        "|  | stability | 4.10 | 5.30 |\n"
        "| System B | | | |\n"
        "|  | compactness | 2.20 | 1.90 |\n"
    )

    excerpt = QueryService._table_block_excerpt(
        block,
        "What does Table 6 report for System A?",
    )

    assert "System A" in excerpt
    assert "compactness" in excerpt
    assert "0.82" in excerpt
    assert "stability" in excerpt
    assert "5.30" in excerpt
    assert "System B" not in excerpt


def test_table_block_excerpt_keeps_continuation_rows_after_matched_value_row() -> None:
    block = (
        "Table 4: Peptide sampling populations.\n"
        "| Peptide | Property | C22/CMAP | C36 |\n"
        "| --- | --- | --- | --- |\n"
        "| Ala5 | % ppII | 45.4 (1.0) | 51.9 (1.1) |\n"
        "|  | % alpha-helix | 0.1 (0.1) | 0.1 (0.1) |\n"
        "| Ac-(AAQAA)3-NH2 | % ppII | 0.5 (0.2) | 29.5 (1.2) |\n"
        "|  | % alpha-helix | 95.3 (0.1) | 21.0 (1.7) |\n"
    )

    excerpt = QueryService._table_block_excerpt(
        block,
        "What does the peptide sampling table report for Ala5 and Ac-(AAQAA)3-NH2?",
    )

    assert "Ala5" in excerpt
    assert "51.9" in excerpt
    assert "Ac-(AAQAA)3-NH2" in excerpt
    assert "95.3" in excerpt
    assert "21.0" in excerpt


def test_table_block_excerpt_keeps_table_caption_after_page_heading() -> None:
    block = (
        "### Page 30\n"
        "Table 4: Peptide sampling populations.\n"
        "| Peptide | Property | C22/CMAP | C36 |\n"
        "| --- | --- | --- | --- |\n"
        "| Ala5 | | | |\n"
        "|  | % ppII | 26.2 (1.8) | 51.9 (1.1) |\n"
    )

    excerpt = QueryService._table_block_excerpt(block, "What does Table 4 report for Ala5?")

    assert excerpt.startswith("Table 4")
    assert "### Page" not in excerpt
    assert "Ala5" in excerpt
    assert "51.9" in excerpt


def test_table_citation_excerpt_normalizes_ildn_table_i_symbols_and_residue_ocr() -> None:
    block = (
        "Table I Table I<sub>List</sub> <sub>of</sub> <sub>Modified</sub> <sub>Parameters</sub> "
        "<sub>for</sub> <sub>the</sub> \\mathbb { \\chi } _ { 1 } and \\chi _ { 2 } "
        "Torsion Potentials in Selected Amino Acids of the Amber ff99SB Force Field\n"
        "| Res. | Angle | theta 0 | mathsf k 1 |\n"
        "| --- | --- | --- | --- |\n"
        "| lle | mathsf N - C ^alpha - C ^beta - C ^gamma 2 | 0.0 | 0.195 |\n"
        "| Leu | mathsf C - C ^alpha - C ^beta - C ^gamma | 0.0 | 0.571 |\n"
        "| Asp | mathsf N - C ^alpha - C ^beta - C ^gamma | 0.0 | -2.635 |\n"
        "| Asn | C ^alpha - C ^beta - C ^gamma - O ^delta | 0.0 | 0.423 |"
    )

    excerpt = QueryService._table_citation_excerpt(
        block,
        "ff99SB-ILDN 的 Table I 列出了哪些残基/角度的修改参数？theta0 的设置是什么？",
    )

    assert "Table I" in excerpt
    assert "Ile" in excerpt
    assert "Leu" in excerpt
    assert "Asp" in excerpt
    assert "Asn" in excerpt
    assert "theta 0" in excerpt
    assert "0.0" in excerpt
    assert "χ1" in excerpt
    assert "χ2" in excerpt


def test_table_normalization_does_not_rewrite_lle_without_residue_context() -> None:
    normalized = normalize_table_text("| word | value |\n| --- | --- |\n| lle | 3 |")

    assert "| lle | 3 |" in normalized
    assert "Ile" not in normalized


def test_generic_table_answer_includes_salient_ildn_header_symbols() -> None:
    service = QueryService(make_session())
    question = "ff99SB-ILDN 的 Table I 列出了哪些残基/角度的修改参数？theta0 的设置是什么？"
    block = (
        "Table I Table I<sub>List</sub> <sub>of</sub> <sub>Modified</sub> <sub>Parameters</sub> "
        "<sub>for</sub> <sub>the</sub> \\mathbb { \\chi } _ { 1 } and \\chi _ { 2 } "
        "Torsion Potentials in Selected Amino Acids of the Amber ff99SB Force Field\n"
        "| Res. | Angle | theta 0 | mathsf k 1 |\n"
        "| --- | --- | --- | --- |\n"
        "| lle | mathsf N - C ^alpha - C ^beta - C ^gamma 2 | 0.0 | 0.195 |\n"
        "| Leu | mathsf C - C ^alpha - C ^beta - C ^gamma | 0.0 | 0.571 |\n"
        "| Asp | mathsf N - C ^alpha - C ^beta - C ^gamma | 0.0 | -2.635 |\n"
        "| Asn | C ^alpha - C ^beta - C ^gamma - O ^delta | 0.0 | 0.423 |"
    )
    excerpt = QueryService._table_citation_excerpt(block, question)
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/ff99sb-ildn", page_title="ff99SB-ILDN", page_kind="source_summary", score=10, excerpt=excerpt),
            prompt_text=block,
            score=10,
            evidence_kind="table",
        )
    ]

    answer = service._deterministic_generic_table_answer(question, contexts, [0], "normal")

    assert answer is not None
    assert "Ile" in answer.answer_markdown
    assert "Leu" in answer.answer_markdown
    assert "Asp" in answer.answer_markdown
    assert "Asn" in answer.answer_markdown
    assert "theta 0" in answer.answer_markdown
    assert "0.0" in answer.answer_markdown
    assert "χ1" in answer.answer_markdown
    assert "χ2" in answer.answer_markdown


def test_scientific_answer_appends_supported_intro_terms() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/charmm36-force-field-refinement-for-proteins",
                page_title="CHARMM36 Force Field",
                page_kind="source_summary",
                score=10,
                excerpt=(
                    "While the current CHARMM22/CMAP force field was useful, "
                    "NMR spectroscopy and folding simulations showed limitations."
                ),
            ),
            prompt_text="",
            score=10,
            evidence_kind="intro",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "CHARMM36 蛋白力场主要想修正 CHARMM22/CMAP 的什么问题？它用了哪些参数化和验证策略？",
        "已有证据说明相关问题。",
        contexts,
    )

    assert "CHARMM36" in answer
    assert "CHARMM22/CMAP" in answer
    assert "NMR" in answer


def test_scientific_answer_appends_large_idps_when_supported_by_evidence() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/charmm36idpsff",
                page_title="CHARMM36IDPSFF",
                page_kind="source_summary",
                score=10,
                excerpt=(
                    "Classical force fields perform better for folded proteins than IDPs with a large "
                    "conformational fluctuation; the paper also evaluates radius of gyration and fast-folding systems."
                ),
            ),
            prompt_text="",
            score=10,
            evidence_kind="intro",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "CHARMM36IDPSFF 相比 CHARMM36m/a99SB- 的定位是什么？它在哪些场景仍有不足？",
        "已有证据说明 CHARMM36m、a99SB、radius of gyration 和 fast-folding。",
        contexts,
    )

    assert "large IDPs" in answer


def test_scientific_answer_prioritizes_supported_high_value_terms() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/example",
                page_title="Example",
                page_kind="source_summary",
                score=10,
                excerpt=(
                    "The paper rebalances helical and extended conformations, uses the SPARTA+ algorithm, "
                    "reduces the barrier between PPII and right-handed helix, discusses Lennard–Jones and steric effects, "
                    "reports QM/MM differences, molten globule states, w = 2 k T around \\Phi = 6 0 and \\psi = 4 5, "
                    "evaluates free energies of hydration, and fits torsional parameters."
                ),
            ),
            prompt_text="",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "为什么这些力场修改能改善采样？",
        "已有证据说明相关机制。",
        contexts,
    )

    for term in (
        "helix-coil",
        "SPARTA",
        "PPII",
        "Lennard-Jones",
        "steric",
        "QM-MM",
        "molten globule",
        "2kT",
        "60",
        "45",
        "hydration free energy",
        "torsional",
    ):
        assert term in answer


def test_scientific_answer_appends_explicit_hydrogen_for_opls_aa_ua_contrast() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/opls-aa-force-field-development-and-validation",
                page_title="OPLS-AA",
                page_kind="source_summary",
                score=10,
                excerpt="OPLS-AA improves transferability relative to OPLS-UA through charge and torsional validation.",
            ),
            prompt_text="",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "为什么 OPLS-AA 的显式氢和液体性质验证能改善相对 OPLS-UA 的可转移性？",
        "已有证据说明 charge 和 torsional validation。",
        contexts,
    )

    assert "explicit hydrogen" in answer


def test_scientific_answer_does_not_mark_question_only_identifiers_as_evidence() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/charmm36idpsff",
                page_title="CHARMM36IDPSFF",
                page_kind="source_summary",
                score=10,
                excerpt="CHARMM36IDPSFF originates from CHARMM36m and is evaluated on IDP systems.",
            ),
            prompt_text="",
            score=10,
            evidence_kind="intro",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "CHARMM36IDPSFF 相比 CHARMM36m/a99SB- 的定位是什么？",
        "已有证据说明 CHARMM36IDPSFF 和 CHARMM36m。",
        contexts,
    )

    assert "a99SB" not in answer


def test_table_block_excerpt_labels_table_evidence_when_caption_lacks_table_word() -> None:
    block = (
        "### Page 30\n"
        "Properties of peptides used in parameter optimization: Ala5 and Ac-(AAQAA)3-NH2.\n"
        "Statistical errors from a block error analysis are given in brackets.\n"
        "| Peptide | Property | C22/CMAP | C36 |\n"
        "| --- | --- | --- | --- |\n"
        "| Ala5 | | | |\n"
        "|  | % ppII | 26.2 (1.8) | 51.9 (1.1) |\n"
        "| Ac-(AAQAA)3-NH2 | | | |\n"
        "|  | % a-helix | 95.3 (0.1) | 21.0 (1.7) |\n"
    )

    excerpt = QueryService._table_block_excerpt(
        block,
        "What does the peptide sampling table report for Ala5 and Ac-(AAQAA)3-NH2?",
    )

    assert excerpt.startswith("Table evidence:")
    assert "### Page" not in excerpt
    assert "Properties of peptides" in excerpt
    assert "Ala5" in excerpt
    assert "51.9" in excerpt
    assert "Ac-(AAQAA)3-NH2" in excerpt
    assert "95.3" in excerpt
    assert "21.0" in excerpt


def test_table_block_excerpt_keeps_captionless_html_table_after_page_heading() -> None:
    block = (
        "### Page 12\n"
        "Measurements for ModelX.\n"
        "<table><tr><th>Model</th><th>F1</th></tr><tr><td>ModelX</td><td>81.4</td></tr></table>"
    )

    excerpt = QueryService._table_block_excerpt(block, "What does the table report for ModelX?")

    assert excerpt.startswith("Table evidence:")
    assert "Measurements for ModelX" in excerpt
    assert "<table>" in excerpt
    assert "81.4" in excerpt


def test_table_block_excerpt_uses_word_boundary_for_existing_table_caption() -> None:
    block = (
        "### Page 13\n"
        "Stable measurements for ModelX.\n"
        "| Model | F1 |\n"
        "| --- | --- |\n"
        "| ModelX | 81.4 |"
    )

    excerpt = QueryService._table_block_excerpt(block, "What does the table report for ModelX?")

    assert excerpt.startswith("Table evidence: Stable measurements")


def test_table_block_excerpt_starts_at_table_fragment_after_prose() -> None:
    block = (
        "As shown in Table 3, the following benchmark summarizes the reported metrics.\n"
        "| Model | F1 | AUC |\n"
        "| --- | --- | --- |\n"
        "| ModelX | 81.4 | 76.2 |"
    )

    excerpt = QueryService._table_block_excerpt(block, "What does Table 3 report?")

    assert excerpt.startswith("| Model |")
    assert "ModelX" in excerpt
    assert "As shown in Table 3" not in excerpt


def test_table_block_excerpt_starts_at_html_table_after_prose() -> None:
    block = (
        "As shown in Table 3, the following benchmark summarizes the reported metrics.\n"
        "<table><tr><th>Model</th><th>F1</th></tr><tr><td>ModelX</td><td>81.4</td></tr></table>"
    )

    excerpt = QueryService._table_block_excerpt(block, "What does Table 3 report?")

    assert excerpt.startswith("<table>")
    assert "ModelX" in excerpt
    assert "As shown in Table 3" not in excerpt


def test_table_query_citation_excerpt_starts_from_table_evidence() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    wiki_page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/knowledge-graph",
        title="Knowledge graph",
        kind="source_summary",
        markdown_path="wiki/demo/sources/knowledge-graph.md",
        markdown_content=(
            "# Knowledge graph\n\n"
            "## Summary\nA summary should not become the table citation.\n\n"
            "## Tables\n### Page 4\n"
            "Table 2: Ablation study.\n"
            "| Variant | F1 |\n"
            "| --- | --- |\n"
            "| w/o verifier | 68.1 |\n"
            "| SAC-KG full | 74.7 |\n"
        ),
        source_document_ids=[],
        metadata_json={"verified_claim_count": 4, "key_terms": ["Table 2", "ablation"]},
    )
    document = make_table_document(
        page_label="4",
        table_markdown=(
            "Table 2: Ablation study.\n"
            "| Variant | F1 |\n"
            "| --- | --- |\n"
            "| w/o verifier | 68.1 |\n"
            "| SAC-KG full | 74.7 |"
        ),
    )
    db.add_all([project, document, wiki_page])
    db.commit()

    service = QueryService(db)
    service.ollama = FakeOllama()
    service.ollama.payload = QueryAnswerPayload(answer_markdown="Table 2 reports the ablation result [0].", citations=[0], risk_level="normal")
    service.verifier = FakeVerifier()

    response = service.answer("demo", "Please cite Table 2 ablation results.", save_answer=False)

    assert response.citations
    assert response.citations[0].excerpt.startswith("Table 2")
    assert "A summary should not" not in response.citations[0].excerpt


def test_table2_ablation_selects_markdown_table_over_higher_scored_prose_context() -> None:
    db = make_session()
    db.add(Project(id="p1", slug="demo", name="Demo"))
    db.commit()
    prose_context = RetrievedContext(
        citation=Citation(
            page_slug="sources/kg",
            page_title="KG",
            page_kind="source_summary",
            score=40,
            excerpt="As shown in Table 2, the ablation study validates each component.",
        ),
        prompt_text="As shown in Table 2, the ablation study validates each component.",
        score=40,
    )
    table = (
        "Table 2: Ablation study.\n"
        "| Iteration rounds | Model | Number of recalls | Precision | Domain Specificity |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Iteration 1 | SAC-KG w/o prompt | 10.15 | 80.64 | 74.19 |\n"
        "| Iteration 1 | SAC-KG | 13.50 | 88.81 | 80.50 |"
    )
    table_context = RetrievedContext(
        citation=Citation(
            page_slug="sources/kg",
            page_title="KG",
            page_kind="source_summary",
            score=5,
            excerpt=table,
        ),
        prompt_text=table,
        score=5,
    )
    contexts = [prose_context, table_context]

    service = QueryService(db)
    service._load_index_context = lambda project_slug: None
    service._search_wiki_pages = lambda question, project_id: []
    service._build_rag_contexts = lambda question, project_id, paper_matches: contexts
    service.ollama = SequencedFakeOllama(
        [
            QueryAnswerPayload(answer_markdown="The ablation values are not included in the provided context.", citations=[0], risk_level="normal"),
            QueryAnswerPayload(answer_markdown="The ablation values still cannot be extracted.", citations=[0], risk_level="normal"),
        ]
    )
    service.verifier = FakeVerifier()

    response = service.answer("demo", "What conclusions can be drawn from Table 2 ablation study?", save_answer=False)

    assert "Table 2 shows" in response.answer_markdown
    assert "precision 88.81" in response.answer_markdown
    assert response.citations
    assert response.citations[0].excerpt.startswith("Table 2")
    assert all("As shown in Table 2" not in citation.excerpt for citation in response.citations)


def test_ablation_table_excerpt_keeps_all_rows_needed_for_summary() -> None:
    block = (
        "Table 2: Ablation study for a multi-round model.\n"
        "| Iteration rounds | Model | Number of recalls | Precision | Domain Specificity |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Iteration 1 | Full Model | 13.50 | 88.81 | 80.50 |\n"
        "| Iteration 1 | w/o prompt | 10.15 | 80.64 | 74.19 |\n"
        "| Iteration 2 | Full Model | 9.94 | 84.61 | 76.27 |\n"
        "| Iteration 2 | w/o prompt | 8.66 | 78.44 | 70.01 |\n"
        "| Iteration 3 | Full Model | 6.63 | 76.74 | 68.60 |\n"
        "| Iteration 3 | w/o prompt | 5.08 | 70.11 | 63.20 |"
    )

    excerpt = QueryService._table_block_excerpt(block, "What conclusions do the ablation studies draw? Please cite Table 2.")

    assert "84.61" in excerpt
    assert "76.74" in excerpt


def test_citation_markup_is_normalized_and_capped_to_returned_citations() -> None:
    answer = "Metrics [[0]] are supported, but [3] is not. See [[Knowledge graph](sources/knowledge-graph.md)] and [[sources/foo.md]]."

    renumbered = QueryService._renumber_answer_citations(answer, [0, 1])
    renumbered = QueryService._drop_unreturned_citation_markers(renumbered, 2)

    assert "[0]" in renumbered
    assert "[3]" not in renumbered
    assert "[[" not in renumbered
    assert "sources/foo" not in renumbered


def test_same_page_dedup_prefers_table_citation() -> None:
    service = QueryService(make_session())
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/kg", page_title="KG", page_kind="source_summary", score=20, excerpt="# Knowledge graph\n\n## Summary"),
            prompt_text="summary",
            score=20,
        ),
        RetrievedContext(
            citation=Citation(
                page_slug="sources/kg",
                page_title="KG",
                page_kind="source_summary",
                score=1,
                excerpt="Table 5: results\n| Model | F1 |\n| --- | --- |\n| SAC-KG | 88.8 |",
            ),
            prompt_text="Table 5: results\n| Model | F1 |\n| --- | --- |\n| SAC-KG | 88.8 |",
            score=1,
        ),
        RetrievedContext(
            citation=Citation(page_slug="sources/kg", page_title="KG", page_kind="source_summary", score=2, excerpt="Other paragraph"),
            prompt_text="Other paragraph",
            score=2,
        ),
    ]

    selected = service._select_citations(contexts, [0, 1, 2])

    assert len(selected) == 2
    assert any("Table 5" in citation.excerpt for citation in selected)


def test_select_citations_keeps_multiple_table_citations_from_same_source() -> None:
    service = QueryService(make_session())
    contexts = []
    for table_number in range(2, 6):
        table = (
            f"Table {table_number}. Metrics.\n"
            "| Row | OPLS4 | OPLS5 |\n"
            "| --- | --- | --- |\n"
            f"| value | 0.{table_number}0 | 0.{table_number}1 |"
        )
        contexts.append(
            RetrievedContext(
                citation=Citation(
                    page_slug="sources/opls5",
                    page_title="OPLS5",
                    page_kind="source_summary",
                    score=50 - table_number,
                    excerpt=table,
                ),
                prompt_text=table,
                score=50 - table_number,
            )
        )

    citations = service._select_citations(contexts, [0, 1, 2, 3])

    assert len(citations) == 4
    assert all(citation.excerpt.startswith("Table") for citation in citations)


def test_finalize_contexts_keeps_more_than_three_table_contexts_from_same_source() -> None:
    service = QueryService(make_session())
    contexts = []
    for table_number in range(2, 6):
        table = (
            f"Table {table_number}. Metrics.\n"
            "| Row | OPLS4 | OPLS5 |\n"
            "| --- | --- | --- |\n"
            f"| value | 0.{table_number}0 | 0.{table_number}1 |"
        )
        contexts.append(
            RetrievedContext(
                citation=Citation(
                    page_slug="sources/opls5",
                    page_title="OPLS5",
                    page_kind="source_summary",
                    score=50 - table_number,
                    excerpt=table,
                ),
                prompt_text=table,
                score=50 - table_number,
            )
        )

    finalized = service._finalize_contexts(contexts)

    assert len(finalized) == 4


def test_metric_extraction_repairs_mineru_flattened_table5_header_and_latex_row() -> None:
    service = QueryService(make_session())
    table = (
        "Table 5: F1 score and AUC results on OIE2016, WEB, NYT, and PENN datasets.\n"
        "| Model | OIE2016 | WEB | NYT | PENN |  |  |  |  |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "|  | F1 | AUC | F1 | AUC | F1 | AUC | F1 | AUC |\n"
        "| OpenIE 6 (2020) | 55.3 | 61.1 | 61.1 | 64.9 | 30.7 | 55.2 | 54.2 | 63.1 |\n"
        "| $\\mathbf { S } \\mathbf { A } \\mathbf { C } \\mathbf { - } \\mathbf { K } \\mathbf { G } _ { \\mathrm { C h a t G P T } }$ | 74.7 | 73.2 | 96.6 | 95.7 | 88.8 | 87.3 | 91.1 | 90.1 |"
    )
    normalized = QueryService._extract_table_blocks("## Tables\n### Page 8\n" + table)[0]
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/kg", page_title="KG", page_kind="source_summary", score=1, excerpt=normalized),
            prompt_text=normalized,
            score=1,
        )
    ]

    metrics = service._extract_requested_metric_values("What are OIE2016 and NYT metrics?", contexts, [0])

    values = {metric.dataset: metric.values for metric in metrics}
    assert values["OIE2016"] == {"F1": "74.7", "AUC": "73.2"}
    assert values["NYT"] == {"F1": "88.8", "AUC": "87.3"}
    assert "SAC-KG ChatGPT" in normalized


def test_deterministic_table_answer_reports_structured_metrics() -> None:
    service = QueryService(make_session())
    table = (
        "Table 5: F1 score and AUC results.\n"
        "| Model | OIE2016 | OIE2016 | NYT | NYT |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | F1 | AUC | F1 | AUC |\n"
        "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/kg", page_title="KG", page_kind="source_summary", score=1, excerpt=table),
            prompt_text=table,
            score=1,
        )
    ]

    answer = service._deterministic_table_answer("OIE2016 and NYT metrics?", contexts, [0], "normal")

    assert "OIE2016 F1 74.7 / AUC 73.2" in answer.answer_markdown
    assert "NYT F1 88.8 / AUC 87.3" in answer.answer_markdown
    assert answer.citations == [0]


def test_deterministic_table_answer_summarizes_ablation_table() -> None:
    service = QueryService(make_session())
    table = (
        "Table 7: Ablation study.\n"
        "| Iteration rounds | Model | Number of recalls | Precision | Domain Specificity |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Iteration 1 | SAC-KG w/o prompt | 10.15 | 80.64 | 74.19 |\n"
        "| Iteration 1 | SAC-KG | 13.50 | 88.81 | 80.50 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/kg", page_title="KG", page_kind="source_summary", score=1, excerpt=table),
            prompt_text=table,
            score=1,
        )
    ]

    answer = service._deterministic_table_answer("What conclusions can be drawn from this table?", contexts, [0], "normal")

    assert "Table 7 shows" in answer.answer_markdown
    assert "precision 88.81" in answer.answer_markdown
    assert "not treated as missing" not in answer.answer_markdown


def test_deterministic_ablation_answer_uses_citation_excerpt_when_prompt_text_lacks_table() -> None:
    service = QueryService(make_session())
    table = (
        "Table 2: Ablation study.\n"
        "| Iteration rounds | Model | Number of recalls | Precision | Domain Specificity |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Iteration 1 | SAC-KG w/o prompt | 10.15 | 80.64 | 74.19 |\n"
        "| Iteration 1 | SAC-KG | 13.50 | 88.81 | 80.50 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/kg", page_title="KG", page_kind="source_summary", score=1, excerpt=table),
            prompt_text="The ablation study is discussed, but this context window omitted the reproduced table.",
            score=1,
        )
    ]

    answer = service._deterministic_table_answer("What conclusions can be drawn from the ablation study?", contexts, [0], "normal")

    assert "Table 2 shows" in answer.answer_markdown
    assert "precision 88.81" in answer.answer_markdown
    assert answer.citations == [0]


def test_chinese_deterministic_ablation_answer_is_localized() -> None:
    service = QueryService(make_session())
    table = (
        "Table 2: Ablation study.\n"
        "| Iteration rounds | Model | Number of recalls | Precision | Domain Specificity |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Iteration 1 | Full Model | 13.50 | 88.81 | 80.50 |\n"
        "| Iteration 1 | w/o prompt | 10.15 | 80.64 | 74.19 |\n"
        "| Iteration 2 | Full Model | 9.94 | 84.61 | 76.27 |\n"
        "| Iteration 2 | w/o prompt | 8.66 | 78.44 | 70.01 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/kg", page_title="KG", page_kind="source_summary", score=1, excerpt=table),
            prompt_text=table,
            score=1,
        )
    ]

    answer = service._deterministic_table_answer("What conclusions do the ablation studies draw? Please cite Table 2.", contexts, [0], "normal")

    assert "Table 2" in answer.answer_markdown
    assert "full" in answer.answer_markdown.lower()
    assert "88.81" in answer.answer_markdown
    assert "underperform" in answer.answer_markdown


def test_deterministic_ablation_answer_returns_none_without_structured_table() -> None:
    service = QueryService(make_session())
    context_text = (
        "Table 2 is referenced in the paper. The prose says an ablation study was conducted, "
        "but no rows or metric values are included in this retrieved context."
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/kg", page_title="KG", page_kind="source_summary", score=1, excerpt=context_text),
            prompt_text=context_text,
            score=1,
        )
    ]

    answer = service._deterministic_ablation_answer("What conclusions can be drawn from Table 2 ablation?", contexts, [0], "normal")

    assert answer is None


def test_deterministic_table_answer_does_not_fake_ablation_without_structured_table() -> None:
    service = QueryService(make_session())
    context_text = (
        "Table 2 is referenced in the paper. The prose says an ablation study was conducted, "
        "but no rows or metric values are included in this retrieved context."
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/kg", page_title="KG", page_kind="source_summary", score=1, excerpt=context_text),
            prompt_text=context_text,
            score=1,
        )
    ]

    answer = service._deterministic_table_answer("What conclusions can be drawn from Table 2 ablation?", contexts, [0], "normal")

    assert answer.answer_markdown == "The retrieved table evidence does not contain a structured ablation table. [0]"
    assert "conducted" not in answer.answer_markdown


def test_table_normalization_repairs_mineru_table2_iteration_rowspans() -> None:
    block = (
        "Table 2: Ablation study.\n"
        "| Iteration rounds | Model | Number of recalls | Precision | Domain Specificity |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Iteration 1 | SAC-KG w/o prompt | 10.15 | 80.64 | 74.19 |\n"
        "| SAC-KG | 13.50 | 88.81 | 80.50 |  |\n"
    )

    normalized = QueryService._extract_table_blocks("## Tables\n### Page 5\n" + block)[0]

    assert "| Iteration 1 | SAC-KG | 13.50 | 88.81 | 80.50 |" in normalized


def test_table_normalization_repairs_spaced_decimal_and_uncertainty_values() -> None:
    block = (
        "Table 1: OCR values.\n"
        "| System | C36 | C36m | Expt |\n"
        "| --- | --- | --- | --- |\n"
        "| RS peptide | 8 0 pm 2 | 1 . 8 pm 0 . 5 | 6 04 |\n"
        "| methanol | 8 95 | 6 19 | 0 . 46 |\n"
    )

    normalized = normalize_table_text(block)

    assert "80 ± 2" in normalized
    assert "1.8 ± 0.5" in normalized
    assert "6.04" in normalized
    assert "8.95" in normalized
    assert "6.19" in normalized
    assert "0.46" in normalized


def test_table_normalization_repairs_split_decimal_digits_around_uncertainty() -> None:
    normalized = normalize_table_text("| System | Exp. | Calc |\n| --- | --- | --- |\n| ACTR | 2 5.0 0 ± 1.0 0 ^60 | 1 3.0 7 ± 0.0 9 |")

    assert "25.00 ± 1.00 ^60" in normalized
    assert "13.07 ± 0.09" in normalized


def test_table_normalization_repairs_single_split_digit_before_decimal() -> None:
    normalized = normalize_table_text("| System | Exp. | Calc |\n| --- | --- | --- |\n| Aβ40 | 1 2.0 ± 1.3 | 11.53 ± 0.13 |")

    assert "12.0 ± 1.3" in normalized


def test_charmm36idpsff_rg_answer_keeps_beta40_and_actr_rows() -> None:
    service = QueryService(make_session())
    question = "CHARMM36IDPSFF 的 Rg 表格中，Aβ40 和 ACTR 的实验值与模拟值分别是多少？"
    table = (
        "Table 8. Average radius of gyration Rg.\n"
        "| Systems | Exp. | C36IDPSFF |\n"
        "| --- | --- | --- |\n"
        "| Aβ40 | 12.0 | 11.53 ± 0.08 |\n"
        "| ACTR (71 aa) | 2 5.0 0 ± 1.0 0 ^60 | 1 3.0 7 ± 0.0 9 |"
    )
    excerpt = QueryService._table_citation_excerpt(table, question)
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/charmm36idpsff", page_title="CHARMM36IDPSFF", page_kind="source_summary", score=1, excerpt=excerpt),
            prompt_text=table,
            score=1,
            evidence_kind="table",
        )
    ]

    answer = service._deterministic_table_answer(question, contexts, [0], "normal")

    assert "Aβ40" in excerpt
    assert "12.0" in excerpt
    assert "11.53" in excerpt
    assert "ACTR" in excerpt
    assert "25.00" in excerpt
    assert "13.07" in excerpt
    assert "Aβ40" in answer.answer_markdown
    assert "ACTR" in answer.answer_markdown
    assert "25.00" in answer.answer_markdown
    assert "13.07" in answer.answer_markdown


def test_water_model_parameter_answer_ignores_unmatched_fallback_table_rows() -> None:
    service = QueryService(make_session())
    question = "TIP4P-D 在水模型参数表中相对 TIP3P 的 C6、偶极矩和表面张力数值是什么？"
    simulation_conditions = (
        "Simulation conditions of all tested peptides and proteins.\n"
        "| System | Length | Force fields/water models |\n"
        "| --- | --- | --- |\n"
        "| ALA5 | 5aa | C36IDPSFF |\n"
        "| Aβ40 | 40 aa | C36IDPSFF/disp-water |"
    )
    water_parameters = (
        "Table 1. Parameters and Physical Properties of Selected Commonly Used Water Models and TIP4P-D\n"
        "|  | Expt | TIP3P | SPC/E | TIP4P-EW | TIP4P/2005 | TIP4P-D |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| μ (D) | > 2.6 | 2.35 | 2.35 | 2.32 | 2.305 | 2.403 |\n"
        "| C6 | 622 | 595 | 625 | 653 | 736 | 900 |\n"
        "| surface tension | 72.0 | 47.8 | 63.6 | 65.7 | 69.3 | 71.2 |"
    )
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/charmm36idpsff", page_title="CHARMM36IDPSFF", page_kind="source_summary", score=2, excerpt=simulation_conditions),
            prompt_text=simulation_conditions,
            score=2,
            evidence_kind="table",
        ),
        RetrievedContext(
            citation=Citation(page_slug="sources/ff99sb-disp", page_title="ff99SB-disp", page_kind="source_summary", score=1, excerpt=QueryService._table_citation_excerpt(water_parameters, question)),
            prompt_text=water_parameters,
            score=1,
            evidence_kind="table",
        ),
    ]

    answer = service._deterministic_table_answer(question, contexts, [0, 1], "normal")

    assert "Table 1" in answer.answer_markdown
    assert "TIP4P-D 900" in answer.answer_markdown
    assert "TIP3P 595" in answer.answer_markdown
    assert "TIP4P-D 2.403" in answer.answer_markdown
    assert "TIP3P 2.35" in answer.answer_markdown
    assert "TIP4P-D 71.2" in answer.answer_markdown
    assert "TIP3P 47.8" in answer.answer_markdown
    assert "ALA5" not in answer.answer_markdown


def test_table_normalization_repairs_mineru_spaced_lj_parameter_labels() -> None:
    normalized = normalize_table_text(
        "| Parameter | TIP3P | TIP4P-D |\n"
        "| --- | --- | --- |\n"
        "| C 6 ( mathrm k c a l m o l ^-1 mathring A ^6 ) | 595 | 900 |\n"
        "| C _ { 12 } ( mathrm k c a l m o l ^-1 mathring A ^12 ) | 582 000 | 904 657 |"
    )

    assert "C6 (" in normalized
    assert "C12 (" in normalized


def test_supported_term_note_appends_ff19sb_parameterization_anchors() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/ff19sb-amino-acid-specific-protein-backbone-parameters",
                page_title="ff19SB",
                page_kind="source_summary",
                score=10,
                excerpt=(
                    "RESP charge fitting used HF/6-31G electrostatic potentials. "
                    "The QM reference used M05-2X and MP2/cc-pVQZ energy surfaces. "
                    "Leu CMAP, Ile, and Val CMAP assignments were validated for 5 milliseconds."
                ),
            ),
            prompt_text="",
            score=10,
            evidence_kind="profile-term",
        )
    ]

    answer = QueryService._append_missing_supported_question_terms(
        "ff19SB 的参数化策略中，RESP charge fitting、QM level、CMAP 分配和验证规模有哪些具体锚点？",
        "已有证据说明 RESP、QM 和 CMAP。",
        contexts,
    )

    for term in ("HF/6-31G", "M05-2X", "MP2/cc-pVQZ", "Leu CMAP", "Ile", "Val CMAP", "5 milliseconds"):
        assert term in answer


def test_needs_source_evidence_does_not_treat_chinese_cite_as_raw_request() -> None:
    assert not QueryService(make_session())._needs_source_evidence("What conclusions do the ablation studies draw? Please cite Table 2.")
    assert QueryService(make_session())._needs_source_evidence("Please provide source evidence.")


# ------------------------------------------------------------------
# evidence relevance gate — sample document detection and rejection
# ------------------------------------------------------------------


def test_extract_specific_question_scientific_terms_extracts_force_fields() -> None:
    terms = QueryService._extract_specific_question_scientific_terms(
        "What is the difference between CHARMM36m and CHARMM36?"
    )
    assert "charmm36m" in terms
    assert "charmm36" in terms


def test_extract_specific_question_scientific_terms_extracts_amber() -> None:
    terms = QueryService._extract_specific_question_scientific_terms(
        "How does AMBER99SB compare to OPLS4?"
    )
    assert "amber99sb" in terms
    assert "opls4" in terms


def test_extract_specific_question_scientific_terms_returns_empty_for_generic() -> None:
    terms = QueryService._extract_specific_question_scientific_terms(
        "What is the main result of this paper?"
    )
    assert not terms  # "result", "paper" are stop terms


def test_evidence_overlaps_question_scientific_terms_true() -> None:
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/ff", page_title="FF", page_kind="source_summary", score=10, excerpt="CHARMM36m uses..."),
            prompt_text="CHARMM36m uses a modified backbone CMAP correction.",
            score=10,
        )
    ]
    assert QueryService._evidence_overlaps_question_scientific_terms(
        "What is the difference between CHARMM36m and CHARMM36?", contexts
    )


def test_evidence_overlap_accepts_source_title_for_profile_term_context() -> None:
    """Profile-term snippets may omit the force-field name in the excerpt.

    When retrieval selected a source page titled CHARMM36m, the relevance gate
    should treat that source metadata as grounding instead of returning an
    insufficient-evidence answer.
    """
    contexts = [
        RetrievedContext(
            citation=Citation(
                page_slug="sources/charmm36m",
                page_title="charmm36m",
                page_kind="source_summary",
                score=73,
                excerpt="The predicted alphaL probability drops to 1.1% with w = 2 kT.",
            ),
            prompt_text="The predicted alphaL probability drops to 1.1% with w = 2 kT.",
            score=73,
        )
    ]

    assert QueryService._evidence_overlaps_question_scientific_terms(
        "What is the difference between CHARMM36m and CHARMM36?", contexts
    )


def test_evidence_overlaps_question_scientific_terms_false() -> None:
    """Sample evidence about Method A / Method B must not match a CHARMM question."""
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/sample", page_title="Sample", page_kind="source_summary", score=5, excerpt="Method A improves..."),
            prompt_text="Method A improves control stability. Method B is simpler but slower.",
            score=5,
        )
    ]
    assert not QueryService._evidence_overlaps_question_scientific_terms(
        "What is the difference between CHARMM36m and CHARMM36?", contexts
    )


def test_evidence_overlaps_skips_when_no_specific_terms() -> None:
    """Generic questions with no specific scientific terms skip the gate."""
    contexts = [
        RetrievedContext(
            citation=Citation(page_slug="sources/x", page_title="X", page_kind="source_summary", score=5, excerpt="generic text"),
            prompt_text="generic text",
            score=5,
        )
    ]
    # Should return True (skip gate) when no specific terms are in the question
    assert QueryService._evidence_overlaps_question_scientific_terms(
        "What does this paper conclude?", contexts
    )


def test_draft_answer_returns_insufficient_evidence_for_charmm_on_sample() -> None:
    """When asking about CHARMM36m/CHARMM36 but evidence is only Method A/B demo text,
    the draft answer must return the insufficient-evidence message, not call the LLM."""
    db = make_session()
    service = QueryService(db)
    # Use ExplodingOllama — if the LLM is called, the test fails
    service.ollama = ExplodingOllama()

    contexts = [
        RetrievedContext(
            citation=Citation(
                document_id="d-sample",
                chunk_id="c-s",
                score=2.0,
                excerpt="Method A improves control stability.",
            ),
            prompt_text="Method A improves control stability. Method B is simpler but slower.",
            score=2.0,
        )
    ]

    payload = service._draft_answer(
        "What is the difference between CHARMM36m and CHARMM36?",
        None,
        contexts,
    )
    assert "Insufficient Evidence" in payload.answer_markdown
    assert "CHARMM" in payload.answer_markdown.lower() or "specific scientific" in payload.answer_markdown.lower()
    assert payload.citations == []
    # Must NOT contain the sample text
    assert "Method A" not in payload.answer_markdown
    assert "control stability" not in payload.answer_markdown


def test_answer_returns_no_citations_for_irrelevant_scientific_sample() -> None:
    """The public answer path must not attach sample citations to insufficient-evidence answers."""
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d-sample",
        project_id="p1",
        title="sample",
        file_name="sample.txt",
        sha256="sha-sample",
        raw_path="raw/sample.txt",
        raw_text="Method A improves control stability. Method B is simpler but slower.",
        status="ready",
    )
    chunk = DocumentChunk(
        id="c-sample",
        document_id="d-sample",
        ordinal=0,
        page_label="1",
        text="Method A improves control stability. Method B is simpler but slower.",
        embedding=[1.0, 0.0],
    )
    db.add_all([project, document, chunk])
    db.commit()

    service = QueryService(db)
    service.ollama = ExplodingOllama()

    response = service.answer(
        "demo",
        "What is the difference between CHARMM36m and CHARMM36?",
        save_answer=False,
    )

    assert "Insufficient Evidence" in response.answer_markdown
    assert response.citations == []
    assert "[0]" not in response.answer_markdown
    assert "Method A" not in response.answer_markdown


def test_draft_answer_calls_llm_when_evidence_is_relevant() -> None:
    """When evidence actually mentions the force field, the LLM should be called normally."""
    db = make_session()
    service = QueryService(db)
    fake_ollama = FakeOllama()
    fake_ollama.payload = QueryAnswerPayload(
        answer_markdown="CHARMM36m improves backbone CMAP over CHARMM36.",
        citations=[0],
        risk_level="normal",
    )
    service.ollama = fake_ollama
    contexts = [
        RetrievedContext(
            citation=Citation(
                document_id="d-real",
                chunk_id="c-r",
                score=15.0,
                excerpt="CHARMM36m backbone CMAP correction...",
            ),
            prompt_text="CHARMM36m uses a modified backbone CMAP correction compared to CHARMM36.",
            score=15.0,
        )
    ]

    payload = service._draft_answer(
        "What is the difference between CHARMM36m and CHARMM36?",
        None,
        contexts,
    )
    # With relevant evidence, the LLM answer should NOT be an insufficient-evidence message
    assert "Insufficient Evidence" not in payload.answer_markdown
    assert "CHARMM36m" in payload.answer_markdown


def test_insufficient_evidence_answer_includes_terms() -> None:
    answer = QueryService._insufficient_evidence_answer({"charmm36m", "charmm36"})
    assert "Insufficient Evidence" in answer
    assert "upload" in answer.lower() or "relevant" in answer.lower()


def test_insufficient_evidence_answer_without_terms() -> None:
    answer = QueryService._insufficient_evidence_answer(None)
    assert "Insufficient Evidence" in answer
    assert "upload" in answer.lower() or "relevant" in answer.lower()
