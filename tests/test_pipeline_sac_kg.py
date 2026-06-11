from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.records import Claim, Document, DocumentChunk, Entity, Project
from app.services.ai import DocumentAnalysisPayload, DocumentExtraction, ExtractedClaim, GeneratedTriple, HeadAnalysisPayload, VerificationPayload
from app.services.pipeline import IngestionPipeline
from app.services.wiki import WikiRenderer


class FakeVerifier:
    def verify_claims(self, summary, claims):
        return VerificationPayload(verdict="local-only", notes="", flagged_claim_indexes=[])


class FakeHeadOllama:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.generate_calls = 0

    def generate_structured(self, schema, *, system_prompt: str, user_prompt: str, model: str | None = None):
        self.prompts.append(user_prompt)
        self.generate_calls += 1
        if self.generate_calls == 1:
            return HeadAnalysisPayload(
                head_entity="高血压",
                summary="初始输出",
                triples=[
                    GeneratedTriple(subject="错误主体", predicate="recommended_follow_up", object_text="3个月后复查"),
                    GeneratedTriple(subject="高血压", predicate="recommended_follow_up", object_text="高血压"),
                    GeneratedTriple(subject="高血压", predicate="", object_text="缺少谓词"),
                ],
            )
        return HeadAnalysisPayload(
            head_entity="高血压",
            summary="修正后输出",
            triples=[
                GeneratedTriple(
                    subject="高血压",
                    predicate="recommended_follow_up",
                    object_text="3个月后复查",
                    evidence_excerpt="医生建议患者在3个月后进行复查。",
                ),
                GeneratedTriple(
                    subject="高血压",
                    predicate="maintains_treatment",
                    object_text="继续当前治疗方案",
                    evidence_excerpt="并继续当前治疗方案。",
                ),
                GeneratedTriple(
                    subject="高血压",
                    predicate="confirmed_by",
                    object_text="当前诊断记录",
                    evidence_excerpt="患者诊断为高血压。",
                ),
            ],
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class FakeEmbeddingOllama:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def test_fallback_generator_preserves_followup_fact_in_source_wiki() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="随访记录",
        file_name="case.md",
        sha256="abc",
        raw_path="raw/case.md",
        raw_text="患者诊断为高血压。医生建议患者在3个月后进行复查，并继续当前治疗方案。",
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        ordinal=0,
        text=document.raw_text,
        page_label="1",
        embedding=None,
    )
    db.add_all([project, document, chunk])
    db.commit()

    pipeline = IngestionPipeline(db)
    pipeline.verifier = FakeVerifier()
    extraction = pipeline._fallback_analysis(document, document.raw_text or "", pipeline._select_generation_contexts(document, document.raw_text or "", [chunk]))
    normalized = pipeline._analysis_to_extraction(document, extraction)
    claims = pipeline._create_claims(document, normalized)

    renderer = WikiRenderer(project)
    _, markdown = renderer.render_document_summary(document, normalized, claims)

    assert "## Key Facts" in markdown
    assert "3个月后进行复查" in markdown
    assert "## Verified Triples By Head" in markdown
    assert 'title: "随访记录"' in markdown
    assert claims[0].verification_status == "verified"


def test_local_verifier_sends_unsupported_claim_to_review_queue() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="Case", file_name="case.md", sha256="abc", raw_path="raw/case.md", raw_text="Only source text.")
    chunk = DocumentChunk(id="c1", document_id="d1", ordinal=0, text="Only source text.", embedding=None)
    db.add_all([project, document, chunk])
    db.commit()

    extraction = DocumentExtraction(
        title="Case",
        summary="Summary",
        claims=[
            ExtractedClaim(
                subject="Unsupported entity",
                predicate="recommends",
                object_text="a two-week follow-up",
                evidence_excerpt="This sentence does not exist in the source.",
                confidence=0.4,
            )
        ],
    )
    pipeline = IngestionPipeline(db)
    pipeline.verifier = FakeVerifier()

    claims = pipeline._create_claims(document, extraction)
    review_count = pipeline._create_review_items(document, extraction, claims)

    assert claims[0].verification_status == "needs-review"
    assert "missing_evidence" in claims[0].metadata_json["verification_errors"]
    assert "low_confidence" in claims[0].metadata_json["verification_errors"]
    assert review_count == 2


def test_pruner_grows_durable_entities_and_prunes_transient_values() -> None:
    db = make_session()
    document = Document(id="d1", project_id="p1", title="Case", file_name="case.md", sha256="abc", raw_path="raw/case.md")
    entities = [
        Entity(id="e1", project_id="p1", name="高血压", entity_type="disease", aliases=[], summary=""),
        Entity(id="e2", project_id="p1", name="二甲双胍", entity_type="drug", aliases=[], summary=""),
        Entity(id="e3", project_id="p1", name="3个月", entity_type="date", aliases=[], summary=""),
        Entity(id="e4", project_id="p1", name="500mg", entity_type="dose", aliases=[], summary=""),
    ]
    pipeline = IngestionPipeline(db)

    decisions = pipeline._decide_entity_growth(document, entities, claims=[])

    assert decisions["高血压"].decision == "grow"
    assert decisions["二甲双胍"].decision == "grow"
    assert decisions["3个月"].decision == "prune"
    assert decisions["500mg"].decision == "prune"


def test_head_context_retriever_prioritizes_relevant_snippets() -> None:
    db = make_session()
    pipeline = IngestionPipeline(db)
    pipeline.ollama = FakeHeadOllama()
    chunks = [
        DocumentChunk(id="c1", document_id="d1", ordinal=0, text="这是无关的背景介绍。", page_label="1", embedding=None),
        DocumentChunk(id="c2", document_id="d1", ordinal=1, text="患者诊断为高血压。医生建议患者在3个月后进行复查。", page_label="2", embedding=None),
    ]

    snippets = pipeline._build_sentence_entries(chunks)
    contexts = pipeline._retrieve_head_contexts({"name": "高血压", "aliases": [], "entity_type": "disease", "summary": ""}, snippets)

    assert contexts
    assert "高血压" in contexts[0]["text"]
    assert "3个月后进行复查" in contexts[0]["text"]


def test_open_kg_examples_support_exact_subentity_and_generic_fallback() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    db.add(project)
    db.add(
        Claim(
            id="c1",
            project_id="p1",
            document_id="d1",
            subject="高血压",
            predicate="recommended_follow_up",
            object_text="3个月后复查",
            confidence=0.9,
            verification_status="verified",
            metadata_json={},
        )
    )
    db.add(
        Claim(
            id="c2",
            project_id="p1",
            document_id="d1",
            subject="Propagation protocol",
            predicate="supports",
            object_text="cell growth",
            confidence=0.9,
            verification_status="verified",
            metadata_json={},
        )
    )
    db.commit()

    pipeline = IngestionPipeline(db)

    exact = pipeline._open_kg_examples("p1", "高血压")
    fuzzy = pipeline._open_kg_examples("p1", "micro propagation")
    generic = pipeline._open_kg_examples("p1", "完全未知实体")

    assert exact[0]["subject"] == "高血压"
    assert any(example["subject"] == "Propagation protocol" for example in fuzzy)
    assert generic[0]["subject"] == "Disease"


def test_head_generator_reprompts_when_verifier_finds_many_errors() -> None:
    db = make_session()
    pipeline = IngestionPipeline(db)
    fake_ollama = FakeHeadOllama()
    pipeline.ollama = fake_ollama
    head = {"name": "高血压", "aliases": [], "entity_type": "disease", "summary": ""}
    document = Document(id="d1", project_id="p1", title="随访记录", file_name="case.md", sha256="abc", raw_path="raw/case.md")
    contexts = [
        {"chunk_ordinal": 0, "sentence_ref": "chunk-0:sentence-0", "text": "患者诊断为高血压。", "page_label": "1"},
        {"chunk_ordinal": 0, "sentence_ref": "chunk-0:sentence-1", "text": "医生建议患者在3个月后进行复查。", "page_label": "1"},
        {"chunk_ordinal": 0, "sentence_ref": "chunk-0:sentence-2", "text": "并继续当前治疗方案。", "page_label": "1"},
    ]

    result = pipeline._generate_head_analysis(
        document=document,
        head=head,
        contexts=contexts,
        open_kg_examples=[{"subject": "高血压", "predicate": "type", "object_text": "慢性病"}],
        wiki_context="",
        previous_claims=[],
    )

    assert fake_ollama.generate_calls == 2
    assert "Target head entity: 高血压" in fake_ollama.prompts[0]
    assert result.triples
    assert all(triple.subject == "高血压" for triple in result.triples)
    assert "Verifier correction reprompt executed." in result.coverage_notes


def test_tail_pruner_uses_verified_triple_objects() -> None:
    db = make_session()
    pipeline = IngestionPipeline(db)
    document = Document(id="d1", project_id="p1", title="Case", file_name="case.md", sha256="abc", raw_path="raw/case.md")
    entities = [Entity(id="e1", project_id="p1", name="高血压", entity_type="disease", aliases=[], summary="")]
    claims = [
        Claim(
            id="c1",
            project_id="p1",
            document_id="d1",
            subject="高血压",
            predicate="recommended_follow_up",
            object_text="3个月",
            confidence=0.9,
            verification_status="verified",
            metadata_json={},
        ),
        Claim(
            id="c2",
            project_id="p1",
            document_id="d1",
            subject="高血压",
            predicate="treated_with",
            object_text="二甲双胍",
            confidence=0.9,
            verification_status="verified",
            metadata_json={},
        ),
    ]

    decisions = pipeline._decide_entity_growth(document, entities, claims)

    assert decisions["3个月"].decision == "prune"
    assert decisions["二甲双胍"].decision == "grow"


def test_head_context_retriever_scores_existing_embeddings() -> None:
    db = make_session()
    pipeline = IngestionPipeline(db)
    pipeline.ollama = FakeEmbeddingOllama()
    entries = [
        {
            "chunk_id": "c1",
            "chunk_ordinal": 0,
            "page_label": "1",
            "heading": None,
            "text": "Background note without matching terms.",
            "sentence_ref": "chunk-0:sentence-0",
            "embedding": [0.0, 1.0],
        },
        {
            "chunk_id": "c2",
            "chunk_ordinal": 1,
            "page_label": "2",
            "heading": None,
            "text": "Follow-up recommendation captured in the document.",
            "sentence_ref": "chunk-1:sentence-0",
            "embedding": [1.0, 0.0],
        },
    ]

    contexts = pipeline._retrieve_head_contexts(
        {"name": "Hypertension", "aliases": [], "entity_type": "disease", "summary": ""},
        entries,
    )

    assert contexts
    assert contexts[0]["chunk_id"] == "c2"
    assert "embedding" not in contexts[0]


def test_merge_document_analysis_flattens_derived_keywords() -> None:
    db = make_session()
    pipeline = IngestionPipeline(db)
    document = Document(id="d1", project_id="p1", title="Case", file_name="case.md", sha256="abc", raw_path="raw/case.md")
    seed = DocumentAnalysisPayload(
        title="Case",
        summary="Seed summary",
        keywords=["seed"],
        key_facts=["Seed fact"],
    )
    head_payload = HeadAnalysisPayload(
        head_entity="Hypertension",
        summary="The doctor made a follow-up recommendation after diagnosis.",
        key_facts=["Follow-up was recommended."],
        triples=[
            GeneratedTriple(
                subject="Hypertension",
                predicate="recommended_follow_up",
                object_text="3 months",
                evidence_excerpt="Follow-up was recommended.",
            )
        ],
    )

    extraction = pipeline._merge_document_analysis(document, seed, [head_payload])

    assert "seed" in extraction.keywords
    assert "recommendation" in extraction.keywords
    assert extraction.claims


def test_source_fact_claim_relaxes_evidence_checks() -> None:
    db = make_session()
    pipeline = IngestionPipeline(db)
    claim = ExtractedClaim(
        subject="PDF Source",
        predicate="states",
        object_text="Figure 1 describes the SAC-KG input and output flow.",
        confidence=0.5,
        relation_type="source_fact",
    )

    errors = pipeline._verify_claim_locally(claim, evidence_chunk=None, previous_claims=[], seen=set())

    assert "missing_evidence" not in errors
    assert "low_confidence" not in errors


def test_source_page_metadata_falls_back_to_chinese_key_terms() -> None:
    db = make_session()
    pipeline = IngestionPipeline(db)
    extraction = DocumentExtraction(
        title="社区慢病随访记录",
        summary="患者需要定期复查。",
        key_facts=["医生建议患者在三个月后复查。"],
        keywords=[],
        concepts=[],
    )

    metadata = pipeline._source_page_metadata(extraction, entities=[], claims=[])

    assert metadata["key_terms"]
    assert any("复查" in term or "患者" in term for term in metadata["key_terms"])
