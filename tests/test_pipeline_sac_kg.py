from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
import pytest

from app.models.records import (
    Claim,
    Document,
    DocumentChunk,
    DocumentParseVersion,
    DocumentStatus,
    Entity,
    PipelineRun,
    Project,
    ReviewItem,
    ReviewStatus,
    RunStatus,
    RunType,
)
from app.services.ai import DocumentAnalysisPayload, DocumentExtraction, ExtractedClaim, GeneratedTriple, HeadAnalysisPayload, VerificationPayload
from app.services.parser import ParsedChunk, ParsedDocument
from app.services.pipeline import IngestionPipeline


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


def test_create_review_items_preserves_resolved_and_ignored_manual_items() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="Case", file_name="case.md", sha256="abc", raw_path="raw/case.md", raw_text="Source text.")
    db.add_all(
        [
            project,
            document,
            ReviewItem(
                id="pending-ingest",
                project_id="p1",
                document_id="d1",
                title="Old ingest",
                detail="old",
                status=ReviewStatus.pending.value,
                payload={"generated_by": "ingest"},
            ),
            ReviewItem(
                id="resolved-manual",
                project_id="p1",
                document_id="d1",
                title="Resolved manual",
                detail="keep",
                status=ReviewStatus.resolved.value,
                payload={"generated_by": "human"},
            ),
            ReviewItem(
                id="ignored-manual",
                project_id="p1",
                document_id="d1",
                title="Ignored manual",
                detail="keep",
                status=ReviewStatus.ignored.value,
                payload={},
            ),
        ]
    )
    db.commit()
    pipeline = IngestionPipeline(db)
    pipeline.verifier = FakeVerifier()

    review_count = pipeline._create_review_items(document, DocumentExtraction(title="Case", summary="Summary", claims=[]), claims=[])

    items = {item.id: item for item in db.query(ReviewItem).all()}
    assert review_count == 1
    assert "pending-ingest" not in items
    assert items["resolved-manual"].status == ReviewStatus.resolved.value
    assert items["ignored-manual"].status == ReviewStatus.ignored.value
    generated = [item for item in items.values() if item.id not in {"resolved-manual", "ignored-manual"}]
    assert generated
    assert all(item.payload.get("generated_by") == "ingest" for item in generated)


def test_process_document_rolls_back_partial_changes_before_marking_failed(monkeypatch) -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="Case", file_name="case.md", sha256="abc", raw_path="raw/case.md")
    run = PipelineRun(id="r1", project_id="p1", document_id="d1", run_type=RunType.ingest.value, status=RunStatus.queued.value, provider_report={})
    db.add_all([project, document, run])
    db.commit()

    from app.services import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "parse_document",
        lambda path: ParsedDocument(
            title="Parsed",
            text="Parsed text",
            chunks=[ParsedChunk(ordinal=0, text="Parsed text")],
            metadata={
                "canonical": {
                    "quality": {"accepted": True, "status": "accepted"},
                    "table_activation_allowed": True,
                }
            },
        ),
    )

    def failing_replace_chunks(self: IngestionPipeline, failed_document: Document, parsed_chunks) -> None:
        self.db.add(DocumentChunk(document_id=failed_document.id, ordinal=99, text="partial"))
        raise RuntimeError("chunk failure")

    monkeypatch.setattr(IngestionPipeline, "_replace_chunks", failing_replace_chunks)

    with pytest.raises(RuntimeError, match="chunk failure"):
        IngestionPipeline(db).process_document("d1")

    db.expire_all()
    assert db.get(Document, "d1").status == DocumentStatus.failed.value
    assert db.get(PipelineRun, "r1").status == RunStatus.failed.value
    assert db.query(DocumentChunk).filter(DocumentChunk.document_id == "d1").count() == 0


def test_process_document_cleans_chunks_when_later_extraction_fails(monkeypatch) -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="Case", file_name="case.md", sha256="abc", raw_path="raw/case.md")
    run = PipelineRun(id="r1", project_id="p1", document_id="d1", run_type=RunType.ingest.value, status=RunStatus.queued.value, provider_report={})
    db.add_all([project, document, run])
    db.commit()

    from app.services import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module.settings, "sac_kg_enabled", True)
    monkeypatch.setattr(
        pipeline_module,
        "parse_document",
        lambda path: ParsedDocument(
            title="Case",
            text="Parsed text",
            chunks=[ParsedChunk(ordinal=0, text="Parsed chunk", page_label="1")],
            metadata={
                "canonical": {
                    "quality": {"accepted": True, "status": "accepted"},
                    "table_activation_allowed": True,
                }
            },
        ),
    )
    monkeypatch.setattr(IngestionPipeline, "_extract_document", lambda self, failed_document, full_text: (_ for _ in ()).throw(RuntimeError("extract failure")))

    pipeline = IngestionPipeline(db)
    pipeline.ollama = FakeEmbeddingOllama()

    with pytest.raises(RuntimeError, match="extract failure"):
        pipeline.process_document("d1")

    db.expire_all()
    assert db.get(Document, "d1").status == DocumentStatus.failed.value
    assert db.get(PipelineRun, "r1").status == RunStatus.failed.value
    assert db.query(DocumentChunk).filter(DocumentChunk.document_id == "d1").count() == 0


def test_process_document_can_complete_rag_only_when_sac_kg_disabled(monkeypatch) -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(id="d1", project_id="p1", title="Case", file_name="case.md", sha256="abc", raw_path="raw/case.md")
    run = PipelineRun(id="r1", project_id="p1", document_id="d1", run_type=RunType.ingest.value, status=RunStatus.queued.value, provider_report={})
    db.add_all([project, document, run])
    db.commit()

    from app.services import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module.settings, "sac_kg_enabled", False)
    monkeypatch.setattr(
        pipeline_module,
        "parse_document",
        lambda path: ParsedDocument(
            title="Parsed",
            text="Parsed text",
            chunks=[ParsedChunk(ordinal=0, text="Parsed text", page_label="1")],
            metadata={
                "canonical": {
                    "quality": {"accepted": True, "status": "accepted"},
                    "table_activation_allowed": True,
                }
            },
        ),
    )

    def fail_if_called(self: IngestionPipeline, processed_document: Document, full_text: str) -> DocumentExtraction:
        raise AssertionError("SAC-KG extraction should be skipped")

    monkeypatch.setattr(IngestionPipeline, "_extract_document", fail_if_called)
    pipeline = IngestionPipeline(db)
    pipeline.ollama = FakeEmbeddingOllama()

    completed = pipeline.process_document("d1")

    db.expire_all()
    assert completed.status == RunStatus.completed.value
    assert db.get(Document, "d1").status == DocumentStatus.ready.value
    assert db.query(DocumentChunk).filter(DocumentChunk.document_id == "d1").count() == 1
    report = db.get(PipelineRun, "r1").provider_report
    assert report["sac_kg_enabled"] is False
    assert report["claims"] == 0
    assert report["progress"]["stage"] == "completed"
    version = db.scalar(
        select(DocumentParseVersion).where(
            DocumentParseVersion.document_id == "d1"
        )
    )
    assert version is not None
    assert version.version_key == "canonical-v1-abc"


@pytest.mark.parametrize(
    "canonical_status,quality_accepted,activation_allowed,issue_code",
    [
        ("validation_failed", False, True, "table_invalid"),
        ("accepted", True, False, "table_invalid"),
        ("rejected", False, True, "page_missing"),
        ("rejected", False, True, "content_empty"),
        ("rejected", False, True, "asset_invalid"),
    ],
)
@pytest.mark.parametrize("suffix", ["pdf", "md"])
def test_process_document_fails_closed_before_chunks_for_canonical_table_failure(
    monkeypatch,
    canonical_status: str,
    quality_accepted: bool,
    activation_allowed: bool,
    issue_code: str,
    suffix: str,
) -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="Case",
        file_name=f"case.{suffix}",
        sha256="abc",
        raw_path=f"raw/case.{suffix}",
    )
    run = PipelineRun(
        id="r1",
        project_id="p1",
        document_id="d1",
        run_type=RunType.ingest.value,
        status=RunStatus.queued.value,
        provider_report={},
    )
    db.add_all([project, document, run])
    db.commit()

    from app.services import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "parse_document",
        lambda _path: ParsedDocument(
            title="Parsed",
            text="Unsafe table evidence",
            chunks=[ParsedChunk(ordinal=0, text="Unsafe table evidence")],
            metadata={
                "canonical": {
                    "quality": {
                        "accepted": quality_accepted,
                        "status": canonical_status,
                        "score": 0.5,
                        "issues": [{"code": issue_code}],
                    },
                    "status": "ready",
                    "table_activation_allowed": activation_allowed,
                    "table_repair_requests": [{"table_id": "table-1"}],
                }
            },
        ),
    )
    monkeypatch.setattr(
        IngestionPipeline,
        "_replace_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rejected canonical evidence must not be persisted")
        ),
    )

    failed = IngestionPipeline(db).process_document("d1")

    db.expire_all()
    stored_document = db.get(Document, "d1")
    stored_run = db.get(PipelineRun, "r1")
    assert failed.status == RunStatus.failed.value
    assert stored_document.status == DocumentStatus.failed.value
    assert db.query(DocumentChunk).filter(DocumentChunk.document_id == "d1").count() == 0
    expected_ingest_status = (
        "validation_failed"
        if canonical_status == "validation_failed" or activation_allowed is False
        else "rejected"
    )
    assert stored_document.metadata_json["ingest_quality"]["status"] == expected_ingest_status
    assert stored_document.metadata_json["ingest_error"]
    assert stored_run.provider_report["ingest_quality"]["status"] == expected_ingest_status
    assert stored_run.provider_report["error"]


def test_ensure_growing_entities_reuses_existing_tail_entity() -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    existing = Entity(id="e1", project_id="p1", name="SAC-KG", entity_type="concept", aliases=[], summary="Existing")
    claim = Claim(
        id="c1",
        project_id="p1",
        document_id="d1",
        subject="Paper",
        predicate="uses",
        object_text="SAC-KG",
        verification_status="verified",
        metadata_json={},
    )
    db.add_all([project, existing, claim])
    db.commit()

    pipeline = IngestionPipeline(db)
    result = pipeline._ensure_growing_entities(
        project_id="p1",
        entities=[],
        claims=[claim],
        entity_decisions={"SAC-KG": pipeline._rule_growth_decision({"name": "SAC-KG", "entity_type": "concept", "claim_count": 1, "item_type": "tail"})},
    )

    assert result == [existing]
    assert db.query(Entity).filter(Entity.project_id == "p1", Entity.name == "SAC-KG").count() == 1


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


def test_pruner_grows_academic_concepts_but_keeps_unsupported_tails() -> None:
    db = make_session()
    pipeline = IngestionPipeline(db)

    concept_decision = pipeline._rule_growth_decision({"name": "Generator", "entity_type": "concept", "claim_count": 1, "item_type": "head"})
    component_decision = pipeline._rule_growth_decision({"name": "Verifier", "entity_type": "component", "claim_count": 1, "item_type": "head"})
    module_decision = pipeline._rule_growth_decision({"name": "Pruner", "entity_type": "module", "claim_count": 1, "item_type": "head"})
    unsupported_tail = pipeline._rule_growth_decision({"name": "Unverified Concept", "entity_type": "concept", "claim_count": 0, "item_type": "tail"})

    assert concept_decision.decision == "grow"
    assert component_decision.decision == "grow"
    assert module_decision.decision == "grow"
    assert unsupported_tail.decision == "keep"


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
        corpus_context="",
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
