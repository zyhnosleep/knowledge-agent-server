"""Frozen scope and rich snapshots, exercised against real canonical files/rows."""
import base64
import hashlib
import json
from dataclasses import replace

import pymupdf
import pytest

from app.models.records import Document, DocumentChunk, DocumentParseVersion
from app.schemas.agent import EvidenceItem, EvidencePack
from app.schemas.common import Citation
from app.services.search import PreparedEvidence, QueryService, RetrievedContext
from app.services.vector_store import PGVectorStore
from app.services.visual_evidence import resolve_context_images
from test_visual_routing import evidence


def add_document(db, root, document_id="d2"):
    db.add(Document(id=document_id, project_id="p1", title="Additional Paper", file_name="extra.pdf",
        raw_path="raw/extra.pdf", sha256="extra", status="ready", active_parse_version="v5"))
    db.flush()
    bundle = root / document_id / "v5"
    (bundle / "assets").mkdir(parents=True)
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=40, height=30)
        page.draw_rect(page.rect, fill=(1, 0, 0))
        page.get_pixmap().save(bundle / "assets/figure.png")
    digest = hashlib.sha256((bundle / "assets/figure.png").read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps({"document_id": document_id, "version": "v5",
        "assets": [{"path": "assets/figure.png", "sha256": digest}]}), encoding="utf-8")
    (bundle / "figures.json").write_text(json.dumps([{"id": "figure-2", "asset_path": "assets/figure.png"}]), encoding="utf-8")
    db.add(DocumentParseVersion(document_id=document_id, version_key="v5", artifact_dir=str(bundle), status="active",
        manifest_json={"ingestion_config": {"embedding": {"provider": "ollama", "model": "qwen3-embedding:4b", "dimensions": 2}}}))
    db.add(DocumentChunk(id=document_id + "-figure", document_id=document_id, parse_version="v5",
        chunk_role="child", block_type="figure", ordinal=0, source_spans=[{"figure_id": "figure-2", "page_index": 0}],
        text="Figure 2: legend. ![figure](assets/figure.png)", embedding=[1.0, 0.0]))
    db.commit()


def snapshot(service, contexts, *, versions=None, document_id=None, question="Compare Figure 2 and Table 3"):
    items, facts = service._evidence_items_and_facts(contexts, len(contexts), question=question)
    return PreparedEvidence("p1", "pilot", question, document_id, versions or {"d1": "v5", "d2": "v5"},
        contexts, EvidencePack(status="ok", items=items, table_facts=facts))


def test_supplement_cannot_add_newly_created_document(evidence):
    db, service, root = evidence
    initial = service.prepare_evidence("pilot", "What is in Figure 1?")
    add_document(db, root, "new_document")
    supplement_service = QueryService(db, parse_version_map=initial.parse_version_map)
    supplement_service._retrieval_token_counter = lambda text: max(1, len(text.split()))
    supplement_service.ollama.embed = lambda texts: [[1, 0] for _ in texts]
    supplement = supplement_service.prepare_evidence("pilot", "Figure 2 legend")
    assert supplement.parse_version_map == initial.parse_version_map
    assert "new_document" not in supplement.parse_version_map
    assert {item.document_id for item in supplement.pack.items} <= {"d1"}
    raw_hits = supplement_service._search_source_chunks("legend", "p1", [], question_vector=[1, 0])
    assert {context.citation.document_id for context in raw_hits} <= {"d1"}


def test_empty_frozen_map_does_not_refresh_active_versions(evidence):
    db, service, root = evidence
    service.parse_version_map = {}
    calls = []
    service.ollama.embed = lambda texts: calls.append(texts) or [[1, 0] for _ in texts]
    empty = service.prepare_evidence("pilot", "What is in Figure 1?")
    assert empty.parse_version_map == {}
    assert empty.contexts == [] and empty.pack.items == []
    assert calls == []


def test_focus_document_cannot_expand_request_scope(evidence):
    db, service, root = evidence
    add_document(db, root)
    service.parse_version_map = {"d1": "v5"}
    with pytest.raises(ValueError, match="frozen|scope"):
        service.prepare_evidence("pilot", "Figure 2 legend", document_id="d2")


def test_explicit_prepare_map_is_supported_and_copied(evidence):
    db, service, root = evidence
    versions = {"d1": "v5"}
    prepared = service.prepare_evidence("pilot", "Figure 1 legend", parse_version_map=versions)
    versions["d1"] = "v4"
    assert prepared.parse_version_map == {"d1": "v5"}
    db.get(Document, "d1").active_parse_version = "v6"
    supplemental = service.prepare_evidence("pilot", "Figure 1 legend", parse_version_map=prepared.parse_version_map)
    assert {item.parse_version for item in supplemental.pack.items} == {"v5"}


def test_empty_frozen_vector_filter_does_not_query_active_rows():
    class Database:
        def execute(self, statement, parameters):
            self.sql = str(statement)
            return type("Result", (), {"all": lambda self: []})()
    db = Database()
    PGVectorStore(db)._search_rows([1, 0], 5, [], parse_version_map={})
    assert "WHERE FALSE" in db.sql


def test_merge_keeps_two_same_named_figures_and_sends_both_pixels(evidence, monkeypatch):
    db, service, root = evidence
    add_document(db, root)
    db.get(DocumentChunk, "v5-fig").text = "Figure 2: legend. ![figure](assets/figure.png)"
    db.commit()
    service.parse_version_map = {"d1": "v5", "d2": "v5"}
    first = service._search_source_chunks("legend", "p1", ["d1"], question_vector=[1, 0])
    second = service._search_source_chunks("legend", "p1", ["d2"], question_vector=[1, 0])
    merged = service.merge_prepared_snapshots(snapshot(service, first), snapshot(service, second, document_id="d2"))
    assert {item.document_id for item in merged.pack.items} == {"d1", "d2"}
    assert [item.index for item in merged.pack.items] == [0, 1]
    payloads = []
    monkeypatch.setattr(service.ollama, "_post_chat", lambda payload: payloads.append(payload) or
        {"message": {"content": '{"answer_markdown":"Both plots have legends [0][1]","citations":[0,1]}'}})
    response = service.answer_from_evidence("pilot", "Compare the legends in Figure 2", merged, visual_intent=True)
    sent = {base64.b64decode(encoded) for encoded in payloads[0]["messages"][1]["images"]}
    assert sent == {(root / "d1/v5/assets/figure.png").read_bytes(), (root / "d2/v5/assets/figure.png").read_bytes()}
    assert {citation.document_id for citation in response.citations} == {"d1", "d2"}
    assert {citation.parse_version for citation in response.citations} == {"v5"}


def test_merge_rebuilds_table_facts_not_forged_wire_values(evidence):
    db, service, root = evidence
    db.add(DocumentChunk(id="v5-table", document_id="d1", parse_version="v5", chunk_role="child",
        block_type="table", ordinal=1, source_block_ids=["table-3"], source_spans=[{"table_id": "table-3", "page_index": 0}],
        text="Table 3\n| Model | Accuracy |\n|---|---|\n| A | 94% |", embedding=[1, 0]))
    db.commit()
    figures = [context for context in service._search_source_chunks("legend", "p1", ["d1"], question_vector=[1, 0])
               if context.evidence_kind == "figure"]
    tables = service._search_source_chunks("Table 3 Model A accuracy", "p1", ["d1"], question_vector=[1, 0])
    base = snapshot(service, figures, versions={"d1": "v5"})
    extra = snapshot(service, tables, versions={"d1": "v5"})
    assert extra.pack.table_facts
    extra.pack.table_facts[0].value = "999%"
    merged = service.merge_prepared_snapshots(base, extra)
    assert any(fact.value == "94%" for fact in merged.pack.table_facts)
    assert not any(fact.value == "999%" for fact in merged.pack.table_facts)
    assert all(fact.document_id == "d1" and fact.parse_version == "v5" for fact in merged.pack.table_facts)
    assert [item.index for item in merged.pack.items] == list(range(len(merged.pack.items)))


def test_merge_drops_cross_version_chunk_from_every_channel(evidence):
    db, service, root = evidence
    legal = service._search_source_chunks("legend", "p1", ["d1"], question_vector=[1, 0])
    base = snapshot(service, legal, versions={"d1": "v5"})
    forged = RetrievedContext(Citation(document_id="d1", chunk_id="v4-fig", parse_version="v4",
        block_type="figure", excerpt="FORGED", score=100), "FORGED", 100, evidence_kind="figure")
    extra = snapshot(service, [forged], versions={"d1": "v5"})
    merged = service.merge_prepared_snapshots(base, extra)
    assert [context.citation.chunk_id for context in merged.contexts] == ["v5-fig"]
    assert [item.chunk_id for item in merged.pack.items] == ["v5-fig"]
    assert "FORGED" not in str(merged.pack.model_dump())


def test_merge_rejects_different_project_or_frozen_version_map(evidence):
    db, service, root = evidence
    base = snapshot(service, [], versions={"d1": "v5"})
    for extra in (replace(base, project_slug="other"), replace(base, parse_version_map={"d1": "v6"})):
        with pytest.raises(ValueError, match="scope|snapshot|version"):
            service.merge_prepared_snapshots(base, extra)


def test_empty_frozen_map_cannot_resolve_active_pixels(evidence):
    db, service, root = evidence
    contexts = service._search_source_chunks("legend", "p1", ["d1"], question_vector=[1, 0])
    db.get(Document, "d1").active_parse_version = "v5"
    images, sent, skipped = resolve_context_images(db, contexts, root, version_map={})
    assert images == [] and sent == []
    assert skipped


def test_strict_figure_route_never_scores_json_embeddings(evidence, monkeypatch):
    import app.services.search as module
    db, service, root = evidence
    monkeypatch.setattr(module.settings, "vector_store_strict", True)
    store = type("Store", (), {"search": lambda self, *a, **kw: []})()
    monkeypatch.setattr(module, "get_vector_store", lambda db: store)
    def forbidden(*args):
        pytest.fail("figure route scored JSON instead of pgvector")
    monkeypatch.setattr(module, "cosine_similarity", forbidden)
    contexts = service._search_document_figure_contexts("Figure 1 legend", "p1", ["d1"])
    assert contexts
    assert max(context.score for context in contexts) <= 1.2
    assert service.retrieval_backend == "pgvector"


def test_table_fill_cannot_use_unmapped_document_via_cartesian_join(evidence):
    db, service, root = evidence
    add_document(db, root)
    db.add(DocumentChunk(id="d2-table", document_id="d2", parse_version="v5", chunk_role="child",
        block_type="table", ordinal=1, source_spans=[{"table_id": "table-3"}],
        text="Table 3\n| Model | Accuracy |\n|---|---|\n| A | 98% |", embedding=[1, 0]))
    db.commit()
    service.parse_version_map = {"d1": "v5"}
    assert service._fill_requested_table_contexts("Table 3 accuracy", project_id="p1",
        missing_table_ids=["table-3"], requested_table_scopes=[("d2", "v5", "table-3")]) == []


def test_same_named_table_fill_preserves_both_document_scopes(evidence):
    db, service, root = evidence
    add_document(db, root)
    for doc, value in (("d1", "94%"), ("d2", "98%")):
        db.add(DocumentChunk(id=doc + "-table", document_id=doc, parse_version="v5", chunk_role="child",
            block_type="table", ordinal=1, source_spans=[{"table_id": "table-3"}],
            text="Table 3\n| Model | Accuracy |\n|---|---|\n| A | " + value + " |", embedding=[1, 0]))
    db.commit()
    service.parse_version_map = {"d1": "v5", "d2": "v5"}
    contexts = service._fill_requested_table_contexts("Table 3 accuracy", project_id="p1",
        missing_table_ids=["table-3"], requested_table_scopes=[("d1", "v5", "table-3"), ("d2", "v5", "table-3")])
    assert {context.citation.document_id for context in contexts} == {"d1", "d2"}


def test_one_filled_table_cannot_complete_same_named_missing_table(evidence, monkeypatch):
    from app.services.canonical_artifacts import CanonicalArtifactStore
    db, service, root = evidence
    add_document(db, root)
    db.add(DocumentChunk(id="d1-table", document_id="d1", parse_version="v5", chunk_role="child",
        block_type="table", ordinal=1, source_spans=[{"table_id": "table-3"}],
        text="Table 3\n| Model | Accuracy |\n|---|---|\n| A | 94% |", embedding=[1, 0]))
    db.commit()
    contexts = [RetrievedContext(Citation(document_id=doc, parse_version="v5", table_id="table-3",
        block_type="table", excerpt="Table 3 accuracy", score=5), "Table 3 accuracy", 5,
        evidence_kind="table") for doc in ("d1", "d2")]
    monkeypatch.setattr(service, "_build_rag_contexts", lambda *a, **kw: contexts)
    monkeypatch.setattr(CanonicalArtifactStore, "load_typed_inventory", lambda self, doc, version:
        {"tables": [{"table_id": "table-3", "row_count": 1, "row_indices": [1]}]})
    prepared = service.prepare_evidence("pilot", "Compare Table 3 accuracy in both papers",
        parse_version_map={"d1": "v5", "d2": "v5"})
    assert prepared.pack.coverage_status == "partial"
    assert "table-3" in prepared.pack.coverage_missing_tables
