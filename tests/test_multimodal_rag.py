from types import SimpleNamespace

import httpx
import pymupdf
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.session import Base
from app.models.records import Document, Project
from app.schemas.agent import EvidenceItem, EvidencePack
from app.services import multimodal_rag as module


def test_physical_page_index_takes_precedence_over_display_label():
    item = EvidenceItem(index=0, page_label="iv", source_spans=[
        {"page_index": 3}, {"page_index": 3}, {"page_index": True}])
    assert module.source_page_numbers(item) == [4]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all([Project(id="p1", slug="pilot", name="Pilot"),
                    Project(id="p2", slug="other", name="Other")])
        db.flush()
        pdf_path = tmp_path / "paper.pdf"
        with pymupdf.open() as pdf:
            for i in range(5):
                pdf.new_page().insert_text((70, 70), f"Physical page {i + 1}")
            pdf.save(pdf_path)
        for identifier, project in [("doc1", "p1"), ("doc2", "p2")]:
            db.add(Document(id=identifier, project_id=project, title="Paper",
                            file_name="paper.pdf", raw_path=str(pdf_path),
                            sha256=identifier, metadata_json={"evaluation_paper_id": identifier}))
        db.commit()
        service = module.MultimodalRAG(db)
        service.settings = SimpleNamespace(cache_dir=tmp_path / "cache")
        yield service
    engine.dispose()


def test_retrieves_once_without_gold_scope_and_preserves_ranked_pages(setup, monkeypatch):
    calls = []
    def retrieve(self, project_slug, question, limit):
        calls.append((project_slug, question, limit))
        return EvidencePack(status="ok", items=[
            EvidenceItem(index=0, document_id="doc1", source_spans=[{"page_index": 2}]),
            EvidenceItem(index=1, document_id="doc1", source_spans=[{"page_index": 2}]),
            EvidenceItem(index=2, document_id="doc1", source_spans=[{"page_index": 0}]),
            EvidenceItem(index=3, document_id="doc1", source_spans=[{"page_index": 4}]),
            EvidenceItem(index=4, document_id="doc1", source_spans=[{"page_index": 3}]),
        ])
    monkeypatch.setattr(module.QueryService, "retrieve_evidence", retrieve)
    result = setup.retrieve("pilot", "Question", max_pages=3)
    assert calls == [("pilot", "Question", 15)]
    assert [p["page"] for p in result["pages"]] == [3, 1, 5]
    assert "Physical page 3" in result["pages"][0]["text"]


def test_cross_project_evidence_is_not_rendered(setup, monkeypatch):
    monkeypatch.setattr(module.QueryService, "retrieve_evidence", lambda *a, **kw:
                        EvidencePack(status="ok", items=[EvidenceItem(
                            index=0, document_id="doc2", page_label="1")]))
    assert setup.retrieve("pilot", "Question")["pages"] == []


def test_empty_retrieval_abstains_without_model_call(setup):
    result = setup.generate({"question": "Question", "pages": []}, "text_image")
    assert result["parsed_output"]["abstain"] is True
    assert result["generation_skipped"] == "no_pages"


@pytest.mark.parametrize("raw,valid,citations_valid", [
    ('{"answer":"A","abstain":false,"citations":["p#page=1"]}', True, True),
    ('{"answer":"A","abstain":"false","citations":[]}', False, False),
    ('{"answer":"A","abstain":false,"citations":["unknown"]}', True, False),
])
def test_paired_modes_share_text_and_validate_output(setup, monkeypatch, raw, valid, citations_valid):
    requests = []
    class Client:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, json):
            requests.append(json)
            return httpx.Response(200, json={"raw_output": raw}, request=httpx.Request("POST", url))
    monkeypatch.setattr(module.httpx, "Client", Client)
    retrieval = {"question": "Question", "pages": [
        {"page_id": "p#page=1", "text": "Evidence", "image_path": "/cache/page.png"}]}
    for mode in ("text", "text_image"):
        result = setup.generate(retrieval, mode)
        assert result["schema_valid"] is valid
        assert result["citation_ids_valid"] is citations_valid
    text_content = requests[0]["messages"][1]["content"]
    image_content = requests[1]["messages"][1]["content"]
    assert [p for p in image_content if p["type"] == "text"] == text_content
    assert [p for p in image_content if p["type"] == "image"] == [
        {"type": "image", "image": "/cache/page.png"}]
