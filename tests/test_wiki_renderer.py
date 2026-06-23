from pathlib import Path

from app.models.records import Claim, Document, Entity, Project, WikiPage
from app.services.ai import DocumentExtraction
from app.services.wiki import WikiRenderer


def test_render_document_summary_contains_sections() -> None:
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="Doc",
        file_name="doc.txt",
        sha256="x",
        raw_path="raw/doc.txt",
        metadata_json={
            "document_intelligence": {
                "page_outputs": [{"page_label": "1", "page_summary": "结构摘要", "text_quality": "high"}],
                "tables": [{"page_label": "1", "markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"}],
                "formulas": [{"page_label": "1", "text": "E = mc^2"}],
                "figures": [{"page_label": "1", "note": "图注说明"}],
            }
        },
    )
    extraction = DocumentExtraction(title="Doc", summary="Summary body", keywords=["rl", "control"], concepts=["Policy"])
    renderer = WikiRenderer(project)

    slug, markdown = renderer.render_document_summary(document, extraction)

    assert slug == "sources/doc"
    assert "# Doc" in markdown
    assert "## Summary" in markdown
    assert "Policy" in markdown
    assert "## Page Structure" in markdown
    assert "## Tables" in markdown
    assert "## Formulas" in markdown


def test_render_entity_pages_collects_claims() -> None:
    project = Project(id="p1", slug="demo", name="Demo")
    renderer = WikiRenderer(project)
    entity = Entity(id="e1", project_id="p1", name="Method A", entity_type="method", aliases=[], summary="Entity summary")
    claim = Claim(
        id="c1",
        project_id="p1",
        document_id="d1",
        subject="Method A",
        predicate="outperforms",
        object_text="baseline B",
        confidence=0.9,
        verification_status="verified",
    )

    pages = renderer.render_entity_pages([entity], [claim])

    assert len(pages) == 1
    assert pages[0][1] == "entities/method-a"
    assert 'title: "Method A"' in pages[0][2]
    assert "outperforms baseline B" in pages[0][2]


def test_render_entity_pages_disambiguates_duplicate_slugs() -> None:
    project = Project(id="p1", slug="demo", name="Demo")
    renderer = WikiRenderer(project)
    entity_links: dict[str, str] = {}
    entities = [
        Entity(id="e1", project_id="p1", name="OPLS4", entity_type="method", aliases=[], summary="Base model"),
        Entity(id="e2", project_id="p1", name="OPLS4*", entity_type="method", aliases=[], summary="Variant model"),
    ]

    pages = renderer.render_entity_pages(entities, [], entity_links=entity_links)

    assert [page[1] for page in pages] == ["entities/opls4", "entities/opls4-2"]
    assert entity_links == {"OPLS4": "entities/opls4", "OPLS4*": "entities/opls4-2"}
    assert 'title: "OPLS4"' in pages[0][2]
    assert 'title: "OPLS4*"' in pages[1][2]


def test_render_index_uses_slug_when_markdown_path_has_windows_separators(tmp_path: Path) -> None:
    project = Project(id="p1", slug="demo", name="Demo")
    renderer = WikiRenderer(project)
    renderer.paths["wiki_root"] = tmp_path

    page = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/sample-doc",
        title="Sample Doc",
        kind="source_summary",
        markdown_path=r"data\\wiki\\demo\\sources\\sample-doc.md",
        markdown_content="# Sample Doc",
        source_document_ids=["d1"],
    )

    index_path = renderer.render_index([page])

    assert index_path == tmp_path / "index.md"
    assert "[Sample Doc](sources/sample-doc.md)" in index_path.read_text(encoding="utf-8")


def test_render_document_summary_keeps_more_tables_and_figures() -> None:
    project = Project(id="p1", slug="demo", name="Demo")
    document = Document(
        id="d1",
        project_id="p1",
        title="Doc",
        file_name="doc.pdf",
        sha256="x",
        raw_path="raw/doc.pdf",
        metadata_json={
            "document_intelligence": {
                "page_outputs": [{"page_label": str(i), "page_summary": f"Page {i}", "text_quality": "mineru"} for i in range(1, 14)],
                "tables": [{"page_label": str(i), "markdown": f"Table {i} content"} for i in range(1, 10)],
                "figures": [{"page_label": str(i), "note": f"Figure {i} note"} for i in range(1, 14)],
            }
        },
    )
    extraction = DocumentExtraction(title="Doc", summary="Summary body")
    renderer = WikiRenderer(project)

    _, markdown = renderer.render_document_summary(document, extraction)

    assert "Table 8 content" in markdown
    assert "Table 9 content" not in markdown
    assert "Figure 12 note" in markdown
    assert "Figure 13 note" not in markdown
