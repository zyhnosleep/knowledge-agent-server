from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.session import Base
from app.models.records import Document, PageKind, Project, WikiPage
from app.services.table_extraction import structure_table_markdown, summarize_ablation_table, table_metric_values
from app.services.wiki_quality import build_ingest_quality_report, lint_project_wiki


def make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)()


def test_structure_table_markdown_extracts_table5_metrics() -> None:
    markdown = (
        "Table 5: F1 score and AUC results.\n"
        "| Model | OIE2016 | OIE2016 | NYT | NYT |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | F1 | AUC | F1 | AUC |\n"
        "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |"
    )

    table = structure_table_markdown(markdown)
    metrics = table_metric_values(markdown, ["OIE2016", "NYT"])

    assert table.label == "Table 5"
    assert table.headers == ["Model", "OIE2016 F1", "OIE2016 AUC", "NYT F1", "NYT AUC"]
    assert metrics[0]["values"] == {"F1": "74.7", "AUC": "73.2"}
    assert metrics[1]["values"] == {"F1": "88.8", "AUC": "87.3"}


def test_summarize_ablation_table_reports_full_model_rows() -> None:
    markdown = (
        "Table 2: Ablation study.\n"
        "| Iteration rounds | Model | Number of recalls | Precision | Domain Specificity |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Iteration 1 | SAC-KG w/o prompt | 10.15 | 80.64 | 74.19 |\n"
        "| Iteration 1 | SAC-KG | 13.50 | 88.81 | 80.50 |\n"
    )

    findings = summarize_ablation_table(markdown)

    assert findings
    assert "Iteration 1" in findings[0]
    assert "precision 88.81" in findings[0]


def test_build_ingest_quality_report_summarizes_tables() -> None:
    document = Document(
        id="d1",
        project_id="p1",
        title="Paper",
        file_name="paper.pdf",
        sha256="abc",
        raw_path="raw/paper.pdf",
        metadata_json={
            "parser_mode": "pdf_mineru",
            "pages": 8,
            "document_intelligence": {
                "tables": [
                    {
                        "page_label": "8",
                        "markdown": (
                            "Table 5: Results\n"
                            "| Model | OIE2016 | OIE2016 |\n"
                            "| --- | --- | --- |\n"
                            "|  | F1 | AUC |\n"
                            "| SAC-KG ChatGPT | 74.7 | 73.2 |"
                        ),
                    }
                ],
                "figures": [],
                "formulas": [],
            },
        },
    )

    report = build_ingest_quality_report(document)

    assert report["table_count"] == 1
    assert report["structured_table_count"] == 1
    assert report["tables"][0]["metric_values"][0]["values"] == {"F1": "74.7", "AUC": "73.2"}


def test_lint_project_wiki_reports_table_quality_and_unresolved_citations(tmp_path, monkeypatch) -> None:
    db = make_session()
    project = Project(id="p1", slug="demo", name="Demo")
    source = WikiPage(
        id="w1",
        project_id="p1",
        slug="sources/paper",
        title="Paper",
        kind=PageKind.source_summary.value,
        markdown_path=str(tmp_path / "sources" / "paper.md"),
        markdown_content="# Paper\n\n## Tables\n### Page 1\nTable 1: Empty\n",
        source_document_ids=["d1"],
    )
    query = WikiPage(
        id="w2",
        project_id="p1",
        slug="queries/q",
        title="Question",
        kind=PageKind.query_answer.value,
        markdown_path=str(tmp_path / "queries" / "q.md"),
        markdown_content="# Question\n\nAnswer cites [Knowledge graph].",
        source_document_ids=[],
    )
    db.add_all([project, source, query])
    db.commit()
    wiki_root = tmp_path / "wiki" / "demo"
    wiki_root.mkdir(parents=True)
    (wiki_root / "index.md").write_text("- [Paper](sources/paper.md)\n", encoding="utf-8")
    from app.services import wiki_quality

    monkeypatch.setattr(wiki_quality.settings, "wiki_dir", tmp_path / "wiki")

    report = lint_project_wiki(db, "demo")

    kinds = {issue["kind"] for issue in report["issues"]}
    assert "tables_section_empty" in kinds
    assert "unresolved_citation_labels" in kinds
