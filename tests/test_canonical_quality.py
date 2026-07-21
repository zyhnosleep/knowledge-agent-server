from __future__ import annotations

from pathlib import Path

import pytest

from app.services import canonical_adapters, parser
from app.services.canonical_adapters import PDFCanonicalAdapter
from app.services.canonical_models import (
    CanonicalAsset,
    CanonicalBlock,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalQualityIssue,
    CanonicalTable,
    SourceSpan,
)
from app.services.canonical_quality import CanonicalQualityGate


def _block(
    text: str = "Body text",
    *,
    order: int = 0,
    page: int = 0,
    parser_source: str = "mineru",
    block_id: str | None = None,
) -> CanonicalBlock:
    return CanonicalBlock(
        block_id=block_id or f"block-{page}-{order}",
        block_type="narrative",
        text=text,
        reading_order=order,
        source_spans=[SourceSpan(page_index=page, page_label=str(page + 1))],
        parser_source=parser_source,
    )


def _document(*blocks: CanonicalBlock, **metadata: object) -> CanonicalDocument:
    return CanonicalDocument(
        document_id="doc-1",
        parser_source="mineru",
        parse_version="canonical-v1",
        title="Paper",
        blocks=list(blocks) or [_block()],
        metadata=dict(metadata),
    )


def _accepted_mineru_document(page_count: int = 1) -> CanonicalDocument:
    return _document(
        _block(),
        expected_page_count=page_count,
        parsed_page_indices=list(range(page_count)),
    )


def test_quality_gate_accepts_a_valid_document_and_writes_report() -> None:
    document = _document(_block())

    report = CanonicalQualityGate().evaluate(document)

    assert report is document.quality
    assert report.accepted is True
    assert report.status == "accepted"
    assert report.score == 1.0
    assert report.issues == []


def test_quality_gate_detects_missing_pages_only_with_explicit_page_contract() -> None:
    gated = _document(
        _block(page=0),
        expected_page_count=2,
        parsed_page_indices=[0],
    )
    ungated = _document(_block(page=0))

    gated_report = CanonicalQualityGate().evaluate(gated)
    ungated_report = CanonicalQualityGate().evaluate(ungated)

    assert [(issue.code, issue.severity) for issue in gated_report.issues] == [
        ("page_missing", "fatal")
    ]
    assert gated_report.status == "rejected"
    assert all(issue.code != "page_missing" for issue in ungated_report.issues)


def test_quality_gate_detects_empty_content() -> None:
    document = _document()
    document.blocks = []

    report = CanonicalQualityGate().evaluate(document)

    assert [(issue.code, issue.severity) for issue in report.issues] == [
        ("content_empty", "fatal")
    ]
    assert report.accepted is False
    assert report.status == "rejected"
    assert report.score == 0.65


def test_quality_gate_does_not_treat_heading_only_document_as_body_content() -> None:
    heading = CanonicalBlock(
        block_id="heading-1",
        block_type="heading",
        text="Paper title",
        reading_order=0,
        parser_source="mineru",
    )

    report = CanonicalQualityGate().evaluate(_document(heading))

    assert [issue.code for issue in report.issues] == ["content_empty"]


def test_quality_gate_detects_invalid_reading_order() -> None:
    document = _document(_block(order=1))

    report = CanonicalQualityGate().evaluate(document)

    assert report.issues[0].code == "reading_order_invalid"
    assert report.issues[0].severity == "fatal"


def test_quality_gate_detects_asset_references_outside_assets() -> None:
    document = _document(_block())
    document.assets = [
        CanonicalAsset(
            asset_id="asset-1",
            path="../outside.png",
            media_type="image/png",
        )
    ]

    report = CanonicalQualityGate().evaluate(document)

    assert report.issues[0].code == "asset_invalid"
    assert report.issues[0].severity == "fatal"


def test_quality_gate_requests_targeted_abstract_repair() -> None:
    document = _document(
        _block("Introduction body"),
        text_layer_pages=["Abstract\nThis paper introduces SAC-KG.", "Introduction"],
    )

    report = CanonicalQualityGate().evaluate(document)

    issue = report.issues[0]
    assert issue.code == "abstract_missing"
    assert issue.severity == "error"
    assert issue.repairable is True
    assert issue.repair_scope == "pages:1-2"
    assert report.fallback_pages == [1, 2]
    assert report.accepted is False
    assert report.status == "validation_failed"


def test_quality_gate_requests_table_page_repair() -> None:
    document = _document(_block())
    document.tables = [
        CanonicalTable(
            table_id="table-1",
            headers=["Model", "F1"],
            rows=[["SAC-KG"]],
            source_spans=[SourceSpan(page_index=2, page_label="3")],
        )
    ]

    report = CanonicalQualityGate().evaluate(document)

    issue = report.issues[0]
    assert issue.code == "table_invalid"
    assert issue.severity == "error"
    assert issue.repairable is True
    assert issue.repair_scope == "page:3"
    assert report.fallback_pages == [3]
    assert report.status == "validation_failed"


def test_figure_and_formula_warnings_do_not_block_acceptance() -> None:
    document = _document(_block())
    document.figures = [CanonicalFigure(figure_id="figure-1")]
    document.formulas = [CanonicalFormula(formula_id="formula-1", latex="x=1")]

    report = CanonicalQualityGate().evaluate(document)

    assert [(issue.code, issue.severity, issue.repairable) for issue in report.issues] == [
        ("figure_caption_missing", "warning", False),
        ("formula_analysis_missing", "warning", False),
    ]
    assert report.accepted is True
    assert report.status == "accepted_with_warnings"
    assert report.score == 0.9


def test_quality_issue_order_and_score_are_deterministic() -> None:
    document = _document(
        _block(order=1, page=0),
        expected_page_count=2,
        parsed_page_indices=[0],
        text_layer_pages=["Abstract\nSource abstract", ""],
    )
    document.assets = [CanonicalAsset(asset_id="a", path="/absolute.png", media_type="image/png")]
    document.tables = [CanonicalTable(table_id="t", headers=["A", "B"], rows=[["1"]])]
    document.figures = [CanonicalFigure(figure_id="f")]
    document.formulas = [CanonicalFormula(formula_id="m", latex="x")]

    first = CanonicalQualityGate().evaluate(document)
    second = CanonicalQualityGate().evaluate(document)

    expected_codes = [
        "page_missing",
        "reading_order_invalid",
        "asset_invalid",
        "abstract_missing",
        "table_invalid",
        "figure_caption_missing",
        "formula_analysis_missing",
    ]
    assert [issue.code for issue in first.issues] == expected_codes
    assert second.model_dump(mode="json") == first.model_dump(mode="json")
    assert first.score == 0.0


def test_quality_issue_repair_scope_round_trips_through_json() -> None:
    issue = CanonicalQualityIssue(
        code="table_invalid",
        severity="error",
        message="table needs repair",
        repairable=True,
        repair_scope="page:7",
    )

    restored = CanonicalQualityIssue.model_validate_json(issue.model_dump_json())

    assert restored.repair_scope == "page:7"


def test_pdf_uses_mineru_when_text_layer_extraction_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")
    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(
        parser,
        "_extract_pdf_text_layer",
        lambda _path: (_ for _ in ()).throw(RuntimeError("page text failed")),
    )
    monkeypatch.setattr(
        canonical_adapters,
        "run_mineru",
        lambda _path, _page_count: _accepted_mineru_document(),
    )
    monkeypatch.setattr(
        canonical_adapters,
        "run_document_intelligence",
        lambda *_args, **_kwargs: pytest.fail("DI must not run for accepted MinerU"),
    )

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert result.metadata["primary_parser"] == "mineru"
    assert result.metadata["text_layer_warnings"]
    assert result.quality.accepted is True


def test_pdf_repairable_issue_calls_document_intelligence_only_for_scoped_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")
    mineru = _accepted_mineru_document(page_count=3)
    mineru.blocks = [
        _block("Abstract source text was left as narrative", page=0),
        _block("Unchanged page three", order=1, page=2),
    ]
    repaired = _document(
        _block("Abstract\nRecovered source abstract", page=0, parser_source="document_intelligence"),
        expected_page_count=3,
        parsed_page_indices=[0],
    )
    repaired.parser_source = "document_intelligence"
    repaired.abstract = "Recovered source abstract"
    calls: list[set[int] | None] = []

    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 3)
    monkeypatch.setattr(
        parser,
        "_extract_pdf_text_layer",
        lambda _path: (["Abstract\nRecovered source abstract", "", ""], 3),
    )
    monkeypatch.setattr(canonical_adapters, "run_mineru", lambda *_args: mineru)

    def fake_di(_path, _page_count, _page_texts, page_indices=None):
        calls.append(None if page_indices is None else set(page_indices))
        return repaired

    monkeypatch.setattr(canonical_adapters, "run_document_intelligence", fake_di)

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert calls == [{0, 1}]
    assert result.abstract == "Recovered source abstract"
    assert any(block.text == "Unchanged page three" for block in result.blocks)
    assert len({block.block_id for block in result.blocks}) == len(result.blocks)
    assert [block.reading_order for block in result.blocks] == list(range(len(result.blocks)))
    assert result.quality.accepted is True
    assert result.metadata["repair_scopes"] == ["pages:1-2"]
    assert result.metadata["fallback_pages"] == [1, 2]
    assert result.metadata["quality"]["status"] == "accepted"


def test_pdf_fatal_mineru_quality_calls_full_document_intelligence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")
    fatal = _accepted_mineru_document()
    fatal.blocks = []
    full_di = _accepted_mineru_document()
    full_di.parser_source = "document_intelligence"
    calls: list[object] = []

    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(parser, "_extract_pdf_text_layer", lambda _path: (["Body"], 1))
    monkeypatch.setattr(canonical_adapters, "run_mineru", lambda *_args: fatal)

    def fake_di(_path, _page_count, _page_texts, page_indices=None):
        calls.append(page_indices)
        return full_di

    monkeypatch.setattr(canonical_adapters, "run_document_intelligence", fake_di)

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert calls == [None]
    assert result.metadata["primary_parser"] == "document_intelligence"
    assert result.quality.accepted is True


def test_pdf_targeted_repair_failure_does_not_escalate_to_full_document_intelligence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")
    mineru = _accepted_mineru_document()
    calls: list[set[int] | None] = []

    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(
        parser,
        "_extract_pdf_text_layer",
        lambda _path: (["Abstract\nSource abstract"], 1),
    )
    monkeypatch.setattr(canonical_adapters, "run_mineru", lambda *_args: mineru)

    def unavailable_di(_path, _page_count, _page_texts, page_indices=None):
        calls.append(None if page_indices is None else set(page_indices))
        return None

    monkeypatch.setattr(canonical_adapters, "run_document_intelligence", unavailable_di)

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert calls == [{0}]
    assert result.metadata["primary_parser"] == "mineru"
    assert result.quality.status == "validation_failed"


def test_pdf_audit_preserves_best_effort_text_layer_warnings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")

    def extract(_path, *, warnings_out=None):
        warnings_out.append("Unable to extract text from page 1: bad stream")
        return ([""], 1)

    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(parser, "_extract_pdf_text_layer", extract)
    monkeypatch.setattr(
        canonical_adapters,
        "run_mineru",
        lambda *_args: _accepted_mineru_document(),
    )

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert result.metadata["text_layer_warnings"] == [
        "Unable to extract text from page 1: bad stream"
    ]


def test_pdf_mineru_exception_and_di_failure_use_complete_text_layer_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")
    long_text = "source-" * 900

    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(parser, "_extract_pdf_text_layer", lambda _path: ([long_text], 1))
    monkeypatch.setattr(
        canonical_adapters,
        "run_mineru",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("mineru crashed")),
    )
    monkeypatch.setattr(canonical_adapters, "run_document_intelligence", lambda *_args, **_kwargs: None)

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert result.metadata["primary_parser"] == "pypdf_text_layer"
    assert result.blocks[0].text == long_text
    assert any("mineru" in attempt and "failed" in attempt for attempt in result.metadata["parser_attempts"])


def test_mineru_long_table_conversion_is_not_truncated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = [f"| row-{index} | {index} |" for index in range(600)]
    markdown = "\n".join(["| Name | Value |", "| --- | --- |", *rows])
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fixture")
    parsed = parser._mineru_content_to_parsed_doc(
        path=pdf_path,
        content_list=[{"type": "table", "table_body": markdown, "page_idx": 0}],
        page_count=1,
    )
    monkeypatch.setattr(parser, "_parse_pdf_with_mineru", lambda *_args: parsed)

    document = canonical_adapters.run_mineru(pdf_path, 1)

    assert document is not None
    table_block = next(block for block in document.blocks if block.block_type == "table")
    assert table_block.text.endswith("| row-599 | 599 |")
    assert document.tables[0].rows[-1] == ["row-599", "599"]


def test_mineru_figure_metadata_is_linked_to_a_figure_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fixture")
    parsed = parser._mineru_content_to_parsed_doc(
        path=pdf_path,
        content_list=[
            {
                "type": "image",
                "image_caption": "Figure 1. Architecture",
                "image_note": "Generator and verifier",
                "img_path": "images/figure-1.png",
                "page_idx": 0,
            }
        ],
        page_count=1,
    )
    monkeypatch.setattr(parser, "_parse_pdf_with_mineru", lambda *_args: parsed)

    document = canonical_adapters.run_mineru(pdf_path, 1)

    assert document is not None
    assert len(document.figures) == 1
    figure_blocks = [block for block in document.blocks if block.block_type == "figure"]
    assert len(figure_blocks) == 1
    assert document.blocks == figure_blocks
    assert figure_blocks[0].figure_id == document.figures[0].figure_id
    assert document.figures[0].metadata["image_path"] == "images/figure-1.png"


def test_mineru_existing_figure_file_is_registered_as_canonical_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fixture")
    output_dir = tmp_path / "mineru-output"
    image_path = output_dir / "images" / "figure-1.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"figure pixels")
    content_list_path = output_dir / "content_list.json"
    content_list_path.write_text("[]", encoding="utf-8")
    parsed = parser._mineru_content_to_parsed_doc(
        path=pdf_path,
        content_list=[
            {
                "type": "image",
                "image_caption": "Figure 1. Architecture",
                "img_path": "images/figure-1.png",
                "page_idx": 0,
            }
        ],
        page_count=1,
        output_dir=output_dir,
        content_list_path=content_list_path,
    )
    monkeypatch.setattr(parser, "_parse_pdf_with_mineru", lambda *_args: parsed)

    document = canonical_adapters.run_mineru(pdf_path, 1)

    assert document is not None
    assert len(document.assets) == 1
    assert document.assets[0].path.startswith("assets/")
    assert document.assets[0].source_path == str(image_path.resolve())
    assert document.figures[0].asset_path == document.assets[0].path


def test_parser_sources_have_no_structured_4000_character_truncation() -> None:
    root = Path(__file__).resolve().parents[1]

    for relative in ("src/app/services/parser.py", "src/app/services/canonical_adapters.py"):
        assert "[:4000]" not in (root / relative).read_text(encoding="utf-8")
