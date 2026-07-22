from __future__ import annotations

from pathlib import Path

import pytest

from app.services import canonical_adapters, parser
from app.services.canonical_adapters import PDFCanonicalAdapter
from app.services.canonical_models import (
    CanonicalAsset,
    CanonicalBlock,
    CanonicalCell,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalQualityIssue,
    CanonicalTable,
    SectionNode,
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


def test_blank_text_layer_page_does_not_count_as_structured_parser_coverage() -> None:
    document = _document(
        _block("Scanned page one", page=0),
        expected_page_count=2,
        parsed_page_indices=[0],
    )

    canonical_adapters._attach_pdf_audit(
        document,
        page_count=2,
        page_texts=["", ""],
        text_layer_warnings=[],
        attempts=["mineru:success"],
        primary_parser="mineru",
    )
    report = CanonicalQualityGate().evaluate(document)

    assert document.metadata["parsed_page_indices"] == [0]
    assert document.metadata["text_layer_blank_page_indices"] == [0, 1]
    issue = next(issue for issue in report.issues if issue.code == "page_missing")
    assert issue.metadata["missing_pages"] == [2]
    assert issue.severity == "fatal"


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


def test_quality_gate_rejects_duplicate_table_ids_even_when_tables_are_valid() -> None:
    def valid_table() -> CanonicalTable:
        return CanonicalTable(
            table_id="duplicate-table",
            headers=["Model", "F1"],
            rows=[["SAC-KG", "74.7"]],
            cells=[
                CanonicalCell(text="Model", row_index=0, column_index=0, is_header=True),
                CanonicalCell(text="F1", row_index=0, column_index=1, is_header=True),
                CanonicalCell(text="SAC-KG", row_index=1, column_index=0),
                CanonicalCell(text="74.7", row_index=1, column_index=1),
            ],
            source_spans=[SourceSpan(page_index=0, page_label="1")],
        )

    document = _document(_block())
    document.tables = [valid_table(), valid_table()]

    report = CanonicalQualityGate().evaluate(document)

    duplicate = next(issue for issue in report.issues if issue.code == "table_id_duplicate")
    assert duplicate.severity == "fatal"
    assert duplicate.metadata["table_ids"] == ["duplicate-table"]
    assert report.accepted is False


def test_table_gate_rejects_missing_and_incomplete_cell_inventory() -> None:
    missing = CanonicalTable(
        table_id="missing-cells",
        headers=["Model", "F1"],
        rows=[["SAC-KG", "74.7"]],
    )
    incomplete = CanonicalTable(
        table_id="incomplete-cells",
        headers=["Model", "F1"],
        rows=[["SAC-KG", "74.7"]],
        cells=[
            CanonicalCell(text="Model", row_index=0, column_index=0, is_header=True),
            CanonicalCell(text="F1", row_index=0, column_index=1, is_header=True),
            CanonicalCell(text="SAC-KG", row_index=1, column_index=0),
        ],
    )

    assert CanonicalQualityGate._invalid_table_reasons(missing) == ["cells_missing"]
    assert "cells_incomplete" in CanonicalQualityGate._invalid_table_reasons(incomplete)


def test_table_gate_compares_cell_and_markdown_representations() -> None:
    table = CanonicalTable(
        table_id="inconsistent",
        headers=["A", "B"],
        rows=[["1", "2"]],
        cells=[
            CanonicalCell(text="A", row_index=0, column_index=0, is_header=True),
            CanonicalCell(text="B", row_index=0, column_index=1, is_header=True),
            CanonicalCell(text="1", row_index=1, column_index=0),
            CanonicalCell(text="WRONG", row_index=1, column_index=1),
        ],
        normalized_markdown="| A | B |\n| --- | --- |\n| 1 | 999 |",
        source_markdown="| A | B |\n| --- | --- |\n| 1 | 888 |",
    )

    reasons = CanonicalQualityGate._invalid_table_reasons(table)

    assert "cell_value_mismatch" in reasons
    assert "normalized_markdown_mismatch" in reasons
    assert "source_markdown_mismatch" in reasons


def test_markdown_table_parser_uses_backslash_parity_for_pipe_escaping() -> None:
    markdown = "\n".join(
        [
            "| Path | Note |",
            "| --- | --- |",
            r"|C:\\|A\|B|",
            r"|D:\\|A\\\|B|",
        ]
    )

    assert CanonicalQualityGate._markdown_table_data(markdown) == (
        ["Path", "Note"],
        [["C:\\", "A|B"], ["D:\\", "A\\|B"]],
    )


def test_table_gate_accepts_complete_rowspan_and_colspan_inventory() -> None:
    table = CanonicalTable(
        table_id="merged-cells",
        headers=["A", "B", ""],
        rows=[["R", "1", "2"], ["", "Wide", ""]],
        cells=[
            CanonicalCell(text="A", row_index=0, column_index=0, is_header=True),
            CanonicalCell(
                text="B", row_index=0, column_index=1, colspan=2, is_header=True
            ),
            CanonicalCell(text="R", row_index=1, column_index=0, rowspan=2),
            CanonicalCell(text="1", row_index=1, column_index=1),
            CanonicalCell(text="2", row_index=1, column_index=2),
            CanonicalCell(text="Wide", row_index=2, column_index=1, colspan=2),
        ],
        normalized_markdown=(
            "| A | B |  |\n| --- | --- | --- |\n"
            "| R | 1 | 2 |\n|  | Wide |  |"
        ),
    )

    assert CanonicalQualityGate._invalid_table_reasons(table) == []


def test_table_gate_reports_short_rows_without_raising() -> None:
    table = CanonicalTable(
        table_id="short-row",
        headers=["A", "B"],
        rows=[["1"]],
        cells=[
            CanonicalCell(text="A", row_index=0, column_index=0, is_header=True),
            CanonicalCell(text="B", row_index=0, column_index=1, is_header=True),
            CanonicalCell(text="1", row_index=1, column_index=0),
            CanonicalCell(text="orphan", row_index=1, column_index=1),
        ],
    )

    reasons = CanonicalQualityGate._invalid_table_reasons(table)

    assert "row_width_mismatch" in reasons
    assert "cell_value_mismatch" in reasons


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


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        (
            [
                "### Abstract\n\nSource abstract sentence.\n\n"
                "### 1. Introduction\n\nIntroduction must not be included."
            ],
            "Source abstract sentence.",
        ),
        (
            ["## ABSTRACT", "First sentence.\nSecond sentence.", "## Introduction", "Intro"],
            "First sentence.\nSecond sentence.",
        ),
        (
            ["Abstract\nSource body.\n# Keywords\nRAG, MinerU"],
            "Source body.",
        ),
    ],
)
def test_explicit_abstract_stops_at_next_markdown_heading(
    candidates: list[str],
    expected: str,
) -> None:
    assert canonical_adapters._extract_explicit_abstract(candidates) == expected


@pytest.mark.parametrize(
    "section_heading",
    ["Introduction", "1 Introduction", "1. Introduction", "Keywords: RAG, MinerU"],
)
def test_plain_pdf_abstract_stops_at_common_section_heading(
    section_heading: str,
) -> None:
    raw_page = (
        "Abstract\nSource abstract sentence one.\nSource abstract sentence two.\n\n"
        f"{section_heading}\nSection content must not be included."
    )

    assert canonical_adapters._extract_explicit_abstract([raw_page]) == (
        "Source abstract sentence one.\nSource abstract sentence two."
    )


def test_plain_pdf_abstract_keeps_introduction_word_inside_body_sentence() -> None:
    raw_page = (
        "Abstract\nThis sentence provides an introduction to the method.\n"
        "A second abstract sentence follows."
    )

    assert canonical_adapters._extract_explicit_abstract([raw_page]) == (
        "This sentence provides an introduction to the method.\n"
        "A second abstract sentence follows."
    )


@pytest.mark.parametrize(
    "section_heading",
    [
        "I. INTRODUCTION",
        "II INTRODUCTION",
        "关键词",
        "关键词：RAG，MinerU",
        "引言",
        "一、引言",
        "1 引言",
        "1. 引言",
    ],
)
def test_plain_pdf_abstract_stops_at_roman_or_chinese_section_heading(
    section_heading: str,
) -> None:
    raw_page = (
        "Abstract\nSource abstract sentence.\n\n"
        f"{section_heading}\nSection content must not be included."
    )

    assert canonical_adapters._extract_explicit_abstract([raw_page]) == (
        "Source abstract sentence."
    )


@pytest.mark.parametrize(
    "body_line",
    [
        "This sentence discusses keywords and introduction in ordinary prose.",
        "本文在正文句子中讨论关键词及其定义。",
        "本文的引言部分解释了研究背景。",
    ],
)
def test_plain_pdf_abstract_keeps_english_and_chinese_boundary_words_in_prose(
    body_line: str,
) -> None:
    raw_page = f"Abstract\n{body_line}\nA final abstract sentence."

    assert canonical_adapters._extract_explicit_abstract([raw_page]) == (
        f"{body_line}\nA final abstract sentence."
    )


@pytest.mark.parametrize(
    "abstract_heading",
    ["Abstract:", "Abstract：", "Abstract -", "Abstract —", "摘要："],
)
def test_quality_detector_recognizes_explicit_abstract_heading_variants(
    abstract_heading: str,
) -> None:
    document = _document(
        _block("Body"),
        text_layer_pages=[f"{abstract_heading}\nSource abstract body."],
    )

    report = CanonicalQualityGate().evaluate(document)

    assert any(issue.code == "abstract_missing" for issue in report.issues)


@pytest.mark.parametrize(
    "section_heading",
    [
        "Background",
        "2. Methods",
        "III. Methodology",
        "Results",
        "Conclusion",
        "Conclusions",
        "背景",
        "3、方法",
        "结果",
        "结论",
    ],
)
def test_plain_pdf_abstract_stops_at_common_english_and_chinese_sections(
    section_heading: str,
) -> None:
    raw_page = (
        "Abstract\nSource abstract sentence.\n\n"
        f"{section_heading}\nSection content must not be included."
    )

    assert canonical_adapters._extract_explicit_abstract([raw_page]) == (
        "Source abstract sentence."
    )


@pytest.mark.parametrize(
    "section_heading",
    [
        "Materials and Methods",
        "MATERIALS AND METHODS",
        "2 Materials and Methods",
        "III. Results and Discussion",
        "Discussion",
        "4. Related Work",
        "研究方法",
        "三、结果与讨论",
    ],
)
def test_plain_pdf_abstract_stops_at_compound_section_headings(
    section_heading: str,
) -> None:
    raw_page = (
        "Abstract\nSource abstract sentence.\n\n"
        f"{section_heading}\nSection content must not be included."
    )

    assert canonical_adapters._extract_explicit_abstract([raw_page]) == (
        "Source abstract sentence."
    )


@pytest.mark.parametrize(
    "body_line",
    [
        "This discussion summarizes the materials and methods used in the study.",
        "本文在正文句子中讨论研究方法和结果与讨论的关系。",
    ],
)
def test_plain_pdf_abstract_keeps_compound_heading_words_in_prose(
    body_line: str,
) -> None:
    raw_page = f"Abstract\n{body_line}\nA final abstract sentence."

    assert canonical_adapters._extract_explicit_abstract([raw_page]) == (
        f"{body_line}\nA final abstract sentence."
    )


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
        _block("MinerU introduction", order=1, page=1),
        _block("Unchanged page three", order=2, page=2),
    ]
    repaired = _document(
        _block("Abstract\nRecovered source abstract", page=0, parser_source="document_intelligence"),
        _block(
            "Introduction source",
            order=1,
            page=1,
            parser_source="document_intelligence",
        ),
        expected_page_count=3,
        parsed_page_indices=[0, 1],
    )
    repaired.parser_source = "document_intelligence"
    repaired.abstract = "Recovered source abstract"
    calls: list[set[int] | None] = []

    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 3)
    monkeypatch.setattr(
        parser,
        "_extract_pdf_text_layer",
        lambda _path: (["Abstract\nRecovered source abstract", "Introduction source", ""], 3),
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


@pytest.mark.parametrize("repair_kind", ["partial", "empty"])
def test_pdf_incomplete_targeted_repair_preserves_mineru_without_full_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repair_kind: str,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")
    mineru = _document(
        _block("MinerU page one", page=0),
        _block("MinerU page two", order=1, page=1),
        expected_page_count=2,
        parsed_page_indices=[0, 1],
    )
    repair = _document(
        _block("Partial repair", page=0, parser_source="document_intelligence"),
        expected_page_count=2,
        parsed_page_indices=[0],
    )
    repair.parser_source = "document_intelligence"
    if repair_kind == "empty":
        repair.blocks = []
        repair.metadata["parsed_page_indices"] = []
    calls: list[set[int] | None] = []

    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 2)
    monkeypatch.setattr(
        parser,
        "_extract_pdf_text_layer",
        lambda _path: (["Abstract\nSource abstract", "Second-page source"], 2),
    )
    monkeypatch.setattr(canonical_adapters, "run_mineru", lambda *_args: mineru)

    def incomplete_di(_path, _page_count, _page_texts, page_indices=None):
        calls.append(None if page_indices is None else set(page_indices))
        return repair

    monkeypatch.setattr(canonical_adapters, "run_document_intelligence", incomplete_di)

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert calls == [{0, 1}]
    assert [block.text for block in result.blocks] == [
        "MinerU page one",
        "MinerU page two",
    ]
    assert result.metadata["primary_parser"] == "mineru"
    assert result.quality.status == "validation_failed"
    assert any("targeted:partial" in attempt for attempt in result.metadata["parser_attempts"])


def _table_document(
    table: CanonicalTable,
    *,
    parser_source: str,
) -> CanonicalDocument:
    return CanonicalDocument(
        document_id=f"doc-{parser_source}",
        parser_source=parser_source,
        parse_version="canonical-v1",
        title="Paper",
        blocks=[
            _block("Page narrative", parser_source=parser_source),
            CanonicalBlock(
                block_id=f"{table.table_id}-block",
                block_type="table",
                text=table.normalized_markdown or table.source_markdown or "table",
                reading_order=1,
                source_spans=[SourceSpan(page_index=0, page_label="1")],
                parser_source=parser_source,
                table_id=table.table_id,
            ),
        ],
        tables=[table],
        metadata={"expected_page_count": 1, "parsed_page_indices": [0]},
    )


def test_table_repair_with_narrative_only_preserves_invalid_mineru_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")
    invalid = CanonicalTable(
        table_id="mineru-invalid",
        headers=["Model", "F1"],
        rows=[["SAC-KG"]],
        cells=[
            CanonicalCell(text="Model", row_index=0, column_index=0, is_header=True),
            CanonicalCell(text="F1", row_index=0, column_index=1, is_header=True),
            CanonicalCell(text="SAC-KG", row_index=1, column_index=0),
        ],
        source_spans=[SourceSpan(page_index=0, page_label="1")],
    )
    mineru = _table_document(invalid, parser_source="mineru")
    narrative_only = _document(
        _block("DI returned only prose", parser_source="document_intelligence"),
        expected_page_count=1,
        parsed_page_indices=[0],
    )
    narrative_only.parser_source = "document_intelligence"
    calls: list[set[int] | None] = []

    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(parser, "_extract_pdf_text_layer", lambda _path: (["Body"], 1))
    monkeypatch.setattr(canonical_adapters, "run_mineru", lambda *_args: mineru)

    def fake_di(_path, _page_count, _page_texts, page_indices=None):
        calls.append(None if page_indices is None else set(page_indices))
        return narrative_only

    monkeypatch.setattr(canonical_adapters, "run_document_intelligence", fake_di)

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert calls == [{0}]
    assert [table.table_id for table in result.tables] == ["mineru-invalid"]
    assert result.quality.accepted is False
    assert any(issue.code == "table_invalid" for issue in result.quality.issues)
    assert any("targeted:incomplete" in attempt for attempt in result.metadata["parser_attempts"])


def test_table_repair_with_valid_replacement_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")
    invalid = CanonicalTable(
        table_id="mineru-invalid",
        headers=["Model", "F1"],
        rows=[["SAC-KG"]],
        cells=[
            CanonicalCell(text="Model", row_index=0, column_index=0, is_header=True),
            CanonicalCell(text="F1", row_index=0, column_index=1, is_header=True),
            CanonicalCell(text="SAC-KG", row_index=1, column_index=0),
        ],
        source_spans=[
            SourceSpan(
                page_index=0,
                page_label="1",
                source_block_id="mineru-table-1",
            )
        ],
    )
    replacement = CanonicalTable(
        table_id="di-valid",
        headers=["Model", "F1"],
        rows=[["SAC-KG", "74.7"]],
        cells=[
            CanonicalCell(text="Model", row_index=0, column_index=0, is_header=True),
            CanonicalCell(text="F1", row_index=0, column_index=1, is_header=True),
            CanonicalCell(text="SAC-KG", row_index=1, column_index=0),
            CanonicalCell(text="74.7", row_index=1, column_index=1),
        ],
        normalized_markdown=(
            "| Model | F1 |\n| --- | --- |\n| SAC-KG | 74.7 |"
        ),
        source_spans=[
            SourceSpan(
                page_index=0,
                page_label="1",
                source_block_id="document_intelligence-table-1",
            )
        ],
    )
    mineru = _table_document(invalid, parser_source="mineru")
    repaired = _table_document(replacement, parser_source="document_intelligence")
    calls: list[set[int] | None] = []

    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(parser, "_extract_pdf_text_layer", lambda _path: (["Body"], 1))
    monkeypatch.setattr(canonical_adapters, "run_mineru", lambda *_args: mineru)

    def fake_di(_path, _page_count, _page_texts, page_indices=None):
        calls.append(None if page_indices is None else set(page_indices))
        return repaired

    monkeypatch.setattr(canonical_adapters, "run_document_intelligence", fake_di)

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert calls == [{0}]
    assert [table.table_id for table in result.tables] == ["di-valid"]
    assert result.quality.accepted is True
    assert all(issue.code != "table_invalid" for issue in result.quality.issues)
    assert result.tables[0].status == "repaired_by_vision"
    assert result.tables[0].metadata["repair_original_table_id"] == "mineru-invalid"
    assert result.tables[0].metadata["repair_proof_validated"] is True
    proof = result.tables[0].metadata["repair_proof"]
    assert proof["match_basis"] == "unique_table_on_page"
    assert proof["original_request"]["table_id"] == "mineru-invalid"
    assert proof["validated_mapping"] == {
        "original_table_id": "mineru-invalid",
        "replacement_table_id": "di-valid",
        "page_index": 0,
    }


def test_table_repair_requires_replacement_inventory_on_each_target_page() -> None:
    def invalid_table(table_id: str, page: int) -> CanonicalTable:
        return CanonicalTable(
            table_id=table_id,
            headers=["A", "B"],
            rows=[["1"]],
            cells=[
                CanonicalCell(text="A", row_index=0, column_index=0, is_header=True),
                CanonicalCell(text="B", row_index=0, column_index=1, is_header=True),
                CanonicalCell(text="1", row_index=1, column_index=0),
            ],
            source_spans=[SourceSpan(page_index=page, page_label=str(page + 1))],
        )

    def valid_table(table_id: str, page: int) -> CanonicalTable:
        return CanonicalTable(
            table_id=table_id,
            headers=["A", "B"],
            rows=[["1", "2"]],
            cells=[
                CanonicalCell(text="A", row_index=0, column_index=0, is_header=True),
                CanonicalCell(text="B", row_index=0, column_index=1, is_header=True),
                CanonicalCell(text="1", row_index=1, column_index=0),
                CanonicalCell(text="2", row_index=1, column_index=1),
            ],
            source_spans=[SourceSpan(page_index=page, page_label=str(page + 1))],
        )

    primary = _document(
        _block("Page one", page=0),
        _block("Page two", order=1, page=1),
        expected_page_count=2,
        parsed_page_indices=[0, 1],
    )
    primary.tables = [invalid_table("bad-1", 0), invalid_table("bad-2", 1)]
    repair = _document(
        _block("Page one repair", page=0, parser_source="document_intelligence"),
        _block("Page two prose", order=1, page=1, parser_source="document_intelligence"),
        expected_page_count=2,
        parsed_page_indices=[0, 1],
    )
    repair.tables = [valid_table("fixed-1", 0), valid_table("fixed-2-wrong-page", 0)]
    original_report = CanonicalQualityGate().evaluate(primary)
    repair_issues = [issue for issue in original_report.issues if issue.repairable]
    candidate = canonical_adapters._merge_pdf_page_repairs(
        primary.model_copy(deep=True),
        repair,
        {0, 1},
        issues=repair_issues,
    )
    CanonicalQualityGate().evaluate(candidate)

    assert canonical_adapters._targeted_repair_satisfies_issues(
        primary,
        repair,
        candidate,
        repair_issues,
        {0, 1},
    ) is False


def test_targeted_table_repair_rejects_duplicate_valid_replacements() -> None:
    invalid = CanonicalTable(
        table_id="bad-table",
        headers=["A", "B"],
        rows=[["1"]],
        source_spans=[SourceSpan(page_index=0, page_label="1")],
    )
    valid = CanonicalTable(
        table_id="fixed-table",
        headers=["A", "B"],
        rows=[["1", "2"]],
        cells=[
            CanonicalCell(text="A", row_index=0, column_index=0, is_header=True),
            CanonicalCell(text="B", row_index=0, column_index=1, is_header=True),
            CanonicalCell(text="1", row_index=1, column_index=0),
            CanonicalCell(text="2", row_index=1, column_index=1),
        ],
        source_spans=[SourceSpan(page_index=0, page_label="1")],
    )
    primary = _document(
        CanonicalBlock(
            block_id="bad-table-block",
            block_type="table",
            text="bad",
            reading_order=0,
            source_spans=[SourceSpan(page_index=0, page_label="1")],
            parser_source="mineru",
            table_id="bad-table",
        ),
        expected_page_count=1,
        parsed_page_indices=[0],
    )
    primary.tables = [invalid]
    repair = _document(
        CanonicalBlock(
            block_id="fixed-table-block-1",
            block_type="table",
            text="fixed one",
            reading_order=0,
            source_spans=[SourceSpan(page_index=0, page_label="1")],
            parser_source="document_intelligence",
            table_id="fixed-table",
        ),
        CanonicalBlock(
            block_id="fixed-table-block-2",
            block_type="table",
            text="fixed two",
            reading_order=1,
            source_spans=[SourceSpan(page_index=0, page_label="1")],
            parser_source="document_intelligence",
            table_id="fixed-table",
        ),
        expected_page_count=1,
        parsed_page_indices=[0],
    )
    repair.parser_source = "document_intelligence"
    repair.tables = [valid.model_copy(deep=True), valid.model_copy(deep=True)]
    issues = [issue for issue in CanonicalQualityGate().evaluate(primary).issues if issue.repairable]
    candidate = canonical_adapters._merge_pdf_page_repairs(
        primary.model_copy(deep=True), repair, {0}, issues=issues
    )
    CanonicalQualityGate().evaluate(candidate)

    assert canonical_adapters._targeted_repair_satisfies_issues(
        primary, repair, candidate, issues, {0}
    ) is False


def test_targeted_table_repair_matches_each_issue_to_a_distinct_locator() -> None:
    def table(
        table_id: str,
        bbox: tuple[float, float, float, float],
        *,
        valid: bool,
    ) -> CanonicalTable:
        rows = [["1", "2"]] if valid else [["1"]]
        cells = (
            [
                CanonicalCell(text="A", row_index=0, column_index=0, is_header=True),
                CanonicalCell(text="B", row_index=0, column_index=1, is_header=True),
                CanonicalCell(text="1", row_index=1, column_index=0),
                CanonicalCell(text="2", row_index=1, column_index=1),
            ]
            if valid
            else []
        )
        return CanonicalTable(
            table_id=table_id,
            headers=["A", "B"],
            rows=rows,
            cells=cells,
            source_spans=[
                SourceSpan(page_index=0, page_label="1", normalized_bbox=bbox)
            ],
        )

    primary = _document(_block("Page body"))
    primary.tables = [
        table("bad-top", (0.0, 0.1, 1.0, 0.2), valid=False),
        table("bad-bottom", (0.0, 0.7, 1.0, 0.8), valid=False),
    ]
    repair = _document(_block("DI body", parser_source="document_intelligence"))
    repair.tables = [
        table("fixed-top-1", (0.0, 0.1, 1.0, 0.2), valid=True),
        table("fixed-top-2", (0.0, 0.15, 1.0, 0.25), valid=True),
    ]
    issues = [issue for issue in CanonicalQualityGate().evaluate(primary).issues if issue.repairable]
    candidate = primary.model_copy(deep=True)
    candidate.tables = [item.model_copy(deep=True) for item in repair.tables]
    CanonicalQualityGate().evaluate(candidate)

    assert canonical_adapters._targeted_repair_satisfies_issues(
        primary, repair, candidate, issues, {0}
    ) is False


def test_targeted_table_repair_rejects_same_content_with_different_ids() -> None:
    def table(table_id: str, *, valid: bool) -> CanonicalTable:
        return CanonicalTable(
            table_id=table_id,
            caption="Main results",
            headers=["Model", "F1"],
            rows=[["SAC-KG", "74.7"]] if valid else [["SAC-KG"]],
            cells=(
                [
                    CanonicalCell(text="Model", row_index=0, column_index=0, is_header=True),
                    CanonicalCell(text="F1", row_index=0, column_index=1, is_header=True),
                    CanonicalCell(text="SAC-KG", row_index=1, column_index=0),
                    CanonicalCell(text="74.7", row_index=1, column_index=1),
                ]
                if valid
                else []
            ),
            source_spans=[
                SourceSpan(
                    page_index=0,
                    page_label="1",
                    source_block_id=(
                        f"document_intelligence-{table_id}"
                        if valid
                        else f"mineru-{table_id}"
                    ),
                )
            ],
        )

    primary = _document(_block("Page body"))
    primary.tables = [table("bad-1", valid=False), table("bad-2", valid=False)]
    repair = _document(_block("DI body", parser_source="document_intelligence"))
    repair.tables = [table("fixed-index-1", valid=True), table("fixed-index-2", valid=True)]
    issues = [issue for issue in CanonicalQualityGate().evaluate(primary).issues if issue.repairable]
    candidate = primary.model_copy(deep=True)
    candidate.tables = [item.model_copy(deep=True) for item in repair.tables]
    CanonicalQualityGate().evaluate(candidate)

    assert canonical_adapters._targeted_repair_satisfies_issues(
        primary, repair, candidate, issues, {0}
    ) is False


def test_targeted_table_repair_requires_exact_replacement_inventory_size() -> None:
    invalid = CanonicalTable(
        table_id="bad",
        headers=["A", "B"],
        rows=[["1"]],
        source_spans=[SourceSpan(page_index=0, page_label="1")],
    )

    def valid(table_id: str, value: str) -> CanonicalTable:
        return CanonicalTable(
            table_id=table_id,
            headers=["A", "B"],
            rows=[[value, "2"]],
            cells=[
                CanonicalCell(text="A", row_index=0, column_index=0, is_header=True),
                CanonicalCell(text="B", row_index=0, column_index=1, is_header=True),
                CanonicalCell(text=value, row_index=1, column_index=0),
                CanonicalCell(text="2", row_index=1, column_index=1),
            ],
            source_spans=[SourceSpan(page_index=0, page_label="1")],
        )

    primary = _document(_block("Page body"))
    primary.tables = [invalid]
    repair = _document(_block("DI body", parser_source="document_intelligence"))
    repair.tables = [valid("fixed-1", "1"), valid("extra", "3")]
    issues = [issue for issue in CanonicalQualityGate().evaluate(primary).issues if issue.repairable]
    candidate = primary.model_copy(deep=True)
    candidate.tables = [item.model_copy(deep=True) for item in repair.tables]
    CanonicalQualityGate().evaluate(candidate)

    assert canonical_adapters._targeted_repair_satisfies_issues(
        primary, repair, candidate, issues, {0}
    ) is False


def test_targeted_merge_rebuilds_references_and_uses_source_order_not_local_reading_order() -> None:
    primary = _document(
        _block("Narrative before", order=100, page=0, block_id="before"),
        CanonicalBlock(
            block_id="old-figure-block",
            block_type="figure",
            text="Old figure",
            reading_order=101,
            source_spans=[SourceSpan(page_index=0, page_label="1")],
            parser_source="mineru",
            figure_id="old-figure",
        ),
        _block("Narrative after", order=102, page=0, block_id="after"),
        _block("Page two", order=103, page=1, block_id="page-two"),
        expected_page_count=2,
        parsed_page_indices=[0, 1],
    )
    primary.figures = [
        CanonicalFigure(
            figure_id="old-figure",
            caption="Old",
            source_spans=[SourceSpan(page_index=0, page_label="1")],
            nearby_block_ids=["before", "old-figure-block", "missing-block"],
        )
    ]
    primary.formulas = [
        CanonicalFormula(
            formula_id="formula-one",
            latex="x = 1",
            source_spans=[SourceSpan(page_index=1, page_label="2")],
            nearby_block_ids=["old-figure-block"],
        )
    ]
    primary.outline = [SectionNode(title="Deleted heading", level=1, block_id="old-figure-block")]
    repair = _document(
        CanonicalBlock(
            block_id="new-figure-block",
            block_type="figure",
            text="New figure",
            reading_order=0,
            source_spans=[SourceSpan(page_index=0, page_label="1")],
            parser_source="document_intelligence",
            figure_id="new-figure",
        ),
        expected_page_count=2,
        parsed_page_indices=[0],
    )
    repair.parser_source = "document_intelligence"
    repair.figures = [
        CanonicalFigure(
            figure_id="new-figure",
            caption="New",
            source_spans=[SourceSpan(page_index=0, page_label="1")],
            nearby_block_ids=["new-figure-block", "before"],
        )
    ]

    issue = CanonicalQualityIssue(
        code="figure_invalid",
        severity="error",
        message="figure repair",
        repairable=True,
        repair_scope="page:1",
    )
    merged = canonical_adapters._merge_pdf_page_repairs(
        primary, repair, {0}, issues=[issue]
    )

    assert [block.text for block in merged.blocks] == [
        "Narrative before",
        "New figure",
        "Narrative after",
        "Page two",
    ]
    block_ids = {block.block_id for block in merged.blocks}
    assert all(
        node.block_id in block_ids
        for node in merged.outline
        if node.block_id is not None
    )
    assert set(merged.formulas[0].nearby_block_ids).issubset(block_ids)
    assert set(merged.figures[0].nearby_block_ids).issubset(block_ids)


def test_abstract_repair_without_explicit_abstract_preserves_mineru_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")
    mineru = _document(
        _block("Original MinerU page"),
        expected_page_count=1,
        parsed_page_indices=[0],
    )
    prose_only = _document(
        _block("DI prose without the source abstract", parser_source="document_intelligence"),
        expected_page_count=1,
        parsed_page_indices=[0],
    )
    prose_only.parser_source = "document_intelligence"
    calls: list[set[int] | None] = []

    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(
        parser,
        "_extract_pdf_text_layer",
        lambda _path: (["Abstract\nSource abstract"], 1),
    )
    monkeypatch.setattr(canonical_adapters, "run_mineru", lambda *_args: mineru)

    def fake_di(_path, _page_count, _page_texts, page_indices=None):
        calls.append(None if page_indices is None else set(page_indices))
        return prose_only

    monkeypatch.setattr(canonical_adapters, "run_document_intelligence", fake_di)

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert calls == [{0}]
    assert [block.text for block in result.blocks] == ["Original MinerU page"]
    assert result.abstract is None
    assert any(issue.code == "abstract_missing" for issue in result.quality.issues)
    assert any("targeted:incomplete" in attempt for attempt in result.metadata["parser_attempts"])


def test_abstract_targeted_repair_preserves_all_mineru_narrative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")

    def valid_table(table_id: str, page: int) -> CanonicalTable:
        span = SourceSpan(page_index=page, page_label=str(page + 1))
        return CanonicalTable(
            table_id=table_id,
            headers=["Metric", "Value"],
            rows=[["F1", "74.7"]],
            cells=[
                CanonicalCell(text="Metric", row_index=0, column_index=0, is_header=True),
                CanonicalCell(text="Value", row_index=0, column_index=1, is_header=True),
                CanonicalCell(text="F1", row_index=1, column_index=0),
                CanonicalCell(text="74.7", row_index=1, column_index=1),
            ],
            source_spans=[span],
        )

    mineru = _document(
        _block("MinerU page one narrative", order=0, page=0, block_id="p1-text"),
        CanonicalBlock(
            block_id="p1-table",
            block_type="table",
            text="MinerU table one",
            reading_order=1,
            source_spans=[SourceSpan(page_index=0, page_label="1")],
            parser_source="mineru",
            table_id="mineru-table-1",
        ),
        _block("MinerU page two narrative", order=2, page=1, block_id="p2-text"),
        CanonicalBlock(
            block_id="p2-table",
            block_type="table",
            text="MinerU table two",
            reading_order=3,
            source_spans=[SourceSpan(page_index=1, page_label="2")],
            parser_source="mineru",
            table_id="mineru-table-2",
        ),
        expected_page_count=2,
        parsed_page_indices=[0, 1],
    )
    mineru.tables = [
        valid_table("mineru-table-1", 0),
        valid_table("mineru-table-2", 1),
    ]

    repair = _document(
        CanonicalBlock(
            block_id="di-table-1-block",
            block_type="table",
            text="DI table one",
            reading_order=0,
            source_spans=[SourceSpan(page_index=0, page_label="1")],
            parser_source="document_intelligence",
            table_id="di-table-1",
        ),
        CanonicalBlock(
            block_id="di-table-2-block",
            block_type="table",
            text="DI table two",
            reading_order=1,
            source_spans=[SourceSpan(page_index=1, page_label="2")],
            parser_source="document_intelligence",
            table_id="di-table-2",
        ),
        expected_page_count=2,
        parsed_page_indices=[0, 1],
    )
    repair.parser_source = "document_intelligence"
    repair.tables = [valid_table("di-table-1", 0), valid_table("di-table-2", 1)]
    repair.abstract = "Recovered source abstract."

    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 2)
    monkeypatch.setattr(
        parser,
        "_extract_pdf_text_layer",
        lambda _path: (["Abstract\nSource abstract", "Body"], 2),
    )
    monkeypatch.setattr(canonical_adapters, "run_mineru", lambda *_args: mineru)
    monkeypatch.setattr(
        canonical_adapters,
        "run_document_intelligence",
        lambda *_args, **_kwargs: repair,
    )

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert result.abstract == "Recovered source abstract."
    assert [
        block.text for block in result.blocks if block.block_type == "narrative"
    ] == ["MinerU page one narrative", "MinerU page two narrative"]
    assert result.quality.accepted is True


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


def test_document_intelligence_model_fallback_is_audited_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")
    monkeypatch.setattr(parser, "_validate_pdf_basic", lambda _path: 1)
    monkeypatch.setattr(parser, "_extract_pdf_text_layer", lambda _path: (["Source body"], 1))
    monkeypatch.setattr(canonical_adapters, "run_mineru", lambda *_args: None)
    monkeypatch.setattr(
        parser,
        "_render_pdf_pages",
        lambda _path, dpi, page_indices=None: {0: b"page"},
    )
    monkeypatch.setattr(
        parser,
        "_analyze_pdf_page",
        lambda **kwargs: parser._fallback_page_analysis(
            page_label=kwargs["page_label"],
            raw_text=kwargs["raw_text"],
            text_quality=kwargs["text_quality"],
        ),
    )

    result = PDFCanonicalAdapter().parse(pdf_path)

    assert result.metadata["primary_parser"] == "pypdf_text_layer"
    assert "document_intelligence:full:unavailable" in result.metadata["parser_attempts"]
    assert "document_intelligence:full:success" not in result.metadata["parser_attempts"]


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
    class TestSettings:
        cache_dir = tmp_path / "adapter-cache"

    monkeypatch.setattr(canonical_adapters, "get_settings", lambda: TestSettings())
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
    original_read_bytes = Path.read_bytes

    def reject_bulk_image_read(candidate: Path) -> bytes:
        if candidate.resolve() == image_path.resolve():
            raise OSError("MinerU assets must use streaming I/O")
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", reject_bulk_image_read)

    document = canonical_adapters.run_mineru(pdf_path, 1)

    assert document is not None
    assert len(document.assets) == 1
    assert document.assets[0].path.startswith("assets/")
    durable_source = Path(document.assets[0].source_path or "")
    assert durable_source.is_file()
    with durable_source.open("rb") as durable, image_path.open("rb") as source:
        assert durable.read() == source.read()
    assert output_dir not in durable_source.parents
    assert tmp_path / "adapter-cache" in durable_source.parents
    assert document.figures[0].asset_path == document.assets[0].path


def test_targeted_page_repair_reconciles_figure_asset_inventory() -> None:
    primary = _document(
        CanonicalBlock(
            block_id="old-figure-block",
            block_type="figure",
            text="Old figure",
            reading_order=0,
            source_spans=[SourceSpan(page_index=0, page_label="1")],
            parser_source="mineru",
            figure_id="old-figure",
        ),
        CanonicalBlock(
            block_id="kept-figure-block",
            block_type="figure",
            text="Kept figure",
            reading_order=1,
            source_spans=[SourceSpan(page_index=1, page_label="2")],
            parser_source="mineru",
            figure_id="kept-figure",
        ),
    )
    primary.figures = [
        CanonicalFigure(
            figure_id="old-figure",
            caption="Old",
            asset_path="assets/old.png",
            source_spans=[SourceSpan(page_index=0, page_label="1")],
        ),
        CanonicalFigure(
            figure_id="kept-figure",
            caption="Kept",
            asset_path="assets/kept.png",
            source_spans=[SourceSpan(page_index=1, page_label="2")],
        ),
    ]
    primary.assets = [
        CanonicalAsset(
            asset_id="old-asset",
            path="assets/old.png",
            media_type="image/png",
            source_spans=[SourceSpan(page_index=0, page_label="1")],
        ),
        CanonicalAsset(
            asset_id="kept-asset",
            path="assets/kept.png",
            media_type="image/png",
            source_spans=[SourceSpan(page_index=1, page_label="2")],
        ),
    ]
    repair = _document(
        CanonicalBlock(
            block_id="new-figure-block",
            block_type="figure",
            text="New figure",
            reading_order=0,
            source_spans=[SourceSpan(page_index=0, page_label="1")],
            parser_source="document_intelligence",
            figure_id="new-figure",
        )
    )
    repair.figures = [
        CanonicalFigure(
            figure_id="new-figure",
            caption="New",
            asset_path="assets/new.png",
            source_spans=[SourceSpan(page_index=0, page_label="1")],
        )
    ]
    repair.assets = [
        CanonicalAsset(
            asset_id="new-asset",
            path="assets/new.png",
            media_type="image/png",
            sha256="a" * 64,
            source_spans=[SourceSpan(page_index=0, page_label="1")],
        ),
        CanonicalAsset(
            asset_id="duplicate-new-asset",
            path="assets/new.png",
            media_type="image/png",
            sha256="a" * 64,
            source_spans=[SourceSpan(page_index=0, page_label="1")],
        ),
    ]

    merged = canonical_adapters._merge_pdf_page_repairs(primary, repair, {0})

    assert {asset.path for asset in merged.assets} == {
        "assets/kept.png",
        "assets/new.png",
    }
    assert len(merged.assets) == 2
    assert {figure.asset_path for figure in merged.figures} == {
        "assets/kept.png",
        "assets/new.png",
    }
    assert all(
        figure.asset_path in {asset.path for asset in merged.assets}
        for figure in merged.figures
        if figure.asset_path
    )
    assert all(issue.code != "asset_invalid" for issue in CanonicalQualityGate().evaluate(merged).issues)


def test_parser_sources_have_no_structured_4000_character_truncation() -> None:
    root = Path(__file__).resolve().parents[1]

    for relative in ("src/app/services/parser.py", "src/app/services/canonical_adapters.py"):
        assert "[:4000]" not in (root / relative).read_text(encoding="utf-8")
