from __future__ import annotations

from pathlib import Path

import pytest

from app.services.canonical_adapters import (
    PDFCanonicalAdapter,
    parse_canonical_document,
)
from app.services.parser import DocumentParseError, parse_document


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canonical"


def _assert_common_contract(document) -> None:
    assert document.blocks
    assert document.source_path
    assert document.source_media_type
    assert document.parser_source
    assert document.parser_metadata
    assert [block.reading_order for block in document.blocks] == list(
        range(len(document.blocks))
    )
    block_ids = [block.block_id for block in document.blocks]
    assert all(block_ids)
    assert len(block_ids) == len(set(block_ids))
    assert all(block.parser_source for block in document.blocks)

    table_ids = {table.table_id for table in document.tables}
    figure_ids = {figure.figure_id for figure in document.figures}
    formula_ids = {formula.formula_id for formula in document.formulas}
    assert all(block.table_id in table_ids for block in document.blocks if block.table_id)
    assert all(block.figure_id in figure_ids for block in document.blocks if block.figure_id)
    assert all(block.formula_id in formula_ids for block in document.blocks if block.formula_id)


@pytest.mark.parametrize("name", ["sample.md", "sample.html", "sample.docx"])
def test_existing_formats_produce_canonical_blocks(name: str) -> None:
    document = parse_canonical_document(FIXTURE_DIR / name)
    _assert_common_contract(document)


def test_block_ids_are_stable_across_repeated_parse() -> None:
    path = FIXTURE_DIR / "sample.md"
    first = parse_canonical_document(path)
    second = parse_canonical_document(path)
    assert [block.block_id for block in first.blocks] == [block.block_id for block in second.blocks]


def test_markdown_preserves_structure_and_exact_source_ranges() -> None:
    path = FIXTURE_DIR / "sample.md"
    source = path.read_text(encoding="utf-8")
    document = parse_canonical_document(path)

    intro = next(block for block in document.blocks if block.text.startswith("This paragraph"))
    assert intro.section_path == ["Canonical Adapter Study", "Introduction"]
    assert source[intro.source_spans[0].char_start : intro.source_spans[0].char_end] == intro.text

    headings = [block for block in document.blocks if block.block_type == "heading"]
    assert headings and all(not block.retrievable for block in headings)
    assert document.outline[0].title == "Canonical Adapter Study"
    assert document.outline[0].children[0].title == "Introduction"

    code = next(block for block in document.blocks if block.metadata.get("kind") == "code")
    code_source = source[code.source_spans[0].char_start : code.source_spans[0].char_end]
    assert code.metadata["language"] == "python"
    assert code.metadata["fence"] == "```"
    assert code_source.startswith("```python") and code_source.endswith("```")
    assert "return value" in code.text

    table_block = next(block for block in document.blocks if block.block_type == "table")
    table = next(table for table in document.tables if table.table_id == table_block.table_id)
    assert table.headers == ["Metric", "Score"]
    assert table.rows == [["Recall", "0.91"], ["Precision", "0.88"]]
    assert table.source_markdown == source[
        table_block.source_spans[0].char_start : table_block.source_spans[0].char_end
    ]
    assert len(table.cells) == 6

    formula_block = next(block for block in document.blocks if block.block_type == "formula")
    formula = next(item for item in document.formulas if item.formula_id == formula_block.formula_id)
    formula_source = source[
        formula_block.source_spans[0].char_start : formula_block.source_spans[0].char_end
    ]
    assert formula.latex == r"F_1 = 2 \frac{PR}{P + R}"
    assert formula_source.startswith("$$") and formula_source.endswith("$$")

    figure_block = next(block for block in document.blocks if block.block_type == "figure")
    figure = next(item for item in document.figures if item.figure_id == figure_block.figure_id)
    assert figure.caption == "Evaluation curve"
    assert figure.metadata["target"] == "figures/evaluation.png"
    assert source[
        figure_block.source_spans[0].char_start : figure_block.source_spans[0].char_end
    ].startswith("![Evaluation curve]")


def test_markdown_references_stop_at_appendix_and_appendix_is_retrievable() -> None:
    document = parse_canonical_document(FIXTURE_DIR / "sample.md")
    references = [block for block in document.blocks if block.block_type == "reference"]
    assert [block.text for block in references] == ["Doe, J. Canonical Parsing. 2026."]
    assert all(not block.retrievable for block in references)

    appendix = next(block for block in document.blocks if block.block_type == "appendix")
    assert appendix.retrievable is True
    assert appendix.section_path[-1] == "Appendix"


def test_html_preserves_dom_order_structures_and_locators() -> None:
    document = parse_canonical_document(FIXTURE_DIR / "sample.html")
    _assert_common_contract(document)
    combined = "\n".join(block.text for block in document.blocks)
    assert "script-content" not in combined
    assert "Navigation content" not in combined
    assert "noscript-content" not in combined
    assert "template-content" not in combined
    assert "Rendered loss" not in combined
    assert document.title == "Canonical HTML Study"

    summary = next(block for block in document.blocks if block.text.startswith("The adapter"))
    assert summary.section_path == ["Canonical HTML Study", "Methods"]
    assert summary.source_spans[0].element_id == "method-summary"
    assert summary.source_spans[0].css_selector == "#method-summary"
    assert summary.source_spans[0].xpath

    code = next(block for block in document.blocks if block.metadata.get("kind") == "code")
    assert code.text == "result = parse(source)"
    assert code.source_spans[0].css_selector == "#example-code"

    table = document.tables[0]
    assert table.headers == ["Metric", "Score"]
    assert table.rows == [["Recall", "0.91", "high"]]
    assert any(cell.rowspan == 2 for cell in table.cells)
    assert any(cell.colspan == 2 for cell in table.cells)
    assert table.source_html.startswith("<table")
    assert table.source_spans[0].css_selector == "#results-table"

    figure = document.figures[0]
    assert figure.caption == "Figure 1. Evaluation curve"
    assert figure.metadata["src"] == "figures/evaluation.png"
    assert figure.metadata["alt"] == "Evaluation curve"
    assert figure.source_spans[0].css_selector == "#evaluation-figure"

    formula = document.formulas[0]
    assert formula.latex == r"L = -\log p(y|x)"
    assert formula.source_spans[0].css_selector == "#loss-equation"


def test_docx_preserves_body_order_locators_image_and_omml() -> None:
    document = parse_canonical_document(FIXTURE_DIR / "sample.docx")
    _assert_common_contract(document)
    block_types = [block.block_type for block in document.blocks]
    paragraph_index = next(
        index for index, block in enumerate(document.blocks) if block.text == "Paragraph before table."
    )
    table_index = block_types.index("table")
    after_index = next(
        index for index, block in enumerate(document.blocks) if block.text == "Paragraph after table."
    )
    assert paragraph_index < table_index < after_index

    heading = next(block for block in document.blocks if block.block_type == "heading")
    assert heading.text == "DOCX Methods"
    assert heading.source_spans[0].paragraph_id
    paragraph = next(block for block in document.blocks if block.text == "Paragraph before table.")
    assert paragraph.section_path == ["DOCX Methods"]
    assert paragraph.source_spans[0].paragraph_id

    table = document.tables[0]
    assert table.headers == ["Metric", "Score"]
    assert table.rows == [["Recall", "0.91"]]
    assert all(cell.source_spans[0].table_id == table.table_id for cell in table.cells)
    assert {(cell.source_spans[0].row_index, cell.source_spans[0].column_index) for cell in table.cells} == {
        (0, 0), (0, 1), (1, 0), (1, 1)
    }

    figure = document.figures[0]
    assert figure.source_spans[0].image_relationship_id
    assert figure.asset_path
    assert document.assets and document.assets[0].media_type == "image/png"

    formula = document.formulas[0]
    assert "x + y = 1" in formula.latex
    assert formula.metadata["source_format"] == "omml"
    assert "oMath" in formula.metadata["omml"]


def test_text_and_unknown_suffix_preserve_exact_ranges_and_empty_file_warning(tmp_path: Path) -> None:
    source = "First paragraph.\ncontinues here.\n\nSecond paragraph stays whole.\n"
    text_path = tmp_path / "notes.txt"
    text_path.write_text(source, encoding="utf-8")
    with text_path.open("r", encoding="utf-8", newline="") as source_file:
        source = source_file.read()
    text_document = parse_canonical_document(text_path)
    assert [block.text.replace("\r\n", "\n") for block in text_document.blocks] == [
        "First paragraph.\ncontinues here.",
        "Second paragraph stays whole.",
    ]
    for block in text_document.blocks:
        span = block.source_spans[0]
        assert source[span.char_start : span.char_end] == block.text
        assert span.line_start is not None and span.line_end is not None

    unknown_path = tmp_path / "notes.custom"
    unknown_path.write_text("Unknown suffix is plain text.", encoding="utf-8")
    unknown = parse_canonical_document(unknown_path)
    assert unknown.parser_source == "text"
    assert unknown.blocks[0].text == "Unknown suffix is plain text."

    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("", encoding="utf-8")
    empty = parse_canonical_document(empty_path)
    assert empty.blocks == []
    assert empty.warnings


def test_text_source_ranges_preserve_crlf_offsets(tmp_path: Path) -> None:
    path = tmp_path / "windows-lines.txt"
    path.write_bytes(b"First paragraph.\r\n\r\nSecond paragraph.")
    with path.open("r", encoding="utf-8", newline="") as source_file:
        source = source_file.read()

    document = parse_canonical_document(path)
    assert len(document.blocks) == 2
    for block in document.blocks:
        span = block.source_spans[0]
        assert source[span.char_start : span.char_end] == block.text


def test_dispatcher_uses_pdf_adapter_without_recursing(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"placeholder")
    sentinel = object()
    monkeypatch.setattr(PDFCanonicalAdapter, "parse", lambda self, path: sentinel)
    assert parse_canonical_document(pdf_path) is sentinel


@pytest.mark.parametrize(
    ("suffix", "content", "parser_source"),
    [(".markdown", "# Alias", "markdown"), (".htm", "<p>Alias</p>", "html")],
)
def test_dispatcher_supports_format_aliases(
    tmp_path: Path, suffix: str, content: str, parser_source: str
) -> None:
    path = tmp_path / f"alias{suffix}"
    path.write_text(content, encoding="utf-8")
    assert parse_canonical_document(path).parser_source == parser_source


def test_missing_path_raises_clear_parse_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"
    with pytest.raises((DocumentParseError, ValueError), match="missing|exist|read"):
        parse_canonical_document(missing)


def test_read_failure_is_wrapped_as_document_parse_error(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "unreadable.txt"
    path.write_text("content", encoding="utf-8")
    original = Path.open

    def fail_for_source(candidate: Path, mode="r", *args, **kwargs):
        if candidate == path.resolve() and "b" in mode:
            raise PermissionError("access denied")
        return original(candidate, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_for_source)
    with pytest.raises(DocumentParseError, match="Unable to read|access denied"):
        parse_canonical_document(path)


def test_compatibility_parse_document_preserves_long_canonical_block(tmp_path: Path) -> None:
    content = "x" * 5001
    path = tmp_path / "long.txt"
    path.write_text(content, encoding="utf-8")
    parsed = parse_document(path)
    assert len(parsed.chunks) == 1
    assert parsed.chunks[0].text == content
    assert parsed.text == content
    assert parsed.metadata["format"] == "text"
    assert parsed.metadata["canonical"]["parser_source"] == "text"
