from __future__ import annotations

from pathlib import Path

from app.services.canonical_adapters import _parsed_pdf_to_canonical, parse_canonical_document
from app.services.canonical_models import (
    CanonicalBlock,
    CanonicalCell,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalTable,
    SourceSpan,
)
from app.services.structured_evidence import StructuredEvidenceBuilder, TableValidator
from app.services.parser import ParsedChunk, ParsedDocument


def _table(
    *,
    table_id: str = "table-1",
    caption: str = "Table 1. Evaluation results",
    rows: list[list[str]] | None = None,
    page_index: int = 0,
    metadata: dict[str, object] | None = None,
) -> CanonicalTable:
    headers = ["Dataset", "Method", "Accuracy"]
    data_rows = rows or [
        ["News", "Base", "81.2%"],
        ["News", "Proposed", "84.9%"],
    ]
    span = SourceSpan(
        page_index=page_index,
        source_block_id=f"source-{table_id}",
        bbox=(10.0, 20.0, 500.0, 700.0),
    )
    cells = [
        CanonicalCell(
            text=value,
            row_index=0,
            column_index=column,
            is_header=True,
            bbox=(float(column * 10), 0.0, float(column * 10 + 10), 10.0),
            source_spans=[span],
        )
        for column, value in enumerate(headers)
    ]
    cells.extend(
        CanonicalCell(
            text=value,
            row_index=row_index,
            column_index=column,
            bbox=(
                float(column * 10),
                float(row_index * 10),
                float(column * 10 + 10),
                float(row_index * 10 + 10),
            ),
            source_spans=[span],
        )
        for row_index, row in enumerate(data_rows, start=1)
        for column, value in enumerate(row)
    )
    markdown = "\n".join(
        [
            "| Dataset | Method | Accuracy |",
            "| --- | --- | --- |",
            *("| " + " | ".join(row) + " |" for row in data_rows),
        ]
    )
    return CanonicalTable(
        table_id=table_id,
        caption=caption,
        headers=headers,
        rows=data_rows,
        cells=cells,
        source_markdown=markdown,
        normalized_markdown=markdown,
        footnotes=["Accuracy is measured on the source test split."],
        source_spans=[span],
        metadata={"table_number": "1", **(metadata or {})},
    )


def test_valid_mineru_table_is_accepted_without_repair_request() -> None:
    result = TableValidator().validate(_table())

    assert result.accepted is True
    assert result.status == "accepted_mineru"
    assert result.repair_request is None
    assert result.table.status == "accepted_mineru"
    assert result.model_dump(mode="json")["accepted"] is True


def test_invalid_table_returns_targeted_repair_request_and_blocks_activation() -> None:
    table = _table()
    table.rows[0] = ["News", "Base"]

    result = TableValidator().validate(table)

    assert result.accepted is False
    assert result.status == "validation_failed"
    assert result.activation_allowed is False
    assert "row_width_mismatch" in result.reasons
    assert result.repair_request is not None
    assert result.repair_request.table_id == table.table_id
    assert result.repair_request.locator["page_index"] == 0
    assert result.repair_request.locator["bbox"] == [10.0, 20.0, 500.0, 700.0]
    assert "row_width_mismatch" in result.repair_request.reasons


def test_invalid_vision_repair_remains_validation_failed() -> None:
    original = _table()
    original.rows[0] = ["News"]
    invalid_repair = _table(table_id=original.table_id)
    invalid_repair.normalized_markdown = "| silently | truncated |"

    result = TableValidator().validate(original, repaired_table=invalid_repair)

    assert result.accepted is False
    assert result.status == "validation_failed"
    assert result.activation_allowed is False
    assert "normalized_markdown_mismatch" in result.reasons


def test_valid_repair_is_marked_repaired_by_vision() -> None:
    original = _table()
    original.metadata["truncated"] = True
    repaired = _table(table_id=original.table_id)

    result = TableValidator().validate(original, repaired_table=repaired)

    assert result.accepted is True
    assert result.status == "repaired_by_vision"
    assert result.table.status == "repaired_by_vision"


def test_repair_cannot_replace_a_different_table_identity() -> None:
    original = _table()
    original.metadata["truncated"] = True
    unrelated = _table(table_id="table-unrelated")

    result = TableValidator().validate(original, repaired_table=unrelated)

    assert result.accepted is False
    assert result.status == "validation_failed"
    assert "repair_identity_mismatch" in result.reasons


def test_table_validator_preserves_legal_spans_bboxes_and_source_values() -> None:
    table = CanonicalTable(
        table_id="table-span",
        caption="Table 2. Grouped results",
        headers=["Group", "Metric", "Value"],
        rows=[["A", "F1", "72.0%"], ["", "Recall", "70.0%"]],
        cells=[
            CanonicalCell(text="Group", row_index=0, column_index=0, is_header=True),
            CanonicalCell(text="Metric", row_index=0, column_index=1, is_header=True),
            CanonicalCell(text="Value", row_index=0, column_index=2, is_header=True),
            CanonicalCell(
                text="A",
                row_index=1,
                column_index=0,
                rowspan=2,
                bbox=(1.0, 2.0, 3.0, 8.0),
            ),
            CanonicalCell(text="F1", row_index=1, column_index=1),
            CanonicalCell(text="72.0%", row_index=1, column_index=2),
            CanonicalCell(text="Recall", row_index=2, column_index=1),
            CanonicalCell(text="70.0%", row_index=2, column_index=2),
        ],
        source_markdown=(
            "| Group | Metric | Value |\n| --- | --- | --- |\n"
            "| A | F1 | 72.0% |\n|  | Recall | 70.0% |"
        ),
        normalized_markdown=(
            "| Group | Metric | Value |\n| --- | --- | --- |\n"
            "| A | F1 | 72.0% |\n|  | Recall | 70.0% |"
        ),
        source_spans=[SourceSpan(page_index=1, bbox=(1.0, 2.0, 100.0, 200.0))],
        metadata={"table_number": "2"},
    )

    result = TableValidator().validate(table)

    assert result.accepted is True
    merged_cell = next(cell for cell in result.table.cells if cell.text == "A")
    assert merged_cell.rowspan == 2
    assert merged_cell.bbox == (1.0, 2.0, 3.0, 8.0)
    assert result.table.rows[0][2] == "72.0%"


def test_table_validator_rejects_caption_number_and_extraction_anomalies() -> None:
    table = _table(caption="Table 7. Wrong identity")
    table.metadata.update(
        {"truncated": True, "fragmented_numeric_tokens": ["84.9", "%"]}
    )

    result = TableValidator().validate(table)

    assert result.accepted is False
    assert {"caption_number_mismatch", "silent_truncation", "numeric_token_fragmented"}.issubset(
        result.reasons
    )


def test_table_validator_detects_percentage_split_across_cells() -> None:
    table = _table(rows=[["News", "84.9", "%"]])

    result = TableValidator().validate(table)

    assert result.accepted is False
    assert "numeric_token_fragmented" in result.reasons


def test_long_table_chunks_repeat_caption_and_header_with_token_bound() -> None:
    rows = [
        [f"Dataset-{index // 3}", f"Method-{index}", f"{80 + index / 10:.1f}%"]
        for index in range(18)
    ]
    table = _table(rows=rows)

    chunks = StructuredEvidenceBuilder().table_chunks(table, max_tokens=48)

    parent, *children = chunks
    assert parent.chunk_role == "parent"
    assert parent.metadata["rows"] == rows
    assert len(children) > 1
    assert all(child.chunk_role == "child" for child in children)
    assert all("Table 1. Evaluation results" in child.text for child in children)
    assert all("| Dataset | Method | Accuracy |" in child.text for child in children)
    assert all(child.token_count <= 48 for child in children)
    assert [row for child in children for row in child.metadata["rows"]] == rows
    assert all(child.metadata["footnotes"] == table.footnotes for child in children)
    assert all(child.metadata["cells"] for child in children)
    assert chunks == StructuredEvidenceBuilder().table_chunks(table, max_tokens=48)


def test_table_chunks_honor_explicit_semantic_groups_when_they_fit() -> None:
    rows = [
        ["News", "A", "80%"],
        ["News", "B", "81%"],
        ["Vision", "A", "82%"],
        ["Vision", "B", "83%"],
    ]
    table = _table(
        rows=rows,
        metadata={"semantic_row_groups": [[0, 1], [2, 3]]},
    )

    children = StructuredEvidenceBuilder().table_chunks(table, max_tokens=45)[1:]

    assert [child.metadata["row_indices"] for child in children] == [[0, 1], [2, 3]]


def test_single_overlong_row_is_explicit_overflow_without_truncation() -> None:
    long_value = " ".join(f"token-{index}" for index in range(80))
    table = _table(rows=[["News", "Verbose", long_value]])

    child = StructuredEvidenceBuilder().table_chunks(table, max_tokens=24)[1]

    assert child.metadata["overflow"] is True
    assert child.token_count > 24
    assert long_value in child.text
    assert child.metadata["rows"] == table.rows


def test_explicit_cross_page_continuation_merges_before_chunking() -> None:
    first = _table(
        table_id="table-page-1",
        rows=[["News", "Base", "81.2%"]],
        page_index=0,
    )
    continuation = _table(
        table_id="table-page-2",
        rows=[
            ["Dataset", "Method", "Accuracy"],
            ["News", "Proposed", "84.9%"],
        ],
        page_index=1,
        metadata={"continuation_of": first.table_id},
    )

    merged = StructuredEvidenceBuilder().merge_cross_page_tables([first, continuation])

    assert len(merged) == 1
    table = merged[0]
    assert table.status == "cross_page_merged"
    assert table.rows == [
        ["News", "Base", "81.2%"],
        ["News", "Proposed", "84.9%"],
    ]
    assert [span.page_index for span in table.source_spans] == [0, 1]
    assert any(cell.metadata.get("original_page_index") == 1 for cell in table.cells)
    assert table.metadata["merged_repeated_header_cells"]
    assert StructuredEvidenceBuilder().table_chunks(table, max_tokens=80)[0].metadata[
        "status"
    ] == "cross_page_merged"


def test_same_headers_without_continuation_are_not_merged() -> None:
    first = _table(table_id="unrelated-1", page_index=0)
    second = _table(table_id="unrelated-2", page_index=1)

    result = StructuredEvidenceBuilder().merge_cross_page_tables([first, second])

    assert [table.table_id for table in result] == ["unrelated-1", "unrelated-2"]


def test_figure_and_formula_evidence_keep_source_and_generated_text_separate() -> None:
    nearby = [
        CanonicalBlock(
            block_id="discussion",
            block_type="narrative",
            text="The source paragraph explains the rising trend and the loss term.",
            reading_order=0,
            parser_source="mineru",
            source_spans=[SourceSpan(page_index=2, source_block_id="paragraph-7")],
        )
    ]
    figure = CanonicalFigure(
        figure_id="figure-1",
        caption="Figure 1. Source accuracy curve",
        description="Original note below the source figure.",
        asset_path="assets/figure.png",
        nearby_block_ids=["discussion"],
        generated_summary="AI sees a sharp increase.",
        analysis_status="complete",
        analysis_model="vision-model",
        source_spans=[SourceSpan(page_index=2, bbox=(1.0, 2.0, 3.0, 4.0))],
    )
    formula = CanonicalFormula(
        formula_id="formula-1",
        latex=r"L = L_{task} + \lambda L_{aux}",
        caption="Equation 1",
        description="The source defines lambda as a balance weight.",
        nearby_block_ids=["discussion"],
        generated_explanation="AI says this regularizes the model.",
        analysis_status="complete",
        analysis_model="language-model",
        source_spans=[SourceSpan(page_index=2, source_block_id="equation-1")],
    )
    builder = StructuredEvidenceBuilder()

    figure_chunk = builder.figure_chunk(figure, nearby)
    formula_chunk = builder.formula_chunk(formula, nearby)

    assert "Figure 1. Source accuracy curve" in figure_chunk.text
    assert "Original note below the source figure." in figure_chunk.text
    assert "source paragraph explains" in figure_chunk.text
    assert "AI sees a sharp increase." not in figure_chunk.text
    assert "AI sees a sharp increase." in figure_chunk.embedding_text
    assert figure_chunk.metadata["provenance"]["generated_summary"]["generated"] is True
    assert r"L = L_{task} + \lambda L_{aux}" in formula_chunk.text
    assert "balance weight" in formula_chunk.text
    assert "AI says this regularizes" not in formula_chunk.text
    assert "AI says this regularizes" in formula_chunk.embedding_text
    assert formula_chunk.metadata["provenance"]["generated_explanation"]["generated"] is True


def test_failed_optional_analysis_is_a_warning_not_rejection() -> None:
    figure = CanonicalFigure(
        figure_id="figure-failed",
        caption="Source caption",
        analysis_status="failed",
        warnings=["Vision analysis timed out."],
    )

    chunk = StructuredEvidenceBuilder().figure_chunk(figure, [])

    assert chunk.metadata["accepted"] is True
    assert chunk.metadata["analysis_status"] == "failed"
    assert chunk.metadata["warnings"] == ["Vision analysis timed out."]


def test_pdf_adapter_preserves_formula_caption_and_source_description(
    tmp_path: Path,
) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"pdf source")
    parsed = ParsedDocument(
        title="Paper",
        text=r"L = L_{task} + \lambda L_{aux}",
        chunks=[
            ParsedChunk(
                ordinal=0,
                text=r"L = L_{task} + \lambda L_{aux}",
                heading="Equation 3",
                page_label="2",
            )
        ],
        metadata={
            "document_intelligence": {
                "formulas": [
                    {
                        "page_label": "2",
                        "latex": r"L = L_{task} + \lambda L_{aux}",
                        "caption": "Equation 3",
                        "description": "The source defines lambda as a balance weight.",
                    }
                ]
            }
        },
    )

    document = _parsed_pdf_to_canonical(source, parsed, "mineru", 2)

    assert document.formulas[0].caption == "Equation 3"
    assert document.formulas[0].description == (
        "The source defines lambda as a balance weight."
    )


def test_markdown_adapter_links_nearby_source_discussion_for_structured_evidence(
    tmp_path: Path,
) -> None:
    image = tmp_path / "curve.png"
    image.write_bytes(b"image")
    source = tmp_path / "paper.md"
    source.write_text(
        "Before the figure, the source introduces the accuracy curve.\n\n"
        "![Figure 1. Accuracy curve](curve.png)\n\n"
        "After the figure, the source discusses the plateau.\n\n"
        "$$\nL = x + y\n$$\n\n"
        "The source explains that x and y are loss terms.\n",
        encoding="utf-8",
    )

    document = parse_canonical_document(source)

    narrative_ids = {
        block.block_id for block in document.blocks if block.block_type == "narrative"
    }
    assert set(document.figures[0].nearby_block_ids).issubset(narrative_ids)
    assert set(document.formulas[0].nearby_block_ids).issubset(narrative_ids)
    assert document.figures[0].nearby_block_ids
    assert document.formulas[0].nearby_block_ids
