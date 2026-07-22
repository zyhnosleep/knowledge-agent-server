from __future__ import annotations

import shutil
from pathlib import Path

from app.services import canonical_adapters
from app.services.canonical_adapters import (
    MarkdownCanonicalAdapter,
    _finalize_structured_evidence,
    _parsed_pdf_to_canonical,
    parse_canonical_document,
)
from app.services.canonical_artifacts import CanonicalArtifactStore
from app.services.canonical_models import (
    CanonicalBlock,
    CanonicalCell,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalTable,
    SourceSpan,
)
from app.services.structured_evidence import (
    StructuredEvidenceBuilder,
    TableRepairMapping,
    TableRepairProof,
    TableValidator,
)
from app.services.canonical_quality import CanonicalQualityGate
from app.services.canonical_table_identity import table_identity_fingerprint
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
    validator = TableValidator()
    request = validator.validate(original).repair_request
    assert request is not None

    missing_proof = validator.validate(original, repaired_table=repaired)
    result = validator.validate(
        original,
        repaired_table=repaired,
        repair_proof=request,
    )

    assert missing_proof.accepted is False
    assert "repair_proof_missing" in missing_proof.reasons
    assert result.accepted is True
    assert result.status == "repaired_by_vision"
    assert result.table.status == "repaired_by_vision"


def test_page_only_repair_proof_is_rejected_without_stable_block_locator() -> None:
    original = _table()
    original.metadata["truncated"] = True
    original.source_spans = [SourceSpan(page_index=0)]
    repaired = _table(table_id=original.table_id)
    repaired.source_spans = [SourceSpan(page_index=0)]
    validator = TableValidator()
    request = validator.validate(original).repair_request
    assert request is not None

    result = validator.validate(original, repaired, repair_proof=request)

    assert result.accepted is False
    assert "repair_stable_locator_missing" in result.reasons


def test_no_bbox_repair_accepts_matching_source_block_id() -> None:
    original = _table()
    original.metadata["truncated"] = True
    original.source_spans = [
        SourceSpan(page_index=0, source_block_id="mineru-table-17")
    ]
    repaired = _table(table_id=original.table_id)
    repaired.source_spans = [
        SourceSpan(page_index=0, source_block_id="mineru-table-17")
    ]
    validator = TableValidator()
    request = validator.validate(original).repair_request
    assert request is not None

    result = validator.validate(original, repaired, repair_proof=request)

    assert result.accepted is True


def test_no_bbox_repair_rejects_mismatched_source_block_id() -> None:
    original = _table()
    original.metadata["truncated"] = True
    original.source_spans = [
        SourceSpan(page_index=0, source_block_id="mineru-table-17")
    ]
    repaired = _table(table_id=original.table_id)
    repaired.source_spans = [
        SourceSpan(page_index=0, source_block_id="mineru-table-99")
    ]
    validator = TableValidator()
    request = validator.validate(original).repair_request
    assert request is not None

    result = validator.validate(original, repaired, repair_proof=request)

    assert result.accepted is False
    assert "repair_locator_mismatch" in result.reasons


def test_unique_page_repair_proof_rejects_model_supplied_mapping_dict() -> None:
    original = _table()
    original.metadata["truncated"] = True
    original.source_spans = [
        SourceSpan(page_index=0, source_block_id="mineru-table-17")
    ]
    repaired = _table(table_id="di-table-17")
    repaired.source_spans = [
        SourceSpan(
            page_index=0,
            source_block_id="document_intelligence-table-17",
        )
    ]
    validator = TableValidator()
    request = validator.validate(original).repair_request
    assert request is not None
    forged_proof = {
        "original_request": request.model_dump(mode="json"),
        "page_index": 0,
        "replacement_content_fingerprint": "forged",
        "match_basis": "unique_table_on_page",
        "validated_mapping": {
            "original_table_id": original.table_id,
            "replacement_table_id": repaired.table_id,
            "page_index": 0,
        },
    }

    result = validator.validate(original, repaired, repair_proof=forged_proof)

    assert result.accepted is False
    assert "repair_proof_invalid" in result.reasons


def test_unique_page_repair_proof_binds_replacement_content() -> None:
    original = _table()
    original.metadata["truncated"] = True
    original.source_spans = [
        SourceSpan(page_index=0, source_block_id="mineru-table-17")
    ]
    repaired = _table(table_id="di-table-17")
    repaired.source_spans = [
        SourceSpan(
            page_index=0,
            source_block_id="document_intelligence-table-17",
        )
    ]
    validator = TableValidator()
    request = validator.validate(original).repair_request
    assert request is not None
    tampered_proof = TableRepairProof(
        original_request=request,
        page_index=0,
        replacement_content_fingerprint="forged",
        match_basis="unique_table_on_page",
        validated_mapping=TableRepairMapping(
            original_table_id=original.table_id,
            replacement_table_id=repaired.table_id,
            page_index=0,
        ),
    )

    result = validator.validate(original, repaired, repair_proof=tampered_proof)

    assert result.accepted is False
    assert "repair_replacement_fingerprint_mismatch" in result.reasons


def test_table_identity_fingerprint_includes_source_block_id() -> None:
    first = _table()
    second = first.model_copy(deep=True)
    assert first.source_spans[0].bbox is not None
    second.source_spans[0].source_block_id = "different-source-block"

    assert table_identity_fingerprint(first) != table_identity_fingerprint(second)


def test_repair_with_same_table_id_but_wrong_source_region_is_rejected() -> None:
    original = _table()
    original.metadata["truncated"] = True
    validator = TableValidator()
    request = validator.validate(original).repair_request
    assert request is not None
    wrong_region = _table(table_id=original.table_id, page_index=7)
    wrong_region.source_spans[0] = SourceSpan(
        page_index=7,
        source_block_id="wrong-region",
        bbox=(20.0, 30.0, 400.0, 600.0),
    )

    result = validator.validate(
        original,
        repaired_table=wrong_region,
        repair_proof=request,
    )

    assert result.accepted is False
    assert "repair_locator_mismatch" in result.reasons


def test_repair_cannot_replace_a_different_table_identity() -> None:
    original = _table()
    original.metadata["truncated"] = True
    unrelated = _table(table_id="table-unrelated")

    result = TableValidator().validate(original, repaired_table=unrelated)

    assert result.accepted is False
    assert result.status == "validation_failed"
    assert "repair_proof_missing" in result.reasons


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


def test_table_validator_rejects_source_html_that_disagrees_with_structure() -> None:
    table = _table()
    table.source_html = (
        "<table><thead><tr><th>Dataset</th><th>Method</th><th>Accuracy</th></tr>"
        "</thead><tbody><tr><td>News</td><td>Wrong</td><td>0%</td></tr>"
        "</tbody></table>"
    )

    result = TableValidator().validate(table)

    assert result.accepted is False
    assert "source_html_mismatch" in result.reasons


def test_table_chunk_metadata_preserves_complete_source_html() -> None:
    table = _table()
    table.source_html = (
        "<table><thead><tr><th>Dataset</th><th>Method</th><th>Accuracy</th></tr>"
        "</thead><tbody><tr><td>News</td><td>Base</td><td>81.2%</td></tr>"
        "<tr><td>News</td><td>Proposed</td><td>84.9%</td></tr>"
        "</tbody></table>"
    )
    builder = StructuredEvidenceBuilder(token_counter=lambda text: len(text))

    parent, children = builder.table_chunks(table, max_tokens=1000)

    assert parent.metadata["source_html"] == table.source_html
    assert children[0].metadata["source_html"] == table.source_html


def test_child_includes_rowspan_cell_whose_occupied_range_intersects_selected_row() -> None:
    span = SourceSpan(page_index=0, source_block_id="merged-source")
    table = CanonicalTable(
        table_id="table-rowspan-child",
        caption="Table 3. Grouped metrics",
        headers=["Group", "Metric"],
        rows=[["A", "F1"], ["", "Recall"]],
        cells=[
            CanonicalCell(text="Group", row_index=0, column_index=0, is_header=True),
            CanonicalCell(text="Metric", row_index=0, column_index=1, is_header=True),
            CanonicalCell(
                text="A",
                row_index=1,
                column_index=0,
                rowspan=2,
                bbox=(1.0, 2.0, 3.0, 9.0),
                source_spans=[span],
            ),
            CanonicalCell(text="F1", row_index=1, column_index=1),
            CanonicalCell(text="Recall", row_index=2, column_index=1),
        ],
        source_markdown=(
            "| Group | Metric |\n| --- | --- |\n| A | F1 |\n|  | Recall |"
        ),
        normalized_markdown=(
            "| Group | Metric |\n| --- | --- |\n| A | F1 |\n|  | Recall |"
        ),
        source_spans=[SourceSpan(page_index=0, source_block_id="table-source")],
        metadata={"table_number": "3"},
    )
    builder = StructuredEvidenceBuilder(token_counter=lambda text: len(text))

    _parent, children = builder.table_chunks(table, max_tokens=1000)
    second_row = next(child for child in children if child.metadata["row_indices"] == [1])
    merged_cell = next(cell for cell in second_row.metadata["cells"] if cell["text"] == "A")

    assert merged_cell["rowspan"] == 2
    assert merged_cell["bbox"] == [1.0, 2.0, 3.0, 9.0]
    assert merged_cell["source_spans"][0]["source_block_id"] == "merged-source"
    assert any(span.source_block_id == "merged-source" for span in second_row.source_spans)


def test_long_table_chunks_repeat_caption_and_header_with_token_bound() -> None:
    rows = [
        [f"Dataset-{index // 3}", f"Method-{index}", f"{80 + index / 10:.1f}%"]
        for index in range(18)
    ]
    table = _table(rows=rows)
    builder = StructuredEvidenceBuilder(token_counter=lambda text: len(text.split()))

    parent, children = builder.table_chunks(table, max_tokens=48)
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
    assert (parent, children) == builder.table_chunks(table, max_tokens=48)


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

    _parent, children = StructuredEvidenceBuilder(
        token_counter=lambda text: len(text.split())
    ).table_chunks(table, max_tokens=45)

    assert [child.metadata["row_indices"] for child in children] == [[0, 1], [2, 3]]


def test_single_overlong_row_is_explicit_overflow_without_truncation() -> None:
    long_value = " ".join(f"token-{index}" for index in range(80))
    table = _table(rows=[["News", "Verbose", long_value]])

    _parent, children = StructuredEvidenceBuilder(
        token_counter=lambda text: len(text.split())
    ).table_chunks(table, max_tokens=24)
    child = children[0]

    assert child.metadata["overflow"] is True
    assert child.token_count > 24
    assert long_value in child.text
    assert child.metadata["rows"] == table.rows


def test_token_count_fallback_is_conservative_for_cjk_formula_and_long_identifier() -> None:
    def unavailable(_name: str):
        raise OSError("tokenizer is not cached")

    long_identifier = "ModelIdentifier" * 30
    table = _table(
        rows=[["中文数据集没有空格", r"损失函数L=Σ_i(x_i-y_i)^2", long_identifier]]
    )
    builder = StructuredEvidenceBuilder(
        tokenizer_name="local/test-tokenizer",
        tokenizer_loader=unavailable,
    )

    parent, children = builder.table_chunks(table, max_tokens=160)

    assert parent.token_count == len(parent.text.encode("utf-8"))
    assert parent.metadata["token_count_mode"] == "utf8_bytes_fallback"
    assert parent.metadata["tokenizer_name"] == "local/test-tokenizer"
    assert children[0].metadata["overflow"] is True
    assert children[0].token_count > 160
    assert long_identifier in children[0].text


def test_configured_tokenizer_loader_drives_real_token_boundaries() -> None:
    loaded: list[str] = []

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return list(text.encode("utf-8"))

    def load(name: str) -> FakeTokenizer:
        loaded.append(name)
        return FakeTokenizer()

    table = _table(rows=[["中文", r"x_i^2", "91.0%"]])
    builder = StructuredEvidenceBuilder(
        tokenizer_name="configured-local-tokenizer",
        tokenizer_loader=load,
    )

    parent, children = builder.table_chunks(table, max_tokens=512)

    assert loaded == ["configured-local-tokenizer"]
    assert parent.token_count == len(parent.text.encode("utf-8"))
    assert children[0].metadata["token_count_mode"] == "transformers"
    assert children[0].metadata["tokenizer_name"] == "configured-local-tokenizer"


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
    first.source_html = (
        "<table><tr><th>Dataset</th><th>Method</th><th>Accuracy</th></tr>"
        "<tr><td>News</td><td>Base</td><td>81.2%</td></tr></table>"
    )
    continuation.source_html = (
        "<table><tr><th>Dataset</th><th>Method</th><th>Accuracy</th></tr>"
        "<tr><td>Dataset</td><td>Method</td><td>Accuracy</td></tr>"
        "<tr><td>News</td><td>Proposed</td><td>84.9%</td></tr></table>"
    )
    first_source_markdown = first.source_markdown
    continuation_source_markdown = continuation.source_markdown
    first_source_html = first.source_html
    continuation_source_html = continuation.source_html

    builder = StructuredEvidenceBuilder(token_counter=lambda _text: 0)
    merged = builder.merge_cross_page_tables([first, continuation])

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
    assert table.source_markdown == first_source_markdown
    assert table.source_html == first_source_html
    assert table.metadata["source_markdowns"] == [
        first_source_markdown,
        continuation_source_markdown,
    ]
    assert table.metadata["source_htmls"] == [first_source_html, continuation_source_html]
    parent, _children = StructuredEvidenceBuilder(
        token_counter=lambda text: len(text.split())
    ).table_chunks(table, max_tokens=80)
    assert parent.metadata["status"] == "cross_page_merged"
    assert parent.metadata["source_markdowns"] == table.metadata["source_markdowns"]
    assert parent.metadata["source_htmls"] == table.metadata["source_htmls"]


def test_quality_gate_accepts_consistent_cross_page_source_segments() -> None:
    first = _table(
        table_id="table-page-1",
        rows=[["News", "Base", "81.2%"]],
        page_index=0,
    )
    continuation = _table(
        table_id="table-page-2",
        rows=[["News", "Proposed", "84.9%"]],
        page_index=1,
        metadata={"continuation_of": first.table_id},
    )
    merged = StructuredEvidenceBuilder(
        token_counter=lambda _text: 0
    ).merge_cross_page_tables([first, continuation])[0]
    document = CanonicalDocument(
        document_id="cross-page",
        parser_source="mineru",
        parse_version="canonical-v1",
        blocks=[
            CanonicalBlock(
                block_id="source-prose",
                block_type="narrative",
                text="Source evidence",
                reading_order=0,
                parser_source="mineru",
            )
        ],
        tables=[merged],
    )

    report = CanonicalQualityGate().evaluate(document)

    assert report.accepted is True
    assert not any(issue.code == "table_invalid" for issue in report.issues)


def test_structured_finalization_recomputes_quality_score_and_fallback_for_bad_html() -> None:
    table = _table(page_index=0)
    table.source_html = (
        "<table><tr><th>Dataset</th><th>Method</th><th>Accuracy</th></tr>"
        "<tr><td>News</td><td>Wrong</td><td>0%</td></tr></table>"
    )
    document = CanonicalDocument(
        document_id="invalid-html",
        parser_source="mineru",
        parse_version="canonical-v1",
        blocks=[
            CanonicalBlock(
                block_id="table-block",
                block_type="table",
                text=table.normalized_markdown or "table",
                reading_order=0,
                parser_source="mineru",
                table_id=table.table_id,
                source_spans=table.source_spans,
            )
        ],
        tables=[table],
    )

    _finalize_structured_evidence(document)

    issue = next(item for item in document.quality.issues if item.code == "table_invalid")
    assert document.quality.status == "validation_failed"
    assert document.quality.accepted is False
    assert document.quality.score == 0.85
    assert document.quality.fallback_pages == [1]
    assert issue.repair_scope == "page:1"
    assert "source_html_mismatch" in issue.metadata["reasons"]


def test_structured_finalization_preserves_strict_only_reasons_in_quality_report() -> None:
    table = _table(page_index=2, metadata={"truncated": True})
    document = CanonicalDocument(
        document_id="silent-truncation",
        parser_source="mineru",
        parse_version="canonical-v1",
        blocks=[
            CanonicalBlock(
                block_id="table-block",
                block_type="table",
                text=table.normalized_markdown or "table",
                reading_order=0,
                parser_source="mineru",
                table_id=table.table_id,
                source_spans=table.source_spans,
            )
        ],
        tables=[table],
    )

    _finalize_structured_evidence(document)

    issue = next(item for item in document.quality.issues if item.code == "table_invalid")
    assert document.quality.score == 0.85
    assert document.quality.fallback_pages == [3]
    assert "silent_truncation" in issue.metadata["reasons"]


def test_same_headers_without_continuation_are_not_merged() -> None:
    first = _table(table_id="unrelated-1", page_index=0)
    second = _table(table_id="unrelated-2", page_index=1)

    result = StructuredEvidenceBuilder(token_counter=lambda _text: 0).merge_cross_page_tables(
        [first, second]
    )

    assert [table.table_id for table in result] == ["unrelated-1", "unrelated-2"]


def test_adapter_completion_merges_explicit_continuations_and_validates_tables(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "paper.md"
    source.write_text("placeholder", encoding="utf-8")
    first = _table(
        table_id="table-page-1",
        rows=[["News", "Base", "81.2%"]],
        page_index=0,
    )
    second = _table(
        table_id="table-page-2",
        rows=[["News", "Proposed", "84.9%"]],
        page_index=1,
        metadata={"continuation_of": first.table_id},
    )
    document = CanonicalDocument(
        document_id="doc-tables",
        source_path=str(source),
        parser_source="mineru",
        parse_version="canonical-v1",
        tables=[first, second],
    )
    monkeypatch.setattr(
        MarkdownCanonicalAdapter,
        "parse",
        lambda self, path: document,
    )

    result = parse_canonical_document(source)

    assert len(result.tables) == 1
    assert result.tables[0].status == "cross_page_merged"
    assert result.metadata["table_repair_requests"] == []


def test_adapter_completion_persists_invalid_status_and_repair_signal(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid.md"
    source.write_text(
        "| Dataset | Method | Accuracy |\n"
        "| --- | --- | --- |\n"
        "| News | Base |\n",
        encoding="utf-8",
    )

    document = parse_canonical_document(source)

    assert document.tables[0].status == "validation_failed"
    assert document.quality.accepted is False
    assert document.quality.status == "validation_failed"
    requests = document.metadata["table_repair_requests"]
    assert requests[0]["table_id"] == document.tables[0].table_id
    assert "row_width_mismatch" in requests[0]["reasons"]


def test_figure_and_formula_evidence_keep_source_and_generated_text_separate() -> None:
    nearby = [
        CanonicalBlock(
            block_id="discussion",
            block_type="narrative",
            text="The source paragraph explains the rising trend and the loss term.",
            reading_order=0,
            parser_source="mineru",
            source_spans=[SourceSpan(page_index=2, source_block_id="paragraph-7")],
        ),
        CanonicalBlock(
            block_id="generated-discussion",
            block_type="narrative",
            text="DO NOT EXPOSE NESTED GENERATED DISCUSSION",
            reading_order=1,
            parser_source="vision-model",
            metadata={"provenance": {"generated": True}},
        ),
    ]
    figure = CanonicalFigure(
        figure_id="figure-1",
        caption="Figure 1. Source accuracy curve",
        description="Original note below the source figure.",
        asset_path="assets/figure.png",
        nearby_block_ids=["discussion", "generated-discussion"],
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
        nearby_block_ids=["discussion", "generated-discussion"],
        generated_explanation="AI says this regularizes the model.",
        analysis_status="complete",
        analysis_model="language-model",
        source_spans=[SourceSpan(page_index=2, source_block_id="equation-1")],
    )
    builder = StructuredEvidenceBuilder(token_counter=lambda text: len(text.split()))

    figure_chunk = builder.figure_chunk(figure, nearby)
    formula_chunk = builder.formula_chunk(formula, nearby)

    assert "Figure 1. Source accuracy curve" in figure_chunk.text
    assert "Original note below the source figure." in figure_chunk.text
    assert "source paragraph explains" in figure_chunk.text
    assert "AI sees a sharp increase." not in figure_chunk.text
    assert "DO NOT EXPOSE NESTED GENERATED DISCUSSION" not in figure_chunk.text
    assert "AI sees a sharp increase." in figure_chunk.embedding_text
    assert figure_chunk.metadata["provenance"]["generated_summary"]["generated"] is True
    assert r"L = L_{task} + \lambda L_{aux}" in formula_chunk.text
    assert "balance weight" in formula_chunk.text
    assert "AI says this regularizes" not in formula_chunk.text
    assert "DO NOT EXPOSE NESTED GENERATED DISCUSSION" not in formula_chunk.text
    assert "AI says this regularizes" in formula_chunk.embedding_text
    assert formula_chunk.metadata["provenance"]["generated_explanation"]["generated"] is True


def test_failed_optional_analysis_is_a_warning_not_rejection() -> None:
    figure = CanonicalFigure(
        figure_id="figure-failed",
        caption="Source caption",
        analysis_status="failed",
        warnings=["Vision analysis timed out."],
    )

    chunk = StructuredEvidenceBuilder(
        token_counter=lambda text: len(text.split())
    ).figure_chunk(figure, [])

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


def test_mineru_figure_asset_is_durable_after_parser_temp_cleanup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class TestSettings:
        cache_dir = tmp_path / "adapter-cache"

    monkeypatch.setattr(canonical_adapters, "get_settings", lambda: TestSettings())
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"pdf source")
    mineru_output = tmp_path / "mineru-temp"
    image = mineru_output / "images" / "figure.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"mineru figure pixels")
    content_list = mineru_output / "content_list.json"
    content_list.write_text("[]", encoding="utf-8")
    parsed = ParsedDocument(
        title="Paper",
        text="Figure 1. Results",
        chunks=[
            ParsedChunk(
                ordinal=0,
                text="Figure 1. Results",
                heading="Figure 1",
                page_label="1",
            )
        ],
        metadata={
            "document_intelligence": {
                "output_dir": str(mineru_output),
                "content_list_path": str(content_list),
                "figures": [
                    {
                        "page_label": "1",
                        "caption": "Figure 1. Results",
                        "image_path": "images/figure.png",
                    }
                ],
            }
        },
    )

    document = _parsed_pdf_to_canonical(source, parsed, "mineru", 1)
    cached_source = Path(document.assets[0].source_path or "")
    assert cached_source.is_file()
    assert mineru_output not in cached_source.parents
    shutil.rmtree(mineru_output)

    staging = CanonicalArtifactStore(tmp_path / "store").write_staging(
        document.document_id,
        document.parse_version,
        document,
    )

    assert (staging / document.assets[0].path).read_bytes() == b"mineru figure pixels"


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
