from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import get_settings
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
from app.services.canonical_table_identity import (
    table_content_fingerprint,
    table_identity_fingerprint,
)
from app.services.parser import ParsedChunk, ParsedDocument
from app.services.pipeline import IngestionPipeline


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


def test_semantic_split_checkpoint_records_completed_fidelity_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import semantic_chunking

    canonical = CanonicalDocument(
        document_id="doc-fidelity",
        parser_source="test",
        parse_version="version-fidelity",
    )

    class FakeChunker:
        def __init__(self, _embedder) -> None:
            pass

        def build(self, loaded: CanonicalDocument):
            assert loaded is canonical
            return [SimpleNamespace(model_dump=lambda **_kwargs: {"local_id": "draft-1"})]

    monkeypatch.setattr(semantic_chunking, "SemanticChunker", FakeChunker)
    monkeypatch.setattr(CanonicalArtifactStore, "load", lambda *_args: canonical)
    captured: dict[str, object] = {}

    pipeline = SimpleNamespace(
        ollama=object(),
        validate_ingestion_identity=lambda *_args: None,
        _write_stage_artifact=lambda _context, _filename, _payload, *, extra: captured.update(extra)
        or dict(extra),
    )
    context = SimpleNamespace(
        document=SimpleNamespace(id=canonical.document_id),
        version=SimpleNamespace(version_key=canonical.parse_version),
        stage="semantic_split",
    )

    result = IngestionPipeline._run_semantic_split_stage(pipeline, context)

    assert result["source_fidelity_completeness"] == 1.0
    assert result["structured_limit_completeness"] == 1.0
    assert captured == {
        "chunk_count": 1,
        "source_fidelity_completeness": 1.0,
        "structured_limit_completeness": 1.0,
    }


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
    bindings = validator.validate_repair_inventory(
        [original], [repaired], {original.table_id}, 0
    )

    assert missing_proof.accepted is False
    assert "repair_proof_missing" in missing_proof.reasons
    assert bindings is not None
    result = bindings[0][2]
    assert result.accepted is True
    assert result.status == "repaired_by_vision"
    assert result.table.status == "repaired_by_vision"


def _set_normalized_table_locator(
    table: CanonicalTable,
    bbox: tuple[float, float, float, float],
    *,
    source_region_id: str | None = None,
) -> None:
    table.source_spans = [
        SourceSpan(
            page_index=0,
            normalized_bbox=bbox,
            metadata=(
                {"source_region_id": source_region_id}
                if source_region_id is not None
                else {}
            ),
        )
    ]


def test_single_repair_inventory_accepts_small_normalized_bbox_shift() -> None:
    original = _table()
    original.metadata["truncated"] = True
    replacement = _table(table_id="vision-table-1")
    _set_normalized_table_locator(original, (0.100, 0.200, 0.400, 0.600))
    _set_normalized_table_locator(replacement, (0.101, 0.199, 0.401, 0.601))

    bindings = TableValidator().validate_repair_inventory(
        [original], [replacement], {original.table_id}, 0
    )

    assert bindings is not None
    assert bindings[0][1].table_id == replacement.table_id


def test_multi_repair_inventory_uniquely_matches_shifted_normalized_bboxes() -> None:
    first = _table(table_id="original-1", rows=[["A", "Base", "1"]])
    second = _table(table_id="original-2", rows=[["B", "Base", "2"]])
    first.metadata["truncated"] = True
    second.metadata["truncated"] = True
    first_repair = _table(table_id="repair-1", rows=[["A", "Fixed", "10"]])
    second_repair = _table(table_id="repair-2", rows=[["B", "Fixed", "20"]])
    _set_normalized_table_locator(first, (0.100, 0.100, 0.400, 0.300))
    _set_normalized_table_locator(second, (0.550, 0.500, 0.850, 0.800))
    _set_normalized_table_locator(first_repair, (0.101, 0.099, 0.401, 0.301))
    _set_normalized_table_locator(second_repair, (0.549, 0.501, 0.851, 0.799))

    bindings = TableValidator().validate_repair_inventory(
        [first, second],
        [second_repair, first_repair],
        {first.table_id, second.table_id},
        0,
    )

    assert bindings is not None
    assert {
        original.table_id: replacement.table_id
        for original, replacement, _, _ in bindings
    } == {"original-1": "repair-1", "original-2": "repair-2"}


def test_repair_inventory_prioritizes_exact_source_region_over_ambiguous_bbox() -> None:
    first = _table(table_id="original-region-a", rows=[["A", "Base", "1"]])
    second = _table(table_id="original-region-b", rows=[["B", "Base", "2"]])
    first.metadata["truncated"] = True
    second.metadata["truncated"] = True
    repair_a = _table(table_id="repair-region-a", rows=[["A", "Fixed", "10"]])
    repair_b = _table(table_id="repair-region-b", rows=[["B", "Fixed", "20"]])
    shared_bbox = (0.100, 0.100, 0.400, 0.400)
    _set_normalized_table_locator(first, shared_bbox, source_region_id="region-a")
    _set_normalized_table_locator(second, shared_bbox, source_region_id="region-b")
    _set_normalized_table_locator(repair_a, shared_bbox, source_region_id="region-a")
    _set_normalized_table_locator(repair_b, shared_bbox, source_region_id="region-b")

    bindings = TableValidator().validate_repair_inventory(
        [first, second],
        [repair_b, repair_a],
        {first.table_id, second.table_id},
        0,
    )

    assert bindings is not None
    assert {
        original.table_id: replacement.table_id
        for original, replacement, _, _ in bindings
    } == {
        "original-region-a": "repair-region-a",
        "original-region-b": "repair-region-b",
    }


def test_repair_inventory_rejects_low_iou_and_ambiguous_bbox_graphs() -> None:
    original = _table(table_id="original-low")
    original.metadata["truncated"] = True
    far_repair = _table(table_id="repair-far")
    _set_normalized_table_locator(original, (0.100, 0.100, 0.300, 0.300))
    _set_normalized_table_locator(far_repair, (0.600, 0.600, 0.800, 0.800))
    validator = TableValidator()

    assert (
        validator.validate_repair_inventory(
            [original], [far_repair], {original.table_id}, 0
        )
        is None
    )

    first = _table(table_id="original-a", rows=[["A", "Base", "1"]])
    second = _table(table_id="original-b", rows=[["B", "Base", "2"]])
    first.metadata["truncated"] = True
    second.metadata["truncated"] = True
    repair_a = _table(table_id="repair-a", rows=[["A", "Fixed", "10"]])
    repair_b = _table(table_id="repair-b", rows=[["B", "Fixed", "20"]])
    _set_normalized_table_locator(first, (0.100, 0.100, 0.400, 0.400))
    _set_normalized_table_locator(second, (0.105, 0.100, 0.405, 0.400))
    _set_normalized_table_locator(repair_a, (0.102, 0.100, 0.402, 0.400))
    _set_normalized_table_locator(repair_b, (0.103, 0.100, 0.403, 0.400))

    assert (
        validator.validate_repair_inventory(
            [first, second],
            [repair_a, repair_b],
            {first.table_id, second.table_id},
            0,
        )
        is None
    )


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

    bindings = validator.validate_repair_inventory(
        [original], [repaired], {original.table_id}, 0
    )

    assert bindings is not None
    assert bindings[0][2].accepted is True


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


def test_public_typed_repair_proof_is_not_an_authorization_capability() -> None:
    original = _table()
    original.metadata["truncated"] = True
    original.source_spans = [
        SourceSpan(page_index=0, source_block_id="mineru-table-17")
    ]
    repaired = _table(table_id="di-table-17")
    repaired.source_spans = [
        SourceSpan(page_index=0, source_block_id="di-table-17")
    ]
    validator = TableValidator()
    request = validator.validate(original).repair_request
    assert request is not None
    public_proof = TableRepairProof(
        original_request=request,
        page_index=0,
        replacement_content_fingerprint=table_content_fingerprint(repaired),
        match_basis="unique_table_on_page",
        validated_mapping=TableRepairMapping(
            original_table_id=original.table_id,
            replacement_table_id=repaired.table_id,
            page_index=0,
        ),
    )

    result = validator.validate(original, repaired, repair_proof=public_proof)

    assert result.accepted is False
    assert "repair_inventory_missing" in result.reasons


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
    table.metadata["fragmented_numeric_tokens"] = ["84.9", "%"]

    result = TableValidator().validate(table)

    assert result.accepted is False
    assert "numeric_token_fragmented" in result.reasons


@pytest.mark.parametrize(
    "headers,row",
    [
        (["Value", "Measure"], ["10", "kg"]),
        (["Magnitude", "Symbol"], ["10", "kg"]),
        (["Value", "Punctuation"], ["10", "."]),
    ],
)
def test_table_validator_does_not_infer_fragmentation_from_adjacent_text_cells(
    headers: list[str], row: list[str]
) -> None:
    markdown = (
        f"| {headers[0]} | {headers[1]} |\n| --- | --- |\n"
        f"| {row[0]} | {row[1]} |"
    )
    span = SourceSpan(page_index=0, source_block_id="table-source")
    table = CanonicalTable(
        table_id="legitimate-columns",
        headers=headers,
        rows=[row],
        cells=[
            CanonicalCell(text=headers[0], row_index=0, column_index=0, is_header=True),
            CanonicalCell(text=headers[1], row_index=0, column_index=1, is_header=True),
            CanonicalCell(text=row[0], row_index=1, column_index=0),
            CanonicalCell(text=row[1], row_index=1, column_index=1),
        ],
        source_markdown=markdown,
        normalized_markdown=markdown,
        source_spans=[span],
    )

    result = TableValidator().validate(table)

    assert result.accepted is True


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


def test_single_overlong_row_is_losslessly_split_without_overflow() -> None:
    long_value = " ".join(f"token-{index}" for index in range(80))
    table = _table(rows=[["News", "Verbose", long_value]])

    _parent, children = StructuredEvidenceBuilder(
        token_counter=lambda text: len(text.split())
    ).table_chunks(table, max_tokens=24)
    long_cell_children = [
        child
        for child in children
        if child.metadata.get("cell_fragment_column_index") == 2
    ]

    assert len(long_cell_children) > 1
    assert all(child.metadata["overflow"] is False for child in children)
    assert all(child.token_count <= 24 for child in children)
    assert "".join(
        child.metadata["source_fragment"] for child in long_cell_children
    ) == long_value
    assert all(child.metadata["rows"] == table.rows for child in children)


def test_overlong_table_row_keeps_short_footnote_in_direct_embedding_children() -> None:
    long_value = " ".join(f"token-{index}" for index in range(80))
    table = _table(rows=[["News", "Verbose", long_value]])
    table.footnotes = ["* denotes statistical significance."]

    _parent, children = StructuredEvidenceBuilder(
        token_counter=lambda text: len(text.split())
    ).table_chunks(table, max_tokens=28)
    row_children = [
        child for child in children if child.metadata.get("row_indices") == [0]
    ]

    assert row_children
    assert all("Footnote: * denotes statistical significance." in child.text for child in row_children)
    assert all(child.embedding_text == child.text for child in row_children)
    assert all(child.token_count <= 28 for child in children)


def test_overlong_table_footnote_gets_lossless_bounded_citation_children() -> None:
    long_value = " ".join(f"value-{index}" for index in range(60))
    long_footnote = " ".join(f"footnote-{index}" for index in range(70))
    table = _table(rows=[["News", "Verbose", long_value]])
    table.footnotes = [long_footnote]

    _parent, children = StructuredEvidenceBuilder(
        token_counter=lambda text: len(text.split())
    ).table_chunks(table, max_tokens=24)
    footnote_children = [
        child for child in children if child.metadata.get("footnote_index") == 0
    ]

    assert len(footnote_children) > 1
    assert all(child.chunk_role == "child" for child in footnote_children)
    assert all(child.source_spans for child in footnote_children)
    assert all(child.token_count <= 24 for child in footnote_children)
    assert "".join(
        child.metadata["source_fragment"] for child in footnote_children
    ) == long_footnote


def test_overlong_identity_cell_is_losslessly_split_without_overflow() -> None:
    long_identity = " ".join(f"dataset-{index}" for index in range(80))
    table = _table(rows=[[long_identity, "Proposed", "84.9%"]])

    _parent, children = StructuredEvidenceBuilder(
        token_counter=lambda text: len(text.split())
    ).table_chunks(table, max_tokens=24)
    identity_children = [
        child
        for child in children
        if child.metadata.get("cell_fragment_column_index") == 0
    ]

    assert len(identity_children) > 1
    assert all(child.metadata["overflow"] is False for child in children)
    assert all(child.token_count <= 24 for child in children)
    assert "".join(
        child.metadata["source_fragment"] for child in identity_children
    ) == long_identity


def test_overlong_formula_description_is_losslessly_split_without_overflow() -> None:
    description = " ".join(f"definition-{index}" for index in range(80))
    formula = CanonicalFormula(
        formula_id="formula-long-description",
        latex="x = y + 1",
        caption="Equation 3.",
        description=description,
        source_spans=[SourceSpan(page_index=2, source_block_id="formula-source")],
    )
    _parent, children = StructuredEvidenceBuilder(
        token_counter=lambda text: len(text.split())
    ).formula_chunks(formula, [], max_tokens=20)
    description_children = [
        child
        for child in children
        if child.metadata.get("source_fragment_kind") == "description"
    ]

    assert len(description_children) > 1
    assert all(child.token_count <= 20 for child in children)
    assert all(child.embedding_text == child.text for child in children)
    assert "".join(
        child.metadata["source_fragment"] for child in description_children
    ) == description


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
    assert all(child.metadata["overflow"] is False for child in children)
    assert all(child.token_count <= 160 for child in children)
    assert long_identifier in "".join(
        str(child.metadata.get("source_fragment") or "") for child in children
    )


def test_strict_tokenizer_loading_fails_instead_of_changing_count_mode() -> None:
    def unavailable(_name: str):
        raise OSError("tokenizer is not cached")

    builder = StructuredEvidenceBuilder(
        tokenizer_name="local/required-tokenizer",
        tokenizer_loader=unavailable,
        strict_tokenizer=True,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "local/required-tokenizer.*5cf2132abc99cad020ac570b19d031efec650f2b"
            ".*local cache"
        ),
    ):
        builder.estimate_tokens("must use real tokenizer tokens")

    assert builder.token_count_mode == "transformers"


def test_tokenizer_asset_hash_is_stable_and_content_addressed(tmp_path: Path) -> None:
    from app.services.ingestion_identity import tokenizer_asset_content_hash

    first = tmp_path / "tokenizer.json"
    second = tmp_path / "tokenizer_config.json"
    first.write_text('{"model":"qwen"}', encoding="utf-8")
    second.write_text('{"padding_side":"left"}', encoding="utf-8")

    initial = tokenizer_asset_content_hash(tmp_path)
    assert initial == tokenizer_asset_content_hash(tmp_path)
    assert len(initial) == 64

    second.write_text('{"padding_side":"right"}', encoding="utf-8")
    assert tokenizer_asset_content_hash(tmp_path) != initial


def _write_local_tokenizer_identity(
    snapshot_path: Path,
    *,
    name: str = "Qwen/Qwen3-Embedding-4B",
    revision: str = "5cf2132abc99cad020ac570b19d031efec650f2b",
    content_sha256: str | None = None,
) -> None:
    from app.services.ingestion_identity import tokenizer_asset_content_hash

    payload = {
        "schema_version": "knowledge-agent-tokenizer-snapshot-v1",
        "name": name,
        "revision": revision,
        "content_sha256": content_sha256 or tokenizer_asset_content_hash(snapshot_path),
    }
    (snapshot_path / ".knowledge-agent-tokenizer-snapshot.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def test_explicit_local_snapshot_requires_trusted_identity_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    from app.services import ingestion_identity as identity_module

    (tmp_path / "tokenizer.json").write_text('{"model":"qwen"}', encoding="utf-8")
    monkeypatch.setattr(
        identity_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    identity_module.resolve_local_tokenizer.cache_clear()

    with pytest.raises(
        identity_module.TokenizerUnavailableError,
        match="identity metadata",
    ):
        identity_module.resolve_local_tokenizer(
            "Qwen/Qwen3-Embedding-4B",
            "5cf2132abc99cad020ac570b19d031efec650f2b",
            local_path=tmp_path,
        )


@pytest.mark.parametrize(
    ("revision", "content_sha256", "error"),
    [
        ("wrong-revision", None, "revision"),
        ("5cf2132abc99cad020ac570b19d031efec650f2b", "b" * 64, "SHA-256"),
    ],
)
def test_explicit_local_snapshot_rejects_untrusted_identity(
    tmp_path: Path,
    monkeypatch,
    revision: str,
    content_sha256: str | None,
    error: str,
) -> None:
    from app.services import ingestion_identity as identity_module

    (tmp_path / "tokenizer.json").write_text('{"model":"qwen"}', encoding="utf-8")
    _write_local_tokenizer_identity(
        tmp_path,
        revision=revision,
        content_sha256=content_sha256,
    )
    monkeypatch.setattr(
        identity_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    identity_module.resolve_local_tokenizer.cache_clear()

    with pytest.raises(identity_module.TokenizerUnavailableError, match=error):
        identity_module.resolve_local_tokenizer(
            "Qwen/Qwen3-Embedding-4B",
            "5cf2132abc99cad020ac570b19d031efec650f2b",
            local_path=tmp_path,
        )


def test_explicit_local_snapshot_keeps_logical_pinned_identity(
    tmp_path: Path, monkeypatch
) -> None:
    from app.services import ingestion_identity as identity_module

    (tmp_path / "tokenizer.json").write_text('{"model":"qwen"}', encoding="utf-8")
    _write_local_tokenizer_identity(tmp_path)
    calls: list[tuple[str, dict[str, object]]] = []
    tokenizer = object()

    def load(path: str, **kwargs):
        calls.append((path, kwargs))
        return tokenizer

    monkeypatch.setattr(identity_module.AutoTokenizer, "from_pretrained", load)
    identity_module.resolve_local_tokenizer.cache_clear()

    resolved = identity_module.resolve_local_tokenizer(
        "Qwen/Qwen3-Embedding-4B",
        "5cf2132abc99cad020ac570b19d031efec650f2b",
        local_path=tmp_path,
    )

    assert resolved.tokenizer is tokenizer
    assert calls == [(str(tmp_path.resolve()), {"local_files_only": True})]
    assert resolved.identity == {
        "name": "Qwen/Qwen3-Embedding-4B",
        "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
        "content_sha256": identity_module.tokenizer_asset_content_hash(tmp_path),
    }


def test_nonexistent_explicit_snapshot_fails_with_diagnostic(tmp_path: Path) -> None:
    from app.services import ingestion_identity as identity_module

    missing = tmp_path / "missing-tokenizer"
    identity_module.resolve_local_tokenizer.cache_clear()

    with pytest.raises(
        identity_module.TokenizerUnavailableError,
        match="Qwen/Qwen3-Embedding-4B.*revision.*local cache.*missing-tokenizer",
    ):
        identity_module.resolve_local_tokenizer(
            "Qwen/Qwen3-Embedding-4B",
            "5cf2132abc99cad020ac570b19d031efec650f2b",
            local_path=missing,
        )


def test_hf_cache_resolution_is_local_only_and_diagnostic(monkeypatch) -> None:
    from app.services import ingestion_identity as identity_module

    calls: list[dict[str, object]] = []

    def unavailable(**kwargs):
        calls.append(kwargs)
        raise OSError("pinned snapshot missing")

    monkeypatch.setattr(identity_module, "snapshot_download", unavailable)
    identity_module.resolve_local_tokenizer.cache_clear()

    with pytest.raises(
        identity_module.TokenizerUnavailableError,
        match="5cf2132abc99cad020ac570b19d031efec650f2b.*local cache",
    ):
        identity_module.resolve_local_tokenizer(
            "Qwen/Qwen3-Embedding-4B",
            "5cf2132abc99cad020ac570b19d031efec650f2b",
        )

    assert calls == [
        {
            "repo_id": "Qwen/Qwen3-Embedding-4B",
            "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
            "local_files_only": True,
        }
    ]


def test_injected_counter_ignores_unavailable_configured_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    settings = get_settings()
    monkeypatch.setattr(
        settings,
        "semantic_tokenizer_local_path",
        tmp_path / "missing-tokenizer",
    )
    builder = StructuredEvidenceBuilder(token_counter=lambda text: len(text.split()))

    assert builder.estimate_tokens("alpha beta gamma") == 3
    assert builder.token_count_mode == "injected_counter"


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


def test_tokenizer_loader_is_cached_per_process_and_model_name() -> None:
    loaded: list[str] = []

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return list(text.encode("utf-8"))

    def load(name: str) -> FakeTokenizer:
        loaded.append(name)
        return FakeTokenizer()

    name = "cache-test-tokenizer-unique"
    first = StructuredEvidenceBuilder(tokenizer_name=name, tokenizer_loader=load)
    second = StructuredEvidenceBuilder(tokenizer_name=name, tokenizer_loader=load)

    assert loaded == []
    assert first.estimate_tokens("abc") == second.estimate_tokens("abc") == 3
    assert loaded == [name]


def test_tokenizer_cache_key_includes_loader_identity() -> None:
    loaded: list[str] = []

    class FakeTokenizer:
        def __init__(self, width: int) -> None:
            self.width = width

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return [self.width] * self.width

    def first_loader(name: str) -> FakeTokenizer:
        loaded.append(f"first:{name}")
        return FakeTokenizer(1)

    def second_loader(name: str) -> FakeTokenizer:
        loaded.append(f"second:{name}")
        return FakeTokenizer(2)

    name = "same-model-different-loader"
    first = StructuredEvidenceBuilder(tokenizer_name=name, tokenizer_loader=first_loader)
    second = StructuredEvidenceBuilder(tokenizer_name=name, tokenizer_loader=second_loader)

    assert first.estimate_tokens("x") == 1
    assert second.estimate_tokens("x") == 2
    assert loaded == [f"first:{name}", f"second:{name}"]


def test_lazy_tokenizer_load_is_singleton_across_concurrent_builders() -> None:
    loaded: list[str] = []

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return list(text.encode("utf-8"))

    def load(name: str) -> FakeTokenizer:
        loaded.append(name)
        time.sleep(0.02)
        return FakeTokenizer()

    name = "concurrent-lazy-tokenizer"
    builders = [
        StructuredEvidenceBuilder(tokenizer_name=name, tokenizer_loader=load)
        for _ in range(8)
    ]
    assert loaded == []

    with ThreadPoolExecutor(max_workers=8) as executor:
        counts = list(executor.map(lambda builder: builder.estimate_tokens("abc"), builders))

    assert counts == [3] * len(builders)
    assert loaded == [name]


def test_token_limited_row_splitting_counts_each_row_once() -> None:
    calls: list[str] = []

    def count(text: str) -> int:
        calls.append(text)
        return len(text)

    rows = [[f"row-{index}", "value"] for index in range(20)]
    table = _table(rows=rows)
    builder = StructuredEvidenceBuilder(token_counter=count)
    one_row_limit = len(builder._table_text(table, [0]))

    groups = builder._split_group_by_token_limit(
        table,
        list(range(len(rows))),
        one_row_limit,
    )

    assert groups == [[index] for index in range(len(rows))]
    assert len(calls) <= len(rows) + 2


def test_child_provenance_uses_only_selected_row_and_rowspan_cell_pages() -> None:
    table = _table(rows=[["A", "Base", "1"], ["B", "Proposed", "2"]])
    table.source_spans = [
        SourceSpan(page_index=0),
        SourceSpan(page_index=1),
        SourceSpan(page_index=2),
    ]
    for cell in table.cells:
        cell.source_spans = []
        if cell.row_index == 1:
            cell.source_spans = [SourceSpan(page_index=0)]
        elif cell.row_index == 2:
            cell.metadata["original_page_index"] = 1
    spanning = next(cell for cell in table.cells if cell.row_index == 1)
    spanning.rowspan = 2

    spans = StructuredEvidenceBuilder._row_source_spans(table, [1])

    assert {span.page_index for span in spans} == {0, 1}


def test_child_provenance_falls_back_to_table_pages_only_without_cell_provenance() -> None:
    table = _table(rows=[["A", "Base", "1"]])
    table.source_spans = [SourceSpan(page_index=3)]
    for cell in table.cells:
        cell.source_spans = []
        cell.metadata.pop("original_page_index", None)

    spans = StructuredEvidenceBuilder._row_source_spans(table, [0])

    assert {span.page_index for span in spans} == {3}


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


def test_continuation_graph_accepts_unordered_input_and_records_aliases() -> None:
    root = _table(table_id="root", rows=[["News", "Base", "81.2%"]], page_index=0)
    continuation = _table(
        table_id="continued",
        rows=[["News", "Proposed", "84.9%"]],
        page_index=1,
        metadata={"continuation_of": "root"},
    )
    builder = StructuredEvidenceBuilder(token_counter=lambda _text: 0)

    merged = builder.merge_cross_page_tables([continuation, root])

    assert [table.table_id for table in merged] == ["root"]
    assert builder.table_aliases == {"continued": "root"}


@pytest.mark.parametrize(
    "tables,error",
    [
        ([_table(table_id="dup", page_index=0), _table(table_id="dup", page_index=1)], "duplicate"),
        ([_table(table_id="orphan", page_index=1, metadata={"continuation_of": "missing"})], "unknown"),
        ([
            _table(table_id="a", page_index=0, metadata={"continuation_of": "b"}),
            _table(table_id="b", page_index=1, metadata={"continuation_of": "a"}),
        ], "cycle"),
        ([
            _table(table_id="root", page_index=0),
            _table(table_id="child-1", page_index=1, metadata={"continuation_of": "root"}),
            _table(table_id="child-2", page_index=1, metadata={"continuation_of": "root"}),
        ], "branch"),
        ([
            _table(table_id="root", page_index=1),
            _table(table_id="child", page_index=0, metadata={"continuation_of": "root"}),
        ], "page order"),
        ([
            _table(table_id="root", page_index=0),
            _table(table_id="child", page_index=2, metadata={"continuation_of": "root"}),
        ], "adjacent"),
        ([
            _table(table_id="root", page_index=0),
            _table(table_id="child", page_index=1, metadata={"continuation_of": "root"}),
        ], "source page"),
    ],
)
def test_continuation_graph_rejects_invalid_topologies(tables, error: str) -> None:
    if error == "source page":
        tables[1].source_spans = []

    with pytest.raises(ValueError, match=error):
        StructuredEvidenceBuilder(token_counter=lambda _text: 0).merge_cross_page_tables(tables)


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


def test_all_td_cross_page_html_uses_the_same_promoted_header_grid_everywhere() -> None:
    first = _table(
        table_id="all-td-page-1",
        rows=[["News", "Base", "81.2%"]],
        page_index=0,
    )
    continuation = _table(
        table_id="all-td-page-2",
        rows=[["News", "Proposed", "84.9%"]],
        page_index=1,
        metadata={"continuation_of": first.table_id},
    )
    first.source_html = (
        "<table><tr><td>Dataset</td><td>Method</td><td>Accuracy</td></tr>"
        "<tr><td>News</td><td>Base</td><td>81.2%</td></tr></table>"
    )
    continuation.source_html = (
        "<table><tr><td>Dataset</td><td>Method</td><td>Accuracy</td></tr>"
        "<tr><td>News</td><td>Proposed</td><td>84.9%</td></tr></table>"
    )

    merged = StructuredEvidenceBuilder(
        token_counter=lambda _text: 0
    ).merge_cross_page_tables([first, continuation])[0]
    validation = TableValidator().validate(merged)
    document = CanonicalDocument(
        document_id="all-td-cross-page",
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

    assert validation.accepted is True
    assert "cross_page_source_html_mismatch" not in validation.reasons
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
        blocks=[
            CanonicalBlock(
                block_id="first-table-block",
                block_type="table",
                text=first.normalized_markdown or "table",
                reading_order=0,
                parser_source="mineru",
                table_id=first.table_id,
                source_spans=first.source_spans,
            ),
            CanonicalBlock(
                block_id="continuation-table-block",
                block_type="table",
                text=second.normalized_markdown or "table",
                reading_order=1,
                parser_source="mineru",
                table_id=second.table_id,
                source_spans=second.source_spans,
                metadata={"parent_table_id": second.table_id},
            ),
        ],
        tables=[second, first],
    )
    monkeypatch.setattr(
        MarkdownCanonicalAdapter,
        "parse",
        lambda self, path: document,
    )

    result = parse_canonical_document(source)

    assert len(result.tables) == 1
    assert result.tables[0].status == "cross_page_merged"
    assert result.metadata["table_aliases"] == {second.table_id: first.table_id}
    assert {block.table_id for block in result.blocks} == {first.table_id}
    assert result.blocks[1].metadata["parent_table_id"] == first.table_id
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


def test_figure_child_windowing_does_not_repeat_overlong_nearby_narrative() -> None:
    nearby_text = " ".join(f"nearby-{index}" for index in range(100))
    nearby = CanonicalBlock(
        block_id="long-nearby",
        block_type="narrative",
        text=nearby_text,
        reading_order=0,
        parser_source="mineru",
        source_spans=[SourceSpan(page_index=2, source_block_id="nearby-source")],
    )
    figure = CanonicalFigure(
        figure_id="figure-short-description",
        caption="Figure 2. Accuracy curve",
        description="The curve rises steadily.",
        asset_path="assets/figure-2.png",
        nearby_block_ids=[nearby.block_id],
        source_spans=[SourceSpan(page_index=2, source_block_id="figure-source")],
    )
    builder = StructuredEvidenceBuilder(token_counter=lambda text: len(text.split()))

    parent, children = builder.figure_chunks(figure, [nearby], max_tokens=20)

    assert nearby_text in parent.text
    assert children
    assert all(nearby_text not in child.text for child in children)
    assert all(child.token_count <= 20 for child in children)
    assert "".join(child.metadata["source_fragment"] for child in children) == figure.description


def test_figure_child_windowing_does_not_duplicate_long_caption_in_image_alt() -> None:
    caption = " ".join(f"caption-{index}" for index in range(12))
    description = " ".join(f"description-{index}" for index in range(40))
    figure = CanonicalFigure(
        figure_id="figure-long-caption",
        caption=caption,
        description=description,
        asset_path="assets/figure-with-a-long-content-hash.png",
        source_spans=[SourceSpan(page_index=3, source_block_id="figure-source")],
    )
    builder = StructuredEvidenceBuilder(token_counter=lambda text: len(text.split()))

    parent, children = builder.figure_chunks(figure, [], max_tokens=20)

    assert caption in parent.text
    assert children
    assert all(child.token_count <= 20 for child in children)
    assert "".join(child.metadata["source_fragment"] for child in children) == description
    assert all(caption in child.text for child in children)
    assert all(
        "![figure-long-caption](assets/figure-with-a-long-content-hash.png)"
        in child.text
        for child in children
    )
    assert all(f"![{caption}]" not in child.text for child in children)


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
