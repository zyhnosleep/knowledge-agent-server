from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.services.canonical_models import (
    CanonicalAsset,
    CanonicalBlock,
    CanonicalCell,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalQualityIssue,
    CanonicalQualityReport,
    CanonicalTable,
    SectionNode,
    SourceSpan,
)


def test_canonical_block_preserves_source_spans() -> None:
    span = SourceSpan(
        page_index=7,
        page_label="8",
        bbox=[10.0, 20.0, 200.0, 120.0],
        source_block_id="b-8",
    )
    block = CanonicalBlock(
        block_id="block-8",
        block_type="table",
        text="| Model | F1 |\n|---|---|\n| SAC-KG | 74.7 |",
        section_path=["Experiments", "Main Results"],
        reading_order=12,
        source_spans=[span],
        parser_source="mineru",
    )

    restored = CanonicalBlock.model_validate_json(block.model_dump_json())

    assert restored.source_spans[0].page_index == 7
    assert restored.block_type == "table"


def test_reference_blocks_are_not_retrievable_by_default() -> None:
    block = CanonicalBlock(
        block_id="r1",
        block_type="reference",
        text="[1] Paper",
        reading_order=1,
        parser_source="mineru",
    )

    assert block.retrievable is False


@pytest.mark.parametrize("block_type", ["heading", "reference"])
def test_non_content_blocks_default_to_not_retrievable(block_type: str) -> None:
    block = CanonicalBlock(
        block_id="b1",
        block_type=block_type,
        text="Source text",
        reading_order=1,
        parser_source="mineru",
    )

    assert block.retrievable is False


def test_narrative_blocks_are_retrievable_by_default() -> None:
    block = CanonicalBlock(
        block_id="n1",
        block_type="narrative",
        text="Source text",
        reading_order=1,
        parser_source="mineru",
    )

    assert block.retrievable is True


@pytest.mark.parametrize(
    ("block_type", "retrievable"),
    [("heading", True), ("narrative", False)],
)
def test_explicit_retrievable_value_is_preserved(
    block_type: str,
    retrievable: bool,
) -> None:
    block = CanonicalBlock(
        block_id="b1",
        block_type=block_type,
        text="Source text",
        reading_order=1,
        parser_source="mineru",
        retrievable=retrievable,
    )

    assert block.retrievable is retrievable


def test_heading_text_does_not_change_block_type() -> None:
    block = CanonicalBlock(
        block_id="n1",
        block_type="narrative",
        text="# Looks like a heading",
        reading_order=1,
        parser_source="markdown",
    )

    assert block.block_type == "narrative"
    assert block.retrievable is True


def test_invalid_block_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CanonicalBlock(
            block_id="b1",
            block_type="unknown",
            text="Source text",
            reading_order=1,
            parser_source="mineru",
        )


@pytest.mark.parametrize(
    "bbox",
    [
        [0.0, 1.0, 2.0],
        [10.0, 0.0, 9.0, 10.0],
        [0.0, 10.0, 10.0, 9.0],
    ],
)
def test_invalid_source_span_bbox_is_rejected(bbox: list[float]) -> None:
    with pytest.raises(ValidationError):
        SourceSpan(bbox=bbox)


def test_source_span_supports_all_adapter_coordinates() -> None:
    span = SourceSpan(
        page_index=0,
        page_label="i",
        bbox=[1, 2, 3, 4],
        normalized_bbox=[0.1, 0.2, 0.3, 0.4],
        source_block_id="pdf-1",
        paragraph_id="p-1",
        table_id="table-1",
        row_index=2,
        column_index=3,
        image_relationship_id="rId7",
        xpath="/html/body/main/p[1]",
        css_selector="#intro",
        element_id="intro",
        heading_path=["Introduction"],
        line_start=10,
        line_end=12,
        char_start=100,
        char_end=150,
        metadata={"adapter": "fixture"},
    )

    restored = SourceSpan.model_validate_json(span.model_dump_json())

    assert restored.normalized_bbox == (0.1, 0.2, 0.3, 0.4)
    assert restored.image_relationship_id == "rId7"
    assert restored.heading_path == ["Introduction"]
    assert restored.char_end == 150


def test_all_canonical_models_round_trip_through_json() -> None:
    span = SourceSpan(page_index=1, bbox=[0, 0, 100, 50])
    cell = CanonicalCell(
        text="74.7",
        row_index=1,
        column_index=1,
        rowspan=1,
        colspan=1,
        bbox=[20, 20, 40, 30],
        source_spans=[span],
    )
    table = CanonicalTable(
        table_id="table-1",
        caption="Main results",
        headers=["Model", "F1"],
        rows=[["SAC-KG", "74.7"]],
        cells=[cell],
        source_markdown="| Model | F1 |",
        normalized_markdown="| Model | F1 |",
        footnotes=["Higher is better."],
        source_spans=[span],
        status="parsed",
    )
    figure = CanonicalFigure(
        figure_id="figure-1",
        caption="System overview",
        asset_path="assets/figure-1.png",
        source_spans=[span],
        nearby_block_ids=["b1"],
        generated_summary="Generated and non-canonical",
        analysis_model="vision-model",
        warnings=["low resolution"],
    )
    formula = CanonicalFormula(
        formula_id="formula-1",
        latex="F_1 = 2PR/(P+R)",
        caption="F1 score",
        description="Source description",
        source_spans=[span],
        nearby_block_ids=["b1"],
        generated_explanation="Generated and non-canonical",
        analysis_model="language-model",
        warnings=["OCR uncertain"],
    )
    asset = CanonicalAsset(
        asset_id="asset-1",
        path="assets/figure-1.png",
        media_type="image/png",
        sha256="a" * 64,
        source_path="/tmp/source.png",
        source_spans=[span],
    )
    issue = CanonicalQualityIssue(
        code="ocr-low-confidence",
        severity="warning",
        message="OCR confidence below threshold",
        block_ids=["b1"],
        repairable=True,
    )
    quality = CanonicalQualityReport(
        accepted=True,
        status="accepted_with_warnings",
        score=0.91,
        issues=[issue],
        warnings=["review formula"],
        fallback_pages=[3],
    )
    document = CanonicalDocument(
        document_id="doc-1",
        source_path="paper.pdf",
        source_media_type="application/pdf",
        parser_source="mineru",
        parse_version="canonical-v1-abcd",
        title="A Paper",
        abstract="Original abstract.",
        keywords=["knowledge graphs"],
        outline=[
            SectionNode(
                title="Introduction",
                level=1,
                block_id="b1",
                children=[SectionNode(title="Background", level=2)],
            )
        ],
        blocks=[
            CanonicalBlock(
                block_id="b1",
                block_type="narrative",
                text="Original source text.",
                reading_order=1,
                source_spans=[span],
                parser_source="mineru",
                parser_confidence=0.98,
                figure_id="figure-1",
                formula_id="formula-1",
                table_id="table-1",
            )
        ],
        tables=[table],
        figures=[figure],
        formulas=[formula],
        assets=[asset],
        quality=quality,
        warnings=["fixture warning"],
        metadata={"language": "en"},
        source_metadata={"filename": "paper.pdf"},
        parser_metadata={"backend_version": "1.0"},
        status="ready",
    )

    restored = CanonicalDocument.model_validate_json(document.model_dump_json())

    assert restored == document
    assert restored.outline[0].children[0].title == "Background"
    assert restored.tables[0].cells[0].text == "74.7"
    assert restored.quality.issues[0].repairable is True


def test_document_core_fields_have_independent_defaults() -> None:
    first = CanonicalDocument()
    second = CanonicalDocument()

    first.blocks.append(
        CanonicalBlock(
            block_id="b1",
            block_type="appendix",
            text="Appendix",
            reading_order=1,
            parser_source="fixture",
        )
    )
    first.metadata["language"] = "en"

    assert first.title == ""
    assert first.abstract is None
    assert first.keywords == []
    assert first.outline == []
    assert second.blocks == []
    assert second.metadata == {}


def test_document_preserves_explicit_missing_abstract() -> None:
    document = CanonicalDocument(abstract=None)

    restored = CanonicalDocument.model_validate_json(document.model_dump_json())

    assert restored.abstract is None


def test_nested_section_outline_round_trips_through_json() -> None:
    document = CanonicalDocument(
        outline=[
            SectionNode(
                title="Experiments",
                level=1,
                block_id="heading-experiments",
                metadata={"number": "4"},
                children=[
                    SectionNode(
                        title="Main Results",
                        level=2,
                        block_id="heading-results",
                    )
                ],
            )
        ]
    )

    restored = CanonicalDocument.model_validate_json(document.model_dump_json())

    assert restored.outline == document.outline
    assert restored.outline[0].children[0].level == 2


def test_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SourceSpan(unknown_coordinate="x")


def test_figure_analysis_fields_round_trip_and_defaults_are_isolated() -> None:
    figure = CanonicalFigure(
        figure_id="figure-analysis",
        analysis_status="complete",
        ai_figure_type="line_chart",
        ai_axes={"x": "epoch", "y": "accuracy"},
        ai_legend=["baseline", "proposed"],
        ai_trends=["increasing"],
        ai_observations=["Proposed remains higher."],
        ai_confidence=0.8,
    )

    restored = CanonicalFigure.model_validate_json(figure.model_dump_json())
    assert restored == figure

    first = CanonicalFigure(figure_id="first")
    second = CanonicalFigure(figure_id="second")
    first.ai_legend.append("only-first")
    assert second.ai_legend == []


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_figure_ai_confidence_is_bounded(confidence: float) -> None:
    with pytest.raises(ValidationError):
        CanonicalFigure(figure_id="figure", ai_confidence=confidence)


def test_formula_analysis_fields_round_trip_with_structured_explanations() -> None:
    formula = CanonicalFormula(
        formula_id="formula-analysis",
        latex="E=mc^2",
        analysis_status="complete",
        ai_variable_explanations=[
            {"variable": "E", "meaning": "energy"},
            {"variable": "m", "meaning": "mass"},
        ],
        ai_method_role="objective_function",
        ai_confidence=0.65,
    )

    restored = CanonicalFormula.model_validate_json(formula.model_dump_json())
    assert restored == formula


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_formula_ai_confidence_is_bounded(confidence: float) -> None:
    with pytest.raises(ValidationError):
        CanonicalFormula(
            formula_id="formula",
            latex="x",
            ai_confidence=confidence,
        )


def test_formula_variable_explanations_reject_non_json_values() -> None:
    with pytest.raises(ValidationError):
        CanonicalFormula(
            formula_id="formula",
            latex="x",
            ai_variable_explanations={"x": {1, 2}},
        )
