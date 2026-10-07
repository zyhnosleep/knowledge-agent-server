from __future__ import annotations

import pytest

from app.services.canonical_models import CanonicalCell, CanonicalDocument, CanonicalTable, SourceSpan
from app.services.canonical_quality import CanonicalQualityGate
from app.services.table_normalization import normalize_fragmented_numeric_spacing, normalize_table_cell


@pytest.mark.parametrize("normalizer", [normalize_table_cell, normalize_fragmented_numeric_spacing])
@pytest.mark.parametrize("value", ["29.4 33.4", "34.2 35.1"])
def test_complete_decimal_scores_are_not_glued(normalizer, value):
    assert normalizer(value) == value


@pytest.mark.parametrize("value", ["29.4 33.4", "29.4.33.4", "34.2 35.1"])
def test_merged_scalar_metric_scores_require_source_page_repair(value):
    table = CanonicalTable(
        table_id="ocr-table", headers=["Method", "HotpotQA (EM)"],
        rows=[["Two merged methods", value]],
        cells=[
            CanonicalCell(text="Method", row_index=0, column_index=0, is_header=True),
            CanonicalCell(text="HotpotQA (EM)", row_index=0, column_index=1, is_header=True),
            CanonicalCell(text="Two merged methods", row_index=1, column_index=0),
            CanonicalCell(text=value, row_index=1, column_index=1),
        ],
        source_spans=[SourceSpan(page_index=4, page_label="5")],
    )
    document = CanonicalDocument(document_id="doc", parse_version="test", parser_source="mineru", tables=[table])
    report = CanonicalQualityGate().evaluate(document)
    issue = next((i for i in report.issues if i.code == "table_invalid"), None)
    assert issue is not None
    assert "scalar_metric_values_merged" in issue.metadata["reasons"]
    assert issue.repairable and issue.repair_scope == "page:5"
    assert not report.accepted


@pytest.mark.parametrize("value", ["27.4", "27.4 ± 0.3", "[27.4, 33.4]"])
def test_scalar_uncertainty_and_explicit_intervals_are_not_merged_scores(value):
    table = CanonicalTable(table_id="t", headers=["Method", "Fever (Acc)"], rows=[["ReAct", value]])
    assert "scalar_metric_values_merged" not in CanonicalQualityGate._invalid_table_reasons(table)


def test_version_identifiers_are_not_metric_scores():
    table = CanonicalTable(table_id="t", headers=["Software", "Version"], rows=[["tool", "1.2.3.4"]])
    assert "scalar_metric_values_merged" not in CanonicalQualityGate._invalid_table_reasons(table)
