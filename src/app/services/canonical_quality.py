from __future__ import annotations

import re
from pathlib import PurePosixPath

from app.services.canonical_models import (
    CanonicalDocument,
    CanonicalQualityIssue,
    CanonicalQualityReport,
    CanonicalTable,
)


_SEVERITY_PENALTIES = {
    "info": 0.0,
    "warning": 0.05,
    "error": 0.15,
    "fatal": 0.35,
}
_ABSTRACT_HEADING = re.compile(r"(?im)^\s*(?:abstract|摘要)\s*(?::|$)")


class CanonicalQualityGate:
    """Apply deterministic, source-only validation to a canonical document."""

    def evaluate(self, document: CanonicalDocument) -> CanonicalQualityReport:
        issues: list[CanonicalQualityIssue] = []
        issues.extend(self._page_issues(document))
        issues.extend(self._content_issues(document))
        issues.extend(self._reading_order_issues(document))
        issues.extend(self._asset_issues(document))
        issues.extend(self._abstract_issues(document))
        issues.extend(self._table_issues(document))
        issues.extend(self._figure_issues(document))
        issues.extend(self._formula_issues(document))

        has_fatal = any(issue.severity == "fatal" for issue in issues)
        has_repairable = any(issue.repairable for issue in issues)
        has_warning = any(issue.severity == "warning" for issue in issues)
        if has_fatal:
            accepted = False
            status = "rejected"
        elif has_repairable:
            accepted = False
            status = "validation_failed"
        elif has_warning:
            accepted = True
            status = "accepted_with_warnings"
        else:
            accepted = True
            status = "accepted"

        fallback_pages = sorted(
            {
                page
                for issue in issues
                if issue.repairable and issue.repair_scope
                for page in self._scope_pages(issue.repair_scope)
            }
        )
        score = max(
            0.0,
            1.0 - sum(_SEVERITY_PENALTIES[issue.severity] for issue in issues),
        )
        report = CanonicalQualityReport(
            accepted=accepted,
            status=status,
            score=round(score, 6),
            issues=issues,
            warnings=[issue.message for issue in issues if issue.severity == "warning"],
            fallback_pages=fallback_pages,
            metadata={
                "gate": "canonical-deterministic-v1",
                "issue_count": len(issues),
            },
        )
        document.quality = report
        return document.quality

    @staticmethod
    def _page_issues(document: CanonicalDocument) -> list[CanonicalQualityIssue]:
        expected = document.metadata.get("expected_page_count")
        explicit_pages = document.metadata.get("parsed_page_indices")
        if not isinstance(expected, int) or expected <= 0 or not isinstance(explicit_pages, list):
            return []
        parsed = {
            value
            for value in explicit_pages
            if isinstance(value, int) and 0 <= value < expected
        }
        missing = [index + 1 for index in range(expected) if index not in parsed]
        if not missing:
            return []
        return [
            CanonicalQualityIssue(
                code="page_missing",
                severity="fatal",
                message=f"Canonical parse is missing source pages: {missing}.",
                metadata={"missing_pages": missing, "expected_page_count": expected},
            )
        ]

    @staticmethod
    def _content_issues(document: CanonicalDocument) -> list[CanonicalQualityIssue]:
        if any(block.retrievable and block.text.strip() for block in document.blocks):
            return []
        return [
            CanonicalQualityIssue(
                code="content_empty",
                severity="fatal",
                message="Canonical parse contains no usable source content.",
            )
        ]

    @staticmethod
    def _reading_order_issues(document: CanonicalDocument) -> list[CanonicalQualityIssue]:
        actual = [block.reading_order for block in document.blocks]
        expected = list(range(len(document.blocks)))
        if actual == expected:
            return []
        return [
            CanonicalQualityIssue(
                code="reading_order_invalid",
                severity="fatal",
                message="Canonical block reading order must be unique and contiguous.",
                block_ids=[block.block_id for block in document.blocks],
                metadata={"actual": actual, "expected": expected},
            )
        ]

    @classmethod
    def _asset_issues(cls, document: CanonicalDocument) -> list[CanonicalQualityIssue]:
        invalid: list[str] = []
        for asset in document.assets:
            if not cls._valid_asset_path(asset.path):
                invalid.append(asset.asset_id)
        valid_paths = {asset.path for asset in document.assets if cls._valid_asset_path(asset.path)}
        for figure in document.figures:
            if figure.asset_path and (
                not cls._valid_asset_path(figure.asset_path)
                or figure.asset_path not in valid_paths
            ):
                invalid.append(figure.figure_id)
        if not invalid:
            return []
        return [
            CanonicalQualityIssue(
                code="asset_invalid",
                severity="fatal",
                message="Canonical assets contain an unsafe or unresolved reference.",
                metadata={"invalid_asset_references": sorted(set(invalid))},
            )
        ]

    @staticmethod
    def _valid_asset_path(value: str) -> bool:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        return bool(
            normalized
            and not path.is_absolute()
            and path.parts
            and path.parts[0] == "assets"
            and ".." not in path.parts
        )

    @staticmethod
    def _abstract_issues(document: CanonicalDocument) -> list[CanonicalQualityIssue]:
        if document.abstract and document.abstract.strip():
            return []
        page_texts = document.metadata.get("text_layer_pages")
        if not isinstance(page_texts, list):
            return []
        source_has_abstract = any(
            isinstance(text, str) and _ABSTRACT_HEADING.search(text)
            for text in page_texts[:2]
        )
        if not source_has_abstract:
            return []
        return [
            CanonicalQualityIssue(
                code="abstract_missing",
                severity="error",
                message="The source has an Abstract on pages 1-2 but canonical Abstract is missing.",
                repairable=True,
                repair_scope="pages:1-2",
            )
        ]

    @classmethod
    def _table_issues(cls, document: CanonicalDocument) -> list[CanonicalQualityIssue]:
        issues: list[CanonicalQualityIssue] = []
        for table in document.tables:
            reasons = cls._invalid_table_reasons(table)
            if not reasons:
                continue
            page = next(
                (
                    span.page_index + 1
                    for span in table.source_spans
                    if span.page_index is not None
                ),
                None,
            )
            issues.append(
                CanonicalQualityIssue(
                    code="table_invalid",
                    severity="error",
                    message=f"Table {table.table_id} failed deterministic validation.",
                    repairable=True,
                    repair_scope=f"page:{page}" if page is not None else "document",
                    metadata={"table_id": table.table_id, "reasons": reasons},
                )
            )
        return issues

    @staticmethod
    def _invalid_table_reasons(table: CanonicalTable) -> list[str]:
        reasons: list[str] = []
        width = len(table.headers)
        if width == 0:
            reasons.append("header_missing")
        if not table.rows:
            reasons.append("data_rows_missing")
        if width and any(len(row) != width for row in table.rows):
            reasons.append("row_width_mismatch")
        if table.cells:
            coordinates = [(cell.row_index, cell.column_index) for cell in table.cells]
            if len(coordinates) != len(set(coordinates)):
                reasons.append("duplicate_cell_coordinate")
            if width and any(cell.column_index >= width for cell in table.cells):
                reasons.append("cell_out_of_bounds")
        return reasons

    @staticmethod
    def _figure_issues(document: CanonicalDocument) -> list[CanonicalQualityIssue]:
        return [
            CanonicalQualityIssue(
                code="figure_caption_missing",
                severity="warning",
                message=f"Figure {figure.figure_id} has no source caption.",
                block_ids=[
                    block.block_id
                    for block in document.blocks
                    if block.figure_id == figure.figure_id
                ],
                metadata={"figure_id": figure.figure_id},
            )
            for figure in document.figures
            if not (figure.caption or "").strip()
        ]

    @staticmethod
    def _formula_issues(document: CanonicalDocument) -> list[CanonicalQualityIssue]:
        return [
            CanonicalQualityIssue(
                code="formula_analysis_missing",
                severity="warning",
                message=f"Formula {formula.formula_id} has no completed optional analysis.",
                block_ids=[
                    block.block_id
                    for block in document.blocks
                    if block.formula_id == formula.formula_id
                ],
                metadata={"formula_id": formula.formula_id},
            )
            for formula in document.formulas
            if formula.analysis_status != "complete"
        ]

    @staticmethod
    def _scope_pages(scope: str) -> list[int]:
        single = re.fullmatch(r"page:(\d+)", scope)
        if single:
            return [int(single.group(1))]
        page_range = re.fullmatch(r"pages:(\d+)-(\d+)", scope)
        if page_range:
            start, end = map(int, page_range.groups())
            return list(range(start, end + 1)) if end >= start else []
        return []
