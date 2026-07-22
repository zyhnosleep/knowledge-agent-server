from __future__ import annotations

import re
from pathlib import PurePosixPath

from app.services.canonical_abstract import has_explicit_abstract
from app.services.canonical_table_identity import table_identity_fingerprint
from app.services.canonical_models import (
    CanonicalCell,
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
class CanonicalQualityGate:
    """Apply deterministic, source-only validation to a canonical document."""

    def evaluate(self, document: CanonicalDocument) -> CanonicalQualityReport:
        issues: list[CanonicalQualityIssue] = []
        issues.extend(self._page_issues(document))
        issues.extend(self._content_issues(document))
        issues.extend(self._reading_order_issues(document))
        issues.extend(self._asset_issues(document))
        issues.extend(self._abstract_issues(document))
        issues.extend(self._table_inventory_issues(document))
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
            isinstance(text, str) and has_explicit_abstract(text)
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
            structured_reasons = table.metadata.get("structured_validation_reasons")
            if isinstance(structured_reasons, list):
                reasons.extend(
                    reason
                    for reason in structured_reasons
                    if isinstance(reason, str) and reason not in reasons
                )
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
                    metadata={
                        "table_id": table.table_id,
                        "reasons": reasons,
                        "locator": cls._table_locator(table),
                    },
                )
            )
        return issues

    @staticmethod
    def _table_locator(table: CanonicalTable) -> dict[str, object]:
        span = next(iter(table.source_spans), None)
        if span is None:
            return {}
        box = span.normalized_bbox or span.bbox
        return {
            key: value
            for key, value in {
                "page_index": span.page_index,
                "bbox": list(box) if box is not None else None,
                "source_block_id": span.source_block_id,
            }.items()
            if value is not None
        }

    @staticmethod
    def _table_inventory_issues(
        document: CanonicalDocument,
    ) -> list[CanonicalQualityIssue]:
        by_id: dict[str, list[CanonicalTable]] = {}
        for table in document.tables:
            by_id.setdefault(table.table_id, []).append(table)

        issues: list[CanonicalQualityIssue] = []
        duplicate_ids = sorted(
            table_id for table_id, items in by_id.items() if len(items) > 1
        )
        if duplicate_ids:
            issues.append(
                CanonicalQualityIssue(
                    code="table_id_duplicate",
                    severity="fatal",
                    message="Canonical table IDs must be unique.",
                    metadata={"table_ids": duplicate_ids},
                )
            )

        by_fingerprint: dict[str, list[CanonicalTable]] = {}
        for table in document.tables:
            by_fingerprint.setdefault(table_identity_fingerprint(table), []).append(table)
        duplicate_content = [
            {
                "fingerprint": fingerprint,
                "table_ids": sorted(table.table_id for table in tables),
            }
            for fingerprint, tables in sorted(by_fingerprint.items())
            if len(tables) > 1
        ]
        if duplicate_content:
            issues.append(
                CanonicalQualityIssue(
                    code="table_content_duplicate",
                    severity="fatal",
                    message=(
                        "Canonical tables contain indistinguishable content and source locators."
                    ),
                    metadata={"duplicates": duplicate_content},
                )
            )

        table_pages = {
            table_id: {
                span.page_index
                for table in items
                for span in table.source_spans
                if span.page_index is not None
            }
            for table_id, items in by_id.items()
        }
        valid_ids = set(by_id)
        for block in document.blocks:
            if not block.table_id:
                continue
            if block.table_id not in valid_ids:
                issues.append(
                    CanonicalQualityIssue(
                        code="table_reference_invalid",
                        severity="fatal",
                        message=f"Block {block.block_id} references an unknown table.",
                        block_ids=[block.block_id],
                        metadata={"table_id": block.table_id},
                    )
                )
                continue
            block_pages = {
                span.page_index
                for span in block.source_spans
                if span.page_index is not None
            }
            known_pages = table_pages[block.table_id]
            if block_pages and known_pages and block_pages.isdisjoint(known_pages):
                issues.append(
                    CanonicalQualityIssue(
                        code="table_reference_conflict",
                        severity="fatal",
                        message=f"Block {block.block_id} is associated with a table on another page.",
                        block_ids=[block.block_id],
                        metadata={
                            "table_id": block.table_id,
                            "block_pages": sorted(block_pages),
                            "table_pages": sorted(known_pages),
                        },
                    )
                )
        return issues

    @staticmethod
    def _invalid_table_reasons(table: CanonicalTable) -> list[str]:
        reasons: list[str] = []

        def add(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        width = len(table.headers)
        if width == 0:
            add("header_missing")
        if not table.rows:
            add("data_rows_missing")
        if width and any(len(row) != width for row in table.rows):
            add("row_width_mismatch")

        grid = [table.headers, *table.rows]
        height = len(grid)
        if not table.cells:
            add("cells_missing")
        elif width:
            coordinates = [(cell.row_index, cell.column_index) for cell in table.cells]
            if len(coordinates) != len(set(coordinates)):
                add("duplicate_cell_coordinate")
            occupied: dict[tuple[int, int], object] = {}
            for cell in table.cells:
                row_end = cell.row_index + cell.rowspan
                column_end = cell.column_index + cell.colspan
                if (
                    cell.row_index >= height
                    or cell.column_index >= width
                    or row_end > height
                    or column_end > width
                ):
                    add("cell_out_of_bounds")
                    continue
                source_row = grid[cell.row_index]
                if (
                    cell.column_index >= len(source_row)
                    or source_row[cell.column_index] != cell.text
                ):
                    add("cell_value_mismatch")
                for row_index in range(cell.row_index, row_end):
                    for column_index in range(cell.column_index, column_end):
                        coordinate = (row_index, column_index)
                        if coordinate in occupied:
                            add("cell_overlap")
                        else:
                            occupied[coordinate] = cell
                        if coordinate != (cell.row_index, cell.column_index):
                            covered_row = grid[row_index]
                            covered_value = (
                                covered_row[column_index]
                                if column_index < len(covered_row)
                                else ""
                            )
                            if covered_value not in {"", cell.text}:
                                add("cell_span_value_mismatch")
            expected_coordinates = {
                (row_index, column_index)
                for row_index in range(height)
                for column_index in range(width)
            }
            if not expected_coordinates.issubset(occupied):
                add("cells_incomplete")

        expected_markdown = CanonicalQualityGate._table_markdown(
            table.headers,
            table.rows,
        )
        if table.normalized_markdown is not None and (
            table.normalized_markdown.strip() != expected_markdown
        ):
            add("normalized_markdown_mismatch")
        source_markdowns = table.metadata.get("source_markdowns")
        if table.status == "cross_page_merged" and isinstance(
            source_markdowns, list
        ):
            if not CanonicalQualityGate._source_markdown_segments_match(
                table, source_markdowns
            ):
                add("cross_page_source_markdown_mismatch")
        elif table.source_markdown is not None:
            source_data = CanonicalQualityGate._markdown_table_data(
                table.source_markdown
            )
            if source_data is None:
                add("source_markdown_invalid")
            elif source_data != (table.headers, table.rows):
                add("source_markdown_mismatch")

        source_htmls = table.metadata.get("source_htmls")
        if (
            table.status == "cross_page_merged"
            and isinstance(source_htmls, list)
            and source_htmls
        ):
            if not CanonicalQualityGate._source_html_segments_match(
                table, source_htmls
            ):
                add("cross_page_source_html_mismatch")
        elif table.source_html is not None:
            try:
                from app.services.canonical_artifacts import CanonicalArtifactStore

                html_cells = CanonicalArtifactStore._table_cells_from_html(
                    table.source_html
                )
                html_headers, html_rows = CanonicalArtifactStore._table_grid_from_cells(
                    html_cells
                )
            except (TypeError, ValueError):
                add("source_html_invalid")
            else:
                if (html_headers, html_rows) != (table.headers, table.rows):
                    add("source_html_mismatch")
                html_signatures = sorted(
                    CanonicalQualityGate._cell_signature(cell) for cell in html_cells
                )
                cell_signatures = sorted(
                    CanonicalQualityGate._cell_signature(cell) for cell in table.cells
                )
                if html_signatures != cell_signatures:
                    add("source_html_cell_mismatch")
        return reasons

    @classmethod
    def _source_markdown_segments_match(
        cls, table: CanonicalTable, markdowns: list[object]
    ) -> bool:
        grids: list[tuple[list[str], list[list[str]]]] = []
        for markdown in markdowns:
            if not isinstance(markdown, str):
                return False
            parsed = cls._markdown_table_data(markdown)
            if parsed is None:
                return False
            grids.append(parsed)
        return cls._combined_segment_rows(table.headers, grids) == table.rows

    @classmethod
    def _source_html_segments_match(
        cls, table: CanonicalTable, htmls: list[object]
    ) -> bool:
        from app.services.canonical_artifacts import CanonicalArtifactStore

        cell_segments = table.metadata.get("source_cell_segments")
        if not isinstance(cell_segments, list) or len(cell_segments) != len(htmls):
            return False
        grids: list[tuple[list[str], list[list[str]]]] = []
        for index, source_html in enumerate(htmls):
            if not isinstance(source_html, str):
                return False
            try:
                cells = CanonicalArtifactStore._table_cells_from_html(source_html)
                grids.append(CanonicalArtifactStore._table_grid_from_cells(cells))
                expected_cells = [
                    CanonicalCell.model_validate(item) for item in cell_segments[index]
                ]
            except (TypeError, ValueError):
                return False
            if sorted(cls._cell_signature(cell) for cell in cells) != sorted(
                cls._cell_signature(cell) for cell in expected_cells
            ):
                return False
        return cls._combined_segment_rows(table.headers, grids) == table.rows

    @staticmethod
    def _combined_segment_rows(
        headers: list[str],
        grids: list[tuple[list[str], list[list[str]]]],
    ) -> list[list[str]] | None:
        combined: list[list[str]] = []
        for segment_headers, segment_rows in grids:
            if segment_headers != headers:
                return None
            rows = list(segment_rows)
            if rows and rows[0] == headers:
                rows = rows[1:]
            combined.extend(rows)
        return combined

    @staticmethod
    def _cell_signature(cell: CanonicalCell) -> tuple[object, ...]:
        return (
            cell.row_index,
            cell.column_index,
            cell.rowspan,
            cell.colspan,
            cell.is_header,
            cell.text,
        )

    @staticmethod
    def _table_markdown(headers: list[str], rows: list[list[str]]) -> str:
        def escape(value: str) -> str:
            return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")

        if not headers:
            return ""
        lines = [
            "| " + " | ".join(escape(cell) for cell in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend(
            "| "
            + " | ".join(
                escape(cell) for cell in (row + [""] * len(headers))[: len(headers)]
            )
            + " |"
            for row in rows
        )
        return "\n".join(lines)

    @staticmethod
    def _markdown_table_data(
        markdown: str,
    ) -> tuple[list[str], list[list[str]]] | None:
        def split_row(value: str) -> list[str]:
            def pipe_is_escaped(index: int) -> bool:
                backslashes = 0
                cursor = index - 1
                while cursor >= 0 and value[cursor] == "\\":
                    backslashes += 1
                    cursor -= 1
                return backslashes % 2 == 1

            def unescape(cell: str) -> str:
                decoded: list[str] = []
                index = 0
                while index < len(cell):
                    if (
                        cell[index] == "\\"
                        and index + 1 < len(cell)
                        and cell[index + 1] in {"\\", "|"}
                    ):
                        decoded.append(cell[index + 1])
                        index += 2
                        continue
                    decoded.append(cell[index])
                    index += 1
                return "".join(decoded)

            value = value.strip()
            if value.startswith("|"):
                value = value[1:]
            if value.endswith("|") and not pipe_is_escaped(len(value) - 1):
                value = value[:-1]
            parts: list[str] = []
            start = 0
            for index, character in enumerate(value):
                if character == "|" and not pipe_is_escaped(index):
                    parts.append(value[start:index])
                    start = index + 1
            parts.append(value[start:])
            return [unescape(part.strip()) for part in parts]

        lines = [line.strip() for line in markdown.splitlines() if line.strip()]
        for index in range(len(lines) - 1):
            separator = split_row(lines[index + 1])
            if not separator or not all(
                re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
                for cell in separator
            ):
                continue
            headers = split_row(lines[index])
            rows: list[list[str]] = []
            for line in lines[index + 2 :]:
                if "|" not in line:
                    break
                rows.append(split_row(line))
            return headers, rows
        return None

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
