from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.canonical_models import (
    CanonicalBlock,
    CanonicalCell,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalTable,
    SourceSpan,
    TableStatus,
)
from app.services.canonical_quality import CanonicalQualityGate
from app.services.canonical_table_identity import table_identity_fingerprint


class TableRepairRequest(BaseModel):
    """A deterministic, page-scoped request for vision table repair."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    reasons: list[str]
    locator: dict[str, Any]
    source_fingerprint: str
    instructions: str


class TableValidationResult(BaseModel):
    """Consumable validation signal used to gate canonical activation."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    activation_allowed: bool
    status: TableStatus
    reasons: list[str] = Field(default_factory=list)
    table: CanonicalTable
    repair_request: TableRepairRequest | None = None


class StructuredEvidenceChunk(BaseModel):
    """Stable source/derived text boundary for structured retrieval evidence."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    parent_chunk_id: str | None = None
    chunk_role: Literal["parent", "child"]
    block_type: Literal["table", "figure", "formula"]
    text: str
    embedding_text: str
    token_count: int = Field(ge=0)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableValidator:
    """Strict source-only table validation with explicit repair routing."""

    _caption_number = re.compile(r"\btable\s*([A-Za-z]?\d+(?:[.-]\d+)*)\b", re.I)

    def validate(
        self,
        table: CanonicalTable,
        repaired_table: CanonicalTable | None = None,
    ) -> TableValidationResult:
        candidate = (repaired_table or table).model_copy(deep=True)
        reasons = self._reasons(candidate)
        if repaired_table is not None and repaired_table.table_id != table.table_id:
            reasons.append("repair_identity_mismatch")
        if reasons:
            candidate.status = "validation_failed"
            return TableValidationResult(
                accepted=False,
                activation_allowed=False,
                status="validation_failed",
                reasons=reasons,
                table=candidate,
                repair_request=self._repair_request(candidate, reasons),
            )

        if repaired_table is not None:
            status: TableStatus = "repaired_by_vision"
        elif candidate.status in {"repaired_by_vision", "cross_page_merged"}:
            status = candidate.status
        else:
            status = "accepted_mineru"
        candidate.status = status
        return TableValidationResult(
            accepted=True,
            activation_allowed=True,
            status=status,
            table=candidate,
        )

    @classmethod
    def _reasons(cls, table: CanonicalTable) -> list[str]:
        reasons = list(CanonicalQualityGate._invalid_table_reasons(table))

        def add(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        if not table.source_spans:
            add("source_spans_missing")
        if not any(value.strip() for value in table.headers):
            add("header_empty")
        if not any(value.strip() for row in table.rows for value in row):
            add("data_rows_empty")

        expected_number = str(table.metadata.get("table_number") or "").strip()
        if expected_number:
            caption_match = cls._caption_number.search(table.caption or "")
            if caption_match is None or caption_match.group(1).casefold() != expected_number.casefold():
                add("caption_number_mismatch")

        if table.metadata.get("truncated") is True:
            add("silent_truncation")
        source_row_count = table.metadata.get("source_row_count")
        if isinstance(source_row_count, int) and source_row_count != len(table.rows):
            add("silent_truncation")
        if table.metadata.get("fragmented_numeric_tokens"):
            add("numeric_token_fragmented")
        if table.metadata.get("fragmented_unit_tokens"):
            add("unit_token_fragmented")
        if cls._has_split_numeric_or_unit(table):
            add("numeric_token_fragmented")
        if table.metadata.get("continuation_expected") and not table.metadata.get(
            "continuation_recovered"
        ):
            add("cross_page_continuation_missing")
        return reasons

    @staticmethod
    def _has_split_numeric_or_unit(table: CanonicalTable) -> bool:
        numeric = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)$")
        unit = re.compile(
            r"^(?:%|‰|°[CFK]|kg|mg|g|km|cm|mm|m|ms|μs|ns|s|Hz|kHz|MHz|GHz|"
            r"KB|MB|GB|TB|bps|kbps|Mbps|Gbps)$",
            re.I,
        )
        explicit_unit_headers = {"unit", "units", "单位", "量纲"}
        for row in table.rows:
            for column_index in range(1, len(row)):
                left = row[column_index - 1].strip()
                right = row[column_index].strip()
                header = (
                    table.headers[column_index].strip().casefold()
                    if column_index < len(table.headers)
                    else ""
                )
                if (
                    numeric.fullmatch(left)
                    and unit.fullmatch(right)
                    and header not in explicit_unit_headers
                ):
                    return True
                if left and right in {".", ","}:
                    return True
        return False

    @staticmethod
    def _repair_request(
        table: CanonicalTable,
        reasons: list[str],
    ) -> TableRepairRequest:
        locator = CanonicalQualityGate._table_locator(table)
        reason_text = ", ".join(reasons)
        return TableRepairRequest(
            table_id=table.table_id,
            reasons=reasons,
            locator=locator,
            source_fingerprint=table_identity_fingerprint(table),
            instructions=(
                "Re-extract only the located source table region. Preserve its caption, "
                "complete header/data grid, merged-cell spans, cell bboxes, numeric tokens, "
                f"units, footnotes, and source markdown. Validation failures: {reason_text}."
            ),
        )


class StructuredEvidenceBuilder:
    """Build table Parent/Child chunks and source-faithful visual evidence."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # A deterministic conservative boundary without coupling this layer to a
        # model-specific tokenizer. Every non-whitespace unit is indivisible.
        return len(re.findall(r"\S+", text))

    def table_chunks(
        self,
        table: CanonicalTable,
        max_tokens: int,
    ) -> list[StructuredEvidenceChunk]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        validation = TableValidator().validate(table)
        if not validation.accepted:
            raise ValueError(
                f"table {table.table_id!r} failed validation: {validation.reasons}"
            )
        table = validation.table
        parent_text = self._table_text(table, list(range(len(table.rows))))
        parent_id = self._stable_id(
            "table-parent",
            table_identity_fingerprint(table),
            parent_text,
        )
        parent = StructuredEvidenceChunk(
            chunk_id=parent_id,
            chunk_role="parent",
            block_type="table",
            text=parent_text,
            embedding_text=parent_text,
            token_count=self.estimate_tokens(parent_text),
            source_spans=self._deduplicate_spans(table.source_spans),
            metadata=self._table_metadata(
                table,
                list(range(len(table.rows))),
                overflow=False,
            ),
        )

        children: list[StructuredEvidenceChunk] = []
        for row_indices in self._semantic_row_groups(table):
            group_text = self._table_text(table, row_indices)
            if self.estimate_tokens(group_text) <= max_tokens:
                child_groups = [row_indices]
            else:
                child_groups = self._split_group_by_token_limit(
                    table,
                    row_indices,
                    max_tokens,
                )
            for child_indices in child_groups:
                child_text = self._table_text(table, child_indices)
                token_count = self.estimate_tokens(child_text)
                overflow = token_count > max_tokens
                child_id = self._stable_id(
                    "table-child",
                    parent_id,
                    json.dumps(child_indices, separators=(",", ":")),
                    child_text,
                )
                children.append(
                    StructuredEvidenceChunk(
                        chunk_id=child_id,
                        parent_chunk_id=parent_id,
                        chunk_role="child",
                        block_type="table",
                        text=child_text,
                        embedding_text=child_text,
                        token_count=token_count,
                        source_spans=self._row_source_spans(table, child_indices),
                        metadata=self._table_metadata(
                            table,
                            child_indices,
                            overflow=overflow,
                        ),
                    )
                )
        return [parent, *children]

    def merge_cross_page_tables(
        self,
        tables: Iterable[CanonicalTable],
    ) -> list[CanonicalTable]:
        merged: list[CanonicalTable] = []
        roots: dict[str, CanonicalTable] = {}
        for source_table in tables:
            table = source_table.model_copy(deep=True)
            continuation_of = str(table.metadata.get("continuation_of") or "").strip()
            root = roots.get(continuation_of)
            if not continuation_of or root is None:
                merged.append(table)
                roots[table.table_id] = table
                continue
            self._merge_continuation(root, table)
            roots[table.table_id] = root
        return merged

    def figure_chunk(
        self,
        figure: CanonicalFigure,
        nearby_blocks: Iterable[CanonicalBlock],
    ) -> StructuredEvidenceChunk:
        nearby = self._nearby_source_text(figure.nearby_block_ids, nearby_blocks)
        source_parts: list[str] = []
        if figure.caption:
            source_parts.append(figure.caption)
        if figure.asset_path:
            source_parts.append(f"![{figure.caption or figure.figure_id}]({figure.asset_path})")
        if figure.description:
            source_parts.append(figure.description)
        source_parts.extend(nearby)
        text = "\n\n".join(self._unique_nonempty(source_parts))
        derived = [figure.generated_summary] if figure.generated_summary else []
        embedding_text = "\n\n".join([text, *derived]).strip()
        warnings = list(figure.warnings)
        if figure.analysis_status == "failed" and not warnings:
            warnings.append("Optional figure analysis failed; source evidence remains accepted.")
        metadata = {
            "figure_id": figure.figure_id,
            "asset_path": figure.asset_path,
            "nearby_block_ids": figure.nearby_block_ids,
            "analysis_status": figure.analysis_status,
            "accepted": True,
            "warnings": warnings,
            "provenance": {
                "source": {"generated": False, "source_spans": self._span_json(figure.source_spans)},
                "generated_summary": {
                    "generated": True,
                    "model": figure.analysis_model,
                    "present": figure.generated_summary is not None,
                },
            },
        }
        return StructuredEvidenceChunk(
            chunk_id=self._stable_id("figure", figure.figure_id, text, embedding_text),
            chunk_role="parent",
            block_type="figure",
            text=text,
            embedding_text=embedding_text,
            token_count=self.estimate_tokens(embedding_text),
            source_spans=figure.source_spans,
            metadata=metadata,
        )

    def formula_chunk(
        self,
        formula: CanonicalFormula,
        nearby_blocks: Iterable[CanonicalBlock],
    ) -> StructuredEvidenceChunk:
        nearby = self._nearby_source_text(formula.nearby_block_ids, nearby_blocks)
        source_parts: list[str] = []
        if formula.caption:
            source_parts.append(formula.caption)
        source_parts.append(f"$$\n{formula.latex}\n$$")
        if formula.description:
            source_parts.append(formula.description)
        source_parts.extend(nearby)
        text = "\n\n".join(self._unique_nonempty(source_parts))
        derived = [formula.generated_explanation] if formula.generated_explanation else []
        embedding_text = "\n\n".join([text, *derived]).strip()
        warnings = list(formula.warnings)
        if formula.analysis_status == "failed" and not warnings:
            warnings.append("Optional formula analysis failed; source evidence remains accepted.")
        metadata = {
            "formula_id": formula.formula_id,
            "nearby_block_ids": formula.nearby_block_ids,
            "analysis_status": formula.analysis_status,
            "accepted": True,
            "warnings": warnings,
            "provenance": {
                "source": {"generated": False, "source_spans": self._span_json(formula.source_spans)},
                "generated_explanation": {
                    "generated": True,
                    "model": formula.analysis_model,
                    "present": formula.generated_explanation is not None,
                },
            },
        }
        return StructuredEvidenceChunk(
            chunk_id=self._stable_id("formula", formula.formula_id, text, embedding_text),
            chunk_role="parent",
            block_type="formula",
            text=text,
            embedding_text=embedding_text,
            token_count=self.estimate_tokens(embedding_text),
            source_spans=formula.source_spans,
            metadata=metadata,
        )

    def _split_group_by_token_limit(
        self,
        table: CanonicalTable,
        indices: list[int],
        max_tokens: int,
    ) -> list[list[int]]:
        groups: list[list[int]] = []
        current: list[int] = []
        for index in indices:
            proposed = [*current, index]
            if current and self.estimate_tokens(self._table_text(table, proposed)) > max_tokens:
                groups.append(current)
                current = [index]
            else:
                current = proposed
            if len(current) == 1 and self.estimate_tokens(
                self._table_text(table, current)
            ) > max_tokens:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _semantic_row_groups(table: CanonicalTable) -> list[list[int]]:
        configured = table.metadata.get("semantic_row_groups")
        if isinstance(configured, list):
            groups: list[list[int]] = []
            seen: set[int] = set()
            for raw_group in configured:
                if not isinstance(raw_group, list):
                    continue
                group = [
                    index
                    for index in raw_group
                    if isinstance(index, int)
                    and 0 <= index < len(table.rows)
                    and index not in seen
                ]
                if group:
                    groups.append(group)
                    seen.update(group)
            groups.extend([[index] for index in range(len(table.rows)) if index not in seen])
            if groups:
                return groups

        groups = []
        current: list[int] = []
        current_key: str | None = None
        for index, row in enumerate(table.rows):
            key = row[0].strip().casefold() if row else ""
            if current and key != current_key:
                groups.append(current)
                current = []
            current.append(index)
            current_key = key
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _table_text(table: CanonicalTable, row_indices: list[int]) -> str:
        lines: list[str] = []
        if table.caption:
            lines.append(table.caption)
        lines.extend(StructuredEvidenceBuilder._markdown_grid(table.headers, [table.rows[i] for i in row_indices]))
        lines.extend(f"Footnote: {footnote}" for footnote in table.footnotes)
        return "\n".join(lines)

    @staticmethod
    def _markdown_grid(headers: list[str], rows: list[list[str]]) -> list[str]:
        def escape(value: str) -> str:
            return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")

        lines = [
            "| " + " | ".join(escape(value) for value in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
        return lines

    @staticmethod
    def _table_metadata(
        table: CanonicalTable,
        row_indices: list[int],
        *,
        overflow: bool,
    ) -> dict[str, Any]:
        selected_grid_rows = {index + 1 for index in row_indices}
        cells = [
            cell.model_dump(mode="json")
            for cell in table.cells
            if cell.row_index == 0 or cell.row_index in selected_grid_rows
        ]
        return {
            "table_id": table.table_id,
            "table_identity": table_identity_fingerprint(table),
            "status": table.status,
            "caption": table.caption,
            "headers": table.headers,
            "rows": [table.rows[index] for index in row_indices],
            "row_indices": row_indices,
            "cells": cells,
            "footnotes": table.footnotes,
            "source_markdown": table.source_markdown,
            "normalized_markdown": StructuredEvidenceBuilder._table_text_without_caption(
                table, row_indices
            ),
            "overflow": overflow,
        }

    @staticmethod
    def _table_text_without_caption(table: CanonicalTable, row_indices: list[int]) -> str:
        return "\n".join(
            StructuredEvidenceBuilder._markdown_grid(
                table.headers,
                [table.rows[index] for index in row_indices],
            )
        )

    @staticmethod
    def _row_source_spans(table: CanonicalTable, row_indices: list[int]) -> list[SourceSpan]:
        selected = {index + 1 for index in row_indices}
        spans = list(table.source_spans)
        for cell in table.cells:
            if cell.row_index == 0 or cell.row_index in selected:
                spans.extend(cell.source_spans)
        return StructuredEvidenceBuilder._deduplicate_spans(spans)

    @staticmethod
    def _deduplicate_spans(spans: Iterable[SourceSpan]) -> list[SourceSpan]:
        result: list[SourceSpan] = []
        seen: set[str] = set()
        for span in spans:
            key = json.dumps(span.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            if key not in seen:
                result.append(span)
                seen.add(key)
        return result

    @staticmethod
    def _merge_continuation(root: CanonicalTable, continuation: CanonicalTable) -> None:
        if continuation.headers != root.headers:
            raise ValueError(
                f"continuation {continuation.table_id!r} header does not match {root.table_id!r}"
            )
        source_markdowns = list(root.metadata.get("source_markdowns") or [])
        if not source_markdowns and root.source_markdown:
            source_markdowns.append(root.source_markdown)
        if continuation.source_markdown:
            source_markdowns.append(continuation.source_markdown)

        repeated_data_header = bool(
            continuation.rows and continuation.rows[0] == root.headers
        )
        skip_data_rows = 1 if repeated_data_header else 0
        base_data_count = len(root.rows)
        repeated_cells: list[dict[str, Any]] = list(
            root.metadata.get("merged_repeated_header_cells") or []
        )
        continuation_page = next(
            (span.page_index for span in continuation.source_spans if span.page_index is not None),
            None,
        )
        for cell in continuation.cells:
            original = cell.model_dump(mode="json")
            if cell.row_index == 0 or (repeated_data_header and cell.row_index == 1):
                repeated_cells.append(original)
                continue
            copied = cell.model_copy(deep=True)
            original_row_index = copied.row_index
            copied.row_index = 1 + base_data_count + (copied.row_index - 1 - skip_data_rows)
            copied.metadata = {
                **copied.metadata,
                "original_table_id": continuation.table_id,
                "original_page_index": continuation_page,
                "original_row_index": original_row_index,
            }
            root.cells.append(copied)

        for cell in root.cells:
            cell.metadata.setdefault("original_table_id", root.table_id)
            if "original_page_index" not in cell.metadata:
                cell.metadata["original_page_index"] = next(
                    (span.page_index for span in root.source_spans if span.page_index is not None),
                    None,
                )
            cell.metadata.setdefault("original_row_index", cell.row_index)
        root.rows.extend(continuation.rows[skip_data_rows:])
        root.source_spans = StructuredEvidenceBuilder._deduplicate_spans(
            [*root.source_spans, *continuation.source_spans]
        )
        root.footnotes = StructuredEvidenceBuilder._unique_nonempty(
            [*root.footnotes, *continuation.footnotes]
        )
        root.status = "cross_page_merged"
        root.metadata.update(
            {
                "continuation_recovered": True,
                "merged_table_ids": [
                    *list(root.metadata.get("merged_table_ids") or [root.table_id]),
                    continuation.table_id,
                ],
                "source_markdowns": source_markdowns,
                "merged_repeated_header_cells": repeated_cells,
            }
        )
        normalized = "\n".join(
            StructuredEvidenceBuilder._markdown_grid(root.headers, root.rows)
        )
        root.normalized_markdown = normalized
        root.source_markdown = normalized

    @staticmethod
    def _nearby_source_text(
        nearby_ids: list[str],
        blocks: Iterable[CanonicalBlock],
    ) -> list[str]:
        wanted = set(nearby_ids)
        return [
            block.text
            for block in blocks
            if block.block_id in wanted
            and block.block_type in {"narrative", "appendix"}
            and not block.metadata.get("generated")
            and block.text.strip()
        ]

    @staticmethod
    def _unique_nonempty(values: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in seen:
                result.append(normalized)
                seen.add(normalized)
        return result

    @staticmethod
    def _span_json(spans: list[SourceSpan]) -> list[dict[str, Any]]:
        return [span.model_dump(mode="json") for span in spans]

    @staticmethod
    def _stable_id(prefix: str, *parts: str) -> str:
        payload = "\x1f".join(parts).encode("utf-8")
        return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"
