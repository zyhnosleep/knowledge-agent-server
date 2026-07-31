"""Lossless canonical table assembly and deterministic fact extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from app.services.table_extraction import structure_table_markdown
from app.services.table_normalization import normalize_table_text


@dataclass(frozen=True)
class CanonicalTableChunk:
    chunk_id: str
    document_id: str
    parse_version: str
    table_id: str
    ordinal: int
    text: str
    page_label: str | None = None
    source_spans: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TableFact:
    table_id: str
    document_id: str
    parse_version: str
    row_label: str
    column: str
    value: str
    row_index: int
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class TableContext:
    document_id: str
    parse_version: str
    table_id: str
    label: str | None
    markdown: str
    headers: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    source_chunk_ids: tuple[str, ...]
    row_count: int
    quality_flags: tuple[str, ...] = ()
    row_source_chunk_ids: tuple[tuple[str, ...], ...] = ()


def assemble_table_context(chunks: Sequence[CanonicalTableChunk]) -> TableContext:
    """Assemble all Children for one canonical table without character slicing."""

    ordered = sorted(chunks, key=lambda item: (item.ordinal, item.chunk_id))
    if not ordered:
        raise ValueError("Cannot assemble an empty canonical table")

    first = ordered[0]
    if any(
        (item.document_id, item.parse_version, item.table_id)
        != (first.document_id, first.parse_version, first.table_id)
        for item in ordered
    ):
        raise ValueError("Canonical table chunks must share document, version, and table ID")

    caption_lines: list[str] = []
    table_lines: list[tuple[str, str]] = []
    for chunk in ordered:
        normalized = normalize_table_text(chunk.text)
        for line in normalized.splitlines():
            stripped = line.strip()
            if stripped.startswith("|") and "|" in stripped[1:]:
                table_lines.append((stripped, chunk.chunk_id))
            elif stripped and not caption_lines:
                caption_lines.append(stripped)

    unique_source_ids = tuple(dict.fromkeys(chunk.chunk_id for chunk in ordered))
    quality_flags: list[str] = []
    if not table_lines:
        quality_flags.append("no_table_rows")
        return TableContext(
            document_id=first.document_id,
            parse_version=first.parse_version,
            table_id=first.table_id,
            label=caption_lines[0] if caption_lines else None,
            markdown="\n".join(caption_lines),
            headers=(),
            rows=(),
            source_chunk_ids=unique_source_ids,
            row_count=0,
            quality_flags=tuple(quality_flags),
        )

    header_line = next(
        (line for line, _chunk_id in table_lines if not _is_separator_line(line)),
        table_lines[0][0],
    )
    data_lines: list[tuple[str, str]] = []
    header_key = _line_key(header_line)
    for line, chunk_id in table_lines:
        if _is_separator_line(line) or _line_key(line) == header_key:
            continue
        if not _line_key(line):
            continue
        data_lines.append((line, chunk_id))

    combined_lines = [*caption_lines, header_line, _separator_for(header_line)]
    combined_lines.extend(line for line, _chunk_id in data_lines)
    combined_markdown = "\n".join(combined_lines).strip()
    structured = structure_table_markdown(combined_markdown)
    rows = tuple(structured.rows)
    row_source_chunk_ids = tuple(
        (data_lines[index][1],) if index < len(data_lines) else ()
        for index in range(len(rows))
    )
    if not rows:
        quality_flags.append("no_data_rows")
    if len(row_source_chunk_ids) != len(rows):
        quality_flags.append("row_source_count_mismatch")

    return TableContext(
        document_id=first.document_id,
        parse_version=first.parse_version,
        table_id=first.table_id,
        label=structured.label or (caption_lines[0] if caption_lines else None),
        markdown=structured.markdown,
        headers=tuple(structured.headers),
        rows=rows,
        source_chunk_ids=unique_source_ids,
        row_count=len(rows),
        quality_flags=tuple(dict.fromkeys([*quality_flags, *structured.quality_flags])),
        row_source_chunk_ids=row_source_chunk_ids,
    )


def extract_table_facts(question: str, table: TableContext) -> list[TableFact]:
    """Extract exact source cells matching row and column selectors in a query."""

    if not table.rows or not table.headers:
        return []
    question_key = _selector_key(question)
    row_header = _row_header(table.headers)
    columns = [
        header
        for header in table.headers
        if header != row_header and _selector_key(header) in question_key
    ]
    if not columns:
        columns = [header for header in table.headers if header != row_header]

    selected_rows: list[tuple[int, dict[str, str]]] = []
    for index, row in enumerate(table.rows):
        label = _row_label(row, row_header)
        if label and _selector_key(label) in question_key:
            selected_rows.append((index, row))
    if not selected_rows:
        selected_rows = list(enumerate(table.rows))

    facts: list[TableFact] = []
    for row_index, row in selected_rows:
        row_label = _row_label(row, row_header)
        source_ids = (
            table.row_source_chunk_ids[row_index]
            if row_index < len(table.row_source_chunk_ids)
            else table.source_chunk_ids
        )
        for column in columns:
            value = str(row.get(column) or "").strip()
            if not value or _is_separator_value(value):
                continue
            facts.append(
                TableFact(
                    table_id=table.table_id,
                    document_id=table.document_id,
                    parse_version=table.parse_version,
                    row_label=row_label,
                    column=column,
                    value=value,
                    row_index=row_index,
                    source_chunk_ids=tuple(source_ids),
                )
            )
    return facts


def _row_header(headers: Sequence[str]) -> str:
    preferred = ("Model", "Variant", "Method", "Approach", "System", "Dataset", "Name")
    for candidate in preferred:
        for header in headers:
            if _selector_key(header) == _selector_key(candidate):
                return header
    return headers[0]


def _row_label(row: dict[str, str], row_header: str) -> str:
    value = str(row.get(row_header) or "").strip()
    if value:
        return value
    for cell in row.values():
        clean = str(cell or "").strip()
        if clean and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", clean):
            return clean
    return ""


def _selector_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _line_key(line: str) -> str:
    return _selector_key(line)


def _is_separator_line(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in cells)


def _separator_for(header_line: str) -> str:
    cells = header_line.strip().strip("|").split("|")
    return "| " + " | ".join("---" for _ in cells) + " |"


def _is_separator_value(value: str) -> bool:
    return bool(re.fullmatch(r"[-–—]+", value.strip()))
