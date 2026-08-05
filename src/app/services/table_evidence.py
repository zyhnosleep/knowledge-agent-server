"""Lossless canonical table assembly and deterministic fact extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from app.services.table_extraction import _classify_semantic_header_row, structure_table_markdown
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

    value_columns = _value_column_indices(data_lines)
    consumed_header_like_rows = False
    while True:
        header_like = next(
            (
                (line, kind)
                for line, _chunk_id in data_lines
                if (kind := _classify_header_like_row(header_line, line, value_columns))
                is not None
            ),
            None,
        )
        if header_like is None:
            break
        header_like_line, header_like_kind = header_like
        consumed_header_like_rows = True
        if header_like_kind == "compose":
            header_line = _compose_header_lines(header_line, header_like_line)
        header_like_key = _line_key(header_like_line)
        data_lines = [
            (line, chunk_id)
            for line, chunk_id in data_lines
            if _line_key(line) != header_like_key
        ]
    if consumed_header_like_rows:
        quality_flags.append("header_like_data_row")

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
    question_terms = _question_match_keys(question)
    label_headers = _row_label_headers(table.headers, table.rows)
    non_label_headers = [
        header
        for header in table.headers
        if header not in label_headers
        and (header_key := _selector_key(header))
    ]
    # A question often names both a row selector and the actual value
    # columns.  The old ``header_key in question_key`` check treated a row
    # header such as ``Solving group`` as the only exact column match and
    # consequently discarded ``O ff99SB`` / ``O ff14SB``.  Prefer headers
    # that share a non-generic selector with the question; fall back to all
    # value columns only when no specific column signal is present.
    strong_columns = [
        header
        for header in non_label_headers
        if _header_has_specific_question_signal(header, question_key, question_terms)
    ]
    if strong_columns:
        columns = strong_columns
    else:
        columns = [
            header
            for header in non_label_headers
            if _header_matches_question(header, question_key, question_terms)
        ]
    if not columns:
        columns = non_label_headers

    # Hierarchical row labels are often rendered once and left blank on the
    # continuation rows (merged cells).  Forward-fill each label column so a
    # repeated child label such as ``C36m`` keeps its parent and distinct rows
    # never share an identical label.
    label_rows: list[tuple[int, str]] = []
    filled: dict[str, str] = {header: "" for header in label_headers}
    for index, row in enumerate(table.rows):
        for header in label_headers:
            value = str(row.get(header) or "").strip()
            if value:
                filled[header] = value
        label_rows.append((index, _join_row_label(row, label_headers, filled)))

    # Prefer exact label matches (the full row label appears in the question)
    # so a question naming one specific row ("Protein Z") never drags in every
    # sibling row that merely shares a generic prefix ("Protein").  When no
    # row label is literally contained in the question, fall back to
    # question-term matching so property rows (C6, mu, surface tension) with
    # unit-bearing labels stay selectable.
    exact_rows = [
        (index, label)
        for index, label in label_rows
        if label and _selector_key(label) in question_key
    ]
    matched_rows = [
        (index, label)
        for index, label in label_rows
        if label and _row_matches_question(label, question_key, question_terms)
    ]
    if exact_rows:
        # Keep the precision of exact row selection, but add property rows
        # selected through a scientific alias.  For example, a query that
        # explicitly names C6 can still request the µ/gamma rows through the
        # Chinese aliases "偶极矩" and "表面张力".  This is a union, not a
        # broad sibling-row fallback, so unrelated rows remain excluded.
        selected_by_index = {index: label for index, label in exact_rows}
        alias_keys = _question_alias_keys(question)
        for index, label in matched_rows:
            label_key = _selector_key(label)
            if index in selected_by_index:
                continue
            if any(
                alias_key and (alias_key in label_key or label_key in alias_key)
                for alias_key in alias_keys
            ):
                selected_by_index[index] = label
        selected_rows = [
            (index, label)
            for index, label in label_rows
            if index in selected_by_index
        ]
    else:
        selected_rows = matched_rows
        if not selected_rows:
            selected_rows = label_rows

    facts: list[TableFact] = []
    for row_index, row_label in selected_rows:
        source_ids = (
            table.row_source_chunk_ids[row_index]
            if row_index < len(table.row_source_chunk_ids)
            else table.source_chunk_ids
        )
        row = table.rows[row_index]
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


def _row_label_headers(headers: Sequence[str], rows: Sequence[dict[str, str]]) -> list[str]:
    """Return the leading column headers that form hierarchical row labels.

    A column is a value column when any of its data cells is numeric; the
    row label is the leading run of non-value columns, so numeric cells are
    never merged into a row label. When the first column is itself a value
    column (an index or rank column such as ``Rank``), the leading run is
    empty and the row label falls back to a preferred non-numeric label
    column (``Model``, ``System``, ...) so indexed/ranked tables keep
    readable row labels instead of numeric ranks. Falls back to the first
    header so fact extraction keeps a deterministic row identity for
    unusual tables.
    """
    if not headers:
        return []
    value_headers = {
        header
        for header in headers
        if any(_is_numeric_cell(str(row.get(header) or "")) for row in rows)
    }
    labels: list[str] = []
    for header in headers:
        if header in value_headers:
            break
        labels.append(header)
    if labels:
        return labels
    return _preferred_label_headers(headers, value_headers)


def _preferred_label_headers(
    headers: Sequence[str], value_headers: set[str]
) -> list[str]:
    """Pick a deterministic non-numeric row-label column when none leads.

    ``_row_label_headers`` has already consumed any leading run of non-value
    headers, so reaching this helper means the table starts with a value
    column (an index or rank). Reuse the historical preferred-label names so
    ranked tables such as ``Rank | Model | Accuracy`` label rows by model
    name; the numeric rank stays a value column. Only non-value columns are
    eligible, so numeric cells are never merged into a row label.
    """
    preferred = ("Model", "Variant", "Method", "Approach", "System", "Dataset", "Name")
    for candidate in preferred:
        for header in headers:
            if (
                header not in value_headers
                and _selector_key(header) == _selector_key(candidate)
            ):
                return [header]
    for header in headers:
        if header not in value_headers and header.strip():
            return [header]
    return [headers[0]]


def _row_label(row: dict[str, str], label_headers: Sequence[str]) -> str:
    parts = [str(row.get(header) or "").strip() for header in label_headers]
    parts = [part for part in parts if part]
    if parts:
        return " / ".join(parts)
    for cell in row.values():
        clean = str(cell or "").strip()
        if clean and not _is_numeric_cell(clean):
            return clean
    return ""


def _join_row_label(
    row: dict[str, str],
    label_headers: Sequence[str],
    filled: dict[str, str],
) -> str:
    """Join hierarchical label cells, inheriting blank parents from ``filled``.

    ``filled`` holds the most recent non-empty value per label header so a
    continuation row whose parent cell is blank (merged-cell layout) keeps the
    parent name and stays distinguishable from other rows with the same child.
    """
    parts: list[str] = []
    for header in label_headers:
        value = str(row.get(header) or "").strip() or filled.get(header, "")
        if value:
            parts.append(value)
    if parts:
        return " / ".join(parts)
    for cell in row.values():
        clean = str(cell or "").strip()
        if clean and not _is_numeric_cell(clean):
            return clean
    return ""


def _row_matches_question(
    label: str,
    question_key: str,
    question_terms: list[str],
) -> bool:
    """Decide whether a row label answers the question.

    A row is selected when its label key appears in the question, or when a
    question term (acronym, property alias, or token) is a substring of the
    label in either direction.  Bidirectional substring matching keeps short
    identifiers such as ``C6`` or ``mu`` able to select property rows whose
    label carries extra unit text.
    """
    label_key = _selector_key(label)
    if not label_key:
        return False
    if label_key in question_key:
        return True
    return any(
        term_key and (term_key in label_key or label_key in term_key)
        for term_key in question_terms
    )


def _question_match_keys(question: str) -> list[str]:
    """Return normalized keys used to match table rows to a question.

    Includes the full question key, individual alphanumeric tokens, and
    Chinese property aliases so a Chinese question such as "C6 偶极矩和
    表面张力" can select the English property rows ``C6``, ``mu``/``dipole``
    and ``surface tension``/``gamma``.
    """
    keys: set[str] = set()
    question_key = _selector_key(question)
    if question_key:
        keys.add(question_key)
    for match in re.finditer(r"[a-z0-9]+", str(question or "").lower()):
        token = match.group(0)
        if len(token) >= 2:
            keys.add(token)
    aliases = _QUESTION_ALIASES
    for marker, alias_keys in aliases.items():
        if marker in question:
            keys.update(alias_keys)
    return [key for key in keys if key]


_QUESTION_ALIASES = {
    "偶极矩": ("mu", "dipole"),
    "表面张力": ("surface", "tension", "gamma"),
    "焓": ("enthalpy", "hvap", "vap"),
    "水化": ("hydration", "hfe"),
    "结合": ("binding", "hfe"),
    "误差": ("rmse", "error"),
    "角度": ("theta", "angle", "torsion"),
    "实验": ("exp", "exptl"),
    "半径": ("radius", "rg"),
    "热容": ("heat", "capacity"),
}


def _question_alias_keys(question: str) -> tuple[str, ...]:
    keys: list[str] = []
    for marker, aliases in _QUESTION_ALIASES.items():
        if marker not in question:
            continue
        for alias in aliases:
            key = _selector_key(alias)
            if key and key not in keys:
                keys.append(key)
    return tuple(keys)


_GENERIC_COLUMN_TERMS = {
    # A shared scientific modifier is not enough to select a column when a
    # table has several ``alpha L ...`` metrics.  The complete requested
    # header (or its other metric token) still matches; this only prevents the
    # bare modifier from making every sibling column look relevant.
    "alpha",
    "table",
    "tables",
    "value",
    "values",
    "metric",
    "metrics",
    "objective",
    "objectives",
    "solving",
    "group",
    "parameter",
    "parameters",
    "param",
    "params",
    "structure",
    "structures",
    "pair",
    "pairs",
    "number",
    "numbers",
    "model",
    "models",
    "row",
    "rows",
}


def _header_matches_question(
    header: str,
    question_key: str,
    question_terms: Sequence[str],
) -> bool:
    header_key = _selector_key(header)
    if not header_key:
        return False
    terms = [_selector_key(term) for term in question_terms]
    return header_key in question_key or any(
        len(term) >= 2 and (term in header_key or header_key in term)
        for term in terms
        if term
    )


def _header_has_specific_question_signal(
    header: str,
    question_key: str,
    question_terms: Sequence[str],
) -> bool:
    header_key = _selector_key(header)
    if not header_key:
        return False
    for term in question_terms:
        term_key = _selector_key(term)
        if len(term_key) < 3 or term_key in _GENERIC_COLUMN_TERMS:
            continue
        if term_key in header_key or header_key in term_key:
            return True
    return False


# Greek letters map to their Latin names so a row label rendered as ``γ``,
# ``μ`` or ``χ`` normalizes to the same key as the ASCII spelling used in a
# question alias (``gamma``, ``mu``, ``chi``).
_GREEK_TO_NAME = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "ς": "sigma",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Α": "alpha",
    "Β": "beta",
    "Γ": "gamma",
    "Δ": "delta",
    "Ε": "epsilon",
    "Ζ": "zeta",
    "Η": "eta",
    "Θ": "theta",
    "Ι": "iota",
    "Κ": "kappa",
    "Λ": "lambda",
    "Μ": "mu",
    "Ν": "nu",
    "Ξ": "xi",
    "Ο": "omicron",
    "Π": "pi",
    "Ρ": "rho",
    "Σ": "sigma",
    "Τ": "tau",
    "Υ": "upsilon",
    "Φ": "phi",
    "Χ": "chi",
    "Ψ": "psi",
    "Ω": "omega",
    "µ": "mu",
    "Å": "a",
}


def _selector_key(value: str) -> str:
    normalized = str(value or "").lower()
    for character, name in _GREEK_TO_NAME.items():
        normalized = normalized.replace(character, name)
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _line_key(line: str) -> str:
    return _selector_key(line)


def _is_separator_line(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in cells)


def _separator_for(header_line: str) -> str:
    cells = header_line.strip().strip("|").split("|")
    return "| " + " | ".join("---" for _ in cells) + " |"


def _classify_header_like_row(
    primary_line: str,
    candidate_line: str,
    value_columns: set[int],
) -> str | None:
    """Classify a candidate table row relative to the current header.

    Decision order: separator -> exact repeated header -> semantic
    header-like row -> data row. Delegates to
    :func:`~app.services.table_extraction._classify_semantic_header_row` so
    canonical assembly and direct table extraction share one classifier.
    ``value_columns`` are the data-line column indices that carry at least
    one numeric cell; they provide the numeric-body support needed to decide
    between compose (multi-level headers), drop (redundant semantic repeats
    such as ``Group | Edgewise | Pairwise`` under ``Group | OPLS4 | OPLS5``),
    and retaining fully-text data rows.
    """
    primary = _line_cells(primary_line)
    candidate = _line_cells(candidate_line)
    return _classify_semantic_header_row(primary, candidate, value_columns)


def _value_column_indices(data_lines: Sequence[tuple[str, str]]) -> set[int]:
    """Return the data-line column indices that hold at least one numeric cell.

    Only refined columns that actually carry numeric values below can be
    sub-divided by a semantic header row; fully-text columns are kept as
    data so text-only rows are never silently dropped.
    """
    value_columns: set[int] = set()
    for line, _chunk_id in data_lines:
        for index, cell in enumerate(_line_cells(line)):
            if _is_numeric_cell(cell):
                value_columns.add(index)
    return value_columns


def _compose_header_lines(primary_line: str, secondary_line: str) -> str:
    primary = _line_cells(primary_line)
    secondary = _line_cells(secondary_line)
    composed: list[str] = []
    group = ""
    for top, bottom in zip(primary, secondary, strict=True):
        if top:
            group = top
        if not bottom:
            composed.append(top)
        elif group and _selector_key(group) != _selector_key(bottom):
            composed.append(f"{group} {bottom}")
        else:
            composed.append(group or bottom)
    return "| " + " | ".join(composed) + " |"


def _line_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


_NUMERIC_CELL_TOKEN_RE = r"[-+]?\d+(?:\.\d+)?%?"


def _is_numeric_cell(value: str) -> bool:
    """Return True when a cell contains only numeric content.

    Accepts plain numbers, signed values, percentages, ``\u00b1`` uncertainty
    pairs, and OCR/LaTeX-flattened cells where the uncertainty marker was
    dropped (``2.0 0.2``) or the sign was split from its number
    (``+ 0.9 0.2``).  A single non-numeric token rejects the cell.
    """
    text = re.sub(r"\s*(?:\u00b1|\+/-)\s*", " ", str(value or "").strip())
    tokens = text.split()
    if not tokens:
        return False
    for index, token in enumerate(tokens):
        if token in {"+", "-"}:
            if index + 1 < len(tokens) and re.fullmatch(
                r"\d+(?:\.\d+)?%?", tokens[index + 1]
            ):
                continue
            return False
        if not re.fullmatch(_NUMERIC_CELL_TOKEN_RE, token):
            return False
    return True


def _is_separator_value(value: str) -> bool:
    return bool(re.fullmatch(r"[-–—]+", value.strip()))
