from __future__ import annotations

import re


_DATASET_NAME_RE = re.compile(
    r"\b(OIE2016|NYT|PENN|WEB|CoNLL|ACE|SemEval|WikiSQL|SQuAD|GLUE|SuperGLUE)\b",
    re.IGNORECASE,
)
_LATEX_TEXT_COMMAND_RE = re.compile(
    r"\\(?:mathbf|mathrm|mathit|text|operatorname)\s*\{\s*([^{}]*?)\s*\}"
)


def normalize_table_text(text: str) -> str:
    """Normalize MinerU table markdown while preserving table shape."""
    lines: list[str] = []
    table_rows: list[list[str]] = []

    def flush_table_rows() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        for row in _repair_markdown_table_rows(table_rows):
            lines.append("| " + " | ".join(row) + " |")
        table_rows = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            cells = [normalize_table_cell(cell) for cell in stripped.strip("|").split("|")]
            table_rows.append(cells)
        else:
            flush_table_rows()
            lines.append(normalize_table_cell(line))
    flush_table_rows()
    return "\n".join(lines).strip()


def normalize_table_cell(cell: str) -> str:
    text = str(cell or "").strip()
    if not text:
        return ""

    text = text.replace("\\text{-}", "-").replace("\\-", "-").strip("$")
    previous = None
    while previous != text:
        previous = text
        text = _LATEX_TEXT_COMMAND_RE.sub(lambda match: _compact_latex_group(match.group(1)), text)

    text = re.sub(r"_\s*\{\s*([^{}]*?)\s*\}", lambda match: " " + _compact_latex_group(match.group(1)), text)
    text = re.sub(r"\^\s*\{\s*([^{}]*?)\s*\}", lambda match: "^" + _compact_latex_group(match.group(1)), text)
    text = re.sub(r"\\([A-Za-z])", r"\1", text)
    text = re.sub(r"\\(?:quad|,|;|!|\s)", " ", text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    text = _compact_common_model_names(text)
    text = re.sub(r"\s*/\s*", "/", text)
    return re.sub(r"\s+", " ", text).strip()


def _compact_latex_group(value: str) -> str:
    text = re.sub(r"\\([A-Za-z])", r"\1", value)
    text = re.sub(r"\s+", " ", text.strip())
    tokens = text.split()
    if tokens and all(len(token) == 1 or token in {"/", "-"} for token in tokens):
        return "".join(tokens)
    return _compact_common_model_names(text)


def _compact_common_model_names(text: str) -> str:
    text = re.sub(r"\bS\s*A\s*C\s*-\s*K\s*G\b", "SAC-KG", text, flags=re.IGNORECASE)
    text = re.sub(r"\bC\s*h\s*a\s*t\s*G\s*P\s*T\b", "ChatGPT", text, flags=re.IGNORECASE)
    for suffix in ("prompt", "text", "verifier", "pruner"):
        spaced_suffix = r"\s*".join(re.escape(char) for char in suffix)
        text = re.sub(rf"w\s*/\s*o\s+{spaced_suffix}", f"w/o {suffix}", text, flags=re.IGNORECASE)
    return text


def _repair_markdown_table_rows(rows: list[list[str]]) -> list[list[str]]:
    rows = [list(row) for row in rows]
    if not rows:
        return rows
    _repair_dataset_metric_header(rows)
    _repair_iteration_rowspans(rows)
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def _repair_dataset_metric_header(rows: list[list[str]]) -> None:
    metric_index = next(
        (
            index
            for index, row in enumerate(rows)
            if sum(1 for cell in row[1:] if _is_metric_cell(cell)) >= 4
        ),
        None,
    )
    if metric_index is None:
        return
    dataset_index = next(
        (
            index
            for index in range(metric_index - 1, -1, -1)
            if any(_DATASET_NAME_RE.search(cell) for cell in rows[index])
        ),
        None,
    )
    if dataset_index is None:
        return

    dataset_row = rows[dataset_index]
    metric_row = rows[metric_index]
    metric_count = max(len(metric_row) - 1, 0)
    if metric_count <= 0:
        return

    dataset_names = [
        match.group(0).upper()
        for cell in dataset_row[1:]
        for match in [_DATASET_NAME_RE.search(cell)]
        if match
    ]
    if not dataset_names or metric_count % len(dataset_names) != 0:
        return
    if len(dataset_names) == metric_count and all(dataset_row[index + 1].strip() for index in range(metric_count)):
        return

    group_size = metric_count // len(dataset_names)
    rows[dataset_index] = [dataset_row[0] if dataset_row else "", *[name for name in dataset_names for _ in range(group_size)]]


def _repair_iteration_rowspans(rows: list[list[str]]) -> None:
    if not rows:
        return
    header = [cell.lower() for cell in rows[0]]
    if not header or "iteration" not in header[0] or len(header) < 2 or "model" not in header[1]:
        return
    width = max(len(row) for row in rows)
    current_iteration = ""
    for index, row in enumerate(rows[1:], start=1):
        if _is_separator_row(row):
            continue
        first = row[0].strip() if row else ""
        if first.lower().startswith("iteration"):
            current_iteration = first
            continue
        if current_iteration and _looks_like_model_name(first):
            repaired = [current_iteration, *row]
            if len(repaired) > width and repaired[-1] == "":
                repaired = repaired[:-1]
            rows[index] = repaired[:width]


def _is_metric_cell(value: str) -> bool:
    lowered = value.lower()
    return bool(re.fullmatch(r"(f\s*1|f1|auc|precision|recall|accuracy)", lowered))


def _is_separator_row(row: list[str]) -> bool:
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in row)


def _looks_like_model_name(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in ("sac-kg", "openie", "stanford", "deepex", "pive", "human"))
