from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.table_normalization import normalize_table_text


@dataclass
class StructuredTable:
    label: str | None
    caption: str
    page_label: str | None
    markdown: str
    headers: list[str] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "caption": self.caption,
            "page_label": self.page_label,
            "markdown": self.markdown,
            "headers": self.headers,
            "rows": self.rows,
            "quality_flags": self.quality_flags,
        }


def structure_table_markdown(markdown: str, page_label: str | None = None) -> StructuredTable:
    normalized = normalize_table_text(markdown)
    lines = [line.rstrip() for line in normalized.splitlines() if line.strip()]
    caption_lines: list[str] = []
    table_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("|"):
            table_lines.append(line)
        else:
            caption_lines.append(line.strip())

    caption = " ".join(caption_lines).strip()
    label = _extract_table_label(caption or normalized)
    rows = _markdown_table_rows("\n".join(table_lines))
    headers, row_dicts = _rows_to_dicts(rows)
    quality_flags = _quality_flags(normalized, headers, row_dicts)
    return StructuredTable(
        label=label,
        caption=caption,
        page_label=page_label,
        markdown=normalized,
        headers=headers,
        rows=row_dicts,
        quality_flags=quality_flags,
    )


def extract_structured_tables(markdown_blocks: list[dict]) -> list[dict]:
    structured: list[dict] = []
    for table in markdown_blocks:
        if not isinstance(table, dict):
            continue
        markdown = str(table.get("markdown") or "").strip()
        if not markdown:
            continue
        page_label = str(table.get("page_label") or "") or None
        structured.append(structure_table_markdown(markdown, page_label=page_label).as_dict())
    return structured


def table_metric_values(
    table_markdown: str,
    requested_datasets: list[str] | None = None,
    row_selectors: list[str] | None = None,
    requested_subjects: list[str] | None = None,
) -> list[dict]:
    table = structure_table_markdown(table_markdown)
    requested = {_normalize_lookup_key(dataset) for dataset in requested_datasets or []}
    selectors = [_normalize_lookup_key(selector) for selector in [*(row_selectors or []), *(requested_subjects or [])] if selector]
    datasets = _dataset_names_from_headers(table.headers)
    candidates: list[dict] = []
    for row in table.rows:
        model = _row_model(row)
        if selectors and not _matches_any_selector(model, selectors):
            continue
        for dataset in datasets:
            dataset_key = _normalize_lookup_key(dataset)
            if requested and dataset_key not in requested:
                continue
            values = _dataset_metric_values(row, dataset)
            if values:
                candidates.append(
                    {
                        "table_label": table.label,
                        "dataset": dataset.upper(),
                        "values": values,
                        "model": model or "",
                    }
                )
    return candidates


def summarize_ablation_table(table_markdown: str) -> list[str]:
    table = structure_table_markdown(table_markdown)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in table.rows:
        iteration = row.get("Iteration rounds") or row.get("Iteration") or ""
        model = _row_model(row)
        if not iteration or not model:
            continue
        grouped.setdefault(iteration, []).append(row)

    findings: list[str] = []
    for iteration, rows in grouped.items():
        full = _select_full_model_row(rows)
        if not full:
            continue
        full_model = _row_model(full)
        precision = full.get("Precision", "")
        specificity = full.get("Domain Specificity", "")
        recalls = full.get("Number of recalls", "")
        parts = []
        if recalls:
            parts.append(f"recalls {recalls}")
        if precision:
            parts.append(f"precision {precision}")
        if specificity:
            parts.append(f"domain specificity {specificity}")
        if parts:
            findings.append(f"{iteration}: full {full_model} reports " + ", ".join(parts) + ".")
        worse = [
            _row_model(row)
            for row in rows
            if row is not full and _row_has_lower_scores(row, full)
        ]
        if worse:
            findings.append(f"{iteration}: ablated variants underperform the full model, including " + ", ".join(worse[:4]) + ".")
    return findings


def _rows_to_dicts(rows: list[list[str]]) -> tuple[list[str], list[dict[str, str]]]:
    if not rows:
        return [], []
    header_index = _header_index(rows)
    headers = _compose_headers(rows, header_index)
    data_start = header_index + 1
    if header_index + 1 < len(rows) and sum(1 for cell in rows[header_index + 1] if _is_metric_cell(cell)) >= 2:
        data_start = header_index + 2
    body_rows = rows[data_start:]
    body_rows = [row for row in body_rows if not _is_separator_row(row)]
    dicts: list[dict[str, str]] = []
    for row in body_rows:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * max(0, len(headers) - len(row))
        record = {headers[index]: padded[index].strip() for index in range(len(headers)) if headers[index]}
        if record:
            dicts.append(record)
    return headers, dicts


def _compose_headers(rows: list[list[str]], header_index: int) -> list[str]:
    header = list(rows[header_index])
    if header_index + 1 < len(rows):
        next_row = rows[header_index + 1]
        metric_count = sum(1 for cell in next_row if _is_metric_cell(cell))
        if metric_count >= 2:
            width = max(len(header), len(next_row))
            composed: list[str] = []
            for index in range(width):
                top = header[index].strip() if index < len(header) else ""
                bottom = next_row[index].strip() if index < len(next_row) else ""
                if top and bottom and _is_metric_cell(bottom):
                    composed.append(f"{top} {bottom}")
                else:
                    composed.append(top or bottom)
            return _dedupe_headers(composed)
    return _dedupe_headers(header)


def _header_index(rows: list[list[str]]) -> int:
    for index, row in enumerate(rows):
        lowered = [cell.lower() for cell in row]
        if any(cell in {"model", "variant"} or "model" in cell for cell in lowered):
            return index
        if any("iteration" in cell for cell in lowered):
            return index
    return 0


def _markdown_table_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "|" not in stripped[1:]:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and _is_separator_row(cells):
            continue
        rows.append(cells)
    return rows


def _quality_flags(markdown: str, headers: list[str], rows: list[dict[str, str]]) -> list[str]:
    flags: list[str] = []
    if not rows:
        flags.append("no_data_rows")
    if "\\mathbf" in markdown or "\\mathrm" in markdown:
        flags.append("contains_latex_markup")
    if any(not header for header in headers):
        flags.append("blank_headers")
    if _extract_table_label(markdown) and not rows:
        flags.append("caption_without_rows")
    return flags


def _dataset_names_from_headers(headers: list[str]) -> list[str]:
    datasets: list[str] = []
    for header in headers:
        dataset = _dataset_name_from_header(header)
        if dataset and dataset.upper() not in {item.upper() for item in datasets}:
            datasets.append(dataset)
    return datasets


def _dataset_metric_values(row: dict[str, str], dataset: str) -> dict[str, str]:
    values: dict[str, str] = {}
    lookup = {_normalize_lookup_key(key): value for key, value in row.items()}
    for metric in ("F1", "AUC", "Precision", "Recall", "Accuracy"):
        value = lookup.get(_normalize_lookup_key(f"{dataset} {metric}"))
        if value and re.search(r"\d+(?:\.\d+)?", value):
            values[metric] = value
    return values


def _row_model(row: dict[str, str]) -> str:
    for key in ("Model", "Variant", "Method", "Approach", "System"):
        if value := row.get(key):
            return value
    for key, value in row.items():
        if value and not _is_metric_header(key) and not re.search(r"\d+(?:\.\d+)?", value):
            return value
    return ""


def _dataset_name_from_header(header: str) -> str:
    clean = re.sub(r"\s+", " ", header.strip())
    for metric in ("F1", "AUC", "Precision", "Recall", "Accuracy"):
        match = re.match(rf"(.+?)\s+{re.escape(metric)}(?:\s+\d+)?$", clean, re.IGNORECASE)
        if match:
            dataset = match.group(1).strip()
            if dataset and dataset.lower() not in {"model", "variant", "method", "approach", "system"}:
                return dataset
    return ""


def _select_full_model_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    full_rows = [row for row in rows if _row_model(row) and not _looks_ablated(_row_model(row))]
    if not full_rows:
        full_rows = [row for row in rows if _row_model(row)]
    if not full_rows:
        return None
    return max(full_rows, key=_numeric_score_sum)


def _looks_ablated(model: str) -> bool:
    lowered = model.lower()
    return bool(
        re.search(r"\bw\s*/\s*o\b", lowered)
        or re.search(r"\bwithout\b", lowered)
        or re.search(r"\bablated?\b", lowered)
        or re.search(r"\bremoved?\b", lowered)
        or re.search(r"\bno\s+\w+", lowered)
    )


def _numeric_score_sum(row: dict[str, str]) -> float:
    total = 0.0
    count = 0
    for key, value in row.items():
        if _is_metric_header(key):
            numeric = _float_value(value)
            if numeric is not None:
                total += numeric
                count += 1
    return total if count else float("-inf")


def _is_metric_header(value: str) -> bool:
    lowered = value.lower()
    return bool(
        "f1" in lowered
        or "auc" in lowered
        or "precision" in lowered
        or "recall" in lowered
        or "accuracy" in lowered
        or "specificity" in lowered
    )


def _normalize_lookup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _matches_any_selector(model: str, selectors: list[str]) -> bool:
    model_key = _normalize_lookup_key(model)
    return bool(model_key and any(selector in model_key or model_key in selector for selector in selectors))


def _row_has_lower_scores(row: dict[str, str], baseline: dict[str, str]) -> bool:
    checked = False
    for key in ("Precision", "Domain Specificity", "Number of recalls"):
        row_value = _float_value(row.get(key, ""))
        baseline_value = _float_value(baseline.get(key, ""))
        if row_value is None or baseline_value is None:
            continue
        checked = True
        if row_value >= baseline_value:
            return False
    return checked


def _float_value(value: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", value or "")
    return float(match.group(0)) if match else None


def _extract_table_label(text: str) -> str | None:
    match = re.search(r"\bTable\s*\d+\b", text, re.IGNORECASE)
    return match.group(0) if match else None


def _is_metric_cell(value: str) -> bool:
    return bool(re.fullmatch(r"(f\s*1|f1|auc|precision|recall|accuracy)", value.strip(), re.IGNORECASE))


def _is_separator_row(row: list[str]) -> bool:
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in row)


def _dedupe_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for header in headers:
        clean = re.sub(r"\s+", " ", header.strip())
        if not clean:
            result.append("")
            continue
        counts[clean] = counts.get(clean, 0) + 1
        result.append(clean if counts[clean] == 1 else f"{clean} {counts[clean]}")
    return result
