from __future__ import annotations

import hashlib
import json
import re

from app.services.canonical_models import CanonicalTable


def _normalized_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def table_content_fingerprint(table: CanonicalTable) -> str:
    headers = [_normalized_text(value) for value in table.headers]
    rows = [[_normalized_text(value) for value in row] for row in table.rows]
    cells = [
        {
            "row": cell.row_index,
            "column": cell.column_index,
            "rowspan": cell.rowspan,
            "colspan": cell.colspan,
            "header": cell.is_header,
            "text": _normalized_text(cell.text),
        }
        for cell in sorted(
            table.cells,
            key=lambda item: (item.row_index, item.column_index, item.rowspan, item.colspan),
        )
    ]
    structured = {"headers": headers, "rows": rows, "cells": cells}
    if not headers and not rows and not cells:
        structured = {
            "markdown": _normalized_text(
                table.normalized_markdown or table.source_markdown or table.source_html
            )
        }
    payload = {
        "caption": _normalized_text(table.caption),
        "structured": structured,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def table_has_precise_locator(table: CanonicalTable) -> bool:
    return any((span.normalized_bbox or span.bbox) is not None for span in table.source_spans)


def table_identity_fingerprint(table: CanonicalTable) -> str:
    locators = []
    for span in table.source_spans:
        box = span.normalized_bbox or span.bbox
        locators.append(
            {
                "page_index": span.page_index,
                "bbox": [round(value, 6) for value in box] if box is not None else None,
                "source_block_id": span.source_block_id,
            }
        )
    locators.sort(
        key=lambda item: (
            item["page_index"] if item["page_index"] is not None else 10**9,
            item["bbox"] or [],
            item["source_block_id"] or "",
        )
    )
    payload = {
        "locators": locators,
        "content": table_content_fingerprint(table),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
