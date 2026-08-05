"""表格身份指纹（Table Identity Fingerprint）工具。

在跨页表格合并、修复、去重等场景中，需要一种稳定的方式来判断两个表格
是否 "相同"。本模块提供两种指纹：

1. table_content_fingerprint(table): 基于表格内容的指纹，用于判断两份
   表格数据是否语义等价（忽略空白、格式）。
2. table_identity_fingerprint(table): 基于内容 + 来源定位信息的指纹，
   用于判断是否在原文档中指向同一个表格（例如跨页合并后的表格）。

主要入口：
- table_content_fingerprint(table): 内容指纹。
- table_identity_fingerprint(table): 身份指纹（内容 + 来源位置）。
- table_has_precise_locator(table): 判断表格是否有精确的页面/ bbox 定位。
"""

from __future__ import annotations

import hashlib
import json
import re

from app.services.canonical_models import CanonicalTable


def _normalized_text(value: str | None) -> str:
    """把文本归一化：所有空白压缩为单个空格并去除首尾空白。"""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def table_content_fingerprint(table: CanonicalTable) -> str:
    """计算表格的内容指纹。

    指纹包含：标题（caption）、表头、行数据、单元格结构（含 rowspan/colspan、
    是否表头）。如果结构化数据为空，则退而使用 markdown / html 文本。
    使用 SHA-256 对规范化后的 JSON 进行哈希。
    """
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
    """判断表格是否有精确的来源定位（bbox 或归一化 bbox）。"""
    return any((span.normalized_bbox or span.bbox) is not None for span in table.source_spans)


def table_identity_fingerprint(table: CanonicalTable) -> str:
    """计算表格的身份指纹（内容 + 来源位置）。

    用于判断两个表格是否指向原文档中的同一个物理表格。指纹包含：
    - 每个 source_span 的 page_index、bbox（精度到 6 位小数）、source_block_id。
    - 内容指纹。
    """
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
