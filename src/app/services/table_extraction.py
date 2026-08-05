"""表格结构化与指标提取模块。

基于 :mod:`~app.services.table_normalization` 产出的干净表格文本,
进一步做结构化解构与指标抽取,服务于 RAG 检索层把表格转成可检索、
可引用、可汇总的结构化证据。

对外主要入口:
- :func:`structure_table_markdown`:把一段表格 Markdown 解析为
  :class:`StructuredTable`(含表头、行字典、质量标记);
- :func:`extract_structured_tables`:批量处理解析器输出的表格块列表;
- :func:`table_metric_values`:从表格中提取 (数据集, 指标) 数值,
  用于回答涉及具体指标数值的问题;
- :func:`summarize_ablation_table`:针对消融实验表(ablation)生成
  每组迭代下"全模型 vs 消融变体"的对比结论文本。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from app.services.table_normalization import normalize_table_text


@dataclass
class StructuredTable:
    """结构化表格的内存表示。

    由 :func:`structure_table_markdown` 生成,字段说明:
    - ``label``:表格编号(如 "Table 1"),没有则为 None;
    - ``caption``:题注文本(非表格行拼接而成);
    - ``page_label``:来源页码标签(可能为 None);
    - ``markdown``:规范化后的完整表格 Markdown 原文;
    - ``headers``:合成后的表头列名列表;
    - ``rows``:数据行,每行是一个 {表头: 单元格值} 的字典;
    - ``quality_flags``:质量标记(如 no_data_rows、blank_headers 等)。
    """

    label: str | None
    caption: str
    page_label: str | None
    markdown: str
    headers: list[str] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """把结构化表格序列化为普通字典(用于 JSON 化/持久化)。"""
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
    """把一段表格 Markdown 解析为 :class:`StructuredTable`。

    步骤:
    1. 先用 normalize_table_text 做文本规范化;
    2. 把行分为"表格行(以 | 开头)"与"题注行(其余)";
    3. 从题注中提取表格编号(label);
    4. 解析表格行为行列表,再合成表头、转成行字典;
    5. 计算质量标记。
    """
    # 1. 规范化文本,并过滤空行。
    normalized = normalize_table_text(markdown)
    lines = [line.rstrip() for line in normalized.splitlines() if line.strip()]
    caption_lines: list[str] = []
    table_lines: list[str] = []
    # 2. 按是否以 | 开头区分表格行与题注行。
    for line in lines:
        if line.strip().startswith("|"):
            table_lines.append(line)
        else:
            caption_lines.append(line.strip())

    # 3. 题注拼接 + 提取表格编号。
    caption = " ".join(caption_lines).strip()
    label = _extract_table_label(caption or normalized)
    # 4. 解析表格行为 [[cell, ...], ...],再合成表头与行字典。
    rows = _markdown_table_rows("\n".join(table_lines))
    headers, row_dicts, consumed_header_like = _rows_to_dicts(rows)
    # 5. 计算质量标记。
    quality_flags = _quality_flags(normalized, headers, row_dicts)
    if consumed_header_like:
        quality_flags.append("header_like_data_row")
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
    """批量处理解析器输出的表格块列表,返回结构化表格字典列表。

    对每个块,跳过非字典条目或缺少 markdown 文本的块,然后调用
    :func:`structure_table_markdown` 并转成字典。
    """
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
    """从表格中提取 (数据集, 指标) 数值。

    参数:
    - ``requested_datasets``:只返回这些数据集的指标(按模糊键匹配);
    - ``row_selectors`` / ``requested_subjects``:只保留模型/主体与这些
      选择器模糊匹配的数据行。

    返回值:候选字典列表,每个包含 table_label、dataset、values(指标:
    数值的映射)与 model(行所属模型/方法,可能为空字符串)。
    """
    table = structure_table_markdown(table_markdown)
    # 把请求的数据集与选择器归一化为"只保留字母数字"的模糊键。
    requested = {_normalize_lookup_key(dataset) for dataset in requested_datasets or []}
    selectors = [_normalize_lookup_key(selector) for selector in [*(row_selectors or []), *(requested_subjects or [])] if selector]
    # 从表头中推断该表包含哪些数据集列。
    datasets = _dataset_names_from_headers(table.headers)
    candidates: list[dict] = []
    for row in table.rows:
        # 识别该行属于哪个模型/方法;若给了选择器则做模糊过滤。
        model = _row_model(row)
        if selectors and not _matches_any_selector(model, selectors):
            continue
        for dataset in datasets:
            dataset_key = _normalize_lookup_key(dataset)
            # 只提取被请求的数据集。
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
    """针对消融实验表生成每组迭代的对比结论文本。

    表格按迭代(Iteration rounds / Iteration)分组,对每组:
    - 挑出"全模型"(未消融)行,提取其 recalls / precision /
      domain specificity 数值;
    - 找出所有分数低于全模型的消融变体;
    - 生成两条人类可读的结论句。

    返回一个结论句列表。
    """
    table = structure_table_markdown(table_markdown)
    # 按迭代值把数据行分组。
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in table.rows:
        iteration = row.get("Iteration rounds") or row.get("Iteration") or ""
        model = _row_model(row)
        if not iteration or not model:
            continue
        grouped.setdefault(iteration, []).append(row)

    findings: list[str] = []
    for iteration, rows in grouped.items():
        # 选出该迭代下的"全模型"行。
        full = _select_full_model_row(rows)
        if not full:
            continue
        full_model = _row_model(full)
        # 提取全模型的关键指标。
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
        # 结论一:全模型报告的指标。
        if parts:
            findings.append(f"{iteration}: full {full_model} reports " + ", ".join(parts) + ".")
        # 结论二:表现低于全模型的消融变体(最多列前 4 个)。
        worse = [
            _row_model(row)
            for row in rows
            if row is not full and _row_has_lower_scores(row, full)
        ]
        if worse:
            findings.append(f"{iteration}: ablated variants underperform the full model, including " + ", ".join(worse[:4]) + ".")
    return findings


def _rows_to_dicts(
    rows: list[list[str]],
) -> tuple[list[str], list[dict[str, str]], bool]:
    """把"行列表"([[cell, ...], ...])转成(表头列表, 行字典列表, 是否丢弃了语义重复表头)。

    处理逻辑:
    - 用 _header_index 定位表头行;
    - 用 _consume_headers 合成任意深度的续行表头,并跳过被消费的续行;
    - 跳过表头行与分隔行,把其余行补齐宽度后按表头转为字典。
    第三个返回值表示是否消费过语义续行表头(合入或丢弃,用于
    header_like_data_row 标记;指标行不计入)。
    """
    if not rows:
        return [], [], False
    header_index = _header_index(rows)
    headers, data_start, consumed_header_like = _consume_headers(rows, header_index)
    headers = _dedupe_headers(headers)
    body_rows = rows[data_start:]
    # 过滤分隔行。
    body_rows = [row for row in body_rows if not _is_separator_row(row)]
    dicts: list[dict[str, str]] = []
    for row in body_rows:
        # 整行空白则跳过。
        if not any(cell.strip() for cell in row):
            continue
        # 行短于表头时用空串补齐,保证按表头索引不越界。
        padded = row + [""] * max(0, len(headers) - len(row))
        record: dict[str, str] = {}
        for index in range(len(headers)):
            cell = padded[index].strip()
            if not headers[index]:
                # 保留首列空表头下的单元格作为行标签来源。许多发表表格把
                # 行标签列留空表头(如属性表),丢掉首列会让后续行标签缺失。
                if index == 0 and cell:
                    record[""] = cell
                continue
            record[headers[index]] = cell
        if record:
            dicts.append(record)
    return headers, dicts, consumed_header_like


def _consume_headers(
    rows: list[list[str]], header_index: int
) -> tuple[list[str], int, bool]:
    """合成所有续行表头,并返回数据行起始下标。

    返回 (headers, data_start, consumed_header_like)。主表头之后连续出现的
    "语义续行表头"按列前向填充逐层合并,或作为冗余语义重复表头被丢弃,
    与 canonical 组装路径的分类器保持一致;任一语义续行被消费都会使第三
    个返回值为 True,供调用方标记 ``header_like_data_row``。指标行(>= 2
    个指标单元格)仍按既有指标路径单独合成一层,不计入该标记。
    """
    headers = list(rows[header_index])
    index = header_index + 1
    consumed_header_like = False
    while index < len(rows):
        candidate = rows[index]
        if _is_metric_row(candidate):
            headers = _compose_metric_headers(headers, candidate)
            index += 1
            break
        body = rows[index + 1 :]
        kind = _classify_direct_header_row(headers, candidate, body)
        if kind is None:
            break
        consumed_header_like = True
        if kind == "compose":
            headers = _compose_secondary_headers(headers, candidate)
        index += 1
    return headers, index, consumed_header_like


def _compose_metric_headers(header: list[str], next_row: list[str]) -> list[str]:
    """把指标行合并进表头(如 "NYT" + "F1" -> "NYT F1")。"""
    width = max(len(header), len(next_row))
    composed: list[str] = []
    for index in range(width):
        top = header[index].strip() if index < len(header) else ""
        bottom = next_row[index].strip() if index < len(next_row) else ""
        # 上下两层都有值且下层是指标:合成 "上层 指标"。
        if top and bottom and _is_metric_cell(bottom):
            composed.append(f"{top} {bottom}")
        else:
            # 否则取任一非空值(优先上层)。
            composed.append(top or bottom)
    return composed


def _classify_direct_header_row(
    header: list[str], candidate: list[str], body: list[list[str]]
) -> str | None:
    """基于候选行下方的数据行,判断它是 compose / drop 还是数据行。"""
    body_numeric_columns = {
        index
        for row in body
        for index, cell in enumerate(row)
        if _is_numeric_cell(cell)
    }
    return _classify_semantic_header_row(header, candidate, body_numeric_columns)


def _header_index(rows: list[list[str]]) -> int:
    """定位表头所在行。

    优先找包含 "model"/"variant" 关键词的行,其次找包含 "iteration"
    关键词的行;都没找到时回退到第 0 行。
    """
    for index, row in enumerate(rows):
        lowered = [cell.lower() for cell in row]
        if any(cell in {"model", "variant"} or "model" in cell for cell in lowered):
            return index
        if any("iteration" in cell for cell in lowered):
            return index
    return 0


def _markdown_table_rows(text: str) -> list[list[str]]:
    """把表格 Markdown 文本解析为行列表(跳过分隔行)。

    只处理以 | 开头且后续还有 | 的行;去掉首尾竖线后按 | 切分为单元格,
    并跳过 Markdown 分隔行(|---|)。
    """
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
    """根据表格内容计算一组质量标记(供下游/诊断使用)。

    可能的标记:
    - ``no_data_rows``:没有任何数据行;
    - ``contains_latex_markup``:还残留 \\mathbf/\\mathrm 等 LaTeX 标记;
    - ``blank_headers``:存在空表头;
    - ``caption_without_rows``:有表格编号但没有数据行(疑似只有题注)。
    """
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
    """从表头列表提取去重后的数据集名列表。

    对每个表头用 _dataset_name_from_header 推断其所属数据集(如
    "NYT F1" -> "NYT"),并按大小写不敏感去重。
    """
    datasets: list[str] = []
    for header in headers:
        dataset = _dataset_name_from_header(header)
        if dataset and dataset.upper() not in {item.upper() for item in datasets}:
            datasets.append(dataset)
    return datasets


def _dataset_metric_values(row: dict[str, str], dataset: str) -> dict[str, str]:
    """从一行数据中提取指定数据集各指标的数值。

    对每个指标(F1/AUC/Precision/Recall/Accuracy),查找形如
    "{dataset} {metric}" 的列(模糊键匹配);仅当单元格包含数字时
    才记录该指标。返回 {指标名: 原始值文本}。
    """
    values: dict[str, str] = {}
    # 先把整行键归一化为模糊键,便于忽略大小写/空格差异的查找。
    lookup = {_normalize_lookup_key(key): value for key, value in row.items()}
    for metric in ("F1", "AUC", "Precision", "Recall", "Accuracy"):
        value = lookup.get(_normalize_lookup_key(f"{dataset} {metric}"))
        # 只保留确实包含数字的值。
        if value and re.search(r"\d+(?:\.\d+)?", value):
            values[metric] = value
    return values


def _row_model(row: dict[str, str]) -> str:
    """识别一行数据属于哪个模型/方法。

    优先看 "Model"/"Variant"/"Method"/"Approach"/"System" 列;
    否则扫描所有列,返回第一个"非指标、且不含数字"的值(即像模型名
    的文本);都找不到时返回空字符串。
    """
    for key in ("Model", "Variant", "Method", "Approach", "System"):
        if value := row.get(key):
            return value
    for key, value in row.items():
        if value and not _is_metric_header(key) and not re.search(r"\d+(?:\.\d+)?", value):
            return value
    return ""


def _dataset_name_from_header(header: str) -> str:
    """从单个表头推断其所属数据集名。

    表头形如 "NYT F1"、"WEB Precision 2" 时,去掉末尾的指标名
    (及可选序号)后剩余部分即数据集名;若剩余部分是 model/variant 等
    通用词,则不视为数据集,返回空字符串。
    """
    clean = re.sub(r"\s+", " ", header.strip())
    for metric in ("F1", "AUC", "Precision", "Recall", "Accuracy"):
        # 匹配 "数据集名 + 指标名(可带序号)" 的表头。
        match = re.match(rf"(.+?)\s+{re.escape(metric)}(?:\s+\d+)?$", clean, re.IGNORECASE)
        if match:
            dataset = match.group(1).strip()
            if dataset and dataset.lower() not in {"model", "variant", "method", "approach", "system"}:
                return dataset
    return ""


def _select_full_model_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """在一组数据行中选出"全模型(未消融)"行。

    优先选择模型名看起来未消融(不含 w/o、without、ablated 等)的行;
    若没有则退而选任意有模型名的行。最终在候选中挑指标数值和最大的行。
    """
    full_rows = [row for row in rows if _row_model(row) and not _looks_ablated(_row_model(row))]
    if not full_rows:
        full_rows = [row for row in rows if _row_model(row)]
    if not full_rows:
        return None
    return max(full_rows, key=_numeric_score_sum)


def _looks_ablated(model: str) -> bool:
    """判断模型名是否暗示这是"消融变体"。

    命中 "w/o"、"without"、"ablated"、"removed"、"no xxx" 等模式时
    视为消融变体,与全模型区分开。
    """
    lowered = model.lower()
    return bool(
        re.search(r"\bw\s*/\s*o\b", lowered)
        or re.search(r"\bwithout\b", lowered)
        or re.search(r"\bablated?\b", lowered)
        or re.search(r"\bremoved?\b", lowered)
        or re.search(r"\bno\s+\w+", lowered)
    )


def _numeric_score_sum(row: dict[str, str]) -> float:
    """计算一行中所有指标数值的总和(用于比较行优劣)。

    只累加指标列(_is_metric_header)中能解析出数字的值;
    若没有任何指标数值则返回负无穷,表示该行不参与比较。
    """
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
    """判断表头/列名是否属于指标列(F1/AUC/Precision 等)。

    采用子串包含判断,兼容 "NYT F1"、"Precision (2)" 等带前缀/后缀
    的写法。
    """
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
    """把文本归一化为"只含小写字母和数字"的模糊匹配键。

    例如 "NYT F1" -> "nyt f1" -> "nytf1",用于忽略大小写、空格和
    标点差异的模糊匹配。
    """
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _matches_any_selector(model: str, selectors: list[str]) -> bool:
    """判断模型名是否与任一选择器模糊匹配。

    归一化后做双向子串判断(选择器是模型名的子串,或模型名是选择器的
    子串),只要命中一个选择器即返回 True。
    """
    model_key = _normalize_lookup_key(model)
    return bool(model_key and any(selector in model_key or model_key in selector for selector in selectors))


def _row_has_lower_scores(row: dict[str, str], baseline: dict[str, str]) -> bool:
    """判断某行在关键指标上是否"全面低于"基线行。

    对 Precision / Domain Specificity / Number of recalls 三个指标:
    只要有一项可比较且不低于基线,就返回 False;若至少比较了一项且
    所有可比较项都更低,才返回 True。
    """
    checked = False
    for key in ("Precision", "Domain Specificity", "Number of recalls"):
        row_value = _float_value(row.get(key, ""))
        baseline_value = _float_value(baseline.get(key, ""))
        # 任一侧缺失数值则跳过该指标。
        if row_value is None or baseline_value is None:
            continue
        checked = True
        # 只要有一项不低于基线,就不算"全面更低"。
        if row_value >= baseline_value:
            return False
    return checked


def _float_value(value: str) -> float | None:
    """从文本中提取第一个数值并转为 float,没有数值则返回 None。

    兼容带符号("-3.2")、带百分号("92.1%")等情况——只取数字部分。
    """
    match = re.search(r"-?\d+(?:\.\d+)?", value or "")
    return float(match.group(0)) if match else None


def _extract_table_label(text: str) -> str | None:
    """从题注/文本中提取表格编号,如 "Table 1"。

    匹配 "Table" + 数字;没有则返回 None。
    """
    match = re.search(r"\bTable\s*\d+\b", text, re.IGNORECASE)
    return match.group(0) if match else None


def _is_metric_cell(value: str) -> bool:
    """判断单元格文本是否恰好是一个指标名(F1/AUC/Precision 等)。"""
    return bool(re.fullmatch(r"(f\s*1|f1|auc|precision|recall|accuracy)", value.strip(), re.IGNORECASE))


def _is_separator_row(row: list[str]) -> bool:
    """判断一行是否为 Markdown 分隔行(如 |---|---|)。"""
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in row)


_NUMERIC_CELL_TOKEN_RE = r"[-+]?\d+(?:\.\d+)?%?"


def _is_numeric_cell(value: str) -> bool:
    """判断单元格文本是否整体是一个数值(可含不确定度/百分号)。

    兼容普通数字、带符号值、百分号与 ``±`` 不确定度对,并识别 OCR/LaTeX
    压平后的数值单元格:不确定度标记丢失(``2.0 0.2``)或符号与数字被拆开
    (``+ 0.9 0.2``)。只要出现一个非数值 token 即判为非数值。
    """
    text = re.sub(r"\s*(?:±|\+/-)\s*", " ", str(value or "").strip())
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


def _is_metric_row(row: list[str]) -> bool:
    """判断一行是否为"指标行"(>= 2 个指标单元格)。"""
    return sum(1 for cell in row if _is_metric_cell(cell)) >= 2


def _classify_semantic_header_row(
    header: Sequence[str],
    candidate: Sequence[str],
    body_numeric_columns: set[int],
) -> str | None:
    """把一个续行分类为 compose / drop / 数据行(与 canonical 组装共用)。

    判定顺序:分隔行 -> 完全重复表头 -> 语义表头行 -> 数据行。候选行至少
    含两个非数值单元格,并与主表头首列对齐时才可能是语义表头行。它在下述
    情形下被合入表头(compose):
    - 细分了重复的组(如 ``OPLS4 | OPLS4`` 下的 ``Edgewise | Pairwise``);
    - 填补了空表头单元格,并且细分了既有列、重复了组值,或所填补的列在
      下方确有数值支撑。
    只细分互异列、且每个细分列下方都有数值的行,作为冗余语义重复表头被
    丢弃(drop);全文本行保留为数据,避免文本表格丢行。
    """
    if len(header) != len(candidate):
        return None
    filled = [cell for cell in candidate if cell.strip()]
    if len(filled) < 2:
        return None
    if any(_is_numeric_cell(cell) for cell in filled):
        return None

    header_first = _normalize_lookup_key(header[0]) if header and header[0] else ""
    candidate_first = _normalize_lookup_key(candidate[0]) if candidate and candidate[0] else ""
    if header_first and candidate_first and candidate_first != header_first:
        return None

    fills = False
    refines_repeat = False
    candidate_repeats = False
    refined_distinct: list[int] = []
    filled_columns: list[int] = []
    for index in range(1, len(header)):
        top = header[index]
        bottom = candidate[index]
        if bottom and not top:
            fills = True
            filled_columns.append(index)
            continue
        if not bottom or not top:
            continue
        top_key = _normalize_lookup_key(top)
        bottom_key = _normalize_lookup_key(bottom)
        if top_key == bottom_key:
            continue
        previous_key = _normalize_lookup_key(header[index - 1]) if header[index - 1] else ""
        if top_key == previous_key:
            refines_repeat = True
        else:
            refined_distinct.append(index)
    for index in range(1, len(candidate)):
        if candidate[index] and candidate[index - 1] and _normalize_lookup_key(
            candidate[index]
        ) == _normalize_lookup_key(candidate[index - 1]):
            candidate_repeats = True

    if refines_repeat:
        return "compose"
    if fills:
        if refined_distinct or candidate_repeats:
            return "compose"
        if any(index in body_numeric_columns for index in filled_columns):
            return "compose"
        return None
    if refined_distinct and all(index in body_numeric_columns for index in refined_distinct):
        return "drop"
    return None


def _compose_secondary_headers(header: list[str], secondary: list[str]) -> list[str]:
    """把语义第二层表头按列前向填充合并进主表头。"""
    width = max(len(header), len(secondary))
    composed: list[str] = []
    group = ""
    for index in range(width):
        top = header[index].strip() if index < len(header) else ""
        bottom = secondary[index].strip() if index < len(secondary) else ""
        if top:
            group = top
        if not bottom:
            composed.append(top)
        elif group and _normalize_lookup_key(group) != _normalize_lookup_key(bottom):
            composed.append(f"{group} {bottom}")
        else:
            composed.append(group or bottom)
    return composed


def _dedupe_headers(headers: list[str]) -> list[str]:
    """对重复的表头列名做去重,重复项追加序号后缀。

    例如 ["NYT", "NYT"] -> ["NYT", "NYT 2"],避免后续按表头建立字典
    时发生键冲突。空表头保留为空字符串。
    """
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
