"""表格文本规范化模块。

负责把 MinerU 抽取出的表格 Markdown 文本规整为标准形式,同时保留
表格的形状(表头、行、列结构)。这是 RAG 检索层处理表格证据的第一环,
为下游的表格结构化与指标提取提供干净、一致的输入。

核心功能:
- :func:`normalize_table_text`:整体规范化一段表格 Markdown——逐行规整
  表格行、修复行内单元格、补齐缺失列,非表格行(如题注)也做单元格级
  规范化;
- :func:`normalize_table_cell`:单个单元格的文本规范化(LaTeX 命令清理、
  OCR 数字碎片拼接、模型名压缩、空白折叠等);
- :func:`normalize_fragmented_numeric_spacing`:专门修复 MinerU/OCR 把
  一个数字拆成多段的问题;
- 各类内部"修复"函数(_repair_*)针对常见真实问题做定向处理:OCR 残留
  单元格、数据集-指标复合表头、迭代(iteration)跨行(rowspan)等。

本模块强调"只改文本、不改结构"。
"""

from __future__ import annotations

import re


# 用于识别常见数据集名称的正则(在表头修复中用于区分"数据集"与"指标")。
_DATASET_NAME_RE = re.compile(
    r"\b(OIE2016|NYT|PENN|WEB|CoNLL|ACE|SemEval|WikiSQL|SQuAD|GLUE|SuperGLUE)\b",
    re.IGNORECASE,
)
# 匹配 LaTeX 文本命令,如 \\mathbf{x},\\text{...} 等,后续会把花括号内
# 的内容压缩成紧凑形式。
_LATEX_TEXT_COMMAND_RE = re.compile(
    r"\\(?:mathbf|mathrm|mathit|text|operatorname)\s*\{\s*([^{}]*?)\s*\}"
)


def normalize_table_text(text: str) -> str:
    """规范化 MinerU 表格 Markdown,同时保留表格形状。

    逐行扫描输入文本:
    - 以 ``|`` 开头且后续还有 ``|`` 的行视为表格行,收集到 table_rows 中;
    - 其他行(题注、说明等)触发"冲刷":先修复并输出已收集的表格行,
      再对非表格行做单元格级规范化。

    表格行按批(_repair_markdown_table_rows)做跨行级修复,保证每行宽度
    一致,最后统一用 ``| a | b |`` 形式重新输出。
    """
    lines: list[str] = []
    table_rows: list[list[str]] = []

    def flush_table_rows() -> None:
        """把当前累积的表格行批修复后写入输出,并清空累积区。"""
        nonlocal table_rows
        if not table_rows:
            return
        for row in _repair_markdown_table_rows(table_rows):
            lines.append("| " + " | ".join(row) + " |")
        table_rows = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            # 去掉首尾竖线后按竖线切分为单元格,并逐个做单元格规范化。
            cells = [normalize_table_cell(cell) for cell in stripped.strip("|").split("|")]
            table_rows.append(cells)
        else:
            # 遇到非表格行:先把前面累积的表格行批冲刷出来。
            flush_table_rows()
            lines.append(normalize_table_cell(line))
    # 文本结束时可能还有未冲刷的表格行。
    flush_table_rows()
    return "\n".join(lines).strip()


def normalize_table_cell(cell: str) -> str:
    """规范化单个表格单元格的文本。

    处理顺序大致为:
    1. 清理明显的 LaTeX 残留(\\text{-}、\\-、\\mathbf{...} 等);
    2. 压缩常见模型名(如 "w / o"、"C 6" 这类被拆散的写法);
    3. 修复 OCR 数字碎片(如 "0. 1" -> "0.1");
    4. 压缩空白并去除首尾空白。
    该函数只处理文本,不涉及表格结构。
    """
    text = str(cell or "").strip()
    if not text:
        return ""

    # 先替换常见的 "-" 写法并去掉首尾的 $ 符号。
    text = text.replace("\\text{-}", "-").replace("\\-", "-").strip("$")
    # 反复折叠 LaTeX 文本命令(可能嵌套,因此循环直到稳定)。
    previous = None
    while previous != text:
        previous = text
        text = _LATEX_TEXT_COMMAND_RE.sub(lambda match: _compact_latex_group(match.group(1)), text)

    # 下标:_{...} 处理成 " 内容";上标:^{...} 处理成 "^内容"。
    text = re.sub(r"_\s*\{\s*([^{}]*?)\s*\}", lambda match: " " + _compact_latex_group(match.group(1)), text)
    text = re.sub(r"\^\s*\{\s*([^{}]*?)\s*\}", lambda match: "^" + _compact_latex_group(match.group(1)), text)
    # 去掉 LaTeX 命令前缀的反斜杠(如 \\alpha -> alpha),再清理命令本身。
    text = re.sub(r"\\([A-Za-z])", r"\1", text)
    text = re.sub(r"\\(?:quad|,|;|!|\s)", " ", text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    # 去掉残留的花括号。
    text = text.replace("{", "").replace("}", "")
    # 把 "mathbb chi1"/"chi 1" 等写法统一为希腊字母 χ1/χ2。
    text = re.sub(r"\b(?:mathbb|mathrm|mathit|mathsf)\s+chi\s*([12])\b", r"χ\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bchi\s*([12])\b", r"χ\1", text, flags=re.IGNORECASE)
    # 在氨基酸/扭转角等上下文中,把 OCR 误识的 "lle" 修正为 "Ile"。
    if re.search(r"\b(?:theta|angle|torsion|chi\s*[12]|χ[12]|Leu|Asp|Asn|Val|Thr|Phe|Tyr)\b", text, flags=re.IGNORECASE):
        text = re.sub(r"\blle\b", "Ile", text)
    # 压缩常见模型名、修复 OCR 数字间距、把带空白的斜杠去掉空白。
    text = _compact_common_model_names(text)
    text = _repair_ocr_numeric_spacing(text)
    text = re.sub(r"\s*/\s*", "/", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_fragmented_numeric_spacing(text: str) -> str:
    """拼接 MinerU/OCR 拆散的单个数字,而不改写周围的 LaTeX。

    处理几类典型碎片:小数点两侧的空格("0 . 1")、数字与小数点的分离
    ("0. 1 2")、以及纯数字段被空格拆开("1 2 3" 被当作 "1.23" 拼接)。
    只作用于"看起来像数字"的片段,不影响周围的公式与文字。
    """
    # 数字与小数点之间的空白:0 . 1 -> 0.1
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    # "x y. z" 四段式:形如 1 2. 3 -> 12.3
    text = re.sub(r"(?<![\d.])(\d)\s+(\d)\.(\d)\s+(\d)(?![\d.])", r"\1\2.\3\4", text)
    # 三段式:形如 1 2.34 -> 12.34(后随 ±、+/-、行尾或非数字)
    text = re.sub(r"(?<![\d.])(\d)\s+(\d)\.(\d+)(?=\s*(?:\\pm|\+/-|$|[^\d.]))", r"\1\2.\3", text)
    # "1.2 3" -> 1.23
    text = re.sub(r"(?<![\d.])(\d)\.(\d)\s+(\d)(?![\d.])", r"\1.\2\3", text)
    # 两位数字被拆开:"1 23" -> "1.23"(类似 1.23 被 OCR 分行的修复)
    text = re.sub(r"(?<!\d)(\d)\s+(\d{2})(?!\d)", r"\1.\2", text)

    def join_single_digit_run(match: re.Match[str]) -> str:
        # 把一段连续的数字+空格(如 "1 2 3")直接拼起来。
        return re.sub(r"\s+", "", match.group(0))

    return re.sub(r"(?<![\d.])\d(?:\s+\d)+(?![\d.])", join_single_digit_run, text)


def _repair_ocr_numeric_spacing(text: str) -> str:
    """修复 OCR 数值间距(内部辅助函数)。

    与 :func:`normalize_fragmented_numeric_spacing` 逻辑一致,但多一步把
    文本形式的 "pm" 替换为 ± 符号。主要用于单元格级规范化流程。
    """
    # 把 "pm" 归一为 ± 符号(±1 这样的常见写法)。
    text = re.sub(r"\bpm\b", "±", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"(?<![\d.])(\d)\s+(\d)\.(\d)\s+(\d)(?![\d.])", r"\1\2.\3\4", text)
    text = re.sub(r"(?<![\d.])(\d)\s+(\d)\.(\d+)(?=\s*(?:±|\+/-|$|[^\d.]))", r"\1\2.\3", text)
    text = re.sub(r"(?<![\d.])(\d)\.(\d)\s+(\d)(?![\d.])", r"\1.\2\3", text)
    text = re.sub(r"(?<!\d)(\d)\s+(\d{2})(?!\d)", r"\1.\2", text)

    def join_single_digit_run(match: re.Match[str]) -> str:
        return re.sub(r"\s+", "", match.group(0))

    return re.sub(r"(?<![\d.])\d(?:\s+\d)+(?![\d.])", join_single_digit_run, text)


def _compact_latex_group(value: str) -> str:
    """压缩 LaTeX 命令花括号内的内容。

    去掉反斜杠命令前缀并折叠空白;如果内容全部是单字符或符号(如
    字母缩写、"/"、"-"),则直接拼成一个紧凑字符串;否则交给
    _compact_common_model_names 做模型名压缩。
    """
    text = re.sub(r"\\([A-Za-z])", r"\1", value)
    text = re.sub(r"\s+", " ", text.strip())
    tokens = text.split()
    if tokens and all(len(token) == 1 or token in {"/", "-"} for token in tokens):
        return "".join(tokens)
    return _compact_common_model_names(text)


def _compact_common_model_names(text: str) -> str:
    """压缩常见模型名/缩写的松散写法。

    处理两类常见问题:
    - 被空格拆开的连续大写字母段(如 "A B C" -> "ABC");
    - "C 6"/"C 12" 这类被拆开的模型名;
    - "w / o XXX" -> "w/o XXX"(without 的缩写)。
    """
    text = _compact_spaced_uppercase_runs(text)
    text = re.sub(r"\bC\s+(6|12)\b", r"C\1", text)
    text = re.sub(
        r"\bw\s*/\s*o\s+((?:[A-Za-z]\s*){2,})",
        lambda match: "w/o " + re.sub(r"\s+", "", match.group(1)),
        text,
        flags=re.IGNORECASE,
    )
    return text


def _compact_spaced_uppercase_runs(text: str) -> str:
    """把被空格拆开的连续大写字母串合并(仅当合并后长度 >= 3)。

    例如 "B E R T" -> "BERT";如果合并后不足 3 个字符则保持原样,
    避免误伤正常的单词间空格。
    """
    def replace(match: re.Match[str]) -> str:
        compact = re.sub(r"\s+", "", match.group(0))
        return compact if len(compact) >= 3 else match.group(0)

    text = re.sub(r"\b[A-Z](?:\s+[A-Z]){2,}(?:\s*-\s*[A-Z](?:\s+[A-Z])*)+\b", replace, text)
    return re.sub(r"\b(?:[A-Z]\s+){2,}[A-Z]\b", replace, text)


def _repair_markdown_table_rows(rows: list[list[str]]) -> list[list[str]]:
    """对一批表格行做跨行级修复,并统一所有行的列宽。

    依次执行:
    - _repair_residue_ocr_cells:修复 OCR 把 "Ile" 识别成 "lle" 等残留;
    - _repair_dataset_metric_header:修复"数据集 x 指标"复合表头;
    - _repair_iteration_rowspans:修复迭代行(iteration)跨行省略。
    最后以最宽行宽度为准,为较短行补齐空单元格,保证矩形结构。
    """
    rows = [list(row) for row in rows]
    if not rows:
        return rows
    _repair_residue_ocr_cells(rows)
    _repair_dataset_metric_header(rows)
    _repair_iteration_rowspans(rows)
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def _repair_residue_ocr_cells(rows: list[list[str]]) -> None:
    """修复氨基酸残基/扭转角表格中 OCR 把 "Ile" 误识成 "lle" 的问题。

    仅当表的前 3 行同时出现"残基(residue)"上下文和"角度/扭转角"上下文
    时才触发,避免误伤其他 "lle" 文本。
    """
    # 取表头区域文本,判断是否具备"残基 + 角度"的上下文。
    table_context = " ".join(cell for row in rows[:3] for cell in row)
    has_residue_angle_context = bool(re.search(r"\b(?:res\.?|residue|amino acid)\b", table_context, flags=re.IGNORECASE)) and bool(
        re.search(r"\b(?:angle|theta|torsion|chi\s*[12]|χ[12]|alpha|beta|gamma)\b", table_context, flags=re.IGNORECASE)
    )
    if not has_residue_angle_context:
        return
    # 把单元格中恰好等于 "lle"(忽略大小写)的修正为 "Ile"。
    for row in rows:
        for index, cell in enumerate(row):
            if re.fullmatch(r"lle", cell.strip(), flags=re.IGNORECASE):
                row[index] = "Ile"


def _repair_dataset_metric_header(rows: list[list[str]]) -> None:
    """修复"数据集 x 指标"复合表头的错位。

    典型场景:表头有两行——第一行是数据集名,第二行是若干指标(F1、
    Accuracy 等),MinerU/OCR 可能把数据集名只写一次而不是按指标重复。
    本函数会找出"指标行"(包含 >= 4 个指标单元格的行)与其上方最近
    的数据集行,然后把数据集名按指标数量展开填充。

    例如:
    |     | NYT |     |
    |     | F1  | AUC |
    会被修复为:
    |     | NYT | NYT |
    |     | F1  | AUC |
    """
    # 找到首个"指标数量 >= 4"的行,视为指标行。
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
    metric_row = rows[metric_index]
    # 指标数量 = 指标行非首列的列数(去掉行首的空白/标签列)。
    metric_count = max(len(metric_row) - 1, 0)
    if metric_count <= 0:
        return
    # 向上寻找最近的"数据集行":该行能解析出与指标数整除的数据集名。
    dataset_index = next(
        (
            index
            for index in range(metric_index - 1, -1, -1)
            if _dataset_names_from_header_row(rows[index], metric_count)
        ),
        None,
    )
    if dataset_index is None:
        return

    dataset_row = rows[dataset_index]
    dataset_names = _dataset_names_from_header_row(dataset_row, metric_count)
    # 数据集名无法解析,或指标数不能被数据集数整除,则放弃修复。
    if not dataset_names or metric_count % len(dataset_names) != 0:
        return
    # 若数据集行已经按指标数完整展开(首列之外都非空),则无需修复。
    if len(dataset_names) == metric_count and all(dataset_row[index + 1].strip() for index in range(metric_count)):
        return

    # 每个数据集名重复 group_size 次,展开成与指标一一对应的表头。
    group_size = metric_count // len(dataset_names)
    rows[dataset_index] = [dataset_row[0] if dataset_row else "", *[name for name in dataset_names for _ in range(group_size)]]


def _dataset_names_from_header_row(row: list[str], metric_count: int) -> list[str]:
    """从表头行解析出数据集名列表。

    跳过空单元格、指标单元格、分隔符和纯数字单元格;若某个单元格匹配
    已知数据集名(_DATASET_NAME_RE)则用规范名(大写),否则用原始文本。
    只有解析出的数据集数能整除 metric_count 时才返回列表,否则返回空,
    表示该行不适合作为数据集行。
    """
    names: list[str] = []
    seen: set[str] = set()
    for cell in row[1:]:
        clean = re.sub(r"\s+", " ", cell.strip())
        # 跳过空值、指标名、分隔符与纯数字。
        if not clean or _is_metric_cell(clean) or _is_separator_row([clean]):
            continue
        if re.fullmatch(r"-?\d+(?:\.\d+)?%?", clean):
            continue
        # 已知数据集名用匹配到的规范名,否则保留清洗后的原始文本。
        match = _DATASET_NAME_RE.search(clean)
        name = match.group(0).upper() if match else clean
        key = name.lower()
        if key not in seen:
            names.append(name)
            seen.add(key)
    # 数据集数必须能整除指标数,否则放弃。
    if not names or metric_count % len(names) != 0:
        return []
    return names


def _repair_iteration_rowspans(rows: list[list[str]]) -> None:
    """修复"迭代(iteration)行跨行"造成的模型名缺行。

    典型表格里 iteration 值只出现在该组第一行,后续行省略。本函数:
    - 仅在首行表头为 ["iteration", "model", ...] 时触发;
    - 逐行扫描,遇到 "iteration ..." 行则记住当前迭代值;
    - 后续某行首列看起来像模型名时,把当前迭代值插入该行首列。
    """
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
        # 遇到迭代标记行("iteration 1" 等),更新当前迭代值并跳过。
        if first.lower().startswith("iteration"):
            current_iteration = first
            continue
        # 若该行首列看似模型名,说明迭代值被省略,补上当前迭代值。
        if current_iteration and _looks_like_model_name(first):
            repaired = [current_iteration, *row]
            # 若补位后超宽且末尾为空单元格,则去掉末尾以对齐。
            if len(repaired) > width and repaired[-1] == "":
                repaired = repaired[:-1]
            rows[index] = repaired[:width]


def _is_metric_cell(value: str) -> bool:
    """判断单元格是否是一个指标名(F1/AUC/Precision/Recall/Accuracy)。"""
    lowered = value.lower()
    return bool(re.fullmatch(r"(f\s*1|f1|auc|precision|recall|accuracy)", lowered))


def _is_separator_row(row: list[str]) -> bool:
    """判断一行是否为 Markdown 分隔行(如 |---|---|)。"""
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in row)


def _looks_like_model_name(value: str) -> bool:
    """判断一个单元格文本"看起来像模型名"。

    非空、非指标名、非分隔符、非纯数字,且包含至少一个字母/汉字
    (ASCII 字母或中日韩统一表意文字)时认为像模型名。
    """
    stripped = value.strip()
    if not stripped or _is_metric_cell(stripped) or _is_separator_row([stripped]):
        return False
    if re.fullmatch(r"-?\d+(?:\.\d+)?%?", stripped):
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", stripped))
