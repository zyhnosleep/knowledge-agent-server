"""摘要（Abstract）提取工具。

负责从文本中识别显式的 "Abstract / 摘要" 章节，并按阅读顺序提取其正文。
支持中英文摘要标题，以及常见的后续章节边界（Introduction、Keywords、
Background、Methods、Results、Discussion、结论 等），避免把正文误收入摘要。

主要入口：
- has_explicit_abstract(text): 判断文本中是否有显式摘要标题。
- extract_explicit_abstract(candidates): 从候选文本列表中提取摘要正文。
"""

from __future__ import annotations

import re


# 匹配显式摘要标题：行首可选 Markdown 标题符号，接着是 "Abstract" 或 "摘要"，
# 后面可跟冒号、破折号或内联摘要内容。
_ABSTRACT_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:abstract|摘要)\s*"
    r"(?:(?::|：|-|—)\s*(?P<inline>.*))?$$",
    re.IGNORECASE,
)
# Markdown ATX 标题：# 开头的 1-6 级标题。
_ATX_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
# 章节编号：阿拉伯数字、罗马数字、中文数字，支持多级如 1.2.3。
_NUMBER = (
    r"(?:\d+(?:\.\d+)*|[ivxlcdm]+(?:\.[ivxlcdm]+)*|"
    r"[零〇一二三四五六七八九十百]+)"
)
# 章节前缀：例如 "第一章"、"第1节"、"(1)"、"1."、"1、" 等。
_SECTION_PREFIX = (
    rf"(?:第{_NUMBER}[章节]\s*|"
    rf"[（(]{_NUMBER}[）)]\s*|"
    rf"{_NUMBER}\s*[.．、)）-]?\s*)?"
)
# 常见后续章节标题：Introduction、Keywords、Methods、Results、Discussion、
# 以及对应的中文（关键词、引言、背景、研究方法、结果与讨论、结论等）。
_SECTION_HEADING = re.compile(
    rf"^\s*{_SECTION_PREFIX}(?:"
    r"introduction|keywords?|background|materials?\s+and\s+methods?|"
    r"methods?|methodology|results?\s+and\s+discussion|results?|discussion|"
    r"related\s+work|conclusions?|关键词|引言|"
    r"背景|研究方法|方法|"
    r"结果与讨论|结果|结论)"
    r"\s*(?:(?::|：)\s*.*)?$$",
    re.IGNORECASE,
)


def is_section_boundary(line: str) -> bool:
    """判断某一行是否是章节边界（ATX 标题或已知章节标题）。"""
    return bool(_ATX_HEADING.match(line) or _SECTION_HEADING.fullmatch(line))


def has_explicit_abstract(text: str) -> bool:
    """判断文本中是否存在显式的 Abstract / 摘要 标题行。"""
    return any(_ABSTRACT_HEADING.fullmatch(line) for line in text.splitlines())


def extract_explicit_abstract(candidates: list[str]) -> str | None:
    """从候选文本列表中提取显式摘要的正文。

    算法：
    1. 遍历 candidates，找到第一个首行匹配摘要标题的候选。
    2. 如果标题行包含内联摘要内容，直接返回。
    3. 否则收集后续候选的正文，直到遇到下一个章节边界或文本结束。
    4. 返回收集到的正文；如果没有找到显式标题则返回 None。
    """
    def abstract_part(value: str) -> tuple[bool, str]:
        lines = value.strip().splitlines()
        if not lines:
            return False, ""
        heading_index = next(
            (
                index
                for index, line in enumerate(lines)
                if _ABSTRACT_HEADING.fullmatch(line)
            ),
            None,
        )
        if heading_index is None:
            return False, ""
        match = _ABSTRACT_HEADING.fullmatch(lines[heading_index])
        if match is None:
            return False, ""
        body_lines: list[str] = []
        # 标题行内联的摘要内容（例如 "Abstract: this is ..."）。
        inline = (match.group("inline") or "").strip()
        if inline:
            body_lines.append(inline)
        for line in lines[heading_index + 1 :]:
            if is_section_boundary(line):
                break
            body_lines.append(line)
        return True, "\n".join(body_lines).strip()

    for index, candidate in enumerate(candidates):
        matched, body = abstract_part(candidate)
        if not matched:
            continue
        if body:
            return body
        # 摘要标题单独占一个候选，正文在后续候选中。
        following: list[str] = []
        for next_candidate in candidates[index + 1 :]:
            next_value = next_candidate.strip()
            if not next_value:
                continue
            if is_section_boundary(next_value):
                break
            lines = next_value.splitlines()
            collected: list[str] = []
            for line in lines:
                if is_section_boundary(line):
                    break
                collected.append(line)
            if collected:
                following.append("\n".join(collected).strip())
            if len(collected) != len(lines):
                break
        joined = "\n\n".join(part for part in following if part).strip()
        return joined or None
    return None
