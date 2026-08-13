"""生成约束的共享出口（draft 与 synthesize 共用，避免各 prompt 漂移）。

2026-08-12 锁定回归实测失败形态（T3）：
- R29/R38：中文问题，答案却输出英文 → 语言必须跟随用户问题；
- R36：答案泄漏内部元数据（"关联地址为 21201 及 48824"、
  "相关文献编号为 2013"）→ 元数据禁止输出；
- R26：基于证据的推断（作者取舍原因）未与事实区分 → 推断需显式标记。

search.py 的 ``_build_answer_constraints``（draft）与
agent_synthesizer.py 的 ``_answer_rules``（synthesize local/external/retry）
都从这里取同一份文本；任何调整只改这一处。本模块只依赖 ``re``。
"""

from __future__ import annotations

import re

# 语言跟随：中文问题 → 简体中文回答（保留术语原样）；英文问题 → 英文。
def language_rule(query: str | None) -> str:
    """返回跟随 *query* 语言的一条约束指令。"""
    if re.search(r"[一-鿿]", query or ""):
        return (
            "IMPORTANT: Answer in Chinese (简体中文) because the user's "
            "question is in Chinese. Keep table labels, dataset names, model "
            "names, acronyms, and technical terms verbatim when needed."
        )
    return (
        "IMPORTANT: Answer in English because the user's question is in "
        "English."
    )


# 元数据禁止：内部文档元数据一律不得出现在答案中，除非问题明确询问且
# 证据原文出现；引用只用 [0] 标记。
METADATA_BAN_RULE = (
    "IMPORTANT: Never output internal document metadata — submission IDs, "
    "author or institution addresses/numbers, upload or submission dates, "
    "version numbers, document IDs, or page numbers — unless the question "
    "explicitly asks about them AND they appear verbatim in the evidence. "
    "Reference sources only with citation markers such as [0]."
)

# 推断标记：超出证据直接陈述的推断/解读必须显式标记（推测/推断/解读），
# 与证据陈述的事实区分开。
INFERENCE_MARKING_RULE = (
    "IMPORTANT: When part of your answer is inference or interpretation "
    "beyond what the evidence states directly (authors' reasoning, "
    "trade-offs, significance), mark it explicitly as inference (e.g. "
    "推测 / 推断 / 基于证据的解读) and keep it clearly distinct from "
    "evidence-stated facts."
)

# 数学可读性：公式与力场术语直接写成 Unicode 可读字符，
# 不得输出 LaTeX 命令源码（前端无法渲染 LaTeX）。
LATEX_FREE_RULE = (
    "IMPORTANT: Write mathematical symbols, chemical formulas, and "
    "force-field terms as readable Unicode text (e.g. χ₁, φ/ψ, Ala₃, "
    "α_R, ≤, °) — never as LaTeX source. In particular: no backslash "
    "escapes of any kind (\\|, \\_, \\theta, \\mathrm), no _{...} / "
    "^{...} / _ { ... } group syntax, and no $...$ math markers. Use "
    "a plain underscore for subscripts (y_<n, p_T) and a plain "
    "vertical bar for conditions (p_T(y_n | x))."
)


# 答案结构：较长答案用固定小节标题分块（结论 / 证据 / 不确定性），
# 前端按标题渲染为色条区块；不适用的小节省略，简单问题可无标题。
def answer_structure_rule(query: str | None) -> str:
    """返回一条跟随 *query* 语言的结构化约束指令。"""
    if re.search(r"[一-鿿]", query or ""):
        return (
            "IMPORTANT: Organize longer answers into clear sections with "
            "exact headings '## 结论', '## 证据', '## 不确定性' (omit any "
            "section that does not apply; a simple direct answer needs no "
            "headings). Keep the headings verbatim so the UI can render "
            "them as sections."
        )
    return (
        "IMPORTANT: Organize longer answers into clear sections with exact "
        "headings '## Conclusion', '## Evidence', '## Uncertainty' (omit "
        "any section that does not apply; a simple direct answer needs no "
        "headings). Keep the headings verbatim so the UI can render them "
        "as sections."
    )


def answer_rules(query: str | None) -> str:
    """组装五条生成约束（T3 + LaTeX 可读性 + 答案结构），供 synthesize system prompt 整块插入。"""
    return (
        language_rule(query)
        + "\n"
        + METADATA_BAN_RULE
        + "\n"
        + INFERENCE_MARKING_RULE
        + "\n"
        + LATEX_FREE_RULE
        + "\n"
        + answer_structure_rule(query)
    )
