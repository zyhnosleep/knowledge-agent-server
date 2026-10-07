"""
answer_verifier.py —— 答案质量校验器（确定性规则检查）模块
=========================================================

职责：
- 对问答系统生成的答案进行"确定性质量门禁"检查：在不调用任何外部
  服务或数据库的前提下，仅依据答案文本、引用列表（citations）与
  路由类型（route_type）给出质量判定（verdict）。
- 本模块以只读工具 ``answer.verify`` 的形式注册，供 agent/工作流在
  返回答案前做一次自动校验，决定是否需要重试（retry）。

判定规则（核心逻辑）：
1. 空答案 → 警告并建议重试。
2. 需要证据的路由（evidence_required / multi_source_compare /
   table_or_metric）缺少引用 → 警告并建议重试。
3. ``table_or_metric`` 路由上未检测到表格或数值证据 → 警告并建议重试。
4. 文本质量启发式：4-gram 短语重复（胡言乱语/复读机信号）或单 token
   主导（≥8 次且占比过半）→ 仅提示，不触发生成重试（2026-08-12 回归 R15
   实测：答案反复重复"改进了Val。"短语并通过旧校验）。
5. 其余情况 → ok，不重试。

设计说明：
- "验证器自身永远成功"（``ok=True`` 恒定）：它只是一个质量门禁，
  其产物是 ``warnings`` / ``retry_recommended`` 等建议字段，而不会让
  校验流程本身失败。
- 本模块刻意保持无副作用、纯函数式，便于单元测试与在 agent 工具
  环境中安全调用。
"""

from __future__ import annotations

import re
from typing import Any


class AnswerVerifier:
    """Deterministic answer quality checker.

    确定性答案质量校验器。校验规则同上。

    Registered as the read-only tool ``answer.verify``.  Does not call
    any external service or database — it inspects the answer text and
    citation list against the route type.

    注册为只读工具 ``answer.verify``，不调用任何外部服务或数据库，
    仅根据答案文本、引用列表与路由类型做静态检查。

    Verdict rules:
    判定规则：

    * **Empty answer** → warn + recommend retry
      空答案 → 警告 + 建议重试
    * **Missing citations** on evidence-requiring routes → warn + retry
      需要证据的路由缺少引用 → 警告 + 建议重试
    * **Missing table / numeric evidence** on ``table_or_metric`` → warn + retry
      ``table_or_metric`` 路由缺少表格/数值证据 → 警告 + 建议重试
    * Otherwise → ok, no retry
      其余情况 → 通过，不重试
    """

    # Routes that require citations
    # 需要引用的路由集合：这些路由的答案必须附带引用来源
    _CITATION_REQUIRED_ROUTES: frozenset[str] = frozenset({
        "evidence_required",  # 强证据要求路由
        "multi_source_compare",  # 多来源比较路由
        "table_or_metric",  # 表格/数值指标路由
    })

    # Numeric / unit patterns for detecting table-like evidence
    # 数值/单位模式：用于在答案文本中检测"表格类证据"
    # 匹配形如 "12.5 kcal"、"30%"、"5 nm"、"100°C" 等数值+可选单位片段
    _NUMERIC_RE: re.Pattern = re.compile(
        r"\d+\.?\d*\s*(?:kcal|%|Å|angstrom|nm|kg|g|ml|L|°C|K|eV|kJ|mol|mM|μM|kcal/mol)?",
        re.IGNORECASE,
    )

    # ---- 文本质量启发式阈值（2026-08-12 回归 R15） ----
    # 相同 4-gram 出现 ≥ 4 次 → 复读机式重复（R15 "改进了Val。" 反复出现）
    _REPEAT_GRAM_SIZE = 4
    _REPEAT_GRAM_MIN_COUNT = 4
    # 单个 token 出现 ≥ 8 次且占总 token 数 ≥ 50% → 单 token 主导噪音
    # （"Val。Val。Val。…" 这类整段只复读一个词的情况，4-gram 统计会漏检）
    _DOMINANT_TOKEN_MIN_COUNT = 8
    _DOMINANT_TOKEN_MIN_RATIO = 0.5

    # 答案 token 化：英文单词（含数字）+ 中文单字，小写去标点
    _TOKEN_RE: re.Pattern = re.compile(r"[a-z0-9]+|[一-鿿]")

    # ------------------------------------------------------------------
    def verify(
        self,
        *,
        question: str,
        answer_markdown: str,
        citations: list[dict[str, Any]] | None = None,
        route_type: str = "simple_rag",
    ) -> dict[str, Any]:
        """Return a quality verdict dict with ``ok``, ``warnings``,
        ``retry_recommended``, and ``reason``.

        返回包含 ``ok``、``warnings``、``retry_recommended``、``reason``
        四个字段的质量判定字典。

        参数：
        - ``question``：原始问题（当前逻辑中主要留作上下文，校验时未直接使用）。
        - ``answer_markdown``：模型生成的答案（Markdown 文本）。
        - ``citations``：答案附带的引用列表；每项为字典，可含
          ``page_kind`` 等字段。为 None 时按空列表处理。
        - ``route_type``：当前回答使用的路由类型，决定校验规则的强弱
          （默认 ``simple_rag``）。

        判定流程：
        1. 若路由为 ``needs_clarification``（需要澄清），其答案天然信息
           不足，跳过所有检查直接返回通过。
        2. 空答案检查 → 警告 + 建议重试。
        3. 需要引用的路由缺少引用 → 警告 + 建议重试。
        4. ``table_or_metric`` 路由且尚未建议重试时，进一步检查是否包含
           表格/数值证据。
        5. 汇总警告为 ``reason`` 字符串并返回。
        """
        warnings: list[str] = []
        retry_recommended = False
        citations = citations or []

        # needs_clarification is inherently underspecified — skip checks
        # needs_clarification 路由本身就意味着信息不充分，跳过质量校验
        if route_type == "needs_clarification":
            return {
                "ok": True,
                "warnings": [],
                "retry_recommended": False,
                "reason": "needs_clarification route — verification skipped",
            }

        # ---- 1. empty answer ----
        # ---- 检查一：空答案 ----
        answer_text = (answer_markdown or "").strip()
        if not answer_text:
            warnings.append("Answer is empty")
            retry_recommended = True
        else:
            # ---- 1b. 文本质量启发式（非空答案才统计） ----
            # ---- 检查一 b：复读机式重复/单 token 主导噪音 ----
            quality_warnings = self._text_quality_warnings(answer_text)
            warnings.extend(quality_warnings)
            # Wording quality is advisory. Only missing answer/evidence gates
            # can request a retry; repetition cannot spend the shared budget.

        # ---- 2. missing citations on evidence routes ----
        # ---- 检查二：需要证据的路由缺少引用 ----
        if route_type in self._CITATION_REQUIRED_ROUTES and not citations:
            warnings.append(
                f"Route '{route_type}' expects citations but none were provided"
            )
            retry_recommended = True

        # ---- 3. missing table / numeric evidence for table_or_metric ----
        # ---- 检查三：table_or_metric 路由缺少表格/数值证据 ----
        # 仅在尚未建议重试时执行，避免重复累积警告
        if route_type == "table_or_metric" and not retry_recommended:
            has_evidence = self._has_table_evidence(answer_text, citations)
            if not has_evidence:
                warnings.append(
                    "Route 'table_or_metric' expects table or numeric evidence "
                    "but none detected in answer or citations"
                )
                retry_recommended = True

        # 汇总：有警告则用分号连接，否则提示答案可接受
        reason = "; ".join(warnings) if warnings else "Answer looks acceptable"

        return {
            "ok": True,  # verifier itself always succeeds (it's a quality gate)
            # ok 恒为 True：验证器本身总是"成功"，因为它只是质量门禁，
            # 真正有意义的输出是 warnings / retry_recommended
            "warnings": warnings,
            "retry_recommended": retry_recommended,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _text_quality_warnings(self, answer: str) -> list[str]:
        """检测复读机式重复与单 token 主导噪音，返回警告列表。

        2026-08-12 回归 R15：模型输出"改进了Val。"反复复读的胡言乱语，
        旧校验只查空/长度/引用全部放行。此处做轻量统计：
        - 相同 4-gram（token 序列）出现 ≥4 次 → 复读机信号；
        - 单个 token ≥8 次且占比 ≥50%（含字母）→ 单 token 主导噪音
          （整段只复读一个词时 4-gram 窗口统计不到，需单独兜底）。
        短答案（<12 token）不做统计，避免小样本抖动误报。
        """
        tokens = self._TOKEN_RE.findall((answer or "").lower())
        if len(tokens) < 12:
            return []

        warnings: list[str] = []

        # 4-gram 重复：R15 "改进了Val。" → (改,进,了,val) 出现 4 次。
        # 纯数字 4-gram 不计数——合法的 markdown 表格答案可能有多行
        # "| 0 | 0 | 0 | 0 |"（占位/空值行），(0,0,0,0) 重复是正常表格
        # 形态而非复读机（2026-08-12 code review 修正误报）。
        if len(tokens) >= self._REPEAT_GRAM_SIZE + self._REPEAT_GRAM_MIN_COUNT - 1:
            gram_counts: dict[tuple[str, ...], int] = {}
            for i in range(len(tokens) - self._REPEAT_GRAM_SIZE + 1):
                gram = tuple(tokens[i : i + self._REPEAT_GRAM_SIZE])
                if all(token.isdigit() for token in gram):
                    continue
                gram_counts[gram] = gram_counts.get(gram, 0) + 1
            if gram_counts:
                top_count = max(gram_counts.values())
                if top_count >= self._REPEAT_GRAM_MIN_COUNT:
                    worst = max(gram_counts, key=gram_counts.get)
                    warnings.append(
                        "Answer repeats identical phrase "
                        f"'{' '.join(worst)}' {top_count} times (possible gibberish)"
                    )

        # 单 token 主导："Val。Val。Val。…" 只复读一个词时 4-gram 无重复。
        # 与 4-gram 规则共用同一 <12 token 短答案门（<12 不做统计）；
        # 纯数字 token（如表格数值 "0"/"74"）不参与主导判定，避免合法
        # 数值列表被误判为噪音。
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        dominant, dominant_count = max(counts.items(), key=lambda kv: kv[1])
        is_word = re.search(r"[a-z一-鿿]", dominant) is not None
        if (
            is_word
            and dominant_count >= self._DOMINANT_TOKEN_MIN_COUNT
            and dominant_count / len(tokens) >= self._DOMINANT_TOKEN_MIN_RATIO
        ):
            warnings.append(
                "Answer dominated by repeated token "
                f"'{dominant}' ({dominant_count}/{len(tokens)} tokens, "
                "possible gibberish)"
            )

        return warnings

    def _has_table_evidence(
        self, answer: str, citations: list[dict[str, Any]]
    ) -> bool:
        """Check whether *answer* or *citations* carry table-like evidence.

        检查答案文本或引用中是否携带"表格类证据"。

        判定依据：
        1. 答案文本中匹配到数值/单位模式（如 "12.5 kcal"、"30%"）。
        2. 任一引用的 ``page_kind`` 字段为 ``"table"``（即该引用指向
           一个表格页）。

        满足其一即认为存在表格证据。
        """
        # Numeric / unit pattern in answer
        # 在答案中查找数值/单位模式
        if self._NUMERIC_RE.search(answer):
            return True
        # page_kind == "table" in any citation
        # 任一引用标注为表格页
        for c in citations:
            if c.get("page_kind") == "table" or c.get('block_type') == 'table' or c.get('table_id'):
                return True
        return False
