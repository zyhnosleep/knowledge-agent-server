"""Agent 查询路由策略：基于关键词的确定性路由。

该模块实现 :class:`PolicyRouter`，在发起任何 RAG 检索之前，用纯规则
（子串匹配 + 正则）对用户查询进行分类，决定后续应采用哪种工具策略。
整个过程不涉及任何大模型调用，因此路由结果完全确定、可复现、可单测。

路由优先级（首个命中即返回，顺序不可调换）：

1. **needs_clarification** —— 空查询或纯空白查询
2. **complex_multi_hop** —— 多部分 / 链式推理问题（含编号子问题）
3. **multi_source_compare** —— 比较 / 对比类问题
4. **table_or_metric** —— 表格 / 指标 / 参数 / 数值单位类问题
5. **evidence_required** —— 需要引用 / 来源 / 证据的问题
6. **simple_rag** —— 兜底默认路由

每个决策都附带 ``requires_citations``（是否强制要求引用）与
``max_retries``（最大重试次数），供下游执行器据此调整生成与校验策略。
"""

from __future__ import annotations

import re

from app.schemas.agent import AgentRouteDecision


class PolicyRouter:
    """基于关键词的确定性 Agent 查询路由器（全程不调用任何大模型）。

    Deterministic keyword-based router for Agent queries.

    Chooses the correct tool strategy before any RAG call.  No LLM is
    involved — all routing is based on substring matching against the
    query text.

    Priority order (first match wins):

    1. **needs_clarification** — empty or whitespace-only
    2. **complex_multi_hop** — multi-part / chained reasoning questions
    3. **multi_source_compare** — comparison / contrast terms
    4. **table_or_metric** — table, metric, parameter, numeric / unit terms
    5. **evidence_required** — citation / source / evidence terms
    6. **simple_rag** — default fallback
    """

    # 复杂多跳 / 多部分问题的指示词（中英双语）。
    # 这些词用于识别"结构型"的查询模式——即问题需要多个推理步骤，
    # 而不是单次检索即可回答，例如"先找到...然后计算..."这类链式指令。
    _COMPLEX_MULTI_HOP_TERMS: tuple[str, ...] = (
        "first find", "first, find", "first determine",
        "then calculate", "then determine", "then find",
        "step by step", "multi-step", "multi-hop", "multi-part",
        "multiple questions", "several parts",
        "首先找到", "首先确定", "然后计算", "然后确定",
        "分步", "多步", "多个问题", "几个部分",
        "先找到", "先确定", "再计算", "再确定",
    )

    # 编号子问题的正则模式，用于识别 "1. ... 2. ..." 之类的多段提问；
    # 每个编号须出现在行首（或紧随换行），编号为 1-9 开头的多位数字。
    _NUMBERED_SUBQUESTION_PATTERN: str = r"(?:^|\n)\s*[1-9]\d*\."

    # 比较 / 对比类指示词（中英双语），命中即路由到 multi_source_compare，
    # 提示执行器需要从多个来源交叉比对后作答。
    _COMPARISON_TERMS: tuple[str, ...] = (
        "compare", "difference", "versus", "vs",
        "对比", "比较", "区别", "不同", "相比",
    )

    # 表格 / 指标 / 数值类指示词，命中即路由到 table_or_metric。
    # "parameter"/"参数" 不在其中：单独出现（如 "torsional parameters 的
    # 改进方向"）是叙述性问题，误路由表格会被 coverage 门禁逼进 synthesize
    # （2026-08-11 topic-switch 案例）——需与数值/表格索取语境词共现才路由。
    _TABLE_METRIC_TERMS: tuple[str, ...] = (
        "table", "metric", "value",
        "表", "指标", "数值",
    )

    # parameter 类词：命中后需与 _PARAMETER_CONTEXT_TERMS 任一语境词共现，
    # 才判定为表格/数值索取型问题（"列出参数值/参数是多少/表里的参数"）。
    # "parameters" 由 "parameter" 子串覆盖，不单列。
    _PARAMETER_TERMS: tuple[str, ...] = (
        "parameter", "参数",
    )

    # 数值 / 表格索取语境词：与 parameter 类词共现时确认用户在索要
    # 表格化/数值化的参数信息，而非叙述性描述。
    _PARAMETER_CONTEXT_TERMS: tuple[str, ...] = (
        "value", "values", "数值", "值", "是多少", "列出", "list",
        "table", "表", "unit", "单位",
    )

    # 数值 / 单位类正则模式：命中即路由到 table_or_metric。
    # 例如 kcal、Å（埃）、angstrom、百分比（如 50%）。
    _UNIT_PATTERNS: tuple[str, ...] = (
        r"\bkcal\b",
        r"Å",
        r"\b[Aa]ngstrom\b",
        r"\d+%",  # e.g. 50%（例如 50%）
    )

    # 证据 / 引用类指示词，命中即路由到 evidence_required，
    # 要求执行器必须在回答中给出引用 / 来源。
    _EVIDENCE_TERMS: tuple[str, ...] = (
        "citation", "cite", "reference", "source", "evidence",
        "引用", "证据", "来源", "原文",
    )

    # ------------------------------------------------------------------
    def route(self, query: str) -> AgentRouteDecision:
        """对 *query* 做出确定性的路由决策并返回。

        Return a deterministic route decision for *query*.

        :param query: 用户原始查询文本（可为空或纯空白）。
        :return: 一个 :class:`AgentRouteDecision`，包含 route、
            requires_citations、max_retries 与 reason 字段。
        """
        # 去除首尾空白；若为空串或纯空白，直接判定为"需要澄清"。
        stripped = (query or "").strip()

        # 1. 空 / 纯空白查询：无需检索也无需重试，交由上游向用户索要更多信息。
        if not stripped:
            return AgentRouteDecision(
                route="needs_clarification",
                requires_citations=False,
                max_retries=0,
                reason="Empty or whitespace-only query",
            )

        # 统一转小写，使英文关键词匹配不区分大小写。
        query_lower = stripped.lower()

        # 2. 复杂多跳 / 多部分指示词：任一命中即视为需要多步推理。
        for term in self._COMPLEX_MULTI_HOP_TERMS:
            if term in query_lower:
                return AgentRouteDecision(
                    route="complex_multi_hop",
                    requires_citations=True,
                    max_retries=0,
                    reason=f"Query contains complex multi-hop term: {term!r}",
                )
        # 编号子问题检测（例如 "1. ... 2. ..."）：至少两段编号才判定为多部分。
        numbered_matches = re.findall(
            self._NUMBERED_SUBQUESTION_PATTERN, stripped
        )
        if len(numbered_matches) >= 2:
            return AgentRouteDecision(
                route="complex_multi_hop",
                requires_citations=True,
                max_retries=0,
                reason="Query contains numbered sub-questions (multi-part)",
            )

        # 3. 比较类指示词：需要从多个来源交叉对比。
        for term in self._COMPARISON_TERMS:
            if term in query_lower:
                return AgentRouteDecision(
                    route="multi_source_compare",
                    requires_citations=True,
                    max_retries=1,
                    reason=f"Query contains comparison term: {term!r}",
                )

        # 4. 表格 / 指标 / 数值类指示词：优先按表格或指标型问题处理。
        for term in self._TABLE_METRIC_TERMS:
            if term in query_lower:
                return AgentRouteDecision(
                    route="table_or_metric",
                    requires_citations=True,
                    max_retries=1,
                    reason=f"Query contains table/metric term: {term!r}",
                )
        # 4b. parameter 类词：需与数值/表格索取语境词共现才路由表格，
        # 否则视为叙述性描述（"…的改进方向/过程"），交给下游默认路径。
        for term in self._PARAMETER_TERMS:
            if term in query_lower:
                for ctx in self._PARAMETER_CONTEXT_TERMS:
                    if ctx in query_lower:
                        return AgentRouteDecision(
                            route="table_or_metric",
                            requires_citations=True,
                            max_retries=1,
                            reason=(
                                f"Query contains parameter term {term!r} "
                                f"with value/table context {ctx!r}"
                            ),
                        )
        # 数值 / 单位正则：忽略大小写匹配（如 kcal、50% 等）。
        for pat in self._UNIT_PATTERNS:
            if re.search(pat, query_lower, re.IGNORECASE):
                return AgentRouteDecision(
                    route="table_or_metric",
                    requires_citations=True,
                    max_retries=1,
                    reason=f"Query contains numeric/unit pattern: {pat!r}",
                )

        # 5. 证据 / 引用类指示词：要求必须附带引用来源。
        for term in self._EVIDENCE_TERMS:
            if term in query_lower:
                return AgentRouteDecision(
                    route="evidence_required",
                    requires_citations=True,
                    max_retries=1,
                    reason=f"Query contains evidence/citation term: {term!r}",
                )

        # 6. 兜底默认：以上规则均未命中时，按普通单来源 RAG 查询处理。
        return AgentRouteDecision(
            route="simple_rag",
            requires_citations=True,
            max_retries=0,
            reason="Default route — single-source RAG query",
        )
