from __future__ import annotations

import re

from app.schemas.agent import AgentRouteDecision


class PolicyRouter:
    """Deterministic keyword-based router for Agent queries.

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

    # Complex multi-hop / multi-part indicators (English + Chinese)
    # These detect structural patterns that indicate the question requires
    # multiple reasoning steps, not a single retrieval.
    _COMPLEX_MULTI_HOP_TERMS: tuple[str, ...] = (
        "first find", "first, find", "first determine",
        "then calculate", "then determine", "then find",
        "step by step", "multi-step", "multi-hop", "multi-part",
        "multiple questions", "several parts",
        "首先找到", "首先确定", "然后计算", "然后确定",
        "分步", "多步", "多个问题", "几个部分",
        "先找到", "先确定", "再计算", "再确定",
    )

    # Regex patterns for numbered sub-questions (e.g. "1. ... 2. ...")
    _NUMBERED_SUBQUESTION_PATTERN: str = r"(?:^|\n)\s*[1-9]\d*\."

    # Comparison / contrast terms (English + Chinese)
    _COMPARISON_TERMS: tuple[str, ...] = (
        "compare", "difference", "versus", "vs",
        "对比", "比较", "区别", "不同", "相比",
    )

    # Table / metric / numeric terms
    _TABLE_METRIC_TERMS: tuple[str, ...] = (
        "table", "metric", "parameter", "value",
        "表", "指标", "参数", "数值",
    )

    # Numeric / unit patterns
    _UNIT_PATTERNS: tuple[str, ...] = (
        r"\bkcal\b",
        r"Å",
        r"\b[Aa]ngstrom\b",
        r"\d+%",  # e.g. 50%
    )

    # Evidence / citation terms
    _EVIDENCE_TERMS: tuple[str, ...] = (
        "citation", "cite", "reference", "source", "evidence",
        "引用", "证据", "来源", "原文",
    )

    # ------------------------------------------------------------------
    def route(self, query: str) -> AgentRouteDecision:
        """Return a deterministic route decision for *query*."""
        stripped = (query or "").strip()

        # 1. empty / whitespace
        if not stripped:
            return AgentRouteDecision(
                route="needs_clarification",
                requires_citations=False,
                max_retries=0,
                reason="Empty or whitespace-only query",
            )

        query_lower = stripped.lower()

        # 2. complex multi-hop / multi-part indicators
        for term in self._COMPLEX_MULTI_HOP_TERMS:
            if term in query_lower:
                return AgentRouteDecision(
                    route="complex_multi_hop",
                    requires_citations=True,
                    max_retries=0,
                    reason=f"Query contains complex multi-hop term: {term!r}",
                )
        # Numbered sub-questions (e.g. "1. ... 2. ...")
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

        # 3. comparison terms
        for term in self._COMPARISON_TERMS:
            if term in query_lower:
                return AgentRouteDecision(
                    route="multi_source_compare",
                    requires_citations=True,
                    max_retries=1,
                    reason=f"Query contains comparison term: {term!r}",
                )

        # 4. table / metric / numeric terms
        for term in self._TABLE_METRIC_TERMS:
            if term in query_lower:
                return AgentRouteDecision(
                    route="table_or_metric",
                    requires_citations=True,
                    max_retries=1,
                    reason=f"Query contains table/metric term: {term!r}",
                )
        for pat in self._UNIT_PATTERNS:
            if re.search(pat, query_lower, re.IGNORECASE):
                return AgentRouteDecision(
                    route="table_or_metric",
                    requires_citations=True,
                    max_retries=1,
                    reason=f"Query contains numeric/unit pattern: {pat!r}",
                )

        # 5. evidence / citation terms
        for term in self._EVIDENCE_TERMS:
            if term in query_lower:
                return AgentRouteDecision(
                    route="evidence_required",
                    requires_citations=True,
                    max_retries=1,
                    reason=f"Query contains evidence/citation term: {term!r}",
                )

        # 6. default
        return AgentRouteDecision(
            route="simple_rag",
            requires_citations=True,
            max_retries=0,
            reason="Default route — single-source RAG query",
        )
