from __future__ import annotations

import pytest

from app.schemas.agent import AgentRouteDecision
from app.services.agent_policy import PolicyRouter


class TestPolicyRouter:
    """RED tests for AgentRouteDecision schema and PolicyRouter."""

    def test_route_decision_schema_fields(self) -> None:
        """AgentRouteDecision has the required public fields."""
        rd = AgentRouteDecision(
            route="simple_rag",
            requires_citations=True,
            max_retries=0,
            reason="default",
        )
        assert rd.route == "simple_rag"
        assert rd.requires_citations is True
        assert rd.max_retries == 0
        assert rd.reason == "default"
        assert rd.confidence > 0.0  # default confidence

    def test_route_decision_valid_routes_only(self) -> None:
        """Route must be one of the 6 public route types."""
        valid = {
            "simple_rag",
            "evidence_required",
            "table_or_metric",
            "multi_source_compare",
            "complex_multi_hop",
            "needs_clarification",
        }
        for route_type in valid:
            rd = AgentRouteDecision(route=route_type, reason="test")
            assert rd.route == route_type

    # ---- PolicyRouter tests ----

    def test_empty_query_returns_needs_clarification(self) -> None:
        router = PolicyRouter()
        result = router.route("")
        assert result.route == "needs_clarification"
        assert result.requires_citations is False
        assert result.max_retries == 0

    def test_whitespace_query_returns_needs_clarification(self) -> None:
        router = PolicyRouter()
        result = router.route("   \t\n  ")
        assert result.route == "needs_clarification"
        assert result.requires_citations is False
        assert result.max_retries == 0

    @pytest.mark.parametrize(
        "query",
        [
            "compare A and B",
            "difference between X and Y",
            "versus",
            "vs",
            "对比一下",
            "比较方法",
            "区别是什么",
            "不同点",
            "相比而言",
        ],
    )
    def test_comparison_terms_route_to_multi_source_compare(self, query: str) -> None:
        router = PolicyRouter()
        result = router.route(query)
        assert result.route == "multi_source_compare", (
            f"Term '{query}' should route to multi_source_compare, got {result.route}"
        )
        assert result.requires_citations is True
        assert result.max_retries == 1

    @pytest.mark.parametrize(
        "query",
        [
            "table of contents",
            "metric value",
            "parameter x",
            "value is 42",
            "这是什么表",
            "指标是多少",
            "参数设置",
            "数值结果",
            "100 kcal",
            "50%",
            "1.5 Å",
            "angstrom units",
        ],
    )
    def test_table_metric_terms_route_to_table_or_metric(self, query: str) -> None:
        router = PolicyRouter()
        result = router.route(query)
        assert result.route == "table_or_metric", (
            f"Term '{query}' should route to table_or_metric, got {result.route}"
        )
        assert result.requires_citations is True
        assert result.max_retries == 1

    @pytest.mark.parametrize(
        "query",
        [
            "citation needed",
            "cite this",
            "reference please",
            "source of this",
            "evidence shows",
            "引用来源",
            "证据是什么",
            "来源在哪",
            "原文怎么写的",
        ],
    )
    def test_evidence_terms_route_to_evidence_required(self, query: str) -> None:
        router = PolicyRouter()
        result = router.route(query)
        assert result.route == "evidence_required", (
            f"Term '{query}' should route to evidence_required, got {result.route}"
        )
        assert result.requires_citations is True
        assert result.max_retries == 1

    @pytest.mark.parametrize(
        "query",
        [
            "hello world",
            "what is something",
            "general question",
            "explain this concept",
        ],
    )
    def test_default_route_is_simple_rag(self, query: str) -> None:
        router = PolicyRouter()
        result = router.route(query)
        assert result.route == "simple_rag"
        assert result.requires_citations is True
        assert result.max_retries == 0

    def test_priority_order_comparison_over_evidence(self) -> None:
        """Comparison terms take priority over evidence terms."""
        router = PolicyRouter()
        # This query contains both "compare" and "evidence"
        result = router.route("compare the evidence")
        assert result.route == "multi_source_compare"

    # ---- complex_multi_hop routing tests ----

    @pytest.mark.parametrize(
        "query",
        [
            "first find the protein structure, then determine its binding affinity",
            "step by step explain how to calculate the energy",
            "multi-step analysis of the molecular dynamics",
            "this is a multi-hop question about protein folding",
            "multi-part question: first find, then determine",
            "multiple questions about this topic",
            "several parts to this query",
            "首先找到蛋白质结构，然后计算结合能",
            "分步骤说明如何分析",
            "先确定参数，再计算结果",
        ],
    )
    def test_complex_multi_hop_terms_route_correctly(self, query: str) -> None:
        router = PolicyRouter()
        result = router.route(query)
        assert result.route == "complex_multi_hop", (
            f"Query '{query[:50]}...' should route to complex_multi_hop, got {result.route}"
        )
        assert result.requires_citations is True
        assert result.max_retries == 0

    @pytest.mark.parametrize(
        "query",
        [
            "1. What is the structure?\n2. How does it bind?\n3. What is the energy?",
            "1. Find the protein\n2. Then determine affinity",
        ],
    )
    def test_numbered_subquestions_route_to_complex_multi_hop(self, query: str) -> None:
        router = PolicyRouter()
        result = router.route(query)
        assert result.route == "complex_multi_hop", (
            f"Numbered sub-question query should route to complex_multi_hop, got {result.route}"
        )

    @pytest.mark.parametrize(
        "query",
        [
            "hello world",
            "what is something",
            "general question",
            "explain this concept",
            "what is the binding affinity of protein X",
            "how does this work",
        ],
    )
    def test_simple_questions_do_not_trigger_complex_multi_hop(
        self, query: str
    ) -> None:
        router = PolicyRouter()
        result = router.route(query)
        assert result.route != "complex_multi_hop", (
            f"Simple query '{query[:50]}...' should NOT route to complex_multi_hop, "
            f"got {result.route}"
        )

    def test_single_numbered_item_does_not_trigger_complex(self) -> None:
        """A single '1.' is not enough — need at least 2 sub-questions."""
        router = PolicyRouter()
        result = router.route("1. What is the structure of this protein?")
        assert result.route != "complex_multi_hop"

    def test_priority_complex_over_comparison(self) -> None:
        """Complex multi-hop terms take priority over comparison terms."""
        router = PolicyRouter()
        # This query has both "first find ... then" (complex) and "compare" (comparison)
        result = router.route("first find the values then compare them")
        assert result.route == "complex_multi_hop", (
            f"Complex multi-hop should win, got {result.route}"
        )

    def test_complex_multi_hop_before_table_or_metric(self) -> None:
        """Complex multi-hop terms take priority over table/metric terms."""
        router = PolicyRouter()
        result = router.route("step by step analyze the table")
        assert result.route == "complex_multi_hop", (
            f"Complex multi-hop should win over table_or_metric, got {result.route}"
        )
