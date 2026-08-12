from __future__ import annotations

from app.services.answer_verifier import AnswerVerifier


class TestAnswerVerifier:
    """RED tests for AnswerVerifier — answer quality check."""

    # ---- ok path ----

    def test_ok_for_valid_answer_with_citations(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="What is X?",
            answer_markdown="X is a well-known concept.",
            citations=[{"document_id": "d1", "excerpt": "X is..."}],
            route_type="simple_rag",
        )
        assert result["ok"] is True
        assert result["warnings"] == []
        assert result["retry_recommended"] is False

    # ---- empty answer ----

    def test_empty_answer_warns_and_recommends_retry(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="What is X?",
            answer_markdown="",
            citations=[],
            route_type="simple_rag",
        )
        assert result["ok"] is True  # verifier itself doesn't fail
        assert any("empty" in w.lower() for w in result["warnings"])
        assert result["retry_recommended"] is True

    def test_whitespace_only_answer_warns(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="What is X?",
            answer_markdown="   \n  ",
            citations=[],
            route_type="simple_rag",
        )
        assert any("empty" in w.lower() for w in result["warnings"])
        assert result["retry_recommended"] is True

    # ---- missing citations for evidence routes ----

    def test_missing_citations_warns_when_evidence_required(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="What is the evidence?",
            answer_markdown="Some answer.",
            citations=[],
            route_type="evidence_required",
        )
        assert any(
            "citation" in w.lower() or "evidence" in w.lower()
            for w in result["warnings"]
        )
        assert result["retry_recommended"] is True

    def test_missing_citations_warns_when_multi_source_compare(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="Compare A and B",
            answer_markdown="They are different.",
            citations=[],
            route_type="multi_source_compare",
        )
        assert any(
            "citation" in w.lower() or "source" in w.lower()
            for w in result["warnings"]
        )
        assert result["retry_recommended"] is True

    def test_missing_citations_warns_when_table_or_metric(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="What is the value?",
            answer_markdown="The value is 42.",
            citations=[],
            route_type="table_or_metric",
        )
        assert any(
            "citation" in w.lower() or "evidence" in w.lower()
            for w in result["warnings"]
        )
        assert result["retry_recommended"] is True

    # ---- table-like evidence ----

    def test_table_or_metric_without_table_evidence_warns(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="What is the value?",
            answer_markdown="The value is something.",
            citations=[{"document_id": "d1"}],  # has citation, no table evidence
            route_type="table_or_metric",
        )
        assert any(
            "table" in w.lower() or "metric" in w.lower() or "numeric" in w.lower()
            for w in result["warnings"]
        )
        assert result["retry_recommended"] is True

    def test_table_or_metric_with_numeric_answer_passes(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="What is the value?",
            answer_markdown="The value is 42.5 kcal/mol.",
            citations=[{"document_id": "d1"}],
            route_type="table_or_metric",
        )
        # Numeric patterns present → no table-evidence warning
        assert not any(
            "table" in w.lower() and "evidence" in w.lower()
            for w in result["warnings"]
        )
        assert result["retry_recommended"] is False

    def test_table_or_metric_with_page_kind_table_passes(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="What is the structure?",
            answer_markdown="It has a complex structure.",
            citations=[{"document_id": "d1", "page_kind": "table"}],
            route_type="table_or_metric",
        )
        # page_kind == "table" → passes
        assert result["retry_recommended"] is False

    # ---- no retry for ok answers ----

    def test_no_retry_for_ok_answer_simple_rag(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="What is X?",
            answer_markdown="X is Y because of Z.",
            citations=[{"document_id": "d1", "excerpt": "X is Y"}],
            route_type="simple_rag",
        )
        assert result["retry_recommended"] is False

    def test_no_retry_for_needs_clarification(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="",
            answer_markdown="",
            citations=[],
            route_type="needs_clarification",
        )
        # needs_clarification doesn't require citations
        assert result["retry_recommended"] is False

    # ---- result shape ----

    def test_result_has_expected_keys(self) -> None:
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="Q",
            answer_markdown="A",
            citations=[],
            route_type="simple_rag",
        )
        for key in ("ok", "warnings", "retry_recommended", "reason"):
            assert key in result, f"Missing key: {key}"
        assert isinstance(result["ok"], bool)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["retry_recommended"], bool)
        assert isinstance(result["reason"], str)


class TestTextQualityHeuristics:
    """T2：复读机式重复与单 token 主导噪音的文本质量启发式（2026-08-12 R15）。"""

    def test_gibberish_repeating_phrase_warns_and_retries(self) -> None:
        """R15 实测形态：'改进了Val。' 反复复读 → 4-gram 重复命中。"""
        verifier = AnswerVerifier()
        answer = (
            "根据提供的证据，改进了Val。虽然改进了Val。而改进了Val。"
            "但改进了Val。因此改进了Val。"
        )
        result = verifier.verify(
            question="χ1 的改进是否意味着所有残基都得到了改善？",
            answer_markdown=answer,
            citations=[],
            route_type="evidence_required",
        )
        assert any("repeats identical phrase" in w for w in result["warnings"]), (
            result["warnings"]
        )
        assert result["retry_recommended"] is True

    def test_dominant_single_token_warns_and_retries(self) -> None:
        """同一 token 高频穿插于不同上下文（4-gram 窗口无重复）→ 单 token 主导命中。"""
        verifier = AnswerVerifier()
        answer = (
            "Val。改。Val。优。Val。增。Val。减。Val。变。"
            "Val。调。Val。保。Val。稳。Val。波。Val。升。"
        )
        result = verifier.verify(
            question="什么是 Val？",
            answer_markdown=answer,
            citations=[],
            route_type="simple_rag",
        )
        assert any("dominated by repeated token" in w for w in result["warnings"]), (
            result["warnings"]
        )
        assert result["retry_recommended"] is True

    def test_normal_answer_not_flagged(self) -> None:
        """正常长答案（含重复出现的常用词）不得误报。"""
        verifier = AnswerVerifier()
        answer = (
            "CHARMM36 力场对蛋白质的改进主要体现在侧链扭转势上。"
            "其中 CMAP 项对 backbone 的二面角分布有显著影响，"
            "Alanine 与 Valine 的 rotamer 分布与实验 NMR 数据一致。"
            "该力场在 298 K 与 300 K 下进行 400 ns 的分子动力学模拟验证，"
            "结果与 SPARTA 预测的化学位移对比表明误差在 1 Å 以内。"
        )
        result = verifier.verify(
            question="CHARMM36 的改进效果如何？",
            answer_markdown=answer,
            citations=[{"document_id": "d1", "excerpt": "charmm36"}],
            route_type="evidence_required",
        )
        assert not any(
            "gibberish" in w for w in result["warnings"]
        ), result["warnings"]
        assert result["retry_recommended"] is False

    def test_short_answer_never_statistically_flagged(self) -> None:
        """短答案不进入统计（<12 token），即使有重复也不误报。"""
        verifier = AnswerVerifier()
        result = verifier.verify(
            question="X?",
            answer_markdown="val val val val val val",
            citations=[],
            route_type="simple_rag",
        )
        assert not any(
            "gibberish" in w for w in result["warnings"]
        ), result["warnings"]
        assert result["retry_recommended"] is False

    def test_markdown_table_with_repeated_zero_rows_not_flagged(self) -> None:
        """纯数字 4-gram 不计数：合法表格答案的多行相同占位值不误报。

        code review 2026-08-12：四行 "| 0 | 0 | 0 | 0 |" 的 (0,0,0,0)
        是正常表格形态，不是复读机。
        """
        verifier = AnswerVerifier()
        answer = (
            "Table 5 reports the missing values as zero entries:\n"
            "| a | b | c | d |\n"
            "| --- | --- | --- | --- |\n"
            "| 0 | 0 | 0 | 0 |\n"
            "| 0 | 0 | 0 | 0 |\n"
            "| 0 | 0 | 0 | 0 |\n"
            "| 0 | 0 | 0 | 0 |"
        )
        result = verifier.verify(
            question="Table 5 的缺失值是什么？",
            answer_markdown=answer,
            citations=[{"page_kind": "table"}],
            route_type="table_or_metric",
        )
        assert not any(
            "gibberish" in w for w in result["warnings"]
        ), result["warnings"]
        assert result["retry_recommended"] is False

    def test_dominant_token_flagged_below_20_tokens(self) -> None:
        """单 token 主导规则在 12-19 token 区间同样生效（AC 对齐）。

        code review 2026-08-12：原实现额外要求 ≥20 token，16 token 答案
        中 val×8（50%）本应命中却被放过。
        """
        verifier = AnswerVerifier()
        answer = (
            "Val。改。Val。优。Val。增。Val。减。"
            "Val。变。Val。调。Val。保。Val。稳。"
        )
        result = verifier.verify(
            question="什么是 Val？",
            answer_markdown=answer,
            citations=[],
            route_type="simple_rag",
        )
        assert any("dominated by repeated token" in w for w in result["warnings"]), (
            result["warnings"]
        )
        assert result["retry_recommended"] is True
