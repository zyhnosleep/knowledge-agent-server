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
