from __future__ import annotations

import re
from typing import Any


class AnswerVerifier:
    """Deterministic answer quality checker.

    Registered as the read-only tool ``answer.verify``.  Does not call
    any external service or database — it inspects the answer text and
    citation list against the route type.

    Verdict rules:

    * **Empty answer** → warn + recommend retry
    * **Missing citations** on evidence-requiring routes → warn + retry
    * **Missing table / numeric evidence** on ``table_or_metric`` → warn + retry
    * Otherwise → ok, no retry
    """

    # Routes that require citations
    _CITATION_REQUIRED_ROUTES: frozenset[str] = frozenset({
        "evidence_required",
        "multi_source_compare",
        "table_or_metric",
    })

    # Numeric / unit patterns for detecting table-like evidence
    _NUMERIC_RE: re.Pattern = re.compile(
        r"\d+\.?\d*\s*(?:kcal|%|Å|angstrom|nm|kg|g|ml|L|°C|K|eV|kJ|mol|mM|μM|kcal/mol)?",
        re.IGNORECASE,
    )

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
        ``retry_recommended``, and ``reason``."""
        warnings: list[str] = []
        retry_recommended = False
        citations = citations or []

        # needs_clarification is inherently underspecified — skip checks
        if route_type == "needs_clarification":
            return {
                "ok": True,
                "warnings": [],
                "retry_recommended": False,
                "reason": "needs_clarification route — verification skipped",
            }

        # ---- 1. empty answer ----
        answer_text = (answer_markdown or "").strip()
        if not answer_text:
            warnings.append("Answer is empty")
            retry_recommended = True

        # ---- 2. missing citations on evidence routes ----
        if route_type in self._CITATION_REQUIRED_ROUTES and not citations:
            warnings.append(
                f"Route '{route_type}' expects citations but none were provided"
            )
            retry_recommended = True

        # ---- 3. missing table / numeric evidence for table_or_metric ----
        if route_type == "table_or_metric" and not retry_recommended:
            has_evidence = self._has_table_evidence(answer_text, citations)
            if not has_evidence:
                warnings.append(
                    "Route 'table_or_metric' expects table or numeric evidence "
                    "but none detected in answer or citations"
                )
                retry_recommended = True

        reason = "; ".join(warnings) if warnings else "Answer looks acceptable"

        return {
            "ok": True,  # verifier itself always succeeds (it's a quality gate)
            "warnings": warnings,
            "retry_recommended": retry_recommended,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _has_table_evidence(
        self, answer: str, citations: list[dict[str, Any]]
    ) -> bool:
        """Check whether *answer* or *citations* carry table-like evidence."""
        # Numeric / unit pattern in answer
        if self._NUMERIC_RE.search(answer):
            return True
        # page_kind == "table" in any citation
        for c in citations:
            if c.get("page_kind") == "table":
                return True
        return False
