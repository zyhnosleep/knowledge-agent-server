from __future__ import annotations

import logging
import re
import time

from sqlalchemy.orm import Session

from app.schemas.common import QueryResponse
from app.schemas.agent import EvidencePack
from app.services.search import QueryService

logger = logging.getLogger(__name__)

_INSUFFICIENT_EVIDENCE_RE = re.compile(
    r"insufficient evidence|no supporting evidence was found|does not contain information",
    re.IGNORECASE,
)


class RAGAdapter:
    """Thin wrapper around the existing QueryService.

    The Agent layer calls ``RAGAdapter.answer()`` as a read-only tool so
    that RAG remains an independent evidence provider and the Agent
    never reaches into QueryService internals directly.
    """

    def answer(
        self,
        db: Session,
        project_slug: str,
        question: str,
        document_id: str | None = None,
    ) -> QueryResponse:
        """Execute a RAG query and return the full response.

        ``save_answer`` is forced to ``False`` so the Agent result is
        the authoritative record and the per-question QA table is not
        cluttered with intermediate tool calls.
        """
        t0 = time.monotonic()
        try:
            result = QueryService(db).answer(
                project_slug, question, save_answer=False, document_id=document_id
            )
        except ValueError:
            # Return an empty-but-valid response so the Agent can
            # record the step and surface the error to the caller.
            logger.warning(
                "RAGAdapter.answer: project_slug=%r not found", project_slug,
            )
            result = QueryResponse(
                answer_markdown="",
                citations=[],
                verification_status="project_not_found",
            )
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.debug("RAGAdapter.answer took %d ms", elapsed_ms)
        return result

    def retrieve_evidence(
        self,
        db: Session,
        project_slug: str,
        question: str,
        limit: int = 15,
        document_id: str | None = None,
    ) -> EvidencePack:
        """Retrieve-only RAG: return an EvidencePack without drafting an answer.

        ``limit`` controls the maximum number of evidence items returned.
        For missing projects, returns an empty EvidencePack with
        ``status="project_not_found"`` instead of crashing.
        """
        t0 = time.monotonic()
        try:
            result = QueryService(db).retrieve_evidence(
                project_slug, question, limit=limit, document_id=document_id
            )
        except ValueError:
            logger.warning(
                "RAGAdapter.retrieve_evidence: project_slug=%r not found",
                project_slug,
            )
            result = EvidencePack(status="project_not_found", items=[])
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.debug("RAGAdapter.retrieve_evidence took %d ms", elapsed_ms)
        return result

    @staticmethod
    def answer_is_insufficient_evidence(answer_text: str) -> bool:
        """Return True when the answer text signals that retrieved evidence
        does not support a grounded answer."""
        return bool(_INSUFFICIENT_EVIDENCE_RE.search(answer_text))
